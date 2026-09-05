"""The read API: SYSTEM_DESIGN.md section 3.1 promises "webhook receiver,
read API, and the demo page from one process" -- this module is the second
of those three. Every route here is read-only and reuses the SAME
report/trace-building functions the HTML dashboard and the CLI use
(`revenew.measure.report.build_report`/`get_decision_trace`), so a number
shown on the dashboard, printed by `revenew trace`, and returned by
`GET /api/decisions/{id}` can never silently diverge -- there is one query
per concept, not three.

This is what makes F9 ("record every decision's full trace") actually
useful: before this module existed, `decision_candidates` held the complete
per-candidate verdict history and nothing could read it back except a direct
SQL query.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query

from revenew.api.theatre import build_timeline
from revenew.api.webhooks import get_conn
from revenew.decide.bandit import PosteriorStore
from revenew.measure.report import build_report, get_decision_trace, lift_to_dict

# /health is deliberately NOT under /api -- it's an operational endpoint,
# not part of the read API's data surface. Everything else in this module is
# registered with an explicit /api/... path, not a router-level prefix, so
# /health can share the router without picking up one it shouldn't.
router = APIRouter()


@router.get("/health")
def health(conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    counts = {}
    for table in ("customers", "opportunities", "decisions", "outcomes", "executions"):
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()  # noqa: S608 -- fixed table names, not user input
        counts[table] = row["n"]
    return {"status": "ok", "counts": counts}


@router.get("/api/report")
def report(conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    """The same numbers the dashboard renders, as JSON. See `build_report`
    for what's included: per-segment and pooled incremental lift, candidate
    validity, no-action reasons, budget consumed, and (if a replay run has
    been exported) the regret curve and posterior recovery table."""
    r = build_report(conn)
    return {
        "run_id": r.run_id,
        "overall": lift_to_dict(r.overall),
        "lifts": [lift_to_dict(lift) for lift in r.lifts],
        "no_action_reasons": r.no_action_reasons,
        "candidate_validity": {
            "validity_rate": r.candidate_validity.validity_rate,
            "total_generated": r.candidate_validity.total_generated,
            "total_valid": r.candidate_validity.total_valid,
            "policy_violations": r.candidate_validity.policy_violations,
            "eligibility_blocked": r.candidate_validity.eligibility_blocked,
            "budget_blocked": r.candidate_validity.budget_blocked,
            "policy_compliance_rate": r.candidate_validity.policy_compliance_rate,
        },
        "budget_consumed": r.budget_consumed,
        "regret_curve": r.regret_curve,
        "regret_curve_all": r.regret_curve_all,
        "learning_curve": r.learning_curve,
        "posterior_recovery": r.posterior_recovery,
    }


@router.get("/api/decisions")
def list_decisions(
    status: str | None = Query(default=None),
    segment: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=1000),
    conn: sqlite3.Connection = Depends(get_conn),
) -> dict:
    clauses = []
    params: list[str] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if segment is not None:
        clauses.append("segment = ?")
        params.append(segment)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT decision_id, opportunity_id, run_id, segment, action_family, status, "  # noqa: S608
        f"no_action_reason, propensity, created_at FROM decisions {where} "
        # opportunity_id (content-addressed), not decision_id (random per
        # run), breaks a same-`created_at` tie -- see theatre.py/dashboard.py
        # for the same fix and why it matters for reproducibility.
        f"ORDER BY created_at DESC, opportunity_id DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    return {"decisions": [dict(r) for r in rows], "count": len(rows)}


@router.get("/api/decisions/{decision_id}")
def decision_trace(decision_id: str, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    """The full trace for one decision -- envelope, every candidate with its
    validator verdict, the chosen candidate, propensity, linked execution,
    and linked outcome. See `get_decision_trace` for the single query this
    endpoint, the CLI's `trace` subcommand, and nothing else, all share."""
    trace = get_decision_trace(conn, decision_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"no decision with id {decision_id!r}")
    return trace


@router.get("/api/opportunities")
def list_opportunities(
    arm: str | None = Query(default=None),
    window_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=1000),
    conn: sqlite3.Connection = Depends(get_conn),
) -> dict:
    clauses = []
    params: list[str] = []
    if arm is not None:
        clauses.append("arm = ?")
        params.append(arm)
    if window_id is not None:
        clauses.append("window_id = ?")
        params.append(window_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT opportunity_id, run_id, customer_id, window_id, segment, arm, assigned_at "  # noqa: S608
        f"FROM opportunities {where} ORDER BY assigned_at DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    return {"opportunities": [dict(r) for r in rows], "count": len(rows)}


@router.get("/api/posteriors")
def posteriors(conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    """The full 20-cell (segment, action_family) grid, including cells with
    no real evidence yet (they report their prior, `n_observed=0`) -- see
    `PosteriorStore.get_all`."""
    store = PosteriorStore(conn)
    rows = store.get_all()
    return {
        "posteriors": [
            {
                "segment": r.segment.value,
                "action_family": r.action_family.value,
                "alpha": r.alpha,
                "beta": r.beta,
                "n_observed": r.n_observed,
                "mean_revenue": r.mean_revenue,
            }
            for r in rows
        ]
    }


# `build_timeline` walks every reward in the run to rebuild the belief
# history, which costs a few seconds on a 90-day database -- fine once,
# unacceptable on every page load. The cache key is (run_id, decision count):
# a replay only ever appends, so a changed count is a changed run and any
# in-flight `revenew demo` invalidates this the moment it writes. Holding one
# built timeline is a few hundred KB, and the alternative -- recomputing on
# each request -- is what would make the theatre stutter live on stage.
_TIMELINE_CACHE: dict[tuple[str | None, int], dict] = {}


@router.get("/api/theatre")
def theatre(conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    """The run as a per-day timeline: cumulative counters, the 20-cell
    posterior grid rebuilt as of each day, and a sampled ticker of the
    offers the bandit actually sent. See `revenew/api/theatre.py` for why
    this is assembled server-side and animated client-side rather than
    streamed."""
    # Same 'live_%' and 'agent_%' exclusion as theatre.py's _latest_run_id() — a live
    # or agent decision must not bust this cache and point the Theatre at a
    # one-decision run.
    key = (
        conn.execute(
            "SELECT run_id FROM decisions WHERE run_id NOT LIKE 'live_%' AND run_id NOT LIKE 'agent_%' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        or {"run_id": None}
    )["run_id"]
    n = conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"]
    cached = _TIMELINE_CACHE.get((key, n))
    if cached is None:
        t = build_timeline(conn)
        cached = {"meta": t.meta, "frames": t.frames, "events": t.events, "truth": t.truth}
        _TIMELINE_CACHE.clear()
        _TIMELINE_CACHE[(key, n)] = cached
    return cached


@router.get("/api/regret")
def regret(conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    """The exported cumulative regret curve for the most recent replay run,
    if one has been exported (see harness/regret.py's `export_to_runtime`).
    Downsampled to at most 500 points, identical to what the dashboard's
    chart plots -- one series, not a separate full-resolution copy that could
    disagree with what's shown on screen."""
    r = build_report(conn)
    return {
        "run_id": r.run_id,
        "regret_curve": r.regret_curve,
        "regret_curve_all": r.regret_curve_all,
        "learning_curve": r.learning_curve,
    }
