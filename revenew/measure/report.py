"""Assembles the read-only summary artifacts the dashboard renders.

Kept separate from `revenew/api/dashboard.py` on purpose: this module knows
how to compute a report from a `revenew.db` connection and nothing about
HTTP, templates, or FastAPI. That split is what makes these numbers
independently testable (and reusable from a CLI or a future export job)
without spinning up a server.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from revenew.measure.incremental import SegmentLift, compute_lift, overall_lift
from revenew.models import Segment


@dataclass(frozen=True)
class CandidateValidity:
    validity_rate: float | None
    total_generated: int
    total_valid: int


@dataclass(frozen=True)
class Report:
    lifts: list[SegmentLift]
    overall: SegmentLift
    no_action_reasons: list[dict] = field(default_factory=list)
    candidate_validity: CandidateValidity = field(
        default_factory=lambda: CandidateValidity(None, 0, 0)
    )
    budget_consumed: float = 0.0
    run_id: str | None = None
    regret_curve: list[dict] = field(default_factory=list)
    posterior_recovery: list[dict] = field(default_factory=list)


def lift_to_dict(lift: SegmentLift) -> dict:
    """JSON-safe shape for one `SegmentLift` -- shared by `revenew/api/read.py`
    and `revenew/cli.py`'s `report --json`, so the two never quietly diverge
    on which fields a lift row carries."""
    return {
        "segment": lift.segment.value if lift.segment else None,
        "n_treatment": lift.n_treatment,
        "n_control": lift.n_control,
        "mean_treatment": lift.mean_treatment,
        "mean_control": lift.mean_control,
        "lift": lift.lift,
        "ci_low": lift.ci_low,
        "ci_high": lift.ci_high,
        "p_value": lift.p_value,
        "is_significant": lift.is_significant,
    }


def _no_action_reasons(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM v_no_action_reasons ORDER BY n DESC").fetchall()
    return [dict(r) for r in rows]


def _candidate_validity(conn: sqlite3.Connection) -> CandidateValidity:
    row = conn.execute("SELECT * FROM v_candidate_validity").fetchone()
    if row is None or row["validity_rate"] is None:
        return CandidateValidity(None, 0, 0)
    return CandidateValidity(row["validity_rate"], row["total_generated"], row["total_valid"])


def _budget_consumed(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT consumed FROM v_budget_consumed").fetchone()
    return float(row["consumed"]) if row and row["consumed"] is not None else 0.0


def _latest_run_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT run_id FROM demo_regret_curve ORDER BY decision_index DESC LIMIT 1"
    ).fetchone()
    return row["run_id"] if row else None


def _regret_curve(conn: sqlite3.Connection, run_id: str | None, *, max_points: int = 500) -> list[dict]:
    if run_id is None:
        return []
    rows = conn.execute(
        "SELECT decision_index, cumulative_regret FROM demo_regret_curve "
        "WHERE run_id = ? ORDER BY decision_index ASC",
        (run_id,),
    ).fetchall()
    points = [dict(r) for r in rows]
    if len(points) > max_points:
        step = len(points) // max_points
        points = points[::step]
    return points


def _posterior_recovery(conn: sqlite3.Connection, run_id: str | None) -> list[dict]:
    if run_id is None:
        return []
    rows = conn.execute(
        "SELECT * FROM demo_posterior_recovery WHERE run_id = ? ORDER BY segment, action_family",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def build_report(conn: sqlite3.Connection) -> Report:
    """Everything the dashboard needs, in one read-only pass over revenew.db.
    Never opens harness.db -- `demo_regret_curve`/`demo_posterior_recovery`
    are already-exported artifacts; see harness/regret.py."""
    run_id = _latest_run_id(conn)
    return Report(
        lifts=compute_lift(conn, list(Segment)),
        overall=overall_lift(conn),
        no_action_reasons=_no_action_reasons(conn),
        candidate_validity=_candidate_validity(conn),
        budget_consumed=_budget_consumed(conn),
        run_id=run_id,
        regret_curve=_regret_curve(conn, run_id),
        posterior_recovery=_posterior_recovery(conn, run_id),
    )


def get_decision_trace(conn: sqlite3.Connection, decision_id: str) -> dict | None:
    """The full audit trail for one decision: F9's promise ("every decision's
    full trace, including the propensity of the chosen arm") finally made
    retrievable, not just persisted. `None` if no such decision exists.

    One function, two callers -- `revenew/api/read.py`'s `/api/decisions/{id}`
    and `revenew/cli.py`'s `trace` subcommand both call this, matching the
    single-source-of-truth pattern the rest of this codebase already uses for
    `EnvelopeEngine`/`EnvelopeValidator`, `decide_one_opportunity`, and
    `PosteriorStore.apply_outcome` -- there is no second, HTTP-shaped or
    CLI-shaped copy of this query to drift out of sync with the other.
    """
    decision = conn.execute(
        "SELECT * FROM decisions WHERE decision_id = ?", (decision_id,)
    ).fetchone()
    if decision is None:
        return None

    opportunity = conn.execute(
        "SELECT customer_id, window_id, arm, assigned_at FROM opportunities WHERE opportunity_id = ?",
        (decision["opportunity_id"],),
    ).fetchone()

    candidates = conn.execute(
        "SELECT candidate_index, candidate_json, valid, violations_json FROM decision_candidates "
        "WHERE decision_id = ? ORDER BY candidate_index ASC",
        (decision_id,),
    ).fetchall()

    execution = conn.execute(
        "SELECT execution_id, idempotency_key, provider_ref, status, created_at FROM executions "
        "WHERE decision_id = ?",
        (decision_id,),
    ).fetchone()

    outcome = conn.execute(
        "SELECT outcome_seq, converted, net_revenue, censored, closed_at FROM outcomes "
        "WHERE opportunity_id = ?",
        (decision["opportunity_id"],),
    ).fetchone()

    return {
        "decision_id": decision["decision_id"],
        "opportunity_id": decision["opportunity_id"],
        "run_id": decision["run_id"],
        "customer_id": opportunity["customer_id"] if opportunity else None,
        "window_id": opportunity["window_id"] if opportunity else None,
        "arm": opportunity["arm"] if opportunity else None,
        "segment": decision["segment"],
        "action_family": decision["action_family"],
        "status": decision["status"],
        "no_action_reason": decision["no_action_reason"],
        "propensity": decision["propensity"],
        "envelope": json.loads(decision["envelope_json"]),
        "candidates_generated": decision["candidates_generated"],
        "candidates_valid": decision["candidates_valid"],
        "candidates": [
            {
                "candidate_index": c["candidate_index"],
                "candidate": json.loads(c["candidate_json"]),
                "valid": bool(c["valid"]),
                "violations": json.loads(c["violations_json"]),
            }
            for c in candidates
        ],
        "chosen_candidate": (
            json.loads(decision["chosen_candidate_json"])
            if decision["chosen_candidate_json"]
            else None
        ),
        "execution": dict(execution) if execution else None,
        "outcome": dict(outcome) if outcome else None,
        "created_at": decision["created_at"],
    }
