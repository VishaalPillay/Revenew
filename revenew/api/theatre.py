"""The Agent Theatre timeline: the run, replayed as something you can watch.

Every other read surface in this package answers "what did the system end up
believing?". This one answers "when did it come to believe it?" -- which is
the question the whole measurement apparatus exists to make answerable, and
the one a static table cannot show.

Two deliberate choices are worth stating, because both could reasonably have
gone the other way:

**Nothing here is recomputed, simulated, or paced by the server.** The
timeline is assembled from rows already in `revenew.db` -- `decisions`,
`outcomes`, `budget_ledger`, and the exported `demo_*` artifacts -- and
handed to the client in one response. The animation is the browser walking a
list of real frames, not a server dribbling out invented ones. That makes the
theatre scrubbable, pausable, and instant on a second view, and it means a
network hiccup mid-demo pauses a local animation instead of stalling a live
stream. It also keeps the honesty claim simple: there is no code path here
that can show a number the database does not already contain.

**Posterior evolution is rebuilt by replaying outcomes, not stored.** The
`posteriors` table holds final state only. But `db/schema.sql` calls that
table "a derived cache over (segment, action_family), fully rebuildable from
`outcomes`" -- so the per-day history is recoverable by doing exactly what
`ledger/replay.py` does: walk `outcomes` in `outcome_seq` order and apply the
same +1 update, from the same priors in `decide/bandit.py`. The last frame
this module emits is therefore not merely consistent with `posteriors`, it is
arithmetically identical to it, and `tests/test_theatre.py` asserts that
rather than trusting it.

Frames are keyed by `outcomes.closed_at`, not by decision date. That is what
makes the 7-day feedback lag visible: the decision ticker for a day runs
ahead of the belief grid that day's rewards will eventually move. The
ENGINEERING_LOG's last entry is about exactly that lag being invisible at a
30-day horizon; here it is on screen.

Like `measure/report.py`, this module knows how to compute from a connection
and nothing about HTTP -- `revenew/api/read.py` is what exposes it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from revenew.decide.bandit import prior_for
from revenew.models import ActionFamily, Segment

# The 20-cell grid, in one fixed order that every frame's `cells` array
# indexes into positionally. Emitting cells as bare triples against a shared
# order rather than as 20 keyed objects per frame is what keeps a 90-frame
# payload at tens of kilobytes instead of megabytes.
SEGMENTS: list[str] = [s.value for s in Segment]
FAMILIES: list[str] = [f.value for f in ActionFamily]
CELL_ORDER: list[tuple[str, str]] = [(s, f) for s in SEGMENTS for f in FAMILIES]
CELL_INDEX: dict[tuple[str, str], int] = {c: i for i, c in enumerate(CELL_ORDER)}

# Events exist to make the ticker feel like a run rather than a progress bar,
# so they are sampled per-day rather than globally.
#
# The cap is deliberately generous because this run's actions are not spread
# evenly: `max_offers_per_customer_per_month = 1` collapses execution into
# monthly bursts -- 1,842 offers go out on Jan 1, the cohort then sits in
# cooldown for four weeks, and the next wave lands on Feb 1. 65 of 90 days
# execute anything at all, and 164,441 of 170,160 decisions are a cooldown
# `no_action`. A global sample would flatten that into a uniform drizzle and
# quietly misrepresent how the system actually behaves; per-day sampling with
# a high ceiling keeps the bursts looking like bursts and the quiet weeks
# looking quiet, which is the truthful shape.
EVENTS_PER_DAY = 40


@dataclass(frozen=True)
class Timeline:
    meta: dict
    frames: list[dict]
    events: list[dict]
    truth: list[dict]


def _latest_run_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        # opportunity_id, not decision_id: the latter is a random uuid4 per
        # run, so tie-breaking on it can pick a different row (and therefore,
        # in principle, a different run_id) between two otherwise-identical
        # replays of the same seed.
        #
        # 'live_%' and 'agent_%' exclusion: Phase 3's live-decision endpoint and
        # Phase 4's agent-commerce endpoint write under run_id = f"live_{...}" and
        # f"agent_{...}". Without this filter, ONE live or agent decision
        # made "now" sorts after the Jan-Mar replay, flips the Theatre to a
        # one-decision "run", and the whole dashboard renders empty.
        "SELECT run_id FROM decisions WHERE run_id NOT LIKE 'live_%' AND run_id NOT LIKE 'agent_%' "
        "ORDER BY created_at DESC, opportunity_id DESC LIMIT 1"
    ).fetchone()
    return row["run_id"] if row else None


def _day_axis(conn: sqlite3.Connection, run_id: str) -> list[str]:
    """Every distinct day on which *either* a decision was made or an outcome
    closed. Built from a union rather than from decisions alone because the
    tail of the run is all reward and no new decisions -- cutting the axis at
    the last decision would hide the final week of learning."""
    rows = conn.execute(
        "SELECT DISTINCT substr(created_at, 1, 10) AS d FROM decisions WHERE run_id = ? "
        "UNION "
        "SELECT DISTINCT substr(o.closed_at, 1, 10) AS d FROM outcomes o "
        "JOIN opportunities p ON p.opportunity_id = o.opportunity_id WHERE p.run_id = ? "
        "ORDER BY d",
        (run_id, run_id),
    ).fetchall()
    return [r["d"] for r in rows]


def _decisions_by_day(conn: sqlite3.Connection, run_id: str) -> dict[str, dict]:
    rows = conn.execute(
        "SELECT substr(created_at, 1, 10) AS d, "
        "       COUNT(*) AS n, "
        "       SUM(CASE WHEN status = 'executed' THEN 1 ELSE 0 END) AS executed, "
        "       SUM(CASE WHEN status = 'no_action' THEN 1 ELSE 0 END) AS no_action, "
        "       SUM(candidates_generated) AS generated, "
        "       SUM(candidates_valid) AS valid "
        "FROM decisions WHERE run_id = ? GROUP BY d",
        (run_id,),
    ).fetchall()
    return {r["d"]: dict(r) for r in rows}


def _budget_by_day(conn: sqlite3.Connection) -> dict[str, float]:
    """Consumed rupees per day, sign-flipped to match `v_budget_consumed`
    (reservations are stored negative; a release is the reversing positive)."""
    rows = conn.execute(
        "SELECT substr(created_at, 1, 10) AS d, COALESCE(SUM(-amount), 0) AS consumed "
        "FROM budget_ledger GROUP BY d"
    ).fetchall()
    return {r["d"]: float(r["consumed"]) for r in rows}


def _regret_by_day(conn: sqlite3.Connection, run_id: str) -> dict[str, dict]:
    """`demo_regret_curve` is indexed by decision ordinal, not by date, so it
    is joined back through `decisions` to land on the day axis. The value
    taken per day is the LAST cumulative figure of that day -- a cumulative
    series must be sampled at its endpoint, never averaged.

    The bandit columns are deliberately left NULL rather than coalesced to
    zero. `bandit_cumulative_regret` is populated only on rows where the
    bandit actually chose, and on this run most days contain no bandit
    decision at all -- every customer is in cooldown. Folding those NULLs to
    0 makes the aggregate collapse to zero on every quiet day, and the
    resulting series saws between the real total and nothing roughly ninety
    times. It still looks like a chart, which is what makes the mistake
    dangerous. Returning None here lets the caller carry the last real value
    forward, which is what a cumulative series does when nothing happens."""
    rows = conn.execute(
        "SELECT substr(d.created_at, 1, 10) AS day, "
        "       MAX(r.decision_index) AS decision_index, "
        "       MAX(r.cumulative_regret) AS cum_regret, "
        "       MAX(r.bandit_decision_index) AS bandit_index, "
        "       MAX(r.bandit_cumulative_regret) AS bandit_cum_regret "
        "FROM demo_regret_curve r JOIN decisions d ON d.decision_id = r.decision_id "
        "WHERE r.run_id = ? GROUP BY day ORDER BY day",
        (run_id,),
    ).fetchall()
    return {r["day"]: dict(r) for r in rows}


def _truth(conn: sqlite3.Connection, run_id: str) -> dict[tuple[str, str], float]:
    """Ground-truth conversion rates, read from the artifact the harness
    already exported into this database. This module does not open
    harness.db, and could not grade anything if it tried -- it is reading a
    scalar the harness chose to publish, exactly as the dashboard does."""
    rows = conn.execute(
        "SELECT segment, action_family, p_true FROM demo_posterior_recovery WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    return {(r["segment"], r["action_family"]): float(r["p_true"]) for r in rows}


def _events(conn: sqlite3.Connection, run_id: str) -> dict[str, list[dict]]:
    """Up to `EVENTS_PER_DAY` executed decisions per day, with the offer the
    bandit actually chose and -- where the window has since closed -- what
    happened. `chosen_candidate_json` is the same blob `get_decision_trace`
    returns, so a ticker row and the trace panel can never disagree about
    what was sent."""
    rows = conn.execute(
        "SELECT d.decision_id, substr(d.created_at, 1, 10) AS day, d.segment, "
        "       d.action_family, d.chosen_candidate_json, d.propensity, "
        "       d.candidates_generated, d.candidates_valid, "
        "       o.converted, o.net_revenue "
        "FROM decisions d LEFT JOIN outcomes o ON o.decision_id = d.decision_id "
        "WHERE d.run_id = ? AND d.status = 'executed' "
        # opportunity_id, not decision_id -- see _latest_run_id's comment
        # above. Here it decides which offers land in the EVENTS_PER_DAY
        # sample on a burst day; without this fix, which specific offers
        # get shown in the ticker was not reproducible run to run.
        "ORDER BY d.created_at ASC, d.opportunity_id ASC",
        (run_id,),
    ).fetchall()

    by_day: dict[str, list[dict]] = {}
    for r in rows:
        bucket = by_day.setdefault(r["day"], [])
        if len(bucket) >= EVENTS_PER_DAY:
            continue
        chosen = json.loads(r["chosen_candidate_json"]) if r["chosen_candidate_json"] else {}
        bucket.append(
            {
                "id": r["decision_id"],
                "segment": r["segment"],
                "family": r["action_family"],
                "headline": chosen.get("headline", ""),
                "discount_pct": chosen.get("discount_pct"),
                "discount_amount": chosen.get("discount_amount"),
                "propensity": r["propensity"],
                "generated": r["candidates_generated"],
                "valid": r["candidates_valid"],
                "converted": None if r["converted"] is None else bool(r["converted"]),
                "revenue": r["net_revenue"],
            }
        )
    return by_day


def _outcome_walk(conn: sqlite3.Connection, run_id: str) -> list[tuple[str, int, int]]:
    """Every outcome that moves a posterior, in the order the ledger applied
    it: (day it closed, cell index, converted). Control-arm outcomes and
    no-action decisions are absent by construction -- they have no
    `action_family` to attribute a reward to, which is the same reason
    `PosteriorStore.apply_outcome` never sees them."""
    rows = conn.execute(
        "SELECT substr(o.closed_at, 1, 10) AS day, d.segment, d.action_family, o.converted "
        "FROM outcomes o JOIN decisions d ON d.decision_id = o.decision_id "
        "WHERE d.run_id = ? AND d.action_family IS NOT NULL "
        "ORDER BY o.outcome_seq ASC",
        (run_id,),
    ).fetchall()
    return [
        (r["day"], CELL_INDEX[(r["segment"], r["action_family"])], int(r["converted"]))
        for r in rows
        if (r["segment"], r["action_family"]) in CELL_INDEX
    ]


def build_timeline(conn: sqlite3.Connection) -> Timeline:
    """One frame per day of the run, each carrying the full 20-cell belief
    state as of that day's close, plus the counters and the sampled ticker.

    Cost is one pass over each contributing table and one pass over the
    reward stream; nothing is quadratic in the number of frames, so this
    stays flat as runs get longer."""
    run_id = _latest_run_id(conn)
    if run_id is None:
        return Timeline(meta={"run_id": None, "days": 0}, frames=[], events=[], truth=[])

    days = _day_axis(conn, run_id)
    per_day = _decisions_by_day(conn, run_id)
    budget = _budget_by_day(conn)
    regret = _regret_by_day(conn, run_id)
    truth = _truth(conn, run_id)
    events_by_day = _events(conn, run_id)

    # Priors first, then replay. `prior_for` is imported rather than
    # re-stated so a change to the cold-start policy cannot leave this
    # module quietly animating from the wrong starting belief.
    alpha = [prior_for(ActionFamily(f))[0] for _, f in CELL_ORDER]
    beta = [prior_for(ActionFamily(f))[1] for _, f in CELL_ORDER]
    observed = [0] * len(CELL_ORDER)

    walk = _outcome_walk(conn, run_id)
    walk_pos = 0

    truth_vec = [truth.get(cell) for cell in CELL_ORDER]

    frames: list[dict] = []
    events: list[dict] = []
    cum_decisions = cum_executed = cum_no_action = cum_outcomes = 0
    cum_generated = cum_valid = 0
    cum_budget = 0.0
    # Carried forward independently: a day can advance the all-decisions
    # regret total while contributing nothing to the bandit-only series.
    cum_regret = 0.0
    bandit_regret = 0.0
    bandit_decisions = 0

    for i, day in enumerate(days):
        d = per_day.get(day)
        if d:
            cum_decisions += d["n"] or 0
            cum_executed += d["executed"] or 0
            cum_no_action += d["no_action"] or 0
            cum_generated += d["generated"] or 0
            cum_valid += d["valid"] or 0
        cum_budget += budget.get(day, 0.0)
        row = regret.get(day)
        if row is not None:
            if row["cum_regret"] is not None:
                cum_regret = float(row["cum_regret"])
            if row["bandit_cum_regret"] is not None:
                bandit_regret = float(row["bandit_cum_regret"])
            if row["bandit_index"] is not None:
                bandit_decisions = int(row["bandit_index"])

        # Apply every reward that closed on this day, in ledger order.
        while walk_pos < len(walk) and walk[walk_pos][0] <= day:
            _, cell, converted = walk[walk_pos]
            if converted:
                alpha[cell] += 1
            else:
                beta[cell] += 1
            observed[cell] += 1
            cum_outcomes += 1
            walk_pos += 1

        # Mean |p_hat - p_true| across all 20 cells -- the same quantity the
        # posterior-recovery table reports, watched as it falls.
        errors = [
            abs(alpha[k] / (alpha[k] + beta[k]) - truth_vec[k])
            for k in range(len(CELL_ORDER))
            if truth_vec[k] is not None
        ]
        live = [
            abs(alpha[k] / (alpha[k] + beta[k]) - truth_vec[k])
            for k in range(len(CELL_ORDER))
            if truth_vec[k] is not None and observed[k] > 0
        ]

        for e in events_by_day.get(day, []):
            events.append({**e, "f": i, "day": day})

        frames.append(
            {
                "i": i,
                "day": day,
                "decisions": cum_decisions,
                "executed": cum_executed,
                "no_action": cum_no_action,
                "outcomes": cum_outcomes,
                "generated": cum_generated,
                "valid": cum_valid,
                "budget": round(cum_budget, 2),
                "cum_regret": round(cum_regret, 2),
                "bandit_cum_regret": round(bandit_regret, 2),
                "bandit_decisions": bandit_decisions,
                "mean_error": round(sum(errors) / len(errors), 5) if errors else None,
                "mean_error_live": round(sum(live) / len(live), 5) if live else None,
                "cells": [
                    [round(alpha[k], 1), round(beta[k], 1), observed[k]]
                    for k in range(len(CELL_ORDER))
                ],
            }
        )

    return Timeline(
        meta={
            "run_id": run_id,
            "days": len(days),
            "day_start": days[0] if days else None,
            "day_end": days[-1] if days else None,
            "segments": SEGMENTS,
            "families": FAMILIES,
            "cell_order": [list(c) for c in CELL_ORDER],
            "total_decisions": cum_decisions,
            "total_executed": cum_executed,
            "total_outcomes": cum_outcomes,
            "total_candidates": cum_generated,
        },
        frames=frames,
        events=events,
        truth=[
            {"segment": s, "action_family": f, "p_true": truth.get((s, f))}
            for s, f in CELL_ORDER
        ],
    )
