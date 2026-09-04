"""Assembles the read-only summary artifacts the dashboard renders.

Kept separate from `revenew/api/dashboard.py` on purpose: this module knows
how to compute a report from a `revenew.db` connection and nothing about
HTTP, templates, or FastAPI. That split is what makes these numbers
independently testable (and reusable from a CLI or a future export job)
without spinning up a server.
"""

from __future__ import annotations

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
