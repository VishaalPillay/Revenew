"""Reconciler: the "no money lost to crashes" guarantee's other half.

`decide_one_opportunity` (decide/__init__.py) already releases a reservation
synchronously the instant it KNOWS an execution failed. This module exists
for what it cannot know about: the process itself dying somewhere between
`budget.reserve` and either outcome. Per SYSTEM_DESIGN.md section 7's
failure table -- "Crash between reserve and commit ->
`action.status='pending'` older than timeout -> reconciler releases the
hold" -- a decision left `pending` for longer than `timeout_minutes` with no
matching `executions` row never actually spent anything, so its hold is
released back to the budget.

The other case a genuine crash can produce -- execution SUCCEEDED (an
`executions` row exists, `status` in `('sent', 'confirmed')`) but the
process died before `trace.mark_executed` flipped `decisions.status` --
must NOT have its budget released; the money really was spent. `reconcile()`
fixes that row forward to `executed` instead.

A third window this also covers: `execute_decision` can return a `failed`
result and `decide_one_opportunity` release its hold synchronously in the
very next line -- but if the process dies in between those two statements,
the `executions` row says `failed` while the reservation is still live.
`reserved_amount()` nets reserve/release either way, so sweeping a
decision whose hold was already released synchronously is a harmless
zero-amount no-op, not a double release.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from revenew.clock import iso
from revenew.decide.trace import mark_executed
from revenew.execute import budget

DEFAULT_TIMEOUT_MINUTES = 30


@dataclass(frozen=True)
class ReconcileResult:
    fixed_forward: int  # pending decisions whose execution had actually succeeded
    released: int  # pending decisions with no execution at all -- hold released
    released_total: float


def reconcile(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
) -> ReconcileResult:
    """Sweeps every `pending` decision older than `timeout_minutes`.

    Idempotent: a decision already fixed forward to `executed` no longer
    matches the `status = 'pending'` filter on a second call, and
    `budget.reserved_amount` already nets reserve/release, so re-running
    this against a decision with nothing left to release costs one harmless
    zero-amount ledger row rather than double-releasing.
    """
    cutoff = iso(now - timedelta(minutes=timeout_minutes))

    succeeded_but_unflipped = conn.execute(
        """
        SELECT d.decision_id
        FROM decisions d
        JOIN executions e ON e.decision_id = d.decision_id
        WHERE d.status = 'pending' AND d.created_at < ?
          AND e.status IN ('sent', 'confirmed')
        """,
        (cutoff,),
    ).fetchall()
    for row in succeeded_but_unflipped:
        mark_executed(conn, row["decision_id"])

    never_executed = conn.execute(
        """
        SELECT d.decision_id
        FROM decisions d
        LEFT JOIN executions e ON e.decision_id = d.decision_id
        WHERE d.status = 'pending' AND d.created_at < ?
          AND (e.execution_id IS NULL OR e.status = 'failed')
        """,
        (cutoff,),
    ).fetchall()

    released_total = 0.0
    for row in never_executed:
        decision_id = row["decision_id"]
        held = budget.reserved_amount(conn, decision_id)
        if held > 0:
            budget.release(conn, decision_id, held, now=now)
            released_total += held

    return ReconcileResult(
        fixed_forward=len(succeeded_but_unflipped),
        released=len(never_executed),
        released_total=released_total,
    )
