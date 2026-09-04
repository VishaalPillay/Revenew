"""Reserve / (implicit commit) / release. See db/schema.sql's `budget_ledger`
comment for why a successful execution needs no second write: the reservation
already holds the estimated cost, so success just means nothing reverses it.

This is what makes a crash between reserve and execute hold budget rather than
lose it (SYSTEM_DESIGN.md section 1.2) -- the money is already accounted for
the instant the decision is made, before Razorpay is ever called. If execution
then fails, `release` writes the reversing entry and the hold disappears. If
the process dies before either happens, the reservation simply stays in place
until a reconciler (see revenew/ledger for the sweep, when built) notices it
and releases it -- but the budget was never silently double-spent in the
meantime, which is the property that actually matters.
"""

from __future__ import annotations

import sqlite3

from revenew.clock import iso


def consumed(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT consumed FROM v_budget_consumed").fetchone()
    return float(row["consumed"]) if row and row["consumed"] is not None else 0.0


def available(conn: sqlite3.Connection, budget_cap: float) -> float:
    return budget_cap - consumed(conn)


def reserve(conn: sqlite3.Connection, decision_id: str, amount: float, *, now) -> None:
    """Write a reservation. Callers must check `available()` against the
    envelope BEFORE calling this -- reserve() itself does not re-check,
    because the envelope already did, and re-checking here would be a second
    source of truth for the same rule."""
    if amount < 0:
        raise ValueError("cannot reserve a negative amount")
    conn.execute(
        "INSERT INTO budget_ledger (decision_id, status, amount, created_at) VALUES (?, 'reserved', ?, ?)",
        (decision_id, -amount, iso(now)),
    )
    conn.commit()


def release(conn: sqlite3.Connection, decision_id: str, amount: float, *, now) -> None:
    """Reverse a reservation that will not be spent (execution failed, or the
    decision never executed at all)."""
    if amount < 0:
        raise ValueError("cannot release a negative amount")
    conn.execute(
        "INSERT INTO budget_ledger (decision_id, status, amount, created_at) VALUES (?, 'released', ?, ?)",
        (decision_id, amount, iso(now)),
    )
    conn.commit()


def reserved_amount(conn: sqlite3.Connection, decision_id: str) -> float:
    """Sum of reservations for one decision, positive. Used to release exactly
    what was reserved rather than a caller-recomputed (and possibly stale)
    figure."""
    row = conn.execute(
        "SELECT COALESCE(SUM(-amount), 0) AS r FROM budget_ledger "
        "WHERE decision_id = ? AND status = 'reserved'",
        (decision_id,),
    ).fetchone()
    already_released = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS r FROM budget_ledger "
        "WHERE decision_id = ? AND status = 'released'",
        (decision_id,),
    ).fetchone()
    return float(row["r"]) - float(already_released["r"])
