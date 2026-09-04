"""Rebuild `posteriors` from `outcomes`, seq 1..N. Must land byte-identical to
whatever the live path produced -- that equality IS the reproducibility claim
in SYSTEM_DESIGN.md section 1.2, not just a nice property to have.

The mechanism that makes this trustworthy rather than merely convenient:
`rebuild_posteriors` calls the exact same `PosteriorStore.apply_outcome` that
`ledger/outcome.py`'s live path calls, in the same order the log recorded
them. There are not two update rules that happen to agree today -- there is
one rule, exercised twice.
"""

from __future__ import annotations

import sqlite3

from revenew.decide.bandit import PosteriorStore
from revenew.models import ActionFamily, Segment


def rebuild_posteriors(conn: sqlite3.Connection) -> PosteriorStore:
    """Wipe and recompute `posteriors` from scratch, from the outcome log alone."""
    store = PosteriorStore(conn)
    conn.execute("DELETE FROM posteriors")
    conn.commit()
    store.ensure_initialized()

    rows = conn.execute(
        """
        SELECT o.outcome_seq, o.converted, o.net_revenue, d.segment, d.action_family
        FROM outcomes o
        JOIN decisions d ON d.decision_id = o.decision_id
        WHERE o.decision_id IS NOT NULL AND d.action_family IS NOT NULL
        ORDER BY o.outcome_seq ASC
        """
    ).fetchall()

    for row in rows:
        store.apply_outcome(
            Segment(row["segment"]),
            ActionFamily(row["action_family"]),
            converted=bool(row["converted"]),
            net_revenue=row["net_revenue"],
            outcome_seq=row["outcome_seq"],
        )
    conn.commit()
    return store


def posteriors_snapshot(conn: sqlite3.Connection) -> dict[tuple[str, str], tuple[float, float, float, int, int]]:
    """A comparable, order-independent snapshot of the whole posteriors table.

    Used by test_replay_equality to compare the live-updated table against a
    freshly rebuilt one without depending on row insertion order.
    """
    rows = conn.execute(
        "SELECT segment, action_family, alpha, beta, revenue_sum, revenue_n, updated_through_seq "
        "FROM posteriors"
    ).fetchall()
    return {
        (r["segment"], r["action_family"]): (
            r["alpha"], r["beta"], r["revenue_sum"], r["revenue_n"], r["updated_through_seq"]
        )
        for r in rows
    }
