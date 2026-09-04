"""OutcomeLedger: the one function allowed to INSERT into `outcomes`.

Appends the outcome row, then -- if and only if a decision exists and it
actually chose an action -- feeds the bandit through the exact same
`PosteriorStore.apply_outcome` call that `ledger/replay.py` uses when
rebuilding posteriors from the log. Same function, two call sites (live and
replay); that identity is what makes replay equality checkable at all rather
than merely hoped for.

Control-arm opportunities record an outcome with `decision_id=None` and never
touch the bandit -- they exist so IncrementalEstimator has a counterfactual,
not so the bandit can learn from them. Feeding the bandit from control-arm
outcomes would contaminate the very comparison the control arm exists to make.
"""

from __future__ import annotations

import sqlite3

from revenew.decide.bandit import PosteriorStore
from revenew.models import ActionFamily, Segment


def record_outcome(
    conn: sqlite3.Connection,
    *,
    opportunity_id: str,
    decision_id: str | None,
    converted: bool,
    net_revenue: float,
    censored: bool,
    closed_at: str,
) -> int:
    """Append one outcome. Returns the assigned outcome_seq.

    Raises sqlite3.IntegrityError on a duplicate opportunity_id (the UNIQUE
    constraint) rather than silently upserting -- an attempt to close the same
    opportunity's window twice is a bug upstream, not something to paper over.
    """
    cur = conn.execute(
        """
        INSERT INTO outcomes (opportunity_id, decision_id, converted, net_revenue, censored, closed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (opportunity_id, decision_id, int(converted), net_revenue, int(censored), closed_at),
    )
    outcome_seq = cur.lastrowid
    conn.commit()

    if decision_id is not None:
        _feed_bandit(conn, decision_id, converted=converted, net_revenue=net_revenue, outcome_seq=outcome_seq)

    return outcome_seq


def _feed_bandit(
    conn: sqlite3.Connection,
    decision_id: str,
    *,
    converted: bool,
    net_revenue: float,
    outcome_seq: int,
) -> None:
    row = conn.execute(
        "SELECT segment, action_family FROM decisions WHERE decision_id = ?",
        (decision_id,),
    ).fetchone()
    if row is None or row["action_family"] is None:
        # A no_action decision has no action_family -- nothing for the bandit
        # to update. This is expected, not an error: doing nothing produces an
        # outcome (usually censored) but teaches the bandit nothing about any
        # family, because no family was chosen.
        return

    store = PosteriorStore(conn)
    store.apply_outcome(
        Segment(row["segment"]),
        ActionFamily(row["action_family"]),
        converted=converted,
        net_revenue=net_revenue,
        outcome_seq=outcome_seq,
    )
    conn.commit()
