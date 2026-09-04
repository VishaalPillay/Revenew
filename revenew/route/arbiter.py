"""AttributionArbiter: one action per (customer, window).

The detector can legitimately find a customer dormant AND a strong cross-sell
match in the same window -- both are true at once. F2 says the customer gets
at most one action, so this module picks a single winner per (customer,
window) among that customer's candidate rows, by rupees_at_risk descending,
ties broken by cohort_id.

The actual enforcement is `UNIQUE (run_id, customer_id, window_id)` on the
`opportunities` table in db/schema.sql. This module's ranking decides WHICH
candidate gets inserted; the database decides whether a second one is allowed
to be. If this module ever had a bug and tried to insert two winners for one
customer, the insert would fail rather than silently create a duplicate --
that is what test_arbiter_uniqueness checks.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from revenew.detect.detector import RawOpportunity, compute_segment_map
from revenew.models import Arm


@dataclass(frozen=True)
class ArbitratedOpportunity:
    opportunity_id: str
    run_id: str
    customer_id: str
    window_id: str
    segment: str
    arm: Arm
    assigned_at: str


def arbitrate(candidates: list[RawOpportunity]) -> list[RawOpportunity]:
    """One winner per (customer_id, window_id): highest rupees_at_risk, ties
    broken by cohort_id (lexicographic, so the choice is deterministic and
    reproducible under replay rather than depending on row order)."""
    best: dict[tuple[str, str], RawOpportunity] = {}
    for c in candidates:
        key = (c.customer_id, c.window_id)
        current = best.get(key)
        if current is None:
            best[key] = c
            continue
        if (c.rupees_at_risk, c.cohort_id) > (current.rupees_at_risk, current.cohort_id):
            best[key] = c
    return list(best.values())


def persist_winners(
    conn: sqlite3.Connection,
    winners: list[RawOpportunity],
    *,
    now,
    salt: str,
    control_pct: int,
) -> list[ArbitratedOpportunity]:
    """Insert arbitrated winners into `opportunities`, with segment and arm.

    A second call for a customer/window already present raises
    sqlite3.IntegrityError from the UNIQUE constraint -- deliberately not
    caught here. Silently swallowing it would hide a real arbitration bug
    behind a database that is doing exactly its job.
    """
    from revenew.clock import iso
    from revenew.route.arm import assign_arm

    seg_map = compute_segment_map(conn, now)
    assigned_at = iso(now)
    out: list[ArbitratedOpportunity] = []

    for w in winners:
        segment = seg_map.get(w.customer_id)
        if segment is None:
            continue  # customer has no order history at all; nothing to segment
        arm = assign_arm(w.customer_id, salt=salt, control_pct=control_pct)
        row = ArbitratedOpportunity(
            opportunity_id=w.opportunity_id,
            run_id=w.run_id,
            customer_id=w.customer_id,
            window_id=w.window_id,
            segment=segment.value,
            arm=arm,
            assigned_at=assigned_at,
        )
        conn.execute(
            """
            INSERT INTO opportunities
                (opportunity_id, run_id, customer_id, window_id, segment, arm, assigned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (row.opportunity_id, row.run_id, row.customer_id, row.window_id,
             row.segment, row.arm.value, row.assigned_at),
        )
        out.append(row)

    conn.commit()
    return out
