"""`recommended_sku` used to be computed by `cross_sell_affinity`'s query and
discarded at the final SELECT -- the query's own `ranked_recommendations` CTE
built it, then `queries.sql`'s two-column contract dropped it on the floor
every run. This is the regression test for its recovery: it must survive
detection, land on `RawOpportunity`, and persist into
`opportunity_candidates`, and it must stay NULL for the two opportunity
types that never had a product recommendation to make.
"""

from __future__ import annotations

from datetime import UTC, datetime

from revenew.detect.detector import OpportunityDetector
from revenew.models import OpportunityType

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_cross_sell_hits_carry_a_real_recommended_sku(seeded_conn):
    detector = OpportunityDetector()
    hits = detector.detect(seeded_conn, run_id="r1", window_id="w1", now=NOW)

    cross_sell = [h for h in hits if h.opportunity_type == OpportunityType.CROSS_SELL_AFFINITY]
    assert cross_sell, "the seeded fixture must produce at least one cross-sell hit for this test to mean anything"

    for h in cross_sell:
        assert h.recommended_sku is not None
        assert h.recommended_sku.startswith("SKU-")


def test_other_opportunity_types_never_fabricate_a_recommended_sku(seeded_conn):
    """dormant_winback and first_order_retention have no product recommendation
    logic at all -- their query blocks never select the column. A `None`
    default here, not an accidental carry-over from a prior row, is what
    proves detector.py reads `recommended_sku` per-row via `row.keys()`
    rather than assuming every query provides it."""
    detector = OpportunityDetector()
    hits = detector.detect(seeded_conn, run_id="r1", window_id="w1", now=NOW)

    non_cross_sell = [h for h in hits if h.opportunity_type != OpportunityType.CROSS_SELL_AFFINITY]
    assert non_cross_sell, "the seeded fixture must also produce non-cross-sell hits"
    assert all(h.recommended_sku is None for h in non_cross_sell)


def test_recommended_sku_persists_into_opportunity_candidates(seeded_conn):
    """The column exists in the schema and detector.py's INSERT actually
    writes it -- not just that the in-memory dataclass carries the value."""
    detector = OpportunityDetector()
    hits = detector.detect(seeded_conn, run_id="r1", window_id="w1", now=NOW)
    detector.persist_candidates(seeded_conn, hits)

    cross_sell_ids = [h.opportunity_id for h in hits if h.opportunity_type == OpportunityType.CROSS_SELL_AFFINITY]
    assert cross_sell_ids

    rows = seeded_conn.execute(
        "SELECT opportunity_id, recommended_sku FROM opportunity_candidates "
        "WHERE opportunity_id IN ({})".format(",".join("?" * len(cross_sell_ids))),
        cross_sell_ids,
    ).fetchall()
    assert len(rows) == len(cross_sell_ids)
    assert all(r["recommended_sku"] is not None for r in rows)

    dormant_id = next(h.opportunity_id for h in hits if h.opportunity_type == OpportunityType.DORMANT_WINBACK)
    row = seeded_conn.execute(
        "SELECT recommended_sku FROM opportunity_candidates WHERE opportunity_id = ?", (dormant_id,)
    ).fetchone()
    assert row["recommended_sku"] is None
