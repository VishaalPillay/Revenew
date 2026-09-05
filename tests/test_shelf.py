"""ShelfGenerator (decide/shelf.py): Arm B's candidate source in PLAN.md
section 5's three-arm ablation. Five templated candidates, one per family,
no LLM call, cohort-level (not customer-level) memoisation, and BUNDLE_OFFER
omitted -- never downgraded to a nudge -- when no global affinity pair
survives the confidence threshold.
"""

from __future__ import annotations

from pathlib import Path

from revenew.db import connect as rconnect
from revenew.db import init_db
from revenew.decide.shelf import (
    FLAT_COUPON_DEPTH,
    LOYALTY_CREDIT_DEPTH,
    PERCENT_DISCOUNT_DEPTH,
    ShelfGenerator,
)
from revenew.models import ActionFamily, Envelope, OpportunityType, Segment

ENVELOPE = Envelope(
    max_discount_pct=0.20, max_absolute_discount=500.0, budget_remaining=10_000.0,
    excluded_skus=[], cooldown_days=30, max_offers_per_customer_per_month=1, cogs_by_sku=None,
)

CATALOG = [
    {"sku": "SKU-A01", "name": "Classic Tee", "category": "apparel", "price": 599.0},
    {"sku": "SKU-A02", "name": "Slim Jeans", "category": "apparel", "price": 1899.0},
]


def _build(conn, catalog=CATALOG, shelf=None):
    shelf = shelf or ShelfGenerator()
    return shelf, shelf.build(
        conn, opportunity_type=OpportunityType.DORMANT_WINBACK, segment=Segment.DORMANT,
        rupees_at_risk=750.0, envelope=ENVELOPE, catalog=catalog,
    )


def test_shelf_covers_every_family_when_a_bundle_pair_exists(seeded_conn):
    """The seeded fixture is already known (test_detector_recommended_sku.py)
    to produce real cross-sell hits, so the pooled/global version of the same
    affinity logic must find at least one pair too."""
    _, result = _build(seeded_conn)
    families = {c.action_family for c in result.candidates}
    assert families == set(ActionFamily), f"expected all 5 families, got {families}"

    bundle = next(c for c in result.candidates if c.action_family == ActionFamily.BUNDLE_OFFER)
    assert len(bundle.skus) == 2
    assert all(sku.startswith("SKU-") for sku in bundle.skus)


def test_shelf_depths_match_the_documented_constants(seeded_conn):
    _, result = _build(seeded_conn)
    by_family = {c.action_family: c for c in result.candidates}
    assert by_family[ActionFamily.PERCENT_DISCOUNT].discount_pct == PERCENT_DISCOUNT_DEPTH
    assert by_family[ActionFamily.FLAT_COUPON].discount_amount == FLAT_COUPON_DEPTH
    assert by_family[ActionFamily.LOYALTY_CREDIT].discount_amount == LOYALTY_CREDIT_DEPTH
    assert by_family[ActionFamily.REMINDER_NUDGE].discount_pct is None
    assert by_family[ActionFamily.REMINDER_NUDGE].discount_amount is None


def test_bundle_offer_is_omitted_not_downgraded_when_no_pair_survives(tmp_path: Path):
    """A sparse fixture -- one order, one item -- cannot possibly meet
    CROSS_SELL_MIN_PAIR_COUNT. The shelf must simply not offer BUNDLE_OFFER,
    NOT fall back to a second REMINDER_NUDGE the way `_template_fallback`
    does for the genuinely-LLM-unavailable path (decide/generator.py) --
    that silent downgrade is the exact bug this arm exists to avoid."""
    db_path = tmp_path / "sparse.db"
    init_db(db_path, reset=True)
    conn = rconnect(db_path)
    conn.execute("INSERT INTO customers VALUES ('cus1', '2026-01-01')")
    conn.execute("INSERT INTO products VALUES ('SKU-X', 'Widget', 'misc', 100.0, NULL)")
    conn.execute("INSERT INTO products VALUES ('SKU-Y', 'Gadget', 'misc', 200.0, NULL)")
    conn.execute("INSERT INTO orders VALUES ('o1', 'cus1', '2026-01-01', 300.0, 'captured')")
    conn.execute("INSERT INTO order_items VALUES ('o1', 'SKU-X', 1, 100.0)")
    conn.execute("INSERT INTO order_items VALUES ('o1', 'SKU-Y', 1, 200.0)")
    conn.commit()

    _, result = _build(conn, catalog=[
        {"sku": "SKU-X", "name": "Widget", "category": "misc", "price": 100.0},
        {"sku": "SKU-Y", "name": "Gadget", "category": "misc", "price": 200.0},
    ])
    families = {c.action_family for c in result.candidates}
    assert families == set(ActionFamily) - {ActionFamily.BUNDLE_OFFER}
    assert sum(1 for c in result.candidates if c.action_family == ActionFamily.REMINDER_NUDGE) == 1
    conn.close()


def test_build_is_cohort_level_not_customer_level(seeded_conn):
    """No customer_id parameter exists at all -- this asserts the stronger
    property that two calls for the SAME cohort return the identical cached
    object (memoised on decide.cassette.cache_key, the same key the LLM
    cassette uses), the way a real cohort-level generator must behave for
    Arm B and Arm C to be a fair comparison."""
    shelf = ShelfGenerator()
    _, first = _build(seeded_conn, shelf=shelf)
    _, second = _build(seeded_conn, shelf=shelf)
    assert first is second


def test_build_recomputes_the_global_bundle_pair_only_once(seeded_conn):
    """The global affinity query has no per-cohort inputs -- it must run at
    most once per ShelfGenerator instance, not once per cohort, or a
    multi-thousand-decision replay would re-scan the whole order history on
    every single decision. `sqlite3.Connection.execute` is a C-level slot
    that cannot be monkeypatched per-instance, so this uses SQLite's own
    `set_trace_callback` to observe which statements actually ran."""
    shelf = ShelfGenerator()
    traced: list[str] = []
    seeded_conn.set_trace_callback(traced.append)
    try:
        shelf.build(
            seeded_conn, opportunity_type=OpportunityType.DORMANT_WINBACK, segment=Segment.DORMANT,
            rupees_at_risk=750.0, envelope=ENVELOPE, catalog=CATALOG,
        )
        shelf.build(
            seeded_conn, opportunity_type=OpportunityType.FIRST_ORDER_RETENTION, segment=Segment.NEW,
            rupees_at_risk=300.0, envelope=ENVELOPE, catalog=CATALOG,
        )
    finally:
        seeded_conn.set_trace_callback(None)

    pair_queries = [sql for sql in traced if "pair_counts" in sql]
    assert len(pair_queries) == 1
