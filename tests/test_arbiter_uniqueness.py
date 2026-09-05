"""F2: at most one action per (customer, window).

The enforcement is a UNIQUE constraint in db/schema.sql, not application logic
-- these tests check both that the arbiter picks a sane winner AND that the
database genuinely refuses a second one if the arbiter is ever wrong.
"""

from __future__ import annotations

import sqlite3

import pytest

from revenew.detect.detector import OpportunityDetector, RawOpportunity
from revenew.route.arbiter import arbitrate, persist_winners
from tests.conftest import NOW


def _raw(customer_id, otype, window_id, rupees, cohort_id=None, oid=None):
    return RawOpportunity(
        opportunity_id=oid or f"opp_{customer_id}_{otype}",
        run_id="run1",
        customer_id=customer_id,
        opportunity_type=otype,
        window_id=window_id,
        cohort_id=cohort_id or otype.value,
        rupees_at_risk=rupees,
        detector_query_hash="testhash",
        detected_at="2026-01-01T00:00:00+00:00",
    )


def test_arbiter_keeps_highest_rupees_at_risk():
    from revenew.models import OpportunityType

    candidates = [
        _raw("c1", OpportunityType.DORMANT_WINBACK, "w1", 500),
        _raw("c1", OpportunityType.CROSS_SELL_AFFINITY, "w1", 1200),
        _raw("c1", OpportunityType.FIRST_ORDER_RETENTION, "w1", 300),
    ]
    winners = arbitrate(candidates)
    assert len(winners) == 1
    assert winners[0].rupees_at_risk == 1200
    assert winners[0].opportunity_type == OpportunityType.CROSS_SELL_AFFINITY


def test_arbiter_ties_break_on_cohort_id():
    from revenew.models import OpportunityType

    candidates = [
        _raw("c1", OpportunityType.DORMANT_WINBACK, "w1", 500, cohort_id="dormant_winback"),
        _raw("c1", OpportunityType.CROSS_SELL_AFFINITY, "w1", 500, cohort_id="cross_sell_affinity"),
    ]
    winners = arbitrate(candidates)
    assert len(winners) == 1
    # Tie-break picks the lexicographically GREATER cohort_id: 'd' > 'c', so
    # "dormant_winback" beats "cross_sell_affinity". The only property that
    # actually matters is that it's deterministic -- asserted by re-running.
    assert winners[0].cohort_id == "dormant_winback"
    assert arbitrate(candidates)[0].cohort_id == winners[0].cohort_id


def test_arbiter_does_not_cross_customers_or_windows():
    from revenew.models import OpportunityType

    candidates = [
        _raw("c1", OpportunityType.DORMANT_WINBACK, "w1", 500),
        _raw("c2", OpportunityType.DORMANT_WINBACK, "w1", 500),
        _raw("c1", OpportunityType.DORMANT_WINBACK, "w2", 500),
    ]
    winners = arbitrate(candidates)
    keys = {(w.customer_id, w.window_id) for w in winners}
    assert keys == {("c1", "w1"), ("c2", "w1"), ("c1", "w2")}


def test_no_customer_appears_twice_after_a_real_detection_run(seeded_conn):
    det = OpportunityDetector()
    raw = det.detect(seeded_conn, run_id="run1", window_id="w1", now=NOW)
    det.persist_candidates(seeded_conn, raw)
    winners = arbitrate(raw)

    customer_ids = [w.customer_id for w in winners]
    assert len(customer_ids) == len(set(customer_ids))

    arbitrated = persist_winners(seeded_conn, winners, now=NOW, salt="test-salt", control_pct=20)
    assert len(arbitrated) == len(winners)


def test_database_rejects_a_second_winner_for_the_same_customer_window(seeded_conn):
    """The actual F2 guarantee. If the arbiter ever had a bug and tried to
    insert two winners for one (customer, window), this is what stops it."""
    seeded_conn.execute(
        "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("opp1", "run1", "cus_000001", "dormant_winback", "w1", "dormant_winback",
         500.0, "hash1", "2026-01-01T00:00:00+00:00", None),
    )
    seeded_conn.execute(
        "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
        ("opp1", "run1", "cus_000001", "w1", "dormant", "treatment", "2026-01-01T00:00:00+00:00"),
    )
    seeded_conn.commit()

    seeded_conn.execute(
        "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("opp2", "run1", "cus_000001", "cross_sell_affinity", "w1", "cross_sell_affinity",
         900.0, "hash2", "2026-01-01T00:00:00+00:00", None),
    )
    with pytest.raises(sqlite3.IntegrityError):
        seeded_conn.execute(
            "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
            ("opp2", "run1", "cus_000001", "w1", "dormant", "treatment", "2026-01-01T00:00:00+00:00"),
        )
