"""F6 / the central safety claim: no executed action ever violates its own
envelope. A model error can only produce a suboptimal LEGAL action -- never an
illegal one -- because EnvelopeValidator re-applies the exact same rule table
EnvelopeEngine rendered into the prompt, programmatically, regardless of what
the model returns.
"""

from __future__ import annotations

from datetime import UTC, datetime

from revenew.decide.envelope import EnvelopeEngine, EnvelopeValidator
from revenew.models import ActionFamily, Candidate
from revenew.settings import PolicyConfig

NOW = datetime(2026, 1, 1, tzinfo=UTC)

TIGHT_POLICY = PolicyConfig(
    max_discount_pct=0.10,
    max_absolute_discount=150,
    budget_cap=1000,
    cooldown_days=14,
    max_offers_per_customer_per_month=1,
    excluded_skus=("SKU-BANNED",),
)


def _setup(conn):
    conn.execute("INSERT INTO products VALUES ('SKU-OK', 'A', 'apparel', 1000, 400)")
    conn.execute("INSERT INTO products VALUES ('SKU-BANNED', 'B', 'apparel', 500, 200)")
    conn.execute("INSERT INTO customers VALUES ('cus1', '2026-01-01')")
    conn.commit()


def test_a_deliberately_illegal_candidate_is_never_marked_valid(seeded_conn):
    _setup(seeded_conn)
    env = EnvelopeEngine.build(seeded_conn, TIGHT_POLICY)

    illegal_candidates = [
        Candidate(action_family=ActionFamily.PERCENT_DISCOUNT, headline="huge", discount_pct=0.99, rationale="r"),
        Candidate(action_family=ActionFamily.FLAT_COUPON, headline="over cap", discount_amount=9999, rationale="r"),
        Candidate(action_family=ActionFamily.BUNDLE_OFFER, headline="banned sku",
                  skus=["SKU-OK", "SKU-BANNED"], rationale="r"),
    ]

    verdicts = EnvelopeValidator.validate_all(
        seeded_conn, env, illegal_candidates, customer_id="cus1", order_value=1000, now=NOW
    )
    assert all(not v.valid for v in verdicts)
    assert all(v.violations for v in verdicts)


def test_a_legal_candidate_is_not_falsely_flagged(seeded_conn):
    _setup(seeded_conn)
    env = EnvelopeEngine.build(seeded_conn, TIGHT_POLICY)
    fine = Candidate(action_family=ActionFamily.PERCENT_DISCOUNT, headline="5% off", discount_pct=0.05, rationale="r")

    verdict = EnvelopeValidator.validate(seeded_conn, env, fine, customer_id="cus1", order_value=1000, now=NOW)
    assert verdict.valid
    assert verdict.violations == []


def test_no_candidate_marked_valid_can_actually_exceed_the_discount_cap(seeded_conn):
    """A fuzz sweep, not one hand-picked case: for every depth from 1% to 99%,
    validity must track the cap exactly. This is the property
    test_envelope_invariant is really standing in for."""
    _setup(seeded_conn)
    env = EnvelopeEngine.build(seeded_conn, TIGHT_POLICY)

    for pct in [i / 100 for i in range(1, 100)]:
        c = Candidate(action_family=ActionFamily.PERCENT_DISCOUNT, headline=f"{pct:.0%} off",
                      discount_pct=pct, rationale="r")
        verdict = EnvelopeValidator.validate(seeded_conn, env, c, customer_id="cus1", order_value=1000, now=NOW)
        should_be_valid = pct <= TIGHT_POLICY.max_discount_pct
        assert verdict.valid == should_be_valid, f"pct={pct} valid={verdict.valid}"


def test_cooldown_blocks_a_second_offer_within_the_window(seeded_conn):
    from revenew.clock import iso

    _setup(seeded_conn)
    env = EnvelopeEngine.build(seeded_conn, TIGHT_POLICY)
    fine = Candidate(action_family=ActionFamily.PERCENT_DISCOUNT, headline="5% off", discount_pct=0.05, rationale="r")

    seeded_conn.execute(
        "INSERT INTO opportunity_candidates VALUES ('opp1','run1','cus1','dormant_winback','w1','dormant_winback',500,'h',?)",
        (iso(NOW),),
    )
    seeded_conn.execute(
        "INSERT INTO opportunities VALUES ('opp1','run1','cus1','w1','dormant','treatment',?)", (iso(NOW),)
    )
    seeded_conn.execute(
        "INSERT INTO decisions VALUES ('dec1','opp1','run1','dormant','percent_discount','{}',1,1,'{}',0.5,'executed',NULL,?)",
        (iso(NOW),),
    )
    seeded_conn.commit()

    verdict = EnvelopeValidator.validate(seeded_conn, env, fine, customer_id="cus1", order_value=1000, now=NOW)
    assert not verdict.valid
    assert "cooldown_days" in verdict.violations
