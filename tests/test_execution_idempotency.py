"""F8/stage 9 closing the loop: decide_one_opportunity now actually executes
the chosen offer, not just reserves budget for it. Covers: a decision that
executes successfully flips pending -> executed and writes exactly one
executions row; a redelivered idempotency key is a database no-op, never a
second charge; a failing adapter releases its budget hold synchronously and
leaves the decision pending (there is no 'failed' decisions status -- see
decide/__init__.py); the reconciler restores available() for a decision that
crashed before ever executing, and fixes forward one whose execution had
actually succeeded; and LiveAdapter retries a transient failure with backoff.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from revenew.clock import iso
from revenew.decide import decide_one_opportunity
from revenew.decide.bandit import PosteriorStore
from revenew.decide.generator import CandidateGenerator
from revenew.execute import budget
from revenew.execute.razorpay import FixtureAdapter, LiveAdapter, execute_decision
from revenew.ledger.reconcile import reconcile
from revenew.models import DecisionStatus, LinkSpec, OfferSpec, OpportunityType, Segment
from revenew.settings import PolicyConfig

NOW = datetime(2026, 1, 1, tzinfo=UTC)
GENEROUS_POLICY = PolicyConfig(budget_cap=100_000.0, cooldown_days=0, max_offers_per_customer_per_month=100)


class FailingAdapter:
    """create_offer always raises -- execute_decision converts that to a
    status='failed' ExecutionResult rather than letting it propagate."""

    def create_offer(self, spec, idempotency_key):
        raise RuntimeError("simulated Razorpay outage")

    def create_payment_link(self, spec, idempotency_key):
        raise RuntimeError("simulated Razorpay outage")


def _seed_opportunity(conn, opp_id: str, customer_id: str, *, created_at=NOW) -> None:
    conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (customer_id, iso(created_at)))
    conn.execute(
        "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?)",
        (opp_id, "run1", customer_id, "dormant_winback", "w1", "dormant_winback", 500, "h", iso(created_at)),
    )
    conn.execute(
        "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
        (opp_id, "run1", customer_id, "w1", "dormant", "treatment", iso(created_at)),
    )
    conn.commit()


def _decide(conn, opp_id: str, customer_id: str, *, adapter=None, now=NOW):
    store = PosteriorStore(conn)
    store.ensure_initialized()
    generator = CandidateGenerator(mode="off")  # deterministic single template candidate
    return decide_one_opportunity(
        conn, opportunity_id=opp_id, customer_id=customer_id, segment=Segment.DORMANT,
        opportunity_type=OpportunityType.DORMANT_WINBACK, rupees_at_risk=500.0, run_id="run1",
        policy=GENEROUS_POLICY, generator=generator, bandit_seed=1, now=now, adapter=adapter,
    )


# ============================================================== execution --


def test_successful_execution_flips_pending_to_executed_and_writes_one_row(seeded_conn):
    _seed_opportunity(seeded_conn, "opp_exec_ok", "cus_exec_ok")
    decision = _decide(seeded_conn, "opp_exec_ok", "cus_exec_ok", adapter=FixtureAdapter())

    assert decision.status == DecisionStatus.EXECUTED
    row = seeded_conn.execute(
        "SELECT status FROM decisions WHERE decision_id = ?", (decision.decision_id,)
    ).fetchone()
    assert row["status"] == "executed"

    executions = seeded_conn.execute(
        "SELECT * FROM executions WHERE decision_id = ?", (decision.decision_id,)
    ).fetchall()
    assert len(executions) == 1
    assert executions[0]["status"] == "confirmed"


def test_redelivered_idempotency_key_is_a_database_no_op(seeded_conn):
    """Calling execute_decision twice with the SAME decision_id (simulating a
    retried request) must write exactly one executions row -- Razorpay's own
    idempotency handling is keyed on this exact header, and the local check
    (UNIQUE idempotency_key) is what makes a redelivery a no-op rather than a
    second charge even before that header is honored upstream. Built directly
    against a hand-inserted decision row (not through decide_one_opportunity,
    which now executes internally) so exactly two execute_decision() calls
    happen -- the ones this test is actually measuring."""
    _seed_opportunity(seeded_conn, "opp_idem", "cus_idem")
    seeded_conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("dec_idem", "opp_idem", "run1", "dormant", "percent_discount", "{}",
         1, 1, "{}", 0.5, "pending", None, iso(NOW)),
    )
    seeded_conn.commit()

    spec = OfferSpec(customer_id="cus_idem", action_family="percent_discount", headline="x", amount=10.0)
    adapter = FixtureAdapter()
    first = execute_decision(seeded_conn, adapter, decision_id="dec_idem", spec=spec, now=NOW)
    second = execute_decision(seeded_conn, adapter, decision_id="dec_idem", spec=spec, now=NOW)

    assert first.provider_ref == second.provider_ref
    assert len(adapter.calls) == 1  # the second call never reached the adapter at all
    rows = seeded_conn.execute(
        "SELECT COUNT(*) AS n FROM executions WHERE decision_id = 'dec_idem'"
    ).fetchone()
    assert rows["n"] == 1


def test_failing_adapter_releases_the_hold_and_leaves_decision_pending(seeded_conn):
    _seed_opportunity(seeded_conn, "opp_fail", "cus_fail")
    available_before = budget.available(seeded_conn, GENEROUS_POLICY.budget_cap)

    decision = _decide(seeded_conn, "opp_fail", "cus_fail", adapter=FailingAdapter())

    assert decision.status == DecisionStatus.PENDING
    row = seeded_conn.execute(
        "SELECT status FROM decisions WHERE decision_id = ?", (decision.decision_id,)
    ).fetchone()
    assert row["status"] == "pending"

    execution = seeded_conn.execute(
        "SELECT status FROM executions WHERE decision_id = ?", (decision.decision_id,)
    ).fetchone()
    assert execution["status"] == "failed"

    # the hold must be released synchronously -- available() is restored to
    # what it was before this decision, not left short by the reserved amount
    assert budget.available(seeded_conn, GENEROUS_POLICY.budget_cap) == pytest.approx(available_before)


# ============================================================ reconciler --


def test_reconcile_releases_a_decision_that_crashed_before_any_execution(seeded_conn):
    """Simulates a genuine crash: a decision persisted and its budget
    reserved, but the process died before execute_decision ever ran -- no
    executions row exists at all."""
    stale = NOW - timedelta(minutes=60)
    _seed_opportunity(seeded_conn, "opp_crash", "cus_crash", created_at=stale)
    seeded_conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("dec_crash", "opp_crash", "run1", "dormant", "percent_discount", "{}",
         1, 1, "{}", 0.5, "pending", None, iso(stale)),
    )
    seeded_conn.commit()
    budget.reserve(seeded_conn, "dec_crash", 300.0, now=stale)
    available_before_reserve = budget.available(seeded_conn, GENEROUS_POLICY.budget_cap) + 300.0

    result = reconcile(seeded_conn, now=NOW, timeout_minutes=30)

    assert result.released == 1
    assert result.fixed_forward == 0
    assert result.released_total == pytest.approx(300.0)
    assert budget.available(seeded_conn, GENEROUS_POLICY.budget_cap) == pytest.approx(available_before_reserve)


def test_reconcile_fixes_forward_a_decision_whose_execution_actually_succeeded(seeded_conn):
    """The other crash window: execute_decision succeeded (a 'confirmed'
    executions row exists) but the process died before mark_executed
    committed. The reconciler must flip the status WITHOUT releasing the
    hold -- the money was genuinely spent."""
    stale = NOW - timedelta(minutes=60)
    _seed_opportunity(seeded_conn, "opp_unflipped", "cus_unflipped", created_at=stale)
    seeded_conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("dec_unflipped", "opp_unflipped", "run1", "dormant", "percent_discount", "{}",
         1, 1, "{}", 0.5, "pending", None, iso(stale)),
    )
    seeded_conn.commit()
    budget.reserve(seeded_conn, "dec_unflipped", 300.0, now=stale)
    seeded_conn.execute(
        "INSERT INTO executions (execution_id, decision_id, idempotency_key, provider_ref, status, created_at) "
        "VALUES ('ex1', 'dec_unflipped', 'revenew-dec_unflipped', 'ref123', 'confirmed', ?)",
        (iso(stale),),
    )
    seeded_conn.commit()
    available_before = budget.available(seeded_conn, GENEROUS_POLICY.budget_cap)

    result = reconcile(seeded_conn, now=NOW, timeout_minutes=30)

    assert result.fixed_forward == 1
    assert result.released == 0
    row = seeded_conn.execute(
        "SELECT status FROM decisions WHERE decision_id = 'dec_unflipped'"
    ).fetchone()
    assert row["status"] == "executed"
    # budget must NOT be released -- the offer was genuinely sent
    assert budget.available(seeded_conn, GENEROUS_POLICY.budget_cap) == pytest.approx(available_before)


def test_reconcile_ignores_pending_decisions_still_within_the_timeout(seeded_conn):
    _seed_opportunity(seeded_conn, "opp_fresh", "cus_fresh", created_at=NOW)
    seeded_conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("dec_fresh", "opp_fresh", "run1", "dormant", "percent_discount", "{}",
         1, 1, "{}", 0.5, "pending", None, iso(NOW)),
    )
    seeded_conn.commit()
    budget.reserve(seeded_conn, "dec_fresh", 300.0, now=NOW)

    result = reconcile(seeded_conn, now=NOW + timedelta(minutes=5), timeout_minutes=30)

    assert result.released == 0
    assert result.fixed_forward == 0


# ============================================================ LiveAdapter --


def test_live_adapter_retries_transient_failure_with_backoff(monkeypatch):
    adapter = LiveAdapter("rzp_test_fake", "fake_secret")

    calls = {"n": 0}

    def flaky_create(payload, idempotency_key=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient 5xx")
        return {"id": "plink_success"}

    monkeypatch.setattr(adapter._client.payment_link, "create", flaky_create)
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    result = adapter.create_payment_link(LinkSpec(customer_id="cus1", amount=100.0, description="test"), "idem1")

    assert result.provider_ref == "plink_success"
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.5]  # LiveAdapter.BACKOFF_SECONDS[:2] -- two retries before success


def test_live_adapter_raises_after_exhausting_all_retries(monkeypatch):
    adapter = LiveAdapter("rzp_test_fake", "fake_secret")

    def always_fails(payload, idempotency_key=None):
        raise RuntimeError("persistent outage")

    monkeypatch.setattr(adapter._client.payment_link, "create", always_fails)
    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="persistent outage"):
        adapter.create_payment_link(LinkSpec(customer_id="cus1", amount=100.0, description="test"), "idem1")


def test_offer_spec_amount_reaches_live_adapter_in_paise(monkeypatch):
    """Finding F-4: LiveAdapter.create_offer used to hardcode amount_paise=0
    unconditionally. It must now use spec.amount, converted to paise."""
    adapter = LiveAdapter("rzp_test_fake", "fake_secret")
    captured = {}

    def capture_create(payload, idempotency_key=None):
        captured.update(payload)
        return {"id": "plink_x"}

    monkeypatch.setattr(adapter._client.payment_link, "create", capture_create)

    from revenew.models import ActionFamily

    spec = OfferSpec(
        customer_id="cus1", action_family=ActionFamily.PERCENT_DISCOUNT,
        headline="10% off", amount=123.45,
    )
    adapter.create_offer(spec, "idem1")

    assert captured["amount"] == 12345  # rupees -> paise, not the old hardcoded 0
