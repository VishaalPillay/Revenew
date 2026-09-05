"""The decision path, and the one function that walks it end to end.

`decide_one_opportunity` is the orchestrator diagram 2 depicts: envelope in,
LLM candidates out, validator drops the illegal ones, bandit picks a winner
from what survives, budget is reserved, the chosen offer is EXECUTED against
Razorpay, and the whole thing is traced.

Its only real callers today are the replay driver (`harness/run_replay.py`)
and tests -- there is one decision path in this codebase, not a live one and
a separate replay one that happen to agree, but "live" here currently means
`revenew replay --llm record|replay` against a real database, not the
webhook receiver. `revenew/api/webhooks.py` records a verified delivery and
stops; it does not call this function, and an earlier version of this
docstring claimed it did. Closing that loop -- a real payment's webhook
feeding an OUTCOME back to the bandit via `revenew/ledger/outcome.py`'s
`record_outcome`, which already exists and already updates posteriors -- is
a different, smaller thing than calling `decide_one_opportunity` again: a
webhook confirms or denies a decision already made, it does not need to make
a new one.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime

from revenew.decide.bandit import BanditScorer, PosteriorStore
from revenew.decide.envelope import EnvelopeEngine, EnvelopeValidator
from revenew.decide.generator import CandidateGenerator
from revenew.decide.trace import mark_executed, persist_decision
from revenew.execute import budget
from revenew.execute.razorpay import RazorpayAdapter, build_adapter, execute_decision
from revenew.models import (
    Decision,
    DecisionStatus,
    NoActionReason,
    OfferSpec,
    OpportunityType,
    Segment,
)
from revenew.settings import PolicyConfig


def _customer_order_value(conn: sqlite3.Connection, customer_id: str) -> float:
    row = conn.execute(
        "SELECT AVG(amount) AS avg_amount FROM orders WHERE customer_id = ? AND status = 'captured'",
        (customer_id,),
    ).fetchone()
    return float(row["avg_amount"]) if row and row["avg_amount"] is not None else 0.0


def decide_one_opportunity(
    conn: sqlite3.Connection,
    *,
    opportunity_id: str,
    customer_id: str,
    segment: Segment,
    opportunity_type: OpportunityType,
    rupees_at_risk: float,
    run_id: str,
    policy: PolicyConfig,
    generator: CandidateGenerator,
    bandit_seed: int,
    now: datetime,
    adapter: RazorpayAdapter | None = None,
    strategy: str = "thompson",
) -> Decision:
    """Runs the full decision path for one TREATMENT-arm opportunity.

    Never called for control-arm opportunities -- those are logged and
    deliberately never reach this function at all, which is what makes them
    the counterfactual. See route/arbiter.py and route/arm.py.

    Callers must have called `PosteriorStore(conn).ensure_initialized()` once
    before the first call here -- it is deliberately NOT called on every
    invocation. It used to be: at a few thousand decisions a day, that meant
    a few thousand redundant 20-row INSERT-OR-IGNORE batches, each forcing its
    own commit, and was a meaningful share of why a 30-day replay was running
    5x over its ~40s budget. `ensure_initialized()` is idempotent, so callers
    that skip it exactly once (a fresh `posteriors` table) will simply see
    `PosteriorStore.get()` fall back to the in-memory prior rather than error
    -- but every real entry point in this codebase initializes it upfront.

    `adapter` defaults to `build_adapter()`'s choice -- `FixtureAdapter`
    (no network call, a synthetic id) unless `REVENEW_EXECUTION_MODE=live` is
    set AND both Razorpay credentials are present, in which case it is a real
    `LiveAdapter`. Every existing caller (every test, every replay run) keeps
    behaving exactly as before unless that env var is explicitly set --
    passing an explicit `adapter=` here always overrides it either way.

    `strategy` ("thompson", the default, or "greedy") is forwarded verbatim
    to `BanditScorer.choose()` -- see bandit.py. It is the entire lever
    PLAN.md section 5's three-arm ablation needs: same candidates, same
    posteriors, same validator, only the scoring rule changes.
    """
    store = PosteriorStore(conn)
    envelope = EnvelopeEngine.build(conn, policy)
    catalog = EnvelopeEngine.load_catalog(conn)
    order_value = _customer_order_value(conn, customer_id)
    decision_id = str(uuid.uuid4())

    def no_action(reason: NoActionReason, candidates=()) -> Decision:
        d = Decision(
            decision_id=decision_id, opportunity_id=opportunity_id, run_id=run_id,
            segment=segment, action_family=None, envelope=envelope, candidates=list(candidates),
            chosen_candidate=None, propensity=None, status=DecisionStatus.NO_ACTION,
            no_action_reason=reason, created_at=now,
        )
        persist_decision(conn, d)
        return d

    if envelope.budget_remaining <= 0:
        return no_action(NoActionReason.BUDGET_EXHAUSTED)

    candidate_set = generator.generate(
        opportunity_type=opportunity_type, segment=segment, rupees_at_risk=rupees_at_risk,
        envelope=envelope, store=store, policy=policy, catalog=catalog, conn=conn,
    )
    if candidate_set is None:
        return no_action(NoActionReason.LLM_UNAVAILABLE)

    verdicts = EnvelopeValidator.validate_all(
        conn, envelope, candidate_set.candidates,
        customer_id=customer_id, order_value=order_value, now=now,
    )
    valid = [v.candidate for v in verdicts if v.valid]
    if not valid:
        return no_action(NoActionReason.ALL_CANDIDATES_INVALID, candidates=verdicts)

    scorer = BanditScorer(store, seed=bandit_seed)
    choice = scorer.choose(segment, valid, fallback_revenue=rupees_at_risk, strategy=strategy)

    cost = choice.candidate.estimated_cost(order_value)
    if cost > envelope.budget_remaining:
        # The bandit picked something that, after the real order_value is
        # accounted for, no longer fits -- re-check rather than trust the
        # envelope snapshot taken before order_value was known.
        return no_action(NoActionReason.BUDGET_EXHAUSTED, candidates=verdicts)

    # Persisted as PENDING, not EXECUTED, before execution is even attempted
    # -- this is F8 (execution) closing the loop with the failure-mode table
    # in SYSTEM_DESIGN.md section 7: "Crash between reserve and commit ->
    # action.status='pending' older than timeout -> reconciler releases the
    # hold." A decision that never gets this far genuinely never spent
    # anything; one that does is exactly as auditable mid-flight as at rest.
    decision = Decision(
        decision_id=decision_id, opportunity_id=opportunity_id, run_id=run_id,
        segment=segment, action_family=choice.candidate.action_family, envelope=envelope,
        candidates=verdicts, chosen_candidate=choice.candidate, propensity=choice.propensity,
        status=DecisionStatus.PENDING, no_action_reason=None, created_at=now,
    )
    # The decision row must exist before budget_ledger can reference it (a
    # foreign key, not an implementation detail), so persist THEN reserve --
    # reserving before the row exists fails outright rather than silently
    # holding budget against a decision nothing else can find.
    persist_decision(conn, decision)
    budget.reserve(conn, decision_id, cost, now=now)

    if adapter is None:
        adapter = build_adapter()
    spec = OfferSpec(
        decision_id=decision_id, customer_id=customer_id, action_family=choice.candidate.action_family,
        headline=choice.candidate.headline, amount=cost,
        discount_pct=choice.candidate.discount_pct, discount_amount=choice.candidate.discount_amount,
        skus=choice.candidate.skus,
    )
    result = execute_decision(conn, adapter, decision_id=decision_id, spec=spec, now=now)

    if result.status == "failed":
        # Known-failed, synchronously -- release the hold right away rather
        # than waiting for the reconciler's timeout sweep. The decision row
        # stays 'pending': the schema has no 'failed' decision status (only
        # executions.status does), and a definitively-failed attempt is
        # indistinguishable from a stalled one for every downstream reader
        # that matters -- neither one spent the reservation.
        budget.release(conn, decision_id, cost, now=now)
        return decision

    mark_executed(conn, decision_id)
    return decision.model_copy(update={"status": DecisionStatus.EXECUTED})
