"""The decision path, and the one function that walks it end to end.

`decide_one_opportunity` is the orchestrator diagram 2 depicts: envelope in,
LLM candidates out, validator drops the illegal ones, bandit picks a winner
from what survives, budget is reserved, and the whole thing is traced. Both
the live webhook path (revenew/api/webhooks.py) and the replay driver
(harness/run_replay.py) call this SAME function -- there is one decision path
in this codebase, not a live one and a separate replay one that happen to
agree.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime

from revenew.decide.bandit import BanditScorer, PosteriorStore
from revenew.decide.envelope import EnvelopeEngine, EnvelopeValidator
from revenew.decide.generator import CandidateGenerator
from revenew.decide.trace import persist_decision
from revenew.execute import budget
from revenew.models import (
    Decision,
    DecisionStatus,
    NoActionReason,
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
    """
    store = PosteriorStore(conn)
    envelope = EnvelopeEngine.build(conn, policy)
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
        envelope=envelope, store=store, policy=policy,
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
    choice = scorer.choose(segment, valid, fallback_revenue=rupees_at_risk)

    cost = choice.candidate.estimated_cost(order_value)
    if cost > envelope.budget_remaining:
        # The bandit picked something that, after the real order_value is
        # accounted for, no longer fits -- re-check rather than trust the
        # envelope snapshot taken before order_value was known.
        return no_action(NoActionReason.BUDGET_EXHAUSTED, candidates=verdicts)

    decision = Decision(
        decision_id=decision_id, opportunity_id=opportunity_id, run_id=run_id,
        segment=segment, action_family=choice.candidate.action_family, envelope=envelope,
        candidates=verdicts, chosen_candidate=choice.candidate, propensity=choice.propensity,
        status=DecisionStatus.EXECUTED, no_action_reason=None, created_at=now,
    )
    # The decision row must exist before budget_ledger can reference it (a
    # foreign key, not an implementation detail), so persist THEN reserve --
    # reserving before the row exists fails outright rather than silently
    # holding budget against a decision nothing else can find.
    persist_decision(conn, decision)
    budget.reserve(conn, decision_id, cost, now=now)
    return decision
