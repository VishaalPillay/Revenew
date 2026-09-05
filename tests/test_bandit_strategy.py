"""BanditScorer.choose(strategy=...): the one substitution PLAN.md section 5's
three-arm ablation rests on. "thompson" (default, unchanged from every caller
before this parameter existed) draws a fresh Beta sample per family and
explores; "greedy" scores every family at its posterior MEAN -- a
deterministic function of the current posteriors -- and always returns
whichever family currently looks best, with propensity exactly 1.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from revenew.decide.bandit import BanditScorer, PosteriorStore
from revenew.models import ActionFamily, Candidate, Segment


def _candidate(family: ActionFamily) -> Candidate:
    return Candidate(action_family=family, headline="offer", rationale="test candidate")


@pytest.fixture
def store(seeded_conn):
    s = PosteriorStore(seeded_conn)
    s.ensure_initialized()
    return s


def test_greedy_picks_the_highest_posterior_mean_deterministically(store):
    """Give PERCENT_DISCOUNT a much stronger posterior than every other
    family in play, then assert greedy picks it on every one of many calls --
    a single lucky Beta draw could pick it under Thompson sampling too, so
    the point of this test is that it NEVER varies."""
    for _ in range(50):
        store.apply_outcome(
            Segment.ACTIVE, ActionFamily.PERCENT_DISCOUNT,
            converted=True, net_revenue=500.0, outcome_seq=1,
        )
    candidates = [_candidate(f) for f in (
        ActionFamily.PERCENT_DISCOUNT, ActionFamily.FLAT_COUPON,
        ActionFamily.LOYALTY_CREDIT, ActionFamily.REMINDER_NUDGE,
    )]

    for seed in range(10):
        scorer = BanditScorer(store, seed=seed)
        choice = scorer.choose(Segment.ACTIVE, candidates, fallback_revenue=500.0, strategy="greedy")
        assert choice.candidate.action_family == ActionFamily.PERCENT_DISCOUNT
        assert choice.propensity == 1.0


def test_greedy_never_advances_the_rng(store):
    """A deterministic policy must not consume randomness -- calling it
    should never perturb what a LATER Thompson call on the same scorer would
    draw, since greedy makes no draws to perturb with."""
    candidates = [_candidate(f) for f in (ActionFamily.PERCENT_DISCOUNT, ActionFamily.FLAT_COUPON)]
    scorer = BanditScorer(store, seed=42)
    state_before = scorer._rng.bit_generator.state
    scorer.choose(Segment.ACTIVE, candidates, fallback_revenue=500.0, strategy="greedy")
    state_after = scorer._rng.bit_generator.state
    assert state_before == state_after


def test_thompson_is_the_default_and_can_explore_away_from_the_point_estimate(store):
    """Thompson sampling must still occasionally pick something other than
    the current point-estimate leader -- that IS exploration. Over many
    seeds, with two families close in posterior mass, both should win at
    least once; a strategy that always agreed with greedy would not be
    Thompson sampling at all."""
    store.apply_outcome(Segment.ACTIVE, ActionFamily.PERCENT_DISCOUNT, converted=True, net_revenue=100.0, outcome_seq=1)
    store.apply_outcome(Segment.ACTIVE, ActionFamily.FLAT_COUPON, converted=True, net_revenue=100.0, outcome_seq=2)
    candidates = [_candidate(f) for f in (ActionFamily.PERCENT_DISCOUNT, ActionFamily.FLAT_COUPON)]

    winners = set()
    for seed in range(200):
        scorer = BanditScorer(store, seed=seed)
        choice = scorer.choose(Segment.ACTIVE, candidates, fallback_revenue=100.0)  # default strategy
        winners.add(choice.candidate.action_family)
    assert winners == {ActionFamily.PERCENT_DISCOUNT, ActionFamily.FLAT_COUPON}


def test_family_values_rng_none_equals_posterior_mean(store):
    from revenew.decide.bandit import BanditScorer as BS

    params = {
        ActionFamily.PERCENT_DISCOUNT: (3.0, 7.0, 200.0),
        ActionFamily.REMINDER_NUDGE: (1.0, 1.0, None),
    }
    values = BS._family_values(params, None, fallback_revenue=50.0)
    assert values[ActionFamily.PERCENT_DISCOUNT] == pytest.approx((3.0 / 10.0) * 200.0)
    assert values[ActionFamily.REMINDER_NUDGE] == pytest.approx((1.0 / 2.0) * 50.0)


def test_family_values_rng_present_draws_from_beta(store):
    from revenew.decide.bandit import BanditScorer as BS

    params = {ActionFamily.PERCENT_DISCOUNT: (3.0, 7.0, 200.0)}
    rng = np.random.default_rng(1)
    values = BS._family_values(params, rng, fallback_revenue=50.0)
    # Not equal to the posterior mean in general -- it's a random draw.
    assert 0.0 <= values[ActionFamily.PERCENT_DISCOUNT] <= 200.0


def test_invalid_strategy_raises(store):
    candidates = [_candidate(ActionFamily.PERCENT_DISCOUNT)]
    scorer = BanditScorer(store, seed=1)
    with pytest.raises(ValueError):
        scorer.choose(Segment.ACTIVE, candidates, fallback_revenue=100.0, strategy="bogus")
