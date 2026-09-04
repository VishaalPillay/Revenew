"""BanditScorer + PosteriorStore: Thompson sampling over (segment, action_family).

Reward model, per SYSTEM_DESIGN.md section 6 -- two parts, because reward is
not Bernoulli:

    p        ~ Beta(alpha, beta)              conversion
    r_bar    = revenue_sum / revenue_n        mean net revenue given conversion
    sampled_value = p_sample * r_bar

Update on each closed window:
    converted     -> alpha += 1, revenue_sum += net_revenue, revenue_n += 1
    not converted -> beta += 1 (censored included -- see the note below)

Priors: discount-bearing families start pessimistic (alpha=1, beta=4);
everything else starts neutral (alpha=1, beta=1). This is the cold-start
margin guard -- on day one the system has not yet earned the right to believe
discounting helps, so it does not default to spraying discounts at customers
who would have converted anyway.

On censoring: the spec is explicit that "censored is not failure" and that
treating silence as failure risks teaching the bandit to avoid slow-converting
arms -- and equally explicit that the update rule is still beta += 1 for a
censored window, with the outcome FLAGGED (outcomes.censored=1) rather than
hidden. That flag is what keeps the caveat auditable instead of silently
biasing every report that reads the outcome log. A more correct treatment
(e.g. modelling time-to-convert directly) is real future work, not a gap
nobody noticed -- see SYSTEM_DESIGN.md's trade-off table.

Cold start across segments: a (segment, family) cell with fewer than
COLD_START_MIN_OBSERVED real outcomes samples from a BLENDED posterior --
this cell's own prior, plus the pooled real-outcome evidence for that same
family across all four segments -- rather than from its own thin sample
alone. Once the cell has enough of its own evidence it detaches and samples
independently. This is recomputable from alpha/beta alone: since both only
ever move by whole +1 increments away from a prior that is a pure function of
`action_family`, "how much is prior vs. real evidence" needs no extra column.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from revenew.models import DISCOUNT_BEARING_FAMILIES, ActionFamily, Candidate, Segment

PRIOR_DISCOUNT_BEARING = (1.0, 4.0)
PRIOR_NEUTRAL = (1.0, 1.0)

COLD_START_MIN_OBSERVED = 20
PROPENSITY_SAMPLES = 4000


def prior_for(family: ActionFamily) -> tuple[float, float]:
    return PRIOR_DISCOUNT_BEARING if family in DISCOUNT_BEARING_FAMILIES else PRIOR_NEUTRAL


@dataclass(frozen=True)
class PosteriorRow:
    segment: Segment
    action_family: ActionFamily
    alpha: float
    beta: float
    revenue_sum: float
    revenue_n: int
    updated_through_seq: int

    @property
    def n_observed(self) -> float:
        """Real outcomes folded in, net of the prior. Prior is a pure function
        of action_family, so this needs no stored column."""
        a0, b0 = prior_for(self.action_family)
        return (self.alpha - a0) + (self.beta - b0)

    @property
    def mean_revenue(self) -> float | None:
        return (self.revenue_sum / self.revenue_n) if self.revenue_n > 0 else None


class PosteriorStore:
    """Thin wrapper over the `posteriors` table. Fully rebuildable from
    `outcomes` -- see ledger/replay.py, which calls `apply_outcome` in
    outcome_seq order and must reach byte-identical rows to the live path."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def ensure_initialized(self) -> None:
        rows = []
        for seg in Segment:
            for fam in ActionFamily:
                a0, b0 = prior_for(fam)
                rows.append((seg.value, fam.value, a0, b0, 0.0, 0, 0))
        self.conn.executemany(
            "INSERT OR IGNORE INTO posteriors "
            "(segment, action_family, alpha, beta, revenue_sum, revenue_n, updated_through_seq) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def get(self, segment: Segment, family: ActionFamily) -> PosteriorRow:
        row = self.conn.execute(
            "SELECT * FROM posteriors WHERE segment = ? AND action_family = ?",
            (segment.value, family.value),
        ).fetchone()
        if row is None:
            a0, b0 = prior_for(family)
            return PosteriorRow(segment, family, a0, b0, 0.0, 0, 0)
        return PosteriorRow(
            segment, family, row["alpha"], row["beta"],
            row["revenue_sum"], row["revenue_n"], row["updated_through_seq"],
        )

    def get_all(self) -> list[PosteriorRow]:
        return [self.get(s, f) for s in Segment for f in ActionFamily]

    def apply_outcome(
        self,
        segment: Segment,
        family: ActionFamily,
        *,
        converted: bool,
        net_revenue: float,
        outcome_seq: int,
    ) -> None:
        current = self.get(segment, family)
        if converted:
            alpha, beta = current.alpha + 1, current.beta
            revenue_sum, revenue_n = current.revenue_sum + net_revenue, current.revenue_n + 1
        else:
            # Both a genuine non-conversion and a censored window land here.
            # See the module docstring for why that is a stated limitation,
            # not an oversight.
            alpha, beta = current.alpha, current.beta + 1
            revenue_sum, revenue_n = current.revenue_sum, current.revenue_n

        self.conn.execute(
            """
            INSERT INTO posteriors
                (segment, action_family, alpha, beta, revenue_sum, revenue_n, updated_through_seq)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(segment, action_family) DO UPDATE SET
                alpha = excluded.alpha,
                beta = excluded.beta,
                revenue_sum = excluded.revenue_sum,
                revenue_n = excluded.revenue_n,
                updated_through_seq = excluded.updated_through_seq
            """,
            (segment.value, family.value, alpha, beta, revenue_sum, revenue_n, outcome_seq),
        )


def _blended_beta_params(store: PosteriorStore, segment: Segment, family: ActionFamily) -> tuple[float, float]:
    own = store.get(segment, family)
    if own.n_observed >= COLD_START_MIN_OBSERVED:
        return own.alpha, own.beta

    a0, b0 = prior_for(family)
    pooled_alpha_extra = 0.0
    pooled_beta_extra = 0.0
    for seg in Segment:
        cell = store.get(seg, family)
        pooled_alpha_extra += cell.alpha - a0
        pooled_beta_extra += cell.beta - b0
    return a0 + pooled_alpha_extra, b0 + pooled_beta_extra


@dataclass(frozen=True)
class BanditChoice:
    candidate: Candidate
    propensity: float


class BanditScorer:
    def __init__(self, store: PosteriorStore, *, seed: int) -> None:
        self.store = store
        self._rng = np.random.default_rng(seed)

    def _resolve_params(
        self, segment: Segment, families: list[ActionFamily]
    ) -> dict[ActionFamily, tuple[float, float, float | None]]:
        """One database pass per `choose()` call: (alpha, beta, mean_revenue)
        for every family in play, including whatever cross-segment pooling
        cold start needs.

        This exists because sampling and propensity estimation both draw
        thousands of times per decision, and every one of those draws used to
        re-read the posterior table from scratch -- fine at unit-test volume,
        catastrophic at a few thousand decisions a day: a 30-day replay was
        timing out past 3 minutes with 17 of 30 days done, entirely inside
        this function. Reading the posteriors ONCE and sampling from plain
        numpy floats afterward is the fix; the math is unchanged.
        """
        return {
            fam: (*_blended_beta_params(self.store, segment, fam), self.store.get(segment, fam).mean_revenue)
            for fam in families
        }

    @staticmethod
    def _sample_values(
        params: dict[ActionFamily, tuple[float, float, float | None]],
        rng: np.random.Generator,
        *,
        fallback_revenue: float,
    ) -> dict[ActionFamily, float]:
        """One Thompson draw per family, from precomputed (alpha, beta, r_bar)
        -- no database access in this function, by design; see `_resolve_params`.
        `fallback_revenue` (the opportunity's own rupees_at_risk) stands in for
        r_bar on a family with zero observed conversions, so an untested
        family does not look worthless and go permanently unsampled."""
        out = {}
        for fam, (a, b, r_bar) in params.items():
            p_sample = rng.beta(a, b)
            out[fam] = p_sample * (r_bar if r_bar is not None else fallback_revenue)
        return out

    def choose(
        self,
        segment: Segment,
        candidates: list[Candidate],
        *,
        fallback_revenue: float,
    ) -> BanditChoice:
        """Thompson-sample over the DISTINCT families present in `candidates`."""
        if not candidates:
            raise ValueError("choose() requires at least one candidate")

        families = sorted({c.action_family for c in candidates}, key=lambda f: f.value)
        params = self._resolve_params(segment, families)

        values = self._sample_values(params, self._rng, fallback_revenue=fallback_revenue)
        best_family = max(values, key=lambda f: values[f])
        # Multiple candidates can share a family; cheapest wins the tie, which
        # maximises margin for the same expected conversion cell.
        same_family = [c for c in candidates if c.action_family == best_family]
        chosen = min(same_family, key=lambda c: c.estimated_cost(fallback_revenue))

        propensity = self._estimate_propensity(params, best_family, fallback_revenue)
        return BanditChoice(candidate=chosen, propensity=propensity)

    def _estimate_propensity(
        self,
        params: dict[ActionFamily, tuple[float, float, float | None]],
        winner: ActionFamily,
        fallback_revenue: float,
    ) -> float:
        """Monte Carlo win-rate of `winner`, vectorised: draw PROPENSITY_SAMPLES
        Beta variates for every family AT ONCE via numpy's array-shaped `beta`,
        instead of one Python-level call per sample per family. Under a FRESH,
        independently-seeded RNG stream, so this never consumes draws from the
        stream `choose()` itself advances -- two calls to `choose()` in a row
        stay reproducible under replay regardless of whether propensity was
        ever computed for the first one.
        """
        rng = np.random.default_rng(self._rng.integers(0, 2**32 - 1))
        families = list(params.keys())

        draws = np.empty((PROPENSITY_SAMPLES, len(families)))
        for i, fam in enumerate(families):
            a, b, r_bar = params[fam]
            p = rng.beta(a, b, size=PROPENSITY_SAMPLES)
            draws[:, i] = p * (r_bar if r_bar is not None else fallback_revenue)

        winner_idx = families.index(winner)
        wins = int((draws.argmax(axis=1) == winner_idx).sum())
        return wins / PROPENSITY_SAMPLES
