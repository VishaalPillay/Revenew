"""Pydantic models: the one contract shared by the API, the LLM structured
output, and every internal component.

Segment x ActionFamily is deliberately a fixed 4x5 grid (SYSTEM_DESIGN.md
section 6). More segments or more families look sophisticated and starve every
learning cell -- the bandit then appears not to learn when it is actually just
underfed. The grid is a scope decision, not a limitation nobody noticed.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ============================================================== vocabulary --


class Segment(StrEnum):
    """Recency/frequency bucket. Computed deterministically from order history."""

    NEW = "new"                # 0-1 orders ever
    ACTIVE = "active"          # 2+ orders, last one recent
    LAPSING = "lapsing"        # 2+ orders, cadence has visibly slowed
    DORMANT = "dormant"        # long since any order


class ActionFamily(StrEnum):
    """The bandit's unit of learning. An LLM candidate belongs to exactly one."""

    PERCENT_DISCOUNT = "percent_discount"
    FLAT_COUPON = "flat_coupon"
    BUNDLE_OFFER = "bundle_offer"
    LOYALTY_CREDIT = "loyalty_credit"
    REMINDER_NUDGE = "reminder_nudge"   # no price change: a message only


# Families whose reward touches margin at the moment of conversion. Everything
# else starts with a neutral prior; these start pessimistic, which is the
# cold-start margin guard described in SYSTEM_DESIGN.md section 6 -- on day one
# the system will not spray discounts at customers who would have converted
# anyway, because it has not yet earned the right to believe discounting helps.
DISCOUNT_BEARING_FAMILIES: frozenset[ActionFamily] = frozenset(
    {
        ActionFamily.PERCENT_DISCOUNT,
        ActionFamily.FLAT_COUPON,
        ActionFamily.BUNDLE_OFFER,
        ActionFamily.LOYALTY_CREDIT,
    }
)


class OpportunityType(StrEnum):
    """What the deterministic detector found. See detect/queries.sql."""

    DORMANT_WINBACK = "dormant_winback"
    FIRST_ORDER_RETENTION = "first_order_retention"
    CROSS_SELL_AFFINITY = "cross_sell_affinity"


class Arm(StrEnum):
    CONTROL = "control"
    TREATMENT = "treatment"


class NoActionReason(StrEnum):
    """Every value here must appear verbatim in SYSTEM_DESIGN.md section 7."""

    ALL_CANDIDATES_INVALID = "all_candidates_invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LLM_UNAVAILABLE = "llm_unavailable"


class DecisionStatus(StrEnum):
    EXECUTED = "executed"
    NO_ACTION = "no_action"
    PENDING = "pending"


# ================================================================ detection --


class Opportunity(BaseModel):
    """One arbitrated, arm-assigned opportunity. Row shape of `opportunities`
    joined back to its winning `opportunity_candidates` row."""

    opportunity_id: str
    run_id: str
    customer_id: str
    opportunity_type: OpportunityType
    window_id: str
    cohort_id: str
    rupees_at_risk: float = Field(ge=0)
    detector_query_hash: str
    segment: Segment
    arm: Arm
    detected_at: datetime
    assigned_at: datetime


# ================================================================== policy --


class Envelope(BaseModel):
    """The single rule table. `EnvelopeEngine` renders this into the prompt;
    `EnvelopeValidator` calls `.violations()` on the model's output. One
    definition, two consumers -- see SYSTEM_DESIGN.md section 5."""

    max_discount_pct: float = Field(ge=0, le=1)
    max_absolute_discount: float = Field(ge=0)
    budget_remaining: float
    excluded_skus: list[str] = Field(default_factory=list)
    cooldown_days: int = Field(ge=0)
    max_offers_per_customer_per_month: int = Field(ge=0)
    cogs_by_sku: dict[str, float] | None = None  # None = unknown, never 0

    def violations(self, c: Candidate) -> list[str]:
        """Every rule this candidate breaks, by name. Empty means legal.

        Deliberately returns ALL violations rather than stopping at the first,
        because decision_candidates.violations_json is audit evidence -- a
        candidate that breaks the discount cap AND touches an excluded SKU
        should show both, not whichever the code happened to check first.
        """
        out: list[str] = []

        if c.discount_pct is not None and c.discount_pct > self.max_discount_pct:
            out.append("max_discount_pct")

        if c.discount_amount is not None and c.discount_amount > self.max_absolute_discount:
            out.append("max_absolute_discount")

        estimated_cost = c.estimated_cost()
        if estimated_cost > self.budget_remaining:
            out.append("budget_remaining")

        if self.excluded_skus and set(c.skus) & set(self.excluded_skus):
            out.append("excluded_skus")

        # cooldown_days and max_offers_per_customer_per_month are checked against
        # the customer's own history by EnvelopeValidator (they need a database
        # lookup this pure function does not have access to); violations() only
        # covers what is decidable from the candidate and the envelope alone.

        return out


# ============================================================= generation --


class Candidate(BaseModel):
    """One offer the LLM proposed, or a template produced as a fallback.

    Exactly one of discount_pct / discount_amount is set for a price-off
    family; both are None for REMINDER_NUDGE, which is the model-level
    expression of "this family has no monetary cost".

    `extra="forbid"` is not just belt-and-braces validation -- it is what
    makes `model_json_schema()` emit `additionalProperties: false`, which the
    Anthropic API's `strict: true` tool mode requires (decide/generator.py).
    Under strict mode a response that doesn't match this shape cannot come
    back as a superficially-valid tool call with junk fields at all; the
    malformed-output retry path exists for the cases strict mode can't catch
    (e.g. a family/discount-shape combination the tool schema allows but
    `_one_discount_shape` below rejects).
    """

    model_config = ConfigDict(extra="forbid")

    action_family: ActionFamily
    headline: str = Field(max_length=140)
    discount_pct: float | None = Field(default=None, ge=0, le=1)
    discount_amount: float | None = Field(default=None, ge=0)
    skus: list[str] = Field(default_factory=list)
    rationale: str = Field(max_length=280)

    @model_validator(mode="after")
    def _one_discount_shape(self) -> Candidate:
        if self.discount_pct is not None and self.discount_amount is not None:
            raise ValueError("candidate must not set both discount_pct and discount_amount")
        if self.action_family == ActionFamily.REMINDER_NUDGE and (
            self.discount_pct is not None or self.discount_amount is not None
        ):
            raise ValueError("reminder_nudge must carry no discount")
        return self

    def estimated_cost(self, order_value: float = 0.0) -> float:
        """Rupees this candidate would consume from the campaign budget.

        `order_value` is the customer's typical basket, supplied by the caller
        (the envelope has no customer context) -- needed to convert a
        percentage discount into an absolute rupee figure for the budget check.
        """
        if self.discount_amount is not None:
            return self.discount_amount
        if self.discount_pct is not None:
            return self.discount_pct * order_value
        return 0.0


class CandidateSet(BaseModel):
    """Structured output contract for the LLM call. Exactly this shape, or the
    call is treated as malformed (see NoActionReason.LLM_UNAVAILABLE)."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[Candidate] = Field(min_length=1, max_length=8)


# ================================================================ decisions --


class DecisionCandidate(BaseModel):
    """One row of the audit trail: a proposed candidate plus its verdict."""

    candidate: Candidate
    valid: bool
    violations: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    decision_id: str
    opportunity_id: str
    run_id: str
    segment: Segment
    action_family: ActionFamily | None = None
    envelope: Envelope
    candidates: list[DecisionCandidate] = Field(default_factory=list)
    chosen_candidate: Candidate | None = None
    propensity: float | None = Field(default=None, ge=0, le=1)
    status: DecisionStatus
    no_action_reason: NoActionReason | None = None
    created_at: datetime
    channel: str = "internal"

    @property
    def candidates_generated(self) -> int:
        return len(self.candidates)

    @property
    def candidates_valid(self) -> int:
        return sum(1 for c in self.candidates if c.valid)

    @model_validator(mode="after")
    def _status_consistency(self) -> Decision:
        if self.status == DecisionStatus.EXECUTED:
            if self.chosen_candidate is None or self.propensity is None:
                raise ValueError("an executed decision must carry a chosen candidate and propensity")
        if self.status == DecisionStatus.NO_ACTION and self.no_action_reason is None:
            raise ValueError("a no_action decision must carry a reason")
        return self


# ================================================================== outcomes --


class Outcome(BaseModel):
    """One append-only row. `converted=False, censored=True` means the window
    closed with no signal either way -- see SYSTEM_DESIGN.md section 6: this is
    NOT the same as a failure, and the bandit update path treats it distinctly."""

    decision_id: str
    opportunity_id: str
    converted: bool
    net_revenue: float = 0.0
    censored: bool = False
    closed_at: datetime

    @model_validator(mode="after")
    def _revenue_only_on_conversion(self) -> Outcome:
        if self.converted and self.censored:
            raise ValueError("an outcome cannot be both converted and censored")
        if not self.converted and self.net_revenue != 0.0:
            raise ValueError("net_revenue must be zero when the outcome did not convert")
        return self


# =========================================================== execution --


class OfferSpec(BaseModel):
    """What RazorpayAdapter.create_offer needs. Derived from a chosen Candidate.

    `amount` is the real rupee figure the offer is worth -- `Candidate.
    estimated_cost(order_value)`, resolved at the call site in
    decide/__init__.py where the customer's actual order_value is already
    known. Earlier, LiveAdapter.create_offer hardcoded this to 0 with no
    caller ever supplying a real figure; carrying it on the spec itself is
    what makes that fixable without LiveAdapter having to look anything up.

    `decision_id` exists so `LiveAdapter.create_offer` can carry it in the
    payment link's `notes` -- without it, a webhook delivery reporting that
    payment has no way to map back to the decision that produced it, and the
    outcome (and the bandit update it feeds) can never be recorded.
    """

    decision_id: str
    customer_id: str
    action_family: ActionFamily
    headline: str
    amount: float = Field(ge=0, default=0.0)
    discount_pct: float | None = None
    discount_amount: float | None = None
    skus: list[str] = Field(default_factory=list)


class LinkSpec(BaseModel):
    """What RazorpayAdapter.create_payment_link needs."""

    customer_id: str
    amount: float
    description: str


class ExecutionResult(BaseModel):
    provider_ref: str
    status: Literal["sent", "confirmed", "failed"]
