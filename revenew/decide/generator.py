"""CandidateGenerator: the only LLM call in the system.

Two distinct failure paths, because the failure-mode table in
SYSTEM_DESIGN.md section 7 specifies two distinct responses:

  connectivity / timeout / API error
      -> `Fall back to highest-posterior family with a templated offer.
          Degraded, not dead.` The system keeps operating; it just stops
      asking the model and uses whichever family currently looks best.

  malformed structured output, even after one retry with the schema echoed
      -> `no_action_reason='llm_unavailable'`. The model responded, but never
      produced anything usable -- retrying blindly forever would burn tokens
      on a call that has already shown it cannot satisfy the contract twice.

`generate()` returns `CandidateSet | None`: a set means "act on this, whether
it came from the model or the template shelf"; `None` means the second
failure path, and the caller records no_action.

Structured output is enforced with a forced tool call, not "please return
JSON" in the prompt -- Candidate's Pydantic schema IS the tool's input schema,
so a response that doesn't match cannot come back as valid JSON in the first
place; it comes back as a tool_use block with a wrong shape, and Pydantic
validation on `.input` is what decides "malformed" here.
"""

from __future__ import annotations

import json
import os

from revenew.decide.bandit import PosteriorStore
from revenew.models import (
    ActionFamily,
    Candidate,
    CandidateSet,
    Envelope,
    OpportunityType,
    Segment,
)
from revenew.settings import PolicyConfig

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024

CANDIDATE_SET_SCHEMA = CandidateSet.model_json_schema()

SYSTEM_PROMPT = """You are composing commercial offers for an e-commerce merchant.

You will be given: an opportunity (why this customer is worth acting on), a
constraint envelope (hard caps you must not exceed), and the customer's
segment. Propose 5 to 8 CANDIDATE offers. Each must:

- pick exactly one action_family from the allowed list
- stay within max_discount_pct and max_absolute_discount from the envelope
- never reference an excluded SKU
- carry a short, specific headline and a one-sentence rationale grounded in
  the opportunity you were given -- not a generic marketing line

Vary the candidates: different families, different depths, at least one
REMINDER_NUDGE (zero cost) among them. You are proposing options for a
downstream ranking system to choose between, not picking the single best one
yourself.
"""


def _prompt_context(
    opportunity_type: OpportunityType,
    segment: Segment,
    rupees_at_risk: float,
    envelope: Envelope,
) -> str:
    return json.dumps(
        {
            "opportunity_type": opportunity_type.value,
            "segment": segment.value,
            "rupees_at_risk": rupees_at_risk,
            "envelope": {
                "max_discount_pct": envelope.max_discount_pct,
                "max_absolute_discount": envelope.max_absolute_discount,
                "budget_remaining": envelope.budget_remaining,
                "excluded_skus": envelope.excluded_skus,
                "cogs_known_for_skus": sorted(envelope.cogs_by_sku or {}),
            },
            "allowed_action_families": [f.value for f in ActionFamily],
        },
        indent=2,
    )


def _client():
    """None if no credential is resolvable. Mirrors the check used elsewhere
    in this codebase's history: constructing the SDK client succeeds with no
    key at all and only fails at request time, so the client object itself is
    not a usable availability check -- the resolved key is."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    import anthropic

    return anthropic.Anthropic()


class CandidateGenerator:
    def __init__(self, client=None) -> None:
        self.client = client if client is not None else _client()

    @property
    def llm_available(self) -> bool:
        return self.client is not None

    def generate(
        self,
        *,
        opportunity_type: OpportunityType,
        segment: Segment,
        rupees_at_risk: float,
        envelope: Envelope,
        store: PosteriorStore,
        policy: PolicyConfig,
    ) -> CandidateSet | None:
        if self.client is None:
            return _template_fallback(segment, envelope, policy, store)

        context = _prompt_context(opportunity_type, segment, rupees_at_risk, envelope)
        try:
            result = self._call(context, retry_hint=None)
        except Exception:
            # Connectivity, auth, rate limit, timeout -- whatever the SDK
            # raises, this is the "unreachable" branch: degrade, don't die.
            return _template_fallback(segment, envelope, policy, store)

        parsed = _try_parse(result)
        if parsed is not None:
            return parsed

        # One retry with the schema echoed back, per the failure table.
        try:
            result2 = self._call(context, retry_hint=_schema_hint())
        except Exception:
            return _template_fallback(segment, envelope, policy, store)

        return _try_parse(result2)  # None here means genuinely give up -> no_action

    def _call(self, context: str, *, retry_hint: str | None) -> dict:
        user_content = context if retry_hint is None else f"{context}\n\n{retry_hint}"
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "name": "propose_candidates",
                    "description": "Propose 5-8 candidate offers for this opportunity.",
                    "input_schema": CANDIDATE_SET_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "propose_candidates"},
            messages=[{"role": "user", "content": user_content}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "propose_candidates":
                return block.input
        raise ValueError("model response contained no propose_candidates tool call")


def _schema_hint() -> str:
    return (
        "Your previous response did not match the required schema. "
        f"It must validate against this JSON schema exactly:\n{json.dumps(CANDIDATE_SET_SCHEMA)}"
    )


def _try_parse(raw: dict) -> CandidateSet | None:
    try:
        return CandidateSet.model_validate(raw)
    except Exception:
        return None


# ============================================================== fallback --


def _point_estimate(store: PosteriorStore, segment: Segment, family: ActionFamily) -> float:
    row = store.get(segment, family)
    p_mean = row.alpha / (row.alpha + row.beta)
    r_bar = row.mean_revenue if row.mean_revenue is not None else 0.0
    return p_mean * r_bar


def _highest_posterior_family(store: PosteriorStore, segment: Segment) -> ActionFamily:
    return max(ActionFamily, key=lambda f: _point_estimate(store, segment, f))


def _template_fallback(
    segment: Segment,
    envelope: Envelope,
    policy: PolicyConfig,
    store: PosteriorStore,
) -> CandidateSet:
    """One templated offer in whichever family currently has the highest
    posterior point estimate for this segment. Deliberately conservative on
    depth -- 60% of the policy's own cap -- so a degraded-mode offer can never
    itself be the thing that pushes a candidate over the envelope."""
    family = _highest_posterior_family(store, segment)
    safe_pct = round(policy.max_discount_pct * 0.6, 3)
    safe_amount = round(policy.max_absolute_discount * 0.6, 2)

    if family == ActionFamily.PERCENT_DISCOUNT:
        c = Candidate(
            action_family=family, headline=f"{safe_pct:.0%} off your next order",
            discount_pct=safe_pct, rationale="Templated fallback: highest posterior family, no LLM call.",
        )
    elif family == ActionFamily.FLAT_COUPON:
        c = Candidate(
            action_family=family, headline=f"Rs {safe_amount:.0f} off",
            discount_amount=safe_amount, rationale="Templated fallback: highest posterior family, no LLM call.",
        )
    elif family == ActionFamily.LOYALTY_CREDIT:
        c = Candidate(
            action_family=family, headline=f"Rs {safe_amount:.0f} loyalty credit",
            discount_amount=safe_amount, rationale="Templated fallback: highest posterior family, no LLM call.",
        )
    elif family == ActionFamily.BUNDLE_OFFER:
        # A bundle needs real SKUs the generator does not have in this
        # degraded path (no catalog reasoning without the model) -- fall
        # through to the always-safe zero-cost nudge instead of guessing SKUs.
        c = Candidate(
            action_family=ActionFamily.REMINDER_NUDGE, headline="We picked something you might like",
            rationale="Templated fallback: bundle needs LLM catalog reasoning, degraded to a nudge.",
        )
    else:
        c = Candidate(
            action_family=ActionFamily.REMINDER_NUDGE, headline="Still thinking of you",
            rationale="Templated fallback: highest posterior family, no LLM call.",
        )

    return CandidateSet(candidates=[c])
