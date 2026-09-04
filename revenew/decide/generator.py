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
it came from the model, the cassette, or the template shelf"; `None` means
the second failure path, and the caller records no_action.

Structured output is enforced with a forced, `strict: true` tool call, not
"please return JSON" in the prompt -- `Candidate`'s Pydantic schema (with
`extra="forbid"`, see models.py) IS the tool's input schema, so a response
that doesn't match the shape cannot come back as a superficially-valid tool
call with junk fields; Pydantic validation on `.input` catches the rest
(family/discount-shape combinations the tool schema allows but
`_one_discount_shape` rejects).

**Modes** (see decide/cassette.py for the full rationale): `mode="off"`
(the default) never touches the cassette or the API -- today's template
fallback, unconditionally, so nothing changes for a caller that doesn't ask
for the LLM. `mode="record"` calls the API on a cassette miss and persists
the result, keyed by COHORT (opportunity_type, segment, a coarse
rupees_at_risk band, and an envelope fingerprint) rather than by decision --
that collapses tens of thousands of decisions down to a few dozen real API
calls. `mode="replay"` never calls the API at all: a cassette hit returns the
recorded candidates byte-identically (the reproducibility claim in
SYSTEM_DESIGN.md section 1.2 survives an LLM in the loop only because of
this), and a miss falls back to the template shelf unless `strict_replay=True`,
which raises instead -- for CI, where a silent fallback would hide a stale or
incomplete cassette.
"""

from __future__ import annotations

import json
import os

from revenew.decide.bandit import PosteriorStore
from revenew.decide.cassette import Cassette, cache_key, rupees_band
from revenew.models import (
    ActionFamily,
    Candidate,
    CandidateSet,
    Envelope,
    OpportunityType,
    Segment,
)
from revenew.settings import CLAUDE_MODEL, PolicyConfig

MODEL = CLAUDE_MODEL
MAX_TOKENS = 4096  # 5-8 candidates with headlines + rationales truncate at 1024

CANDIDATE_SET_SCHEMA = CandidateSet.model_json_schema()

SYSTEM_PROMPT = """You are composing commercial offers for an e-commerce merchant.

You will be given: an opportunity (why this cohort of customers is worth
acting on), a constraint envelope (hard caps you must not exceed), and the
customer segment. Propose 5 to 8 CANDIDATE offers. Each must:

- pick exactly one action_family from the allowed list
- stay within max_discount_pct and max_absolute_discount from the envelope
- never reference an excluded SKU
- carry a short, specific headline and a one-sentence rationale grounded in
  the opportunity you were given -- not a generic marketing line

Vary the candidates: different families, different depths, at least one
REMINDER_NUDGE (zero cost) among them. You are proposing options for a
downstream ranking system to choose between, not picking the single best one
yourself. These candidates will be reused across every customer in this
cohort, not just one -- do not reference any one customer's specific name or
order history, only the cohort-level facts given to you.
"""


class CassetteMissError(RuntimeError):
    """Raised in replay mode with strict_replay=True when a cohort has no
    recorded candidates. A silent fallback here would hide a stale or
    incomplete cassette instead of failing loudly -- see cassette.py."""


def _prompt_context(
    opportunity_type: OpportunityType,
    segment: Segment,
    rupees_band_label: str,
    rupees_band_value: float,
    envelope: Envelope,
) -> str:
    """Built from the COHORT, not one customer -- deliberately no exact
    rupees_at_risk and no budget_remaining. Both change on every decision;
    baking either in would make two customers in the same cohort produce
    different prompts (and therefore different cache keys in all but name),
    which defeats cohort-level generation. See cassette.py's
    envelope_fingerprint for the same exclusion, applied to the cache key.
    """
    return json.dumps(
        {
            "opportunity_type": opportunity_type.value,
            "segment": segment.value,
            "rupees_at_risk_band": rupees_band_label,
            "rupees_at_risk_representative": rupees_band_value,
            "envelope": {
                "max_discount_pct": envelope.max_discount_pct,
                "max_absolute_discount": envelope.max_absolute_discount,
                "excluded_skus": envelope.excluded_skus,
                "cogs_known_for_skus": sorted(envelope.cogs_by_sku or {}),
            },
            "allowed_action_families": [f.value for f in ActionFamily],
        },
        indent=2,
        sort_keys=True,
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
    def __init__(
        self,
        client=None,
        *,
        mode: str = "off",
        cassette: Cassette | None = None,
        strict_replay: bool = False,
    ) -> None:
        if mode not in ("off", "record", "replay"):
            raise ValueError(f"mode must be 'off', 'record', or 'replay', got {mode!r}")
        self.mode = mode
        self.cassette = cassette if cassette is not None else (Cassette() if mode != "off" else None)
        self.strict_replay = strict_replay
        # In "off" mode this stays None even if a credential is present --
        # the default must never make a live call just because ANTHROPIC_API_KEY
        # happens to be set in the environment. Only "record"/"replay" resolve
        # a client (and even "replay" never actually calls it -- see generate()).
        self.client = None if mode == "off" else (client if client is not None else _client())

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
        if self.mode == "off":
            return _template_fallback(segment, envelope, policy, store)

        key = cache_key(opportunity_type, segment, rupees_at_risk, envelope)

        if self.mode == "replay":
            cached = self.cassette.get(key)
            if cached is not None:
                return cached
            if self.strict_replay:
                raise CassetteMissError(
                    f"no recorded candidates for cohort {key} "
                    f"({opportunity_type.value}/{segment.value}) -- run in "
                    "mode='record' first, or drop strict_replay"
                )
            return _template_fallback(segment, envelope, policy, store)

        # mode == "record"
        cached = self.cassette.get(key)
        if cached is not None:
            return cached  # a hit never calls the API, even in record mode

        if self.client is None:
            # No credential to fill the miss with. Fall back, but do NOT
            # persist a template candidate to the cassette as if it were a
            # real recording -- a later run with a real credential must still
            # see this as a miss and try again.
            return _template_fallback(segment, envelope, policy, store)

        import anthropic  # local: keeps `anthropic` optional for off/replay-only callers

        band_label, band_value = rupees_band(rupees_at_risk)
        context = _prompt_context(opportunity_type, segment, band_label, band_value, envelope)
        try:
            result = self._call(context, retry_hint=None)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
            # A bad key must be loud in record mode -- this is precisely how
            # "the LLM never actually ran" stayed invisible for so long: a
            # bare `except Exception` here would silently downgrade to the
            # template shelf and look, from the outside, exactly like success.
            raise
        except Exception:
            # Connectivity, rate limit, timeout, 5xx -- degrade, don't die.
            return _template_fallback(segment, envelope, policy, store)

        parsed = _try_parse(result)
        if parsed is None:
            # One retry with the schema echoed back, per the failure table.
            try:
                result2 = self._call(context, retry_hint=_schema_hint())
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
                raise
            except Exception:
                return _template_fallback(segment, envelope, policy, store)
            parsed = _try_parse(result2)

        if parsed is None:
            return None  # genuinely malformed twice -> no_action, per the failure table

        self.cassette.put(key, parsed)
        return parsed

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
                    # Guarantees tool_use.input validates exactly against the
                    # schema server-side -- CandidateSet/Candidate's
                    # extra="forbid" (models.py) is what makes
                    # additionalProperties:false true of the schema, which
                    # this requires. Most of the "malformed JSON" failure
                    # path becomes unreachable in practice; the retry-then-
                    # None path stays for what strict mode can't catch (e.g.
                    # a family/discount-shape combination the schema allows
                    # but Candidate._one_discount_shape rejects).
                    "strict": True,
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
