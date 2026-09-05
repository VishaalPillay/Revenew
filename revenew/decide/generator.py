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

**Provider: Groq, not Anthropic.** SYSTEM_DESIGN.md section 3.1 names "Claude
(Sonnet tier)" -- this is a disclosed, constraint-driven deviation (no
Anthropic billing available in this environment; see ENGINEERING_LOG.md and
revenew/settings.py). Structured output is enforced via Groq's
`response_format={"type": "json_schema", ..., "strict": True}`, which -- as
of this writing -- only a handful of Groq-hosted models honor
(openai/gpt-oss-20b/120b, qwen/qwen3-32b); GROQ_MODEL defaults to the
20b variant. `strict: true` plus `Candidate`/`CandidateSet`'s
`extra="forbid"` (models.py) is what makes a response that doesn't match the
schema structurally impossible rather than merely likely -- the
malformed-output retry path stays for what constrained decoding can't catch
(a family/discount-shape combination the JSON schema allows but
`Candidate._one_discount_shape` rejects).

Every other piece of this design -- cohort-level cache keys, the cassette,
the off/record/replay modes, the typed-exception split between "loud" and
"degrade" -- is provider-agnostic by construction; the swap from Anthropic
to Groq touched only this module's `_client()`/`_call()` and its exception
types.

**A fourth mode, `"shelf"`, makes no LLM call at all** -- it delegates to
`decide.shelf.ShelfGenerator`, which lives in its OWN module precisely so
this docstring's "the only LLM call in the system" claim stays true even
with a fourth generation mode. It exists for PLAN.md section 5's Arm B: a
fixed, five-family template shelf that gives the bandit a real multi-family
menu to Thompson-sample over WITHOUT an LLM in the loop, isolating what
*learning* adds (Arm A -> B) from what the *LLM* adds (Arm B -> C).
"""

from __future__ import annotations

import json
import os
import sqlite3

from revenew.decide.bandit import PosteriorStore
from revenew.decide.cassette import Cassette, cache_key, rupees_band
from revenew.decide.shelf import ShelfGenerator
from revenew.models import (
    ActionFamily,
    Candidate,
    CandidateSet,
    Envelope,
    OpportunityType,
    Segment,
)
from revenew.settings import GROQ_MODEL, PolicyConfig

MODEL = GROQ_MODEL
MAX_TOKENS = 4096  # 5-8 candidates with headlines + rationales truncate at 1024


def _require_every_property(schema: dict | list) -> dict | list:
    """Groq's (and OpenAI's) `strict: true` json_schema mode requires EVERY
    property of an object to be listed in `required`, with optionality
    expressed through the property's own type (`anyOf: [..., {"type":
    "null"}]`) rather than omission -- the opposite of Pydantic's default,
    which only lists fields that have no default. Discovered against the
    live API, not by inspection: `model_json_schema()` on `Candidate` leaves
    `discount_amount`/`discount_pct`/`skus` out of `required` (they all have
    defaults), and Groq's API rejects that shape outright with a 400 before
    ever running inference -- the exact three field names it lists as
    missing. Recurses into `$defs` since `Candidate`, nested under
    `CandidateSet`, has the same requirement.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
        for value in schema.values():
            _require_every_property(value)
    elif isinstance(schema, list):
        for item in schema:
            _require_every_property(item)
    return schema


CANDIDATE_SET_SCHEMA = _require_every_property(CandidateSet.model_json_schema())

SYSTEM_PROMPT = """You are composing commercial offers for an e-commerce merchant.

You will be given: an opportunity (why this cohort of customers is worth
acting on), a constraint envelope (hard caps you must not exceed), the
customer segment, and the merchant's product catalog. Propose 5 to 8
CANDIDATE offers. Each must:

- pick exactly one action_family from the allowed list
- stay within max_discount_pct and max_absolute_discount from the envelope
- never reference an excluded SKU
- carry a short, specific headline and a one-sentence rationale grounded in
  the opportunity you were given -- not a generic marketing line

If you propose a BUNDLE_OFFER, name real products from the given catalog in
`skus` (by their SKU code) -- pick items that plausibly complement each
other (same or adjacent category is a reasonable signal), not an arbitrary
pair. Do not invent a SKU that is not in the catalog you were given.

Vary the candidates: different families, different depths, at least one
REMINDER_NUDGE (zero cost) among them. You are proposing options for a
downstream ranking system to choose between, not picking the single best one
yourself. These candidates will be reused across every customer in this
cohort, not just one -- do not reference any one customer's specific name,
order history, or which product THEY specifically own; reason only from the
cohort-level facts and the catalog you were given.

Respond with a JSON object matching the required schema exactly -- no prose,
no markdown fencing, just the JSON object.
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
    catalog: list[dict],
) -> str:
    """Built from the COHORT, not one customer -- deliberately no exact
    rupees_at_risk and no budget_remaining. Both change on every decision;
    baking either in would make two customers in the same cohort produce
    different prompts (and therefore different cache keys in all but name),
    which defeats cohort-level generation. See cassette.py's
    envelope_fingerprint for the same exclusion, applied to the cache key.

    `catalog` is the full merchant catalog (name/category/price per SKU),
    not any one customer's `recommended_sku` -- that field is per-customer
    (detect/queries.sql's cross-sell CTE ranks one best pairing per
    customer's own basket) and injecting it here would fragment the cache
    back toward one entry per customer, the same mistake `rupees_at_risk`
    and `budget_remaining` are already excluded to avoid. Giving the model
    the whole catalog instead lets it reason about plausible bundles at the
    cohort level using only information every customer in the cohort shares
    -- real product identity, not one query's specific recommendation.
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
            "catalog": [
                {"sku": item["sku"], "name": item["name"], "category": item["category"], "price": item["price"]}
                for item in catalog
            ],
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
    if not os.environ.get("GROQ_API_KEY"):
        return None
    import groq

    return groq.Groq()


class CandidateGenerator:
    def __init__(
        self,
        client=None,
        *,
        mode: str = "off",
        cassette: Cassette | None = None,
        strict_replay: bool = False,
        shelf: ShelfGenerator | None = None,
    ) -> None:
        if mode not in ("off", "record", "replay", "shelf"):
            raise ValueError(f"mode must be 'off', 'record', 'replay', or 'shelf', got {mode!r}")
        self.mode = mode
        self.cassette = cassette if cassette is not None else (Cassette() if mode in ("record", "replay") else None)
        self.strict_replay = strict_replay
        self.shelf = shelf if shelf is not None else (ShelfGenerator() if mode == "shelf" else None)
        # In "off"/"shelf" this stays None even if a credential is present --
        # neither mode may ever make a live call just because GROQ_API_KEY
        # happens to be set in the environment. Only "record"/"replay" resolve
        # a client (and even "replay" never actually calls it -- see generate()).
        self.client = None if mode in ("off", "shelf") else (client if client is not None else _client())

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
        catalog: list[dict],
        conn: sqlite3.Connection | None = None,
    ) -> CandidateSet | None:
        if self.mode == "off":
            return _template_fallback(segment, envelope, policy, store)

        if self.mode == "shelf":
            if conn is None:
                raise ValueError("mode='shelf' requires generate(conn=...) -- see decide/shelf.py")
            return self.shelf.build(
                conn, opportunity_type=opportunity_type, segment=segment,
                rupees_at_risk=rupees_at_risk, envelope=envelope, catalog=catalog,
            )

        key = cache_key(opportunity_type, segment, rupees_at_risk, envelope, catalog)

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

        import groq  # local: keeps `groq` optional for off/replay-only callers

        band_label, band_value = rupees_band(rupees_at_risk)
        context = _prompt_context(opportunity_type, segment, band_label, band_value, envelope, catalog)
        try:
            raw = self._call(context, retry_hint=None)
        except (groq.AuthenticationError, groq.PermissionDeniedError):
            # A bad key must be loud in record mode -- this is precisely how
            # "the LLM never actually ran" stayed invisible for so long: a
            # bare `except Exception` here would silently downgrade to the
            # template shelf and look, from the outside, exactly like success.
            raise
        except Exception:
            # Connectivity, rate limit, timeout, 5xx -- degrade, don't die.
            # NOTE: this is SDK-level failure only -- `_call()` itself never
            # raises on a response that came back but wasn't valid JSON or
            # didn't match the schema; that's "malformed", handled below via
            # the retry, not "unreachable".
            return _template_fallback(segment, envelope, policy, store)

        parsed = _try_parse(raw) if raw is not None else None
        if parsed is None:
            # One retry with the schema echoed back, per the failure table.
            # Covers both a non-JSON response and JSON that doesn't match
            # CandidateSet's shape -- both are "the model responded but never
            # produced anything usable", not a connectivity problem.
            try:
                raw2 = self._call(context, retry_hint=_schema_hint())
            except (groq.AuthenticationError, groq.PermissionDeniedError):
                raise
            except Exception:
                return _template_fallback(segment, envelope, policy, store)
            parsed = _try_parse(raw2) if raw2 is not None else None

        if parsed is None:
            return None  # genuinely malformed twice -> no_action, per the failure table

        self.cassette.put(key, parsed)
        return parsed

    def _call(self, context: str, *, retry_hint: str | None) -> dict | None:
        """Returns the parsed response body, or None if the model returned no
        content or content that isn't valid JSON -- both count as "malformed
        output" for the caller's retry logic, not as an SDK/connectivity
        failure, so this method itself never raises on either.
        """
        user_content = context if retry_hint is None else f"{context}\n\n{retry_hint}"
        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "propose_candidates",
                    "strict": True,
                    # Guarantees the response validates exactly against the
                    # schema via constrained decoding -- CandidateSet/
                    # Candidate's extra="forbid" (models.py) is what makes
                    # additionalProperties:false true of the schema, which
                    # Groq's strict mode requires. Only honored on a handful
                    # of Groq-hosted models (GROQ_MODEL defaults to one); on
                    # any model that ignores strict, the retry-then-None path
                    # above is what catches the resulting malformed output.
                    "schema": CANDIDATE_SET_SCHEMA,
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None


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
