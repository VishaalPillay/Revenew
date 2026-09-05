"""Candidate cassette: recorded LLM responses keyed by a COHORT, not a
decision, so a real, multi-candidate LLM can sit in the decision loop without
breaking either of the two things SYSTEM_DESIGN.md refuses to trade away.

**Cost/latency (SYSTEM_DESIGN.md section 9).** A 3,000-customer, 30-day replay
makes on the order of tens of thousands of decisions. One LLM call per
decision is tens of millions of tokens and hours of wall time -- section 9
names the fix directly: "generate per (segment, opportunity_type) rather than
per customer -- candidates are cohort-level already, so this is caching, not
degradation." `cache_key()` below is that cache: it collapses the call volume
from "one per decision" to "one per distinct cohort", typically a few dozen.

**Reproducibility (SYSTEM_DESIGN.md section 1.2).** "Same fixture + seed =>
byte-identical posteriors" is non-negotiable, and an LLM is not deterministic.
A live call inside the replayed decision path would make two runs of the same
seed diverge the moment the model's phrasing (not even its chosen family)
differed by a token. The cassette is the answer: record once against the real
API, commit the JSON files, and every subsequent run -- including on a
judge's machine with no credential at all -- replays the exact same
candidates byte-for-byte, because it never calls the API again.

Modes on `CandidateGenerator` (decide/generator.py):

    off     -- ignore the cassette entirely. Today's template fallback,
               unconditionally. The default, so a credential-less run behaves
               exactly as it always has.
    record  -- on a cache miss, call the API, persist the parsed CandidateSet,
               then return it. A cache HIT never calls the API even here --
               "record" means "fill what's missing", not "always call".
    replay  -- cassette only. A miss returns None (the caller falls through to
               the templated fallback) unless `strict=True`, in which case a
               miss raises -- for CI, where a silent fallback would hide a
               stale or incomplete cassette rather than fail loudly.

The cassette itself is a plain directory of one JSON file per key --
diffable, reviewable in a PR, and exactly what "commit the cassette" means in
practice.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from revenew.models import CandidateSet, Envelope, OpportunityType, Segment

DEFAULT_CASSETTE_DIR = Path(__file__).resolve().parent.parent.parent / "cassettes" / "candidates"

# Bump whenever the PROMPT TEXT itself changes shape (a new field added to
# `_prompt_context`, the system prompt's instructions rewritten, and so on)
# in a way that is not already covered by `envelope_fingerprint` or
# `catalog_fingerprint` below. A cassette recorded against an older prompt
# answers a question the current prompt no longer asks -- replaying it as if
# it still applies is exactly the silent-degradation failure class
# ENGINEERING_LOG.md #11 and #13 both describe. Folded into every cache key
# unconditionally, so bumping this is the deliberate, visible way to force a
# full re-record; forgetting to bump it when it matters is the accident this
# constant exists to make rare, not the accident it can prevent by itself.
#
# History: 1 -- original prompt, no catalog. 2 -- `_prompt_context` gained a
# `catalog` field (see `catalog_fingerprint`); bumped alongside it because
# the catalog fingerprint alone only protects against catalog *content*
# changing, not against the prompt learning to use that content differently.
PROMPT_VERSION = 2

# Coarse rupees_at_risk buckets, with round boundaries rather than ones
# derived from any one fixture's product catalog -- the cache key has to stay
# meaningful for a merchant whose prices look nothing like this project's
# synthetic apparel/footwear catalog. Each band carries a REPRESENTATIVE
# figure that goes into the prompt IN PLACE OF a customer's exact
# rupees_at_risk -- that substitution is what lets two different customers in
# the same band collapse onto one cache key and one LLM call, rather than the
# literal float re-fragmenting the cache on every decision.
RUPEES_BANDS: list[tuple[float, float, str, float]] = [
    (0, 500, "under Rs 500", 300),
    (500, 2000, "Rs 500-2000", 1200),
    (2000, float("inf"), "over Rs 2000", 3000),
]


def rupees_band(rupees_at_risk: float) -> tuple[str, float]:
    """(label, representative_value) for the band `rupees_at_risk` falls into."""
    for lo, hi, label, rep in RUPEES_BANDS:
        if lo <= rupees_at_risk < hi:
            return label, rep
    return RUPEES_BANDS[-1][2], RUPEES_BANDS[-1][3]


def envelope_fingerprint(envelope: Envelope) -> str:
    """Hash of the COMPOSITION inputs only -- the fields that actually change
    what offer the model should compose.

    Everything excluded here is excluded for the same reason: it governs
    whether an offer may be *sent*, not what a good offer *is*.

      `budget_remaining`     a validity gate `EnvelopeValidator` applies
                             downstream, and it moves on every single decision
                             as the ledger is spent down -- keying on it would
                             refragment the cache back to one entry per
                             decision, defeating cohort-level generation
                             entirely.

      `cooldown_days`,       pure ELIGIBILITY rules: they decide whether this
      `max_offers_per_       customer may receive anything at all right now,
       customer_per_month`   and have no bearing whatsoever on what the model
                             should propose. Keying on them meant that merely
                             tuning a merchant's contact policy silently
                             invalidated every recorded cohort, and -- because
                             a non-strict replay falls back to a templated
                             single candidate on a miss -- quietly reverted the
                             whole system to the one-candidate greedy argmax
                             that ENGINEERING_LOG.md #11 exists to describe.
                             Found exactly that way: a policy-tuning experiment
                             produced 55,535 decisions with one candidate each
                             and propensity 1.0 on every one.

    The lesson generalises past this function: composition inputs and
    eligibility gates are different things, and conflating them is the same
    mistake `v_candidate_compliance` (db/schema.sql) exists to undo on the
    reporting side.
    """
    payload = {
        "max_discount_pct": envelope.max_discount_pct,
        "max_absolute_discount": envelope.max_absolute_discount,
        "excluded_skus": sorted(envelope.excluded_skus),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def catalog_fingerprint(catalog: Sequence[Mapping[str, object]]) -> str:
    """Hash of the catalog CONTENT the prompt is built from.

    This is the fix for a real, already-caught failure mode: enriching
    `_prompt_context` with product names/prices changes what the model is
    asked, but if the cache key does not change too, every already-recorded
    cassette entry keeps matching and keeps returning candidates composed in
    ignorance of the catalog -- tests pass, the demo looks fine, and the
    feature silently does nothing. Folding a hash of the actual catalog rows
    into the key means a changed catalog (a new SKU, a repriced product) is
    caught automatically the moment it changes the SORTED, JSON-serialized
    row set, with no reliance on a human remembering to bump
    `PROMPT_VERSION` for every future catalog edit.

    Sorted by `sku` before hashing (independent of the order the query
    happened to return rows in) so the fingerprint is a function of content
    alone, not incidental row order.
    """
    normalized = sorted(
        (
            {"sku": item["sku"], "name": item["name"], "category": item["category"], "price": item["price"]}
            for item in catalog
        ),
        key=lambda item: item["sku"],
    )
    payload = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_key(
    opportunity_type: OpportunityType,
    segment: Segment,
    rupees_at_risk: float,
    envelope: Envelope,
    catalog: Sequence[Mapping[str, object]] = (),
) -> str:
    """The cohort key: (prompt_version, opportunity_type, segment, rupees_band,
    envelope_fingerprint, catalog_fingerprint).

    Deliberately NOT keyed on customer_id or opportunity_id -- candidates are
    a property of the cohort, and the bandit (not the generator) is what
    personalizes the final choice. This is the same "cohort-level, not
    per-customer" boundary SYSTEM_DESIGN.md section 9 draws.

    `catalog` defaults to empty so every call site that does not care about
    catalog-driven cache invalidation (most of this module's own tests, which
    are about envelope-fingerprint exclusion behaviour) keeps working
    unchanged -- only `CandidateGenerator.generate`, which has a real catalog
    to pass, needs to supply one.
    """
    band_label, _ = rupees_band(rupees_at_risk)
    fp = envelope_fingerprint(envelope)
    cfp = catalog_fingerprint(catalog)
    raw = f"{PROMPT_VERSION}|{opportunity_type.value}|{segment.value}|{band_label}|{fp}|{cfp}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class Cassette:
    """A directory of recorded `CandidateSet`s, one JSON file per cache key."""

    def __init__(self, directory: Path | str = DEFAULT_CASSETTE_DIR) -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> CandidateSet | None:
        path = self._path(key)
        if not path.exists():
            return None
        return CandidateSet.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, candidate_set: CandidateSet) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(candidate_set.model_dump_json(indent=2), encoding="utf-8")
