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
from pathlib import Path

from revenew.models import CandidateSet, Envelope, OpportunityType, Segment

DEFAULT_CASSETTE_DIR = Path(__file__).resolve().parent.parent.parent / "cassettes" / "candidates"

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
    what offer the model should compose. Deliberately EXCLUDES
    `budget_remaining`: budget is a validity gate `EnvelopeValidator` applies
    to each candidate downstream, not an input to what an offer should look
    like, and it moves on every single decision as the ledger is spent down --
    keying on it would refragment the cache back to one entry per decision,
    which defeats the entire point of cohort-level generation.
    """
    payload = {
        "max_discount_pct": envelope.max_discount_pct,
        "max_absolute_discount": envelope.max_absolute_discount,
        "excluded_skus": sorted(envelope.excluded_skus),
        "cooldown_days": envelope.cooldown_days,
        "max_offers_per_customer_per_month": envelope.max_offers_per_customer_per_month,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def cache_key(
    opportunity_type: OpportunityType,
    segment: Segment,
    rupees_at_risk: float,
    envelope: Envelope,
) -> str:
    """The cohort key: (opportunity_type, segment, rupees_band, envelope_fingerprint).

    Deliberately NOT keyed on customer_id or opportunity_id -- candidates are
    a property of the cohort, and the bandit (not the generator) is what
    personalizes the final choice. This is the same "cohort-level, not
    per-customer" boundary SYSTEM_DESIGN.md section 9 draws.
    """
    band_label, _ = rupees_band(rupees_at_risk)
    fp = envelope_fingerprint(envelope)
    raw = f"{opportunity_type.value}|{segment.value}|{band_label}|{fp}"
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
