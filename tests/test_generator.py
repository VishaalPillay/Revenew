"""CandidateGenerator: the four LLM failure-mode rows SYSTEM_DESIGN.md
section 7 marks "unit"-tested (malformed JSON -> one retry, all candidates
invalid, LLM unreachable -> fallback, and the connectivity/auth split that
makes a bad key loud instead of a silent downgrade), plus the cassette modes
that make cohort-level generation reproducible. None of this needs a real
credential -- every case here injects a fake client shaped like Groq's
OpenAI-compatible chat.completions response.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import groq
import httpx
import pytest

from revenew.decide.bandit import PosteriorStore
from revenew.decide.cassette import Cassette, cache_key
from revenew.decide.generator import CANDIDATE_SET_SCHEMA, CandidateGenerator, CassetteMissError
from revenew.models import CandidateSet, Envelope, OpportunityType, Segment
from revenew.settings import DEFAULT_POLICY

NOW = datetime(2026, 1, 1, tzinfo=UTC)

ENVELOPE = Envelope(
    max_discount_pct=0.20, max_absolute_discount=500.0, budget_remaining=10_000.0,
    excluded_skus=[], cooldown_days=30, max_offers_per_customer_per_month=1, cogs_by_sku=None,
)

CATALOG = [
    {"sku": "SKU-A01", "name": "Classic Tee", "category": "apparel", "price": 599.0},
    {"sku": "SKU-A02", "name": "Slim Jeans", "category": "apparel", "price": 1899.0},
]


def _response(content: str) -> SimpleNamespace:
    """A response shaped like Groq's chat.completions.create() return value."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _valid_payload(n: int = 2) -> dict:
    candidates = [
        {
            "action_family": "percent_discount", "headline": f"{5 + i}% off",
            "discount_pct": 0.05 + i * 0.01, "discount_amount": None, "skus": [],
            "rationale": "cohort-level rationale",
        }
        for i in range(n)
    ]
    candidates.append(
        {"action_family": "reminder_nudge", "headline": "Still thinking of you",
         "discount_pct": None, "discount_amount": None, "skus": [], "rationale": "zero-cost option"}
    )
    return {"candidates": candidates}


class FakeClient:
    """Records every call and returns queued responses/exceptions in order."""

    def __init__(self, *reactions) -> None:
        self._reactions = list(reactions)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reaction = self._reactions.pop(0)
        if isinstance(reaction, Exception):
            raise reaction
        return reaction


def _auth_error() -> groq.AuthenticationError:
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(401, request=req)
    return groq.AuthenticationError("invalid api key", response=resp, body=None)


@pytest.fixture
def store(seeded_conn):
    s = PosteriorStore(seeded_conn)
    s.ensure_initialized()
    return s


def _generate(gen: CandidateGenerator, store):
    return gen.generate(
        opportunity_type=OpportunityType.DORMANT_WINBACK, segment=Segment.DORMANT,
        rupees_at_risk=750.0, envelope=ENVELOPE, store=store, policy=DEFAULT_POLICY,
        catalog=CATALOG,
    )


# ========================================================= cassette keying --


def test_eligibility_policy_changes_do_not_invalidate_the_cassette():
    """Regression for a silent, severe failure: `envelope_fingerprint` used to
    include `cooldown_days` and `max_offers_per_customer_per_month`, so merely
    tuning a merchant's contact policy changed every cache key. In a
    non-strict replay that means a miss on every cohort, a fallback to the
    single templated candidate, and the whole system quietly reverting to the
    one-candidate greedy argmax of ENGINEERING_LOG.md #11 -- propensity 1.0 on
    every decision, with nothing in the output saying so. Those two fields
    govern whether a customer may be contacted, not what to offer them."""
    lenient = ENVELOPE.model_copy(update={"cooldown_days": 7, "max_offers_per_customer_per_month": 4})
    assert cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE) == cache_key(
        OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, lenient
    )


def test_budget_remaining_does_not_invalidate_the_cassette():
    """Same principle, the original case: budget moves on every decision as
    the ledger is spent down, so keying on it would refragment the cache to
    one entry per decision."""
    spent = ENVELOPE.model_copy(update={"budget_remaining": 12.5})
    assert cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE) == cache_key(
        OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, spent
    )


def test_composition_rules_DO_invalidate_the_cassette():
    """The other half of the contract: a genuinely different discount cap or
    SKU exclusion list must produce a different key, or the cassette would
    serve offers composed against rules that no longer apply."""
    base = cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE)
    assert base != cache_key(
        OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0,
        ENVELOPE.model_copy(update={"max_discount_pct": 0.05}),
    )
    assert base != cache_key(
        OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0,
        ENVELOPE.model_copy(update={"excluded_skus": ["SKU-BANNED"]}),
    )


def test_catalog_content_change_invalidates_the_cassette():
    """The landmine this project's own history warned about: enriching the
    prompt with product data (name/category/price) changes what the model is
    asked, but if `cache_key` does not change too, every already-recorded
    cassette entry keeps hitting and keeps returning candidates composed in
    ignorance of the catalog -- tests pass, the demo looks fine, and the
    feature silently does nothing. Same failure class as
    ENGINEERING_LOG.md #11 (a single-candidate fallback silently defeating
    Thompson sampling) and #13 (a schema mismatch silently defeating every
    live call) -- both were invisible until someone looked at the actual
    values flowing through, not just whether the code ran without error."""
    base = cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE, CATALOG)

    reordered_same_content = list(reversed(CATALOG))
    assert base == cache_key(
        OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE, reordered_same_content
    ), "row order must not matter -- the fingerprint is a function of content, not incidental order"

    repriced = [dict(CATALOG[0], price=CATALOG[0]["price"] + 1), CATALOG[1]]
    assert base != cache_key(
        OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE, repriced
    ), "a repriced product must be a cache miss, not a stale hit"

    fewer_products = CATALOG[:1]
    assert base != cache_key(
        OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE, fewer_products
    ), "a changed catalog SIZE must also be a cache miss"

    assert base != cache_key(
        OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE
    ), "an empty catalog (the pre-Phase-0 prompt shape) must not collide with a real one"


# ============================================================= strict schema --


def _object_schemas(schema) -> list[dict]:
    """Every object-type schema reachable from `schema`, including nested
    `$defs` -- Groq's strict json_schema mode enforces the "every property is
    required" rule at EVERY level, not just the top one."""
    found = []
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            found.append(schema)
        for value in schema.values():
            found.extend(_object_schemas(value))
    elif isinstance(schema, list):
        for item in schema:
            found.extend(_object_schemas(item))
    return found


def test_candidate_set_schema_lists_every_property_as_required():
    """Regression for the bug this hardens against: Groq's `strict: true`
    json_schema mode rejects a schema outright (HTTP 400, before running any
    inference at all) unless EVERY property of EVERY object -- including
    `Candidate`, nested under `$defs` -- appears in `required`, with
    optionality expressed via the property's own `anyOf: [..., {"type":
    "null"}]` rather than omission. Pydantic's default `model_json_schema()`
    only lists fields with no default (so `discount_pct`/`discount_amount`/
    `skus` were missing), which is exactly what a live call against the real
    API caught and every fake-client test above did not, because none of them
    exercise Groq's own schema validator. This test does not need a
    credential or the network -- it checks the same invariant statically."""
    object_schemas = _object_schemas(CANDIDATE_SET_SCHEMA)
    assert len(object_schemas) >= 2  # CandidateSet itself, plus Candidate under $defs
    for obj in object_schemas:
        assert set(obj["required"]) == set(obj["properties"]), (
            f"schema titled {obj.get('title')!r} has properties not listed in required: "
            f"{set(obj['properties']) - set(obj['required'])}"
        )


# ==================================================================== off --


def test_off_mode_never_touches_an_injected_client(store):
    client = FakeClient()  # no reactions queued -- a call would raise IndexError
    gen = CandidateGenerator(client, mode="off")
    result = _generate(gen, store)
    assert result is not None and len(result.candidates) == 1
    assert client.calls == []


# ==================================================================== shelf --


def test_shelf_mode_never_touches_an_injected_client(store, seeded_conn):
    client = FakeClient()  # no reactions queued -- a call would raise IndexError
    gen = CandidateGenerator(client, mode="shelf")
    assert gen.client is None
    result = gen.generate(
        opportunity_type=OpportunityType.DORMANT_WINBACK, segment=Segment.DORMANT,
        rupees_at_risk=750.0, envelope=ENVELOPE, store=store, policy=DEFAULT_POLICY,
        catalog=CATALOG, conn=seeded_conn,
    )
    assert result is not None and len(result.candidates) >= 4
    assert client.calls == []


def test_shelf_mode_without_conn_raises():
    gen = CandidateGenerator(mode="shelf")
    with pytest.raises(ValueError):
        gen.generate(
            opportunity_type=OpportunityType.DORMANT_WINBACK, segment=Segment.DORMANT,
            rupees_at_risk=750.0, envelope=ENVELOPE, store=None, policy=DEFAULT_POLICY,
            catalog=CATALOG,
        )


def test_shelf_mode_delegates_to_the_shelf_generator(store, seeded_conn):
    """A regression against generator.py accidentally growing its own
    template-composition logic for `mode="shelf"` -- it must be a pure
    delegation to `decide.shelf.ShelfGenerator`, the module PLAN.md section 5
    keeps separate precisely so `generator.py`'s "only LLM call" docstring
    claim stays true."""
    from revenew.decide.shelf import ShelfGenerator

    shelf = ShelfGenerator()
    gen = CandidateGenerator(mode="shelf", shelf=shelf)
    assert gen.shelf is shelf

    result = gen.generate(
        opportunity_type=OpportunityType.DORMANT_WINBACK, segment=Segment.DORMANT,
        rupees_at_risk=750.0, envelope=ENVELOPE, store=store, policy=DEFAULT_POLICY,
        catalog=CATALOG, conn=seeded_conn,
    )
    direct = shelf.build(
        seeded_conn, opportunity_type=OpportunityType.DORMANT_WINBACK, segment=Segment.DORMANT,
        rupees_at_risk=750.0, envelope=ENVELOPE, catalog=CATALOG,
    )
    assert result is direct  # same cache_key -> the memoised object, not a fresh build


# ================================================================= record --


def test_record_mode_calls_once_parses_and_persists_to_cassette(store, tmp_path):
    client = FakeClient(_response(json.dumps(_valid_payload())))
    cassette = Cassette(tmp_path)
    gen = CandidateGenerator(client, mode="record", cassette=cassette)

    result = _generate(gen, store)

    assert result is not None
    assert len(result.candidates) == 3
    assert len(client.calls) == 1

    key = cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE, CATALOG)
    assert cassette.has(key)
    assert cassette.get(key) == result


def test_record_mode_cache_hit_never_calls_the_client(store, tmp_path):
    cassette = Cassette(tmp_path)
    key = cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE, CATALOG)
    cassette.put(key, CandidateSet.model_validate(_valid_payload(1)))

    client = FakeClient()  # any call would raise IndexError
    gen = CandidateGenerator(client, mode="record", cassette=cassette)
    result = _generate(gen, store)

    assert result is not None
    assert client.calls == []


def test_record_mode_retries_once_on_malformed_then_succeeds(store, tmp_path):
    malformed = _response("not json at all")
    valid = _response(json.dumps(_valid_payload()))
    client = FakeClient(malformed, valid)
    gen = CandidateGenerator(client, mode="record", cassette=Cassette(tmp_path))

    result = _generate(gen, store)

    assert result is not None
    assert len(client.calls) == 2
    # the retry call must echo the schema, per the failure-mode table
    assert "schema" in client.calls[1]["messages"][1]["content"].lower()


def test_record_mode_empty_content_retries_rather_than_degrading(store, tmp_path):
    """Regression: `_call()` must treat 'no content' and 'content that isn't
    valid JSON' as MALFORMED OUTPUT (retry, then give up), not as a
    connectivity failure (immediate degrade) -- the two are easy to conflate
    if `_call()` raises on either, which would skip the retry path entirely
    on the very first attempt."""
    empty = _response("")
    valid = _response(json.dumps(_valid_payload()))
    client = FakeClient(empty, valid)
    gen = CandidateGenerator(client, mode="record", cassette=Cassette(tmp_path))

    result = _generate(gen, store)

    assert result is not None
    assert len(client.calls) == 2


def test_record_mode_malformed_twice_returns_none(store, tmp_path):
    malformed = _response(json.dumps({"candidates": []}))  # violates min_length=1
    client = FakeClient(malformed, malformed)
    gen = CandidateGenerator(client, mode="record", cassette=Cassette(tmp_path))

    result = _generate(gen, store)

    assert result is None
    assert len(client.calls) == 2


def test_record_mode_connectivity_error_degrades_to_template(store, tmp_path):
    client = FakeClient(RuntimeError("connection reset"))
    gen = CandidateGenerator(client, mode="record", cassette=Cassette(tmp_path))

    result = _generate(gen, store)

    assert result is not None
    assert len(result.candidates) == 1
    assert "Templated fallback" in result.candidates[0].rationale


def test_record_mode_auth_error_is_raised_not_swallowed(store, tmp_path):
    """The bug this hardens against: a bare `except Exception` here would
    silently downgrade a bad API key to template candidates, which is
    precisely how 'the LLM never ran' stayed invisible in this codebase."""
    client = FakeClient(_auth_error())
    gen = CandidateGenerator(client, mode="record", cassette=Cassette(tmp_path))

    with pytest.raises(groq.AuthenticationError):
        _generate(gen, store)


def test_record_mode_no_credential_degrades_without_caching_the_miss(store, tmp_path, monkeypatch):
    # This dev environment may have a real GROQ_API_KEY in .env, so
    # simulating "no credential resolvable" means patching `_client()`
    # itself, not just passing client=None -- passing None with a real key
    # present would resolve a live client, which is not what this test means
    # to exercise.
    monkeypatch.setattr("revenew.decide.generator._client", lambda: None)
    cassette = Cassette(tmp_path)
    gen = CandidateGenerator(None, mode="record", cassette=cassette)
    assert gen.client is None

    result = _generate(gen, store)

    assert result is not None
    key = cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE, CATALOG)
    assert not cassette.has(key)  # a template fallback must never be cached as a real recording


# ================================================================= replay --


def test_replay_mode_hit_returns_recorded_set_with_no_client_at_all(store, tmp_path, monkeypatch):
    monkeypatch.setattr("revenew.decide.generator._client", lambda: None)
    cassette = Cassette(tmp_path)
    key = cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE, CATALOG)
    recorded = CandidateSet.model_validate(_valid_payload(2))
    cassette.put(key, recorded)

    gen = CandidateGenerator(None, mode="replay", cassette=cassette)
    assert gen.client is None  # replay never needs a credential

    result = _generate(gen, store)
    assert result == recorded


def test_replay_mode_miss_falls_back_to_template_by_default(store, tmp_path, monkeypatch):
    monkeypatch.setattr("revenew.decide.generator._client", lambda: None)
    gen = CandidateGenerator(None, mode="replay", cassette=Cassette(tmp_path))
    result = _generate(gen, store)
    assert result is not None
    assert "Templated fallback" in result.candidates[0].rationale


def test_replay_mode_miss_raises_under_strict(store, tmp_path, monkeypatch):
    monkeypatch.setattr("revenew.decide.generator._client", lambda: None)
    gen = CandidateGenerator(None, mode="replay", cassette=Cassette(tmp_path), strict_replay=True)
    with pytest.raises(CassetteMissError):
        _generate(gen, store)


# ============================================================ integration --


def test_generator_output_that_violates_the_envelope_never_reaches_execution(seeded_conn, tmp_path):
    """End-to-end through decide_one_opportunity: an LLM that proposes only
    illegal candidates must still result in ALL_CANDIDATES_INVALID, not an
    executed decision -- the envelope is enforced regardless of what the
    model (or, here, a fake standing in for it) returns."""
    from revenew.decide import decide_one_opportunity
    from revenew.models import DecisionStatus, NoActionReason

    illegal_payload = {
        "candidates": [
            {"action_family": "percent_discount", "headline": "way too much off",
             "discount_pct": 0.99, "discount_amount": None, "skus": [], "rationale": "illegal"},
        ]
    }
    client = FakeClient(_response(json.dumps(illegal_payload)))
    gen = CandidateGenerator(client, mode="record", cassette=Cassette(tmp_path))

    seeded_conn.execute("INSERT OR IGNORE INTO customers VALUES ('cus_gen_test', ?)", ("2026-01-01",))
    seeded_conn.execute(
        "INSERT INTO opportunity_candidates VALUES ('opp_gen','run1','cus_gen_test',"
        "'dormant_winback','w1','dormant_winback',500,'h',?,NULL)",
        (str(NOW),),
    )
    seeded_conn.execute(
        "INSERT INTO opportunities VALUES ('opp_gen','run1','cus_gen_test','w1','dormant','treatment',?)",
        (str(NOW),),
    )
    seeded_conn.commit()

    store_ = PosteriorStore(seeded_conn)
    store_.ensure_initialized()

    decision = decide_one_opportunity(
        seeded_conn, opportunity_id="opp_gen", customer_id="cus_gen_test", segment=Segment.DORMANT,
        opportunity_type=OpportunityType.DORMANT_WINBACK, rupees_at_risk=500.0, run_id="run1",
        policy=DEFAULT_POLICY, generator=gen, bandit_seed=1, now=NOW,
    )

    assert decision.status == DecisionStatus.NO_ACTION
    assert decision.no_action_reason == NoActionReason.ALL_CANDIDATES_INVALID
