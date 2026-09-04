"""CandidateGenerator: the four LLM failure-mode rows SYSTEM_DESIGN.md
section 7 marks "unit"-tested (malformed JSON -> one retry, all candidates
invalid, LLM unreachable -> fallback, and the connectivity/auth split that
makes a bad key loud instead of a silent downgrade), plus the cassette modes
that make cohort-level generation reproducible. None of this needs a real
credential -- every case here injects a fake client.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from revenew.decide.bandit import PosteriorStore
from revenew.decide.cassette import Cassette, cache_key
from revenew.decide.generator import CandidateGenerator, CassetteMissError
from revenew.models import CandidateSet, Envelope, OpportunityType, Segment
from revenew.settings import DEFAULT_POLICY

NOW = datetime(2026, 1, 1, tzinfo=UTC)

ENVELOPE = Envelope(
    max_discount_pct=0.20, max_absolute_discount=500.0, budget_remaining=10_000.0,
    excluded_skus=[], cooldown_days=30, max_offers_per_customer_per_month=1, cogs_by_sku=None,
)


def _tool_response(payload: dict | None) -> SimpleNamespace:
    """A response shaped like the SDK's Message, carrying one tool_use block
    (or none, to simulate a response with no tool call at all)."""
    if payload is None:
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="I decline.")])
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="propose_candidates", input=payload)]
    )


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
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reaction = self._reactions.pop(0)
        if isinstance(reaction, Exception):
            raise reaction
        return reaction


def _auth_error() -> anthropic.AuthenticationError:
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx2.Response(401, request=req)
    return anthropic.AuthenticationError("invalid x-api-key", response=resp, body=None)


@pytest.fixture
def store(seeded_conn):
    s = PosteriorStore(seeded_conn)
    s.ensure_initialized()
    return s


def _generate(gen: CandidateGenerator, store):
    return gen.generate(
        opportunity_type=OpportunityType.DORMANT_WINBACK, segment=Segment.DORMANT,
        rupees_at_risk=750.0, envelope=ENVELOPE, store=store, policy=DEFAULT_POLICY,
    )


# ==================================================================== off --


def test_off_mode_never_touches_an_injected_client(store):
    client = FakeClient()  # no reactions queued -- a call would raise IndexError
    gen = CandidateGenerator(client, mode="off")
    result = _generate(gen, store)
    assert result is not None and len(result.candidates) == 1
    assert client.calls == []


# ================================================================= record --


def test_record_mode_calls_once_parses_and_persists_to_cassette(store, tmp_path):
    client = FakeClient(_tool_response(_valid_payload()))
    cassette = Cassette(tmp_path)
    gen = CandidateGenerator(client, mode="record", cassette=cassette)

    result = _generate(gen, store)

    assert result is not None
    assert len(result.candidates) == 3
    assert len(client.calls) == 1

    key = cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE)
    assert cassette.has(key)
    assert cassette.get(key) == result


def test_record_mode_cache_hit_never_calls_the_client(store, tmp_path):
    cassette = Cassette(tmp_path)
    key = cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE)
    cassette.put(key, CandidateSet.model_validate(_valid_payload(1)))

    client = FakeClient()  # any call would raise IndexError
    gen = CandidateGenerator(client, mode="record", cassette=cassette)
    result = _generate(gen, store)

    assert result is not None
    assert client.calls == []


def test_record_mode_retries_once_on_malformed_then_succeeds(store, tmp_path):
    malformed = _tool_response({"candidates": "not a list"})
    valid = _tool_response(_valid_payload())
    client = FakeClient(malformed, valid)
    gen = CandidateGenerator(client, mode="record", cassette=Cassette(tmp_path))

    result = _generate(gen, store)

    assert result is not None
    assert len(client.calls) == 2
    # the retry call must echo the schema, per the failure-mode table
    assert "schema" in client.calls[1]["messages"][0]["content"].lower()


def test_record_mode_malformed_twice_returns_none(store, tmp_path):
    malformed = _tool_response({"candidates": []})  # violates min_length=1
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

    with pytest.raises(anthropic.AuthenticationError):
        _generate(gen, store)


def test_record_mode_no_credential_degrades_without_caching_the_miss(store, tmp_path, monkeypatch):
    # This dev environment has a real ANTHROPIC_API_KEY in .env (Phase 0), so
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
    key = cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE)
    assert not cassette.has(key)  # a template fallback must never be cached as a real recording


# ================================================================= replay --


def test_replay_mode_hit_returns_recorded_set_with_no_client_at_all(store, tmp_path, monkeypatch):
    monkeypatch.setattr("revenew.decide.generator._client", lambda: None)
    cassette = Cassette(tmp_path)
    key = cache_key(OpportunityType.DORMANT_WINBACK, Segment.DORMANT, 750.0, ENVELOPE)
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
    client = FakeClient(_tool_response(illegal_payload))
    gen = CandidateGenerator(client, mode="record", cassette=Cassette(tmp_path))

    seeded_conn.execute("INSERT OR IGNORE INTO customers VALUES ('cus_gen_test', ?)", ("2026-01-01",))
    seeded_conn.execute(
        "INSERT INTO opportunity_candidates VALUES ('opp_gen','run1','cus_gen_test',"
        "'dormant_winback','w1','dormant_winback',500,'h',?)",
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
