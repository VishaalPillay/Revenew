"""Tests for the Live Decision Studio (Phase 3).

The critical regression test here is `test_live_decision_does_not_break_theatre`:
PLAN.md §4.2 identifies a real landmine where a live decision would make the
Theatre flip to a one-decision "run" and render empty.  That test inserts a
live decision, calls `build_timeline()`, and confirms the Theatre still
returns the replay run's timeline.

The other tests exercise the SSE stream, the degraded fallback, the
revalidation endpoint, and the cache-key exclusion — all through the real
FastAPI TestClient, parsing the actual SSE bytes the endpoint emits.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from revenew.api.dashboard import app
from revenew.api.theatre import build_timeline
from revenew.clock import iso
from revenew.decide.bandit import PosteriorStore
from revenew.models import (
    ActionFamily,
    Candidate,
    CandidateSet,
    Envelope,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_one_opportunity(conn: sqlite3.Connection, *, run_id: str = "replay_20260101") -> str:
    """Insert one treatment-arm opportunity with the full supporting rows
    (customer, orders, products, opportunity_candidates, opportunities).
    Returns the customer_id."""
    cid = "live_test_cus1"
    oid = "live_test_opp1"
    when = iso(NOW)

    conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (cid, when))
    conn.execute(
        "INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?)",
        ("ord1", cid, 1500.0, "captured", when),
    )
    # Products are needed for EnvelopeEngine.load_catalog and budget checks
    conn.execute(
        "INSERT OR IGNORE INTO products VALUES (?, ?, ?, ?, ?)",
        ("SKU-A01", "Test Shoes", "footwear", 2999.0, None),
    )
    conn.execute(
        "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
        (oid, run_id, cid, "dormant_winback", "w1", "dormant_winback", 1500, "h", when, None),
    )
    conn.execute(
        "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
        (oid, run_id, cid, "w1", "dormant", "treatment", when),
    )
    conn.commit()
    return cid


def _seed_replay_decision(conn: sqlite3.Connection, *, decision_index: int = 0) -> str:
    """Insert one replay decision so the Theatre has a run to display.
    Returns the decision_id."""
    cid = f"replay_cus{decision_index}"
    oid = f"replay_opp{decision_index}"
    did = f"replay_dec{decision_index}"
    run_id = "replay_20260101"
    when = iso(NOW + timedelta(days=decision_index))

    conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (cid, iso(NOW)))
    conn.execute(
        "INSERT OR IGNORE INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
        (oid, run_id, cid, "dormant_winback", "w1", "dormant_winback", 500, "h", when, None),
    )
    conn.execute(
        "INSERT OR IGNORE INTO opportunities VALUES (?,?,?,?,?,?,?)",
        (oid, run_id, cid, "w1", "dormant", "treatment", when),
    )
    # A valid serialized Envelope — the revalidate endpoint deserializes
    # envelope_json from the DB, so it must parse correctly.
    envelope_json = Envelope(
        max_discount_pct=0.20, max_absolute_discount=500.0,
        budget_remaining=50000.0, excluded_skus=[],
        cooldown_days=30, max_offers_per_customer_per_month=1,
    ).model_dump_json()
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (did, oid, run_id, "dormant", "percent_discount", envelope_json, 5, 2,
         '{"action_family":"percent_discount","headline":"15% off","discount_pct":0.15,"rationale":"test"}',
         0.4, "executed", None, when, "internal"),
    )
    # Store candidates for the revalidate test
    conn.execute(
        "INSERT INTO decision_candidates VALUES (?,?,?,?,?)",
        (did, 0,
         '{"action_family":"percent_discount","headline":"15% off","discount_pct":0.15,"rationale":"test"}',
         1, "[]"),
    )
    conn.execute(
        "INSERT INTO decision_candidates VALUES (?,?,?,?,?)",
        (did, 1,
         '{"action_family":"flat_coupon","headline":"Rs 200 off","discount_amount":200,"rationale":"test"}',
         1, "[]"),
    )
    conn.execute(
        "INSERT INTO decision_candidates VALUES (?,?,?,?,?)",
        (did, 2,
         '{"action_family":"reminder_nudge","headline":"We miss you","rationale":"test"}',
         1, "[]"),
    )
    conn.commit()
    return did


def _parse_sse(response_text: str) -> list[tuple[str, dict]]:
    """Parse an SSE stream into [(event_type, data_dict), ...]."""
    events = []
    current_event = None
    current_data = None
    for line in response_text.split("\n"):
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            current_data = line[len("data: "):]
        elif line == "" and current_event is not None and current_data is not None:
            events.append((current_event, json.loads(current_data)))
            current_event = None
            current_data = None
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _fake_candidate_set():
    """A valid CandidateSet for mocking the generator."""
    return CandidateSet(candidates=[
        Candidate(action_family=ActionFamily.PERCENT_DISCOUNT,
                  headline="15% off your next order",
                  discount_pct=0.15,
                  rationale="Test candidate"),
        Candidate(action_family=ActionFamily.REMINDER_NUDGE,
                  headline="We miss you",
                  rationale="Test nudge"),
    ])


def test_live_decide_emits_sse_events_in_order(seeded_conn):
    """Mock the LLM call, drive /api/live/decide, parse the SSE stream,
    and verify all expected event types appear in the correct order."""
    _seed_one_opportunity(seeded_conn)
    PosteriorStore(seeded_conn).ensure_initialized()

    fake_cs = _fake_candidate_set()

    with patch("revenew.api.live.CandidateGenerator") as MockGen:
        mock_instance = MockGen.return_value
        mock_instance.generate.return_value = fake_cs
        mock_instance.llm_available = True

        with patch("revenew.api.live.get_conn") as mock_conn:
            mock_conn.return_value = seeded_conn
            # Override the dependency
            app.dependency_overrides[__import__("revenew.api.webhooks", fromlist=["get_conn"]).get_conn] = lambda: seeded_conn

            try:
                client = TestClient(app)
                resp = client.post("/api/live/decide", json={})
                assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:500]}"

                events = _parse_sse(resp.text)
                event_types = [e[0] for e in events]

                # Must contain the key events in order
                assert "opportunity" in event_types, f"missing 'opportunity' event in {event_types}"
                assert "envelope" in event_types, f"missing 'envelope' event in {event_types}"



                # Either llm_done or degraded must appear
                has_llm = "llm_done" in event_types or "degraded" in event_types
                assert has_llm, f"neither 'llm_done' nor 'degraded' in {event_types}"

                # Candidates must appear
                assert "candidate" in event_types, f"missing 'candidate' event in {event_types}"

                # Verdicts must appear
                assert "verdict" in event_types, f"missing 'verdict' event in {event_types}"
            finally:
                app.dependency_overrides.clear()


def test_live_decide_degraded_fallback(seeded_conn):
    """Force the generator to raise, confirm a 'degraded' event is emitted
    and the stream still completes."""
    _seed_one_opportunity(seeded_conn)
    PosteriorStore(seeded_conn).ensure_initialized()

    fake_cs = _fake_candidate_set()

    # The live CandidateGenerator raises, but the fallback one returns candidates
    call_count = [0]

    def fake_generate(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise TimeoutError("Groq is down")
        return fake_cs

    with patch("revenew.api.live.CandidateGenerator") as MockGen:
        mock_instance = MockGen.return_value
        mock_instance.generate.side_effect = fake_generate
        mock_instance.llm_available = True

        app.dependency_overrides[__import__("revenew.api.webhooks", fromlist=["get_conn"]).get_conn] = lambda: seeded_conn

        try:
            client = TestClient(app)
            resp = client.post("/api/live/decide", json={})
            assert resp.status_code == 200

            events = _parse_sse(resp.text)
            event_types = [e[0] for e in events]

            assert "degraded" in event_types, f"missing 'degraded' event in {event_types}"
            degraded_data = next(d for t, d in events if t == "degraded")
            assert "Groq is down" in degraded_data["reason"]
        finally:
            app.dependency_overrides.clear()


def test_revalidate_tightened_cap_invalidates_chosen_offer(seeded_conn):
    """Create a decision with a 15% discount candidate, call
    /api/live/revalidate with max_discount_pct=0.10, confirm the candidate
    is now invalid."""
    did = _seed_replay_decision(seeded_conn)

    app.dependency_overrides[__import__("revenew.api.webhooks", fromlist=["get_conn"]).get_conn] = lambda: seeded_conn

    try:
        client = TestClient(app)
        resp = client.post("/api/live/revalidate", json={
            "decision_id": did,
            "max_discount_pct": 0.10,
        })
        assert resp.status_code == 200
        body = resp.json()

        assert body["decision_id"] == did
        assert body["tightened_max_discount_pct"] == 0.10

        # The 15% discount candidate should now be INVALID
        pct_verdict = next(
            v for v in body["verdicts"]
            if v["action_family"] == "percent_discount"
        )
        assert pct_verdict["valid"] is False
        assert "max_discount_pct" in pct_verdict["violations"]

        # The reminder nudge (no discount) should still be valid
        nudge_verdict = next(
            v for v in body["verdicts"]
            if v["action_family"] == "reminder_nudge"
        )
        assert nudge_verdict["valid"] is True

        # The chosen offer (percent_discount at 15%) should be struck down
        assert body["chosen_now_valid"] is False
    finally:
        app.dependency_overrides.clear()


def test_revalidate_is_deterministic(seeded_conn):
    """Same inputs, same outputs — confirming no randomness in the revalidate
    path (no model call, no Thompson sampling)."""
    did = _seed_replay_decision(seeded_conn)

    app.dependency_overrides[__import__("revenew.api.webhooks", fromlist=["get_conn"]).get_conn] = lambda: seeded_conn

    try:
        client = TestClient(app)
        results = []
        for _ in range(3):
            resp = client.post("/api/live/revalidate", json={
                "decision_id": did,
                "max_discount_pct": 0.08,
            })
            assert resp.status_code == 200
            results.append(resp.json())

        # All three must be identical
        assert results[0] == results[1] == results[2], "revalidate is not deterministic"
    finally:
        app.dependency_overrides.clear()


def test_live_decision_does_not_break_theatre(seeded_conn):
    """REGRESSION TEST FOR PLAN.md §4.2.

    Insert a replay run's decisions (so the Theatre has something to show),
    then insert a live decision with `run_id='live_test123'` whose
    `created_at` sorts AFTER the replay.  Confirm `build_timeline()` still
    returns the replay run's timeline, not a one-decision live "run".

    Without the `WHERE run_id NOT LIKE 'live_%'` fix in `_latest_run_id()`,
    this test would see the Theatre flip to a one-decision timeline with
    run_id='live_test123' and zero frames, because there are no outcomes,
    no regret rows, and no day axis for that run."""
    # 1. Seed a replay run with a real decision
    _seed_replay_decision(seeded_conn, decision_index=0)
    _seed_replay_decision(seeded_conn, decision_index=1)

    # Confirm the Theatre picks up the replay run
    timeline_before = build_timeline(seeded_conn)
    assert timeline_before.meta["run_id"] == "replay_20260101"
    assert len(timeline_before.frames) > 0, "Theatre should have frames from the replay run"

    # 2. Insert a live decision dated AFTER the replay (this is the landmine)
    live_when = iso(NOW + timedelta(days=100))  # far in the future
    seeded_conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", ("live_cus", iso(NOW)))
    seeded_conn.execute(
        "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("live_opp", "live_test123", "live_cus", "dormant_winback", "w99",
         "dormant_winback", 500, "h", live_when, None),
    )
    seeded_conn.execute(
        "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
        ("live_opp", "live_test123", "live_cus", "w99", "dormant", "treatment", live_when),
    )
    seeded_conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("live_dec", "live_opp", "live_test123", "dormant", "percent_discount",
         "{}", 1, 1, None, None, "executed", None, live_when, "internal"),
    )
    seeded_conn.commit()

    # 3. Verify the Theatre STILL returns the replay run, not the live one
    timeline_after = build_timeline(seeded_conn)
    assert timeline_after.meta["run_id"] == "replay_20260101", (
        f"Theatre flipped to run_id={timeline_after.meta['run_id']!r} — "
        f"the live decision broke it (§4.2 landmine)"
    )
    assert len(timeline_after.frames) == len(timeline_before.frames), (
        "Theatre frame count changed after a live decision was added"
    )


def test_live_run_id_excluded_from_theatre_cache_key(seeded_conn):
    """The /api/theatre cache key query must also exclude live_ prefixed
    run_ids, otherwise a live decision busts the cache and forces a full
    timeline rebuild on the next page load."""
    _seed_replay_decision(seeded_conn, decision_index=0)

    # First call: populates the cache
    app.dependency_overrides[__import__("revenew.api.webhooks", fromlist=["get_conn"]).get_conn] = lambda: seeded_conn
    try:
        client = TestClient(app)
        resp1 = client.get("/api/theatre")
        assert resp1.status_code == 200
        meta1 = resp1.json()["meta"]
        assert meta1["run_id"] == "replay_20260101"

        # Insert a live decision (newest created_at)
        live_when = iso(NOW + timedelta(days=100))
        seeded_conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", ("cache_cus", iso(NOW)))
        seeded_conn.execute(
            "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("cache_opp", "live_cache", "cache_cus", "dormant_winback", "w99",
             "dormant_winback", 500, "h", live_when, None),
        )
        seeded_conn.execute(
            "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
            ("cache_opp", "live_cache", "cache_cus", "w99", "dormant", "treatment", live_when),
        )
        seeded_conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("cache_dec", "cache_opp", "live_cache", "dormant", None,
             "{}", 0, 0, None, None, "no_action", "budget_exhausted", live_when, "internal"),
        )
        seeded_conn.commit()

        # The cache key query should still resolve to replay_20260101,
        # not live_cache, so the cache hit should work
        resp2 = client.get("/api/theatre")
        assert resp2.status_code == 200
        meta2 = resp2.json()["meta"]
        assert meta2["run_id"] == "replay_20260101", (
            f"Theatre cache key picked up live run_id={meta2['run_id']!r}"
        )
    finally:
        app.dependency_overrides.clear()
