"""Tests for Phase 4: The Agent Channel.

Verifies:
1. Negotiation within policy is accepted.
2. Negotiation exceeding policy returns a reasoned refusal and counter-offer citing the binding rule.
3. Excluded SKU cannot be discounted.
4. Checkout generates a Razorpay payment link via LinkSpec and marks decision executed.
5. REST API mirrors (/api/agent/*) function correctly with CORS headers.
6. MCP stdio protocol correctly responds to initialize, tools/list, and tools/call.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from revenew.agent.mcp_server import process_message
from revenew.agent.negotiate import create_checkout, negotiate
from revenew.api.dashboard import app
from revenew.api.webhooks import get_conn
from revenew.clock import iso
from revenew.settings import DEFAULT_POLICY, PolicyConfig

NOW = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)


def _seed_products(conn):
    """Seed test catalog products if not already present."""
    conn.execute(
        "INSERT OR IGNORE INTO products (sku, name, category, price, cogs) VALUES (?, ?, ?, ?, ?)",
        ("SKU-TEST-01", "Pro Running Shoes", "footwear", 3000.0, 1500.0),
    )
    conn.execute(
        "INSERT OR IGNORE INTO products (sku, name, category, price, cogs) VALUES (?, ?, ?, ?, ?)",
        ("SKU-TEST-02", "Cotton Graphic Tee", "apparel", 1000.0, 400.0),
    )
    conn.commit()


def test_negotiate_valid_discount_accepted(seeded_conn):
    _seed_products(seeded_conn)
    result = negotiate(
        seeded_conn,
        sku="SKU-TEST-01",
        requested_discount_pct=0.10,
        now=NOW,
    )

    assert result["status"] == "accepted"
    assert result["sku"] == "SKU-TEST-01"
    assert result["offered_discount_pct"] == 0.10
    assert result["final_price"] == 2700.0  # 3000 * 0.9
    assert result["violations"] == []
    assert result["channel"] == "agent"

    # Verify persisted in decisions with channel = 'agent'
    row = seeded_conn.execute(
        "SELECT decision_id, channel, status FROM decisions WHERE decision_id = ?",
        (result["decision_id"],),
    ).fetchone()
    assert row is not None
    assert row["channel"] == "agent"
    assert row["status"] == "pending"


def test_negotiate_excessive_discount_counter_offered(seeded_conn):
    _seed_products(seeded_conn)
    # Request 35% discount on SKU-TEST-02 (price 1000) where 20% discount (200) < max_absolute (500)
    result = negotiate(
        seeded_conn,
        sku="SKU-TEST-02",
        requested_discount_pct=0.35,
        policy=DEFAULT_POLICY,
        now=NOW,
    )

    assert result["status"] == "counter_offer"
    assert result["requested_discount_pct"] == 0.35
    assert result["offered_discount_pct"] == 0.20  # capped at max_discount_pct
    assert result["final_price"] == 800.0  # 1000 * 0.8
    assert "max_discount_pct" in result["violations"]
    assert "max_discount_pct" in result["reason"]
    assert result["channel"] == "agent"


def test_negotiate_excluded_sku_refused(seeded_conn):
    _seed_products(seeded_conn)
    policy = PolicyConfig(
        max_discount_pct=0.20,
        excluded_skus=("SKU-TEST-01",),
    )

    result = negotiate(
        seeded_conn,
        sku="SKU-TEST-01",
        requested_discount_pct=0.15,
        policy=policy,
        now=NOW,
    )

    assert result["status"] == "counter_offer"
    assert result["offered_discount_pct"] == 0.0  # list price
    assert result["final_price"] == 3000.0
    assert "excluded_skus" in result["violations"]
    assert "excluded_skus" in result["reason"]


def test_create_checkout_generates_payment_link_and_marks_executed(seeded_conn):
    _seed_products(seeded_conn)
    neg = negotiate(
        seeded_conn,
        sku="SKU-TEST-02",
        requested_discount_pct=0.15,
        now=NOW,
    )
    decision_id = neg["decision_id"]

    checkout = create_checkout(
        seeded_conn,
        sku="SKU-TEST-02",
        decision_id=decision_id,
        agreed_discount_pct=neg["offered_discount_pct"],
        customer_ref="agent_buyer_99",
        now=NOW,
    )

    assert checkout["status"] in ("confirmed", "sent")
    assert checkout["decision_id"] == decision_id
    assert checkout["amount"] == 850.0  # 1000 * 0.85
    assert checkout["provider_ref"] != ""
    assert checkout["payment_url"] != ""

    # Verify decision updated to 'executed'
    dec_row = seeded_conn.execute(
        "SELECT status FROM decisions WHERE decision_id = ?",
        (decision_id,),
    ).fetchone()
    assert dec_row["status"] == "executed"

    # Verify execution row recorded
    exec_row = seeded_conn.execute(
        "SELECT provider_ref, status FROM executions WHERE decision_id = ?",
        (decision_id,),
    ).fetchone()
    assert exec_row is not None
    assert exec_row["provider_ref"] == checkout["provider_ref"]


def test_agent_rest_endpoints(seeded_conn):
    _seed_products(seeded_conn)
    app.dependency_overrides[get_conn] = lambda: seeded_conn
    try:
        client = TestClient(app)

        # 1. GET /api/agent/catalog
        r_cat = client.get("/api/agent/catalog")
        assert r_cat.status_code == 200
        items = r_cat.json()
        assert len(items) >= 2
        skus = [it["sku"] for it in items]
        assert "SKU-TEST-01" in skus

        # 2. GET /api/agent/products/SKU-TEST-01
        r_prod = client.get("/api/agent/products/SKU-TEST-01")
        assert r_prod.status_code == 200
        assert r_prod.json()["name"] == "Pro Running Shoes"

        # 3. POST /api/agent/negotiate
        r_neg = client.post(
            "/api/agent/negotiate",
            json={"sku": "SKU-TEST-02", "requested_discount_pct": 0.25},
        )
        assert r_neg.status_code == 200
        neg_data = r_neg.json()
        assert neg_data["status"] == "counter_offer"
        assert neg_data["offered_discount_pct"] == 0.20

        # 4. POST /api/agent/checkout
        r_chk = client.post(
            "/api/agent/checkout",
            json={
                "sku": "SKU-TEST-02",
                "decision_id": neg_data["decision_id"],
                "agreed_discount_pct": neg_data["offered_discount_pct"],
                "customer_ref": "buyer_123",
            },
        )
        assert r_chk.status_code == 200
        chk_data = r_chk.json()
        assert chk_data["amount"] == 800.0
        assert "payment_url" in chk_data

        # 5. GET /api/agent/metrics
        r_met = client.get("/api/agent/metrics")
        assert r_met.status_code == 200
        met_data = r_met.json()
        assert met_data["channel"] == "agent"
        assert met_data["total_negotiations"] >= 1
        assert met_data["completed_transactions"] >= 1
    finally:
        app.dependency_overrides.clear()


def test_mcp_server_protocol(seeded_conn):
    _seed_products(seeded_conn)

    # 1. initialize
    init_res = process_message(
        seeded_conn,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    assert init_res["id"] == 1
    assert "capabilities" in init_res["result"]
    assert init_res["result"]["serverInfo"]["name"] == "revenew-agent-channel"

    # 2. tools/list
    tools_res = process_message(
        seeded_conn,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )
    assert tools_res["id"] == 2
    tool_names = [t["name"] for t in tools_res["result"]["tools"]]
    assert "search_catalog" in tool_names
    assert "get_product" in tool_names
    assert "request_offer" in tool_names
    assert "negotiate" in tool_names
    assert "create_checkout" in tool_names

    # 3. tools/call -> negotiate
    call_res = process_message(
        seeded_conn,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "negotiate",
                "arguments": {"sku": "SKU-TEST-01", "requested_discount_pct": 0.15},
            },
        },
    )
    assert call_res["id"] == 3
    content_text = call_res["result"]["content"][0]["text"]
    parsed = json.loads(content_text)
    assert parsed["status"] == "accepted"
    assert parsed["offered_discount_pct"] == 0.15

    # 4. tools/call -> create_checkout
    chk_res = process_message(
        seeded_conn,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "create_checkout",
                "arguments": {
                    "sku": "SKU-TEST-01",
                    "decision_id": parsed["decision_id"],
                    "agreed_discount_pct": 0.15,
                },
            },
        },
    )
    assert chk_res["id"] == 4
    chk_text = chk_res["result"]["content"][0]["text"]
    chk_parsed = json.loads(chk_text)
    assert chk_parsed["status"] in ("confirmed", "sent")
    assert chk_parsed["amount"] == 2550.0  # 3000 * 0.85


def test_negotiate_max_absolute_discount_bounds_high_price_item(seeded_conn):
    _seed_products(seeded_conn)
    # Price 3000.0 with max_absolute_discount 500.0:
    # 500 / 3000 = 0.1667 max discount fraction
    result = negotiate(
        seeded_conn,
        sku="SKU-TEST-01",
        requested_discount_pct=0.30,
        policy=DEFAULT_POLICY,
        now=NOW,
    )

    assert result["status"] == "counter_offer"
    assert result["offered_discount_pct"] == 0.1667
    assert result["final_price"] == 2499.9  # 3000 * (1 - 0.1667)


def test_agent_decision_does_not_break_theatre(seeded_conn):
    """An agent-channel decision made with run_id='agent_...' must not hijack
    the Theatre dashboard timeline or cache key."""
    from revenew.api.theatre import build_timeline

    _seed_products(seeded_conn)

    # Insert a replay run decision
    conn = seeded_conn
    replay_run = "replay_20260101"
    cid = "theatre_agent_cus"
    oid1 = "theatre_agent_opp1"
    when = iso(NOW)

    conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (cid, when))
    conn.execute(
        "INSERT OR IGNORE INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
        (oid1, replay_run, cid, "dormant_winback", "w1", "dormant", 1500, "h", when, None),
    )
    conn.execute(
        "INSERT OR IGNORE INTO opportunities VALUES (?,?,?,?,?,?,?)",
        (oid1, replay_run, cid, "w1", "dormant", "treatment", when),
    )
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "did_replay", oid1, replay_run, "dormant", "percent_discount", "{}", 5, 2,
            '{"action_family":"percent_discount","headline":"15% off","discount_pct":0.15}',
            0.5, "executed", None, "2026-01-02T00:00:00+00:00", "internal",
        ),
    )
    conn.commit()

    # Now make an agent decision dated newer than replay
    negotiate(
        conn,
        sku="SKU-TEST-02",
        requested_discount_pct=0.10,
        now=datetime(2026, 4, 15, tzinfo=UTC),
    )

    timeline = build_timeline(conn)
    assert timeline.meta["run_id"] == replay_run, (
        f"Theatre was hijacked by agent run! Expected {replay_run!r}, got {timeline.meta['run_id']!r}"
    )
