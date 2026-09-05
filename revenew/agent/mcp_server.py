"""Model Context Protocol (MCP) stdio server for Revenew Agent Channel.

Provides a standard JSON-RPC 2.0 interface for AI shopping agents (e.g. Claude
Desktop, Cursor, custom shopping bots) to:
1. `search_catalog`: Search merchant catalog by keyword/category
2. `get_product`: Retrieve product details by SKU
3. `request_offer`: Request standard merchant promotion
4. `negotiate`: Propose custom terms and receive reasoned acceptance/counter-offer
5. `create_checkout`: Generate a Razorpay Payment Link for agreed terms

Implemented using standard Python stdio and JSON-RPC 2.0 without heavy third-party
dependencies to keep the deployment lightweight and robust across environments.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from revenew.agent.negotiate import create_checkout, negotiate
from revenew.db import DEFAULT_DB_PATH, connect

TOOLS_DEFINITIONS = [
    {
        "name": "search_catalog",
        "description": "Search the merchant's live product catalog by keyword or category.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional product name search term"},
                "category": {"type": "string", "description": "Optional category filter"},
            },
        },
    },
    {
        "name": "get_product",
        "description": "Retrieve detailed product specifications, list price, and stock for a SKU.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "The product SKU identifier"},
            },
            "required": ["sku"],
        },
    },
    {
        "name": "request_offer",
        "description": "Request the merchant's standard promotional offer for a product.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "The product SKU identifier"},
                "customer_ref": {"type": "string", "description": "Optional external customer identifier"},
            },
            "required": ["sku"],
        },
    },
    {
        "name": "negotiate",
        "description": (
            "Propose custom commercial terms (e.g. requested discount percentage) for a product. "
            "Revenew evaluates your proposal against merchant policies and either accepts or returns "
            "a reasoned refusal and counter-offer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "The product SKU identifier"},
                "requested_discount_pct": {
                    "type": "number",
                    "description": "Requested discount fraction between 0.0 and 1.0 (e.g., 0.15 for 15% off)",
                },
                "customer_ref": {"type": "string", "description": "Optional external customer identifier"},
            },
            "required": ["sku", "requested_discount_pct"],
        },
    },
    {
        "name": "create_checkout",
        "description": "Create an active Razorpay payment link for agreed product terms.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "The product SKU identifier"},
                "decision_id": {"type": "string", "description": "The decision ID received from negotiate or request_offer"},
                "agreed_discount_pct": {"type": "number", "description": "The agreed discount fraction (e.g. 0.15)"},
                "customer_ref": {"type": "string", "description": "Optional external customer identifier"},
            },
            "required": ["sku"],
        },
    },
]


def search_catalog(conn: sqlite3.Connection, query: str | None = None, category: str | None = None) -> list[dict]:
    sql = "SELECT sku, name, category, price FROM products WHERE 1=1"
    params: list[Any] = []
    if query:
        sql += " AND (name LIKE ? OR sku LIKE ?)"
        params.extend([f"%{query}%", f"%{query}%"])
    if category:
        sql += " AND category LIKE ?"
        params.append(f"%{category}%")
    sql += " ORDER BY price DESC"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_product(conn: sqlite3.Connection, sku: str) -> dict:
    row = conn.execute(
        "SELECT sku, name, category, price FROM products WHERE sku = ?",
        (sku,),
    ).fetchone()
    if row is None:
        raise ValueError(f"SKU {sku!r} not found")
    return dict(row)


def handle_tool_call(conn: sqlite3.Connection, name: str, arguments: dict) -> Any:
    if name == "search_catalog":
        return search_catalog(conn, arguments.get("query"), arguments.get("category"))
    elif name == "get_product":
        return get_product(conn, arguments["sku"])
    elif name == "request_offer":
        return negotiate(
            conn,
            sku=arguments["sku"],
            requested_discount_pct=0.0,
            customer_ref=arguments.get("customer_ref"),
        )
    elif name == "negotiate":
        return negotiate(
            conn,
            sku=arguments["sku"],
            requested_discount_pct=float(arguments["requested_discount_pct"]),
            customer_ref=arguments.get("customer_ref"),
        )
    elif name == "create_checkout":
        return create_checkout(
            conn,
            sku=arguments["sku"],
            decision_id=arguments.get("decision_id"),
            agreed_discount_pct=float(arguments.get("agreed_discount_pct", 0.0)),
            customer_ref=arguments.get("customer_ref"),
        )
    else:
        raise ValueError(f"Unknown tool {name!r}")


def process_message(conn: sqlite3.Connection, message: dict) -> dict | None:
    """Process a single JSON-RPC 2.0 message."""
    method = message.get("method")
    msg_id = message.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": "revenew-agent-channel",
                    "version": "0.1.0",
                },
            },
        }
    elif method == "notifications/initialized":
        return None
    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {},
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": TOOLS_DEFINITIONS,
            },
        }
    elif method == "tools/call":
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            result_data = handle_tool_call(conn, tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result_data, indent=2),
                        }
                    ],
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32000,
                    "message": str(exc),
                },
            }
    else:
        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method {method!r} not found",
                },
            }
        return None


def run_stdio_server(db_path: str | Path | None = None) -> None:
    """Run the stdio MCP server loop."""
    if db_path is None:
        db_path = os.environ.get("REVENEW_DB_PATH", DEFAULT_DB_PATH)
    conn = connect(db_path)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = process_message(conn, message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
