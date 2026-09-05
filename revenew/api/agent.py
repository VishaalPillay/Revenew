"""REST API endpoints for the Revenew Agent Channel.

Allows external AI shopping agents and client applications to interact with the
agent commerce layer over HTTP, complementing the stdio MCP server.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from revenew.agent.mcp_server import get_product, search_catalog
from revenew.agent.negotiate import create_checkout, negotiate
from revenew.api.webhooks import get_conn

router = APIRouter(prefix="/api/agent", tags=["agent"])


class NegotiateRequest(BaseModel):
    sku: str
    requested_discount_pct: float = Field(ge=0.0, le=1.0, description="Requested discount fraction (e.g. 0.15)")
    customer_ref: str | None = Field(default=None, description="Optional customer or agent reference")


class CheckoutRequest(BaseModel):
    sku: str
    decision_id: str | None = Field(default=None, description="Decision ID from negotiate")
    agreed_discount_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    customer_ref: str | None = Field(default=None)


@router.get("/catalog")
def api_search_catalog(
    query: str | None = Query(default=None, description="Product search term"),
    category: str | None = Query(default=None, description="Category filter"),
    conn: sqlite3.Connection = Depends(get_conn),
):
    """List or search products in the catalog."""
    return search_catalog(conn, query=query, category=category)


@router.get("/products/{sku}")
def api_get_product(
    sku: str,
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Retrieve details for a specific product SKU."""
    try:
        return get_product(conn, sku=sku)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/negotiate")
def api_negotiate(
    request: NegotiateRequest,
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Negotiate terms for a product against merchant policy."""
    try:
        return negotiate(
            conn,
            sku=request.sku,
            requested_discount_pct=request.requested_discount_pct,
            customer_ref=request.customer_ref,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/checkout")
def api_checkout(
    request: CheckoutRequest,
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Generate an active Razorpay payment link for agreed terms."""
    try:
        return create_checkout(
            conn,
            sku=request.sku,
            decision_id=request.decision_id,
            agreed_discount_pct=request.agreed_discount_pct,
            customer_ref=request.customer_ref,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/metrics")
def api_agent_metrics(
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Summary metrics for the Agent Channel demand source (for Act 5 panel)."""
    # Total agent decisions
    total_decisions = conn.execute(
        "SELECT COUNT(*) AS count FROM decisions WHERE channel = 'agent'"
    ).fetchone()["count"]

    # Decisions executed
    executed_decisions = conn.execute(
        "SELECT COUNT(*) AS count FROM decisions WHERE channel = 'agent' AND status = 'executed'"
    ).fetchone()["count"]

    # Total agent GMV
    gmv_row = conn.execute(
        """
        SELECT COALESCE(SUM(-amount), 0.0) AS gmv
        FROM budget_ledger bl
        JOIN decisions d ON d.decision_id = bl.decision_id
        WHERE d.channel = 'agent' AND d.status = 'executed'
        """
    ).fetchone()
    agent_gmv = float(gmv_row["gmv"]) if gmv_row else 0.0

    return {
        "channel": "agent",
        "total_negotiations": total_decisions,
        "completed_transactions": executed_decisions,
        "agent_gmv": agent_gmv,
    }
