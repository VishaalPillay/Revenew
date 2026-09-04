"""FastTrigger: the live webhook receiver.

Dedupes on event id before anything else runs -- a redelivered webhook (a
routine occurrence with any provider's at-least-once delivery guarantee) must
be a 200 OK no-op, not a second detection cycle. `events.event_id` is UNIQUE
in db/schema.sql; that constraint is what actually enforces this, the code
below just turns the resulting IntegrityError into the right HTTP response
instead of a 500.

This module does not verify a Razorpay signature -- SYSTEM_DESIGN.md's
assumption ledger (section 11) flags the exact webhook payload shape as
UNKNOWN pending a captured real payload, and signature verification needs the
real header name and signing scheme to get right rather than guess at. Wiring
it in is one function once that payload is captured; shipping a guessed
verifier that always passes would be worse than no verifier with an honest
TODO.
"""

from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from revenew.clock import WallClock, iso
from revenew.db import connect

router = APIRouter()


def get_conn() -> sqlite3.Connection:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


@router.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
    body = await request.body()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    event_id = payload.get("id") or payload.get("event_id")
    if not event_id:
        return JSONResponse({"error": "missing event id"}, status_code=400)

    event_type = payload.get("event", "unknown")
    now = WallClock().now()

    try:
        conn.execute(
            "INSERT INTO events (event_id, event_type, payload_json, received_at) VALUES (?, ?, ?, ?)",
            (event_id, event_type, body.decode("utf-8", errors="replace"), iso(now)),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Duplicate delivery. Per the failure-mode table: ignore, 200 OK.
        return JSONResponse({"status": "duplicate_ignored"}, status_code=200)

    # The fast trigger's job ends at "an event is durably recorded". Waking
    # the detector for a single customer in response to one webhook, rather
    # than waiting for the nightly SlowTrigger, is real future work -- this
    # process currently relies on the nightly rebuild to pick up what this
    # event implies. Recording it now means nothing is lost in the meantime.
    return JSONResponse({"status": "recorded", "event_id": event_id}, status_code=200)
