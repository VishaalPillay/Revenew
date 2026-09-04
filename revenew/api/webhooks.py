"""FastTrigger: the live webhook receiver.

Dedupes on event id before anything else runs -- a redelivered webhook (a
routine occurrence with any provider's at-least-once delivery guarantee) must
be a 200 OK no-op, not a second detection cycle. `events.event_id` is UNIQUE
in db/schema.sql; that constraint is what actually enforces this, the code
below just turns the resulting IntegrityError into the right HTTP response
instead of a 500.

**Both the dedup key and the signature scheme are now verified against a
real captured delivery (2026-09-04), not guessed.** Two things this
correction fixed, found only by capturing a genuine webhook through a tunnel
and reading exactly what came back:

1. There is no `id` or `event_id` field anywhere in Razorpay's webhook BODY
   -- the original code's `payload.get("id") or payload.get("event_id")`
   would return `None` for every real delivery Razorpay has ever sent,
   rejecting all of them with "missing event id" before the dedup logic
   even ran. The real per-event identifier is the `X-Razorpay-Event-Id`
   HTTP header. This was a live bug sitting behind the UNKNOWN assumption
   ledger row the whole time, invisible until a real payload was captured.
2. The signature scheme -- `X-Razorpay-Signature`, an HMAC-SHA256 hex digest
   of the exact raw request body using the webhook's configured secret --
   is now confirmed byte-for-byte: computing
   `hmac.new(secret, raw_body, hashlib.sha256).hexdigest()` against a real
   captured `(body, signature)` pair reproduced the signature exactly. See
   SYSTEM_DESIGN.md section 11 and ENGINEERING_LOG.md for the full story.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from revenew.clock import WallClock, iso
from revenew.db import connect
from revenew.settings import RAZORPAY_WEBHOOK_SECRET, RAZORPAY_WEBHOOK_SECRET_PLACEHOLDER

router = APIRouter()


def _verify_signature(raw_body: bytes, signature: str | None) -> bool:
    """True if `signature` is a valid HMAC-SHA256 hex digest of `raw_body`
    under `RAZORPAY_WEBHOOK_SECRET`. Uses `hmac.compare_digest` -- a plain
    `==` on two hex strings is a timing side-channel an attacker forging
    signatures could exploit to recover the correct one byte by byte."""
    if not signature:
        return False
    expected = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def get_conn() -> sqlite3.Connection:
    # REVENEW_DB_PATH lets `revenew serve --db PATH` point every route at a
    # non-default database without threading a CLI flag through FastAPI's
    # dependency injection -- unset in normal operation, in which case
    # connect() falls back to its own default exactly as before.
    db_path = os.environ.get("REVENEW_DB_PATH")
    conn = connect(db_path) if db_path else connect()
    try:
        yield conn
    finally:
        conn.close()


@router.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
    body = await request.body()

    # Signature check runs on the RAW bytes, before json.loads -- re-serializing
    # the parsed dict for verification would silently change key order/
    # whitespace and break every signature, since Razorpay signs exactly what
    # it sent over the wire, not a semantically-equivalent re-encoding.
    if RAZORPAY_WEBHOOK_SECRET and RAZORPAY_WEBHOOK_SECRET != RAZORPAY_WEBHOOK_SECRET_PLACEHOLDER:
        signature = request.headers.get("X-Razorpay-Signature")
        if not _verify_signature(body, signature):
            return JSONResponse({"error": "invalid signature"}, status_code=400)
    else:
        # Deliberately not silent: a placeholder secret means verification is
        # OFF, and every unsigned request is being accepted -- that must be
        # visible in the logs, not a fact only discoverable by reading this
        # module's source. See the plan this was built against: distinguish
        # "not configured yet" from "configured and enforced" explicitly,
        # rather than either hard-failing every webhook before setup is done
        # or silently accepting forever with no signal either way.
        print(
            "WARNING: RAZORPAY_WEBHOOK_SECRET is not set (or still the .env.example "
            "placeholder) -- webhook signature verification is DISABLED. Every "
            "request to /webhooks/razorpay is being accepted unverified."
        )

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    # The real per-delivery identifier, confirmed against a captured webhook:
    # Razorpay's JSON body carries no id/event_id field at all -- the
    # per-event id lives in the X-Razorpay-Event-Id header. Every prior
    # version of this handler looked in the body and would have rejected
    # every real delivery Razorpay has ever sent with "missing event id".
    event_id = request.headers.get("X-Razorpay-Event-Id")
    if not event_id:
        return JSONResponse({"error": "missing X-Razorpay-Event-Id header"}, status_code=400)

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
