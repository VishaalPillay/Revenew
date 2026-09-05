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

**What a verified delivery does beyond recording it, and the one thing left
UNVERIFIED.** `payment.captured`/`payment.failed` are mapped back to a
decision via `notes.decision_id` (`LiveAdapter.create_offer` sets it --
`revenew/execute/razorpay.py`) and fed to `record_outcome`
(`revenew/ledger/outcome.py`), which already updates the bandit's posteriors
-- no new learning code, this only supplies it a real trigger. This is the
one assumption in this module that is NOT yet confirmed against a real
capture the way the dedup key and signature scheme are: whether Razorpay
copies a Payment Link's `notes` onto the `payment` entity created when a
customer pays it is standard, documented Razorpay behavior, but unlike the
dedup/signature fixes above, nobody has captured a real delivery through
THIS project's own payment-link flow to confirm it end to end. If it turns
out not to hold, the failure mode is silent-safe by construction: `notes`
missing or not a dict (the real capture below shows it can be an empty
LIST, not a dict, when nothing set it) means `_decision_id_from_notes`
returns `None`, the event is still recorded, and no outcome is written --
never a crash, never a wrong outcome attributed to the wrong decision.
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
from revenew.ledger.outcome import record_outcome
from revenew.settings import RAZORPAY_WEBHOOK_SECRET, RAZORPAY_WEBHOOK_SECRET_PLACEHOLDER

router = APIRouter()

# The only two events this handler acts on beyond recording. Everything else
# Razorpay might send (payment.authorized, order.paid, refund.*, ...) is
# recorded into `events` like any other delivery and otherwise ignored --
# explicitly, not by omission, so a future reader can see the boundary was a
# choice.
_OUTCOME_EVENTS = {"payment.captured", "payment.failed"}


def _decision_id_from_notes(payload: dict) -> str | None:
    """`payload.payment.entity.notes.decision_id`, or `None` if any step of
    that path is missing or the wrong shape.

    `notes` is a Razorpay-controlled field: it is a dict when the merchant
    set one (LiveAdapter always does now), but a REAL captured delivery for
    a payment made with no notes set shows `"notes":[]` -- an empty LIST,
    not `{}`. `.get` on a list would raise `AttributeError`, so the isinstance
    check is load-bearing, not defensive boilerplate."""
    try:
        notes = payload["payload"]["payment"]["entity"]["notes"]
    except (KeyError, TypeError):
        return None
    if not isinstance(notes, dict):
        return None
    decision_id = notes.get("decision_id")
    return decision_id if isinstance(decision_id, str) and decision_id else None


def _payment_amount_rupees(payload: dict) -> float | None:
    """`payload.payment.entity.amount` is in paise, per Razorpay convention
    (see LiveAdapter's own `amount_paise` conversion) -- converted back to
    rupees here since that is the unit every other outcome in this system
    (`outcomes.net_revenue`, the fixture's `OutcomeOracle`) is recorded in."""
    try:
        amount_paise = payload["payload"]["payment"]["entity"]["amount"]
    except (KeyError, TypeError):
        return None
    if not isinstance(amount_paise, int | float):
        return None
    return amount_paise / 100.0


def _record_outcome_from_webhook(conn: sqlite3.Connection, *, event_type: str, payload: dict, now) -> None:
    """Best-effort: a payment this handler cannot attribute to a decision is
    not an error, it is simply outside what this handler can act on -- see
    the module docstring for exactly which failure this degrades from."""
    decision_id = _decision_id_from_notes(payload)
    if decision_id is None:
        return

    decision = conn.execute(
        "SELECT opportunity_id FROM decisions WHERE decision_id = ?", (decision_id,)
    ).fetchone()
    if decision is None:
        # A decision_id that does not exist in THIS database -- plausible if
        # the webhook secret/URL is shared across environments, or a replay
        # database was swapped out from under a running server. Recording
        # nothing is the safe choice; there is no decision here to attach an
        # outcome to.
        return

    converted = event_type == "payment.captured"
    net_revenue = (_payment_amount_rupees(payload) or 0.0) if converted else 0.0

    try:
        record_outcome(
            conn,
            opportunity_id=decision["opportunity_id"],
            decision_id=decision_id,
            converted=converted,
            net_revenue=net_revenue,
            censored=False,  # a captured or failed payment is definitive, not a timeout
            closed_at=iso(now),
        )
    except sqlite3.IntegrityError:
        # This opportunity's window was already closed -- by an earlier
        # delivery under a different event_id, or by the nightly resolver
        # racing this webhook. outcomes.opportunity_id is UNIQUE and the
        # table is append-only by trigger; a second attempt is a no-op, not
        # a failure worth surfacing as one.
        pass


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
    signature_verified = False
    if RAZORPAY_WEBHOOK_SECRET and RAZORPAY_WEBHOOK_SECRET != RAZORPAY_WEBHOOK_SECRET_PLACEHOLDER:
        signature = request.headers.get("X-Razorpay-Signature")
        if not _verify_signature(body, signature):
            return JSONResponse({"error": "invalid signature"}, status_code=400)
        signature_verified = True
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
            "INSERT INTO events (event_id, event_type, payload_json, received_at, signature_verified) "
            "VALUES (?, ?, ?, ?, ?)",
            (event_id, event_type, body.decode("utf-8", errors="replace"), iso(now), int(signature_verified)),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Duplicate delivery. Per the failure-mode table: ignore, 200 OK.
        return JSONResponse({"status": "duplicate_ignored"}, status_code=200)

    # The fast trigger's job used to end here, at "an event is durably
    # recorded" -- waking the detector for a single customer in response to
    # one webhook, rather than waiting for the nightly SlowTrigger, is still
    # real future work. But a payment outcome is not a new decision to make;
    # it is the answer to a decision already made, and recording that answer
    # (which is what actually feeds the bandit) does not need to wait for
    # any nightly rebuild -- see `_record_outcome_from_webhook`.
    #
    # GATED ON A VERIFIED SIGNATURE, and that gate is the whole point.
    # Recording an unverified delivery into `events` is harmless: nothing
    # downstream reads that table, so the placeholder-secret setup path can
    # accept-and-warn without consequence. Recording an OUTCOME is the
    # opposite -- `outcomes` is append-only by trigger (no UPDATE, no DELETE)
    # and `record_outcome` feeds `posteriors` directly, so a single forged
    # `payment.captured` naming a real decision_id would permanently teach
    # the bandit a conversion that never happened, with no way to retract it.
    # An unauthenticated caller must never be able to reach that. This
    # asymmetry is deliberate: keep accepting unsigned deliveries during
    # setup, but never let one change what the system believes.
    if event_type in _OUTCOME_EVENTS:
        if signature_verified:
            _record_outcome_from_webhook(conn, event_type=event_type, payload=payload, now=now)
        else:
            print(
                f"WARNING: {event_type} delivery {event_id} recorded but NOT applied as an "
                "outcome -- its signature was not verified (RAZORPAY_WEBHOOK_SECRET unset or "
                "still the .env.example placeholder). Configure the secret to close the "
                "learning loop; an unverified payment must not move the bandit."
            )

    return JSONResponse({"status": "recorded", "event_id": event_id}, status_code=200)
