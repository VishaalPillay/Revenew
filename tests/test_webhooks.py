"""FastTrigger / webhook receiver: dedup on X-Razorpay-Event-Id, HMAC-SHA256
signature verification, and the placeholder-secret degrade path.

`REAL_CAPTURED_BODY` and `REAL_CAPTURED_SIGNATURE` below are byte-for-byte
what Razorpay actually sent to a real webhook, captured via a real test-mode
payment through an ngrok tunnel on 2026-09-04 -- not a hand-built fixture.
That capture is what proved two things the code previously guessed wrong:
there is no `id`/`event_id` field anywhere in the JSON body (the real
identifier lives in the `X-Razorpay-Event-Id` header), and the signature
scheme is exactly `hmac.new(secret, raw_body, hashlib.sha256).hexdigest()`.
`test_the_real_captured_payload_is_accepted_end_to_end` replays that exact
delivery through the real endpoint and asserts it is finally accepted --
every version of this handler before the fix would have rejected it.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from revenew.api.webhooks import get_conn
from revenew.api.webhooks import router as webhooks_router
from revenew.db import init_db

REAL_SECRET = "Superce11"

# Exact bytes captured from a real Razorpay test-mode webhook delivery. Do
# not reformat this string -- HMAC verification is byte-exact, and
# reformatting (even just whitespace) would change the signature this file
# tests against.
REAL_CAPTURED_BODY = (
    b'{"entity":"event","account_id":"acc_TTT6vjsZ731b5f","event":"payment.failed",'
    b'"contains":["payment"],"payload":{"payment":{"entity":{"id":"pay_TXwnFQisOagXOA",'
    b'"entity":"payment","amount":1000,"currency":"INR","status":"failed",'
    b'"order_id":"order_TXwlXT4y0Y61cR","international":true,"method":"card",'
    b'"amount_refunded":0,"refund_status":null,"captured":false,'
    b'"description":"#TXwlJqco2LxWow","card_id":"card_TXwnFcAURFiO7B",'
    b'"card":{"id":"card_TXwnFcAURFiO7B","entity":"card","name":"test","last4":"1111",'
    b'"network":"Visa","type":"debit","issuer":null,"international":true,"emi":false,'
    b'"sub_type":"consumer","token_iin":"411111111"},"email":"test@gmail.com",'
    b'"contact":"+919940903891","notes":[],"fee":null,"tax":null,'
    b'"error_code":"BAD_REQUEST_ERROR",'
    b'"error_description":"Your payment could not be completed as this business '
    b'accepts domestic (Indian) card payments only. Try another payment method.",'
    b'"error_source":"business","error_step":"payment_initiation",'
    b'"error_reason":"international_transaction_not_allowed",'
    b'"acquirer_data":{"auth_code":null},"created_at":1788522184}}},'
    b'"created_at":1788522184}'
)
REAL_CAPTURED_SIGNATURE = "fc9bb95437cd80ea562e2e7cf0894e051d48ee30103a673aff3dddca3169339e"
REAL_CAPTURED_EVENT_ID = "TXwnG8FYhPdrMZ"


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "revenew.db"
    init_db(db_path, reset=True)

    app = FastAPI()
    app.include_router(webhooks_router)
    app.dependency_overrides[get_conn] = _conn_override(db_path)

    monkeypatch.setattr("revenew.api.webhooks.RAZORPAY_WEBHOOK_SECRET", REAL_SECRET)
    return TestClient(app)


def _conn_override(db_path):
    from revenew.db import connect

    def _get_conn():
        conn = connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    return _get_conn


def test_the_real_captured_payload_is_accepted_end_to_end(client):
    """The regression this whole file exists for: replay the exact real
    delivery byte-for-byte and confirm it's accepted, not rejected with
    'missing event id' the way every prior version of this handler would."""
    response = client.post(
        "/webhooks/razorpay",
        content=REAL_CAPTURED_BODY,
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Event-Id": REAL_CAPTURED_EVENT_ID,
            "X-Razorpay-Signature": REAL_CAPTURED_SIGNATURE,
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "recorded", "event_id": REAL_CAPTURED_EVENT_ID}


def test_redelivering_the_same_event_id_is_a_no_op(client):
    headers = {
        "Content-Type": "application/json",
        "X-Razorpay-Event-Id": REAL_CAPTURED_EVENT_ID,
        "X-Razorpay-Signature": REAL_CAPTURED_SIGNATURE,
    }
    first = client.post("/webhooks/razorpay", content=REAL_CAPTURED_BODY, headers=headers)
    second = client.post("/webhooks/razorpay", content=REAL_CAPTURED_BODY, headers=headers)
    assert first.status_code == 200
    assert first.json()["status"] == "recorded"
    assert second.status_code == 200
    assert second.json() == {"status": "duplicate_ignored"}


def test_a_tampered_body_fails_signature_verification(client):
    """Same signature, different body -- exactly what a forged webhook looks
    like. Must be rejected, not accepted because the JSON still parses."""
    tampered = REAL_CAPTURED_BODY.replace(b'"amount":1000', b'"amount":9999999')
    response = client.post(
        "/webhooks/razorpay",
        content=tampered,
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Event-Id": REAL_CAPTURED_EVENT_ID,
            "X-Razorpay-Signature": REAL_CAPTURED_SIGNATURE,
        },
    )
    assert response.status_code == 400
    assert "signature" in response.json()["error"]


def test_a_missing_signature_header_is_rejected_when_a_real_secret_is_configured(client):
    response = client.post(
        "/webhooks/razorpay",
        content=REAL_CAPTURED_BODY,
        headers={"Content-Type": "application/json", "X-Razorpay-Event-Id": REAL_CAPTURED_EVENT_ID},
    )
    assert response.status_code == 400


def test_wrong_configured_secret_is_rejected(client, monkeypatch):
    """The real body and its real signature, but the SERVER is configured
    with a different secret than the one Razorpay actually signed with --
    must fail exactly like a forgery would, since from the server's
    perspective the two are indistinguishable."""
    monkeypatch.setattr("revenew.api.webhooks.RAZORPAY_WEBHOOK_SECRET", "a-different-secret-than-superce11")
    response = client.post(
        "/webhooks/razorpay",
        content=REAL_CAPTURED_BODY,
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Event-Id": REAL_CAPTURED_EVENT_ID,
            "X-Razorpay-Signature": REAL_CAPTURED_SIGNATURE,
        },
    )
    assert response.status_code == 400


def test_missing_event_id_header_is_rejected(client):
    """Confirms the dedup key really did move to the header -- a request
    with no X-Razorpay-Event-Id at all (valid signature notwithstanding)
    must be rejected, not silently accepted with event_id=None."""
    response = client.post(
        "/webhooks/razorpay",
        content=REAL_CAPTURED_BODY,
        headers={"Content-Type": "application/json", "X-Razorpay-Signature": REAL_CAPTURED_SIGNATURE},
    )
    assert response.status_code == 400
    assert "X-Razorpay-Event-Id" in response.json()["error"]


def test_placeholder_secret_degrades_to_accept_without_verification(client, monkeypatch):
    """Before a real secret is configured (.env still has the .env.example
    placeholder, or the var is unset), the handler must keep accepting
    webhooks -- not hard-fail every delivery during setup -- but this is the
    ONE test allowed to send a bogus signature and still get a 200, exactly
    because verification is explicitly off in this state."""
    monkeypatch.setattr("revenew.api.webhooks.RAZORPAY_WEBHOOK_SECRET", "your_webhook_secret_here")
    response = client.post(
        "/webhooks/razorpay",
        content=REAL_CAPTURED_BODY,
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Event-Id": REAL_CAPTURED_EVENT_ID,
            "X-Razorpay-Signature": "totally-bogus-not-even-hex",
        },
    )
    assert response.status_code == 200


def test_invalid_json_body_is_still_rejected(client):
    response = client.post(
        "/webhooks/razorpay",
        content=b"not json at all",
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Event-Id": "evt_whatever",
            "X-Razorpay-Signature": hmac.new(REAL_SECRET.encode(), b"not json at all", hashlib.sha256).hexdigest(),
        },
    )
    assert response.status_code == 400
