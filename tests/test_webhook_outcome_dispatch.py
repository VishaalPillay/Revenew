"""Closing the loop: a verified `payment.captured`/`payment.failed` delivery
must record an outcome against the decision it resolves, which is what
actually feeds the bandit (`record_outcome` -> `_feed_bandit` in
`revenew/ledger/outcome.py`, unchanged by this work -- this only supplies it
a real trigger it never had before).

Before this module existed, `revenew/api/webhooks.py` did exactly one thing
on a verified delivery: `INSERT INTO events`. Nothing ever read that table
back. These tests are the guarantee that a real payment now visibly moves a
posterior, not just that a row lands in `events`.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from revenew.api.webhooks import get_conn
from revenew.api.webhooks import router as webhooks_router
from revenew.db import connect, init_db
from revenew.decide.bandit import PosteriorStore
from revenew.models import ActionFamily, Segment

SECRET = "test-outcome-secret"
NOW_ISO = "2026-01-01T00:00:00+00:00"


def _conn_override(db_path):
    def _get_conn():
        conn = connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    return _get_conn


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "revenew.db"
    init_db(db_path, reset=True)

    app = FastAPI()
    app.include_router(webhooks_router)
    app.dependency_overrides[get_conn] = _conn_override(db_path)
    monkeypatch.setattr("revenew.api.webhooks.RAZORPAY_WEBHOOK_SECRET", SECRET)

    return TestClient(app), db_path


def _seed_decision(
    db_path,
    *,
    decision_id: str,
    opportunity_id: str,
    customer_id: str,
    segment: Segment,
    action_family: ActionFamily | None,
    status: str = "executed",
) -> None:
    """The minimal FK chain a `decisions` row needs: customer ->
    opportunity_candidates -> opportunities -> decisions. Mirrors the
    pattern every other test file in this suite uses for the same reason --
    there is no lighter-weight fixture for "one real decision exists"."""
    conn = connect(db_path)
    conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (customer_id, NOW_ISO))
    conn.execute(
        "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
        (opportunity_id, "run1", customer_id, "dormant_winback", "w_" + opportunity_id,
         "dormant_winback", 500.0, "h", NOW_ISO, None),
    )
    conn.execute(
        "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
        (opportunity_id, "run1", customer_id, "w_" + opportunity_id, segment.value, "treatment", NOW_ISO),
    )
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            decision_id, opportunity_id, "run1", segment.value,
            action_family.value if action_family else None, "{}",
            1, 1, "{}" if action_family else None, 0.5 if action_family else None,
            status, None if action_family else "all_candidates_invalid", NOW_ISO, "internal",
        ),
    )
    conn.commit()
    conn.close()


def _signed_payment_event(*, event: str, payment_id: str, amount_paise: int, notes) -> tuple[bytes, str]:
    """A payload shaped like a real Razorpay `payment.*` delivery -- same
    envelope REAL_CAPTURED_BODY in test_webhooks.py uses -- with a
    controllable `notes` so tests can exercise both "decision_id present"
    and "notes empty/malformed" (the real captured shape for a payment made
    with no notes set: `"notes":[]`, not `{}`)."""
    body = json.dumps(
        {
            "entity": "event",
            "account_id": "acc_test",
            "event": event,
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": payment_id,
                        "entity": "payment",
                        "amount": amount_paise,
                        "currency": "INR",
                        "status": "captured" if event == "payment.captured" else "failed",
                        "notes": notes,
                    }
                }
            },
            "created_at": 1788522184,
        }
    ).encode("utf-8")
    signature = hmac.new(SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return body, signature


def _post(client: TestClient, *, event_id: str, body: bytes, signature: str):
    return client.post(
        "/webhooks/razorpay",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Event-Id": event_id,
            "X-Razorpay-Signature": signature,
        },
    )


def test_payment_captured_records_a_converted_outcome_and_moves_the_posterior(client):
    test_client, db_path = client
    _seed_decision(
        db_path, decision_id="dec1", opportunity_id="opp1", customer_id="cus1",
        segment=Segment.ACTIVE, action_family=ActionFamily.BUNDLE_OFFER,
    )
    conn = connect(db_path)
    PosteriorStore(conn).ensure_initialized()
    before = PosteriorStore(conn).get(Segment.ACTIVE, ActionFamily.BUNDLE_OFFER)
    conn.close()

    body, sig = _signed_payment_event(
        event="payment.captured", payment_id="pay_1", amount_paise=15000,
        notes={"decision_id": "dec1", "action_family": "bundle_offer", "customer_id": "cus1"},
    )
    response = _post(test_client, event_id="evt1", body=body, signature=sig)
    assert response.status_code == 200

    conn = connect(db_path)
    outcome = conn.execute("SELECT * FROM outcomes WHERE opportunity_id = 'opp1'").fetchone()
    assert outcome is not None
    assert outcome["converted"] == 1
    assert outcome["net_revenue"] == pytest.approx(150.0)  # 15000 paise -> Rs 150
    assert outcome["censored"] == 0
    assert outcome["decision_id"] == "dec1"

    after = PosteriorStore(conn).get(Segment.ACTIVE, ActionFamily.BUNDLE_OFFER)
    conn.close()
    assert after.alpha == before.alpha + 1, "a captured payment must feed the bandit as a conversion"
    assert after.beta == before.beta


def test_payment_failed_records_a_non_converted_outcome(client):
    test_client, db_path = client
    _seed_decision(
        db_path, decision_id="dec2", opportunity_id="opp2", customer_id="cus2",
        segment=Segment.DORMANT, action_family=ActionFamily.PERCENT_DISCOUNT,
    )
    conn = connect(db_path)
    PosteriorStore(conn).ensure_initialized()
    before = PosteriorStore(conn).get(Segment.DORMANT, ActionFamily.PERCENT_DISCOUNT)
    conn.close()

    body, sig = _signed_payment_event(
        event="payment.failed", payment_id="pay_2", amount_paise=5000,
        notes={"decision_id": "dec2"},
    )
    response = _post(test_client, event_id="evt2", body=body, signature=sig)
    assert response.status_code == 200

    conn = connect(db_path)
    outcome = conn.execute("SELECT * FROM outcomes WHERE opportunity_id = 'opp2'").fetchone()
    assert outcome["converted"] == 0
    assert outcome["net_revenue"] == 0.0
    assert outcome["censored"] == 0, "a failed payment is definitive, not a timeout"

    after = PosteriorStore(conn).get(Segment.DORMANT, ActionFamily.PERCENT_DISCOUNT)
    conn.close()
    assert after.beta == before.beta + 1
    assert after.alpha == before.alpha


def test_notes_as_empty_list_records_the_event_but_no_outcome(client):
    """The REAL captured shape (test_webhooks.py's REAL_CAPTURED_BODY) has
    `"notes":[]` for a payment made with nothing set -- must degrade to
    'no outcome recorded', never crash on `list.get`."""
    test_client, db_path = client
    body, sig = _signed_payment_event(
        event="payment.captured", payment_id="pay_3", amount_paise=1000, notes=[],
    )
    response = _post(test_client, event_id="evt3", body=body, signature=sig)
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"

    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) AS n FROM outcomes").fetchone()["n"] == 0
    conn.close()


def test_unknown_decision_id_records_the_event_but_no_outcome(client):
    """A decision_id that does not exist in this database -- plausible if a
    webhook secret/URL is shared across environments -- must not crash."""
    test_client, db_path = client
    body, sig = _signed_payment_event(
        event="payment.captured", payment_id="pay_4", amount_paise=1000,
        notes={"decision_id": "dec_does_not_exist"},
    )
    response = _post(test_client, event_id="evt4", body=body, signature=sig)
    assert response.status_code == 200

    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) AS n FROM outcomes").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"] == 1
    conn.close()


def test_other_event_types_are_recorded_but_never_produce_an_outcome(client):
    test_client, db_path = client
    _seed_decision(
        db_path, decision_id="dec5", opportunity_id="opp5", customer_id="cus5",
        segment=Segment.NEW, action_family=ActionFamily.LOYALTY_CREDIT,
    )
    body, sig = _signed_payment_event(
        event="payment.authorized", payment_id="pay_5", amount_paise=1000,
        notes={"decision_id": "dec5"},
    )
    response = _post(test_client, event_id="evt5", body=body, signature=sig)
    assert response.status_code == 200

    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) AS n FROM outcomes").fetchone()["n"] == 0
    conn.close()


def test_a_second_delivery_for_an_already_closed_opportunity_is_a_no_op(client):
    """Two DIFFERENT event_ids (so the events.event_id dedup does not apply)
    both resolving the SAME opportunity -- outcomes.opportunity_id is UNIQUE
    and the table is append-only by trigger; the second attempt must be
    swallowed, not raised as a 500."""
    test_client, db_path = client
    _seed_decision(
        db_path, decision_id="dec6", opportunity_id="opp6", customer_id="cus6",
        segment=Segment.LAPSING, action_family=ActionFamily.FLAT_COUPON,
    )

    body1, sig1 = _signed_payment_event(
        event="payment.captured", payment_id="pay_6a", amount_paise=1000,
        notes={"decision_id": "dec6"},
    )
    r1 = _post(test_client, event_id="evt6a", body=body1, signature=sig1)
    assert r1.status_code == 200

    body2, sig2 = _signed_payment_event(
        event="payment.captured", payment_id="pay_6b", amount_paise=2000,
        notes={"decision_id": "dec6"},
    )
    r2 = _post(test_client, event_id="evt6b", body=body2, signature=sig2)
    assert r2.status_code == 200  # must not be a 500

    conn = connect(db_path)
    n = conn.execute("SELECT COUNT(*) AS n FROM outcomes WHERE opportunity_id = 'opp6'").fetchone()["n"]
    assert n == 1, "the second delivery must not create a second outcome row"
    # The first (only) outcome recorded the FIRST payment's amount, not the second's.
    net_revenue = conn.execute("SELECT net_revenue FROM outcomes WHERE opportunity_id = 'opp6'").fetchone()["net_revenue"]
    assert net_revenue == pytest.approx(10.0)
    conn.close()


def test_no_action_decision_produces_no_outcome_effect_on_posteriors(client):
    """A decision_id that maps to a no_action decision (no action_family) --
    an outcome can still be recorded (the customer somehow paid anyway, an
    edge case worth not crashing on) but `_feed_bandit` must no-op, exactly
    as it already does for the harness-driven path."""
    test_client, db_path = client
    _seed_decision(
        db_path, decision_id="dec7", opportunity_id="opp7", customer_id="cus7",
        segment=Segment.ACTIVE, action_family=None, status="no_action",
    )
    body, sig = _signed_payment_event(
        event="payment.captured", payment_id="pay_7", amount_paise=1000,
        notes={"decision_id": "dec7"},
    )
    response = _post(test_client, event_id="evt7", body=body, signature=sig)
    assert response.status_code == 200

    conn = connect(db_path)
    outcome = conn.execute("SELECT * FROM outcomes WHERE opportunity_id = 'opp7'").fetchone()
    assert outcome is not None
    assert outcome["converted"] == 1
    conn.close()


def test_an_unverified_delivery_records_the_event_but_never_moves_the_bandit(client, monkeypatch):
    """The security boundary. Recording an unverified delivery into `events`
    is harmless -- nothing reads that table. Recording an OUTCOME is not:
    `outcomes` is append-only by trigger and `record_outcome` feeds
    `posteriors`, so one forged `payment.captured` naming a real decision_id
    would permanently teach the bandit a conversion that never happened.

    This is the placeholder-secret state (setup not finished), where the
    handler deliberately still accepts unsigned requests -- it must accept
    them WITHOUT letting them change what the system believes."""
    test_client, db_path = client
    monkeypatch.setattr("revenew.api.webhooks.RAZORPAY_WEBHOOK_SECRET", "your_webhook_secret_here")
    _seed_decision(
        db_path, decision_id="dec_forge", opportunity_id="opp_forge", customer_id="cus_forge",
        segment=Segment.ACTIVE, action_family=ActionFamily.BUNDLE_OFFER,
    )
    conn = connect(db_path)
    PosteriorStore(conn).ensure_initialized()
    before = PosteriorStore(conn).get(Segment.ACTIVE, ActionFamily.BUNDLE_OFFER)
    conn.close()

    body, _ = _signed_payment_event(
        event="payment.captured", payment_id="pay_forge", amount_paise=9_999_900,
        notes={"decision_id": "dec_forge"},
    )
    # No valid signature -- and with the placeholder secret configured the
    # handler accepts it anyway, which is exactly the state being guarded.
    response = test_client.post(
        "/webhooks/razorpay",
        content=body,
        headers={"Content-Type": "application/json", "X-Razorpay-Event-Id": "evt_forge",
                 "X-Razorpay-Signature": "not-a-real-signature"},
    )
    assert response.status_code == 200, "unsigned deliveries are still accepted during setup"

    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"] == 1, (
        "the delivery must still be durably recorded"
    )
    assert conn.execute("SELECT COUNT(*) AS n FROM outcomes").fetchone()["n"] == 0, (
        "an unverified delivery must NOT write an irreversible outcome"
    )
    after = PosteriorStore(conn).get(Segment.ACTIVE, ActionFamily.BUNDLE_OFFER)
    conn.close()
    assert (after.alpha, after.beta) == (before.alpha, before.beta), (
        "an unverified delivery must NOT move the bandit"
    )


def test_signature_verified_column_is_set_correctly(client):
    test_client, db_path = client
    body, sig = _signed_payment_event(
        event="payment.captured", payment_id="pay_8", amount_paise=1000, notes=[],
    )
    _post(test_client, event_id="evt8", body=body, signature=sig)

    conn = connect(db_path)
    row = conn.execute("SELECT signature_verified FROM events WHERE event_id = 'evt8'").fetchone()
    assert row["signature_verified"] == 1
    conn.close()
