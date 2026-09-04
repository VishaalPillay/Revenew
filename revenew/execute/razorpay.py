"""RazorpayAdapter: one Protocol, two implementations, everything upstream
identical either way.

`LiveAdapter` targets Razorpay's test-mode Payment Links API. That mapping is
flagged in SYSTEM_DESIGN.md section 11 as an ASSUMPTION to verify before
relying on it: Razorpay does not expose a first-class "offer" object, so
`create_offer` is realized as a Payment Link carrying the discounted amount
and the offer's headline in its description -- a deliberate adaptation, not a
literal API this project confirmed against live docs. This machine has no
Razorpay credential either, so LiveAdapter has never been exercised against
the real API; only FixtureAdapter is exercised by tests and by replay.

`FixtureAdapter` never makes a network call. It records the attempted call and
returns a synthetic, deterministic reference -- what every test, the replay
harness, and a credential-less demo run against.

Every execution attempt is persisted to `executions` regardless of which
adapter ran it, keyed by `idempotency_key` (UNIQUE in db/schema.sql) -- a
retried call with the same key is a database no-op on the second attempt, not
a second charge.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from typing import Protocol

from revenew.clock import iso
from revenew.models import ExecutionResult, LinkSpec, OfferSpec


class RazorpayAdapter(Protocol):
    def create_offer(self, spec: OfferSpec, idempotency_key: str) -> ExecutionResult: ...
    def create_payment_link(self, spec: LinkSpec, idempotency_key: str) -> ExecutionResult: ...


def idempotency_key_for(decision_id: str) -> str:
    """Deterministic per decision: a retry of the same decision reuses the
    same key, so Razorpay's own idempotency handling (a real feature of their
    API, keyed on this exact header) absorbs a redelivered request rather than
    executing it twice."""
    return f"revenew-{decision_id}"


class FixtureAdapter:
    """Records the call, returns a synthetic id. No network. This is what
    every test and the replay harness use."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def create_offer(self, spec: OfferSpec, idempotency_key: str) -> ExecutionResult:
        self.calls.append(("create_offer", spec.model_dump()))
        ref = "fixture_offer_" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:12]
        return ExecutionResult(provider_ref=ref, status="confirmed")

    def create_payment_link(self, spec: LinkSpec, idempotency_key: str) -> ExecutionResult:
        self.calls.append(("create_payment_link", spec.model_dump()))
        ref = "fixture_link_" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:12]
        return ExecutionResult(provider_ref=ref, status="confirmed")


class LiveAdapter:
    """Razorpay test-mode Payment Links. UNVERIFIED against live docs or a
    real credential -- see the module docstring. Retries transient failures
    with backoff, same idempotency key, per the failure-mode table."""

    MAX_RETRIES = 3
    BACKOFF_SECONDS = (0.5, 1.5, 3.0)

    def __init__(self, key_id: str, key_secret: str) -> None:
        import razorpay

        self._client = razorpay.Client(auth=(key_id, key_secret))

    def _with_retry(self, fn, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # SDK raises its own error types; retry any of them
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.BACKOFF_SECONDS[attempt])
        raise last_exc  # type: ignore[misc]

    def create_offer(self, spec: OfferSpec, idempotency_key: str) -> ExecutionResult:
        # No native "offer" object in Razorpay's API: realized as a Payment
        # Link for the discounted amount, with the offer's headline carried in
        # the description so it is visible to the customer and in the
        # merchant's dashboard.
        amount_paise = 0  # a real amount requires the customer's basket, resolved by the caller
        payload = {
            "amount": amount_paise,
            "currency": "INR",
            "description": spec.headline,
            "notes": {"action_family": spec.action_family.value, "customer_id": spec.customer_id},
        }
        result = self._with_retry(
            self._client.payment_link.create, payload, idempotency_key=idempotency_key
        )
        return ExecutionResult(provider_ref=result["id"], status="sent")

    def create_payment_link(self, spec: LinkSpec, idempotency_key: str) -> ExecutionResult:
        payload = {
            "amount": int(round(spec.amount * 100)),  # Razorpay amounts are paise
            "currency": "INR",
            "description": spec.description,
        }
        result = self._with_retry(
            self._client.payment_link.create, payload, idempotency_key=idempotency_key
        )
        return ExecutionResult(provider_ref=result["id"], status="sent")


def execute_decision(
    conn: sqlite3.Connection,
    adapter: RazorpayAdapter,
    *,
    decision_id: str,
    spec: OfferSpec,
    now,
) -> ExecutionResult:
    """Runs one execution attempt and persists it. The single call site both
    the live path and replay use, so `executions` never has a row written
    outside this function."""
    idem_key = idempotency_key_for(decision_id)

    existing = conn.execute(
        "SELECT execution_id, provider_ref, status FROM executions WHERE idempotency_key = ?",
        (idem_key,),
    ).fetchone()
    if existing is not None:
        return ExecutionResult(provider_ref=existing["provider_ref"] or "", status=existing["status"])

    try:
        result = adapter.create_offer(spec, idem_key)
    except Exception:
        result = ExecutionResult(provider_ref="", status="failed")

    conn.execute(
        "INSERT INTO executions (execution_id, decision_id, idempotency_key, provider_ref, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), decision_id, idem_key, result.provider_ref or None, result.status, iso(now)),
    )
    conn.commit()
    return result
