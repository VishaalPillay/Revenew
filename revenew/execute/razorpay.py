"""RazorpayAdapter: one Protocol, two implementations, everything upstream
identical either way.

`LiveAdapter` targets Razorpay's test-mode Payment Links API. That mapping was
flagged in SYSTEM_DESIGN.md section 11 as an ASSUMPTION pending verification;
it is now CONFIRMED against a real test-mode call (2026-09-04): Razorpay does
not expose a first-class "offer" object, so `create_offer` is realized as a
Payment Link carrying the discounted amount and the offer's headline in its
description, and that payload shape (`amount`/`currency`/`description`)
works exactly as assumed.

**What did NOT survive verification: provider-side idempotency.** The
original code passed `idempotency_key=` as a keyword straight into
`razorpay.Client().payment_link.create(...)`, on the assumption Razorpay's
SDK would turn it into a dedup header. Two things were wrong with that,
found only by tracing the installed SDK and then confirming live: (1) the
`razorpay` package's `Resource.create(data, **kwargs)` forwards unrecognized
kwargs all the way to `requests.Session.post(...)`, which does not accept
`idempotency_key` and raises `TypeError` on the very first call -- this
would have crashed every real execution attempt, never mind a redelivered
one; and (2) even routed correctly as a header
(`headers={"X-Razorpay-Idempotency-Key": ...}`), two live calls with an
identical key and identical payload created two DIFFERENT payment links --
Razorpay's Payment Links API does not deduplicate on it. Fixed by dropping
the kwarg entirely rather than chasing a header name that does nothing on
this endpoint. This does not weaken the actual guarantee: idempotency here
was never Razorpay's job to begin with, see `execute_decision` below.

`FixtureAdapter` never makes a network call. It records the attempted call and
returns a synthetic, deterministic reference -- what every test, the replay
harness, and a credential-less demo run against.

Every execution attempt is persisted to `executions` regardless of which
adapter ran it, keyed by `idempotency_key` (UNIQUE in db/schema.sql).
`execute_decision` checks for an existing row BEFORE ever calling the
adapter -- that check, not anything Razorpay does, is what makes a redelivered
request a database no-op instead of a second real payment link: it is the
sole call site, so a decision that already has an `executions` row never
reaches the network a second time regardless of what the provider supports.
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
    same key, which is what lets `execute_decision`'s own lookup against
    `executions.idempotency_key` (UNIQUE) recognize a redelivered request and
    skip calling the adapter a second time -- see the module docstring for why
    this, not anything Razorpay does, is the actual dedup mechanism."""
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
    """Razorpay test-mode Payment Links. Verified against a real credential --
    see the module docstring for what did and didn't survive that. Retries
    transient failures with backoff, same payload, per the failure-mode
    table -- `idempotency_key` is accepted (it's part of `RazorpayAdapter`'s
    shape, and `FixtureAdapter` uses it) but deliberately NOT forwarded to the
    SDK call: Razorpay's Payment Links endpoint doesn't honor it, and
    `execute_decision`'s own pre-check is what actually prevents a retry from
    reaching this method at all for an already-executed decision."""

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
        # merchant's dashboard. spec.amount is resolved by the caller
        # (decide/__init__.py, from the customer's real order_value) -- this
        # adapter has no basket to look one up from itself.
        amount_paise = int(round(spec.amount * 100))  # Razorpay amounts are paise
        payload = {
            "amount": amount_paise,
            "currency": "INR",
            "description": spec.headline,
            "notes": {"action_family": spec.action_family.value, "customer_id": spec.customer_id},
        }
        # NOT idempotency_key=idempotency_key here -- the razorpay SDK forwards
        # unrecognized kwargs straight through to requests.Session.post(),
        # which raises TypeError on it. Confirmed live: it crashed the very
        # first real call. See the module docstring.
        result = self._with_retry(self._client.payment_link.create, payload)
        return ExecutionResult(provider_ref=result["id"], status="sent")

    def create_payment_link(self, spec: LinkSpec, idempotency_key: str) -> ExecutionResult:
        payload = {
            "amount": int(round(spec.amount * 100)),  # Razorpay amounts are paise
            "currency": "INR",
            "description": spec.description,
        }
        # See create_offer above: idempotency_key is not forwarded to the SDK.
        result = self._with_retry(self._client.payment_link.create, payload)
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
