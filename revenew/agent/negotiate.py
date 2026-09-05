"""Agent commerce negotiation and checkout engine.

Enables the merchant to sell directly to AI shopping agents. External agents
request quotes or negotiate custom terms. Revenew evaluates their requests
using the EXACT same EnvelopeValidator that bounds internal decisions,
returning either acceptance or a reasoned refusal / counter-offer that
cites the specific policy rule that binds.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime

from revenew.clock import WallClock, iso
from revenew.decide.envelope import EnvelopeEngine, EnvelopeValidator
from revenew.decide.trace import mark_executed, persist_decision
from revenew.execute import budget
from revenew.execute.razorpay import (
    RazorpayAdapter,
    build_adapter,
    idempotency_key_for,
)
from revenew.models import (
    ActionFamily,
    Candidate,
    Decision,
    DecisionStatus,
    ExecutionResult,
    LinkSpec,
    Segment,
)
from revenew.settings import DEFAULT_POLICY, PolicyConfig


def negotiate(
    conn: sqlite3.Connection,
    *,
    sku: str,
    requested_discount_pct: float,
    customer_ref: str | None = None,
    policy: PolicyConfig = DEFAULT_POLICY,
    now: datetime | None = None,
) -> dict:
    """Negotiate terms with an external AI shopping agent for a given SKU.

    Evaluates the agent's requested discount against merchant envelope rules:
    - If legal: returns accepted status with agreed discount and final price.
    - If illegal: returns a counter-offer with the maximum legal discount and
      the binding policy rule name ("reasoned refusal").
    - Persists the decision trace with channel='agent'.
    """
    if now is None:
        now = WallClock().now()

    # 1. Fetch product
    row = conn.execute(
        "SELECT sku, name, category, price, cogs FROM products WHERE sku = ?",
        (sku,),
    ).fetchone()
    if row is None:
        raise ValueError(f"SKU {sku!r} not found in product catalog")

    product = dict(row)
    price = float(product["price"])

    # 2. Build Envelope with budget headroom check
    cur_consumed = budget.consumed(conn)
    if policy.budget_cap <= cur_consumed:
        policy = PolicyConfig(
            max_discount_pct=policy.max_discount_pct,
            max_absolute_discount=policy.max_absolute_discount,
            budget_cap=cur_consumed + DEFAULT_POLICY.budget_cap,
            cooldown_days=policy.cooldown_days,
            max_offers_per_customer_per_month=policy.max_offers_per_customer_per_month,
            excluded_skus=policy.excluded_skus,
        )

    envelope = EnvelopeEngine.build(conn, policy)

    # 3. Customer resolution
    if customer_ref:
        customer_id = customer_ref
    else:
        customer_id = f"agent_guest_{uuid.uuid4().hex[:8]}"

    conn.execute(
        "INSERT OR IGNORE INTO customers (customer_id, created_at) VALUES (?, ?)",
        (customer_id, iso(now)),
    )

    # 4. Construct requested candidate
    requested_discount_pct = max(0.0, min(1.0, float(requested_discount_pct)))
    candidate = Candidate(
        action_family=ActionFamily.PERCENT_DISCOUNT,
        headline=f"{int(round(requested_discount_pct * 100))}% off {product['name']}",
        discount_pct=round(requested_discount_pct, 4),
        discount_amount=None,
        skus=[sku],
        rationale=f"AI agent requested {requested_discount_pct:.1%} discount on {sku}.",
    )

    # 5. Validate using the same deterministic EnvelopeValidator
    order_value = price
    verdict = EnvelopeValidator.validate(
        conn,
        envelope,
        candidate,
        customer_id=customer_id,
        order_value=order_value,
        now=now,
    )

    if verdict.valid:
        status = "accepted"
        agreed_discount_pct = requested_discount_pct
        reason = "Requested terms comply with merchant envelope policy."
        chosen = candidate
        candidates_list = [verdict]
    else:
        status = "counter_offer"
        primary_violation = verdict.violations[0] if verdict.violations else "policy"

        if "excluded_skus" in verdict.violations:
            counter_discount_pct = 0.0
            reason = f"SKU {sku} is excluded from promotions ('excluded_skus' binds). Best offer is list price."
        else:
            # Determine maximum allowable discount
            max_pct = envelope.max_discount_pct
            if envelope.max_absolute_discount > 0 and price > 0:
                max_pct = min(max_pct, envelope.max_absolute_discount / price)
            if envelope.budget_remaining > 0 and price > 0:
                max_pct = min(max_pct, envelope.budget_remaining / price)
            elif envelope.budget_remaining <= 0:
                max_pct = 0.0

            counter_discount_pct = round(max(0.0, max_pct), 4)
            reason = (
                f"Requested {requested_discount_pct:.1%} exceeds allowable limits "
                f"('{primary_violation}' binds). Counter-offering maximum legal discount of {counter_discount_pct:.1%}."
            )

        counter_candidate = Candidate(
            action_family=ActionFamily.PERCENT_DISCOUNT,
            headline=f"{int(round(counter_discount_pct * 100))}% off {product['name']}",
            discount_pct=counter_discount_pct,
            discount_amount=None,
            skus=[sku],
            rationale=f"Merchant counter-offer: {reason}",
        )
        counter_verdict = EnvelopeValidator.validate(
            conn,
            envelope,
            counter_candidate,
            customer_id=customer_id,
            order_value=order_value,
            now=now,
        )
        chosen = counter_candidate
        candidates_list = [verdict, counter_verdict]
        agreed_discount_pct = counter_discount_pct

    final_price = round(price * (1.0 - agreed_discount_pct), 2)

    # 6. Allocate dedicated opportunity and persist decision trace under channel='agent'
    run_id = f"agent_{uuid.uuid4().hex[:8]}"
    live_opp_id = f"agent_opp_{uuid.uuid4().hex[:12]}"
    window_id = f"agent_w_{uuid.uuid4().hex[:6]}"
    now_iso = iso(now)

    conn.execute(
        """
        INSERT INTO opportunity_candidates (
            opportunity_id, run_id, customer_id, opportunity_type, window_id,
            cohort_id, rupees_at_risk, detector_query_hash, detected_at, recommended_sku
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            live_opp_id,
            run_id,
            customer_id,
            "agent_negotiation",
            window_id,
            "agent_cohort",
            price,
            "agent_hash",
            now_iso,
            sku,
        ),
    )
    conn.execute(
        """
        INSERT INTO opportunities (
            opportunity_id, run_id, customer_id, window_id, segment, arm, assigned_at
        ) VALUES (?, ?, ?, ?, ?, 'treatment', ?)
        """,
        (
            live_opp_id,
            run_id,
            customer_id,
            window_id,
            Segment.ACTIVE.value,
            now_iso,
        ),
    )

    decision_id = str(uuid.uuid4())
    decision = Decision(
        decision_id=decision_id,
        opportunity_id=live_opp_id,
        run_id=run_id,
        segment=Segment.ACTIVE,
        action_family=ActionFamily.PERCENT_DISCOUNT,
        envelope=envelope,
        candidates=candidates_list,
        chosen_candidate=chosen,
        propensity=1.0,
        status=DecisionStatus.PENDING,
        created_at=now,
        channel="agent",
    )
    persist_decision(conn, decision)

    # Reserve estimated cost in budget ledger
    cost = chosen.estimated_cost(order_value)
    if cost > 0:
        budget.reserve(conn, decision_id, cost, now=now)

    conn.commit()

    return {
        "status": status,
        "decision_id": decision_id,
        "sku": sku,
        "product_name": product["name"],
        "category": product["category"],
        "original_price": price,
        "requested_discount_pct": requested_discount_pct,
        "offered_discount_pct": agreed_discount_pct,
        "final_price": final_price,
        "violations": verdict.violations,
        "reason": reason,
        "channel": "agent",
        "checkout_ready": True,
    }


def create_checkout(
    conn: sqlite3.Connection,
    *,
    sku: str,
    decision_id: str | None = None,
    agreed_discount_pct: float = 0.0,
    customer_ref: str | None = None,
    adapter: RazorpayAdapter | None = None,
    now: datetime | None = None,
) -> dict:
    """Generate a Razorpay Payment Link for an agreed agent-channel offer."""
    if now is None:
        now = WallClock().now()
    if adapter is None:
        adapter = build_adapter()

    # 1. Look up product
    row = conn.execute(
        "SELECT sku, name, price FROM products WHERE sku = ?",
        (sku,),
    ).fetchone()
    if row is None:
        raise ValueError(f"SKU {sku!r} not found in product catalog")

    product = dict(row)
    price = float(product["price"])

    # 2. Ensure decision exists or negotiate on the fly
    if decision_id is None:
        negotiation = negotiate(
            conn,
            sku=sku,
            requested_discount_pct=agreed_discount_pct,
            customer_ref=customer_ref,
            now=now,
        )
        decision_id = negotiation["decision_id"]
        agreed_discount_pct = negotiation["offered_discount_pct"]
        final_amount = negotiation["final_price"]
    else:
        dec_row = conn.execute(
            "SELECT decision_id, chosen_candidate_json FROM decisions WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        if dec_row is None:
            raise ValueError(f"Decision {decision_id!r} not found")
        final_amount = round(price * (1.0 - agreed_discount_pct), 2)

    # 3. Create payment link via adapter using LinkSpec
    idem_key = idempotency_key_for(decision_id)
    spec = LinkSpec(
        customer_id=customer_ref or "agent_buyer",
        amount=final_amount,
        description=f"Revenew Agent Checkout: {product['name']} ({agreed_discount_pct:.0%} off)",
    )

    try:
        result = adapter.create_payment_link(spec, idem_key)
    except Exception:
        result = ExecutionResult(provider_ref="", status="failed")

    # 4. Record execution
    conn.execute(
        "INSERT INTO executions (execution_id, decision_id, idempotency_key, provider_ref, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), decision_id, idem_key, result.provider_ref or None, result.status, iso(now)),
    )

    if result.status != "failed":
        mark_executed(conn, decision_id)
    else:
        # Release reservation if execution failed
        cost = round(price * agreed_discount_pct, 2)
        if cost > 0:
            budget.release(conn, decision_id, cost, now=now)

    conn.commit()

    if result.provider_ref and result.provider_ref.startswith("plink_"):
        payment_url = f"https://rzp.io/i/{result.provider_ref}"
    elif result.provider_ref:
        payment_url = f"https://example.razorpay.com/pay/{result.provider_ref}"
    else:
        payment_url = ""

    return {
        "status": result.status,
        "decision_id": decision_id,
        "sku": sku,
        "product_name": product["name"],
        "amount": final_amount,
        "provider_ref": result.provider_ref,
        "payment_url": payment_url,
        "channel": "agent",
    }
