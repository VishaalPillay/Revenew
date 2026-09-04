"""EnvelopeEngine builds the Envelope; EnvelopeValidator applies it.

`Envelope.violations()` in revenew/models.py is the single rule table both
sides read. EnvelopeEngine renders the resulting Envelope into the LLM prompt
as context; EnvelopeValidator calls `.violations()` on every candidate the
model returns, plus two checks `.violations()` cannot make on its own because
they need a database lookup against the specific customer's history:
cooldown and monthly offer cap. One rule definition, two consumers -- they
cannot drift apart because there is only one place the discount-cap/budget/
excluded-SKU logic is written down at all.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from revenew.clock import iso
from revenew.execute import budget
from revenew.models import Candidate, DecisionCandidate, Envelope
from revenew.settings import PolicyConfig


class EnvelopeEngine:
    @staticmethod
    def build(conn: sqlite3.Connection, policy: PolicyConfig) -> Envelope:
        cogs_rows = conn.execute("SELECT sku, cogs FROM products").fetchall()
        cogs_by_sku = {r["sku"]: r["cogs"] for r in cogs_rows if r["cogs"] is not None}
        return Envelope(
            max_discount_pct=policy.max_discount_pct,
            max_absolute_discount=policy.max_absolute_discount,
            budget_remaining=budget.available(conn, policy.budget_cap),
            excluded_skus=list(policy.excluded_skus),
            cooldown_days=policy.cooldown_days,
            max_offers_per_customer_per_month=policy.max_offers_per_customer_per_month,
            cogs_by_sku=cogs_by_sku or None,
        )


def _executed_decisions_since(conn: sqlite3.Connection, customer_id: str, since_iso: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM decisions d
        JOIN opportunities o ON o.opportunity_id = d.opportunity_id
        WHERE o.customer_id = ? AND d.status = 'executed' AND d.created_at >= ?
        """,
        (customer_id, since_iso),
    ).fetchone()
    return int(row["n"])


class EnvelopeValidator:
    @staticmethod
    def validate(
        conn: sqlite3.Connection,
        envelope: Envelope,
        candidate: Candidate,
        *,
        customer_id: str,
        order_value: float,
        now: datetime,
    ) -> DecisionCandidate:
        """Full verdict for one candidate: the pure rules plus the two
        history-dependent ones `Envelope.violations()` cannot see."""
        violations = list(envelope.violations(candidate))

        # estimated_cost() needs order_value for a percentage discount, which
        # Envelope.violations() does not have -- re-check the budget rule here
        # with the real figure rather than trust whatever violations() saw.
        cost = candidate.estimated_cost(order_value)
        if cost > envelope.budget_remaining and "budget_remaining" not in violations:
            violations.append("budget_remaining")

        if envelope.cooldown_days > 0:
            cutoff = iso(now - timedelta(days=envelope.cooldown_days))
            if _executed_decisions_since(conn, customer_id, cutoff) > 0:
                violations.append("cooldown_days")

        if envelope.max_offers_per_customer_per_month >= 0:
            month_cutoff = iso(now - timedelta(days=30))
            if _executed_decisions_since(conn, customer_id, month_cutoff) >= envelope.max_offers_per_customer_per_month:
                violations.append("max_offers_per_customer_per_month")

        return DecisionCandidate(candidate=candidate, valid=len(violations) == 0, violations=violations)

    @staticmethod
    def validate_all(
        conn: sqlite3.Connection,
        envelope: Envelope,
        candidates: list[Candidate],
        *,
        customer_id: str,
        order_value: float,
        now: datetime,
    ) -> list[DecisionCandidate]:
        return [
            EnvelopeValidator.validate(conn, envelope, c, customer_id=customer_id, order_value=order_value, now=now)
            for c in candidates
        ]
