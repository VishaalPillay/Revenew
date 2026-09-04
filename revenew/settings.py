"""Merchant policy configuration. One place, read by both EnvelopeEngine
(which renders it into the prompt) and EnvelopeValidator (which checks
against it programmatically) -- see decide/envelope.py.

A real deployment would load this per-merchant from a config table or an
onboarding form. For a solo 48-hour build, one importable constant is the
honest amount of configurability to build -- multi-tenant policy storage is
scope this project does not claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyConfig:
    max_discount_pct: float = 0.20
    max_absolute_discount: float = 500.0
    budget_cap: float = 50_000.0
    cooldown_days: int = 30
    max_offers_per_customer_per_month: int = 1
    excluded_skus: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_POLICY = PolicyConfig()
