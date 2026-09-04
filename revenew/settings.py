"""Merchant policy configuration. One place, read by both EnvelopeEngine
(which renders it into the prompt) and EnvelopeValidator (which checks
against it programmatically) -- see decide/envelope.py.

A real deployment would load this per-merchant from a config table or an
onboarding form. For a solo 48-hour build, one importable constant is the
honest amount of configurability to build -- multi-tenant policy storage is
scope this project does not claim.

This is also the one place `.env` gets loaded. Every entry point in this
codebase (CLI, API, harness) imports `revenew.settings` for `PolicyConfig` or
`DEFAULT_POLICY` before it does anything else, so `load_dotenv()` here runs
before any code reads `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `RAZORPAY_KEY_ID`,
`RAZORPAY_KEY_SECRET`, or `RAZORPAY_WEBHOOK_SECRET` -- see .env.example for
the full list. `load_dotenv()` never overwrites a variable already set in the
real environment (its default `override=False`), so an explicit `export`
still wins over `.env` for anyone who wants that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# Fallback matches decide/generator.py's historical hardcoded default. Reading
# it here, once, rather than in generator.py directly, keeps "where env vars
# get resolved" in one place -- and makes it trivial to point the whole system
# at claude-haiku-4-5-20251001 to conserve credit, per the .env.example comment.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")


@dataclass(frozen=True)
class PolicyConfig:
    max_discount_pct: float = 0.20
    max_absolute_discount: float = 500.0
    budget_cap: float = 50_000.0
    cooldown_days: int = 30
    max_offers_per_customer_per_month: int = 1
    excluded_skus: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_POLICY = PolicyConfig()
