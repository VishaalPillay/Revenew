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
before any code reads `GROQ_API_KEY`, `GROQ_MODEL`, `RAZORPAY_KEY_ID`,
`RAZORPAY_KEY_SECRET`, or `RAZORPAY_WEBHOOK_SECRET` -- see .env.example for
the full list. `load_dotenv()` never overwrites a variable already set in the
real environment (its default `override=False`), so an explicit `export`
still wins over `.env` for anyone who wants that.

**LLM provider is Groq, not Anthropic, as a disclosed, constraint-driven
deviation from SYSTEM_DESIGN.md section 3.1's stated "Claude (Sonnet tier)"
-- there was no Anthropic billing available in this environment. See
ENGINEERING_LOG.md for the switch and decide/generator.py for how the swap
was kept contained to one module: cassette.py's cohort-keying, the record/
replay modes, and every test are provider-agnostic by construction. The
default model, gpt-oss-20b, is one of the few Groq-hosted models that honors
`strict: true` json_schema structured output -- the same schema-guarantee
property the original design got from Anthropic's forced `strict` tool call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# Reading this here, once, rather than in generator.py directly keeps "where
# env vars get resolved" in one place -- and makes it easy to swap toward
# openai/gpt-oss-120b for quality or qwen/qwen3-32b, the other Groq models
# that currently honor strict json_schema mode.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


@dataclass(frozen=True)
class PolicyConfig:
    max_discount_pct: float = 0.20
    max_absolute_discount: float = 500.0
    budget_cap: float = 50_000.0
    cooldown_days: int = 30
    max_offers_per_customer_per_month: int = 1
    excluded_skus: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_POLICY = PolicyConfig()
