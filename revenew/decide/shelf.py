"""ShelfGenerator: Arm B's candidate source in PLAN.md section 5's three-arm
ablation -- five TEMPLATED offers, one per action family, with no LLM call
anywhere in this module.

This lives here, not as a branch inside `generator.py`, because that module's
own docstring claims to be "the only LLM call in the system" -- keeping shelf
construction in its own module keeps that claim true. `CandidateGenerator`
still exposes it as `mode="shelf"` (see generator.py), so callers that already
speak in terms of `CandidateGenerator.generate()` do not need to know two
classes exist.

**Why this arm exists at all.** The obvious ablation -- `--llm off` (today's
single-candidate template) vs `--llm replay` (the LLM) -- is rigged twice
over: `_template_fallback` can never propose `BUNDLE_OFFER` and always
returns exactly one candidate, which collapses `BanditScorer.choose()` to a
propensity-1.0 no-op regardless of strategy. Arm B is the honest middle
term: a FIXED shelf with the same five families the LLM is allowed to use,
so `Arm B -> Arm C` isolates what the LLM specifically adds (composition,
personalized copy) from `Arm A -> Arm B`, which isolates what *learning*
(Thompson sampling over a real, multi-family menu) adds on its own.

**Cohort-level, not customer-level -- the same boundary the cassette draws.**
`build()` takes no `customer_id` and is memoised on the exact `cache_key` the
LLM cassette uses (decide/cassette.py), for the same reason: if the shelf
looked up anything customer-specific (their own basket, their own recommended
SKU), Arm B would carry information Arm C's LLM never receives ONLY the
cohort-level catalog, never one customer's basket, per generator.py's
`_prompt_context` docstring -- and the A/B/C comparison would be unsound by
construction before a single decision is scored. The bundle SKU pair below is
therefore drawn from a GLOBAL affinity query (shelf_queries.sql), not
detect/queries.sql's cross_sell_affinity, which is deliberately per-customer.

**If no pair survives the confidence threshold, `BUNDLE_OFFER` is omitted
outright** -- never downgraded to a second REMINDER_NUDGE. Silently
downgrading a family that should exist is exactly the failure class this
whole arm was built to stop reproducing (see generator.py's
`_template_fallback`, which does downgrade a bundle to a nudge because it has
no catalog reasoning available -- correct for the genuine LLM-unavailable
path, wrong for a shelf that is supposed to be a fair comparison point).
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from revenew.decide.cassette import cache_key
from revenew.detect.detector import CROSS_SELL_MIN_CONFIDENCE, CROSS_SELL_MIN_PAIR_COUNT
from revenew.models import (
    ActionFamily,
    Candidate,
    CandidateSet,
    Envelope,
    OpportunityType,
    Segment,
)

QUERIES_PATH = Path(__file__).resolve().parent / "shelf_queries.sql"

# Depths chosen to match the committed cassette's OWN medians (see PLAN.md
# section 5), not scaled off `policy.max_discount_pct` the way
# `_template_fallback`'s 60%-of-cap is -- the point of Arm B is a shelf whose
# spend pressure looks like what the LLM actually proposed, not a
# conservative fallback. A depth here that exceeds a stricter envelope is
# simply invalid and dropped by `EnvelopeValidator`, exactly as an
# LLM-proposed candidate would be.
PERCENT_DISCOUNT_DEPTH = 0.15
FLAT_COUPON_DEPTH = 150.0
LOYALTY_CREDIT_DEPTH = 100.0

_UNSET = object()


def _load_query(name: str) -> str:
    text = QUERIES_PATH.read_text(encoding="utf-8")
    blocks = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    for i in range(1, len(blocks), 2):
        if blocks[i] == name:
            return blocks[i + 1].strip()
    raise ValueError(f"shelf_queries.sql has no block named {name!r}")


GLOBAL_BUNDLE_PAIR_SQL = _load_query("global_bundle_pair")


class ShelfGenerator:
    """One instance per run (harness/run_replay.py holds it exactly like a
    `CandidateGenerator`/`Cassette`) so its caches live for the run's
    duration and are rebuilt fresh next time -- the population is re-seeded
    every run, and a stale global bundle pair from a PREVIOUS fixture must
    never leak into this one."""

    def __init__(self) -> None:
        self._shelf_cache: dict[str, CandidateSet] = {}
        self._bundle_pair: tuple[str, str] | None = _UNSET  # type: ignore[assignment]

    def _global_bundle_pair(self, conn: sqlite3.Connection) -> tuple[str, str] | None:
        if self._bundle_pair is _UNSET:
            row = conn.execute(
                GLOBAL_BUNDLE_PAIR_SQL,
                {
                    "min_pair_count": CROSS_SELL_MIN_PAIR_COUNT,
                    "min_confidence": CROSS_SELL_MIN_CONFIDENCE,
                },
            ).fetchone()
            self._bundle_pair = (row["sku_from"], row["sku_to"]) if row is not None else None
        return self._bundle_pair

    def build(
        self,
        conn: sqlite3.Connection,
        *,
        opportunity_type: OpportunityType,
        segment: Segment,
        rupees_at_risk: float,
        envelope: Envelope,
        catalog: list[dict],
    ) -> CandidateSet:
        """The cohort-level shelf. Memoised on the same `cache_key` the LLM
        cassette uses -- see the module docstring for why that specific key,
        not a customer-level one, is what keeps Arm B and Arm C comparable."""
        key = cache_key(opportunity_type, segment, rupees_at_risk, envelope, catalog)
        cached = self._shelf_cache.get(key)
        if cached is not None:
            return cached

        candidates = [
            Candidate(
                action_family=ActionFamily.PERCENT_DISCOUNT,
                headline=f"{PERCENT_DISCOUNT_DEPTH:.0%} off your next order",
                discount_pct=PERCENT_DISCOUNT_DEPTH,
                rationale="Shelf template: fixed depth, one per family, no LLM call.",
            ),
            Candidate(
                action_family=ActionFamily.FLAT_COUPON,
                headline=f"Rs {FLAT_COUPON_DEPTH:.0f} off",
                discount_amount=FLAT_COUPON_DEPTH,
                rationale="Shelf template: fixed depth, one per family, no LLM call.",
            ),
            Candidate(
                action_family=ActionFamily.LOYALTY_CREDIT,
                headline=f"Rs {LOYALTY_CREDIT_DEPTH:.0f} loyalty credit",
                discount_amount=LOYALTY_CREDIT_DEPTH,
                rationale="Shelf template: fixed depth, one per family, no LLM call.",
            ),
            Candidate(
                action_family=ActionFamily.REMINDER_NUDGE,
                headline="We picked something you might like",
                rationale="Shelf template: zero-cost option, one per family, no LLM call.",
            ),
        ]

        pair = self._global_bundle_pair(conn)
        if pair is not None:
            sku_from, sku_to = pair
            catalog_by_sku = {item["sku"]: item for item in catalog}
            name_to = catalog_by_sku.get(sku_to, {}).get("name", sku_to)
            candidates.append(
                Candidate(
                    action_family=ActionFamily.BUNDLE_OFFER,
                    headline=f"Complete the set: add {name_to}",
                    skus=[sku_from, sku_to],
                    rationale="Shelf template: global affinity pair, no LLM call.",
                )
            )
        # else: BUNDLE_OFFER is simply absent from this shelf, not downgraded
        # to a second nudge -- see the module docstring.

        result = CandidateSet(candidates=candidates)
        self._shelf_cache[key] = result
        return result
