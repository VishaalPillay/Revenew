"""Declares ground truth, and only ground truth. Nothing here is a suggestion.

The (segment, action_family) table below is not filler data. It is built so
that the flat "always pick the free option" policy and the flat "always pick
the biggest discount" policy are both wrong, and wrong in different segments:

  NEW      PERCENT_DISCOUNT wins (143) over REMINDER_NUDGE (131) -- a real
           discount narrowly beats a free nudge once its conversion lift is
           weighed against its cost. Close enough that a bandit needs real
           evidence, not a prior, to separate them.

  ACTIVE   PERCENT_DISCOUNT (400) is WORSE than doing nothing (420). Active
           customers convert organically; discounting them mostly pays people
           who were already going to buy. BUNDLE_OFFER (594) is the true best
           cell, and it is nowhere near the cheapest option -- it wins on
           genuine incremental lift, not on cost.

  LAPSING  FLAT_COUPON (229.6) wins. Cheap and well-timed beats both a bigger
           discount and a bare nudge.

  DORMANT  Baseline reward is nearly zero (14); everything beats doing
           nothing, but only PERCENT_DISCOUNT (77) is worth the spend --
           REMINDER_NUDGE (28) badly underperforms because a customer who
           already ignored organic touchpoints needs a real incentive.

A bandit that converges on this table has learned something a fixed rule
("always discount" or "always nudge") could not have gotten right in every
segment at once. That is what test_replay_equality and the regret curve in
harness/regret.py are checking.

Reward throughout is expected net revenue per decision: p_convert * mean_revenue,
zero when the customer does not convert -- matching Outcome.net_revenue exactly,
which is what BanditRewardFeed consumes (SYSTEM_DESIGN.md section 6).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

from revenew.models import ActionFamily, Segment

# ============================================================= ground truth --


@dataclass(frozen=True)
class Cell:
    p_convert: float
    mean_revenue: float

    @property
    def expected_reward(self) -> float:
        return self.p_convert * self.mean_revenue


BASELINE: dict[Segment, Cell] = {
    Segment.NEW: Cell(0.08, 800),
    Segment.ACTIVE: Cell(0.35, 1200),
    Segment.LAPSING: Cell(0.10, 900),
    Segment.DORMANT: Cell(0.02, 700),
}

TRUTH: dict[tuple[Segment, ActionFamily], Cell] = {
    (Segment.NEW, ActionFamily.PERCENT_DISCOUNT): Cell(0.22, 650),
    (Segment.NEW, ActionFamily.FLAT_COUPON): Cell(0.18, 700),
    (Segment.NEW, ActionFamily.BUNDLE_OFFER): Cell(0.14, 780),
    (Segment.NEW, ActionFamily.LOYALTY_CREDIT): Cell(0.12, 760),
    (Segment.NEW, ActionFamily.REMINDER_NUDGE): Cell(0.16, 820),
    (Segment.ACTIVE, ActionFamily.PERCENT_DISCOUNT): Cell(0.40, 1000),
    (Segment.ACTIVE, ActionFamily.FLAT_COUPON): Cell(0.38, 1100),
    (Segment.ACTIVE, ActionFamily.BUNDLE_OFFER): Cell(0.44, 1350),
    (Segment.ACTIVE, ActionFamily.LOYALTY_CREDIT): Cell(0.37, 1150),
    (Segment.ACTIVE, ActionFamily.REMINDER_NUDGE): Cell(0.36, 1220),
    (Segment.LAPSING, ActionFamily.PERCENT_DISCOUNT): Cell(0.24, 750),
    (Segment.LAPSING, ActionFamily.FLAT_COUPON): Cell(0.28, 820),
    (Segment.LAPSING, ActionFamily.BUNDLE_OFFER): Cell(0.15, 880),
    (Segment.LAPSING, ActionFamily.LOYALTY_CREDIT): Cell(0.20, 800),
    (Segment.LAPSING, ActionFamily.REMINDER_NUDGE): Cell(0.13, 900),
    (Segment.DORMANT, ActionFamily.PERCENT_DISCOUNT): Cell(0.14, 550),
    (Segment.DORMANT, ActionFamily.FLAT_COUPON): Cell(0.09, 650),
    (Segment.DORMANT, ActionFamily.BUNDLE_OFFER): Cell(0.05, 700),
    (Segment.DORMANT, ActionFamily.LOYALTY_CREDIT): Cell(0.11, 600),
    (Segment.DORMANT, ActionFamily.REMINDER_NUDGE): Cell(0.04, 700),
}

assert set(TRUTH) == {(s, f) for s in Segment for f in ActionFamily}, (
    "ground truth must cover every one of the 20 (segment, action_family) cells"
)


def best_family(segment: Segment) -> tuple[ActionFamily, float]:
    """The oracle's answer for one segment: family and its expected reward,
    already compared against doing nothing."""
    candidates = [(f, TRUTH[(segment, f)].expected_reward) for f in ActionFamily]
    best = max(candidates, key=lambda x: x[1])
    baseline = BASELINE[segment].expected_reward
    if baseline >= best[1]:
        return None, baseline  # type: ignore[return-value]  # oracle correctly does nothing
    return best


def write_ground_truth(conn: sqlite3.Connection, *, seed: int) -> None:
    """Populate harness.db. One-time, at the start of a replay run."""
    conn.executemany(
        "INSERT INTO ground_truth (segment, action_family, p_convert, mean_revenue) "
        "VALUES (?, ?, ?, ?)",
        [(s.value, f.value, cell.p_convert, cell.mean_revenue) for (s, f), cell in TRUTH.items()],
    )
    conn.executemany(
        "INSERT INTO ground_truth_baseline (segment, p_convert, mean_revenue) VALUES (?, ?, ?)",
        [(s.value, c.p_convert, c.mean_revenue) for s, c in BASELINE.items()],
    )
    conn.executemany(
        "INSERT INTO fixture_meta (key, value) VALUES (?, ?)",
        [("seed", str(seed)), ("generated_at", datetime.now(UTC).isoformat())],
    )
    conn.commit()


# ======================================================= synthetic population --

CATEGORIES = ("apparel", "footwear", "accessories")

# (customer_count_share, orders_range, days_since_LAST_order_range) per intended
# segment. The detector re-derives segment from raw orders, not from this label
# -- this only controls how the synthetic order history is shaped so that
# derivation lands where intended.
#
# The ranges are recency of the LAST order specifically, and are kept clear of
# the ACTIVE/LAPSING (30d) and LAPSING/DORMANT (90d) boundaries in
# detect/detector.py's segment_of() on purpose, so a customer never straddles a
# cutoff by construction. Earlier orders (when orders_count > 1) are placed
# BEFORE the last one by ORDER_GAP_DAYS, not drawn independently -- see the
# note in generate_population for why that distinction is load-bearing.
POPULATION_SHAPE = {
    Segment.NEW: (0.35, (0, 1), (1, 20)),
    Segment.ACTIVE: (0.25, (2, 8), (2, 25)),
    Segment.LAPSING: (0.20, (2, 6), (35, 85)),
    # orders_count starts at 2, not 1: a customer needs to have RE-ordered and
    # then gone quiet to be meaningfully dormant. One old order with no second
    # is FIRST_ORDER_RETENTION's target population, and segment_of() correctly
    # classifies orders_count<=1 as NEW regardless of its age -- letting this
    # range include 1 was double-counting that case as dormant by construction.
    Segment.DORMANT: (0.20, (2, 5), (100, 400)),
}

# Typical days between consecutive orders for a customer who is (or was)
# actively buying. Drives how far back earlier orders land relative to the
# last one -- not how recent the last one is, which POPULATION_SHAPE controls
# directly.
ORDER_GAP_DAYS = (15, 45)


@dataclass(frozen=True)
class SeedData:
    products: list[tuple[str, str, str, float, float | None]]  # sku, name, category, price, cogs
    customers: list[tuple[str, str]]  # customer_id, created_at
    orders: list[tuple[str, str, str, float, str]]  # order_id, customer_id, placed_at, amount, status
    order_items: list[tuple[str, str, int, float]]  # order_id, sku, qty, unit_price
    intended_segment: dict[str, Segment] = field(default_factory=dict)


def _products(rng: np.random.Generator) -> list[tuple[str, str, str, float, float | None]]:
    names = [
        ("SKU-A01", "Classic Tee", "apparel", 599),
        ("SKU-A02", "Slim Jeans", "apparel", 1899),
        ("SKU-A03", "Hoodie", "apparel", 1499),
        ("SKU-F01", "Runner Low", "footwear", 2999),
        ("SKU-F02", "Trail Boot", "footwear", 3999),
        ("SKU-C01", "Cap", "accessories", 399),
        ("SKU-C02", "Backpack", "accessories", 1299),
        ("SKU-C03", "Socks 3pk", "accessories", 299),
    ]
    out = []
    for sku, name, cat, price in names:
        # ~70% of products have merchant-supplied COGS; the rest are NULL on
        # purpose, so margin-aware code paths have a real "unknown" case to hit.
        cogs = round(price * float(rng.uniform(0.45, 0.65)), 2) if rng.random() < 0.7 else None
        out.append((sku, name, cat, float(price), cogs))
    return out


def generate_population(seed: int, n_customers: int, now: datetime) -> SeedData:
    """Synthetic customers + order history shaped to land in each Segment.

    `now` is the virtual clock's start time: all "days since" figures in
    POPULATION_SHAPE are relative to it, so a replay run's segment mix is
    stable regardless of when in wall-clock time it happens to run.
    """
    rng = np.random.default_rng(seed)
    products = _products(rng)
    skus = [p[0] for p in products]
    prices = {p[0]: p[3] for p in products}

    customers: list[tuple[str, str]] = []
    orders: list[tuple[str, str, str, float, str]] = []
    order_items: list[tuple[str, str, int, float]] = []
    intended: dict[str, Segment] = {}

    shares = np.array([POPULATION_SHAPE[s][0] for s in Segment])
    shares = shares / shares.sum()
    seg_choices = rng.choice(list(Segment), size=n_customers, p=shares)

    for i, seg in enumerate(seg_choices):
        cid = f"cus_{i:06d}"
        _, orders_range, recency_range = POPULATION_SHAPE[seg]
        n_orders = int(rng.integers(orders_range[0], orders_range[1] + 1))
        days_since_last = int(rng.integers(recency_range[0], recency_range[1] + 1))

        if n_orders == 0:
            created_at = now - timedelta(days=days_since_last + 5)
            customers.append((cid, created_at.isoformat()))
            intended[cid] = seg
            continue

        # The LAST order is placed exactly `days_since_last` days before now --
        # set directly, not left to emerge from independently drawn offsets.
        # Independent draws were the bug: with the last order sampled uniformly
        # over the customer's whole history, its expected recency shrinks as
        # order count grows (max of k uniforms drifts toward the upper bound),
        # so a DORMANT customer with several orders was landing with a RECENT
        # last order and getting classified ACTIVE. Fixing the last order's
        # position and placing earlier ones progressively further back removes
        # that drift entirely: recency is a direct parameter, not an emergent one.
        gaps = rng.integers(ORDER_GAP_DAYS[0], ORDER_GAP_DAYS[1] + 1, size=max(n_orders - 1, 0))
        offsets_before_last = np.concatenate([[0], np.cumsum(gaps)])[::-1]  # oldest first, last = 0
        created_at = now - timedelta(days=days_since_last + int(offsets_before_last[0]) + 10)
        customers.append((cid, created_at.isoformat()))
        intended[cid] = seg

        for j, extra_days_back in enumerate(offsets_before_last):
            oid = f"ord_{cid}_{j:02d}"
            placed_at = now - timedelta(days=days_since_last + int(extra_days_back))
            n_items = int(rng.integers(1, 4))
            chosen = rng.choice(skus, size=n_items, replace=False)
            total = 0.0
            for sku in chosen:
                qty = int(rng.integers(1, 3))
                unit_price = prices[sku]
                order_items.append((oid, str(sku), qty, unit_price))
                total += qty * unit_price
            orders.append((oid, cid, placed_at.isoformat(), round(total, 2), "captured"))

    return SeedData(products, customers, orders, order_items, intended)


def seed_runtime_db(conn: sqlite3.Connection, data: SeedData) -> None:
    conn.executemany(
        "INSERT INTO products (sku, name, category, price, cogs) VALUES (?, ?, ?, ?, ?)",
        data.products,
    )
    conn.executemany("INSERT INTO customers (customer_id, created_at) VALUES (?, ?)", data.customers)
    conn.executemany(
        "INSERT INTO orders (order_id, customer_id, placed_at, amount, status) VALUES (?, ?, ?, ?, ?)",
        data.orders,
    )
    conn.executemany(
        "INSERT INTO order_items (order_id, sku, qty, unit_price) VALUES (?, ?, ?, ?)",
        data.order_items,
    )
    conn.commit()


# ================================================================ resolution --


class OutcomeOracle:
    """Draws whether a decision converts, using ground truth the runtime cannot see.

    One instance per replay run, seeded once. `resolve` is the only method the
    runtime-facing replay driver calls -- see harness/run_replay.py -- and it
    never exposes `TRUTH` or `BASELINE` directly to anything downstream of it.
    """

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def resolve(self, segment: Segment, action_family: ActionFamily | None) -> tuple[bool, float]:
        """`action_family=None` means the control arm / no-action case."""
        cell = BASELINE[segment] if action_family is None else TRUTH[(segment, action_family)]
        converted = bool(self._rng.random() < cell.p_convert)
        if not converted:
            return False, 0.0
        # Revenue itself is noisy around the declared mean -- a fixture where
        # every conversion pays out exactly the mean would make the Welch
        # interval in IncrementalEstimator artificially tight.
        noise = float(self._rng.lognormal(mean=-0.02, sigma=0.18))
        return True, round(cell.mean_revenue * noise, 2)
