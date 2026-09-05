"""`learning_curve()` / `compute_decision_regret()` bucket a run's decisions by
`ORDER BY created_at ASC, opportunity_id ASC`. Real regression: the tie-break
used to be `decision_id`, which is `str(uuid.uuid4())` (`decide/__init__.py`)
-- a fresh random value every run. Two decisions made in the same simulated
day (a routine occurrence: the virtual clock advances once per day, not once
per decision) share `created_at` exactly, so their relative order depended on
a value that is, by construction, never the same twice. Discovered by running
the same 90-day/3000-customer replay twice at the same seed and finding the
exported `demo_learning_curve` bucket percentages differed (10.3%/71.9% vs
10.9%/76.1%) even though every decision's own (segment, action_family,
propensity) was, when compared by the content-addressed `opportunity_id`
rather than by the random `decision_id`, byte-identical between the two runs.

`opportunity_id` is content-addressed (`detect/detector.py`'s
`_opportunity_id`, a function of `run_id`/`window_id`/`opportunity_type`/
`customer_id`), so it gives the same total order every time at the same
seed. This test proves the fix directly: the same logical decisions, entered
under two different random `decision_id` values and two different insertion
orders, must bucket identically.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from harness.regret import compute_decision_regret, learning_curve
from revenew.clock import iso
from revenew.db import connect, init_db
from revenew.models import ActionFamily, Segment

NOW = datetime(2026, 1, 1, tzinfo=UTC)

# Two distinct timestamps, several decisions sharing each -- exactly the
# "same simulated day" shape the real bug needs to reproduce. Segment/family
# pairs are fixed so `learning_curve`'s optimal/regret math has real ground
# truth to compare against, not just a row count.
_SCENARIO = [
    # (opportunity_id, created_at_offset_days, segment, family)
    ("opp_a1", 0, Segment.ACTIVE, ActionFamily.BUNDLE_OFFER),
    ("opp_a2", 0, Segment.ACTIVE, ActionFamily.REMINDER_NUDGE),
    ("opp_a3", 0, Segment.ACTIVE, ActionFamily.PERCENT_DISCOUNT),
    ("opp_a4", 0, Segment.DORMANT, ActionFamily.FLAT_COUPON),
    ("opp_a5", 1, Segment.DORMANT, ActionFamily.PERCENT_DISCOUNT),
    ("opp_a6", 1, Segment.LAPSING, ActionFamily.LOYALTY_CREDIT),
    ("opp_a7", 1, Segment.LAPSING, ActionFamily.BUNDLE_OFFER),
    ("opp_a8", 1, Segment.NEW, ActionFamily.REMINDER_NUDGE),
]


def _build_db(db_path: Path, *, decision_ids: list[str], insertion_order: list[int]) -> None:
    """One fresh database with `_SCENARIO`'s decisions, using the given
    (arbitrary, per-call-different) `decision_id`s, inserted in the given
    (arbitrary, per-call-different) row order. If the bucketing were still
    keyed on decision_id, varying either of these would change which
    same-`created_at` decision lands in which bucket."""
    init_db(db_path, reset=True)
    conn = connect(db_path)
    conn.execute("INSERT INTO customers VALUES ('cus1', ?)", (iso(NOW),))

    for idx in insertion_order:
        opp_id, day_offset, segment, family = _SCENARIO[idx]
        window_id = f"w{idx}"  # UNIQUE(run_id, customer_id, window_id) needs one per opportunity here
        created_at = iso(NOW + timedelta(days=day_offset))
        conn.execute(
            "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
            (opp_id, "run1", "cus1", "dormant_winback", window_id, "dormant_winback", 500, "h", created_at, None),
        )
        conn.execute(
            "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
            (opp_id, "run1", "cus1", window_id, segment.value, "treatment", created_at),
        )
        conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                decision_ids[idx], opp_id, "run1", segment.value, family.value, "{}",
                1, 1, "{}", 0.5, "executed", None, created_at, "internal",
            ),
        )
    conn.commit()
    conn.close()


def test_learning_curve_is_independent_of_random_decision_id(tmp_path: Path):
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"

    # Deliberately different "random" decision_ids and a deliberately
    # different insertion order between the two databases -- both are things
    # that legitimately differ between two real runs of the same seed
    # (decision_id is uuid4; insertion order follows whatever order the
    # detector's SQL happened to return rows in).
    _build_db(
        db_a,
        decision_ids=["dec_00000001", "dec_00000002", "dec_00000003", "dec_00000004",
                       "dec_00000005", "dec_00000006", "dec_00000007", "dec_00000008"],
        insertion_order=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    _build_db(
        db_b,
        decision_ids=["dec_zzzzzzzz", "dec_yyyyyyyy", "dec_xxxxxxxx", "dec_wwwwwwww",
                       "dec_vvvvvvvv", "dec_uuuuuuuu", "dec_tttttttt", "dec_ssssssss"],
        insertion_order=[7, 3, 1, 5, 0, 6, 2, 4],
    )

    conn_a, conn_b = connect(db_a), connect(db_b)
    try:
        curve_a = learning_curve(conn_a, buckets=2)
        curve_b = learning_curve(conn_b, buckets=2)
        assert curve_a == curve_b, (
            "bucketed learning curve must not depend on decision_id or insertion order"
        )

        # compute_decision_regret returns DecisionRegret dataclasses whose
        # decision_id/created_at legitimately differ between the two DBs --
        # compare the fields that should be order-stable: which opportunity
        # landed in which POSITION once both are sorted the same real way.
        regret_a = compute_decision_regret(conn_a)
        regret_b = compute_decision_regret(conn_b)
        positions_a = [(r.segment, r.chosen_family) for r in regret_a]
        positions_b = [(r.segment, r.chosen_family) for r in regret_b]
        assert positions_a == positions_b, (
            "compute_decision_regret's row ORDER must not depend on decision_id"
        )
    finally:
        conn_a.close()
        conn_b.close()
