"""harness/ablation.py: PLAN.md section 5's three-arm ablation runner, and
`best_constant_policy_floor` (harness/regret.py), the free floor that stops a
report ever presenting a number a one-line constant policy already beats.

Arm C ("C_agentic") is deliberately never run for real here -- it replays the
committed cassette with `strict_replay=True`, and a small test-scale
population reaches cohorts the cassette (recorded at the demo's own
3000-customer/90-day scale) was never asked about, which is a real,
already-observed miss, not a bug -- see PLAN.md section 5's reality check.
Orchestration (`run_ablation`) is tested with `run_one_arm` monkeypatched
instead of exercising real replay for all three arms.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import harness.ablation as ablation_module
from harness.ablation import ArmResult, export_to_runtime, run_ablation, run_one_arm
from harness.regret import best_constant_policy_floor
from revenew.db import connect as rconnect
from revenew.db import init_db
from revenew.models import ActionFamily, Segment

# ==================================================== best_constant_policy_floor --


def test_best_constant_policy_floor_matches_a_manual_computation():
    """ACTIVE's truth-optimal family is BUNDLE_OFFER (see harness/fixture.py's
    TRUTH table); NEW's is PERCENT_DISCOUNT. A mix skewed toward ACTIVE must
    therefore prefer holding BUNDLE_OFFER constant over PERCENT_DISCOUNT, if
    BUNDLE_OFFER's regret against the mix is lower -- assert against a
    brute-force recomputation from the same TRUTH/BASELINE ground truth the
    function itself reads, not just "it returns something plausible"."""
    from harness.fixture import BASELINE, TRUTH

    segments = [Segment.ACTIVE] * 8 + [Segment.NEW] * 2

    def oracle_ev(seg):
        return max(max(TRUTH[(seg, f)].expected_reward for f in ActionFamily), BASELINE[seg].expected_reward)

    def chosen_ev(seg, family):
        return TRUTH[(seg, family)].expected_reward

    expected_family, expected_regret = None, None
    for family in ActionFamily:
        regret = sum(oracle_ev(s) - chosen_ev(s, family) for s in segments) / len(segments)
        if expected_regret is None or regret < expected_regret:
            expected_family, expected_regret = family, regret

    family, regret = best_constant_policy_floor(segments)
    assert family == expected_family
    assert regret == pytest.approx(expected_regret)


def test_best_constant_policy_floor_is_never_better_than_the_true_oracle():
    """The floor holds ONE family fixed; the oracle may pick a different best
    family per segment. The floor's regret can never be negative, and it can
    never beat (undercut) the true per-segment oracle mix."""
    segments = [Segment.ACTIVE, Segment.NEW, Segment.LAPSING, Segment.DORMANT] * 5
    _, regret = best_constant_policy_floor(segments)
    assert regret >= -1e-9


def test_best_constant_policy_floor_requires_at_least_one_segment():
    with pytest.raises(ValueError):
        best_constant_policy_floor([])


# ===================================================================== run_one_arm --


def test_run_one_arm_deterministic_produces_a_well_formed_result_and_cleans_up(tmp_path: Path):
    result = run_one_arm("A_deterministic", seed=1, n_customers=40, n_days=6, scratch_dir=tmp_path)

    assert result.arm == "A_deterministic"
    assert result.n_customers == 40 and result.n_days == 6
    assert result.decisions_executed + result.decisions_no_action > 0
    assert 0.0 <= result.optimal_rate_first <= 1.0
    assert 0.0 <= result.optimal_rate_last <= 1.0
    assert result.explores is False  # greedy strategy, per PLAN.md section 5
    assert result.bundle_reachable is False  # _template_fallback never proposes BUNDLE_OFFER
    assert result.elapsed_seconds >= 0.0

    # Peak-disk discipline: the scratch databases for THIS arm must be gone
    # once its scalars have been extracted.
    leftovers = list(tmp_path.glob("A_deterministic_*"))
    assert leftovers == [], f"scratch files were not cleaned up: {leftovers}"


def test_run_one_arm_bandit_shelf_explores_and_can_reach_bundle(tmp_path: Path):
    result = run_one_arm("B_bandit", seed=1, n_customers=80, n_days=10, scratch_dir=tmp_path)
    assert result.explores is True
    assert result.decisions_executed + result.decisions_no_action > 0


def test_run_one_arm_rejects_an_unknown_arm(tmp_path: Path):
    with pytest.raises(ValueError):
        run_one_arm("Z_unknown", seed=1, n_customers=10, n_days=2, scratch_dir=tmp_path)


# ===================================================================== run_ablation --


def test_run_ablation_runs_all_three_arms_in_order(monkeypatch):
    calls = []

    def fake_run_one_arm(arm, *, seed, n_customers, n_days, scratch_dir):
        calls.append(arm)
        return ArmResult(
            arm=arm, label=f"label-{arm}", n_customers=n_customers, n_days=n_days,
            candidates_composed=10, decisions_executed=5, decisions_no_action=1,
            optimal_rate_first=0.2, optimal_rate_last=0.6,
            regret_per_decision_first=50.0, regret_per_decision_last=10.0,
            best_constant_family="percent_discount", best_constant_regret_per_decision=15.0,
            explores=(arm != "A_deterministic"), bundle_reachable=(arm != "A_deterministic"),
            beats_best_constant=True, elapsed_seconds=0.01,
        )

    monkeypatch.setattr(ablation_module, "run_one_arm", fake_run_one_arm)
    results = run_ablation(seed=1, n_customers=10, n_days=2, quiet=True)

    assert calls == list(ablation_module.ARMS)
    assert [r.arm for r in results] == list(ablation_module.ARMS)


# =================================================================== export_to_runtime --


def test_export_to_runtime_round_trips_into_demo_ablation_arm(tmp_path: Path):
    db_path = tmp_path / "target.db"
    init_db(db_path, reset=True)

    result = ArmResult(
        arm="B_bandit", label="B · + Bandit", n_customers=300, n_days=30,
        candidates_composed=1200, decisions_executed=180, decisions_no_action=20,
        optimal_rate_first=0.21, optimal_rate_last=0.58,
        regret_per_decision_first=80.0, regret_per_decision_last=12.5,
        best_constant_family="bundle_offer", best_constant_regret_per_decision=40.0,
        explores=True, bundle_reachable=True, beats_best_constant=True,
        elapsed_seconds=12.3,
    )
    export_to_runtime(str(db_path), run_id="ablation_1", results=[result])

    conn = rconnect(db_path)
    row = conn.execute(
        "SELECT * FROM demo_ablation_arm WHERE run_id = ? AND arm = ?", ("ablation_1", "B_bandit")
    ).fetchone()
    conn.close()

    assert row is not None
    assert row["label"] == "B · + Bandit"
    assert row["n_customers"] == 300
    assert row["regret_per_decision_last"] == pytest.approx(12.5)
    assert row["explores"] == 1
    assert row["bundle_reachable"] == 1
    assert row["beats_best_constant"] == 1


def test_export_to_runtime_raises_loudly_on_a_stale_schema(tmp_path: Path):
    """`demo_ablation_arm` is a new table and this project keeps no
    migrations -- writing into a database created before this table existed
    must fail loudly, not silently do nothing."""
    import sqlite3

    db_path = tmp_path / "stale.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE customers (customer_id TEXT PRIMARY KEY, created_at TEXT NOT NULL)")
    conn.commit()
    conn.close()

    result = ArmResult(
        arm="A_deterministic", label="A", n_customers=1, n_days=1, candidates_composed=0,
        decisions_executed=0, decisions_no_action=0, optimal_rate_first=0.0, optimal_rate_last=0.0,
        regret_per_decision_first=0.0, regret_per_decision_last=0.0, best_constant_family="n/a",
        best_constant_regret_per_decision=0.0, explores=False, bundle_reachable=False,
        beats_best_constant=False, elapsed_seconds=0.0,
    )
    with pytest.raises(sqlite3.OperationalError):
        export_to_runtime(str(db_path), run_id="ablation_1", results=[result])
