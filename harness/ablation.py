"""Three-arm ablation: PLAN.md section 5. Proves "the deterministic layer
finds the money -> the agentic layer makes more of it" with a real measured
number, not a claim.

The obvious A/B (`--llm off` vs `--llm replay`) is rigged twice over:
`_template_fallback` (decide/generator.py) can never propose `BUNDLE_OFFER`
and always returns exactly one candidate, which collapses
`BanditScorer.choose()` to a propensity-1.0 no-op regardless of strategy --
the baseline's ACTIVE optimal-rate would be structurally zero, not a learning
result. The corrected arms:

    A. Deterministic -- one templated candidate (today's `llm_mode="off"`),
       scored GREEDILY (`strategy="greedy"`). Not "learning slowly" -- a
       constant policy: `_family_values` under `rng=None` scores every
       family at its posterior mean, the tie always breaks the same way on a
       cold store, and only one family is EVER chosen. Label it "always
       offer 12% off", not a degraded learner.
    B. + Bandit -- a fixed five-family template shelf (`llm_mode="shelf"`,
       decide/shelf.py), Thompson-sampled (`strategy="thompson"`). Isolates
       what LEARNING adds over a constant policy, with no LLM anywhere.
    C. Agentic -- the real, cassette-replayed LLM (`llm_mode="replay"`),
       Thompson-sampled. Isolates what the LLM adds over a fixed shelf.

Each arm runs into its OWN scratch database but at the IDENTICAL `seed` --
`run_id` inside `run_replay()` is `f"replay_{seed}"`, a pure function of
`seed` alone, so passing the same seed to all three arms already gives them
the identical run_id the paired design needs (it seeds `opportunity_id` via
detect/detector.py and, from there, every decision's bandit RNG stream via
`_stable_seed` in harness/run_replay.py) -- without threading a second
parameter through `run_replay()` just to duplicate what `seed` already pins
down.

Sequential, not parallel: each arm's scratch databases are deleted
immediately after its scalars are extracted, so peak disk usage is one run's
worth, not three (see `run_one_arm`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from harness.regret import best_constant_policy_floor, learning_curve
from harness.run_replay import run_replay
from revenew.db import connect as rconnect
from revenew.models import Segment

ARMS: tuple[str, ...] = ("A_deterministic", "B_bandit", "C_agentic")

_ARM_CONFIG: dict[str, dict] = {
    "A_deterministic": {
        "llm_mode": "off", "strategy": "greedy",
        "label": "A · Deterministic (one templated offer, constant policy)",
    },
    "B_bandit": {
        "llm_mode": "shelf", "strategy": "thompson",
        "label": "B · + Bandit (five-family shelf, Thompson sampling)",
    },
    "C_agentic": {
        "llm_mode": "replay", "strategy": "thompson",
        "label": "C · Agentic (LLM-composed candidates, Thompson sampling)",
    },
}

# Last 20% of a run's bandit-chosen decisions, in decision order -- "did it
# converge", not "how did it do on day one". Matches PLAN.md section 5's
# metric-discipline note.
CONVERGED_SLICE_FRACTION = 0.2


@dataclass(frozen=True)
class ArmResult:
    arm: str
    label: str
    n_customers: int
    n_days: int
    candidates_composed: int
    decisions_executed: int
    decisions_no_action: int
    optimal_rate_first: float
    optimal_rate_last: float
    regret_per_decision_first: float
    regret_per_decision_last: float
    best_constant_family: str
    best_constant_regret_per_decision: float
    explores: bool
    bundle_reachable: bool
    beats_best_constant: bool
    elapsed_seconds: float


def _scalar(conn, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row[0] is not None else 0


def _bundle_reachable(conn) -> bool:
    """Whether BUNDLE_OFFER ever appeared among GENERATED candidates -- not
    just chosen ones. Arm A's `_template_fallback` never proposes it at all;
    Arm B/C may or may not, depending on whether the affinity threshold
    (detect/detector.py's constants, reused by decide/shelf.py) was met."""
    return (
        _scalar(
            conn,
            "SELECT COUNT(*) FROM decision_candidates "
            "WHERE json_extract(candidate_json, '$.action_family') = 'bundle_offer'",
        )
        > 0
    )


def _converged_slice_segments(conn) -> list[Segment]:
    """Segments of the last `CONVERGED_SLICE_FRACTION` of BANDIT-CHOSEN
    decisions (status='executed', action_family set), in the same order and
    over the same filter `learning_curve()` uses -- this is the decision mix
    `best_constant_policy_floor` needs to be a fair floor for THIS arm, not
    the population's raw segment distribution."""
    rows = conn.execute(
        "SELECT segment FROM decisions "
        "WHERE status = 'executed' AND action_family IS NOT NULL "
        "ORDER BY created_at ASC, opportunity_id ASC"
    ).fetchall()
    if not rows:
        return []
    cutoff = int(len(rows) * (1 - CONVERGED_SLICE_FRACTION))
    return [Segment(r["segment"]) for r in rows[cutoff:]]


def run_one_arm(
    arm: str,
    *,
    seed: int,
    n_customers: int,
    n_days: int,
    scratch_dir: Path,
) -> ArmResult:
    if arm not in _ARM_CONFIG:
        raise ValueError(f"unknown arm {arm!r}, must be one of {ARMS}")
    config = _ARM_CONFIG[arm]

    revenew_db_path = scratch_dir / f"{arm}_revenew.db"
    harness_db_path = scratch_dir / f"{arm}_harness.db"

    t0 = time.perf_counter()
    run_replay(
        seed=seed, n_customers=n_customers, n_days=n_days,
        revenew_db_path=str(revenew_db_path), harness_db_path=str(harness_db_path),
        quiet=True, llm_mode=config["llm_mode"], strategy=config["strategy"],
        # Only Arm C touches the cassette -- a miss there must fail loudly,
        # exactly like `revenew demo`, rather than quietly degrade to a
        # single template and collapse back into looking like Arm A.
        strict_replay=(config["llm_mode"] == "replay"),
    )
    elapsed = time.perf_counter() - t0

    conn = rconnect(revenew_db_path)
    try:
        learning = learning_curve(conn, buckets=5)
        first = learning[0] if learning else {"optimal_rate": 0.0, "regret_per_decision": 0.0}
        last = learning[-1] if learning else {"optimal_rate": 0.0, "regret_per_decision": 0.0}

        converged_segments = _converged_slice_segments(conn)
        if converged_segments:
            best_family, best_regret = best_constant_policy_floor(converged_segments)
        else:
            best_family, best_regret = None, 0.0

        result = ArmResult(
            arm=arm,
            label=config["label"],
            n_customers=n_customers,
            n_days=n_days,
            candidates_composed=_scalar(conn, "SELECT COALESCE(SUM(candidates_generated), 0) FROM decisions"),
            decisions_executed=_scalar(conn, "SELECT COUNT(*) FROM decisions WHERE status = 'executed'"),
            decisions_no_action=_scalar(conn, "SELECT COUNT(*) FROM decisions WHERE status = 'no_action'"),
            optimal_rate_first=first["optimal_rate"],
            optimal_rate_last=last["optimal_rate"],
            regret_per_decision_first=first["regret_per_decision"],
            regret_per_decision_last=last["regret_per_decision"],
            best_constant_family=best_family.value if best_family else "n/a",
            best_constant_regret_per_decision=best_regret,
            explores=(config["strategy"] == "thompson"),
            bundle_reachable=_bundle_reachable(conn),
            beats_best_constant=(bool(converged_segments) and last["regret_per_decision"] < best_regret),
            elapsed_seconds=elapsed,
        )
    finally:
        conn.close()

    # Delete this arm's scratch databases now, before the next arm starts --
    # peak disk usage is one run's worth, not three. WAL/SHM sidecars too;
    # harness.db's own connection (inside run_replay) is already closed by
    # the time we get here.
    for path in (revenew_db_path, harness_db_path):
        for candidate in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            candidate.unlink(missing_ok=True)

    return result


def run_ablation(
    *,
    seed: int = 20260101,
    n_customers: int = 3000,
    n_days: int = 90,
    quiet: bool = False,
) -> list[ArmResult]:
    """Runs all three arms sequentially at the same `seed` (see module
    docstring for why that alone is enough to keep run_id, and therefore
    every opportunity_id and bandit seed, identical across arms) and returns
    one `ArmResult` per arm, in `ARMS` order. Writing the results into a
    target database's `demo_ablation_arm` table is the caller's job (see
    `export_to_runtime` below) -- this function only runs and measures.
    """
    results = []
    with TemporaryDirectory(prefix="revenew_ablation_") as tmp:
        scratch_dir = Path(tmp)
        for arm in ARMS:
            if not quiet:
                print(f"  {arm}: {n_customers} customers x {n_days} days...", flush=True)
            result = run_one_arm(arm, seed=seed, n_customers=n_customers, n_days=n_days, scratch_dir=scratch_dir)
            results.append(result)
            if not quiet:
                print(
                    f"    regret/decision {result.regret_per_decision_first:.1f} -> "
                    f"{result.regret_per_decision_last:.1f} "
                    f"(best-constant floor {result.best_constant_regret_per_decision:.1f}, "
                    f"beats it: {result.beats_best_constant}) -- "
                    f"explores={result.explores} bundle_reachable={result.bundle_reachable} "
                    f"[{result.elapsed_seconds:.1f}s]",
                    flush=True,
                )
    return results


def export_to_runtime(target_db_path: str, *, run_id: str, results: list[ArmResult]) -> None:
    """Writes the already-computed `ArmResult`s into `target_db_path`'s
    `demo_ablation_arm` table -- the one-way export, same discipline as
    `harness/regret.py`'s `export_to_runtime`: only scalars cross into the
    runtime database, and only this function (called from harness code, never
    from revenew/) writes this table.

    `target_db_path` must already carry the CURRENT schema (db/schema.sql) --
    `demo_ablation_arm` is a new table, and this project keeps no migrations
    (see PLAN.md section 3): a stale `revenew.db` predating this table raises
    `sqlite3.OperationalError: no such table` here, loudly, rather than
    silently doing nothing.
    """
    conn = rconnect(target_db_path)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO demo_ablation_arm
                (run_id, arm, label, n_customers, n_days, candidates_composed,
                 decisions_executed, decisions_no_action, optimal_rate_first, optimal_rate_last,
                 regret_per_decision_first, regret_per_decision_last, best_constant_family,
                 best_constant_regret_per_decision, explores, bundle_reachable, beats_best_constant,
                 elapsed_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id, r.arm, r.label, r.n_customers, r.n_days, r.candidates_composed,
                    r.decisions_executed, r.decisions_no_action, r.optimal_rate_first, r.optimal_rate_last,
                    r.regret_per_decision_first, r.regret_per_decision_last, r.best_constant_family,
                    r.best_constant_regret_per_decision, int(r.explores), int(r.bundle_reachable),
                    int(r.beats_best_constant), r.elapsed_seconds,
                )
                for r in results
            ],
        )
        conn.commit()
    finally:
        conn.close()
