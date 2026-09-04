"""Entry point: replays a full fixture run under a virtual clock.

One call per simulated day: detect -> arbitrate -> assign arms -> decide
every treatment opportunity -> resolve whatever outcome windows have closed.
The SAME `decide_one_opportunity` the live webhook path would call is used
here -- there is one decision path, exercised under two different clocks.

Nothing in this module ever imports `TRUTH` or `BASELINE` directly; it holds
one `OutcomeOracle` and calls `.resolve()` on it, exactly as a live merchant's
actual customers would "resolve" by either buying or not. That is what makes
this a blind run rather than a rigged one -- see harness/fixture.py.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

from harness.db import connect as hconnect
from harness.db import init_harness_db
from harness.fixture import OutcomeOracle, generate_population, seed_runtime_db, write_ground_truth
from revenew.clock import VirtualClock, iso
from revenew.db import connect as rconnect
from revenew.db import init_db
from revenew.decide import decide_one_opportunity
from revenew.decide.bandit import PosteriorStore
from revenew.decide.cassette import DEFAULT_CASSETTE_DIR, Cassette
from revenew.decide.generator import CandidateGenerator
from revenew.detect.detector import OpportunityDetector
from revenew.ledger.outcome import record_outcome
from revenew.models import ActionFamily, DecisionStatus, Segment
from revenew.route.arbiter import arbitrate, persist_winners
from revenew.settings import PolicyConfig

ATTRIBUTION_WINDOW_DAYS = 7
CENSOR_PROBABILITY = 0.03  # a fraction of windows close with no signal at all


def _stable_seed(*parts: str) -> int:
    """A deterministic 32-bit seed from string parts.

    NOT Python's built-in `hash()`: hash() on strings is salted per-process
    (PYTHONHASHSEED) specifically so untrusted input can't be used to mount a
    hash-flooding attack -- exactly the right default for a dict key, and
    exactly wrong for an RNG seed that a reproducibility claim depends on.
    The same (run_id, opportunity_id) must seed the bandit identically in
    every process that ever replays this fixture at this seed, forever.
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass
class PendingResolution:
    opportunity_id: str
    decision_id: str | None
    segment: Segment
    action_family: ActionFamily | None
    resolve_on_day: int


@dataclass
class ReplayResult:
    run_id: str
    n_days: int
    n_customers: int
    seed: int
    opportunities_detected: int = 0
    decisions_executed: int = 0
    decisions_no_action: int = 0
    outcomes_recorded: int = 0
    elapsed_seconds: float = 0.0
    no_action_reasons: dict = field(default_factory=dict)


def run_replay(
    *,
    seed: int = 20260101,
    n_customers: int = 3000,
    n_days: int = 30,
    control_pct: int = 20,
    policy: PolicyConfig | None = None,
    revenew_db_path: str = "revenew.db",
    harness_db_path: str = "harness.db",
    quiet: bool = True,
    llm_mode: str = "off",
    cassette_dir: str | None = None,
    strict_replay: bool = False,
) -> ReplayResult:
    t0 = time.perf_counter()
    if policy is None:
        # PolicyConfig's own default (Rs 50,000) is sized for one modest
        # campaign, not a multi-thousand-customer, multi-week simulation --
        # against this population it exhausted inside day 0 and every
        # following day ran almost entirely `budget_exhausted`, which
        # demonstrates the failure path (already covered by
        # test_budget_conservation.py) instead of the learning behaviour a
        # 30-day replay exists to show. Scaling the cap to the population
        # keeps budget from being the accidental bottleneck.
        policy = PolicyConfig(budget_cap=max(200_000.0, n_customers * 200.0))
    # Purely a function of `seed` -- no uuid4 suffix. A random suffix would
    # flow into opportunity_id (content-addressed from run_id, see
    # detect/detector.py) and from there into the bandit's RNG seed for every
    # decision, making two runs of the identical fixture at the identical
    # seed produce different decisions. Two run_replay() calls that must stay
    # distinct (e.g. parallel experiments) should pass different `seed`s.
    run_id = f"replay_{seed}"

    start = datetime(2026, 1, 1, tzinfo=UTC)
    clock = VirtualClock(start)

    init_harness_db(harness_db_path, reset=True)
    hconn = hconnect(harness_db_path)
    write_ground_truth(hconn, seed=seed)

    init_db(revenew_db_path, reset=True)
    conn = rconnect(revenew_db_path)
    data = generate_population(seed=seed, n_customers=n_customers, now=clock.now())
    seed_runtime_db(conn, data)

    store = PosteriorStore(conn)
    store.ensure_initialized()

    detector = OpportunityDetector()
    cassette = Cassette(cassette_dir) if cassette_dir else Cassette(DEFAULT_CASSETTE_DIR)
    generator = CandidateGenerator(mode=llm_mode, cassette=cassette, strict_replay=strict_replay)
    oracle = OutcomeOracle(seed=seed + 1)
    resolution_rng = np.random.default_rng(seed + 2)

    pending: list[PendingResolution] = []
    result = ReplayResult(run_id=run_id, n_days=n_days, n_customers=n_customers, seed=seed)
    no_action_counts: dict[str, int] = {}

    for day in range(n_days):
        window_id = f"day_{day:03d}"
        now = clock.now()

        raw = detector.detect(conn, run_id=run_id, window_id=window_id, now=now)
        detector.persist_candidates(conn, raw)
        winners = arbitrate(raw)
        arbitrated = persist_winners(conn, winners, now=now, salt=run_id, control_pct=control_pct)
        result.opportunities_detected += len(arbitrated)

        raw_by_id = {r.opportunity_id: r for r in raw}

        for opp in arbitrated:
            resolve_on = day + ATTRIBUTION_WINDOW_DAYS
            if opp.arm.value == "control":
                pending.append(
                    PendingResolution(opp.opportunity_id, None, Segment(opp.segment), None, resolve_on)
                )
                continue

            r = raw_by_id[opp.opportunity_id]
            decision = decide_one_opportunity(
                conn,
                opportunity_id=opp.opportunity_id,
                customer_id=opp.customer_id,
                segment=Segment(opp.segment),
                opportunity_type=r.opportunity_type,
                rupees_at_risk=r.rupees_at_risk,
                run_id=run_id,
                policy=policy,
                generator=generator,
                bandit_seed=_stable_seed(run_id, opp.opportunity_id),
                now=now,
            )
            if decision.status == DecisionStatus.EXECUTED:
                result.decisions_executed += 1
            else:
                result.decisions_no_action += 1
                reason = decision.no_action_reason.value if decision.no_action_reason else "unknown"
                no_action_counts[reason] = no_action_counts.get(reason, 0) + 1

            pending.append(
                PendingResolution(
                    opp.opportunity_id, decision.decision_id, Segment(opp.segment),
                    decision.action_family, resolve_on,
                )
            )

        due, pending = _partition(pending, day)
        for p in due:
            censored = bool(resolution_rng.random() < CENSOR_PROBABILITY)
            if censored:
                converted, revenue = False, 0.0
            else:
                converted, revenue = oracle.resolve(p.segment, p.action_family)
            record_outcome(
                conn,
                opportunity_id=p.opportunity_id,
                decision_id=p.decision_id,
                converted=converted,
                net_revenue=revenue,
                censored=censored,
                closed_at=iso(now),
            )
            result.outcomes_recorded += 1

        clock.advance(timedelta(days=1))
        if not quiet:
            print(f"  day {day:02d}: {len(arbitrated)} opportunities, {len(due)} outcomes closed", flush=True)

    # Resolve everything still pending at the end of the run, rather than
    # dropping it -- a replay that silently discards its last week of
    # attribution windows would understate n on every downstream metric.
    now = clock.now()
    for p in pending:
        censored = bool(resolution_rng.random() < CENSOR_PROBABILITY)
        converted, revenue = (False, 0.0) if censored else oracle.resolve(p.segment, p.action_family)
        record_outcome(
            conn, opportunity_id=p.opportunity_id, decision_id=p.decision_id,
            converted=converted, net_revenue=revenue, censored=censored, closed_at=iso(now),
        )
        result.outcomes_recorded += 1

    result.no_action_reasons = no_action_counts
    result.elapsed_seconds = time.perf_counter() - t0
    conn.close()
    hconn.close()
    return result


def _partition(pending: list[PendingResolution], today: int) -> tuple[list[PendingResolution], list[PendingResolution]]:
    due = [p for p in pending if p.resolve_on_day <= today]
    remaining = [p for p in pending if p.resolve_on_day > today]
    return due, remaining


if __name__ == "__main__":
    import argparse

    from harness.regret import compute_decision_regret, export_to_runtime, posterior_recovery_error
    from revenew.db import connect as rconnect

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=20260101)
    p.add_argument("--customers", type=int, default=3000)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--revenew-db", default="revenew.db")
    p.add_argument("--harness-db", default="harness.db")
    p.add_argument("--no-export", action="store_true", help="skip writing the regret curve into revenew.db")
    p.add_argument(
        "--llm", choices=["off", "record", "replay"], default="off",
        help="off (default): templated fallback only, matches every prior run. "
             "record: fill cassette misses with real API calls, keyed by cohort "
             "(opportunity_type, segment, rupees band, envelope fingerprint) -- "
             "~dozens of calls, not one per decision. replay: cassette only, "
             "never touches the API -- what a judge with no credential runs.",
    )
    p.add_argument(
        "--cassette-dir", default=None,
        help="defaults to cassettes/candidates/ at the repo root (see decide/cassette.py)",
    )
    p.add_argument(
        "--strict-replay", action="store_true",
        help="with --llm replay, raise on a cassette miss instead of falling back to a template",
    )
    args = p.parse_args()

    r = run_replay(
        seed=args.seed, n_customers=args.customers, n_days=args.days,
        revenew_db_path=args.revenew_db, harness_db_path=args.harness_db, quiet=False,
        llm_mode=args.llm, cassette_dir=args.cassette_dir, strict_replay=args.strict_replay,
    )
    print()
    print(f"run_id: {r.run_id}")
    print(f"opportunities detected: {r.opportunities_detected}")
    print(f"decisions: {r.decisions_executed} executed, {r.decisions_no_action} no_action")
    print(f"no_action reasons: {r.no_action_reasons}")
    print(f"outcomes recorded: {r.outcomes_recorded}")
    print(f"elapsed: {r.elapsed_seconds:.1f}s")

    if not args.no_export:
        # A fresh connection: run_replay() has already closed its own. This is
        # the one-way export described in harness/regret.py -- ground truth is
        # read here, in harness code, and only the resulting scalars cross
        # into revenew.db.
        conn = rconnect(args.revenew_db)
        regrets = compute_decision_regret(conn)
        recovery = posterior_recovery_error(conn)
        export_to_runtime(conn, run_id=r.run_id, regrets=regrets, recovery=recovery)
        conn.close()
        final_regret = regrets and sum(x.regret for x in regrets)
        print(f"regret curve exported: {len(regrets)} decisions, final cumulative regret {final_regret:.1f}" if regrets else "no decisions to export")
