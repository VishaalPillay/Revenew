"""`revenew`: the operator CLI. Every subcommand takes its own `--db` (or
`--revenew-db`/`--harness-db` where a replay needs both), so which database a
command touches is always an explicit argument, never an accident of which
directory the shell happened to be in -- the exact cwd-relative-path mismatch
`run_replay.py`'s bare `revenew.db` default has always been vulnerable to
(see ENGINEERING_LOG.md).

Thin wiring only. Every subcommand calls the same functions the tests and
the API already call -- `init_db`, `run_and_report`, `build_report`,
`get_decision_trace`, `PosteriorStore`, `reconcile` -- nothing here is a
second implementation of any of them.

**On the one import that crosses the isolation boundary.** `demo` and
`replay` import `harness.run_replay`, and every other module under
`revenew/` is forbidden from touching `harness/` (see db/schema.sql: the
runtime process opens revenew.db and nothing else). That claim is about the
RUNTIME DECISION PATH, and it still holds exactly: `decide_one_opportunity`
and everything it reaches never import harness, and `revenew serve` never
does either. This module is an operator multi-tool, not the runtime, and the
two subcommands that drive an evaluation import it lazily, inside the
function body, so merely importing `revenew.cli` does not pull the harness
in. Keeping the boundary meaningful means being precise about where it is,
not pretending a CLI that can launch an evaluation doesn't exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

# The parameters the committed cassette was recorded against. Defined once so
# the argparse defaults and the cassette-miss message cannot drift apart --
# a help text that names different numbers than the code uses is worse than
# no help text.
DEMO_CUSTOMERS = 3000
# 90 days, not 30. Outcomes close on a 7-day attribution lag, so in a 30-day
# run most feedback arrives at or after the last decision -- the bandit ends
# with posteriors that are correct (replayed, they pick the truth-optimal
# action 72% of the time) while the decisions the run actually made never got
# to use them. At 90 days the loop closes inside the run: truth-optimal picks
# go 10.5% -> 76.1% and regret per decision falls from 75.5 to 4.0. Nothing
# about the bandit changed; the horizon did.
DEMO_DAYS = 90


def _cmd_init(args: argparse.Namespace) -> None:
    from revenew.db import init_db

    init_db(args.db, reset=args.reset)
    print(f"initialized {args.db}" + (" (reset)" if args.reset else ""))


def _cmd_replay(args: argparse.Namespace) -> None:
    from harness.run_replay import run_and_report

    run_and_report(
        seed=args.seed, n_customers=args.customers, n_days=args.days,
        revenew_db_path=args.revenew_db, harness_db_path=args.harness_db,
        llm_mode=args.llm, cassette_dir=args.cassette_dir, strict_replay=args.strict_replay,
        strategy=args.strategy, export=not args.no_export,
    )


def _cmd_demo(args: argparse.Namespace) -> None:
    """One command, known-good state, nothing to mistype on camera.

    Two defaults here are deliberate and load-bearing:

    `--llm replay` -- never `record`. A demo must not depend on a network
    call, a credential, or a rate limit, and the committed cassette means it
    doesn't: the candidates are real model output, recorded once, replayed
    byte-identically by anyone who clones the repo.

    `strict_replay=True` -- a cassette MISS raises instead of silently falling
    back to a single templated candidate. That fallback is exactly how this
    project twice ended up measuring a greedy argmax while believing it was
    measuring a bandit (ENGINEERING_LOG.md #11, and again when a policy tweak
    silently changed every cache key). On a demo run, a miss must stop the
    show loudly rather than quietly produce a worse-but-plausible number.
    """
    from harness.run_replay import run_and_report
    from revenew.decide.generator import CassetteMissError

    print(f"Revenew demo -- replaying a {args.days}-day fixture from the committed cassette.")
    print("No API key required: the LLM candidates were recorded once and are replayed exactly.\n")

    try:
        run_and_report(
            seed=args.seed, n_customers=args.customers, n_days=args.days,
            revenew_db_path=args.db, harness_db_path=args.harness_db,
            llm_mode="replay", cassette_dir=args.cassette_dir, strict_replay=True,
            export=True,
        )
    except CassetteMissError as exc:
        # The cassette is recorded against the DEFAULT demo parameters. Change
        # the population or the horizon and the run reaches cohorts nobody
        # recorded -- which strict mode correctly refuses to paper over, but a
        # raw traceback makes a working system look broken. Say what happened
        # and what to do about it.
        print(f"\n  Cassette miss: {exc}\n", file=sys.stderr)
        print(
            "  The committed cassette covers the default demo "
            f"(--customers {DEMO_CUSTOMERS} --days {DEMO_DAYS}). "
            "Other parameters reach\n"
            "  cohorts that were never recorded. Either run the defaults:\n"
            "\n      revenew demo\n\n"
            "  or record the missing cohorts first (needs GROQ_API_KEY):\n"
            "\n      revenew replay --llm record "
            f"--customers {args.customers} --days {args.days}\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    print()
    print("=" * 66)
    print(f"  Dashboard:  http://127.0.0.1:{args.port}")
    print(f"  Start it:   revenew serve --db {args.db} --port {args.port}")
    print("=" * 66)


def _cmd_ablation(args: argparse.Namespace) -> None:
    """Runs PLAN.md section 5's three-arm ablation and writes the result into
    `--db`'s `demo_ablation_arm` table. `--db` must already carry the CURRENT
    schema (`demo_ablation_arm` is a new table; this project keeps no
    migrations -- see PLAN.md section 3) -- reinitialize it first
    (`revenew init --db ... --reset`) or regenerate it via `revenew replay`
    if it predates this command.
    """
    import sqlite3

    from harness.ablation import export_to_runtime, run_ablation
    from revenew.db import connect

    conn = connect(args.db)
    try:
        conn.execute("SELECT 1 FROM demo_ablation_arm LIMIT 1")
    except sqlite3.OperationalError:
        print(
            f"  {args.db} does not have the demo_ablation_arm table yet -- its schema predates "
            "this command.\n  Reinitialize it (revenew init --db "
            f"{args.db} --reset) or regenerate it via `revenew replay` first.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    finally:
        conn.close()

    print(f"Revenew ablation -- {args.customers} customers x {args.days} days, 3 arms, seed {args.seed}.\n")
    results = run_ablation(seed=args.seed, n_customers=args.customers, n_days=args.days, quiet=False)
    export_to_runtime(args.db, run_id=f"ablation_{args.seed}", results=results)

    print()
    print(f"{'arm':10} {'regret/dec (first->last)':28} {'floor':>8} {'beats floor':>12} {'explores':>9} {'bundle':>7}")
    for r in results:
        curve = f"{r.regret_per_decision_first:7.1f} -> {r.regret_per_decision_last:7.1f}"
        print(
            f"{r.arm:10} {curve:28} {r.best_constant_regret_per_decision:8.1f} "
            f"{str(r.beats_best_constant):>12} {str(r.explores):>9} {str(r.bundle_reachable):>7}"
        )
    print(f"\nWritten to {args.db}: demo_ablation_arm (run_id=ablation_{args.seed})")


def _cmd_report(args: argparse.Namespace) -> None:
    from revenew.db import connect
    from revenew.measure.report import build_report

    conn = connect(args.db)
    r = build_report(conn)
    conn.close()

    if args.json:
        print(json.dumps(_report_to_dict(r), indent=2))
        return

    print(f"run_id: {r.run_id or '(none exported yet)'}")
    print()
    print("Incremental lift, pooled:")
    print(
        f"  {r.overall.lift:+.1f} / decision  95% CI [{r.overall.ci_low:.1f}, {r.overall.ci_high:.1f}]  "
        f"n_t={r.overall.n_treatment} n_c={r.overall.n_control}  "
        f"{'significant' if r.overall.is_significant else 'not yet significant'}"
    )
    print()
    print("Lift by segment:")
    for lift in r.lifts:
        seg = lift.segment.value if lift.segment else "?"
        print(f"  {seg:10} {lift.lift:+8.1f}  CI [{lift.ci_low:8.1f}, {lift.ci_high:8.1f}]  n_t={lift.n_treatment:5} n_c={lift.n_control:5}")
    print()
    cv = r.candidate_validity
    if cv.policy_compliance_rate is not None:
        print(f"LLM policy compliance: {cv.policy_compliance_rate:.2%} "
              f"({cv.policy_violations:,} illegal offers proposed out of {cv.total_generated:,})")
        print(f"  eligibility-blocked (cooldown / monthly cap, not a model failure): "
              f"{cv.eligibility_blocked:,}")
    else:
        print("LLM policy compliance: no decisions yet")
    print(f"Budget consumed: {r.budget_consumed:.1f}")
    if r.no_action_reasons:
        print("No-action reasons:")
        for row in r.no_action_reasons:
            print(f"  {row['no_action_reason']}: {row['n']}")
    if r.regret_curve:
        print(f"Regret (bandit decisions only): {len(r.regret_curve)} points, "
              f"final cumulative {r.regret_curve[-1]['cumulative_regret']:,.0f}")
    if r.regret_curve_all:
        print(f"Regret (all decisions, incl. envelope-forced): {len(r.regret_curve_all)} points, "
              f"final cumulative {r.regret_curve_all[-1]['cumulative_regret']:,.0f}")
    if r.learning_curve:
        first, last = r.learning_curve[0], r.learning_curve[-1]
        print()
        print("Did it learn? (share of decisions landing on the truth-optimal action; chance = 20%)")
        print(f"  first slice: {first['optimal_rate']:6.1%}   regret/decision {first['regret_per_decision']:6.1f}")
        print(f"  last  slice: {last['optimal_rate']:6.1%}   regret/decision {last['regret_per_decision']:6.1f}")


def _report_to_dict(r) -> dict:
    from revenew.measure.report import lift_to_dict

    return {
        "run_id": r.run_id,
        "overall": lift_to_dict(r.overall),
        "lifts": [lift_to_dict(lift) for lift in r.lifts],
        "no_action_reasons": r.no_action_reasons,
        "candidate_validity": {
            "validity_rate": r.candidate_validity.validity_rate,
            "total_generated": r.candidate_validity.total_generated,
            "total_valid": r.candidate_validity.total_valid,
            "policy_violations": r.candidate_validity.policy_violations,
            "eligibility_blocked": r.candidate_validity.eligibility_blocked,
            "policy_compliance_rate": r.candidate_validity.policy_compliance_rate,
        },
        "budget_consumed": r.budget_consumed,
        "regret_curve": r.regret_curve,
        "regret_curve_all": r.regret_curve_all,
        "learning_curve": r.learning_curve,
        "posterior_recovery": r.posterior_recovery,
    }


def _cmd_trace(args: argparse.Namespace) -> None:
    from revenew.db import connect
    from revenew.measure.report import get_decision_trace

    conn = connect(args.db)
    trace = get_decision_trace(conn, args.decision_id)
    conn.close()

    if trace is None:
        print(f"no decision with id {args.decision_id!r}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(trace, indent=2))


def _cmd_posteriors(args: argparse.Namespace) -> None:
    from revenew.db import connect
    from revenew.decide.bandit import PosteriorStore

    conn = connect(args.db)
    rows = PosteriorStore(conn).get_all()
    conn.close()

    print(f"{'segment':10} {'family':18} {'alpha':>8} {'beta':>8} {'n_obs':>7} {'mean_rev':>10}")
    for row in rows:
        mean_rev = f"{row.mean_revenue:.1f}" if row.mean_revenue is not None else "-"
        print(f"{row.segment.value:10} {row.action_family.value:18} {row.alpha:8.1f} {row.beta:8.1f} "
              f"{row.n_observed:7.0f} {mean_rev:>10}")


def _cmd_reconcile(args: argparse.Namespace) -> None:
    from revenew.db import connect
    from revenew.ledger.reconcile import reconcile

    conn = connect(args.db)
    result = reconcile(conn, now=datetime.now(UTC), timeout_minutes=args.timeout_minutes)
    conn.close()
    print(f"fixed forward: {result.fixed_forward}")
    print(f"released (never executed): {result.released}")
    print(f"released_total: {result.released_total:.1f}")


def _cmd_mcp(args: argparse.Namespace) -> None:
    from revenew.agent.mcp_server import run_stdio_server

    run_stdio_server(args.db)


def _cmd_serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    if args.db:
        # See revenew/api/webhooks.py's get_conn(): this is the one place a
        # CLI flag reaches FastAPI's dependency-injected connection, since
        # the dependency itself takes no arguments uvicorn could pass through.
        os.environ["REVENEW_DB_PATH"] = str(args.db)
    uvicorn.run("revenew.api.dashboard:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="revenew", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    p_mcp = sub.add_parser("mcp", help="run the MCP stdio server for agent commerce")
    p_mcp.add_argument("--db", default="revenew.db")
    p_mcp.set_defaults(func=_cmd_mcp)

    p_init = sub.add_parser("init", help="create the schema in a fresh revenew.db")
    p_init.add_argument("--db", default="revenew.db")
    p_init.add_argument("--reset", action="store_true", help="delete an existing file first")
    p_init.set_defaults(func=_cmd_init)

    p_replay = sub.add_parser("replay", help="run a full fixture replay under the virtual clock")
    p_replay.add_argument("--seed", type=int, default=20260101)
    p_replay.add_argument("--customers", type=int, default=3000)
    p_replay.add_argument("--days", type=int, default=30)
    p_replay.add_argument("--revenew-db", default="revenew.db")
    p_replay.add_argument("--harness-db", default="harness.db")
    p_replay.add_argument("--llm", choices=["off", "record", "replay", "shelf"], default="off")
    p_replay.add_argument("--cassette-dir", default=None)
    p_replay.add_argument("--strict-replay", action="store_true")
    p_replay.add_argument("--strategy", choices=["thompson", "greedy"], default="thompson")
    p_replay.add_argument("--no-export", action="store_true")
    p_replay.set_defaults(func=_cmd_replay)

    p_demo = sub.add_parser(
        "demo",
        help="one-command demo: replay the committed cassette, export regret, print the dashboard URL",
    )
    p_demo.add_argument("--seed", type=int, default=20260101)
    p_demo.add_argument("--customers", type=int, default=DEMO_CUSTOMERS)
    p_demo.add_argument("--days", type=int, default=DEMO_DAYS)
    p_demo.add_argument("--db", default="revenew.db")
    p_demo.add_argument("--harness-db", default="harness.db")
    p_demo.add_argument("--cassette-dir", default=None)
    p_demo.add_argument("--port", type=int, default=8000)
    p_demo.set_defaults(func=_cmd_demo)

    p_ablation = sub.add_parser(
        "ablation",
        help="run the three-arm ablation (deterministic / bandit / agentic) and write demo_ablation_arm",
    )
    p_ablation.add_argument("--seed", type=int, default=20260101)
    p_ablation.add_argument("--customers", type=int, default=DEMO_CUSTOMERS)
    p_ablation.add_argument("--days", type=int, default=DEMO_DAYS)
    p_ablation.add_argument("--db", default="revenew.db", help="target database demo_ablation_arm is written into")
    p_ablation.set_defaults(func=_cmd_ablation)

    p_report = sub.add_parser("report", help="print the measurement report (lift, validity, budget, regret)")
    p_report.add_argument("--db", default="revenew.db")
    p_report.add_argument("--json", action="store_true", help="machine-readable output")
    p_report.set_defaults(func=_cmd_report)

    p_trace = sub.add_parser("trace", help="print one decision's full audit trail as JSON")
    p_trace.add_argument("decision_id")
    p_trace.add_argument("--db", default="revenew.db")
    p_trace.set_defaults(func=_cmd_trace)

    p_post = sub.add_parser("posteriors", help="print the (segment, action_family) posterior grid")
    p_post.add_argument("--db", default="revenew.db")
    p_post.set_defaults(func=_cmd_posteriors)

    p_recon = sub.add_parser("reconcile", help="sweep stale pending decisions, releasing or fixing forward")
    p_recon.add_argument("--db", default="revenew.db")
    p_recon.add_argument("--timeout-minutes", type=int, default=30)
    p_recon.set_defaults(func=_cmd_reconcile)

    p_serve = sub.add_parser("serve", help="run the FastAPI app (dashboard + webhooks + read API)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--db", default=None, help="defaults to revenew.db next to the package root")
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
