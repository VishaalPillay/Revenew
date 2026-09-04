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
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime


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
        export=not args.no_export,
    )


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
    if r.candidate_validity.validity_rate is not None:
        print(f"Candidate validity: {r.candidate_validity.validity_rate:.0%} "
              f"({r.candidate_validity.total_valid}/{r.candidate_validity.total_generated})")
    else:
        print("Candidate validity: no decisions yet")
    print(f"Budget consumed: {r.budget_consumed:.1f}")
    if r.no_action_reasons:
        print("No-action reasons:")
        for row in r.no_action_reasons:
            print(f"  {row['no_action_reason']}: {row['n']}")
    if r.regret_curve:
        print(f"Regret curve: {len(r.regret_curve)} points, final cumulative regret "
              f"{r.regret_curve[-1]['cumulative_regret']:.1f}")


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
        },
        "budget_consumed": r.budget_consumed,
        "regret_curve": r.regret_curve,
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
    p_replay.add_argument("--llm", choices=["off", "record", "replay"], default="off")
    p_replay.add_argument("--cassette-dir", default=None)
    p_replay.add_argument("--strict-replay", action="store_true")
    p_replay.add_argument("--no-export", action="store_true")
    p_replay.set_defaults(func=_cmd_replay)

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
