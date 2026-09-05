"""RegretCalculator: chosen EV against oracle EV, using ground truth the
runtime never sees.

Per-decision regret is computed in Python, not as a single opaque SQL
expression -- the "chosen" side needs a different ground-truth table for an
executed decision (`ground_truth`, keyed by action_family) than a no_action
one (`ground_truth_baseline`, since doing nothing gets you the organic rate),
and getting that branch wrong silently would produce a plausible-looking
regret curve for the wrong reason. Keeping it in Python keeps it testable
directly against hand-built fixtures, the way tests/ already tests everything
else in this codebase.

`v_cumulative_regret` and `v_posterior_recovery`, named in SYSTEM_DESIGN.md
section 8, ARE created here as SQL views -- but as TEMP views on a connection
that has ATTACHed revenew.db, per the note in db/harness_schema.sql. They
exist for ad hoc inspection (`sqlite3 harness.db` after calling
`attach_and_create_views`); the numbers this module actually returns to
callers come from the Python path above, which is what is tested.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from harness.fixture import BASELINE, TRUTH
from revenew.models import ActionFamily, Segment


@dataclass(frozen=True)
class DecisionRegret:
    decision_id: str
    created_at: str
    segment: Segment
    chosen_family: ActionFamily | None  # None means the decision was no_action
    chosen_true_ev: float
    oracle_true_ev: float

    @property
    def regret(self) -> float:
        return self.oracle_true_ev - self.chosen_true_ev


def _oracle_ev(segment: Segment) -> float:
    best = max(TRUTH[(segment, f)].expected_reward for f in ActionFamily)
    return max(best, BASELINE[segment].expected_reward)


def _chosen_ev(segment: Segment, family: ActionFamily | None) -> float:
    if family is None:
        return BASELINE[segment].expected_reward
    return TRUTH[(segment, family)].expected_reward


def _truth_optimal_family(segment: Segment) -> ActionFamily:
    return max(ActionFamily, key=lambda f: TRUTH[(segment, f)].expected_reward)


def learning_curve(conn: sqlite3.Connection, *, buckets: int = 12) -> list[dict]:
    """How often the bandit picked the TRUTH-OPTIMAL action, over time.

    Cumulative regret is the headline metric, but it is a rupee figure whose
    slope is easy to misread and whose level is dominated by how many
    decisions there were. This is the same question asked directly: of the
    decisions the bandit actually made in this slice of the run, what share
    landed on the action that ground truth says is best for that segment?
    Chance is 20% (five families); a system that is not learning sits there.

    Harness-side by construction -- it reads `TRUTH`, which is exactly what
    the runtime is never allowed to see. Like the regret curve, only the
    resulting scalars cross into revenew.db.
    """
    rows = conn.execute(
        "SELECT segment, action_family FROM decisions "
        "WHERE status = 'executed' AND action_family IS NOT NULL "
        # opportunity_id, not decision_id, breaks a same-`created_at` tie --
        # decision_id is `str(uuid.uuid4())` (decide/__init__.py), a fresh
        # random value every run, so ordering by it means two runs of the
        # SAME seed can bucket a handful of same-timestamp decisions
        # differently even though every decision's own content is identical.
        # opportunity_id is content-addressed (detect/detector.py), so it
        # gives the same total order on every run at the same seed -- this
        # is what "byte-identical" needs to mean for a BUCKETED curve, not
        # just for the posteriors the existing replay-equality test checks.
        "ORDER BY created_at ASC, opportunity_id ASC"
    ).fetchall()
    if not rows:
        return []

    # Even boundaries, not a fixed stride. `range(0, n, n // buckets)` leaves a
    # ragged remainder as its own bucket -- on one run that was a final bucket
    # of SEVEN decisions reporting "100% optimal", which is both true and
    # completely useless as a headline. Sizes here differ by at most one.
    n = len(rows)
    bounds = [(i * n) // buckets for i in range(buckets + 1)]
    out = []
    # Pairwise over consecutive boundaries, so the offset slice is one shorter
    # by construction -- not a strict-zip case.
    for lo, hi in zip(bounds, bounds[1:]):  # noqa: B905
        chunk = rows[lo:hi]
        if not chunk:
            continue
        i = lo
        hits = sum(
            1 for r in chunk
            if ActionFamily(r["action_family"]) is _truth_optimal_family(Segment(r["segment"]))
        )
        chosen_ev = sum(_chosen_ev(Segment(r["segment"]), ActionFamily(r["action_family"])) for r in chunk)
        oracle_ev = sum(_oracle_ev(Segment(r["segment"])) for r in chunk)
        out.append({
            "decision_index": i + len(chunk),
            "n": len(chunk),
            "optimal_rate": hits / len(chunk),
            "regret_per_decision": (oracle_ev - chosen_ev) / len(chunk),
        })
    return out


def compute_decision_regret(conn: sqlite3.Connection) -> list[DecisionRegret]:
    """One row per treatment-arm decision (executed or no_action), in the
    order they were made. Control-arm opportunities never reach `decisions`
    at all, so they contribute nothing here -- regret is about the quality of
    decisions actually taken, and the control arm by definition took none."""
    rows = conn.execute(
        "SELECT decision_id, created_at, segment, action_family, status "
        "FROM decisions WHERE status IN ('executed', 'no_action') "
        # opportunity_id, not decision_id -- see learning_curve()'s comment
        # above for why the tie-break must be the content-addressed id, not
        # the random one, for this ordering to be reproducible run to run.
        "ORDER BY created_at ASC, opportunity_id ASC"
    ).fetchall()

    out = []
    for r in rows:
        segment = Segment(r["segment"])
        family = ActionFamily(r["action_family"]) if r["action_family"] else None
        out.append(
            DecisionRegret(
                decision_id=r["decision_id"],
                created_at=r["created_at"],
                segment=segment,
                chosen_family=family,
                chosen_true_ev=_chosen_ev(segment, family),
                oracle_true_ev=_oracle_ev(segment),
            )
        )
    return out


def best_constant_policy_floor(segments: list[Segment]) -> tuple[ActionFamily, float]:
    """The best a single FIXED action family -- applied to every decision in
    `segments`, chosen once and never adapted -- could have scored against
    ground truth over that exact segment mix. `(best_family, regret_per_decision)`.

    This is PLAN.md section 5's "best-constant-policy floor": a pure function
    of `TRUTH`/`BASELINE` and an arm's own OBSERVED decision mix (how many of
    its decisions fell in each segment), free to compute because it needs no
    learning and no per-segment specialization -- just picking the single
    best-on-average family and holding it fixed. It exists to stop a report
    ever presenting a number that a one-line "always offer X" policy already
    beats for free; the measured example that motivated it: a constant
    "always 12% off" policy scores 58.6% cumulative optimal-rate against a
    learned bandit's 45.7% (the bandit is still winning on the metric that
    matters -- converged regret/decision -- but publishing the cumulative
    number alone would hand a reader exactly the wrong comparison).

    Deliberately over the SAME segments an arm actually decided on, not the
    full segment distribution of the population -- different arms drain
    ACTIVE into LAPSING/DORMANT at different rates depending what they chose,
    so the fair floor for arm X is "the best constant GIVEN X's own mix", not
    a population-wide constant every arm is held to identically.
    """
    if not segments:
        raise ValueError("best_constant_policy_floor() requires at least one segment")
    oracle_total = sum(_oracle_ev(s) for s in segments)
    best_family, best_regret = None, None
    for family in ActionFamily:
        chosen_total = sum(_chosen_ev(s, family) for s in segments)
        regret = (oracle_total - chosen_total) / len(segments)
        if best_regret is None or regret < best_regret:
            best_family, best_regret = family, regret
    return best_family, best_regret


def cumulative_regret_curve(regrets: list[DecisionRegret]) -> list[tuple[int, float]]:
    """(decision_index, cumulative_regret) pairs, in decision order -- the
    series the dashboard's regret chart plots. A learning bandit's curve
    should visibly flatten as posteriors sharpen; a flat baseline (every
    decision equally wrong) would instead climb as a straight line."""
    out = []
    total = 0.0
    for i, r in enumerate(regrets, start=1):
        total += r.regret
        out.append((i, total))
    return out


@dataclass(frozen=True)
class CellRecovery:
    segment: Segment
    action_family: ActionFamily
    n_observed: float
    p_hat: float
    p_true: float
    revenue_hat: float | None
    revenue_true: float
    p_error: float
    revenue_error: float | None


def posterior_recovery_error(conn: sqlite3.Connection) -> list[CellRecovery]:
    """How close the LEARNED posterior point estimate is to the DECLARED true
    rate, per cell. This is what proves the bandit found the true rates
    rather than merely converged on a stable but wrong belief -- a Thompson
    sampler can look perfectly well-behaved (stable choices, tight posterior)
    while being confidently wrong if it never explored enough, and only
    grading against the ground truth this codebase is not allowed to see
    would catch that.
    """
    rows = conn.execute("SELECT * FROM posteriors").fetchall()
    out = []
    for r in rows:
        segment, family = Segment(r["segment"]), ActionFamily(r["action_family"])
        p_hat = r["alpha"] / (r["alpha"] + r["beta"])
        p_true = TRUTH[(segment, family)].p_convert
        revenue_hat = (r["revenue_sum"] / r["revenue_n"]) if r["revenue_n"] > 0 else None
        revenue_true = TRUTH[(segment, family)].mean_revenue
        out.append(
            CellRecovery(
                segment=segment,
                action_family=family,
                n_observed=(r["alpha"] + r["beta"]) - _prior_sum(family),
                p_hat=p_hat,
                p_true=p_true,
                revenue_hat=revenue_hat,
                revenue_true=revenue_true,
                p_error=abs(p_hat - p_true),
                revenue_error=abs(revenue_hat - revenue_true) if revenue_hat is not None else None,
            )
        )
    return out


def _prior_sum(family: ActionFamily) -> float:
    from revenew.decide.bandit import prior_for

    a0, b0 = prior_for(family)
    return a0 + b0


def export_to_runtime(
    runtime_conn: sqlite3.Connection,
    *,
    run_id: str,
    regrets: list[DecisionRegret],
    recovery: list[CellRecovery],
    learning: list[dict] | None = None,
) -> None:
    """Writes the ALREADY-COMPUTED regret curve and posterior recovery into
    `revenew.db`'s two demo_* tables, so the dashboard can render them without
    ever opening harness.db. This is the one-way door: everything on the right
    of it is scalars derived from ground truth, never the ground truth itself.

    Called by harness code (harness/run_replay.py's __main__), never by
    anything under revenew/ -- `runtime_conn` here is a connection the CALLER
    owns and passes in, not something this module opens itself.
    """
    # Two running totals, not one: `cumulative` over every decision, and
    # `bandit_cumulative` over only those the bandit actually chose
    # (chosen_family is not None). See db/schema.sql's demo_regret_curve
    # comment for why conflating them hides the learning signal entirely.
    cumulative = 0.0
    bandit_cumulative = 0.0
    bandit_index = 0
    rows = []
    for i, r in enumerate(regrets, start=1):
        cumulative += r.regret
        bandit_chose = r.chosen_family is not None
        if bandit_chose:
            bandit_index += 1
            bandit_cumulative += r.regret
        rows.append((
            run_id, i, r.decision_id, r.segment.value, r.regret, cumulative,
            int(bandit_chose),
            bandit_index if bandit_chose else None,
            bandit_cumulative if bandit_chose else None,
        ))
    runtime_conn.executemany(
        "INSERT OR REPLACE INTO demo_regret_curve "
        "(run_id, decision_index, decision_id, segment, regret, cumulative_regret, "
        " bandit_chose, bandit_decision_index, bandit_cumulative_regret) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    recovery_rows = [
        (
            run_id, c.segment.value, c.action_family.value, c.n_observed,
            c.p_hat, c.p_true, c.p_error, c.revenue_hat, c.revenue_true, c.revenue_error,
        )
        for c in recovery
    ]
    runtime_conn.executemany(
        "INSERT OR REPLACE INTO demo_posterior_recovery "
        "(run_id, segment, action_family, n_observed, p_hat, p_true, p_error, "
        " revenue_hat, revenue_true, revenue_error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        recovery_rows,
    )

    if learning:
        runtime_conn.executemany(
            "INSERT OR REPLACE INTO demo_learning_curve "
            "(run_id, decision_index, n, optimal_rate, regret_per_decision) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (run_id, b["decision_index"], b["n"], b["optimal_rate"], b["regret_per_decision"])
                for b in learning
            ],
        )

    runtime_conn.commit()


# ============================================== SQL views, for inspection --


def attach_and_create_views(harness_conn: sqlite3.Connection, revenew_db_path: str) -> None:
    """ATTACHes revenew.db read-only and creates the views SYSTEM_DESIGN.md
    section 8 names -- `v_cumulative_regret`, `v_posterior_recovery` -- as TEMP
    views on THIS connection only. Never called by anything in revenew/; this
    is explicitly a harness-side, read-only, after-the-fact operation, for ad
    hoc inspection (`sqlite3 harness.db` after a run). The SQL here mirrors
    `compute_decision_regret`/`posterior_recovery_error` above exactly -- same
    branch for a no_action decision's chosen EV, same oracle definition.

    That mirroring is a convention, NOT a checked invariant: an earlier
    version of this docstring claimed "tests/ cross-checks the two against
    each other on a real run", and no such test exists or ever did. Nothing
    calls this function either -- it is a diagnostic you invoke by hand. It
    is kept because SYSTEM_DESIGN.md section 8 names these views by name, so
    deleting them would break a documented promise; but if you change the
    Python path above, nothing will tell you this SQL has drifted from it.
    """
    # Not opened with the URI mode=ro flag: that requires the connection
    # itself to have been created with sqlite3.connect(..., uri=True), which
    # harness/db.py's connect() deliberately does not do (URI-filename
    # parsing has its own footguns and this attach is diagnostic-only). The
    # actual safety property -- revenew/ never opens harness.db -- does not
    # depend on this attach being read-only; nothing here issues a write.
    harness_conn.execute(f"ATTACH DATABASE '{revenew_db_path}' AS runtime")
    harness_conn.executescript(
        """
        CREATE TEMP VIEW v_no_action_reasons AS
        SELECT no_action_reason, COUNT(*) AS n
        FROM runtime.decisions
        WHERE status = 'no_action'
        GROUP BY no_action_reason;

        CREATE TEMP VIEW v_posterior_recovery AS
        SELECT
            p.segment,
            p.action_family,
            p.alpha / (p.alpha + p.beta) AS p_hat,
            gt.p_convert AS p_true,
            ABS(p.alpha / (p.alpha + p.beta) - gt.p_convert) AS p_error,
            CASE WHEN p.revenue_n > 0 THEN p.revenue_sum / p.revenue_n END AS revenue_hat,
            gt.mean_revenue AS revenue_true
        FROM runtime.posteriors p
        JOIN ground_truth gt ON gt.segment = p.segment AND gt.action_family = p.action_family;

        CREATE TEMP VIEW v_decision_ev AS
        SELECT
            d.decision_id, d.opportunity_id, d.created_at, d.segment, d.action_family, d.status,
            CASE
                WHEN d.action_family IS NOT NULL THEN
                    (SELECT gt.p_convert * gt.mean_revenue FROM ground_truth gt
                     WHERE gt.segment = d.segment AND gt.action_family = d.action_family)
                ELSE
                    (SELECT gtb.p_convert * gtb.mean_revenue FROM ground_truth_baseline gtb
                     WHERE gtb.segment = d.segment)
            END AS chosen_true_ev,
            (SELECT MAX(v) FROM (
                SELECT MAX(gt2.p_convert * gt2.mean_revenue) AS v
                FROM ground_truth gt2 WHERE gt2.segment = d.segment
                UNION ALL
                SELECT gtb2.p_convert * gtb2.mean_revenue AS v
                FROM ground_truth_baseline gtb2 WHERE gtb2.segment = d.segment
            )) AS oracle_true_ev
        FROM runtime.decisions d
        WHERE d.status IN ('executed', 'no_action');

        CREATE TEMP VIEW v_cumulative_regret AS
        SELECT
            decision_id, opportunity_id, created_at, segment,
            (oracle_true_ev - chosen_true_ev) AS regret,
            -- opportunity_id, not decision_id -- see compute_decision_regret()'s
            -- comment above for why the tie-break must be content-addressed.
            SUM(oracle_true_ev - chosen_true_ev) OVER (ORDER BY created_at ASC, opportunity_id ASC) AS cumulative_regret
        FROM v_decision_ev;
        """
    )
