"""The two reporting metrics that were previously wrong in ways that made the
system look worse than it is.

`v_candidate_compliance` splits "the model proposed something illegal" from
"this customer was ineligible for anything" -- conflating them reported 3%
validity on a run where the model's actual violation count was zero.

`demo_regret_curve`'s bandit series separates decisions the bandit actually
made from ones the envelope forced before `BanditScorer.choose()` was ever
reached. Averaging the forced ones in measures cooldown policy, not learning.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from harness.fixture import TRUTH
from harness.regret import DecisionRegret, export_to_runtime, learning_curve
from revenew.clock import iso
from revenew.measure.report import build_report, get_decision_trace
from revenew.models import ActionFamily, Segment

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _decision(conn, did: str, *, status: str, family: str | None, candidates: list[tuple[bool, list[str]]]):
    """One decision plus its candidate verdicts. `candidates` is
    (valid, violations) per candidate."""
    cid, oid = f"cus_{did}", f"opp_{did}"
    conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (cid, iso(NOW)))
    conn.execute(
        "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
        (oid, "run1", cid, "dormant_winback", "w1", "dormant_winback", 500, "h", iso(NOW), None),
    )
    conn.execute(
        "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
        (oid, "run1", cid, "w1", "dormant", "treatment", iso(NOW)),
    )
    n_valid = sum(1 for v, _ in candidates if v)
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (did, oid, "run1", "dormant", family, "{}", len(candidates), n_valid,
         "{}" if family else None, 0.5 if family else None, status,
         None if status == "executed" else "all_candidates_invalid", iso(NOW), "internal"),
    )
    for i, (valid, violations) in enumerate(candidates):
        conn.execute(
            "INSERT INTO decision_candidates VALUES (?,?,?,?,?)",
            (did, i, "{}", int(valid), json.dumps(violations)),
        )
    conn.commit()


def test_eligibility_blocks_are_not_counted_against_the_model(seeded_conn):
    """The bug this guards: a customer blocked by cooldown invalidates every
    candidate for that customer identically, however good they are. Counting
    those as model failures reported 3% validity on a run with zero actual
    policy violations."""
    _decision(seeded_conn, "d_elig", status="no_action", family=None, candidates=[
        (False, ["cooldown_days", "max_offers_per_customer_per_month"]),
        (False, ["cooldown_days", "max_offers_per_customer_per_month"]),
        (False, ["cooldown_days", "max_offers_per_customer_per_month"]),
    ])

    cv = build_report(seeded_conn).candidate_validity

    assert cv.total_generated == 3
    assert cv.policy_violations == 0
    assert cv.eligibility_blocked == 3
    assert cv.policy_compliance_rate == 1.0  # the model did nothing wrong
    assert cv.validity_rate == 0.0           # ...and yet nothing survived


def test_budget_exhaustion_is_not_counted_against_the_model(seeded_conn):
    """The same category error as cooldown, one rule later. `budget_remaining`
    used to sit in `v_candidate_compliance`'s policy bucket, so a perfectly
    legal offer dropped because the campaign ledger had drained counted as
    the MODEL proposing something illegal. It stayed invisible while offers
    were cheap enough that the budget never bound; once the real catalog
    reached the prompt and candidates started costing real money, the panel
    fell from 100% to 93.7% on a run where the model's actual illegal-offer
    count was still exactly zero.

    Budget is a property of how much money is left when an offer is costed,
    not a property of the offer the model composed."""
    _decision(seeded_conn, "d_budget", status="no_action", family=None, candidates=[
        (False, ["budget_remaining"]),
        (False, ["budget_remaining"]),
    ])

    cv = build_report(seeded_conn).candidate_validity

    assert cv.total_generated == 2
    assert cv.policy_violations == 0, "budget exhaustion is not a model error"
    assert cv.budget_blocked == 2
    assert cv.policy_compliance_rate == 1.0


def test_report_still_builds_against_a_database_without_budget_blocked(seeded_conn):
    """`budget_blocked` was added to `v_candidate_compliance` after databases
    already existed, and this project has no migration path -- `init_db`
    refuses to touch an existing file. `build_report` backs GET /, /classic,
    /api/report and /api/regret, so reading that column unconditionally 500s
    the ENTIRE console on any older database instead of degrading one number.
    Caught live: every one of those routes returned 500 against the existing
    revenew.db."""
    _decision(seeded_conn, "d_oldschema", status="executed", family="percent_discount",
              candidates=[(True, [])])
    # Recreate the pre-`budget_blocked` view exactly as older databases carry it.
    seeded_conn.executescript(
        """
        DROP VIEW v_candidate_compliance;
        CREATE VIEW v_candidate_compliance AS
        WITH classified AS (
            SELECT dc.decision_id, dc.candidate_index, dc.valid,
                MAX(CASE WHEN v.value IN (
                    'max_discount_pct', 'max_absolute_discount', 'excluded_skus', 'budget_remaining'
                ) THEN 1 ELSE 0 END) AS broke_policy,
                MAX(CASE WHEN v.value IN (
                    'cooldown_days', 'max_offers_per_customer_per_month'
                ) THEN 1 ELSE 0 END) AS blocked_eligibility
            FROM decision_candidates dc
            LEFT JOIN json_each(dc.violations_json) v ON 1 = 1
            GROUP BY dc.decision_id, dc.candidate_index
        )
        SELECT COUNT(*) AS total_generated, SUM(valid) AS total_valid,
               SUM(broke_policy) AS policy_violations,
               SUM(blocked_eligibility) AS eligibility_blocked,
               1.0 - (CAST(SUM(broke_policy) AS REAL) / NULLIF(COUNT(*), 0)) AS policy_compliance_rate
        FROM classified;
        """
    )
    seeded_conn.commit()

    cv = build_report(seeded_conn).candidate_validity  # must not raise

    assert cv.total_generated == 1
    assert cv.budget_blocked == 0, "absent column degrades to 0, it does not take down the report"


def test_a_genuine_policy_violation_does_count_against_the_model(seeded_conn):
    """The other half: when the model really does propose something illegal,
    compliance must drop. Otherwise the metric is just always 100%."""
    _decision(seeded_conn, "d_pol", status="executed", family="percent_discount", candidates=[
        (True, []),
        (False, ["max_discount_pct"]),
        (False, ["excluded_skus", "cooldown_days"]),  # both kinds at once
    ])

    cv = build_report(seeded_conn).candidate_validity

    assert cv.total_generated == 3
    assert cv.policy_violations == 2          # the cap breach and the banned SKU
    assert cv.eligibility_blocked == 1        # the one that was also cooldown-blocked
    assert cv.policy_compliance_rate == 1 - (2 / 3)


def test_the_chosen_candidate_is_identified_by_index_not_by_headline(seeded_conn):
    """Both trace views used to find the chosen candidate by comparing
    HEADLINE STRINGS. Nothing forbids two candidates sharing a headline -- the
    LLM is asked for 5-8 of them and the shelf's templates are fixed strings --
    so a collision starred multiple rows, and could star a candidate the
    validator had REJECTED, in the one view whose entire purpose is
    auditability.

    Here candidate 0 is INVALID and candidate 1 is VALID, and they share a
    headline. The chosen index must be 1."""
    did, cid, oid = "d_dupe", "cus_d_dupe", "opp_d_dupe"
    duplicate = {
        "action_family": "percent_discount", "headline": "10% off", "discount_pct": 0.10,
        "discount_amount": None, "skus": [], "rationale": "r",
    }
    seeded_conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (cid, iso(NOW)))
    seeded_conn.execute(
        "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
        (oid, "run1", cid, "dormant_winback", "w1", "dormant_winback", 500, "h", iso(NOW), None),
    )
    seeded_conn.execute(
        "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
        (oid, "run1", cid, "w1", "dormant", "treatment", iso(NOW)),
    )
    seeded_conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (did, oid, "run1", "dormant", "percent_discount", "{}", 2, 1,
         json.dumps(duplicate), 0.5, "executed", None, iso(NOW), "internal"),
    )
    # index 0: same headline, but the validator rejected it.
    seeded_conn.execute(
        "INSERT INTO decision_candidates VALUES (?,?,?,?,?)",
        (did, 0, json.dumps(duplicate), 0, json.dumps(["max_discount_pct"])),
    )
    # index 1: the one actually chosen.
    seeded_conn.execute(
        "INSERT INTO decision_candidates VALUES (?,?,?,?,?)",
        (did, 1, json.dumps(duplicate), 1, "[]"),
    )
    seeded_conn.commit()

    trace = get_decision_trace(seeded_conn, did)

    assert trace["chosen_candidate_index"] == 1, (
        "must resolve to the VALID candidate, never the rejected one that shares its headline"
    )


def test_a_no_action_decision_has_no_chosen_candidate_index(seeded_conn):
    _decision(seeded_conn, "d_noact", status="no_action", family=None, candidates=[(False, ["cooldown_days"])])
    trace = get_decision_trace(seeded_conn, "d_noact")
    assert trace["chosen_candidate"] is None
    assert trace["chosen_candidate_index"] is None


def test_learning_curve_detects_a_policy_that_improves(seeded_conn):
    """A bandit that starts picking badly and ends picking the truth-optimal
    action must show a rising `optimal_rate` and a falling regret/decision.
    Built from real ground truth (`TRUTH`), not a hand-picked family name, so
    the test stays correct if the fixture's declared rates ever change."""
    seg = Segment.ACTIVE
    best = max(ActionFamily, key=lambda f: TRUTH[(seg, f)].expected_reward)
    worst = min(ActionFamily, key=lambda f: TRUTH[(seg, f)].expected_reward)

    # First half all-wrong, second half all-right, timestamps in order.
    for i in range(20):
        family = worst if i < 10 else best
        did = f"lc{i}"
        cid, oid = f"cus_{did}", f"opp_{did}"
        seeded_conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (cid, iso(NOW)))
        seeded_conn.execute(
            "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
            (oid, "run1", cid, "dormant_winback", "w1", "dormant_winback", 500, "h", iso(NOW), None),
        )
        seeded_conn.execute(
            "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
            (oid, "run1", cid, "w1", seg.value, "treatment", iso(NOW)),
        )
        seeded_conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (did, oid, "run1", seg.value, family.value, "{}", 1, 1, "{}", 0.5, "executed", None,
             iso(NOW.replace(minute=i)), "internal"),
        )
    seeded_conn.commit()

    curve = learning_curve(seeded_conn, buckets=2)

    assert len(curve) == 2
    assert curve[0]["optimal_rate"] == 0.0   # every early pick was the worst family
    assert curve[1]["optimal_rate"] == 1.0   # every late pick was the best
    assert curve[1]["regret_per_decision"] < curve[0]["regret_per_decision"]
    assert curve[1]["regret_per_decision"] == 0.0  # picking the oracle's own answer


def test_learning_curve_is_flat_for_a_policy_that_never_changes(seeded_conn):
    """The control case -- a metric that only ever goes up is not a metric.
    A policy that picks the same wrong family throughout must show no
    improvement."""
    seg = Segment.ACTIVE
    worst = min(ActionFamily, key=lambda f: TRUTH[(seg, f)].expected_reward)
    for i in range(20):
        did = f"flat{i}"
        cid, oid = f"cus_{did}", f"opp_{did}"
        seeded_conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (cid, iso(NOW)))
        seeded_conn.execute(
            "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
            (oid, "run1", cid, "dormant_winback", "w1", "dormant_winback", 500, "h", iso(NOW), None),
        )
        seeded_conn.execute(
            "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
            (oid, "run1", cid, "w1", seg.value, "treatment", iso(NOW)),
        )
        seeded_conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (did, oid, "run1", seg.value, worst.value, "{}", 1, 1, "{}", 0.5, "executed", None,
             iso(NOW.replace(minute=i)), "internal"),
        )
    seeded_conn.commit()

    curve = learning_curve(seeded_conn, buckets=2)

    assert [b["optimal_rate"] for b in curve] == [0.0, 0.0]
    assert curve[0]["regret_per_decision"] == curve[1]["regret_per_decision"]


def test_regret_curve_separates_bandit_decisions_from_forced_no_actions(seeded_conn):
    """`regret_curve` must contain only decisions the bandit actually chose.
    A `no_action` forced by the envelope was never a choice -- including it
    measures the cooldown policy, not learning."""
    regrets = [
        DecisionRegret("d1", iso(NOW), Segment.DORMANT, ActionFamily.PERCENT_DISCOUNT, 60.0, 100.0),
        DecisionRegret("d2", iso(NOW), Segment.DORMANT, None, 10.0, 100.0),  # forced no_action
        DecisionRegret("d3", iso(NOW), Segment.DORMANT, ActionFamily.FLAT_COUPON, 80.0, 100.0),
        DecisionRegret("d4", iso(NOW), Segment.DORMANT, None, 10.0, 100.0),  # forced no_action
    ]
    export_to_runtime(seeded_conn, run_id="run1", regrets=regrets, recovery=[])

    report = build_report(seeded_conn)

    # Bandit series: only d1 and d3, cumulative 40 then 40+20 = 60.
    assert [p["cumulative_regret"] for p in report.regret_curve] == [40.0, 60.0]
    assert [p["decision_index"] for p in report.regret_curve] == [1, 2]

    # All-decisions series keeps every row: 40, 130, 150, 240.
    assert [p["cumulative_regret"] for p in report.regret_curve_all] == [40.0, 130.0, 150.0, 240.0]

    # The forced no-actions carry the larger share -- which is exactly why
    # plotting the combined series buries the learning signal.
    assert report.regret_curve_all[-1]["cumulative_regret"] > 3 * report.regret_curve[-1]["cumulative_regret"]
