"""The theatre animates a belief history that is nowhere stored.

`posteriors` holds final state only, so `revenew/api/theatre.py` reconstructs
the per-day history by replaying `outcomes` from the priors -- the same thing
`ledger/replay.py` does, for a different reason. That reconstruction is the
one place in the read path where a plausible-looking number could be silently
wrong: an animation that drifts from the real posteriors would still rise,
still converge, and still look convincing, while showing a belief the system
never actually held.

So the last frame is checked against `posteriors` cell by cell. If they agree
exactly, every earlier frame was produced by the same accumulation and the
curve is the system's real history rather than a smooth-looking fiction.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from revenew.api.theatre import CELL_INDEX, build_timeline
from revenew.clock import iso
from revenew.decide.bandit import PosteriorStore
from revenew.ledger.outcome import record_outcome
from revenew.models import ActionFamily, Segment

NOW = datetime(2026, 1, 1, tzinfo=UTC)

# Two cells, one discount-bearing and one not, so the assertion covers both
# cold-start priors rather than only the neutral (1, 1) one.
CASES = [
    (Segment.DORMANT, ActionFamily.PERCENT_DISCOUNT),
    (Segment.ACTIVE, ActionFamily.REMINDER_NUDGE),
]


def _executed_decision(conn, i: int, segment: str, family: str, day: int) -> tuple[str, str]:
    """One treatment-arm opportunity that reached the decision path and was
    executed, dated `day` days into the run so the timeline has an axis to
    build frames along."""
    cid, oid, did = f"cus{i}", f"opp{i}", f"dec{i}"
    when = iso(NOW + timedelta(days=day))
    conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (cid, iso(NOW)))
    conn.execute(
        "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
        (oid, "run1", cid, "dormant_winback", "w1", "dormant_winback", 500, "h", when, None),
    )
    conn.execute(
        "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
        (oid, "run1", cid, "w1", segment, "treatment", when),
    )
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (did, oid, "run1", segment, family, "{}", 5, 2, '{"headline":"x"}', 0.4,
         "executed", None, when, "internal"),
    )
    conn.commit()
    return oid, did


def _seed_run(conn) -> None:
    """A small run with a mix of conversions and non-conversions, outcomes
    landing on a later day than the decision -- the 7-day lag the theatre
    exists to make visible.

    Only `record_outcome` is called: it feeds the bandit itself via
    `_feed_bandit`, so posting the posterior update a second time here would
    double-count every reward and corrupt the very table this test checks
    against."""
    i = 0
    for day in range(0, 12, 3):
        for segment, family in CASES:
            for k in range(4):
                i += 1
                oid, did = _executed_decision(conn, i, segment.value, family.value, day)
                converted = k % 2 == 0
                record_outcome(
                    conn,
                    opportunity_id=oid,
                    decision_id=did,
                    converted=converted,
                    net_revenue=800.0 if converted else 0.0,
                    censored=False,
                    closed_at=iso(NOW + timedelta(days=day + 7)),
                )
    conn.commit()


def test_last_frame_reproduces_the_stored_posteriors_exactly(seeded_conn):
    _seed_run(seeded_conn)

    timeline = build_timeline(seeded_conn)
    assert timeline.frames, "a seeded run must produce at least one frame"
    last = timeline.frames[-1]

    store = PosteriorStore(seeded_conn)
    for segment, family in CASES:
        alpha, beta, n = last["cells"][CELL_INDEX[(segment.value, family.value)]]
        stored = store.get(segment, family)
        assert alpha == stored.alpha, f"{segment}/{family} alpha drifted from the ledger"
        assert beta == stored.beta, f"{segment}/{family} beta drifted from the ledger"
        assert n == stored.n_observed


def test_frames_are_monotonic_because_the_run_only_ever_appends(seeded_conn):
    """Every counter the theatre animates is cumulative. A frame that went
    backwards would mean the walk double-counted or reordered rewards."""
    _seed_run(seeded_conn)
    frames = build_timeline(seeded_conn).frames

    for key in ("decisions", "executed", "outcomes", "generated", "valid"):
        series = [f[key] for f in frames]
        assert series == sorted(series), f"{key} is not monotonic across frames"

    total_observed = [sum(c[2] for c in f["cells"]) for f in frames]
    assert total_observed == sorted(total_observed)
    assert total_observed[-1] == frames[-1]["outcomes"]


def test_cumulative_regret_never_falls_on_a_day_the_bandit_sat_out(seeded_conn):
    """`demo_regret_curve.bandit_cumulative_regret` is NULL on every row where
    the bandit did not choose, and on this workload most days consist of
    nothing but those rows -- every customer is in cooldown.

    Aggregating them with COALESCE(..., 0) makes the day's total collapse to
    zero, and the plotted series then saws between the real figure and nothing
    once per quiet day. It still renders as a chart, which is exactly what
    makes it dangerous: the bug is invisible unless someone notices that a
    cumulative quantity is going down. The regret series must be
    non-decreasing, always.
    """
    _seed_run(seeded_conn)

    # Day 0 has bandit activity; day 3 is a quiet day whose only regret rows
    # carry NULL bandit columns.
    rows = seeded_conn.execute(
        "SELECT decision_id, segment, substr(created_at, 1, 10) AS day FROM decisions "
        "ORDER BY created_at"
    ).fetchall()
    for idx, r in enumerate(rows):
        quiet = r["day"] != rows[0]["day"]
        seeded_conn.execute(
            "INSERT INTO demo_regret_curve (run_id, decision_index, decision_id, segment, "
            "regret, cumulative_regret, bandit_chose, bandit_decision_index, "
            "bandit_cumulative_regret) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "run1", idx, r["decision_id"], r["segment"], 1.0, float(idx + 1),
                0 if quiet else 1,
                None if quiet else idx,
                None if quiet else float(idx + 1) * 10,
            ),
        )
    seeded_conn.commit()

    frames = build_timeline(seeded_conn).frames
    for key in ("cum_regret", "bandit_cum_regret", "bandit_decisions"):
        series = [f[key] for f in frames]
        assert series == sorted(series), f"{key} fell on a quiet day: {series}"


def test_a_database_with_no_run_yields_an_empty_timeline(seeded_conn):
    """`revenew serve` on a fresh database must render the theatre's empty
    state, not raise -- the seeded fixture has customers and orders but has
    never made a decision."""
    timeline = build_timeline(seeded_conn)
    assert timeline.frames == []
    assert timeline.events == []
    assert timeline.meta["run_id"] is None
