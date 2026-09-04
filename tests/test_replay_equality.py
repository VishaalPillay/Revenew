"""Reproducibility is ranked above availability (SYSTEM_DESIGN.md section 1.2):
"same fixture + seed => byte-identical posteriors. Non-negotiable."

This is the test that makes that non-negotiable rather than aspirational.
`rebuild_posteriors` recomputes the entire `posteriors` table from the
`outcomes` log alone; a live run that updates posteriors incrementally as
outcomes arrive must land on the exact same numbers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from revenew.clock import iso
from revenew.decide.bandit import PosteriorStore
from revenew.ledger.outcome import record_outcome
from revenew.ledger.replay import posteriors_snapshot, rebuild_posteriors
from revenew.models import ActionFamily, Segment

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _make_executed_decision(conn, i: int, segment: str, family: str) -> tuple[str, str]:
    cid, oid, did = f"cus{i}", f"opp{i}", f"dec{i}"
    conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (cid, iso(NOW)))
    conn.execute(
        "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?)",
        (oid, "run1", cid, "dormant_winback", "w1", "dormant_winback", 500, "h", iso(NOW)),
    )
    conn.execute(
        "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
        (oid, "run1", cid, "w1", segment, "treatment", iso(NOW)),
    )
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (did, oid, "run1", segment, family, "{}", 3, 1, "{}", 0.4, "executed", None, iso(NOW)),
    )
    conn.commit()
    return oid, did


def test_replay_matches_the_live_updated_posteriors_exactly(seeded_conn):
    store = PosteriorStore(seeded_conn)
    store.ensure_initialized()

    rng = np.random.default_rng(99)
    segments = list(Segment)
    families = list(ActionFamily)

    for i in range(150):
        seg = segments[i % len(segments)].value
        fam = families[i % len(families)].value
        oid, did = _make_executed_decision(seeded_conn, i, seg, fam)
        converted = bool(rng.random() < 0.4)
        revenue = float(rng.uniform(200, 900)) if converted else 0.0
        closed_at = (NOW + timedelta(days=i % 5)).isoformat()
        record_outcome(
            seeded_conn, opportunity_id=oid, decision_id=did,
            converted=converted, net_revenue=revenue, censored=False, closed_at=closed_at,
        )

    live_snapshot = posteriors_snapshot(seeded_conn)

    rebuild_posteriors(seeded_conn)
    rebuilt_snapshot = posteriors_snapshot(seeded_conn)

    assert live_snapshot == rebuilt_snapshot


def test_replay_is_itself_deterministic_across_repeated_rebuilds(seeded_conn):
    store = PosteriorStore(seeded_conn)
    store.ensure_initialized()
    rng = np.random.default_rng(7)
    for i in range(40):
        oid, did = _make_executed_decision(
            seeded_conn, i, Segment.LAPSING.value, ActionFamily.FLAT_COUPON.value
        )
        converted = bool(rng.random() < 0.3)
        record_outcome(
            seeded_conn, opportunity_id=oid, decision_id=did,
            converted=converted, net_revenue=400.0 if converted else 0.0,
            censored=False, closed_at=iso(NOW),
        )

    rebuild_posteriors(seeded_conn)
    first = posteriors_snapshot(seeded_conn)
    rebuild_posteriors(seeded_conn)
    second = posteriors_snapshot(seeded_conn)
    assert first == second


def test_control_arm_outcomes_never_move_the_posteriors(seeded_conn):
    """Control-arm opportunities record outcomes but carry decision_id=None,
    so they must be invisible to both the live bandit feed and replay."""
    store = PosteriorStore(seeded_conn)
    store.ensure_initialized()
    before = posteriors_snapshot(seeded_conn)

    for i in range(30):
        cid, oid = f"ccus{i}", f"copp{i}"
        seeded_conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (cid, iso(NOW)))
        seeded_conn.execute(
            "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?)",
            (oid, "run1", cid, "dormant_winback", "w1", "dormant_winback", 500, "h", iso(NOW)),
        )
        seeded_conn.execute(
            "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
            (oid, "run1", cid, "w1", "dormant", "control", iso(NOW)),
        )
        seeded_conn.commit()
        record_outcome(
            seeded_conn, opportunity_id=oid, decision_id=None,
            converted=True, net_revenue=550.0, censored=False, closed_at=iso(NOW),
        )

    after = posteriors_snapshot(seeded_conn)
    assert before == after

    rebuild_posteriors(seeded_conn)
    after_rebuild = posteriors_snapshot(seeded_conn)
    assert after_rebuild == before
