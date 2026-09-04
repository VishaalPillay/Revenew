"""No money lost to crashes. A reservation that is never released or spent
still shows up as consumed -- that IS the "hold, don't lose" guarantee, so a
reserve with no matching release must reduce `available()` and stay reduced.
A release must restore it exactly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from revenew.clock import iso
from revenew.execute import budget

NOW = datetime(2026, 1, 1, tzinfo=UTC)
BUDGET_CAP = 10_000.0


def _make_decision(conn, decision_id: str, customer_id: str = "cus1") -> None:
    conn.execute("INSERT OR IGNORE INTO customers VALUES (?, ?)", (customer_id, iso(NOW)))
    conn.execute(
        "INSERT INTO opportunity_candidates VALUES (?,?,?,?,?,?,?,?,?)",
        (f"opp_{decision_id}", "run1", customer_id, "dormant_winback", "w1",
         "dormant_winback", 500, "h", iso(NOW)),
    )
    conn.execute(
        "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?)",
        (f"opp_{decision_id}", "run1", customer_id, "w1", "dormant", "treatment", iso(NOW)),
    )
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (decision_id, f"opp_{decision_id}", "run1", "dormant", "percent_discount", "{}",
         1, 1, "{}", 0.5, "executed", None, iso(NOW)),
    )
    conn.commit()


def test_available_starts_at_the_full_cap(seeded_conn):
    assert budget.available(seeded_conn, BUDGET_CAP) == BUDGET_CAP


def test_a_reservation_reduces_available_and_a_release_restores_it(seeded_conn):
    _make_decision(seeded_conn, "d1")
    budget.reserve(seeded_conn, "d1", 300, now=NOW)
    assert budget.available(seeded_conn, BUDGET_CAP) == BUDGET_CAP - 300

    budget.release(seeded_conn, "d1", 300, now=NOW)
    assert budget.available(seeded_conn, BUDGET_CAP) == BUDGET_CAP


def test_an_unreleased_reservation_holds_budget_rather_than_losing_it(seeded_conn):
    """The crash-recovery property, stated as an invariant. A reservation with
    no release must NOT silently vanish from the ledger -- if it did, a crash
    between reserve and execute would let the merchant overspend on the next
    decision, which is exactly the failure this design exists to prevent."""
    _make_decision(seeded_conn, "d1")
    budget.reserve(seeded_conn, "d1", 400, now=NOW)
    # Simulate a crash: nothing more happens for this decision. available()
    # must still reflect the hold on every subsequent read.
    for _ in range(3):
        assert budget.available(seeded_conn, BUDGET_CAP) == BUDGET_CAP - 400


def test_ledger_never_lets_available_go_negative_of_its_own_accord(seeded_conn):
    """budget.reserve() does not enforce the cap itself (the envelope does,
    before reserve() is ever called) -- but the LEDGER's arithmetic must still
    be exact: reserving more than the cap makes available() negative, visibly,
    rather than clamping or silently discarding the overage."""
    _make_decision(seeded_conn, "d1")
    budget.reserve(seeded_conn, "d1", BUDGET_CAP + 500, now=NOW)
    assert budget.available(seeded_conn, BUDGET_CAP) == -500


def test_conservation_across_many_reserve_release_cycles(seeded_conn):
    """SUM(reserved) + SUM(released) must equal -(total ever spent), for any
    sequence of operations -- checked directly against the ledger rather than
    against available(), so this test would catch a bug in available() itself
    as well as one in reserve/release."""
    net_spent = 0.0
    for i in range(20):
        did = f"d{i}"
        _make_decision(seeded_conn, did, customer_id=f"cus{i}")
        amount = 50 + i * 10
        budget.reserve(seeded_conn, did, amount, now=NOW)
        if i % 3 != 0:  # most get released; some (crash simulation) do not
            budget.release(seeded_conn, did, amount, now=NOW)
        else:
            net_spent += amount

    ledger_sum = seeded_conn.execute("SELECT COALESCE(SUM(amount), 0) AS s FROM budget_ledger").fetchone()["s"]
    assert ledger_sum == pytest.approx(-net_spent)
    assert budget.available(seeded_conn, BUDGET_CAP) == pytest.approx(BUDGET_CAP - net_spent)


def test_reserved_amount_nets_reserve_and_release_for_one_decision(seeded_conn):
    _make_decision(seeded_conn, "d1")
    budget.reserve(seeded_conn, "d1", 500, now=NOW)
    assert budget.reserved_amount(seeded_conn, "d1") == 500
    budget.release(seeded_conn, "d1", 500, now=NOW)
    assert budget.reserved_amount(seeded_conn, "d1") == 0


def test_negative_amounts_are_rejected_outright(seeded_conn):
    _make_decision(seeded_conn, "d1")
    with pytest.raises(ValueError):
        budget.reserve(seeded_conn, "d1", -100, now=NOW)
    with pytest.raises(ValueError):
        budget.release(seeded_conn, "d1", -100, now=NOW)
