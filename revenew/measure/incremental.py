"""IncrementalEstimator: treatment minus control, per segment, with a Welch
interval. The only externally-reported number in the system -- see
SYSTEM_DESIGN.md section 8's anti-metric: gross revenue from targeted
customers is never reported, because most of them would have converted anyway.

Welch's t-test, not Student's: the two arms have very different sample sizes
by construction (80/20 split) and there is no reason to assume equal variance
between "got an offer" and "got nothing" -- assuming it would understate the
uncertainty exactly where the split is most lopsided.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
from scipy import stats

from revenew.models import Segment

CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class SegmentLift:
    segment: Segment
    n_treatment: int
    n_control: int
    mean_treatment: float
    mean_control: float
    lift: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def is_significant(self) -> bool:
        # bool(...), not the bare comparison: ci_low/ci_high are frequently
        # numpy.float64 (welch_interval does its arithmetic in numpy/scipy),
        # and a numpy.float64 comparison returns numpy.bool -- which is NOT a
        # subclass of Python's bool and is not JSON-serializable, unlike this
        # property's own declared `-> bool` return type promises. Found via
        # `revenew report --json`, which is exactly the kind of caller this
        # type contract exists to keep honest.
        return bool(self.ci_low > 0 or self.ci_high < 0)


def _fetch_revenue(conn: sqlite3.Connection, segment: Segment) -> tuple[np.ndarray, np.ndarray]:
    """Net revenue per outcome, split by arm. Non-converting outcomes
    contribute 0.0, exactly as `Outcome.net_revenue` requires -- excluding
    them would bias the mean upward by conditioning on conversion, which is
    precisely the "gross revenue from targeted customers" anti-metric."""
    rows = conn.execute(
        """
        SELECT o.net_revenue, opp.arm
        FROM outcomes o
        JOIN opportunities opp ON opp.opportunity_id = o.opportunity_id
        WHERE opp.segment = ?
        """,
        (segment.value,),
    ).fetchall()
    treatment = np.array([r["net_revenue"] for r in rows if r["arm"] == "treatment"], dtype=float)
    control = np.array([r["net_revenue"] for r in rows if r["arm"] == "control"], dtype=float)
    return treatment, control


def welch_interval(
    treatment: np.ndarray, control: np.ndarray, *, confidence: float = CONFIDENCE_LEVEL
) -> tuple[float, float, float, float]:
    """(lift, ci_low, ci_high, p_value) via Welch's t-test.

    Returns a maximally-wide interval, not a crash, when either arm is too
    small to say anything -- a demo run at small N should show a wide interval
    honestly, per SYSTEM_DESIGN.md section 8, not fail outright.
    """
    n1, n2 = len(treatment), len(control)
    if n1 < 2 or n2 < 2:
        lift = (float(treatment.mean()) if n1 else 0.0) - (float(control.mean()) if n2 else 0.0)
        return lift, float("-inf"), float("inf"), 1.0

    m1, m2 = treatment.mean(), control.mean()
    v1, v2 = treatment.var(ddof=1), control.var(ddof=1)
    lift = float(m1 - m2)

    se = np.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return lift, lift, lift, 0.0 if lift != 0 else 1.0

    # Welch-Satterthwaite degrees of freedom.
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df)
    margin = t_crit * se
    return lift, lift - margin, lift + margin, float(p_value)


def compute_lift(conn: sqlite3.Connection, segments: list[Segment] | None = None) -> list[SegmentLift]:
    segments = segments or list(Segment)
    out = []
    for seg in segments:
        treatment, control = _fetch_revenue(conn, seg)
        lift, lo, hi, p = welch_interval(treatment, control)
        out.append(
            SegmentLift(
                segment=seg,
                n_treatment=len(treatment),
                n_control=len(control),
                mean_treatment=float(treatment.mean()) if len(treatment) else 0.0,
                mean_control=float(control.mean()) if len(control) else 0.0,
                lift=lift,
                ci_low=lo,
                ci_high=hi,
                p_value=p,
            )
        )
    return out


def overall_lift(conn: sqlite3.Connection) -> SegmentLift:
    """Pooled across all segments. Reported alongside the per-segment table,
    never in place of it -- an aggregate can hide a segment where the true
    effect runs the other way."""
    rows = conn.execute(
        """
        SELECT o.net_revenue, opp.arm
        FROM outcomes o
        JOIN opportunities opp ON opp.opportunity_id = o.opportunity_id
        """
    ).fetchall()
    treatment = np.array([r["net_revenue"] for r in rows if r["arm"] == "treatment"], dtype=float)
    control = np.array([r["net_revenue"] for r in rows if r["arm"] == "control"], dtype=float)
    lift, lo, hi, p = welch_interval(treatment, control)
    return SegmentLift(
        segment=None,  # type: ignore[arg-type]  # pooled, not one segment
        n_treatment=len(treatment),
        n_control=len(control),
        mean_treatment=float(treatment.mean()) if len(treatment) else 0.0,
        mean_control=float(control.mean()) if len(control) else 0.0,
        lift=lift,
        ci_low=lo,
        ci_high=hi,
        p_value=p,
    )
