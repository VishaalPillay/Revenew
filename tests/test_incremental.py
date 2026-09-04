"""SegmentLift.is_significant: a small but real correctness contract. Its
type annotation says `-> bool`; the bug this guards against is that the
comparison it's built from (`ci_low > 0 or ci_high < 0`) frequently operates
on `numpy.float64` values (Welch's interval is computed via numpy/scipy), and
a numpy comparison returns `numpy.bool` -- which is not a subclass of
Python's `bool` and is not JSON-serializable, breaking every caller that
promises a plain JSON-shaped dict (`revenew/api/read.py`, `revenew report
--json`) the moment a real lift is significant. Found by actually running
`revenew report --json` against a real replay's data, not by inspection.
"""

from __future__ import annotations

import json

import numpy as np

from revenew.measure.incremental import SegmentLift, welch_interval
from revenew.measure.report import lift_to_dict
from revenew.models import Segment


def _lift(ci_low: float, ci_high: float) -> SegmentLift:
    return SegmentLift(
        segment=Segment.DORMANT, n_treatment=10, n_control=10,
        mean_treatment=100.0, mean_control=50.0, lift=50.0,
        ci_low=ci_low, ci_high=ci_high, p_value=0.01,
    )


def test_is_significant_is_a_real_python_bool_not_a_numpy_bool():
    numpy_ci_low = np.float64(5.0)  # exactly what welch_interval actually returns
    lift = _lift(numpy_ci_low, np.float64(20.0))
    assert lift.is_significant is True
    assert type(lift.is_significant) is bool  # noqa: E721 -- deliberately not isinstance; numpy.bool would pass that


def test_is_significant_false_is_also_a_real_python_bool():
    lift = _lift(np.float64(-5.0), np.float64(5.0))  # interval straddles zero
    assert lift.is_significant is False
    assert type(lift.is_significant) is bool  # noqa: E721


def test_welch_interval_output_round_trips_through_json_via_lift_to_dict():
    """The actual symptom: a lift built from welch_interval's own return
    values (not a hand-constructed numpy.float64, in case some other path
    produces plain floats) must survive `json.dumps` end to end."""
    treatment = np.array([120.0, 130.0, 90.0, 200.0, 0.0, 150.0])
    control = np.array([20.0, 10.0, 0.0, 30.0, 15.0, 5.0])
    lift_val, ci_low, ci_high, p = welch_interval(treatment, control)
    lift = SegmentLift(
        segment=Segment.ACTIVE, n_treatment=len(treatment), n_control=len(control),
        mean_treatment=float(treatment.mean()), mean_control=float(control.mean()),
        lift=lift_val, ci_low=ci_low, ci_high=ci_high, p_value=p,
    )
    json.dumps(lift_to_dict(lift))  # must not raise
