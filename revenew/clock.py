"""The sole source of `now()`.

Every timestamp written to the database goes through a Clock, never through a
bare `datetime.now()` call. That single discipline is what makes the virtual
clock possible: replace `WallClock` with `VirtualClock` and every downstream
component -- cooldown windows, attribution windows, the nightly rebuild --
runs identically, just against ticks instead of wall time. Nothing downstream
needs to know which one it's holding.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class WallClock:
    """Real time. What the live webhook receiver and the nightly job use."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class VirtualClock:
    """Deterministic time for replay. Advances only when told to.

    A 30-day fixture run advances this in fixed steps (see harness/run_replay.py)
    rather than sleeping, which is the entire reason 30 simulated days take
    ~40 seconds of wall time instead of 30 real ones -- and the entire reason
    two replay runs from the same seed produce byte-identical timestamps.
    """

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("VirtualClock start must be timezone-aware")
        self._t = start

    def now(self) -> datetime:
        return self._t

    def advance(self, delta: timedelta) -> datetime:
        self._t = self._t + delta
        return self._t

    def set(self, when: datetime) -> None:
        if when.tzinfo is None:
            raise ValueError("VirtualClock.set requires a timezone-aware datetime")
        if when < self._t:
            raise ValueError(f"VirtualClock cannot move backward: {when} < {self._t}")
        self._t = when


def iso(dt: datetime) -> str:
    """Canonical timestamp string for the database. Always UTC, always explicit."""
    if dt.tzinfo is None:
        raise ValueError("refusing to serialize a naive datetime")
    return dt.astimezone(UTC).isoformat()


def parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
