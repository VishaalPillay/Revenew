"""OpportunityDetector: parameterised SQL, nothing else.

No opportunity is ever constructed by Python arithmetic over rows fetched
generically -- every one comes from a named block in queries.sql, run with
bound parameters, so `detector_query_hash` is a real fingerprint of the exact
logic that produced the row, not a label chosen after the fact.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from revenew.clock import iso
from revenew.models import OpportunityType, Segment

QUERIES_PATH = __import__("pathlib").Path(__file__).resolve().parent / "queries.sql"

# Thresholds. Named constants, not magic numbers scattered through the SQL or
# buried in a config file nobody reads before the demo.
DORMANT_THRESHOLD_DAYS = 60
FIRST_ORDER_THRESHOLD_DAYS = 14
CROSS_SELL_MIN_PAIR_COUNT = 8
CROSS_SELL_MIN_CONFIDENCE = 0.12

# Segment boundaries. The single definition every component (detector,
# dashboard, replay) must agree on -- see `segment_of`.
ACTIVE_RECENCY_DAYS = 30
LAPSING_RECENCY_DAYS = 90


def _parse_named_queries(sql_text: str) -> dict[str, str]:
    blocks = re.split(r"^-- name:\s*(\S+)\s*$", sql_text, flags=re.MULTILINE)
    # blocks[0] is leading comment/whitespace before the first marker.
    out: dict[str, str] = {}
    for i in range(1, len(blocks), 2):
        name, body = blocks[i], blocks[i + 1]
        out[name] = body.strip()
    return out


def _load_queries() -> dict[str, str]:
    text = QUERIES_PATH.read_text(encoding="utf-8")
    named = _parse_named_queries(text)
    expected = {t.value for t in OpportunityType}
    missing = expected - named.keys()
    if missing:
        raise ValueError(f"queries.sql is missing named blocks for: {sorted(missing)}")
    return named


def _query_hash(sql_text: str) -> str:
    return hashlib.sha256(sql_text.encode("utf-8")).hexdigest()[:16]


def _opportunity_id(run_id: str, window_id: str, opportunity_type: OpportunityType, customer_id: str) -> str:
    """Deterministic, not `uuid.uuid4()`.

    A raw UUID here would mean the exact same detection input produces a
    different opportunity_id on every run -- and that id later seeds the
    bandit's RNG stream (see decide/__init__.py), so a random id would make
    the choice of action itself non-reproducible. Content-addressing it from
    the four values that actually identify "this opportunity" is what makes
    two runs of the same fixture with the same seed replay to byte-identical
    decisions, per SYSTEM_DESIGN.md section 1.2.
    """
    raw = f"{run_id}|{window_id}|{opportunity_type.value}|{customer_id}"
    return "opp_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def segment_of(orders_count: int, days_since_last_order: int | None) -> Segment:
    """The one place recency/frequency becomes a Segment label.

    A customer with zero or one order is NEW regardless of how long ago that
    order was -- they haven't had the chance to lapse yet, and
    FIRST_ORDER_RETENTION is the opportunity type that watches for exactly
    this case aging past its window.
    """
    if orders_count <= 1:
        return Segment.NEW
    if days_since_last_order is None:
        return Segment.NEW
    if days_since_last_order <= ACTIVE_RECENCY_DAYS:
        return Segment.ACTIVE
    if days_since_last_order <= LAPSING_RECENCY_DAYS:
        return Segment.LAPSING
    return Segment.DORMANT


def compute_segment_map(conn: sqlite3.Connection, now: datetime) -> dict[str, Segment]:
    """Segment for every customer with at least one captured order, plus NEW
    for everyone else. Shared by the detector, the dashboard, and replay."""
    rows = conn.execute(
        """
        SELECT
            c.customer_id,
            COUNT(o.order_id) AS orders_count,
            MAX(o.placed_at) AS last_order_at
        FROM customers c
        LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'captured'
        GROUP BY c.customer_id
        """
    ).fetchall()

    out: dict[str, Segment] = {}
    for row in rows:
        days_since = None
        if row["last_order_at"] is not None:
            last = datetime.fromisoformat(row["last_order_at"])
            days_since = (now.astimezone(last.tzinfo) - last).days if last.tzinfo else (now.replace(tzinfo=None) - last).days
        out[row["customer_id"]] = segment_of(row["orders_count"], days_since)
    return out


@dataclass(frozen=True)
class RawOpportunity:
    """One detector hit, before arbitration. Maps 1:1 to `opportunity_candidates`."""

    opportunity_id: str
    run_id: str
    customer_id: str
    opportunity_type: OpportunityType
    window_id: str
    cohort_id: str
    rupees_at_risk: float
    detector_query_hash: str
    detected_at: str
    # Only cross_sell_affinity's query projects this column; every other
    # query's rows simply don't have the key, which is why this is read via
    # `row.keys()` rather than `row["recommended_sku"]` at the call site --
    # sqlite3.Row raises IndexError on a column the query never selected.
    recommended_sku: str | None = None


class OpportunityDetector:
    def __init__(self) -> None:
        self._queries = _load_queries()
        self._hashes = {name: _query_hash(sql) for name, sql in self._queries.items()}

    def query_hash(self, opportunity_type: OpportunityType) -> str:
        return self._hashes[opportunity_type.value]

    def detect(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        window_id: str,
        now: datetime,
    ) -> list[RawOpportunity]:
        """Run every named query, tag results with identity, return raw hits.

        `cohort_id` is the opportunity_type itself: for tie-breaking in the
        arbiter, "which cohort" only needs to be stable and comparable, and the
        opportunity type already is both.
        """
        detected_at = iso(now)
        out: list[RawOpportunity] = []

        params_by_type = {
            OpportunityType.DORMANT_WINBACK: {
                "now": iso(now),
                "dormant_threshold_days": DORMANT_THRESHOLD_DAYS,
            },
            OpportunityType.FIRST_ORDER_RETENTION: {
                "now": iso(now),
                "first_order_threshold_days": FIRST_ORDER_THRESHOLD_DAYS,
            },
            OpportunityType.CROSS_SELL_AFFINITY: {
                "min_pair_count": CROSS_SELL_MIN_PAIR_COUNT,
                "min_confidence": CROSS_SELL_MIN_CONFIDENCE,
            },
        }

        for otype in OpportunityType:
            sql = self._queries[otype.value]
            params = params_by_type[otype]
            for row in conn.execute(sql, params).fetchall():
                keys = row.keys()
                out.append(
                    RawOpportunity(
                        opportunity_id=_opportunity_id(run_id, window_id, otype, row["customer_id"]),
                        run_id=run_id,
                        customer_id=row["customer_id"],
                        opportunity_type=otype,
                        window_id=window_id,
                        cohort_id=otype.value,
                        rupees_at_risk=float(row["rupees_at_risk"]),
                        detector_query_hash=self._hashes[otype.value],
                        detected_at=detected_at,
                        recommended_sku=row["recommended_sku"] if "recommended_sku" in keys else None,
                    )
                )
        return out

    def persist_candidates(self, conn: sqlite3.Connection, candidates: list[RawOpportunity]) -> None:
        conn.executemany(
            """
            INSERT INTO opportunity_candidates
                (opportunity_id, run_id, customer_id, opportunity_type, window_id,
                 cohort_id, rupees_at_risk, detector_query_hash, detected_at, recommended_sku)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.opportunity_id, c.run_id, c.customer_id, c.opportunity_type.value,
                    c.window_id, c.cohort_id, c.rupees_at_risk, c.detector_query_hash, c.detected_at,
                    c.recommended_sku,
                )
                for c in candidates
            ],
        )
        conn.commit()
