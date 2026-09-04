"""DecisionTrace: the sink every stage of one decision feeds into.

Persists a completed `Decision` to `decisions` + `decision_candidates`. This
is deliberately the ONLY place either table is written -- the audit trail is
exactly as trustworthy as the guarantee that nothing bypasses it, so nothing
else in this codebase INSERTs into `decisions` directly.
"""

from __future__ import annotations

import json
import sqlite3

from revenew.clock import iso
from revenew.models import Decision


def persist_decision(conn: sqlite3.Connection, decision: Decision) -> None:
    conn.execute(
        """
        INSERT INTO decisions
            (decision_id, opportunity_id, run_id, segment, action_family, envelope_json,
             candidates_generated, candidates_valid, chosen_candidate_json, propensity,
             status, no_action_reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision.decision_id,
            decision.opportunity_id,
            decision.run_id,
            decision.segment.value,
            decision.action_family.value if decision.action_family else None,
            decision.envelope.model_dump_json(),
            decision.candidates_generated,
            decision.candidates_valid,
            decision.chosen_candidate.model_dump_json() if decision.chosen_candidate else None,
            decision.propensity,
            decision.status.value,
            decision.no_action_reason.value if decision.no_action_reason else None,
            iso(decision.created_at),
        ),
    )
    conn.executemany(
        "INSERT INTO decision_candidates (decision_id, candidate_index, candidate_json, valid, violations_json) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (
                decision.decision_id, i, dc.candidate.model_dump_json(),
                int(dc.valid), json.dumps(dc.violations),
            )
            for i, dc in enumerate(decision.candidates)
        ],
    )
    conn.commit()
