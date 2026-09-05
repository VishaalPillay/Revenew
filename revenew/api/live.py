"""Live Decision Studio: the demo endpoint where a judge drives one real
decision through the full pipeline and watches it happen.

Two endpoints:

    POST /api/live/decide   → SSE stream of the decision pipeline unfolding
    POST /api/live/revalidate → deterministic re-verdict under a tightened cap

**Why this is NOT `run_replay` with a small window.**  `run_replay` calls
`init_db(reset=True)` at the very top — it is designed to produce a fresh
database from scratch, and calling it against a live `revenew.db` would
delete every prior decision mid-demo.  Phase 3 calls the same PIECES
(`EnvelopeEngine.build`, `CandidateGenerator.generate`,
`EnvelopeValidator.validate_all`, `BanditScorer.choose`, `execute_decision`)
directly, emitting SSE events between each step so the judge sees the
pipeline unfold without ever touching `run_replay`'s orchestration.

**Why `run_id` starts with `live_`.**  PLAN.md §4.2 identifies a real
landmine: `theatre.py`'s `_latest_run_id()` picks the newest `created_at`
with no filter, so a single live decision written under any run_id that sorts
newest would flip the Theatre to a one-decision "run" and render empty.  All
live decisions use `run_id = f"live_{uuid4().hex[:8]}"`, and `_latest_run_id`
(and the cache key in `read.py`) now exclude that prefix.  Regression test:
`test_live.py::test_live_decision_does_not_break_theatre`.

**Fallback.**  If the live Groq call times out or errors, we fall through to
the committed cassette and emit a visible `degraded` SSE event saying so.
An honest, visible fallback reads as engineering maturity; a silent hang or a
500 reads as broken.

**Revalidation is the money shot.**  `POST /api/live/revalidate` re-runs
`EnvelopeValidator.validate_all` over the ALREADY-STORED `decision_candidates`
for a given decision against a tightened cap passed in the request.  The
judge moves a discount-cap slider and watches the AI's chosen offer get
struck down live.  Deterministic, instant, no model call — it cannot be
allowed to fail on stage.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from revenew.api.webhooks import get_conn
from revenew.clock import WallClock
from revenew.db import connect
from revenew.decide.bandit import BanditScorer, PosteriorStore
from revenew.decide.cassette import Cassette
from revenew.decide.envelope import EnvelopeEngine, EnvelopeValidator
from revenew.decide.generator import CandidateGenerator
from revenew.decide.trace import mark_executed, persist_decision
from revenew.execute import budget
from revenew.execute.razorpay import RazorpayAdapter, build_adapter, execute_decision
from revenew.models import (
    Candidate,
    Decision,
    DecisionStatus,
    Envelope,
    NoActionReason,
    OfferSpec,
)
from revenew.settings import DEFAULT_POLICY, GROQ_MODEL, PolicyConfig

router = APIRouter()


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: dict) -> str:
    """One SSE frame.  `data:` is a single JSON line — multi-line `data:`
    blocks are legal SSE but gratuitously harder to parse on the client."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _is_connection_open(conn: sqlite3.Connection | None) -> bool:
    """Check if a sqlite3 connection is valid and open.

    FastAPI request-scoped generator dependencies (`get_conn`) run their
    teardown (`finally: conn.close()`) as soon as the route handler returns.
    Because StreamingResponse iterates *after* the handler returns, any
    connection injected from the route is already closed by the time the
    generator runs.  This helper lets _decide_generator detect that and
    open its own connection safely."""
    if conn is None:
        return False
    try:
        conn.execute("SELECT 1")
        return True
    except (sqlite3.ProgrammingError, sqlite3.OperationalError):
        return False


# ---------------------------------------------------------------------------
# Opportunity selection
# ---------------------------------------------------------------------------

def _pick_opportunity(conn: sqlite3.Connection) -> dict | None:
    """Pick a random customer who has an existing opportunity in the DB.

    Deliberately picks from the TREATMENT arm only (control-arm opportunities
    never reach the decision path — that's what makes them the counterfactual).

    Returns None if no eligible opportunity exists (empty DB)."""
    row = conn.execute(
        """
        SELECT o.customer_id, o.segment,
               oc.opportunity_type, oc.rupees_at_risk,
               oc.cohort_id, oc.detector_query_hash, oc.recommended_sku
        FROM opportunities o
        JOIN opportunity_candidates oc ON oc.opportunity_id = o.opportunity_id
        WHERE o.arm = 'treatment'
        ORDER BY RANDOM()
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# POST /api/live/decide  →  SSE
# ---------------------------------------------------------------------------

class DecideRequest(BaseModel):
    """Optional overrides for the live decision.  All fields are optional —
    calling with an empty body picks a random opportunity automatically."""
    customer_id: str | None = None


def _decide_generator(
    conn: sqlite3.Connection | None,
    request: DecideRequest,
    policy: PolicyConfig,
    adapter: RazorpayAdapter | None,
):
    """Sync generator that yields SSE frames.  FastAPI runs this in a
    threadpool via `StreamingResponse` — fine with the existing sync sqlite3,
    no async rewrite needed."""
    own_conn = False
    if not _is_connection_open(conn):
        db_path = os.environ.get("REVENEW_DB_PATH")
        conn = connect(db_path) if db_path else connect()
        own_conn = True

    try:
        yield from _decide_stream(conn, request, policy, adapter)
    finally:
        if own_conn and conn is not None:
            conn.close()


def _decide_stream(
    conn: sqlite3.Connection,
    request: DecideRequest,
    policy: PolicyConfig,
    adapter: RazorpayAdapter | None,
):
    clock = WallClock()
    now = clock.now()

    # ── 1. Select opportunity ──────────────────────────────────────────
    if request.customer_id:
        row = conn.execute(
            """
            SELECT o.customer_id, o.segment,
                   oc.opportunity_type, oc.rupees_at_risk,
                   oc.cohort_id, oc.detector_query_hash, oc.recommended_sku
            FROM opportunities o
            JOIN opportunity_candidates oc ON oc.opportunity_id = o.opportunity_id
            WHERE o.arm = 'treatment' AND o.customer_id = ?
            ORDER BY RANDOM() LIMIT 1
            """,
            (request.customer_id,),
        ).fetchone()
        if row is None:
            yield _sse_event("error", {"message": f"no treatment opportunity for customer {request.customer_id}"})
            return
        opp = dict(row)
    else:
        opp = _pick_opportunity(conn)
        if opp is None:
            yield _sse_event("error", {"message": "no eligible opportunity found — is the DB populated?"})
            return

    from revenew.models import OpportunityType, Segment

    opportunity_type = OpportunityType(opp["opportunity_type"])
    segment = Segment(opp["segment"])
    customer_id = opp["customer_id"]
    rupees_at_risk = float(opp["rupees_at_risk"])

    # Establish the live run and opportunity IDs.
    # decisions.opportunity_id has a UNIQUE constraint in schema.sql.
    # Therefore, each live decision must link to its own live opportunity record
    # rather than reusing a replay run's already-decided opportunity.
    run_id = f"live_{uuid.uuid4().hex[:8]}"
    live_opp_id = f"live_opp_{uuid.uuid4().hex[:12]}"
    window_id = f"live_w_{uuid.uuid4().hex[:6]}"
    now_iso = now.isoformat()

    conn.execute(
        """
        INSERT INTO opportunity_candidates (
            opportunity_id, run_id, customer_id, opportunity_type, window_id,
            cohort_id, rupees_at_risk, detector_query_hash, detected_at, recommended_sku
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            live_opp_id,
            run_id,
            customer_id,
            opportunity_type.value,
            window_id,
            opp.get("cohort_id") or "live_cohort",
            rupees_at_risk,
            opp.get("detector_query_hash") or "live_hash",
            now_iso,
            opp.get("recommended_sku"),
        ),
    )
    conn.execute(
        """
        INSERT INTO opportunities (
            opportunity_id, run_id, customer_id, window_id, segment, arm, assigned_at
        ) VALUES (?, ?, ?, ?, ?, 'treatment', ?)
        """,
        (
            live_opp_id,
            run_id,
            customer_id,
            window_id,
            segment.value,
            now_iso,
        ),
    )
    conn.commit()

    yield _sse_event("opportunity", {
        "customer_id": customer_id,
        "segment": segment.value,
        "opportunity_type": opportunity_type.value,
        "rupees_at_risk": rupees_at_risk,
    })

    # ── 2. Build envelope ──────────────────────────────────────────────
    # For live decisions against an existing DB (where a historical replay run
    # has already consumed budget), ensure the live session has budget headroom
    # rather than being blocked by past replay runs.
    cur_consumed = budget.consumed(conn)
    if policy.budget_cap <= cur_consumed:
        policy = PolicyConfig(
            max_discount_pct=policy.max_discount_pct,
            max_absolute_discount=policy.max_absolute_discount,
            budget_cap=cur_consumed + DEFAULT_POLICY.budget_cap,
            cooldown_days=policy.cooldown_days,
            max_offers_per_customer_per_month=policy.max_offers_per_customer_per_month,
            excluded_skus=policy.excluded_skus,
        )

    envelope = EnvelopeEngine.build(conn, policy)
    catalog = EnvelopeEngine.load_catalog(conn)

    yield _sse_event("envelope", {
        "max_discount_pct": envelope.max_discount_pct,
        "max_absolute_discount": envelope.max_absolute_discount,
        "budget_remaining": envelope.budget_remaining,
        "excluded_skus": envelope.excluded_skus,
        "cooldown_days": envelope.cooldown_days,
    })

    # ── 3. Generate candidates (live LLM call) ─────────────────────────
    yield _sse_event("llm_start", {"model": GROQ_MODEL})

    # Scratch tmpdir for the live cassette — never writes into committed
    # cassettes/candidates/.  Not cleaned up here (OS handles tmpdir GC),
    # because on Windows the sqlite connection in the same process can hold
    # a lock that blocks deletion.
    scratch_dir = tempfile.mkdtemp(prefix="revenew_live_")
    live_cassette = Cassette(scratch_dir)

    degraded = False
    t0 = time.perf_counter()

    try:
        gen = CandidateGenerator(mode="record", cassette=live_cassette)
        store = PosteriorStore(conn)
        store.ensure_initialized()

        candidate_set = gen.generate(
            opportunity_type=opportunity_type,
            segment=segment,
            rupees_at_risk=rupees_at_risk,
            envelope=envelope,
            store=store,
            policy=policy,
            catalog=catalog,
            conn=conn,
        )
    except Exception as exc:
        # Groq timeout, rate limit, auth error, anything — fall through to
        # the committed cassette and tell the audience what happened.
        degraded = True
        latency_ms = round((time.perf_counter() - t0) * 1000)

        yield _sse_event("degraded", {
            "reason": f"Live LLM call failed ({type(exc).__name__}: {exc}). "
                      "Falling back to the committed cassette.",
            "latency_ms": latency_ms,
        })

        # Fallback: replay from the committed cassette (the default dir).
        fallback_gen = CandidateGenerator(mode="replay", cassette=Cassette())
        store = PosteriorStore(conn)
        store.ensure_initialized()
        candidate_set = fallback_gen.generate(
            opportunity_type=opportunity_type,
            segment=segment,
            rupees_at_risk=rupees_at_risk,
            envelope=envelope,
            store=store,
            policy=policy,
            catalog=catalog,
            conn=conn,
        )

    if not degraded:
        latency_ms = round((time.perf_counter() - t0) * 1000)
        yield _sse_event("llm_done", {"latency_ms": latency_ms})

    if candidate_set is None:
        yield _sse_event("error", {"message": "no candidates available (LLM and cassette both missed)"})
        return

    # ── 4. Emit candidates ─────────────────────────────────────────────
    for i, c in enumerate(candidate_set.candidates):
        yield _sse_event("candidate", {
            "index": i,
            "action_family": c.action_family.value,
            "headline": c.headline,
            "discount_pct": c.discount_pct,
            "discount_amount": c.discount_amount,
            "skus": c.skus,
            "rationale": c.rationale,
        })

    # ── 5. Validate ────────────────────────────────────────────────────
    order_value = _customer_order_value(conn, customer_id)

    verdicts = EnvelopeValidator.validate_all(
        conn, envelope, candidate_set.candidates,
        customer_id=customer_id, order_value=order_value, now=now,
    )

    for i, v in enumerate(verdicts):
        yield _sse_event("verdict", {
            "index": i,
            "action_family": v.candidate.action_family.value,
            "headline": v.candidate.headline,
            "valid": v.valid,
            "violations": v.violations,
        })

    valid = [v.candidate for v in verdicts if v.valid]
    if not valid:
        # All candidates invalid — still a real outcome, show it.
        yield _sse_event("no_action", {
            "reason": "all_candidates_invalid",
            "message": "Every candidate was struck down by the envelope validator.",
        })
        # Persist the no-action decision so it's in the audit trail.
        decision_id = str(uuid.uuid4())
        decision = Decision(
            decision_id=decision_id,
            opportunity_id=live_opp_id,
            run_id=run_id,
            segment=segment,
            action_family=None,
            envelope=envelope,
            candidates=verdicts,
            chosen_candidate=None,
            propensity=None,
            status=DecisionStatus.NO_ACTION,
            no_action_reason=NoActionReason.ALL_CANDIDATES_INVALID,
            created_at=now,
        )
        persist_decision(conn, decision)
        return

    # ── 6. Bandit choice ───────────────────────────────────────────────
    decision_id = str(uuid.uuid4())

    # Use a stable seed derived from customer_id so repeated demo clicks
    # for the same customer produce consistent choices.
    import hashlib
    bandit_seed = int(hashlib.sha256(
        customer_id.encode()
    ).hexdigest(), 16) % (2**31)

    scorer = BanditScorer(store, seed=bandit_seed)
    choice = scorer.choose(segment, valid, fallback_revenue=rupees_at_risk)

    yield _sse_event("bandit", {
        "decision_id": decision_id,
        "chosen_family": choice.candidate.action_family.value,
        "chosen_headline": choice.candidate.headline,
        "propensity": round(choice.propensity, 4),
    })

    # ── 7. Persist + Execute ───────────────────────────────────────────
    cost = choice.candidate.estimated_cost(order_value)

    decision = Decision(
        decision_id=decision_id,
        opportunity_id=live_opp_id,
        run_id=run_id,
        segment=segment,
        action_family=choice.candidate.action_family,
        envelope=envelope,
        candidates=verdicts,
        chosen_candidate=choice.candidate,
        propensity=choice.propensity,
        status=DecisionStatus.PENDING,
        no_action_reason=None,
        created_at=now,
    )
    persist_decision(conn, decision)
    budget.reserve(conn, decision_id, cost, now=now)

    if adapter is None:
        adapter = build_adapter()

    spec = OfferSpec(
        decision_id=decision_id,
        customer_id=customer_id,
        action_family=choice.candidate.action_family,
        headline=choice.candidate.headline,
        amount=cost,
        discount_pct=choice.candidate.discount_pct,
        discount_amount=choice.candidate.discount_amount,
        skus=choice.candidate.skus,
    )
    result = execute_decision(conn, adapter, decision_id=decision_id, spec=spec, now=now)

    if result.status == "failed":
        budget.release(conn, decision_id, cost, now=now)
        yield _sse_event("execution", {
            "decision_id": decision_id,
            "provider_ref": result.provider_ref,
            "status": "failed",
        })
        return

    mark_executed(conn, decision_id)

    yield _sse_event("execution", {
        "decision_id": decision_id,
        "provider_ref": result.provider_ref,
        "status": result.status,
        "degraded": degraded,
    })


def _customer_order_value(conn: sqlite3.Connection, customer_id: str) -> float:
    """Same logic as decide/__init__.py's _customer_order_value — duplicated
    here rather than imported because decide/__init__.py's version is a
    module-private function (leading underscore), and the boundary this module
    respects is calling the same PIECES, not reaching into private helpers."""
    row = conn.execute(
        "SELECT AVG(amount) AS avg_amount FROM orders "
        "WHERE customer_id = ? AND status = 'captured'",
        (customer_id,),
    ).fetchone()
    return float(row["avg_amount"]) if row and row["avg_amount"] is not None else 0.0


@router.post("/api/live/decide")
def live_decide(
    request: DecideRequest | None = None,
    conn: sqlite3.Connection = Depends(get_conn),
):
    """SSE stream of a live decision pipeline.

    Returns `text/event-stream`: the client reads events as they arrive,
    watching the pipeline unfold step by step.  FastAPI runs the sync
    generator in a threadpool, which suits the existing sync sqlite3 —
    no async rewrite needed."""
    if request is None:
        request = DecideRequest()
    return StreamingResponse(
        _decide_generator(conn, request, DEFAULT_POLICY, adapter=None),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# POST /api/live/revalidate  →  deterministic re-verdict
# ---------------------------------------------------------------------------

class RevalidateRequest(BaseModel):
    """The judge moved a slider.  Re-run validation under a tightened cap."""
    decision_id: str
    max_discount_pct: float = Field(ge=0, le=1)
    max_absolute_discount: float | None = Field(default=None, ge=0)


@router.post("/api/live/revalidate")
def live_revalidate(
    request: RevalidateRequest,
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Re-run `EnvelopeValidator.validate_all` over ALREADY-STORED candidates
    for a given decision, against a tightened cap.

    Deterministic, instant, no model call.  The judge moves the discount-cap
    slider and watches the AI's chosen offer get struck down live.

    This endpoint cannot be allowed to fail on stage — it reads only from the
    database and applies pure, deterministic rules.  If the decision_id
    doesn't exist, it returns a clear 404 rather than a traceback."""
    # Read the decision row
    dec_row = conn.execute(
        "SELECT decision_id, segment, action_family, envelope_json, "
        "       chosen_candidate_json, opportunity_id "
        "FROM decisions WHERE decision_id = ?",
        (request.decision_id,),
    ).fetchone()
    if dec_row is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"no decision with id {request.decision_id!r}")

    # Read stored candidates
    cand_rows = conn.execute(
        "SELECT candidate_index, candidate_json FROM decision_candidates "
        "WHERE decision_id = ? ORDER BY candidate_index",
        (request.decision_id,),
    ).fetchall()
    if not cand_rows:
        return {"decision_id": request.decision_id, "verdicts": [], "message": "no candidates stored"}

    candidates = [Candidate.model_validate_json(r["candidate_json"]) for r in cand_rows]

    # Build a tightened envelope from the stored one, with the judge's new cap
    stored_envelope = Envelope.model_validate_json(dec_row["envelope_json"])
    tightened = stored_envelope.model_copy(update={
        "max_discount_pct": request.max_discount_pct,
        **({"max_absolute_discount": request.max_absolute_discount}
           if request.max_absolute_discount is not None else {}),
    })

    # Look up the customer for this decision
    opp_row = conn.execute(
        "SELECT customer_id FROM opportunities WHERE opportunity_id = ?",
        (dec_row["opportunity_id"],),
    ).fetchone()
    customer_id = opp_row["customer_id"] if opp_row else "unknown"
    order_value = _customer_order_value(conn, customer_id)

    # Re-run validation — deterministic, instant, no model call.
    # Uses only the pure envelope rules (violations()), not the
    # customer-history-dependent cooldown/monthly-cap checks, because those
    # are customer-state-at-decision-time properties that a slider shouldn't
    # change.  The point of this endpoint is "tighten the cap, watch the
    # offer die" — cooldown is irrelevant.
    verdicts = []
    for c in candidates:
        violations = list(tightened.violations(c))
        # Also check budget against the tightened envelope
        cost = c.estimated_cost(order_value)
        if cost > tightened.budget_remaining and "budget_remaining" not in violations:
            violations.append("budget_remaining")
        verdicts.append({
            "action_family": c.action_family.value,
            "headline": c.headline,
            "discount_pct": c.discount_pct,
            "discount_amount": c.discount_amount,
            "valid": len(violations) == 0,
            "violations": violations,
        })

    # Identify the originally chosen candidate
    chosen_json = dec_row["chosen_candidate_json"]
    chosen = Candidate.model_validate_json(chosen_json) if chosen_json else None
    chosen_now_valid = None
    if chosen is not None:
        chosen_violations = list(tightened.violations(chosen))
        cost = chosen.estimated_cost(order_value)
        if cost > tightened.budget_remaining and "budget_remaining" not in chosen_violations:
            chosen_violations.append("budget_remaining")
        chosen_now_valid = len(chosen_violations) == 0

    return {
        "decision_id": request.decision_id,
        "original_max_discount_pct": stored_envelope.max_discount_pct,
        "tightened_max_discount_pct": request.max_discount_pct,
        "chosen_family": dec_row["action_family"],
        "chosen_headline": chosen.headline if chosen else None,
        "chosen_now_valid": chosen_now_valid,
        "verdicts": verdicts,
    }
