-- Revenew runtime schema. Lives in revenew.db.
--
-- This file and harness_schema.sql are the only two schemas in the system, and
-- they are never opened by the same connection: the runtime process opens only
-- this file, so the isolation between "what the system knows" and "what is
-- actually true" (harness_schema.sql) is a filesystem boundary, not a promise
-- kept by application code. See harness/regret.py for the one place that reads
-- across both, deliberately outside the runtime.
--
-- WAL mode is set by the connection layer (revenew/db.py), not here.

PRAGMA foreign_keys = ON;

-- ============================================================== identity --

CREATE TABLE customers (
    customer_id     TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL
);

CREATE TABLE products (
    sku             TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    price           REAL NOT NULL,
    -- A payment gateway cannot observe cost of goods. NULL means "not supplied
    -- by the merchant", and margin ranking must degrade explicitly rather than
    -- silently treating an unknown cost as zero. See SYSTEM_DESIGN.md section 11.
    cogs            REAL
);

CREATE TABLE orders (
    order_id        TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id),
    placed_at       TEXT NOT NULL,
    amount          REAL NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('captured', 'failed', 'refunded'))
);

CREATE TABLE order_items (
    order_id        TEXT NOT NULL REFERENCES orders(order_id),
    sku             TEXT NOT NULL REFERENCES products(sku),
    qty             INTEGER NOT NULL CHECK (qty > 0),
    unit_price      REAL NOT NULL,
    PRIMARY KEY (order_id, sku)
);

CREATE INDEX idx_orders_customer ON orders(customer_id, placed_at);
CREATE INDEX idx_order_items_sku ON order_items(sku);

-- ============================================================== ingestion --

-- Fast-trigger dedup. `event_id` is whatever the source (Razorpay webhook, or
-- the virtual clock's synthetic feed) calls it; uniqueness is what makes a
-- redelivered webhook a no-op instead of a double detection.
CREATE TABLE events (
    event_id        TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    received_at     TEXT NOT NULL
);

-- =========================================================== opportunities --

-- Raw detector output. One row per (customer, window, opportunity_type) that
-- a named query in detect/queries.sql found. Several rows can name the same
-- customer in the same window -- that is exactly what the arbiter resolves.
CREATE TABLE opportunity_candidates (
    opportunity_id      TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    customer_id         TEXT NOT NULL REFERENCES customers(customer_id),
    opportunity_type    TEXT NOT NULL,
    window_id           TEXT NOT NULL,
    cohort_id           TEXT NOT NULL,
    rupees_at_risk      REAL NOT NULL,
    detector_query_hash TEXT NOT NULL,
    detected_at         TEXT NOT NULL
);

CREATE INDEX idx_candidates_window ON opportunity_candidates(customer_id, window_id);

-- Arbitrated winner: at most one row per (run_id, customer_id, window_id). The
-- UNIQUE constraint is the enforcement mechanism for F2, not application code --
-- a second insert attempt for the same customer/window fails at the database,
-- which is what test_arbiter_uniqueness checks.
CREATE TABLE opportunities (
    opportunity_id  TEXT PRIMARY KEY REFERENCES opportunity_candidates(opportunity_id),
    run_id          TEXT NOT NULL,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id),
    window_id       TEXT NOT NULL,
    segment         TEXT NOT NULL,
    arm             TEXT NOT NULL CHECK (arm IN ('control', 'treatment')),
    assigned_at     TEXT NOT NULL,
    UNIQUE (run_id, customer_id, window_id)
);

-- Without this, EnvelopeValidator's cooldown/max-offers check -- decisions
-- JOIN opportunities WHERE opportunities.customer_id = ? -- has no way to
-- seek a customer's opportunities and falls back to scanning every decision
-- made so far, for every new decision. That is an O(n^2) cost across a
-- replay run, and it is what turned a ~40s, 30-day replay into one that had
-- not finished after 3 minutes.
CREATE INDEX idx_opportunities_customer ON opportunities(customer_id);

-- ================================================================ decisions --

-- One row per treatment-arm opportunity that reached the decision path.
-- Control-arm opportunities are logged (above) and never reach this table --
-- that is what makes them the counterfactual.
CREATE TABLE decisions (
    decision_id         TEXT PRIMARY KEY,
    opportunity_id      TEXT NOT NULL UNIQUE REFERENCES opportunities(opportunity_id),
    run_id              TEXT NOT NULL,
    segment             TEXT NOT NULL,
    action_family       TEXT,               -- NULL when status = 'no_action'
    envelope_json       TEXT NOT NULL,
    candidates_generated INTEGER NOT NULL DEFAULT 0,
    candidates_valid    INTEGER NOT NULL DEFAULT 0,
    chosen_candidate_json TEXT,             -- NULL when status = 'no_action'
    propensity          REAL,               -- NULL when status = 'no_action'
    status              TEXT NOT NULL CHECK (
                             status IN ('executed', 'no_action', 'pending')
                         ),
    no_action_reason    TEXT,               -- NULL unless status = 'no_action'
    created_at          TEXT NOT NULL
);

CREATE INDEX idx_decisions_segment_family ON decisions(segment, action_family);

-- Every candidate the generator produced, with its validator verdict. This is
-- the evidence for candidates_valid / candidates_generated and for
-- test_envelope_invariant: no candidate with a non-empty violations list may
-- ever appear as a decision's chosen_candidate_json.
CREATE TABLE decision_candidates (
    decision_id     TEXT NOT NULL REFERENCES decisions(decision_id),
    candidate_index INTEGER NOT NULL,
    candidate_json  TEXT NOT NULL,
    valid           INTEGER NOT NULL CHECK (valid IN (0, 1)),
    violations_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (decision_id, candidate_index)
);

-- ================================================================ execution --

CREATE TABLE executions (
    execution_id    TEXT PRIMARY KEY,
    decision_id     TEXT NOT NULL UNIQUE REFERENCES decisions(decision_id),
    idempotency_key TEXT NOT NULL UNIQUE,
    provider_ref    TEXT,
    status          TEXT NOT NULL CHECK (status IN ('sent', 'confirmed', 'failed')),
    created_at      TEXT NOT NULL
);

-- Reserve / (implicit commit) / release. A reservation is a negative delta
-- written at decision time, before execution. Execution succeeding leaves it
-- in place permanently -- that IS the commit, no second write needed. A
-- release writes the reversing positive delta. This is what makes a crash
-- between reserve and execute hold budget rather than lose it: nothing is
-- double-spent, and nothing vanishes.
CREATE TABLE budget_ledger (
    entry_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id     TEXT NOT NULL REFERENCES decisions(decision_id),
    status          TEXT NOT NULL CHECK (status IN ('reserved', 'released')),
    amount          REAL NOT NULL,          -- reserved: negative. released: positive, reverses it.
    created_at      TEXT NOT NULL
);

-- ================================================================= outcomes --

-- Append-only by construction: the triggers below reject UPDATE and DELETE
-- outright. `outcome_seq` is the monotonic sequence the replay test rebuilds
-- posteriors from -- see ledger/replay.py.
--
-- Linked to `opportunities`, not `decisions`. Diagram 3 says "both arms" for a
-- reason: IncrementalEstimator needs a converted/not-converted outcome for
-- every opportunity, including control-arm ones that were logged and never
-- actioned -- those never get a `decisions` row at all, since the control arm
-- exists precisely by skipping the decision path. decision_id is therefore
-- NULLable, populated only when a decision (and possibly a bandit reward
-- update) exists to attach it to.
CREATE TABLE outcomes (
    outcome_seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id  TEXT NOT NULL UNIQUE REFERENCES opportunities(opportunity_id),
    decision_id     TEXT UNIQUE REFERENCES decisions(decision_id),  -- NULL for control arm
    converted       INTEGER NOT NULL CHECK (converted IN (0, 1)),
    net_revenue     REAL NOT NULL DEFAULT 0,
    censored        INTEGER NOT NULL DEFAULT 0 CHECK (censored IN (0, 1)),
    closed_at       TEXT NOT NULL
);

CREATE TRIGGER outcomes_no_update
BEFORE UPDATE ON outcomes
BEGIN
    SELECT RAISE(ABORT, 'outcomes is append-only: UPDATE is forbidden');
END;

CREATE TRIGGER outcomes_no_delete
BEFORE DELETE ON outcomes
BEGIN
    SELECT RAISE(ABORT, 'outcomes is append-only: DELETE is forbidden');
END;

-- ============================================================== posteriors --

-- Derived cache over (segment, action_family). Fully rebuildable from
-- `outcomes` by ledger/replay.py -- this table is an optimization, not a
-- source of truth. Fixed at 4 segments x 5 families = 20 possible rows;
-- see revenew/models.py for the two enums that define the grid.
CREATE TABLE posteriors (
    segment             TEXT NOT NULL,
    action_family       TEXT NOT NULL,
    alpha               REAL NOT NULL,
    beta                REAL NOT NULL,
    revenue_sum         REAL NOT NULL DEFAULT 0,
    revenue_n           INTEGER NOT NULL DEFAULT 0,
    updated_through_seq INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (segment, action_family)
);

-- =================================================================== views --

-- Failure-recovery evidence: a measured distribution, not a claim. See
-- SYSTEM_DESIGN.md section 7.
CREATE VIEW v_no_action_reasons AS
SELECT no_action_reason, COUNT(*) AS n
FROM decisions
WHERE status = 'no_action'
GROUP BY no_action_reason;

-- Budget available right now = total configured budget minus every ledger
-- delta that has not been reversed. The budget cap itself is a runtime
-- constant (see revenew/execute/budget.py), not stored here, so this view
-- reports the *consumed* side of the ledger; the caller subtracts from the cap.
CREATE VIEW v_budget_consumed AS
SELECT COALESCE(SUM(-amount), 0) AS consumed
FROM budget_ledger;

-- Decisions still holding a budget reservation with no matching execution
-- row -- either a genuine crash between reserve and execute, or an
-- execution that succeeded but the process died before trace.mark_executed
-- committed. What ledger/reconcile.py sweeps once these age past its
-- timeout. See SYSTEM_DESIGN.md section 7: "Crash between reserve and
-- commit -> action.status='pending' older than timeout -> reconciler
-- releases the hold."
CREATE VIEW v_pending_executions AS
SELECT d.decision_id, d.created_at, d.segment, d.action_family
FROM decisions d
LEFT JOIN executions e ON e.decision_id = d.decision_id
WHERE d.status = 'pending' AND e.execution_id IS NULL;

-- ============================================== demo-only exported artifacts --
--
-- Written ONLY by harness/regret.py, after a replay run, and ONLY these two
-- tables -- everything else in this file is written exclusively by revenew/.
-- The dashboard (revenew/api/dashboard.py) reads them to draw the regret
-- curve, but the runtime process still never opens harness.db to get there:
-- by the time these rows exist, harness code has already reduced ground
-- truth down to a plain (decision_index, cumulative_regret) series with no
-- way to recover the underlying true rates from it. The one-way export IS
-- the isolation boundary in practice, not a exception to it -- see
-- harness/regret.py's `export_to_runtime`.

CREATE TABLE demo_regret_curve (
    run_id              TEXT NOT NULL,
    decision_index      INTEGER NOT NULL,
    decision_id         TEXT NOT NULL,
    segment             TEXT NOT NULL,
    regret              REAL NOT NULL,
    cumulative_regret   REAL NOT NULL,
    PRIMARY KEY (run_id, decision_index)
);

CREATE TABLE demo_posterior_recovery (
    run_id          TEXT NOT NULL,
    segment         TEXT NOT NULL,
    action_family   TEXT NOT NULL,
    n_observed      REAL NOT NULL,
    p_hat           REAL NOT NULL,
    p_true          REAL NOT NULL,
    p_error         REAL NOT NULL,
    revenue_hat     REAL,
    revenue_true    REAL NOT NULL,
    revenue_error   REAL,
    PRIMARY KEY (run_id, segment, action_family)
);

-- Candidate validity: how often the LLM stays inside the envelope on its own,
-- before the validator has to drop anything.
CREATE VIEW v_candidate_validity AS
SELECT
    CAST(SUM(candidates_valid) AS REAL) / NULLIF(SUM(candidates_generated), 0) AS validity_rate,
    SUM(candidates_generated) AS total_generated,
    SUM(candidates_valid) AS total_valid
FROM decisions
WHERE status IN ('executed', 'no_action');
