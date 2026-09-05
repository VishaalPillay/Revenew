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
    received_at     TEXT NOT NULL,
    -- 1 once a real HMAC check has passed for this delivery, 0 for the
    -- placeholder-secret "accept and warn" path. Lets the Rail console show
    -- verification status honestly rather than implying every recorded
    -- delivery was cryptographically checked when the secret may not have
    -- been configured yet.
    signature_verified INTEGER NOT NULL DEFAULT 0
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
    detected_at         TEXT NOT NULL,
    -- Only cross_sell_affinity's query populates this; NULL for every other
    -- opportunity_type. Previously computed by that query and discarded at
    -- its final SELECT -- see detect/queries.sql. Feeds the template shelf's
    -- BUNDLE_OFFER (harness/ablation) and the agent channel's catalog
    -- awareness (revenew/agent), neither of which existed when this column
    -- was added; it is recovered here because the data was already being
    -- thrown away, not because either consumer exists yet.
    recommended_sku     TEXT REFERENCES products(sku)
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
    created_at          TEXT NOT NULL,
    -- 'agent' marks a decision made through the agent-commerce negotiation
    -- path (revenew/agent/negotiate.py) rather than the internal detector ->
    -- bandit path. Same table, same trace machinery, same holdout discipline
    -- -- so agent-driven revenue is measured, not a separate untracked
    -- channel. Defaults to 'internal' so every existing call site (and every
    -- test) that persists a decision without knowing this column exists
    -- keeps working unchanged.
    channel             TEXT NOT NULL DEFAULT 'internal'
                             CHECK (channel IN ('internal', 'agent'))
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

-- Two series live in this one table, and the distinction matters more than it
-- looks. Regret means "how much worse was the action you CHOSE than the best
-- one available". A `no_action` decision forced by the cooldown/monthly-cap
-- envelope was never a choice at all -- `BanditScorer.choose()` is not even
-- reached for it (see revenew/decide/__init__.py: the validator drops every
-- candidate first). On a 30-day run those are 96.7% of all decisions, each
-- carrying a fixed baseline regret the bandit has no way to improve on, which
-- drowns the learning signal completely: per-decision regret over ALL
-- decisions falls 104.4 -> 99.1 across the run (5%, a straight line on a
-- chart), while over the decisions the bandit actually made it falls
-- 82.8 -> 69.2 (16%).
--
-- So: `cumulative_regret` is the whole-system number (total money left on the
-- table, dominated by eligibility policy), and `bandit_cumulative_regret` --
-- populated only where `bandit_chose = 1` -- is the learning curve. The
-- dashboard plots the second by default because "did the bandit learn" is the
-- question it is being asked. Both are honest; they answer different things.
CREATE TABLE demo_regret_curve (
    run_id                   TEXT NOT NULL,
    decision_index           INTEGER NOT NULL,
    decision_id              TEXT NOT NULL,
    segment                  TEXT NOT NULL,
    regret                   REAL NOT NULL,
    cumulative_regret        REAL NOT NULL,
    -- 1 when the bandit actually ran and picked an action_family; 0 for a
    -- no_action decision that never reached it.
    bandit_chose             INTEGER NOT NULL DEFAULT 0 CHECK (bandit_chose IN (0, 1)),
    -- Index and running total WITHIN the bandit-only series. NULL when
    -- bandit_chose = 0, so the learning curve is a plain WHERE away.
    bandit_decision_index    INTEGER,
    bandit_cumulative_regret REAL,
    PRIMARY KEY (run_id, decision_index)
);

-- "Did the bandit learn?" asked directly, in the bandit's own terms: of the
-- decisions it made in this slice of the run, what share landed on the action
-- ground truth says is best for that segment? Chance is 20% (five families).
-- Written by harness/regret.py's `learning_curve` -- like the regret curve,
-- only the resulting scalars cross the isolation boundary, never TRUTH itself.
CREATE TABLE demo_learning_curve (
    run_id              TEXT NOT NULL,
    decision_index      INTEGER NOT NULL,
    n                   INTEGER NOT NULL,
    optimal_rate        REAL NOT NULL,
    regret_per_decision REAL NOT NULL,
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

-- PLAN.md section 5's three-arm ablation: one row per (run_id, arm), scalars
-- only -- same one-way-export discipline as the two tables above. Written by
-- harness/ablation.py after all three arms have run into their own scratch
-- databases and been reduced to numbers; never by revenew/.
--
-- Caveats are COLUMNS, not something a reader has to know to go dig for in a
-- write-up: `explores` (0 for Arm A's constant greedy policy, 1 for Thompson
-- arms), `bundle_reachable` (0 unless a global affinity pair survived the
-- confidence threshold -- Arm A never offers BUNDLE_OFFER at all), and
-- `beats_best_constant` (whether this arm's own converged regret/decision is
-- actually better than the free floor computed over its own decision mix --
-- see harness/regret.py's `best_constant_policy_floor`). A UI or report that
-- reads this table renders all three automatically; nobody has to remember
-- to ask.
CREATE TABLE demo_ablation_arm (
    run_id                          TEXT NOT NULL,
    arm                             TEXT NOT NULL,   -- 'A_deterministic' | 'B_bandit' | 'C_agentic'
    label                           TEXT NOT NULL,   -- human-readable, for direct display
    n_customers                     INTEGER NOT NULL,
    n_days                          INTEGER NOT NULL,
    candidates_composed             INTEGER NOT NULL,
    decisions_executed              INTEGER NOT NULL,
    decisions_no_action             INTEGER NOT NULL,
    -- "first" = earliest learning-curve slice, "last" = the CONVERGED slice
    -- (the last 20%) -- the headline number is regret_per_decision_last, per
    -- PLAN.md section 5's metric-discipline note: cumulative optimal_rate
    -- rewards a constant policy that got lucky early and stopped exploring.
    optimal_rate_first              REAL NOT NULL,
    optimal_rate_last               REAL NOT NULL,
    regret_per_decision_first       REAL NOT NULL,
    regret_per_decision_last        REAL NOT NULL,
    best_constant_family            TEXT NOT NULL,
    best_constant_regret_per_decision REAL NOT NULL,
    explores                        INTEGER NOT NULL CHECK (explores IN (0, 1)),
    bundle_reachable                INTEGER NOT NULL CHECK (bundle_reachable IN (0, 1)),
    beats_best_constant             INTEGER NOT NULL CHECK (beats_best_constant IN (0, 1)),
    elapsed_seconds                 REAL NOT NULL,
    PRIMARY KEY (run_id, arm)
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

-- v_candidate_validity above answers "what fraction of candidates survived
-- validation", which conflates two completely different things and badly
-- understates the model. A candidate can be dropped because:
--
--   POLICY      the model proposed something illegal -- a discount over the
--               cap, an excluded SKU, a spend over the remaining budget.
--               This is the LLM's fault, and it is the number the safety
--               claim in SYSTEM_DESIGN.md section 1.2 actually rests on.
--
--   ELIGIBILITY this customer cannot receive ANY offer right now (cooldown,
--               monthly cap). Nothing to do with what was proposed -- it
--               invalidates every candidate for that customer identically,
--               however good they are.
--
-- On a 30-day/3,000-customer run those were 0 and 313,949 respectively out of
-- 324,578 candidates: the model never once proposed an illegal offer, while
-- the headline "3% validity" made it look like it almost always did. Splitting
-- them is not presentation polish -- reporting the conflated number as a model
-- quality metric is simply wrong.
CREATE VIEW v_candidate_compliance AS
WITH classified AS (
    SELECT
        dc.decision_id,
        dc.candidate_index,
        dc.valid,
        -- ONLY the three rules the MODEL itself can break by composing a bad
        -- offer. This is the number the safety claim rests on, so nothing
        -- that is not a model error may be counted here.
        MAX(CASE WHEN v.value IN (
            'max_discount_pct', 'max_absolute_discount', 'excluded_skus'
        ) THEN 1 ELSE 0 END) AS broke_policy,
        MAX(CASE WHEN v.value IN (
            'cooldown_days', 'max_offers_per_customer_per_month'
        ) THEN 1 ELSE 0 END) AS blocked_eligibility,
        -- `budget_remaining` used to be lumped in with broke_policy above,
        -- and that was the same category error this view already exists to
        -- undo for cooldown/max_offers: it is not a property of the offer the
        -- model composed, it is a property of how much campaign money is left
        -- when the offer is costed. A perfectly legal 15%-off bundle becomes
        -- a `budget_remaining` violation purely because the ledger drained.
        -- It stayed invisible while offers were cheap enough that the budget
        -- never bound; enriching the prompt with the real catalog made
        -- candidates cost real money, the cap started binding, and the panel
        -- dropped from 100% to 93.7% with the model still having proposed
        -- exactly ZERO illegal offers. Counted separately, never against the
        -- model.
        MAX(CASE WHEN v.value = 'budget_remaining'
            THEN 1 ELSE 0 END) AS blocked_budget
    FROM decision_candidates dc
    -- LEFT JOIN ... ON 1=1 so a candidate with an EMPTY violations array (the
    -- valid ones) still contributes a row; a plain comma-join against a
    -- table-valued function drops exactly the rows we most want to count.
    LEFT JOIN json_each(dc.violations_json) v ON 1 = 1
    GROUP BY dc.decision_id, dc.candidate_index
)
SELECT
    COUNT(*)                                                        AS total_generated,
    SUM(valid)                                                      AS total_valid,
    SUM(broke_policy)                                               AS policy_violations,
    SUM(blocked_eligibility)                                        AS eligibility_blocked,
    SUM(blocked_budget)                                             AS budget_blocked,
    1.0 - (CAST(SUM(broke_policy) AS REAL) / NULLIF(COUNT(*), 0))   AS policy_compliance_rate
FROM classified;
