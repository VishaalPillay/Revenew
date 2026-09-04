-- Harness schema. Lives in harness.db, a file the runtime process never opens.
--
-- This table is the ONLY place the true response rates exist. The fixture
-- generator (harness/fixture.py) writes it once, at the start of a replay run,
-- and the runtime (revenew/) is fed the resulting event stream through the
-- exact same webhook/detection path a live merchant would use. Nothing in
-- revenew/ ever queries this table directly -- there is no code path that
-- could, since the runtime's database connection never attaches this file.
--
-- v_cumulative_regret and v_posterior_recovery are NOT defined here as
-- ordinary views, because SQLite views cannot embed an ATTACH: the join
-- against revenew.db's `decisions` and `posteriors` tables requires the
-- calling connection to attach that file first. harness/regret.py does this
-- explicitly (`ATTACH DATABASE ... AS runtime`) and creates them as TEMP
-- views on that connection. That attachment happens only inside harness code,
-- read-only, after a run completes -- never inside the runtime process.

CREATE TABLE ground_truth (
    segment         TEXT NOT NULL,
    action_family   TEXT NOT NULL,
    p_convert       REAL NOT NULL CHECK (p_convert >= 0 AND p_convert <= 1),
    mean_revenue    REAL NOT NULL,
    PRIMARY KEY (segment, action_family)
);

-- The counterfactual truth for doing nothing, per segment. Needed so the
-- oracle in regret.py can compare "best action" against "no action" and not
-- just against the worst action -- a fixture where every action is worse than
-- doing nothing must let the oracle choose that, too.
CREATE TABLE ground_truth_baseline (
    segment         TEXT PRIMARY KEY,
    p_convert       REAL NOT NULL CHECK (p_convert >= 0 AND p_convert <= 1),
    mean_revenue    REAL NOT NULL
);

-- Config the fixture used to generate this run, kept for reproducibility of
-- the fixture itself (distinct from the runtime's own replay determinism).
CREATE TABLE fixture_meta (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
);
