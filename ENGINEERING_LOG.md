# Engineering Log

Chronological, honest, and this is the file that answers "what broke." Every
entry below is a real bug that existed in a working state of this repository
at some point, found by running the code against real (synthetic) data rather
than by inspection. None of them were caught by reading the code twice.

---

## Build order followed

Stages 1-9 exactly as sequenced in `SYSTEM_DESIGN.md` section 12: schema and
clock first, the fixture and its declared ground truth second (so measurement
exists before anything measurable does), detection/arbitration/arm assignment
third, the outcome ledger and incremental estimator fourth -- "the core claim
becomes measurable" -- envelope fifth, the bandit and replay-equality test
sixth, regret and the dashboard seventh, LLM candidate generation eighth,
Razorpay adapters ninth. Stages 1-7 were fully working, with templated
candidates, before stage 8 (the LLM) was written at all.

---

## Real bugs found while building, in order

### 1. `outcomes` could not represent a control-arm outcome at all

Original schema had `outcomes.decision_id NOT NULL REFERENCES decisions`. But
control-arm opportunities never reach `decide_one_opportunity` -- that is
exactly what makes them the counterfactual -- so they never get a `decisions`
row. As written, the schema made it *impossible* to record a control arm's
outcome, which would have broken `IncrementalEstimator`'s entire
treatment-minus-control calculation before a single row was inserted. Fixed
by linking `outcomes` to `opportunities` (which both arms have) and making
`decision_id` nullable. Found before any code was written against the old
shape, by re-reading what `IncrementalEstimator` would actually need to query.

### 2. Fixture population: recency drifted with order count

`generate_population`'s first draft placed each customer's orders at
independently-drawn offsets over their whole history. The *last* order's
recency is then the max of several draws, which drifts toward the present as
order count grows -- so a DORMANT customer with several orders was landing
with a recent last order and getting classified ACTIVE. Segment mix came out
as `active=971, dormant=65` against an intended `active=491, dormant=413`.
Fixed by setting the last order's recency directly and placing earlier orders
progressively further back from it, rather than leaving recency to emerge
from independent draws. Verified with an exact intended-vs-computed segment
comparison (0 mismatches out of 2,000 after the fix).

### 3. Fixture: `orders_count=1` was double-counted as dormant

A follow-on to #2: `POPULATION_SHAPE`'s DORMANT range allowed `orders_count=1`,
but `detector.segment_of()` classifies orders_count<=1 as NEW regardless of
age (a customer with one old order never had the chance to lapse -- that's
FIRST_ORDER_RETENTION's population, not dormant winback's). ~20% of intended
DORMANT customers were rolling exactly 1 order and getting reclassified NEW by
the detector's own, correct rule. Fixed by starting DORMANT's order-count
range at 2.

### 4. Bandit: O(decisions) database reads inside a 4,000-sample Monte Carlo loop

`_estimate_propensity` re-read the posterior table from `self.store.get()`
inside every one of 4,000 Monte Carlo draws, for every family, for every
decision. At a few thousand decisions a day this was not slow, it was
catastrophic: a 30-day replay (target ~40s, per SYSTEM_DESIGN.md section 6's
"30 virtual days in ~40 seconds") had not finished after 3 minutes. Fixed by
reading each family's posterior parameters ONCE per `choose()` call and
sampling from plain numpy arrays afterward -- the math is identical, only the
number of database round-trips changed (from ~4-5 per sample to ~4-5 per
decision).

### 5. SQLite default `synchronous=FULL` was forcing an fsync per commit

Never explicitly set, so it stayed at SQLite's default. In WAL mode this is
unnecessarily strong -- `NORMAL` keeps the same crash safety WAL already
provides and only trades away durability of the last few commits against a
full power loss, which is an acceptable trade for a batch/replay system this
codebase's own NFR table already calls "best effort" availability. Set
explicitly in `revenew/db.py`.

### 6. `decide_one_opportunity` re-initialized posteriors on every call

`PosteriorStore.ensure_initialized()` -- a 20-row `INSERT OR IGNORE` batch
plus a commit -- was being called unconditionally inside
`decide_one_opportunity`, once per decision, even though every real entry
point already calls it once up front. At ~1,240 treatment decisions/day this
was a meaningful share of the same performance problem as #4 and #5. Removed;
documented as a caller responsibility instead.

### 7. Missing index made the replay O(decisions^2)

`EnvelopeValidator`'s cooldown and monthly-offer-cap checks join `decisions`
to `opportunities` filtered on `opportunities.customer_id` -- and
`opportunities.customer_id` had no index. SQLite had no way to seek a
customer's opportunities and fell back to scanning every decision made so
far, for every new decision. By day 30 of a 2,000-customer replay that is
tens of thousands of decisions each triggering a scan proportional to
decisions-so-far -- fixes #4-#6 brought 5 days down to 6 seconds, but 30 days
still would not finish inside 3 minutes until this index was added. After
adding `idx_opportunities_customer`, the full 30-day/2,000-customer replay
runs in ~31 seconds, inside the ~40s target.

### 8. Non-deterministic replay: `hash()`, and two `uuid4()`s that fed the RNG

The most important bug in this list, because it directly contradicted the
project's own non-negotiable claim (SYSTEM_DESIGN.md section 1.2: "same
fixture + seed => byte-identical posteriors").  Three compounding sources of
nondeterminism:

- `bandit_seed=hash((run_id, opportunity_id))` used Python's built-in
  `hash()`, which is salted per-process (`PYTHONHASHSEED`) by design, so the
  identical input produces a different value in every separate process.
- `opportunity_id` was `str(uuid.uuid4())` in the detector -- random on every
  detection run, not a function of what was actually detected.
- `run_id` carried a `uuid.uuid4().hex[:8]` suffix in `run_replay`, also
  random per call.

Any one of these breaks "run the same seed twice, get the same posteriors."
Fixed all three: `opportunity_id` is now content-addressed
(`sha256(run_id|window_id|opportunity_type|customer_id)`), `run_id` is purely
`f"replay_{seed}"`, and the bandit seed uses `hashlib.sha256`, not the
built-in `hash()`. Verified, not just argued: ran the identical seed in two
separate Python processes with different `PYTHONHASHSEED` values (1 and
99999) and diffed the resulting `posteriors` tables and all 8,959 decision
rows -- exact match, zero mismatches. `tests/test_replay_equality.py` checks
live-vs-rebuilt-from-log agreement *within* one process; it would not have
caught this class of bug, which only shows up *across* processes. Worth
recording as a gap in the test suite as currently written, not just a bug
that got fixed.

### 9. FastAPI + sqlite3 thread-affinity mismatch

`sqlite3.Connection` checks that it's used from the thread that created it.
FastAPI runs a sync generator dependency's setup/teardown via
`run_in_threadpool` while an `async def` endpoint itself runs on the event
loop thread -- so a connection created in the dependency and used in the
endpoint crossed threads and raised `ProgrammingError`. Fixed with
`check_same_thread=False` on the connection: safe here because each
connection is request-scoped and never accessed concurrently by more than one
thread, which is the actual property SQLite's check exists to protect and the
property that still holds.

### 10. Budget default was sized for one campaign, not a 30-day replay

Not a code bug, a configuration mismatch worth recording anyway because it
initially looked like one: `PolicyConfig`'s default `budget_cap` (Rs 50,000)
exhausted inside day 0 of a 2,000-customer replay, after which every
subsequent day of the 30-day run was almost entirely `no_action` /
`budget_exhausted` -- correct behaviour, demonstrating the wrong thing (the
already-unit-tested exhaustion path, not the learning behaviour a 30-day
replay exists to show). `harness/run_replay.py` now scales its default budget
to the population size rather than inheriting the single-campaign default.

---

## Known limitations, disclosed rather than hidden

**No Anthropic credential exists in this development environment.** Every
full replay run to date has exercised `CandidateGenerator`'s degraded
template-fallback path exclusively -- one candidate per decision, chosen by a
greedy point-estimate argmax over the posterior, not genuine Thompson-sampled
exploration across several LLM-composed options. `BanditScorer.choose()` is
built, tested, and correctly implements Thompson sampling with cold-start
pooling (verified directly: cold cells borrow cross-segment evidence, cells
past 20 observations detach and use their own posterior) -- but a full
replay's regret curve and posterior-recovery numbers, AS MEASURED SO FAR,
report on the fallback path's performance, not the full system's. The
regret/recovery machinery (`harness/regret.py`) is verified correct via two
independent implementations (Python and SQL) agreeing exactly on a real run
(1,002,998.2 == 1,002,998.2 cumulative regret, 8,959/8,959 matching decision
rows) -- what's unverified is how the NUMBERS look once the LLM is actually in
the loop, because it never has been in this environment.

**`LiveAdapter` (Razorpay) has never been exercised against a real
credential or the live API.** The `create_offer` -> Payment Link mapping is
an explicit, stated assumption (SYSTEM_DESIGN.md section 11), not a verified
integration. `FixtureAdapter` is what every test and every replay run
actually exercises.

**Webhook signature verification is not implemented.** The exact Razorpay
webhook payload shape and signing header are labelled UNKNOWN in the
assumption ledger, pending a captured real payload. Shipping a guessed
verifier that always passes would be worse than an honest gap.

**Budget is a single, non-renewing pool for the life of a run.** There is no
periodic (e.g. monthly) reset. This is a scope decision, not an oversight --
`Envelope.budget_remaining` in SYSTEM_DESIGN.md section 5 has no period
concept, and adding one was not asked for. `run_replay`'s budget scaling
(bug/note #10 above) works around the mismatch this creates for a multi-week
simulation without changing the underlying semantics.
