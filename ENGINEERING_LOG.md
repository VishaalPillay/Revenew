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

**No cassette has been recorded against a real LLM yet (see bugs #11-12
below).** `CandidateGenerator` now supports genuine multi-candidate
generation via Groq, cached per cohort, with `off`/`record`/`replay` modes
and 12 tests passing against fakes -- but every full replay run to date still
predates that fix, and no `--llm record` run has completed against the real
API (blocked first on Anthropic credit, now on adding `GROQ_API_KEY`). Until
that run happens, every existing regret curve and posterior-recovery table
reports on the single-candidate fallback path's greedy argmax, not genuine
Thompson-sampled exploration. `BanditScorer.choose()` is built, tested, and
correctly implements Thompson sampling with cold-start pooling (verified
directly: cold cells borrow cross-segment evidence, cells past 20
observations detach and use their own posterior) -- what's unverified is how
the numbers look once a real, multi-candidate LLM is actually in the loop.
The regret/recovery machinery (`harness/regret.py`) is verified correct via
two independent implementations (Python and SQL) agreeing exactly on a real
run (1,002,998.2 == 1,002,998.2 cumulative regret, 8,959/8,959 matching
decision rows) -- that verification is unaffected by which policy produced
the decisions being graded.

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

---

## 11. The bandit had never actually chosen anything

Found by reading the code, not by a failing test -- which is itself the
lesson: nothing above caught this because everything above was written
*assuming* the LLM path would eventually run, and it never had.
`CandidateGenerator`'s no-credential fallback (`_template_fallback`) always
returned a `CandidateSet` with exactly ONE candidate. `BanditScorer.choose()`
computes `families = sorted({c.action_family for c in candidates})` and
Thompson-samples over that set -- with one candidate, `families` has exactly
one element, `max()` over one value is forced, and
`_estimate_propensity` returns 1.0 on every single decision. Every regret
curve and posterior-recovery table this project had produced up to this
point graded a **greedy argmax over posterior point estimates**, not the
Thompson sampling SYSTEM_DESIGN.md section 6 describes and
`tests/test_replay_equality.py` exercises directly (correctly, but only ever
against hand-built multi-candidate fixtures, never through a real replay).

Fixed by putting a genuine, multi-candidate LLM in the loop for real, which
required solving two things the original single-LLM-call-per-decision design
would have made infeasible or unreproducible: cost (a 3,000-customer,
30-day replay is on the order of tens of thousands of decisions; one call
each is tens of millions of tokens) and determinism (an LLM call inside a
replayed decision path breaks "same seed => byte-identical posteriors").
`revenew/decide/cassette.py` keys generation on the COHORT --
`(opportunity_type, segment, a coarse rupees_at_risk band, an envelope
composition fingerprint)` -- collapsing the call count to a few dozen, and
persists every recorded `CandidateSet` to a committed JSON directory so a
later run reproduces byte-identically without ever calling the API again.
`CandidateGenerator` gained `off`/`record`/`replay` modes, defaulting to
`off` so no existing caller's behaviour changed just because a credential
happened to be present.

## 12. No Anthropic billing available; switched the LLM provider to Groq

While wiring the above fix, `ANTHROPIC_API_KEY` resolved correctly and the
network path to `api.anthropic.com` worked -- but every real call returned
`400 invalid_request_error: Your credit balance is too low`, confirmed with a
minimal direct call (`max_retries=0`, 0.6s to fail). Worth recording
precisely because of what it looked like from the *outside* first: a
`--llm record` replay run produced zero output and zero cassette files for
several minutes before being killed, which looked like a hang. It wasn't --
`_call()`'s failure path degrades to the templated fallback WITHOUT caching
the miss (deliberately: a transient failure must not poison the cassette
with a fake "recording"), so every decision in an affected cohort was
retrying the same instantly-failing request, once per decision, for the
whole run. A `time.perf_counter()` around the very first live call before
trusting a multi-thousand-decision run would have surfaced this in under a
second; killing and re-diagnosing with a bounded, `max_retries=0` direct
script did the same after the fact.

No Anthropic credit was available to add, so the LLM provider was switched to
Groq -- a disclosed deviation from SYSTEM_DESIGN.md section 3.1's stated
"Claude (Sonnet tier)". The swap stayed contained to
`revenew/decide/generator.py`'s `_client()`/`_call()` and its exception
types: cohort-level cache keys, the cassette, the three modes, and the
loud-vs-degrade exception split are all provider-agnostic by construction.
Structured output uses Groq's `response_format={"type": "json_schema", ...,
"strict": True}` rather than Anthropic's forced `strict` tool call --
functionally equivalent (constrained decoding guarantees schema-valid
output), but as of this writing only honored by a handful of Groq-hosted
models (`openai/gpt-oss-20b`/`120b`, `qwen/qwen3-32b`); `GROQ_MODEL` defaults
to the 20b variant. `GROQ_API_KEY` still needs to be added to `.env` before
the actual cassette-recording run can happen -- the generator, cassette, and
all 12 of `tests/test_generator.py`'s cases are verified against fakes, but
no cassette has been recorded against the real API yet.
