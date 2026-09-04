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

`FixtureAdapter` is still what every replay run exercises -- `LiveAdapter`
is opt-in, never the default -- but it has now been verified against the real
API at least once; see bug #14.

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

## 13. Groq's `strict: true` rejected the schema outright -- every real call
silently degraded to the template shelf

Found the moment `GROQ_API_KEY` was added and the first real `--llm record`
run was attempted: every single decision in a 200-customer/5-day smoke run
came back with exactly one candidate and propensity 1.0 -- i.e. bug #11 all
over again, despite #11's fix being fully in place and unit-tested. The
`except Exception: return _template_fallback(...)` branch in `_call()` (the
"connectivity/timeout/API error -> degrade" path, correct behaviour for what
it's meant to catch) was silently swallowing a completely different failure:
calling `_call()` directly, bypassing the try/except, surfaced

    groq.BadRequestError: 400 - invalid JSON schema for response_format:
    'propose_candidates': /$defs/Candidate/required: `required` is required
    to be supplied and to be an array including every key in properties. The
    following properties must be listed in `required`: discount_amount,
    discount_pct, skus

Groq's (and OpenAI's) `strict: true` json_schema mode requires EVERY property
of EVERY object to appear in `required`, with optionality expressed through
the property's own type (`anyOf: [..., {"type": "null"}]`) rather than
omission. Pydantic's `model_json_schema()` does the opposite by default: it
only lists fields with no default, so `discount_pct`/`discount_amount`/
`skus` (all `Optional` with defaults on `Candidate`) were missing, and the
API rejected the request before running any inference at all -- every call,
every time, which is exactly why it looked identical to bug #11 from the
outside. None of `tests/test_generator.py`'s 12 fake-client cases caught
this because a fake client never runs Groq's own schema validator; only a
live call does. Fixed with `_require_every_property()`, a small recursive
transform over the generated schema (including nested `$defs`) run once at
import time, and hardened with a new, fake-free regression test
(`test_candidate_set_schema_lists_every_property_as_required`) that checks
the same invariant statically -- it would have caught this without ever
touching the network.

Verified against the real API afterward, not just re-run past the error: a
direct `_call()` returned a genuine 6-7 candidate response spanning multiple
action families with grounded, cohort-specific rationale text, which then
parsed cleanly through `CandidateSet.model_validate`. A full canonical
30-day/3,000-customer `--llm record` run followed (16 distinct cohorts, one
real API call each, 64.7s total -- the cassette from the smoke run supplied
15 of those 16 for free), then two independent `--llm replay --strict-replay`
runs of the identical seed were diffed directly: `posteriors` tables
identical, all 55,535 decisions identical when keyed by the content-addressed
`opportunity_id` (not `decision_id`, which is intentionally a fresh UUID per
run -- see bug #8), all 70,133 outcomes identical. Propensity across the
1,857 executed decisions now spans 256 distinct values from 0.030 to 0.673
(mean 0.425), and all 5 action families are chosen across all 4 segments --
bug #11 is closed for real, not just in unit tests.

**A separate, non-bug observation surfaced by this same run, worth recording
because it shapes how the regret curve should be read:** of the 55,535
treatment-arm decisions in the 30-day run, 53,678 (96.7%) are `no_action` /
`all_candidates_invalid`, and every single one of those is blocked on
`cooldown_days` + `max_offers_per_customer_per_month` together, never on a
discount-cap or budget violation. This is `EnvelopeValidator` working
correctly, not an envelope bug -- but it is a real property of running a
30-day replay against `DEFAULT_POLICY`'s `cooldown_days=30`,
`max_offers_per_customer_per_month=1` over a fixture population whose order
history is fixed at generation time (no new orders are created from a
conversion during replay, so a customer's detected segment barely moves
inside 30 days). The practical effect: most customers who get one offer
early in the run are then correctly cooldown-blocked for the rest of it, so
the majority of "decisions" the regret curve grades are `no_action` outcomes
graded against `BASELINE`, not genuine bandit choices -- a real dynamic, but
one that dilutes how informative the regret curve's slope is about bandit
learning specifically, since so much of the run's decision volume never
reaches `BanditScorer.choose()` at all. Not fixed here -- `cooldown_days` and
`max_offers_per_customer_per_month` are policy values, not bugs, and
loosening either is a product decision, not a code change to make
unilaterally.

## 14. `LiveAdapter` would have crashed on its very first real call

Found by tracing the installed `razorpay` SDK before ever making a live call
-- the same discipline as bug #12, applied before the fact this time rather
than after a confusing symptom. `LiveAdapter.create_offer`/`create_payment_link`
called `self._client.payment_link.create(payload, idempotency_key=idempotency_key)`,
on the assumption the SDK would turn that kwarg into a dedup header. Reading
`razorpay.resources.payment_link.PaymentLink.create` -> `Resource.post_url`
-> `Client.post` -> `Client.request` showed every one of those forwards
unrecognized kwargs straight through, ending at
`getattr(self.session, method)(url, auth=..., verify=..., **options)` --
i.e. `requests.Session.post(..., idempotency_key=...)`, which `requests` does
not accept. Confirmed live, not just by reading: a direct call with that
exact kwarg raised `TypeError: Session.request() got an unexpected keyword
argument 'idempotency_key'` immediately, before any network request was
even attempted.

Three of this project's own tests exercised `LiveAdapter` and all three
passed anyway, which is the real lesson here: each one mocks
`adapter._client.payment_link.create` with a fake shaped
`def fake(payload, idempotency_key=None)` -- a signature that *accepts* the
exact keyword the real SDK rejects. A mock more permissive than the thing it
stands in for is worse than no mock, because it certifies behavior the real
dependency doesn't have. Fixed by dropping the kwarg from both call sites
(it was never going to work) and tightening all three fakes to
`def fake(payload)`, so passing it back in the future raises the identical
`TypeError` class these tests are meant to guard against, not silently pass.

While tracing this, tested whether the header form actually works instead --
`headers={"X-Razorpay-Idempotency-Key": ...}` -- since the SDK does correctly
route a `headers` kwarg through to the request. It didn't: two live calls
with an identical key and identical payload created two distinct payment
links (`plink_TXu1iO9ysEroc1` and `plink_TXu1iqsLHYpjKP`). Razorpay's Payment
Links API does not deduplicate on this, at least not via this header name --
official docs on the exact header returned 404 on every plausible path tried,
so this is reported as an empirical finding, not a documented one. It doesn't
weaken the actual safety property: `execute_decision` already checks
`executions.idempotency_key` (UNIQUE) before ever calling the adapter, and it
is the sole call site (see `revenew/decide/trace.py`-style single-writer
discipline applied to `revenew/execute/razorpay.py`), so a redelivered
request never reaches the network a second time regardless of what the
provider does or doesn't support. Every docstring that claimed otherwise
(`idempotency_key_for`, the module docstring, one test's docstring) has been
corrected; SYSTEM_DESIGN.md section 11's assumption ledger now carries both
the confirmed payload shape and the rejected idempotency-header claim
explicitly, rather than leaving the original ASSUMPTION row looking settled
when only half of it was ever checked.

Verified end to end after the fix: a real `create_offer` call against
test-mode credentials succeeded (`status="sent"`, a genuine `plink_...`
reference), now runs as `test_live_adapter_create_offer_against_real_razorpay_test_mode`
-- gated on `RAZORPAY_KEY_ID`/`RAZORPAY_KEY_SECRET` being present so it skips
cleanly (not hidden, not hardcoded off) on a machine without credentials,
and is the one test in the suite that touches the real network.

## 15. `SegmentLift.is_significant` broke JSON serialization the moment a
real lift was significant

Found by building the read API (stage 3: JSON endpoints + `revenew` CLI)
and actually running `revenew report --json` against the real 30-day
replay's data, rather than against a hand-built fixture: it crashed with
`TypeError: Object of type bool is not JSON serializable` -- a genuinely
confusing message, because `bool` unqualified sounds like it must mean
Python's own builtin, which every `json.dumps` call handles natively.

`SegmentLift.is_significant` is `-> bool` by its own type annotation but its
body is a bare comparison: `self.ci_low > 0 or self.ci_high < 0`.
`welch_interval` (`revenew/measure/incremental.py`) computes `ci_low`/
`ci_high` via numpy/scipy, so both are `numpy.float64`, and a `numpy.float64`
comparison returns `numpy.bool` -- a real, distinct type from Python's
`bool` (it cannot subclass it; `bool` cannot be subclassed at all) and one
the standard `json` module has never supported. `numpy.float64` itself
serializes fine because it IS a `float` subclass, which is exactly why only
`is_significant` broke and every other numeric field on the same object
didn't -- a plausible reason this went unnoticed through every previous test
and every dashboard render (Jinja2's `{{ }}` just calls `str()`, which numpy
bools handle fine; only strict JSON serialization cares about the type,
not just the value).

Fixed at the source, not downstream: `return bool(self.ci_low > 0 or
self.ci_high < 0)`, so the property's own declared return type is actually
true at runtime for every caller, not just the ones that happen to only
check truthiness. Regression test
(`tests/test_incremental.py`) checks `type(...) is bool` explicitly rather
than `isinstance` -- `isinstance(numpy.bool(True), bool)` is also `False`,
so either assertion would have caught the original bug, but `type() is`
is the stricter, more literal statement of the actual contract being
protected. A third test builds a `SegmentLift` from `welch_interval`'s own
real return values (not a hand-constructed `numpy.float64`) and round-trips
it through `json.dumps`, matching the exact call path that surfaced this
against real data.

## 16. The webhook handler had been rejecting every real Razorpay delivery,
looking for a field that doesn't exist

SYSTEM_DESIGN.md's assumption ledger flagged the exact webhook payload shape
as UNKNOWN from the start, precisely because shipping a guessed verifier
would be worse than an honest gap -- but the ORIGINAL code had already
guessed at something else, quietly: `payload.get("id") or
payload.get("event_id")`, used to dedupe redelivered webhooks. No test ever
caught this because no test had a real payload to check it against.

Captured one for real: exposed the local server through a tunnel, registered
it as a webhook URL in the Razorpay dashboard, and completed a test-mode
payment (which itself failed for an unrelated reason -- Razorpay's checkout
flagged a generic Visa test card as an international card the merchant
account doesn't accept -- but a `payment.failed` event is just as real a
delivery as a `payment.captured` one for this purpose). The real envelope:

```json
{"entity":"event","account_id":"acc_...","event":"payment.failed",
 "contains":["payment"],"payload":{"payment":{"entity":{...}}},
 "created_at":1788522184}
```

There is no `id` or `event_id` field anywhere in that body. Every one of the
three real deliveries captured had been rejected by the running server with
`400 {"error": "missing event id"}` -- confirmed directly in ngrok's request
inspector, which shows the exact response the server sent back. The real
per-delivery identifier is the `X-Razorpay-Event-Id` HTTP header, sent
alongside the body, never inside it. This is exactly the failure mode the
UNKNOWN label was guarding against, except it had already happened silently
on the dedup path rather than the (correctly unimplemented) signature path.

While capturing this, also captured `X-Razorpay-Signature` and verified the
signing scheme directly rather than trusting a remembered pattern: computed
`hmac.new(secret, raw_body, hashlib.sha256).hexdigest()` against the real
`(body, signature)` pair using the actual webhook secret configured in the
Razorpay dashboard, and it matched byte-for-byte on the first attempt. Fixed
both issues together in `revenew/api/webhooks.py`: dedup now reads
`X-Razorpay-Event-Id`, and signature verification is implemented for real
using `hmac.compare_digest` (not `==`, which leaks timing information a
forger could exploit to recover the correct signature byte by byte). A
placeholder secret (still `.env.example`'s `your_webhook_secret_here`, or
unset) degrades to accept-with-a-printed-warning rather than either
hard-failing every webhook before setup is complete or silently accepting
forever with no signal that verification is off.

`tests/test_webhooks.py` replays the exact captured bytes -- not a
reconstructed approximation -- through the real endpoint via FastAPI's
`TestClient`, and asserts it is now accepted. Every version of this handler
before today's fix would have rejected that exact request. Also covers:
redelivery of the same event id is a no-op; a tampered body fails signature
verification even though the signature string is unchanged; a wrong
configured secret is rejected the same as a forgery, since the server can't
tell them apart; a missing `X-Razorpay-Event-Id` header is rejected; and the
placeholder-secret path still accepts an unsigned request, deliberately,
during setup.
