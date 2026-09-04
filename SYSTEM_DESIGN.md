# Revenew — System Design

**Status:** Accepted
**Track:** Razorpay AI Buildathon, Track 01 (AI Growth & Agentic Commerce)
**Scope:** Solo build

> A decision layer above Razorpay that finds where a merchant's revenue is leaking,
> proposes bounded commercial actions, learns which ones actually work, and reports
> incremental impact against a held-out control arm.

---

## 1. Requirements

### 1.1 Functional

| # | Requirement |
|---|---|
| F1 | Detect revenue opportunities deterministically from transaction history, each carrying a rupee figure and the query that produced it |
| F2 | Resolve collisions so a customer receives at most one action per window |
| F3 | Assign every opportunity to a control or treatment arm by deterministic hash |
| F4 | Compute a constraint envelope from merchant policy and current budget state |
| F5 | Generate 5–8 candidate offers per opportunity via LLM, constrained by the envelope |
| F6 | Validate every candidate programmatically against the same envelope; drop violators |
| F7 | Rank surviving candidates by Thompson sampling over per-segment action families |
| F8 | Execute the selected action against Razorpay with an idempotency key |
| F9 | Record every decision's full trace, including the propensity of the chosen arm |
| F10 | Close attribution windows and append outcomes, distinguishing censored from failed |
| F11 | Update posteriors from the outcome log, reconstructibly |
| F12 | Report incremental lift as treatment minus control, with a confidence interval |
| F13 | Replay a full fixture run under a virtual clock and produce a regret curve |

### 1.2 Non-functional

| Property | Target | Rationale |
|---|---|---|
| Decision latency (fast path) | < 5 s p95 | Dominated by one LLM call; nothing is user-facing |
| Decision latency (slow path) | Unbounded | Nightly batch, no pressure |
| Throughput | 10k decisions/night | One merchant's monthly volume in a single batch |
| Availability | Best effort | Batch system; a missed night is recoverable, not lost |
| Reproducibility | **Exact** | Same fixture + seed ⇒ byte-identical posteriors. Non-negotiable. |
| Auditability | Every action traceable end to end | Money moves; no unexplained decisions |
| Safety | No action outside envelope, ever | Enforced twice, fails closed |

Reproducibility is ranked above availability on purpose. This system's central claim is
measured learning. A result that cannot be re-derived is not a result.

### 1.3 Constraints

- Solo builder, ~48 hours to submission.
- No production merchant data. All transaction data is synthetic or Razorpay test-mode.
- COGS is not observable from a payment gateway. It must be merchant-supplied or absent.
- Real market/trend signals are not available offline and are not simulated.

---

## 2. Component inventory

Mapped to `docs/img/01`, `02`, `03`.

### Diagram 1 — Ingestion and routing

| Component | Type | Responsibility |
|---|---|---|
| `ClockAdapter` | Deterministic | Sole source of `now()`. Wall implementation or virtual tick driver. |
| `FastTrigger` | Deterministic | Razorpay webhook receiver. Dedupes on event id, appends, wakes the detector. |
| `SlowTrigger` | Deterministic | Nightly cohort rebuild via APScheduler. |
| `OpportunityDetector` | Deterministic | Parameterised SQL. Emits `Opportunity` rows with `rupees_at_risk` and `detector_query_hash`. |
| `AttributionArbiter` | Deterministic | One action per `(customer, window)`. Highest rupees-at-risk wins; tie broken by cohort id. |
| `ArmAssigner` | Deterministic | `crc32(customer_id + salt) % 100 < 20 → control`. Stable across runs. |

### Diagram 2 — Decision path

| Component | Type | Responsibility |
|---|---|---|
| `EnvelopeEngine` | Deterministic | Builds the `Envelope` from policy config, budget balance, cooldown history, COGS table. |
| `CandidateGenerator` | **LLM** | The only LLM in the system. Envelope-in, structured candidates-out. |
| `EnvelopeValidator` | Deterministic | Re-applies every envelope rule programmatically. Records per-candidate verdicts. |
| `BanditScorer` | Deterministic | Thompson sampling over `(segment, action_family)`. Returns choice + propensity. |
| `PosteriorStore` | Derived cache | Beta + revenue moments. Rebuildable from the outcome log. |
| `DecisionTrace` | Sink | JSON record of every stage of one decision. |

### Diagram 3 — Measurement harness

| Component | Type | Responsibility |
|---|---|---|
| `FixtureGenerator` | Harness only | Declares true `(segment, family) → (p_convert, mean_revenue)`. Emits an event stream. |
| `OutcomeLedger` | Append-only | One row per decision when its window closes. Monotonic `outcome_seq`. |
| `BanditRewardFeed` | Estimator | Raw net revenue. Feeds posteriors. |
| `IncrementalEstimator` | Estimator | Treatment minus control per segment, with CI. |
| `RegretCalculator` | Harness only | Compares chosen EV against oracle EV using ground truth. |

**The runtime process never opens the harness database.** `revenew.db` and `harness.db`
are separate SQLite files. The isolation is a file boundary, not a convention.

---

## 3. Stack

### 3.1 Chosen

| Layer | Choice | Why this and not the obvious alternative |
|---|---|---|
| Language | Python 3.11 | Fastest path to statistics + SQL + LLM SDK in one process |
| Storage | **SQLite (WAL)** | Entire system state is one diffable file. Makes replay and reproducibility trivial. Postgres buys concurrency this system does not need. |
| HTTP | FastAPI + uvicorn | Webhook receiver, read API, and the demo page from one process |
| Schemas | Pydantic v2 | Doubles as the LLM structured-output contract and the API contract |
| Scheduler | APScheduler, in-process | Cron for the slow clock. Celery would add a broker for one recurring job. |
| LLM | **Groq** (`openai/gpt-oss-20b`, structured-output strict mode), single call per cohort | Originally specified as Claude (Sonnet tier); switched when no Anthropic billing was available in the build environment. Groq's `response_format` strict `json_schema` mode gives the same schema-guarantee property the design originally got from Anthropic's forced `strict` tool call. See ENGINEERING_LOG.md and `revenew/decide/generator.py`. |
| Sampling | `numpy.random.Generator` seeded per run | Seeded RNG is what makes replay exact |
| Statistics | `scipy.stats` | Welch's t-interval for the reported lift |
| Dashboard | Jinja2 page + Chart.js from CDN | One server, no separate frontend build |
| Tests | pytest | Replay-equality and envelope-invariant tests are the important ones |

### 3.2 Deliberately not used

Stated explicitly because "where you chose not to use a tool" is a scored dimension.

| Rejected | Why |
|---|---|
| **Kafka / RabbitMQ** | The append-only outcome table with a monotonic sequence already gives ordering and replay. A broker adds an operational dependency and no capability. |
| **Redis** | Nothing here is hot enough to need a cache. The posterior table is < 100 rows. |
| **Celery** | One nightly job. APScheduler is 3 lines. |
| **Postgres** | Single writer, single node, < 1 GB. SQLite's file-level portability is a feature for a replayable system. |
| **LangChain / agent frameworks** | There is exactly one LLM call with a fixed schema. An agent loop would add nondeterminism to a system whose core claim is reproducibility. |
| **Vector DB / RAG** | No corpus to retrieve from. Merchant state is structured and fits in a prompt. |
| **LLM for detection** | Counting and joining is what SQL is for. An LLM would be slower, unauditable, and arithmetically unreliable. |
| **LLM for ranking** | Ranking is where evidence should decide, not priors. That's the bandit's job. |
| **Fine-tuning** | No labelled corpus, no time, and the learning we need is per-merchant online adaptation, not weight updates. |

The last three rows are the argument that this is not an AI wrapper.

---

## 4. Repository layout

```
revenew/
├── README.md
├── SYSTEM_DESIGN.md
├── ENGINEERING_LOG.md          # daily; feeds the "what broke" submission field
├── docs/img/                   # the three architecture diagrams
├── db/
│   ├── schema.sql
│   └── harness_schema.sql
├── revenew/
│   ├── clock.py                # ClockAdapter: WallClock | VirtualClock
│   ├── models.py               # Pydantic: Opportunity, Envelope, Candidate, Decision, Outcome
│   ├── detect/
│   │   ├── queries.sql         # one named query per opportunity_type
│   │   └── detector.py
│   ├── route/
│   │   ├── arbiter.py
│   │   └── arm.py
│   ├── decide/
│   │   ├── envelope.py         # build + validate, same rule table
│   │   ├── generator.py        # the only LLM call
│   │   ├── bandit.py           # Thompson sampling, posterior update
│   │   └── trace.py
│   ├── execute/
│   │   ├── razorpay.py         # RazorpayAdapter protocol
│   │   └── budget.py           # reserve / commit / release
│   ├── ledger/
│   │   ├── outcome.py
│   │   └── replay.py           # rebuild posteriors from seq 1..N
│   ├── measure/
│   │   ├── incremental.py
│   │   └── report.py
│   └── api/
│       ├── webhooks.py
│       └── dashboard.py
├── harness/
│   ├── fixture.py              # declares ground truth, emits events
│   ├── regret.py
│   └── run_replay.py           # entrypoint: 30 virtual days in ~40 s
└── tests/
    ├── test_replay_equality.py     # posterior cache == log replay
    ├── test_envelope_invariant.py  # no executed action ever violates its envelope
    ├── test_arbiter_uniqueness.py
    └── test_budget_conservation.py
```

---

## 5. Key interfaces

```python
class Clock(Protocol):
    def now(self) -> datetime: ...

class RazorpayAdapter(Protocol):
    def create_offer(self, spec: OfferSpec, idempotency_key: str) -> ExecutionResult: ...
    def create_payment_link(self, spec: LinkSpec, idempotency_key: str) -> ExecutionResult: ...
```

Two implementations of `RazorpayAdapter`: `LiveAdapter` (test-mode API) and
`FixtureAdapter` (records the call, returns a synthetic id). The demo can run either.
Everything upstream is identical.

```python
class Envelope(BaseModel):
    max_discount_pct: float
    max_absolute_discount: float
    budget_remaining: float
    excluded_skus: list[str]
    cooldown_days: int
    max_offers_per_customer_per_month: int
    cogs_by_sku: dict[str, float] | None   # None = unknown, never 0

    def violations(self, c: Candidate) -> list[str]: ...
```

`violations()` is the single rule table. `EnvelopeEngine` renders it into the prompt;
`EnvelopeValidator` calls it on the output. One definition, two consumers — they cannot
drift apart.

```python
class BanditScorer:
    def choose(self, segment: str, candidates: list[Candidate]) -> tuple[Candidate, float]:
        """Returns the winner and the propensity with which it was chosen."""
```

---

## 6. The learning model

Reward is not Bernoulli, so a plain Beta is wrong. Two-part model per
`(segment_key, action_family)`:

```
p ~ Beta(α, β)                      # conversion
r̄ = revenue_sum / revenue_n         # mean net revenue given conversion
sampled_value = p_sample × r̄
```

Update on each closed window:
- converted → `α += 1`, `revenue_sum += net_revenue`, `revenue_n += 1`
- not converted → `β += 1`
- censored → `β += 1`, flagged. **Censored is not failure**; recording silence as failure
  teaches the bandit to avoid slow-converting arms.

**Priors.** Discount-bearing families start pessimistic (`α=1, β=4`); zero-cost families
start neutral (`α=1, β=1`). This is the cold-start margin guard: on day one the system
will not spray discounts at customers who would have converted anyway.

**Cold start across segments.** Hierarchical shrinkage — a new segment's prior is the
pooled global mean until `n_observed ≥ 20`, then it detaches.

**Segmentation budget.** 4 segments × 5 action families = 20 cells. Fixed deliberately.
More segments look sophisticated and starve every cell; the bandit then appears not to
learn when in fact it is underfed.

**Why two estimators, not one.** The bandit's reward is *raw* net revenue, not the
holdout difference. At cell level the control sample is ~20 customers and differencing
injects more variance than signal. All arms face identical selection bias, so relative
ordering stays valid. The holdout difference is computed once, at segment level, and is
the only number reported externally.

---

## 7. Failure modes

| Failure | Detection | Response | Test |
|---|---|---|---|
| LLM returns malformed JSON | Pydantic parse error | One retry with schema echo, then `no_action_reason='llm_unavailable'` | unit |
| LLM returns policy-violating candidate | `violations()` non-empty | Drop candidate, record verdict in trace | `test_envelope_invariant` |
| All candidates invalid | `candidates_valid == 0` | No action; `no_action_reason='all_candidates_invalid'` | unit |
| LLM unreachable | Timeout / API error | Fall back to highest-posterior family with a templated offer. Degraded, not dead. | unit |
| Razorpay 5xx or timeout | Non-2xx | Retry with backoff, same idempotency key | integration |
| Crash between reserve and commit | `action.status='pending'` older than timeout | Reconciler releases the hold | `test_budget_conservation` |
| Duplicate webhook | Event id already present | Ignore | unit |
| Two opportunities, one customer | `UNIQUE(run_id, customer_id, window_id)` | Insert fails; arbiter must pick one first | `test_arbiter_uniqueness` |
| Outcome never arrives | Window closes with no signal | Append `status='censored'` | unit |
| Posterior cache diverges from log | `updated_through_seq` mismatch | Rebuild from log. **The log is right.** | `test_replay_equality` |
| Budget exhausted mid-run | `v_budget_available <= 0` | No action; `no_action_reason='budget_exhausted'` | unit |

Every `no_action_reason` is queryable via `v_no_action_reasons`. That view is the
failure-recovery evidence — a measured distribution, not a claim.

---

## 8. Measurement methodology

**Primary metric.** Net incremental revenue per eligible customer:

```
lift = mean(net_revenue | treatment) − mean(net_revenue | control)
```

reported per segment with a Welch 95% interval. At demo N the interval will be wide.
Report it wide. A weak number honestly presented outscores a strong number that cannot
be defended.

**Secondary metrics.**

| Metric | Source | What it proves |
|---|---|---|
| Cumulative regret vs oracle | `v_cumulative_regret` | The bandit converges, quantified |
| Posterior recovery error | `v_posterior_recovery` | It found the *true* rates, not just a stable one |
| Candidate validity rate | `decision.candidates_valid / candidates_generated` | How often the LLM stays inside the envelope |
| No-action distribution | `v_no_action_reasons` | Failure handling actually fires |
| Budget conservation | `v_budget_available` vs allocation | No money lost to crashes |

**Anti-metric, stated in the pitch:** gross revenue from customers who received an offer.
Most of them would have bought anyway. Only the differenced number is reported.

**Off-policy evaluation.** Propensities are logged on every treatment decision, so an
IPS estimator can answer "what would revenue have been under a different policy" without
running it. Stretch goal; the logging is cheap and enables it later regardless.

---

## 9. Scale

Current load is trivial: 10k decisions/night, one LLM call each, ~100 posterior rows.
SQLite handles this on a laptop.

What breaks first at 10×–100×, in order:

1. **LLM cost and latency.** 10k calls/night dominates everything. *Fix:* generate per
   `(segment, opportunity_type)` rather than per customer — candidates are cohort-level
   already, so this is caching, not degradation. Cuts calls by ~1000×.
2. **Single-writer SQLite.** At multi-merchant scale, contention. *Fix:* one database
   file per merchant before reaching for Postgres. Merchants share nothing.
3. **Nightly batch window.** *Fix:* shard by merchant; the pipeline is embarrassingly
   parallel across merchants.
4. **Posterior cardinality.** Segments × families × merchants. *Fix:* this is when the
   hierarchical prior stops being a nicety and becomes load-bearing.

Nothing in the current design blocks any of these. That is the reason for the
`RazorpayAdapter` protocol and the injected clock.

---

## 10. Trade-offs made

| Decision | Gained | Paid | Revisit when |
|---|---|---|---|
| SQLite over Postgres | Replayability, zero setup, diffable state | Single writer | Multi-merchant concurrent writes |
| One LLM call, no agent loop | Determinism, auditability, low cost | No multi-step reasoning | A task genuinely needs planning |
| Envelope enforced twice | Model errors can only be suboptimal, never illegal | Duplicate rule evaluation (cheap) | Never |
| Raw reward for bandit, differenced for reporting | Stable learning at small N | Two estimators to explain | Cell sizes exceed ~200 |
| 4 segments only | Every cell gets fed | Coarse personalisation | Volume supports finer cells |
| Fixture with declared ground truth | Regret is measurable, not asserted | Not real-world evidence | Real merchant data available |
| Merchant-supplied COGS | Honest margin, or explicit absence | Requires merchant input | Never — gateways cannot see COGS |

---

## 11. Assumption ledger

| Claim | Label |
|---|---|
| Razorpay exposes test-mode APIs for offers and payment links | **ASSUMPTION** — verify against live docs before writing `LiveAdapter` |
| Exact webhook event names and payload shape | **UNKNOWN** — must capture real payloads; no schema written until then |
| A payment gateway cannot observe COGS | **FACT** — hence merchant-supplied config |
| Thompson sampling converges on stationary Bernoulli bandits | **FACT** — standard result |
| Declared-truth fixtures make regret computable | **FACT** — oracle is known by construction |
| Fixture-measured lift transfers to real merchants | **ASSUMPTION** — explicitly not claimed in the pitch |
| Bandit needs roughly 10k monthly transactions to separate | **INFERENCE** — from cell-size arithmetic, not measured |
| LLM has current market or trend knowledge | **REJECTED** — training cutoff; no unsourced market claims anywhere in the system |

The last row is a design constraint, not a footnote. Every merchant-specific fact in an
LLM output must trace to data placed in the prompt.

---

## 12. Build order

Strictly sequenced. Each stage is demoable on its own, so the submission degrades
gracefully if time runs out.

| # | Stage | Unblocks |
|---|---|---|
| 1 | `schema.sql` + `clock.py` + Pydantic models | Everything |
| 2 | `harness/fixture.py` with declared ground truth | All measurement |
| 3 | Detector + arbiter + arm assignment | Opportunities exist |
| 4 | Outcome ledger + incremental estimator | **The core claim becomes measurable** |
| 5 | Envelope engine + validator | Safety guarantee holds |
| 6 | Bandit + posterior store + replay test | Learning is provable |
| 7 | `regret.py` + dashboard | The centrepiece chart |
| 8 | LLM candidate generation | Replaces templated candidates |
| 9 | `LiveAdapter` against Razorpay test mode | Real execution in the demo |

The LLM lands at stage 8 deliberately. Everything around it must be able to prove
whether it helped before it is allowed in.

Minimum viable submission is stages 1–7 with templated candidates: a measured,
reproducible learning system with a safety guarantee. Stage 8 upgrades it from a
learning system to an AI system. Stage 9 upgrades it from synthetic to real.

---

## 13. Interview defence

| Question | Answer |
|---|---|
| Why does AI need to be here at all? | It doesn't, for detection or ranking — those are SQL and statistics. It's needed for composing a specific offer from catalog, cohort, brand voice, and constraints. That's combinatorial with no closed form. |
| What happens when the model is wrong? | It produces a suboptimal legal action. The envelope is enforced before generation and again after, programmatically, from one rule table. |
| How do you know it worked? | 20% holdout, deterministic bucketing, differenced per segment with a confidence interval. Gross recovered revenue is a vanity metric and is not reported. |
| How do you know it *learned*? | Ground truth is declared in a separate database the runtime cannot open. We report cumulative regret against an oracle and posterior recovery error. |
| Why not Kafka / Redis / an agent framework? | Append-only log with a monotonic sequence already gives ordering and replay. Nothing is hot. One fixed-schema LLM call needs no agent loop, and an agent loop would break reproducibility. |
| Why SQLite? | The system's core claim is exact replay. One file that can be copied, diffed, and re-run is worth more here than concurrency we don't need. |
| Where does margin come from? | Merchant-supplied config. Razorpay cannot see COGS. When it's absent the field is NULL, not zero, and margin ranking degrades explicitly. |
| Where does this stop working? | Below ~10k monthly transactions the cells starve and a rules engine would do better. Named, not hidden. |
| What did you sacrifice? | Breadth. No market signals, no multi-channel campaigns, no pricing engine. Depth in measurement and safety instead. |
