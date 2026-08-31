# Paydger

**Recurring payments fail. Retrying all of them is not a strategy.**

Paydger is a decision layer that sits above Razorpay. It looks ahead at scheduled recurring debits to catch failures that are structurally preventable, and when a payment does fail, it decides whether another attempt is actually worth spending — instead of retrying blindly on a fixed clock.

> Razorpay already moves the money. Paydger decides when trying again is worth it.

---

## Why this exists now

The RBI's **Digital Payments — E-mandate Framework, 2026** (notified 21 April 2026, effective immediately) consolidated eight prior circulars into one rulebook. Two provisions change the shape of the problem:

| Provision | Consequence for recovery |
|---|---|
| Recurring debits up to **₹15,000** clear without AFA; above that, AFA is required per transaction | Amount becomes a **decision variable**, not a passive field. A price change across the ceiling breaks an entire mandate cohort at once. |
| Carve-out: **₹1,00,000** ceiling for insurance premiums, mutual fund subscriptions, credit card bills | The ceiling is category-dependent, so risk must be evaluated per merchant category. |
| Issuers must send pre-debit notification ≥24h before every debit | Every scheduled debit has a **known 24-hour window** before money moves. That window is an intervention opportunity that reactive systems never use. |

> **Scope note.** The pre-debit notification obligation belongs to *issuers*, not merchants. Paydger does not send it and does not claim to. It uses the merchant's own knowledge of scheduled debit dates as an operational prevention window.

Razorpay Subscriptions currently retries a failed charge on **T+1, T+2, T+3**, then halts the subscription. That schedule is fixed, amount-blind, and identical whether the failure was insufficient funds, an expired credential, a cancelled mandate, or an amount that just crossed the AFA ceiling. Under a regime where attempts are constrained and amount-sensitive, "retry three times and give up" leaves value on the table in both directions — wasted attempts on dead payments, and no attempt at all on recoverable ones after day three.

---

## What Paydger is and is not

| Layer | Owner | Paydger's role |
|---|---|---|
| Payment execution, settlement, mandate rails | Razorpay | Consumer. Never reimplemented. |
| Issuer/PSP downtime detection | Razorpay Payment Downtime API | **Consumed as an input signal**, corroborated against local population data. |
| Fixed retry schedule (T+1/2/3) | Razorpay Subscriptions | Used as the **control arm baseline** in measurement. |
| Deciding *whether*, *when*, and *at what amount* to attempt | — | **This is Paydger.** |

Paydger is not a CRM, a checkout product, a coupon engine, a chatbot, or a payment gateway.

---

## Architecture

```
        ┌────────────────────────────┐      ┌────────────────────────────┐
        │  Razorpay (test mode)      │      │  Synthetic population      │
        │                            │      │  (causal generator)        │
        │  subscription.charged      │      │                            │
        │  subscription.pending      │      │  10k mandates, 100k events │
        │  subscription.halted       │      │  known latent ground truth │
        │  payment.failed            │      │                            │
        │  payment.downtime.*        │      │                            │
        └──────────────┬─────────────┘      └──────────────┬─────────────┘
                       │ HMAC-verified webhook             │
                       └──────────────┬────────────────────┘
                                      │  identical normalized schema
                                      ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  INGEST                                                      │
        │  verify signature → dedupe on source_event_id → normalize    │
        └──────────────────────────────┬───────────────────────────────┘
                                       ▼
        ╔══════════════════════════════════════════════════════════════╗
        ║  EVENT LOG — append-only, seq-ordered, SOURCE OF TRUTH       ║
        ╚══════════════════════════════┬═══════════════════════════════╝
                                       │  deterministic projection
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
  payment_ledger  customer_memory  attempt_ledger  population_   upcoming_
                                   (double-entry)   health        debits
        └──────────────┴───────────────┼───────────────┴──────────────┘
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  CONTEXT BUILDER — assembles one compact decision context    │
        │  per case (payment · customer · population · system health)  │
        └──────────────────────────────┬───────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │ FAILURE          │   │ UPLIFT MODEL     │   │ PREVENTION       │
    │ NORMALIZER       │   │                  │   │ DETECTOR         │
    │                  │   │ P(rec│attempt)   │   │                  │
    │ table → LLM      │   │  − P(rec│none)   │   │ AFA breach       │
    │ (cold path only) │   │ calibrated GBMs  │   │ expiry cliffs    │
    │ → closed enum    │   │ → uplift, EV     │   │ → risk cohort    │
    └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
             └──────────────────────┼──────────────────────┘
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  ENVELOPE ENGINE   (deterministic, versioned, pure function) │
        │                                                              │
        │  case → { permissible actions, earliest_at, refusal reasons }│
        │  Runs BEFORE ranking. Model never sees an illegal action.    │
        │  Empty set → NO_ACTION. Fails closed.                        │
        └──────────────────────────────┬───────────────────────────────┘
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  ALLOCATOR — rank permissible actions by uplift × value,     │
        │  take top-K under attempt budget (greedy knapsack)           │
        └──────────────────────────────┬───────────────────────────────┘
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  EXECUTION — reserve attempt → act → commit or release       │
        │  retry · schedule · merchant alert · suppress · escalate     │
        └──────────────────────────────┬───────────────────────────────┘
                                       ▼
                          outcome event → EVENT LOG
```

Every arrow above is a function call inside one process. There is no message broker and no service mesh. See [Deliberate non-choices](#deliberate-non-choices).

### Decision path for a single failed payment

```
payment.failed webhook
   │
   ├─ dedupe: source_event_id already in event_log? ──── yes ──▶ drop, 200 OK
   │
   ├─ normalize failure string
   │     exact match in failure_taxonomy? ── yes ──▶ class (no LLM call)
   │                                        no  ──▶ LLM → class → write back to table
   │
   ├─ entity collapse: find or create case by
   │     (customer_id, credential_fingerprint, failure_class, 6h bucket)
   │     → one case may cover N mandates
   │
   ├─ build context: customer memory · attempt history · issuer health · amount vs ceiling
   │
   ├─ ENVELOPE ENGINE → permissible action set
   │     empty ──▶ NO_ACTION, log refusal reasons, done
   │
   ├─ uplift model → P(recover│attempt) − P(recover│no attempt)
   │
   ├─ allocator: uplift × recoverable_value vs budget shadow price
   │     below threshold ──▶ SUPPRESS_RETRY (deliberate non-action, logged)
   │
   ├─ reserve attempt in attempt_ledger  (atomic, in the same tx as the decision)
   │
   ├─ execute
   │
   └─ outcome ──▶ commit or release reservation ──▶ append outcome event
```

---

## Core components

### 1. Event log and projections

The `event_log` table is the write-side source of truth. Every derived table (`payment_ledger`, `customer_memory`, `attempt_ledger`, `population_health`) is a **projection** — a pure function of the log up to a sequence number.

This is not full CQRS/event sourcing with a framework. It is one append-only table with a monotonic `seq`, plus projection functions that can rebuild every derived table from scratch. That buys deterministic replay for roughly 200 lines of code.

**Replay contract:** same event log + same `model_version` + `policy_version` + `taxonomy_version` + fixed seed ⇒ byte-identical decisions. This is a test, not a claim:

```bash
make replay-verify   # rebuilds all projections, re-decides every case, diffs against recorded decisions
```

### 2. Attempt ledger (double-entry)

A recovery attempt is a scarce resource, so it is accounted for like money. Every attempt produces a balanced posting group across two accounts.

```
posting_group_id  account              amount  case_id
─────────────────────────────────────────────────────────
pg_001            BUDGET_AVAILABLE       -1    case_88
pg_001            ATTEMPTS_RESERVED      +1    case_88

pg_002            ATTEMPTS_RESERVED      -1    case_88     (on success/failure)
pg_002            ATTEMPTS_CONSUMED      +1    case_88

pg_003            ATTEMPTS_RESERVED      -1    case_91     (on crash/timeout)
pg_003            BUDGET_AVAILABLE       +1    case_91
```

**Reserve → commit/release** exists so that a crash between the decision and the execution cannot silently consume budget or, worse, permit a duplicate attempt. Reservation and decision are written in the same database transaction.

**Invariants** (enforced as property tests over generated event streams, not assertions in prose):

```
I1  SUM(amount) GROUP BY posting_group_id = 0        for every group
I2  SUM(amount) WHERE merchant_id = M    = 0         at every seq
I3  BUDGET_AVAILABLE >= 0                            at every seq
I4  COUNT(postings) WHERE event_id = E   <= 2        idempotency holds under replay
```

`tests/test_ledger_invariants.py` runs these under Hypothesis against randomly generated interleavings including duplicate webhooks and mid-flight crashes.

### 3. Envelope engine

A deterministic, versioned pure function. **It runs before the model ranks anything**, so a model error can only produce a suboptimal legal action, never an illegal one.

```python
def envelope(ctx: CaseContext, policy: PolicyVersion) -> PermissibleSet:
    """Returns permitted actions with time bounds, plus a reason for each refusal."""
```

Rules are individually versioned and each carries its source:

| Rule | Condition | Effect |
|---|---|---|
| `R1_AFA_CEILING` | `amount > ceiling(category)` and mandate not AFA-provisioned | all `RETRY_*` excluded; only `CUSTOMER_ACTION`, `MERCHANT_ALERT` |
| `R2_ATTEMPT_CAP` | `attempts_used >= policy.max_attempts` | empty set |
| `R3_MIN_GAP` | `now < last_attempt_at + policy.min_gap` | `RETRY_NOW` excluded; `RETRY_LATER.earliest_at` set |
| `R4_DEAD_MANDATE` | `failure_class == MANDATE_INVALID` | all `RETRY_*` excluded |
| `R5_ISSUER_DOWN` | issuer health `DEGRADED` or `DOWN` | `RETRY_NOW` excluded; `WAIT_FOR_SYSTEM_RECOVERY` permitted |
| `R6_UNKNOWN_CAUSE` | `failure_class == UNKNOWN` | `ESCALATE_TO_HUMAN` only |
| `R7_FAIL_CLOSED` | permissible set empty | `NO_ACTION`, refusal reasons persisted |

`ceiling(category)` returns ₹15,000 by default and ₹1,00,000 for the insurance / mutual fund / credit card bill carve-out.

### 4. Failure normalizer (LLM, cold path)

Issuer and PSP decline reasons arrive as heterogeneous, vendor-specific, semi-natural-language strings. A lookup table handles the head of that distribution; the tail is unbounded and keeps growing.

```
raw string → normalize whitespace/case → hash
   │
   ├─ hit in failure_taxonomy  ──▶ class, confidence 1.0, source=table
   │
   └─ miss ──▶ LLM (structured output, closed enum) ──▶ class
                  │
                  └─ write back to failure_taxonomy with taxonomy_version
```

**The LLM is a cold-path component.** In steady state roughly 30 strings cover most volume, so the table absorbs almost every call. This keeps latency, cost, and determinism where they should be, and the table grows on its own.

Output space is a closed enum: `TEMPORARY_ISSUER_FAILURE`, `INSUFFICIENT_FUNDS`, `CUSTOMER_ACTION_REQUIRED`, `CREDENTIAL_EXPIRED`, `MANDATE_INVALID`, `AFA_REQUIRED`, `STRUCTURALLY_UNRECOVERABLE`, `UNKNOWN`.

**Evaluation:** 200 hand-labelled strings across providers. Report macro-F1 and `UNKNOWN` detection rate. Fallback on API failure or low confidence is `UNKNOWN`, which routes to `R6` and escalates rather than retrying.

### 5. Uplift model (classical ML, deliberately not an LLM)

The wrong question is *"will this payment recover?"* A loyal customer with a clean history scores high on that question and also recovers fine without help. Spending a scarce attempt on them buys almost nothing.

The right question is the difference between the two arms:

```
uplift(x) = P(recover | attempt, x) − P(recover | no attempt, x)
```

**T-learner.** Two gradient-boosted models trained on the two arms of the randomized holdout, each calibrated with isotonic regression on a held-out split. Because assignment is randomized, the naive difference of the two calibrated outputs is unbiased. Uplift trees would be more sample-efficient; two calibrated GBMs are simpler to build, easier to debug, and easier to defend in five days.

An LLM is not used here because the decision needs a **calibrated number**, and calibration is the property being measured.

**Evaluation:** ROC-AUC and PR-AUC per arm, Brier score, expected calibration error, and a Qini curve for the uplift ranking. The dashboard annotates every displayed probability with observed calibration from the holdout, e.g. *"80–90% bucket resolved at 84% over 1,127 cases."*

### 6. Population detector (deterministic, deliberately not an agent)

Two independent signals, then corroboration:

```
External:  payment.downtime.started / .updated / .resolved   (Razorpay)
Local:     EWMA baseline of success rate per (method, issuer)
           two-proportion z-test vs baseline + failure concentration share
```

| External | Local | Verdict | Action bias |
|---|---|---|---|
| down | degraded | `CONFIRMED_SYSTEMIC` | `WAIT_FOR_SYSTEM_RECOVERY`, preserve budget |
| quiet | degraded | `LOCAL_TO_MERCHANT` | investigate merchant-side cause (e.g. AFA breach) |
| down | normal | `TRUST_EXTERNAL` | conservative, cohort may not be exposed |
| quiet | normal | `HEALTHY` | normal allocation |

**We deliberately did not build an agent here.** Failure concentration by issuer and time window is a statistical question with a reliable deterministic answer, and Razorpay's own downtime signal already provides authoritative external corroboration. An LLM hypothesis loop would add latency and a hallucination surface for no accuracy gain. The `LOCAL_TO_MERCHANT` cell above is the interesting one, and it is reached by arithmetic.

### 7. Prevention engine

A scheduled sweep over debits due in the next 24 hours. For each upcoming charge:

| Risk class | Condition | Why it matters |
|---|---|---|
| `AFA_THRESHOLD_BREACH` | `amount > ceiling(category)` and mandate not AFA-provisioned | **The failure has not happened yet and retrying will never fix it.** Only a pricing or re-authorization decision will. |
| `CREDENTIAL_EXPIRY` | credential expires before `scheduled_debit_at` | recoverable only by customer action, before the debit |
| `MANDATE_EXPIRY_CLIFF` | mandate validity ends before `scheduled_debit_at` | requires re-registration with AFA |
| `ISSUER_RISK` | issuer currently `DEGRADED` | reschedule within the permitted window |
| `AMOUNT_CHANGE` | large delta vs historical, ceiling not crossed | weak signal, informational only |

`AFA_THRESHOLD_BREACH` and `AMOUNT_CHANGE` are **separate classes on purpose**. Percent change is close to meaningless here: ₹14,000 → ₹14,900 is a large jump that breaks nothing, while ₹14,950 → ₹15,050 is a 0.7% change that breaks the entire cohort. The threshold crossing is the signal.

Output is a risk cohort with exposed scheduled value, labelled explicitly as a prediction, not a guaranteed failure.

---

## Measurement

The measurement harness is built **before** the decision logic. Without it, every number the system produces is unfalsifiable.

**Assignment.** Deterministic so replay reproduces the split:

```python
arm = CONTROL if sha256(f"{case_id}{EXPERIMENT_SALT}").digest_int % 100 < HOLDOUT_PCT else TREATMENT
```

**Control arm** reproduces Razorpay's actual default behaviour: blind retry on T+1, T+2, T+3, then halt. This is the honest baseline and the real competitor.

**Treatment arm** uses the Paydger envelope + uplift allocation.

**Primary metric** — net incremental recovered value per 1,000 affected payments, with a bootstrap confidence interval. Not gross recovered GMV, because a large share of failures self-recover with no intervention at all and gross numbers silently take credit for them.

**Secondary** — recovery rate by arm, attempts consumed, attempts saved, recovered value per attempt, prevention precision and false-positive rate.

**The simulator holds a `would_have_self_recovered` flag that the system never observes**, so the holdout estimate can be checked against known truth. That validates the *method*. It does not validate real-world lift, and the README, dashboard, and pitch all say so.

---

## Failure handling

Default posture for any money-affecting action is **fail closed**.

| Failure | Detection | Behaviour |
|---|---|---|
| Duplicate webhook | unique index on `source_event_id` | drop, return 200 |
| Bad webhook signature | HMAC verify | reject 401, log, no ingest |
| LLM unavailable / timeout | API error, 3s budget | `UNKNOWN` class → `R6` → escalate |
| LLM low confidence | structured output confidence | `UNKNOWN` |
| Prompt injection in failure string | output space is a closed enum | cannot reach financial authority; envelope is deterministic |
| Uplift model unavailable | model load / inference error | conservative baseline policy, log degraded mode |
| Model outputs NaN or out of range | range assertion | treat as no-uplift, suppress |
| Envelope returns empty set | by construction | `NO_ACTION` with persisted refusal reasons |
| Ledger write fails | transaction rollback | **no execution**, case requeued |
| Crash between reserve and execute | reservation timeout sweep | release reservation, case re-decided |
| Razorpay API timeout | client timeout | reservation released, no attempt counted |
| Conflicting signals (external healthy, local degraded) | corroboration matrix | downgrade confidence, prefer `WAIT` |
| Clock skew on scheduled windows | server time only, UTC | envelope refuses actions with ambiguous timing |

---

## Security

- **No PAN, CVV, or full credential is stored.** Entity collapse keys on a `credential_fingerprint` derived from token id, or a salted hash of `last4 + network + expiry`.
- Razorpay webhooks are HMAC-verified before any parsing or ingest.
- The LLM receives **only** the raw failure string and the taxonomy. No customer identifiers, no amounts, no PII. Its output is a closed enum. Prompt injection therefore cannot cause a financial action.
- Every decision persists `decision_id`, `model_version`, `policy_version`, `taxonomy_version`, input snapshot, candidate actions, refusal reasons, selected action, and outcome.
- Secrets via environment only. `.env` is gitignored; `.env.example` documents required keys.

---

## Stack

| Layer | Choice | Reason |
|---|---|---|
| API | Python 3.11 + FastAPI | The uplift model requires scikit-learn, so the decision path must be Python. Splitting languages across the boundary costs a day and buys nothing. |
| Database | PostgreSQL 16 | Reserve→commit must be atomic in one transaction. Window functions carry the EWMA baselines. |
| Scheduler | APScheduler, in-process | One scheduled job (the 24h prevention sweep). A separate worker tier is not warranted. |
| ML | scikit-learn, `HistGradientBoostingClassifier`, isotonic calibration | Fast, calibrated, reproducible with a fixed seed. |
| LLM | Claude API, structured output | Closed-enum classification on the cold path only. |
| Frontend | Next.js + Tailwind + Recharts | Single dashboard; server components keep data fetching simple. |
| Tests | pytest + Hypothesis | Ledger invariants are property tests over generated event streams. |
| Local env | Docker Compose (`api`, `db`, `web`) | One command to a running system. |

### Deliberate non-choices

Documented because *not* adding infrastructure is an engineering decision.

- **No Kafka, no Redis Streams.** The system is single-writer over roughly 100k events with batch replay. The `event_log` table with a monotonic `seq` *is* the queue, and it gives ordering and replay for free. A broker would add operational surface and remove the replay guarantee.
- **No vector database.** All retrieval is by key over structured facts. Nothing here is a semantic search problem.
- **No microservices.** Six modules behind one process boundary. Splitting them would multiply failure modes without changing the decision logic.
- **No agent for root-cause diagnosis.** See [Population detector](#6-population-detector-deterministic-deliberately-not-an-agent).
- **No LLM in the money path.** The LLM classifies text. It never selects, schedules, or authorizes an attempt.

---

## Data model

```
event_log             seq, event_id, source_event_id, type, payload, occurred_at, ingested_at
payments              payment_id, subscription_id, customer_id, amount, method, issuer, status
payment_attempts      attempt_id, payment_id, attempt_no, attempted_at, result, raw_failure_reason
subscriptions         subscription_id, customer_id, plan_amount, category, mandate_type,
                      mandate_valid_until, afa_provisioned, next_debit_at
customers             customer_id, created_at, segment
customer_memory       customer_id, successes, failures, recoveries, typical_window,
                      credential_fingerprint, updated_at_seq
failure_taxonomy      hash, raw_pattern, failure_class, source, taxonomy_version, confidence
cases                 case_id, customer_id, credential_fingerprint, failure_class,
                      bucket_start, affected_payment_ids[], arm
attempt_ledger        entry_id, posting_group_id, account, amount, case_id, decision_id,
                      event_id, seq, created_at
decisions             decision_id, case_id, trigger_event_id, permissible_set, refusal_reasons,
                      uplift, expected_value, selected_action, model_version, policy_version,
                      taxonomy_version, created_at
outcomes              outcome_id, decision_id, result, recovered_amount, observed_at
population_health     method, issuer, window_start, success_rate, baseline, z_score,
                      external_signal, verdict
upcoming_debits       subscription_id, scheduled_debit_at, amount, risk_class, exposed_value
```

Unique constraints that matter: `event_log.source_event_id`, `attempt_ledger.(event_id, account)`, `decisions.(case_id, trigger_event_id)`.

---

## Repository layout

```
paydger/
├── api/
│   ├── ingest/            webhook verification, dedupe, normalization
│   ├── projections/       event_log → derived tables (pure, replayable)
│   ├── context/           case context assembly
│   ├── intelligence/
│   │   ├── normalizer/    taxonomy table + LLM cold path
│   │   ├── uplift/        T-learner, calibration, Qini
│   │   ├── population/    EWMA baseline, z-test, corroboration matrix
│   │   └── prevention/    24h sweep, risk classes
│   ├── policy/            envelope engine, versioned rules
│   ├── decision/          allocator, expected value, budget shadow price
│   ├── execution/         reserve → act → commit/release
│   └── replay/            deterministic replay + diff
├── bench/
│   ├── generator/         causal synthetic population
│   ├── scenarios/         threshold_break, issuer_outage, self_heal, entity_collapse
│   └── evaluation/        holdout scoreboard, calibration report
├── web/                   Next.js dashboard
├── tests/
│   ├── test_ledger_invariants.py    Hypothesis property tests
│   ├── test_envelope_rules.py       table-driven, one case per rule
│   └── test_replay_determinism.py
└── docker-compose.yml
```

---

## Quick start

```bash
git clone <repo> && cd paydger
cp .env.example .env          # add RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET,
                              # RAZORPAY_WEBHOOK_SECRET, ANTHROPIC_API_KEY
docker compose up -d          # api :8000, db :5432, web :3000
make migrate
make seed                     # generates the synthetic population
make bench                    # runs the holdout, prints the scoreboard
open http://localhost:3000
```

Live Razorpay test-mode events:

```bash
make tunnel                   # exposes /webhooks/razorpay
# register the URL in Razorpay Dashboard → Settings → Webhooks
```

---

## Status and limitations

**Working:** ingestion, event log and projections, attempt ledger with invariant tests, envelope engine, failure normalizer, uplift model, population detector, prevention sweep, allocator, holdout scoreboard, deterministic replay, dashboard.

**Known limitations, stated deliberately:**

1. **Benchmark results are simulated.** They demonstrate that the decision method does not fool itself. They are not evidence of production lift. Real validation needs a merchant cohort experiment over historical data.
2. **The Payment Downtime API is not enabled by default** on a Razorpay account and requires a support request. Where it is unavailable, the population detector falls back to local signals only and marks its verdict `LOCAL_ONLY`.
3. **Uplift models are trained on synthetic outcomes** and will not transfer to real merchants without retraining. The pipeline transfers; the weights do not.
4. **AFA provisioning state is inferred**, not read from a mandate field. Where the real mandate object exposes it, that inference should be replaced with the authoritative value.
5. **Single-tenant.** Merchant isolation exists in the schema but has no enforcement layer.
6. **Regulatory rules are encoded as of the 2026 framework.** They live in one versioned policy module precisely so they can be updated in one place, and they must be re-verified against current RBI and network rules before any production use.

---

## What I would revisit at scale

- The single-process design holds to roughly 10k decisions/minute. Past that, the ingest and decision paths separate first, with the event log as the boundary.
- Projections are currently rebuilt in full for replay. At real volume they need snapshots plus incremental catch-up from the last checkpoint.
- The greedy knapsack allocator is optimal for a single budget period. Across periods it needs a shadow price learned from budget exhaustion history rather than a fixed threshold.
- The T-learner will drift as merchant mix changes. Production wants out-of-time validation and automatic recalibration on a schedule.
