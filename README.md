# eval-tinder

Human-aligned evaluation of production traces. An expert labels cases
PASS / FAIL / CANNOT_JUDGE, DSPy GEPA evolves a grader prompt to match those
labels, a small committee of retained candidate graders picks the next cases
worth a human's time, and an independent locked audit decides whether a frozen
grader may automate anything.

No task-specific rubric is required. Predictions are never treated as human
labels. TRAIN, DEV, and AUDIT_RESERVE material are kept apart end to end.

## Stack

| Part | Technology |
|---|---|
| API | Python 3.11, FastAPI, SQLAlchemy 2, Alembic, PostgreSQL 16 |
| Worker | separate Python process leasing jobs from a PostgreSQL job table |
| Optimizer | `dspy==3.3.1` with the `gepa==0.1.4` engine (pinned in `backend/uv.lock`) |
| Frontend | Vite, React 18, TypeScript |
| Models | any LiteLLM provider through `dspy.LM`; a deterministic scripted fake for tests and the offline demo |

## Quick start

### Option A: Docker Compose

```bash
cp .env.example .env            # fill in models/credentials, or keep LLM_PROVIDER=fake
docker compose up --build       # db, api (:8000), worker, frontend (:5173)
```

### Option B: local processes

```bash
# PostgreSQL 16 must be running; create the databases once:
psql -U postgres -c 'CREATE DATABASE eval_tinder' -c 'CREATE DATABASE eval_tinder_test'

cd backend
cp ../.env.example .env         # edit as needed
uv sync --extra dev             # installs the pinned lockfile
uv run alembic upgrade head     # migrations
uv run eval-tinder serve        # API on http://localhost:8000 (docs at /docs)
uv run eval-tinder worker       # in a second terminal

cd ../frontend
pnpm install && pnpm dev        # http://localhost:5173, proxies /api to :8000
```

### Configuration

Everything comes from the environment or `backend/.env` (see `.env.example`):

- `LLM_PROVIDER=fake` runs the scripted grading/reflection models. The UI shows a
  SIMULATED badge; nothing is sent to a provider.
- `LLM_PROVIDER=litellm` with `GRADER_MODEL` and `REFLECTION_MODEL`
  (for example `openai/gpt-5.6-luna`) plus the provider's own credential
  variable (`OPENAI_API_KEY`, ...). Model ids are configuration; none is hardcoded.
- Budgets: `MAX_PROVIDER_CALLS_PER_JOB`, `MAX_TOKENS_PER_CALL`,
  `MAX_TOTAL_TOKENS_PER_JOB` cap every job in addition to GEPA's metric-call budget.
  `PRICING_TABLE` (JSON) enables cost estimates; without it no cost is claimed.
- `API_TOKEN` is required for anything but a loopback single-expert deployment.

## The demo

```bash
cd backend
uv run eval-tinder demo seed --simulate-expert   # synthetic cancellation fixture + simulated labels
```

The fixture (`backend/fixtures/`) is SYNTHETIC. Its expert policy is *truthful
status reporting*, not task completion; the initial grader prompt contains no
such rule. With the fake provider the scripted grader behaves like a
"completion" grader until an evolved instruction mentions truthful reporting, so
the whole loop can be exercised offline. Simulated judgments are stored with
reviewer `simulated-expert` and an explanation prefix `[SIMULATED]`.

With a real model, run the same flow through the UI: import, seed review, DEV
review, optimization run, candidate comparison, selection round, next batch,
second run, audit, export. Measured real-model outcomes are in
`docs/REAL_MODEL_RESULTS.md`.

## Tests

```bash
cd backend
uv run pytest -q                       # unit, integration (PostgreSQL), API, GEPA contract with fakes
uv run ruff check .
cd ../frontend && pnpm test && pnpm lint && pnpm build
```

Integration tests create a private PostgreSQL database per test process from
the Alembic migrations (`TEST_DATABASE_URL`, default
`postgresql+psycopg://postgres:postgres@127.0.0.1:5432/eval_tinder_test`).

Opt-in tests against a real provider (they spend tokens and report the measured
outcome; they never require GEPA to beat the seed):

```bash
LLM_PROVIDER=litellm GRADER_MODEL=openai/... REFLECTION_MODEL=openai/... \
REAL_MODEL=1 REAL_GEPA=1 uv run pytest backend/tests/optimizer/test_real_model.py -s
```

The always-on contract test `tests/optimizer/test_gepa_contract.py` runs the
real `dspy.GEPA` optimizer with deterministic scripted models. It verifies the
integration against the pinned release (metric convention, explicit nonempty
DEV, candidate extraction, instance-id mapping, prompt-only save/reload, separate
grading and reflection contexts, budget abort) and fails clearly on dependency
incompatibility. It is not evidence that optimization learns.

## How the pieces fit

```
backend/eval_tinder/
  domain/       pure logic: partitions, rendering, manifests, metrics, disagreement, committee, selection
  grader/       the single dspy.Predict predictor, evidence validation, deterministic grade() boundary
  gepa/         dspy.Example construction, 5-argument agreement metric, GepaOptimizerService, result extraction
  llm/          LM factory (one real provider adapter), scripted fakes, budget guard, recording wrapper
  services/     DB-backed application services: imports, review, snapshots, optimization, grading,
                selection, audits, automation, bulk grading, exports, jobs
  worker/       leased job runner and handler registry
  api/          FastAPI app, schemas, routers (one module per feature area)
  experiments/  random-vs-committee selection experiment under a fixed labeling budget
  cli.py        migrate / serve / worker / grade (portable grader bundle) / export-grader / demo / experiment
backend/alembic/   migrations
backend/fixtures/  synthetic demo data with its truth table
frontend/          React UI: dashboard, blind review, optimization + diffs, traces, audits, exports
```

Key rules the code enforces (see the plan's section 3):

- Judgments are append-only; corrections supersede. One active judgment per trace and policy epoch.
- Groups (revisions, alternates, exact duplicates) share a seeded partition that is never rearranged.
- Optimization runs freeze TRAIN and DEV snapshots; labels collected mid-run belong to the next run.
  Empty DEV fails preflight. Reflection feedback comes from TRAIN only.
- Every returned GEPA candidate is an immutable grader version with lineage and a JSON manifest.
  DEV comparisons are recomputed through the same validated grading path used by probes and audits.
- Invalid output, invalid evidence, provider errors, or budget exhaustion never become PASS.
- The committee only chooses questions. Random slots are drawn before any model output is consulted.
- Audits lock the pipeline hash, sample, and risk targets before any audit label is seen; zero
  denominators are `NOT_ESTIMABLE`; automation is off by default and enabled only for the exact
  frozen pipeline when the predeclared gate passes; corrections and pipeline changes invalidate it.
- Exports contain JSONL, JSON manifests, and reports only: no secrets, no sealed audit material, no pickles.

## Verification and known limitations

Every milestone's tests were re-run by independent adversarial reviewers (mutation
checks, throwaway probes) whose confirmed findings were fixed and covered by
regression tests in `backend/tests/integration/test_verifier_fixes.py`. Points
worth knowing before relying on the system:

- **Random review slots and context repair.** Random slots are drawn before any
  disagreement score is consulted, but cases whose committee votes were all
  REVIEW are routed to context repair first, so the random draw is conditional
  on that routing. Random exploration never replaces the independent audit.
- **Contrary audits.** A released audit of the same pipeline and policy epoch
  whose gate failed blocks enablement unless its risk targets were strictly
  tighter than the enabling audit's. Re-sampling the reserve until a pass is
  refused by design.
- **Simulated expert.** The demo's `--simulate-expert` labels and the experiment
  runner's expert come from the fixture truth table. They are marked as simulated
  and are not expert evidence; the experiment reports ties and regressions as
  measured.
- **Synthetic data never enters an audit.** The demo fixture is SYNTHETIC, so the
  audit screen needs PRODUCTION-sourced imports before a sample can be locked.
- **Metric-call budgets are approximate.** GEPA may overshoot `max_metric_calls`
  by one in-flight minibatch; the application budget guard is the hard cap and
  ends the run in an explicit BUDGET_EXHAUSTED state.

## Portable grader

```bash
uv run eval-tinder export-grader --grader-id <id> --output grader.json
uv run eval-tinder grade --bundle grader.json --input new_cases.jsonl --output predictions.jsonl \
    --provider litellm --model openai/...
```

Predictions are written one per line, tagged `MACHINE`, with the grader's manifest
hash, status, verdict, validated evidence, and the bundle's audit/automation status.
