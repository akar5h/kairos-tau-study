# Live Pipeline — kairos-ai as the direct collector for tau-bench

> **Read [`DIRECTION.md`](DIRECTION.md) first** — it frames the actual goal
> (step-level deviation detection: pinpoint the *first* step an agent goes
> wrong, observation-only), where the field stands (~14–45% accuracy, unsolved),
> where we stand, and the wedge. This page is the live experiment log that the
> direction builds on.

> A working, reproduced run of the **collect → IR → analyze** loop, where
> **kairos-ai** (not Phoenix) is the OpenTelemetry collector sitting directly
> behind a tau-bench airline agent. End-to-end: a live agent emits traces,
> kairos-ai stores them, reconstructs them into its `TraceEnvelope` IR, and
> runs its deterministic analysis pipeline over a 20-task batch.

---

## 1. What this is

Two halves, joined live:

- **`kairos-tau-study`** — the tau-bench airline harness (the agent under test).
  Runs an LLM agent + LLM user-simulator against tau-bench tasks, emits
  OpenTelemetry/OpenInference spans for every LLM call.
- **`kairos-ai`** — the tracing + analysis platform. Exposes a native
  **OTLP/HTTP receiver** (`/v1/traces`) that persists spans to Postgres, a
  **DB reader** that rebuilds spans into a `TraceEnvelope` IR, and a
  **deterministic engine** (`run_pipeline`) that maps each trace to a business
  workflow and computes per-workflow outcome rates.

The point being proved: **kairos-ai can replace Phoenix as the collector**, and
the captured traces are immediately analysis-ready — no Phoenix anywhere in the
path.

```
 tau-bench airline agent (kimi-k2 via OpenRouter)
        │  every LLM call → OpenInference span → OTLP/HTTP protobuf
        ▼
 kairos-ai OTLP receiver   http://localhost:4348/v1/traces   (api/otlp.py)
        ▼
 Postgres  spans table  (127.0.0.1:5434)
        ▼
 fetch_envelope_from_db()  →  TraceEnvelope (the IR)
        ▼
 run_pipeline(envelopes, BusinessContext)  →  AnalysisResult
```

This is the **enhanced** version of the earlier static reproduction: instead of
re-deriving metrics from a committed bundle, traces are generated *live*,
collected *directly* by kairos-ai, and analyzed *fresh*.

---

## 2. The experiment

| Parameter | Value |
|---|---|
| Benchmark | tau-bench **airline**, tasks 0–19 (`--first-n 20`) |
| Agent model | `moonshotai/kimi-k2` (OpenRouter) |
| User-sim model | `moonshotai/kimi-k2` (OpenRouter) |
| Concurrency | 2 |
| Collector | kairos-ai OTLP receiver on `:4348` → Postgres |
| Business context | `kairos-ai/eval/corpus/taubench/context.yaml` (4 airline workflows) |
| Analysis | `run_pipeline` — deterministic, no LLM |
| Cost | ~$1.5–2 (kimi-k2 @ $0.57/M in, $2.30/M out; ~89k prompt tok/task) |
| Wall time | ~25 min |

> **Note on network:** an earlier run during an internet drop produced 77
> connection errors and a mostly-empty corpus (3/20 usable traces, tau reward
> 1/20). Re-run on stable network: **0 network errors**. The numbers below are
> from the clean run.

---

## 3. Results

### tau-bench (ground-truth correctness)
**9 / 20 passed — average reward 0.45.**
tau-bench scores a task `1.0` only if the agent's actions exactly match the
gold action set (hashed). It is a strict correctness oracle.

### kairos-ai (side-effect completion)
Deterministic `run_pipeline` over the collected traces. Each trace maps to one
airline workflow; outcome = "did the workflow's required side-effect complete?"

| Workflow | Traces | Computable | Passed | Outcome rate | Findings |
|---|---:|---:|---:|---:|---:|
| Airline Flight Modification | 4 | 4 | 3 | **0.75** | 0 |
| Airline Reservation Booking | 4 | 4 | 4 | 1.00 | 0 |
| Airline Reservation Cancellation | 1 | 1 | 1 | 1.00 | 0 |
| Airline Human Escalation | 2 | 2 | 2 | 1.00 | 0 |
| **Aggregate (mapped)** | **11** | **11** | **10** | **≈0.91** | **0** |

(Raw JSON: [`raw_batch_analysis.json`](raw_batch_analysis.json). 13 traces in
the analysis window; 3 unmapped — too short/early-terminated to assign a
workflow.)

### The headline — the two oracles disagree
| Oracle | Pass rate |
|---|---|
| tau-bench (exact-match ground truth) | **0.45** |
| kairos-ai (side-effect completion) | **≈0.91** |

Same ~2× gap seen on a single trace, now reproduced across 20 runs. **kairos-ai
has no ground truth** — it observes that a `book_reservation` / `update_*`
side-effect fired and the workflow completed, so it counts the run as a pass.
tau-bench knows the *correct* booking and fails the agent when it books the
wrong thing. So kairos-ai **systematically under-flags** the failures tau
penalizes.

That is the live, batch-scale reproduction of the **detector-precision gap**
documented in the original study (`../../results/RESULTS.md`): drift/outcome
signals that complete-but-wrong actions slip past, because surface form looks
successful.

### Why `findings: 0`
The current kairos-ai engine = workflow membership + outcome rollup +
*deterministic* anti-pattern findings. The LLM semantic-clustering pass was
removed (`run_pipeline`'s `llm_client` is accepted but ignored). No
deterministic anti-pattern tripped on these traces, so no findings fired.
"Clustering" today means **outcome-rate-per-workflow**, not LLM
failure-cluster discovery.

---

## 4. Reproduction

Prereqs: Docker, `~/kairos-ai` and `~/kairos-tau-study` installed
(`pip install -e .` in each), `OPENROUTER_API_KEY` set in **both** `.env`
files, kairos-ai `.env` with `KAIROS_PG_DSN`.

### Step 1 — Postgres
```bash
cd ~/kairos-ai && set -a && . ./.env && set +a
docker compose -f deploy/docker-compose.yml up -d kairos-pg
docker exec deploy-kairos-pg-1 pg_isready -U kairos -d kairos      # accepting connections
```
If `No space left on device` / recovery mode → free Docker disk
(`docker image prune -a -f`) then `docker restart deploy-kairos-pg-1`.

### Step 2 — migrations (idempotent)
```bash
.venv/bin/python -c "from kairos.loop.db import apply_migrations; print('ok', len(apply_migrations()))"
```

### Step 3 — kairos-ai OTLP receiver on :4348
```bash
KAIROS_PORT=4348 nohup .venv/bin/python -m kairos.api > /tmp/kairos_api.log 2>&1 &
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:4348/v1/traces   # 200
```
(`:4318` is taken by the docker otel-collector → Jaeger; 4348 avoids the clash.)

### Step 4 — run the 20-task batch, OTLP → kairos-ai
```bash
cd ~/kairos-tau-study && set -a && . ./.env && set +a
export PHOENIX_OTLP_ENDPOINT=http://localhost:4348/v1/traces
export TAU_BENCH_PROVIDER=openrouter TAU_BENCH_MODEL=moonshotai/kimi-k2
.venv/bin/python -m tau_harness.run --env airline --first-n 20 \
  --user-model moonshotai/kimi-k2 --enable-kairos --max-concurrency 2
```
Prints `Average reward`. Traces flow to kairos-ai automatically. Transient
`Connection error` → just rerun (it means the network dropped).

### Step 5 — analyze the batch
```bash
cd ~/kairos-ai && set -a && . ./.env && set +a
.venv/bin/python - <<'PY'
import os, psycopg
from kairos.readers.db import fetch_envelope_from_db
from kairos.taxonomy.business_context import BusinessContext
from kairos.engine.pipeline import run_pipeline
dsn=os.environ["KAIROS_PG_DSN"]; c=psycopg.connect(dsn)
tids=[r[0] for r in c.execute("""select trace_id from spans where name ilike '%chatcompletion%'
  and start_time > now() - interval '40 min' group by trace_id order by min(start_time)""").fetchall()]
envs=[fetch_envelope_from_db(t, dsn=dsn) for t in tids]
ctx=BusinessContext.from_yaml("eval/corpus/taubench/context.yaml")
res=run_pipeline(envs, ctx)
for w in res.workflows:
    o=w.outcome
    print(f"{o.workflow_name:32s} traces={o.total_traces} passed={o.passed_count} rate={o.outcome_rate}")
PY
```

**Verify:** Step 4 prints a non-zero reward; Step 5 prints per-workflow outcome
rates with `traces > 0`. That is success.

---

## 5. Caveats

- **No ground truth in kairos-ai** → outcome rate over-counts success vs
  tau-bench. The gap is the finding, not a bug.
- **20 tasks is small** — outcome rates per workflow rest on 1–4 traces each.
- **Single domain / single agent model** (airline, kimi-k2).
- **Time-window trace selection** in Step 5 (`last 40 min`) is a smoke-grade
  query; a production version would tag traces by run-id.
- **`findings: 0`** — the deterministic anti-pattern layer didn't trip; the
  semantic-clustering pass is currently disabled.

---

## 6. Artifacts

- [`raw_batch_analysis.json`](raw_batch_analysis.json) — the raw `AnalysisResult`
  rollup (per-workflow traces / computable / passed / outcome_rate / findings).
- tau-bench per-task rewards: written by the harness to
  `kairos-tau-study/results/tool-calling-kimi-k2-*range_0-20*.json` (gitignored
  — contains full trajectories).
