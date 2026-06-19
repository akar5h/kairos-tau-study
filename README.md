# kairos-tau-study

A minimal, Apache-2.0 reproduction of an ablation experiment testing whether
Kairos drift detection and memory-injection interventions improve agent reward
on the tau-bench airline benchmark. The answer is: not meaningfully.

---

## TL;DR (null result)

Final 20-task ladder — Azure gpt-4.1 agent + gpt-4.1-mini user sim:

| Config | Score | vs baseline |
|---|---|---|
| baseline_no_kairos | 10/20 | floor |
| v4 cascade (advisory_v2) | 11/20 | +1 — ceiling, held since 2026-05-19 |
| v5 Mem0 content rewrite | 9/20 | −2 vs v4 |
| v6 full strip | ~5/20 | −6 vs v4 (run killed at task 8 for budget) |
| v7 surgical relabel (Fix A v3) | 11/20 | 0 vs v4, different per-task shape |

No memory or intervention configuration beat the baseline by more than the +1
task that plain cascade memory retrieval gave at v4, and that +1 is within
noise on 20 tasks. Per-task movements happened under every intervention but
never compounded to net improvement.

---

## What's in here

```
src/
  kairos/          vendored Kairos detection subset (drift detector, host SDK,
                   intercept engine, semantic recovery, diagnostic catalog)
  tau_harness/     benchmark harness (OpenAI agent loop, tau-bench wiring,
                   cascade retriever, Kairos session setup)
scripts/
  run_ablation.py  run a fresh ablation bundle (costs API $)
  analyze.py       regenerate metrics from a committed bundle (no API spend)
data/
  airline_success_workflows.json  memory pool used by cascade retriever
  anti_patterns.json              11-AP detector database
  labeled_alerts.jsonl            labeled drift observations from study runs
  pattern_catalog_v0.json         diagnostic pattern catalog
results/
  ablation_summary.json           committed sample bundle (tasks 19-21,
                                  baseline_no_kairos + kairos_detect_nomem_noplan)
```

---

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

This pulls tau-bench from a pinned git commit
(`sierra-research/tau-bench@59a200c`). You need git on the PATH.

Copy `.env.example` to `.env` and fill in your API keys (OpenRouter or Azure
endpoint + key). The harness reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` for
the agent and `OPENAI_USER_MODEL_API_KEY` / `OPENAI_USER_BASE_URL` for the
user sim. See `.env.example` for the full variable list.

---

## Reproduce the metrics (no API spend)

The committed bundle in `results/ablation_summary.json` covers tasks 19-21
with two modes. Run:

```bash
.venv/bin/python scripts/analyze.py results/ablation_summary.json
```

Expected output:

```
=== Reward ladder ===
baseline_no_kairos          0.50  (tasks 19-21, 2 tasks)
kairos_detect_nomem_noplan  0.50  (tasks 19-21, 2 tasks)

=== Detection confusion matrix (19-21 smoke, raw drift) ===
TP=1  FP=1  TN=0  FN=0  precision=0.50  recall=1.00

Note: FP is task 20 (passed); detector flagged update_reservation_baggages
as missing. Positive class = task failure.
```

The range 10-18 confusion matrices in `results/RESULTS.md` were derived from
full study runs and are not reproduced from `ablation_summary.json`; they are
documented there for reference.

---

## Run a fresh ablation (costs API $)

```bash
.venv/bin/python scripts/run_ablation.py \
  --env airline \
  --start-index 19 \
  --count 10 \
  --model gpt-4.1
```

This runs tasks 19-29 sequentially across the configured modes and writes a
JSON bundle to `results/`. Then analyze it:

```bash
.venv/bin/python scripts/analyze.py results/<bundle-name>.json
```

Runtime notes: sequential execution, roughly 30 requests/min, expect 1-2 hours
for a 10-task slice. Real API cost applies. Slices this small have high
variance — numbers are noisy and should not be treated as directional claims
without multiple trials.

---

## Caveats / honest limitations

- **Small slices.** The committed smoke bundle covers 2 tasks. The 20-task
  final ladder used one trial per config. At this scale, 1-2 task-level flips
  per run are expected from LLM temperature variance alone, independent of any
  intervention.
- **LLM user-sim noise.** The user simulator is gpt-4.1-mini. Reward varies
  across repeated runs on identical tasks; we observed reward swings of 0.33+
  across runs on the same 6-task slice.
- **Detector FP risk.** The raw drift detector has recall ~1.00 on the smoke
  slice but precision ~0.50. The dominant FP source is missing-action detection
  on tasks that pass (e.g., baggage update flagged as missing when the task
  succeeded without it).
- **Single benchmark domain.** All results are from the tau-bench airline
  domain with gpt-4.1 as the agent. Results may not generalise to other domains
  or models.

---

## License

Apache-2.0. See `LICENSE`.

The Kairos detection code under `src/kairos/` is vendored from a separate SDK
and relicensed Apache-2.0 for this study reproduction.
