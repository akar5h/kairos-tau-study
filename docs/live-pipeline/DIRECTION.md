# Step-Level Deviation Detection — what this is, where we stand, what we can do

> **The category:** detect that an agent has gone off-track **during or right
> after the step it happens** — pinpoint the *first* action where it deviated
> from a successful trajectory, and the deviated steps that follow.
> **Observation only.** Identify the failure; do not correct it.

This is the thing this project is actually about. Everything else — the
tau-bench harness, the kairos-ai collector, the outcome rollups — is
scaffolding to get to *this* question.

---

## 1. The problem, stated precisely

An agent runs a multi-step task (call tools, read results, respond). Sometimes
it goes wrong. End-of-run pass/fail (an **Outcome Reward Model**, ORM) tells you
*that* it failed but not *where* — sparse, delayed, useless for debugging or
live intervention.

What we want is the opposite: a **per-step signal** that answers

> "At step *k*, did the agent just deviate from a path that still leads to
> success?"

and in particular **First-Error localization** — the earliest step *k\** where
it broke. Once you have *k\**, everything downstream is explainable; without it
you only have a verdict.

Crucially this is **observation, not correction.** We are not steering the agent
back. We are raising a flag: *"here, this step, something went wrong."* That
flag is the product.

---

## 2. Where the field stands (June 2026)

This is an active, **unsolved** subfield — "step-level failure attribution."
The headline is how *bad* the state of the art still is, which is the opening.

| Benchmark / method | Step-pinpoint accuracy | Note |
|---|---|---|
| **Who&When** (ICML'25, 127 MAS logs) | **14.2%** | even o1 / DeepSeek-R1 near-random at step localization |
| **AgentProcessBench** (1k traj, 8.5k steps) | 65.8% best (Gemini-3), **~45%** open-source | First-Error Accuracy |
| Human inter-annotator agreement | 89.1% (κ=0.77) | humans agree; models don't |

**The families of approach:**

- **Process Reward Models (PRMs)** — train a model to score each step
  (ToolPRM, STeCa). Strong but needs labeled step data.
- **LLM-judge localization** — feed the trajectory to a judge: *all-at-once*,
  *step-by-step*, or **binary-search**. Cheap; suffers positional bias and weak
  hypothesis revision over long chains.
- **Dependency / causal search** — FALAT (dependency-guided), causal-inference
  critical-step. Heavier, more principled.
- **Cheap training-free signals** —
  - **Thought↔action misalignment**: does stated reasoning match the action
    taken? ~1% of *successful* steps vs **3.5–4.8%** of *failing* ones. Per-step,
    no training.
  - **Reference-trajectory distance** (nDTW): cheap deviation metric vs a known
    good path.
  - ⚠️ **Raw action divergence is a trap** — divergence is *more* common in
    **successful** traces (agents legitimately explore). "Different from
    reference = wrong" **overfires**. Any naive drift detector hits this wall.

---

## 3. Where *we* currently stand

What we have working (proven live, see [`README.md`](README.md)):

- A **live collection pipeline**: tau-bench agent → OTLP → kairos-ai →
  Postgres → `TraceEnvelope` IR → analysis. No Phoenix. 20-task batch, reproducible.
- **Reference behavior** per workflow (kairos-ai already builds known-good
  trajectories per business operation).
- **Per-step drift observations** (the older kairos drift detector emits a
  per-step "this looks inconsistent" signal with a `would_break_task` flag).

What we **don't** have yet — and it's exactly the gap above:

- **No step localization.** Current live analysis is **outcome-only**: it asks
  "did the workflow's side-effect complete?" → it scored ~0.91 while tau-bench
  scored 0.45 on the same runs. It cannot say *which step* broke; it can't even
  see *that* a completed-but-wrong action is wrong (no ground truth).
- **Overfiring.** The old drift detector had recall but **precision ≈ 0.50** on
  tiny slices — its dominant false-positive source was flagging legitimate
  *exploratory / missing-action* steps. That is the exact "divergence trap" the
  literature names.

So we sit on the right raw material (reference paths + per-step signals) but with
the wrong *labeling scheme* and *no first-error metric*. The gap is framing and
scoring, **not infrastructure.**

---

## 4. The wedge — our thesis

A small, defensible, **observation-only** contribution, lifted from the strongest
cheap ideas in the literature and grounded in data we already collect.

Three moves, none of which require training a PRM:

1. **3-class step labels, not binary.** `+1 correct / 0 neutral / −1 wrong`.
   The **neutral** class absorbs legitimate trial-and-error and info-seeking
   steps so they stop being flagged. This is the direct fix for our
   precision-0.5 overfiring and the field's "divergence trap."
   *(from AgentProcessBench)*

2. **Error-propagation rule.** Once a step is `−1`, label every causally
   dependent downstream step `−1` until the agent corrects it or starts a new
   subtask. Consistent supervision over long trajectories; no spurious
   downstream blame. *(from AgentProcessBench)*

3. **Score with First-Error Accuracy (FirstErrAcc).** Did we pinpoint the
   *first* `−1` step? This metric **is** the goal restated. Report it against
   our reference good-paths.

Plus one cheap per-step detector we can compute from data already in the trace:

4. **Thought↔action consistency.** The trace has both the agent's stated
   reasoning and the tool call it made. Flag steps where they disagree — a
   training-free signal the literature shows separates 1% (success) from
   ~4% (failure).

The one-line claim we'd be testing:

> *Neutral-aware 3-class step labeling + error-propagation + a thought-action
> consistency signal reduces drift false-positives and improves first-error
> localization on tau-bench airline — observation only.*

---

## 5. What we can do (roadmap, observation-only)

Ordered by effort. All build on the live pipeline that already works.

- **A. First-error scorer — DONE.** `scripts/first_divergence.py`, run on the
  20-task corpus (no new spend). Diffs agent write actions vs tau gold writes,
  localizes the first divergence, then applies error-propagation (neutralize
  divergences later corrected). **Measured:**
  | signal | precision | recall |
  |---|---|---|
  | raw (positional first-write) | 0.55 | 1.00 |
  | propagated (drop self-corrected) | 0.60 | 0.82 |
  Error-propagation trades recall for a small precision gain and hits a **~0.6
  ceiling** — because tau scores final **DB state**, not action order, so
  exact-kwargs diff cannot see state-equivalence (e.g. an agent that splits a
  payment differently but reaches the same balance). **Takeaway: pure action-list
  diff tops out near 0.6; closing the gap needs DB-state replay or an
  LLM/judge correctness check.** This is the empirical motivation for moves
  B–C below.
- **B. Neutral-class drift filter.** Reclassify the old drift detector's
  per-step output into the 3-class scheme; measure the false-positive drop vs
  the precision-0.5 baseline.
- **C. Thought-action consistency detector.** Add the per-step reasoning↔action
  check; measure standalone precision/recall and as an ensemble with drift.
- **D. Reference-distance signal.** nDTW (or simpler step-set overlap) vs the
  per-workflow reference good-paths; combine, don't use alone.
- **E. Honest benchmark.** Report FirstErrAcc + step-precision/recall on a
  widened tau-bench corpus (50+ tasks, stronger agent model), against the
  field's 14–45% bar.

Out of scope by design: **correction / intervention.** We identify, we do not
steer. (The original study's lesson: intervention contaminated the signal;
keep detection clean and standalone.)

---

## 5b. LLM step-judge plan — beating the 0.6 ceiling

Action-diff caps at ~0.6 because it judges *exact match*, not *correctness*. An
LLM judge reads instruction + state + action and can flag **wrong-but-completed**
steps (the kairos outcome-layer blind spot) — observation only, no ground truth.

**Key reuse:** kairos-ai already ships an LLM step-judge — the drift detector
(`kairos.semantic_recovery`) emits a per-step verdict (`consistent`,
`would_break_task`, `drift_label`) from an LLM. So the first move is *wire +
benchmark*, not *build*.

### A — LLM step-judge (drift-detector-style) — DONE, NEGATIVE
`scripts/llm_step_judge.py`. Per agent WRITE action, an LLM is asked
"does this deviate from completing the task?" (`would_break_task`), reusing
kairos's `OpenRouterExpectationClient`. First `True` = predicted first-error;
same error-propagation as the baseline. (Not a faithful drift-detector replay —
that needs a live `SessionContext`/`PolicyPack`; this is the same idea, minimal.)

**Measured (20 tasks), raw / propagated precision · recall:**
| judge | precision | recall |
|---|---|---|
| gpt-4o-mini | 0.50 / 0.44 | 0.45 / 0.36 |
| **claude-sonnet-4.6** | 0.56 / 0.50 | **0.45** / 0.36 |
| action-diff baseline | 0.55 / 0.60 | 1.00 / 0.82 |

**Both LLM judges LOSE to the deterministic baseline, and a 5× stronger model
barely moved the needle (recall flat at 0.45).** So **model capability is NOT
the bottleneck — the framing is.** The naive per-write "does this deviate?"
prompt cannot flag *plausible-but-wrong* actions without knowing the correct
answer, and structurally can't catch failures where the agent never writes at
all (e.g. a task that needs a `send_certificate` the agent simply never issues).

This kills the "just use a stronger model" lever and sharpens B: the judge needs
**grounding** — the expected terminal actions / policy (exactly what kairos's
*real* drift detector feeds it via `SessionExpectation`, which this minimal
judge omits). Next move is grounding + 3-class, not a bigger model.

### B — Focused 3-class step judge (build, if A underperforms)
- Per step, prompt a judge (kimi-k2 / claude-sonnet via OpenRouter) with task
  instruction + trajectory-so-far + the step's action → `+1 / 0 / −1`
  (advance / neutral-exploratory / wrong), structured output.
- The **neutral class** is the precision lever — it stops exploratory-step
  over-firing.
- **Binary-search localization** (~log N judge calls/trace), not judge-every-step.

### C — Ensemble + honest benchmark
- Combine: action-diff (recall) + thought↔action consistency + LLM judge
  (precision). Report marginal contribution of each.
- FirstErrAcc on a widened corpus (50+ tasks) vs the field's 14–45% bar.

**Realistic target: beat 0.6, not solve it** — best LLM judges in the
literature reach ~65% FirstErrAcc. Calibrate against tau reward as a weak label
(failed tasks should surface ≥1 uncorrected `−1`; passed tasks ideally none).

---

## 6. Honest limitations

- tau-bench gives gold *actions*, so we can build `−1` labels semi-cheaply here;
  in the wild (no ground truth) the labels need a judge or reference paths,
  which reintroduces cost and noise.
- Small corpora: per-workflow outcome rests on 1–4 traces today. Numbers are
  directional until widened.
- Single domain (airline), single agent model (kimi-k2). Generalization unproven.
- Thought-action consistency needs the agent to *emit* reasoning; silent agents
  give nothing to check.
- First-error labeling has genuine ambiguity (humans agree only ~89%); our
  scorer inherits that ceiling.

---

## Sources

- [Who&When — Automated Failure Attribution of LLM Multi-Agent Systems (ICML 2025)](https://arxiv.org/abs/2505.00212)
- [AgentProcessBench — Step-Level Process Quality in Tool-Using Agents](https://arxiv.org/html/2603.14465)
- [AgenTracer — Who Induces Failure in LLM Agentic Systems](https://arxiv.org/pdf/2509.03312)
- [FALAT — Tracing Failures via Dependency-Guided Search](https://arxiv.org/html/2606.00765)
- [TrajAD — Trajectory Anomaly Detection for LLM Agents](https://arxiv.org/html/2602.06443)
- [STeCa — Step-level Trajectory Calibration](https://arxiv.org/pdf/2502.14276)
- [Understanding SWE Agents — Thought-Action-Result Trajectories](https://arxiv.org/html/2506.18824v1)
- [When Agents go Astray — Course-Correcting with PRMs](https://arxiv.org/pdf/2509.02360)
