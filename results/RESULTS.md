# Results

## How to read this

- **Positive class** = task failure (reward 0.0).
- **Detector-positive** = drift fired (raw, filtered, or matched+task-breaking).
- TP: failure detected. FP: pass flagged as failure. TN: pass not flagged. FN: failure missed.
- All confusion matrices treat each task as one sample.
- Slices are small; numbers are illustrative, not statistically stable.

---

## 20-task reward ladder (final ablation)

Azure gpt-4.1 agent + gpt-4.1-mini user sim. One trial per config.

| Config | Score | Verdict |
|---|---|---|
| baseline_no_kairos | 10/20 | Floor. No Kairos at all. |
| v4 cascade (advisory_v2) | 11/20 | Ceiling. Plain cascade memory retrieval + advisory renderer. Never improved upon across 3 subsequent intervention phases. |
| v5 Mem0 content rewrite | 9/20 | −2 vs v4. Rewriting memory content from advisory to imperative hurt net. |
| v6 full strip | ~5/20 | −6 vs v4. Removing the directive blocks killed distributed-help tasks faster than it fixed concentrated-harm tasks. Run stopped early for budget. |
| v7 surgical relabel (Fix A v3) | 11/20 | Tied v4. 4 per-task movements (2 lifts + 2 regressions) — net zero. Cleanest mechanism story of the study; doesn't survive variance at N=20. |

Source: originally docs/phase7-active-harness-results.md in the experiment repo.

---

## Detection confusion matrices

### 19-21 clean smoke (reproducible from committed bundle)

Ablation bundle: `results/ablation_summary.json`
Modes: `baseline_no_kairos` and `kairos_detect_nomem_noplan` (no memory, no plan injection, no intervention).
Tasks: 19 (fail) and 20 (pass). 2 tasks total.

Reproduce with: `scripts/analyze.py results/ablation_summary.json`

| Detector view | TP | FP | TN | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Raw drift | 1 | 1 | 0 | 0 | 0.50 | 1.00 |
| Filtered task-breaking | 1 | 1 | 0 | 0 | 0.50 | 1.00 |
| Matched + task-breaking | 1 | 1 | 0 | 0 | 0.50 | 1.00 |

FP detail: task 20 passed (reward 1.0) but the detector flagged
`update_reservation_baggages` as a missing expected action. The obligation
ledger did not suppress it because the baggage expectation was not closed as
satisfied or accepted-handoff before session end.

### Range 10-18 (from full study runs)

Three checkpoints from progressive detector calibration. Memory was on in all
runs; plan injection and intervention state not separately recorded for
checkpoints 1 and 2. Source: originally
docs/kairos-tau-detection-evolution-report.md in the experiment repo.

| Checkpoint | Avg reward | Raw matrix | Filtered matrix | Note |
|---|---|---|---|---|
| 0512132218 | 0.375 | TP=5 FP=3 TN=0 FN=0 | TP=0 FP=0 TN=3 FN=5 | Early calibration layer was too suppressed — filtered recall collapsed to zero. |
| 0512155428 | 0.25 | TP=6 FP=2 TN=0 FN=0 | TP=6 FP=2 TN=0 FN=0 | High recall, FP still high; calibration not yet effective on this slice. |
| 0512201336 | 0.25 | TP=6 FP=2 TN=0 FN=0 | TP=6 FP=0 TN=2 FN=0 | Hard-constraint precision hardening worked: filtered FP dropped to 0. |

Interpretation: the raw detector had consistent recall across all three
checkpoints. Hard-constraint calibration (cabin mismatch, route/date mismatch,
budget, payment) moved precision from ~0.63 to 1.00 on this slice. The
filtered-then-zero recall at checkpoint 1 shows that early suppression logic
was too aggressive before the calibration rules were in place.

---

## Conclusion

v4 cascade (11/20) is the ceiling. Every subsequent intervention — memory
content rewrite (v5), full strip (v6), surgical relabel (v7) — produced
per-task movements that summed to net zero or net negative at 20 tasks.
Detection-only mode (no memory, no plan injection, no intervention gates)
does not perturb reward, which confirms Kairos is usable as a passive
measurement and diagnostic layer. It is not (yet) an intervention that
improves agent reward: gpt-4.1 + cascade failures are confident-wrong actions
in varying surface forms that substring, regex, and hash detectors miss, and
the dominant source of detector false positives (missing-action on tasks that
pass) remains an open engineering problem in the obligation ledger.
