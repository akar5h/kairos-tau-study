"""Layer 1 — diagnostic / offline analysis of kairos run artifacts.

This module owns the offline side of the architecture: read traces from
``data/runs/<run_id>/``, label drift alerts against task outcomes, cluster
by intent, extract baselines per cluster, mine failure-pattern rubrics. The
output feeds Layer 2 (runtime detection) as priors.

V0 ships two CLIs:

  * ``python -m kairos.diagnostic.labeler`` — interactive hand-labeler that
    walks a ``drift_observations.jsonl`` row by row, lets the user tag each
    alert with ``true_failure_signal`` / ``would_break_task`` / ``recoverable``
    / ``proposed_pattern_id``, and writes a JSONL of labels.

  * ``python -m kairos.diagnostic.catalog`` — groups a labeled JSONL by
    ``proposed_pattern_id`` and produces a ``pattern_catalog_v0.json`` with
    per-pattern positive/negative examples, severity default, recoverability
    default. This file becomes the RAG corpus the runtime judge consumes.

Neither CLI runs an LLM. Hand-grading produces the seed corpus; later
versions may auto-grade with a stronger judge.
"""
