"""Tests for scripts/analyze.py — deterministic, stdlib+pytest only."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SUMMARY_PATH = REPO_ROOT / "results" / "ablation_summary.json"


def _import_analyze():
    if "analyze" in sys.modules:
        return sys.modules["analyze"]
    spec = importlib.util.spec_from_file_location("analyze", SCRIPTS_DIR / "analyze.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


analyze = _import_analyze()


def test_raw_drift_confusion_kairos_detect_nomem_noplan() -> None:
    import json

    bundle = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    modes = analyze.get_modes(bundle)

    target = next(m for m in modes if m["mode"] == "kairos_detect_nomem_noplan")
    per_task = analyze.get_per_task(target)
    assert per_task, "expected per_task rows for kairos_detect_nomem_noplan"

    result = analyze.compute_confusion(per_task, "drift_detected_count")

    assert result["TP"] == 1, f"expected TP=1, got {result['TP']}"
    assert result["FP"] == 1, f"expected FP=1, got {result['FP']}"
    assert result["TN"] == 0, f"expected TN=0, got {result['TN']}"
    assert result["FN"] == 0, f"expected FN=0, got {result['FN']}"
    assert result["precision"] == pytest.approx(0.5, abs=1e-9)
    assert result["recall"] == pytest.approx(1.0, abs=1e-9)
