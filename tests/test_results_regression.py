"""Regression guard on the committed full-data results (skipped when artifacts are absent, e.g. in CI)."""
import json
from pathlib import Path

import pandas as pd
import pytest

ART = Path(__file__).resolve().parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "model_metrics.csv").exists(), reason="full-data artifacts not built")


def test_models_beat_baselines_on_test():
    m = pd.read_csv(ART / "model_metrics.csv").query("split == 'test' and calibration == 'none'").set_index("model")
    assert m.loc["Logistic regression", "pr_auc"] > m.loc["Recency rule", "pr_auc"]
    assert m.loc["Logistic regression", "roc_auc"] > m.loc["RFM heuristic", "roc_auc"]
    assert 0.70 < m.loc["Logistic regression", "roc_auc"] < 0.85


def test_clv_holdout_aggregate_within_tolerance():
    s = json.loads((ART / "clv_summary.json").read_text())
    assert abs(s["pred_purchases_total"] / s["actual_purchases_total"] - 1) < 0.10
    assert s["purchases_corr"] > 0.7
