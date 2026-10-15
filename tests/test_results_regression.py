"""Regression guard on the committed full-data results (skipped when artifacts are absent)."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "model_metrics.csv").exists(), reason="full-data artifacts not built")


def js(name):
    return json.loads((ART / name).read_text())


def test_models_beat_baselines_on_test():
    m = pd.read_csv(ART / "model_metrics.csv").query("split == 'test' and calibration == 'none'").set_index("model")
    assert m.loc["Logistic regression", "roc_auc"] > m.loc["RFM heuristic", "roc_auc"]
    assert m.loc["Logistic regression", "roc_auc"] > m.loc["Recency rule", "roc_auc"]
    assert m.loc["LightGBM", "roc_auc"] > m.loc["RFM heuristic", "roc_auc"]
    assert 0.75 < m.loc["LightGBM", "roc_auc"] < 0.92


def test_final_model_selected_on_validation_and_not_isotonic():
    t = js("train_summary.json")
    assert not t["final_model"].endswith("isotonic")
    for name, c in t["calibration"].items():
        assert c["isotonic_eligible"] is False and c["valid_positives"] < 1000, name
    m = pd.read_csv(ART / "model_metrics.csv").query("split == 'valid' and calibration == 'none'")
    ml = m[m.model.isin(["Logistic regression", "LightGBM"])].sort_values("pr_auc", ascending=False)
    assert t["final_model"].startswith(ml.iloc[0]["model"])


def test_label_prevalence_is_stable_and_low():
    t = js("train_summary.json")
    for k in ("train", "valid", "test"):
        assert 0.02 < t["churn_rate"][k] < 0.08
    assert t["n_test_positives"] >= 50


def test_backtest_has_closed_training_windows():
    bt = pd.read_csv(ART / "backtest_metrics.csv")
    assert (bt["n_train_snapshots"] >= 3).all()
    assert pd.to_datetime(bt["train_last_cutoff"]).add(pd.Timedelta(days=90)).le(pd.to_datetime(bt["eval_cutoff"])).all()


def test_clv_holdout_aggregate_within_tolerance():
    s = js("clv_summary.json")
    assert abs(s["pred_purchases_total"] / s["actual_purchases_total"] - 1) < 0.05
    assert abs(s["pred_spend_total"] / s["actual_spend_total"] - 1) < 0.05
    assert s["purchases_corr"] > 0.7
    assert s["scoring_params"]["bgf"]["r"] > 0


def test_cleaning_counts_reconcile():
    sq = js("sql_summary.json")
    wf = sq["waterfall"]
    dropped = sum(v for k, v in wf.items() if k.startswith("dropped"))
    assert wf["raw lines"] - dropped == wf["kept lines (merchandise, free items, coupon-only, returns)"]
    assert sq["n_households"] == 2469 and sq["dq_report"]["total_lines"] == 1_469_307
    assert sq["n_households_with_demographics"] == 801


def test_targeting_outputs_are_labelled_assumed():
    t = js("targeting_summary.json")
    assert "base_case_ASSUMED" in t
    assert "expected_net_value_target_all_usd" in t["base_case_ASSUMED"]


def test_campaign_analysis_is_flagged_observational():
    c = js("campaign_summary.json")
    assert "Observational" in c["caveat"]
    assert c["max_abs_smd_targeted"] > 0.5   # targeted and untargeted households were very different at baseline


def test_demographics_flagged_partial():
    d = js("demographics_summary.json")
    assert d["n_with_demographics"] == 801 and 0.3 < d["coverage_share"] < 0.35
    assert "Partial coverage" in d["caveat"]


def test_survival_outputs_present():
    r = ART / "r_survival"
    for f in ("km_overall.csv", "logrank_tests.csv", "cox_hazard_ratios.csv", "cox_zph.csv",
              "cox_stratified_hazard_ratios.csv"):
        assert (r / f).exists()
    hr = pd.read_csv(r / "cox_hazard_ratios.csv").set_index("term")
    assert hr.loc["log_trips", "hr"] < 1   # more baseline trips, slower lapse
    assert (ROOT / "reports" / "survival_report.html").exists()
