"""Export compact, aggregated artifacts for the Streamlit app (app/data/, committed).

The deployed app reads ONLY these files: no raw transactions, no pipeline, no ML libraries.
Household-level content is limited to the households scored at the test cutoff (pseudonymous public-dataset ids)
with model features, scores, CLV and segment. Demographic attributes are NOT exported at household level, only the
aggregated breakout.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
ART, PROC, OUT = ROOT / "artifacts", ROOT / "data" / "processed", ROOT / "app" / "data"


def thin(df: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    if len(df) <= n:
        return df
    idx = np.unique(np.linspace(0, len(df) - 1, n).astype(int))
    return df.iloc[idx]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)  # never leave stale files from a previous schema
    OUT.mkdir(parents=True)
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    names = ["sql_summary", "train_summary", "clv_summary", "targeting_summary", "campaign_summary",
             "demographics_summary"]
    summary = {n: json.loads((ART / f"{n}.json").read_text()) for n in names}
    summary["config"] = {k: cfg[k] for k in ("churn", "clv", "targeting", "survival", "demographics")}
    summary["config"]["data"] = {k: cfg["data"][k] for k in ("max_quantity", "high_value_line_usd")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    # --- aggregated tables copied as-is ---------------------------------------------------
    for f in ["model_metrics.csv", "backtest_metrics.csv", "lift_tables.csv", "targeting_matrix.csv",
              "targeting_sensitivity.csv", "shap_importance.csv", "shap_importance_logreg.csv",
              "lgbm_grid_valid.csv", "test_metric_ci.csv", "campaign_outcomes_targeted.csv",
              "campaign_outcomes_type_a.csv", "campaign_outcomes_prior_redeemer.csv",
              "campaign_balance_targeted.csv", "campaign_coupon_gradient.csv", "campaign_type_summary.csv",
              "demographics_breakout.csv", "demographics_coverage.csv"]:
        shutil.copy(ART / f, OUT / f)
    for f in ["km_curves.csv", "logrank_tests.csv", "cox_hazard_ratios.csv", "cox_zph.csv",
              "cox_stratified_hazard_ratios.csv", "cox_stratified_zph.csv", "km_overall.csv"]:
        shutil.copy(ART / "r_survival" / f, OUT / f"surv_{f}")
    for f in ["dq_report.csv", "dq_cleaning_waterfall.csv", "cohort_retention.csv", "dim_campaigns.csv"]:
        shutil.copy(PROC / f, OUT / f)
    pd.read_parquet(PROC / "monthly_active_households.parquet").to_csv(OUT / "monthly_active_households.csv",
                                                                       index=False)
    b = pd.read_parquet(PROC / "fct_baskets.parquet")
    b["month"] = pd.to_datetime(b["basket_date"]).dt.to_period("M").dt.to_timestamp()
    b.groupby("month").agg(spend=("spend", "sum"), baskets=("basket_id", "count"),
                           households=("household_id", "nunique"), coupon_baskets=("has_coupon", "sum"),
                           fresh=("fresh_spend", "sum"), grocery=("grocery_spend", "sum"), fuel=("fuel_spend", "sum"),
                           health_beauty=("health_beauty_spend", "sum")).round(2).reset_index() \
        .to_csv(OUT / "monthly_sales.csv", index=False)
    rfm = pd.read_parquet(PROC / "household_rfm.parquet")
    rfm.groupby("rfm_segment").agg(households=("household_id", "size"), avg_recency=("recency_days", "mean"),
                                   avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean"),
                                   total_monetary=("monetary", "sum")).reset_index().round(2) \
        .to_csv(OUT / "rfm_segments.csv", index=False)

    # --- curves (precomputed so the app needs no sklearn) -----------------------------------
    probs = pd.read_parquet(ART / "test_probs.parquet")
    y = probs["churned"].to_numpy()
    rows_roc, rows_pr, rows_rel = [], [], []
    for m in [c for c in probs.columns if c not in ("household_id", "churned")]:
        p = probs[m].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        rows_roc.append(thin(pd.DataFrame({"model": m, "fpr": fpr, "tpr": tpr})))
        pr, rc, _ = precision_recall_curve(y, p)
        rows_pr.append(thin(pd.DataFrame({"model": m, "recall": rc, "precision": pr})))
        frac, mean = calibration_curve(y, p, n_bins=8, strategy="quantile")
        rows_rel.append(pd.DataFrame({"model": m, "mean_pred": mean, "observed": frac}))
    pd.concat(rows_roc).round(5).to_csv(OUT / "curve_roc.csv", index=False)
    pd.concat(rows_pr).round(5).to_csv(OUT / "curve_pr.csv", index=False)
    pd.concat(rows_rel).round(5).to_csv(OUT / "curve_reliability.csv", index=False)

    # --- CLV holdout aggregates ----------------------------------------------------------
    ch = pd.read_parquet(ART / "clv_holdout.parquet")
    ch.assign(fbin=(np.minimum(ch["frequency_cal"], 60) // 5) * 5).groupby("fbin").agg(
        households=("household_id", "size"), actual=("frequency_holdout", "mean"),
        predicted=("pred_purchases", "mean")).reset_index().to_csv(OUT / "clv_holdout_by_freq.csv", index=False)
    ch.assign(decile=pd.qcut(ch["pred_spend"].rank(method="first"), 10, labels=range(1, 11)).astype(int)) \
        .groupby("decile").agg(households=("household_id", "size"), actual_spend=("actual_spend", "mean"),
                               predicted_spend=("pred_spend", "mean")).reset_index() \
        .to_csv(OUT / "clv_holdout_by_decile.csv", index=False)

    # --- scored households (test cutoff) + SHAP -------------------------------------------
    from churn_clv.features import FEATURE_COLUMNS

    cutoff = cfg["churn"]["test_cutoff"]
    sc = pd.read_parquet(ART / "test_scored.parquet")
    sc["household_id"] = sc["household_id"].astype(str)
    clv = pd.read_parquet(ART / "clv_scores_test_cutoff.parquet")[
        ["household_id", "p_alive", "pred_purchases_90d", "expected_aov", "clv_12m"]]
    clv["household_id"] = clv["household_id"].astype(str)
    seg = pd.read_csv(ART / f"household_scores_{cutoff}.csv", dtype={"household_id": str})[
        ["household_id", "risk_segment", "value_segment"]]
    keep = ["household_id"] + FEATURE_COLUMNS + ["recency_days", "frequency", "monetary_total", "churn_prob",
                                                 "churn_prob_raw", "churned", "future_trips", "future_spend"]
    keep = list(dict.fromkeys(keep))
    hh = sc[keep].merge(clv, on="household_id").merge(seg, on="household_id")
    for c in hh.select_dtypes("float").columns:
        hh[c] = hh[c].astype("float32")
    hh.to_parquet(OUT / "households.parquet", index=False)
    for m in ["lgbm", "logreg"]:
        sv = pd.read_parquet(ART / f"shap_values_{m}.parquet")
        sv["household_id"] = sv["household_id"].astype(str)
        sv[FEATURE_COLUMNS] = sv[FEATURE_COLUMNS].astype("float32")
        sv.to_parquet(OUT / f"shap_values_{m}.parquet", index=False)
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"app/data: {len(list(OUT.iterdir()))} files, {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
