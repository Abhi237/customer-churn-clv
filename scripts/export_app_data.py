"""Export compact, aggregated artifacts for the Streamlit app (app/data/, committed).

The deployed app reads ONLY these files: no raw transactions, no pipeline, no ML libraries.
Customer-level content is limited to the 4,323 customers scored at the test cutoff (pseudonymous
public-dataset IDs) with model features, scores, CLV and segment.
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


def thin(df: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    if len(df) <= n:
        return df
    idx = np.unique(np.linspace(0, len(df) - 1, n).astype(int))
    return df.iloc[idx]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    # --- summaries (json) -------------------------------------------------------------
    summary = {n: json.loads((ART / f"{n}.json").read_text())
               for n in ["sql_summary", "train_summary", "clv_summary", "targeting_summary"]}
    summary["config"] = {"churn": cfg["churn"], "clv": cfg["clv"], "targeting": cfg["targeting"],
                         "non_product_codes": cfg["data"]["non_product_codes"],
                         "max_abs_quantity": cfg["data"]["max_abs_quantity"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    # --- aggregated tables copied as-is ------------------------------------------------
    for f in ["model_metrics.csv", "backtest_metrics.csv", "lift_tables.csv", "targeting_matrix.csv",
              "targeting_sensitivity.csv", "shap_importance.csv", "shap_importance_logreg.csv",
              "lgbm_grid_valid.csv"]:
        shutil.copy(ART / f, OUT / f)
    for f in ["km_curves.csv", "logrank_tests.csv", "cox_hazard_ratios.csv", "cox_zph.csv",
              "cox_stratified_hazard_ratios.csv", "cox_stratified_zph.csv", "km_overall.csv"]:
        shutil.copy(ART / "r_survival" / f, OUT / f"surv_{f}")
    for f in ["dq_report.csv", "dq_cleaning_waterfall.csv", "cohort_retention.csv"]:
        shutil.copy(PROC / f, OUT / f)
    pd.read_parquet(PROC / "monthly_active_customers.parquet").to_csv(OUT / "monthly_active_customers.csv",
                                                                       index=False)
    inv = pd.read_parquet(PROC / "fct_invoices.parquet")
    inv["month"] = pd.to_datetime(inv["invoice_date"]).dt.to_period("M").dt.to_timestamp()
    inv.groupby("month").agg(revenue=("revenue", "sum"), invoices=("invoice", "count"),
                             customers=("customer_id", "nunique")).reset_index() \
        .to_csv(OUT / "monthly_revenue.csv", index=False)
    rfm = pd.read_parquet(PROC / "customer_rfm.parquet")
    rfm.groupby("rfm_segment").agg(customers=("customer_id", "size"), avg_recency=("recency_days", "mean"),
                                   avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean"),
                                   total_monetary=("monetary", "sum")).reset_index() \
        .to_csv(OUT / "rfm_segments.csv", index=False)
    dim = pd.read_parquet(PROC / "dim_customers.parquet")
    dim.groupby("home_country").agg(customers=("customer_id", "size"), revenue=("lifetime_revenue", "sum")) \
        .reset_index().sort_values("customers", ascending=False).to_csv(OUT / "countries.csv", index=False)

    # --- curves (precomputed so the app needs no sklearn) -------------------------------
    probs = pd.read_parquet(ART / "test_probs.parquet")
    y = probs["churned"].to_numpy()
    rows_roc, rows_pr, rows_rel = [], [], []
    for m in [c for c in probs.columns if c not in ("customer_id", "churned")]:
        p = probs[m].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        rows_roc.append(thin(pd.DataFrame({"model": m, "fpr": fpr, "tpr": tpr})))
        pr, rc, _ = precision_recall_curve(y, p)
        rows_pr.append(thin(pd.DataFrame({"model": m, "recall": rc, "precision": pr})))
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        rows_rel.append(pd.DataFrame({"model": m, "mean_pred": mean, "observed": frac}))
    pd.concat(rows_roc).to_csv(OUT / "curve_roc.csv", index=False)
    pd.concat(rows_pr).to_csv(OUT / "curve_pr.csv", index=False)
    pd.concat(rows_rel).to_csv(OUT / "curve_reliability.csv", index=False)

    # --- CLV holdout aggregates --------------------------------------------------------
    ch = pd.read_parquet(ART / "clv_holdout.parquet")
    ch.assign(fbin=np.minimum(ch["frequency_cal"], 15)).groupby("fbin").agg(
        customers=("customer_id", "size"), actual=("frequency_holdout", "mean"),
        predicted=("pred_purchases", "mean")).reset_index().to_csv(OUT / "clv_holdout_by_freq.csv", index=False)
    ch.assign(decile=pd.qcut(ch["pred_spend"].rank(method="first"), 10, labels=range(1, 11)).astype(int)) \
        .groupby("decile").agg(customers=("customer_id", "size"), actual_spend=("actual_spend", "mean"),
                               predicted_spend=("pred_spend", "mean")).reset_index() \
        .to_csv(OUT / "clv_holdout_by_decile.csv", index=False)

    # --- scored customers (test cutoff) + SHAP ------------------------------------------
    sc = pd.read_parquet(ART / "test_scored.parquet")
    clv = pd.read_parquet(ART / "clv_scores_test_cutoff.parquet")[
        ["customer_id", "p_alive", "pred_purchases_90d", "expected_aov", "clv_12m"]]
    seg = pd.read_csv(ART / "customer_scores_2011-09-01.csv", dtype={"customer_id": str})[
        ["customer_id", "risk_segment", "value_segment"]]
    from churn_clv.features import FEATURE_COLUMNS

    cust = sc[["customer_id"] + FEATURE_COLUMNS + ["churn_prob", "churn_prob_raw", "churned", "future_invoices",
                                                   "future_revenue"]] \
        .merge(clv, on="customer_id").merge(seg, on="customer_id")
    for c in cust.select_dtypes("float").columns:
        cust[c] = cust[c].astype("float32")
    cust.to_parquet(OUT / "customers.parquet", index=False)
    for m in ["lgbm", "logreg"]:
        sv = pd.read_parquet(ART / f"shap_values_{m}.parquet")
        sv[FEATURE_COLUMNS] = sv[FEATURE_COLUMNS].astype("float32")
        sv.to_parquet(OUT / f"shap_values_{m}.parquet", index=False)
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"app/data: {len(list(OUT.iterdir()))} files, {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
