"""End-to-end orchestration: SQL model -> snapshots -> models -> CLV -> targeting -> campaigns -> demographics."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import campaigns as camp
from . import demographics as demo
from . import plots
from .clv import holdout_validation, score_clv
from .config import load_config, resolve
from .db import connect, export_tables, run_sql_model
from .features import (
    FEATURE_COLUMNS,
    build_panel,
    build_snapshot,
    data_end_date,
    label_window_complete,
    validate_split_design,
)
from .metrics import bootstrap_ci, classification_metrics, lift_table, paired_bootstrap_diff
from .models import LGBM_GRID, Calibrator, RecencyRule, RFMHeuristic, make_lgbm, make_logreg, proba
from .targeting import assign_segments, expected_net_value, segment_matrix, sensitivity_grid, targeting_summary

EXPORT_TABLES = ["dq_report", "dq_cleaning_waterfall", "fct_baskets", "fct_days", "dim_households",
                 "cohort_retention", "monthly_active_households", "household_rfm", "survival_households",
                 "survival_exclusions", "dim_campaigns", "campaign_exposure", "coupon_redemptions_clean"]


def _dirs(cfg):
    art, fig = resolve(cfg["paths"]["artifacts"]), resolve(cfg["paths"]["figures"])
    art.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    return art, fig


def _dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _scalar(con, sql):
    v = con.execute(sql).fetchone()[0]
    return float(v) if v is not None and not isinstance(v, (int, str)) and not hasattr(v, "isoformat") else v


# ----------------------------------------------------------------------------------------- SQL
def run_sql(cfg=None) -> dict:
    cfg = cfg or load_config()
    art, fig = _dirs(cfg)
    con = connect()
    run_sql_model(con, cfg=cfg)
    export_tables(con, resolve(cfg["paths"]["processed"]), EXPORT_TABLES)
    q = lambda s: _scalar(con, s)  # noqa: E731
    total_spend = q("SELECT CAST(SUM(spend) AS DOUBLE) FROM fct_baskets")
    s = {
        "dq_report": dict(con.execute("SELECT * FROM dq_report").fetchall()),
        "waterfall": dict(con.execute("SELECT * FROM dq_cleaning_waterfall").fetchall()),
        "n_baskets_clean": q("SELECT COUNT(*) FROM fct_baskets"),
        "n_households": q("SELECT COUNT(*) FROM dim_households"),
        "n_shopping_days": q("SELECT COUNT(*) FROM fct_days"),
        "n_stores": q("SELECT COUNT(DISTINCT store_id) FROM fct_baskets"),
        "spend_clean_usd": total_spend,
        "returns_clean_usd": q("SELECT COALESCE(CAST(SUM(returned_amount) AS DOUBLE), 0) FROM fct_returns"),
        "date_min": str(q("SELECT MIN(basket_date) FROM fct_baskets")),
        "date_max": str(q("SELECT MAX(basket_date) FROM fct_baskets")),
        "baskets_dropped_not_purchases": q("SELECT COUNT(DISTINCT basket_id) FROM stg_transactions") - q("SELECT COUNT(*) FROM fct_baskets"),
        "median_shopping_days_per_household": q("SELECT MEDIAN(n_shopping_days) FROM dim_households"),
        "median_basket_spend_usd": q("SELECT MEDIAN(CAST(spend AS DOUBLE)) FROM fct_baskets"),
        "mean_basket_spend_usd": total_spend / q("SELECT COUNT(*) FROM fct_baskets"),
        "single_shopping_day_share": q("SELECT AVG(CASE WHEN n_shopping_days = 1 THEN 1.0 ELSE 0 END) FROM dim_households"),
        "top10pct_spend_share": float(con.execute(
            "SELECT CAST(lifetime_spend AS DOUBLE) s FROM dim_households ORDER BY s DESC").df()["s"]
            .pipe(lambda x: x.head(int(len(x) * 0.1)).sum() / x.sum())),
        "spend_share_by_dept_group": {k: float(v) / total_spend for k, v in con.execute(
            "SELECT dept_group, CAST(SUM(sales_value) AS DOUBLE) FROM fct_lines WHERE line_type IN ('merchandise','free_item') "
            "GROUP BY 1 ORDER BY 2 DESC").fetchall()},
        "retail_disc_share_of_shelf": q("SELECT CAST(SUM(retail_disc) AS DOUBLE) / CAST(SUM(shelf_value) AS DOUBLE) FROM fct_baskets"),
        "coupon_disc_share_of_shelf": q("SELECT CAST(SUM(coupon_disc) + SUM(coupon_match_disc) AS DOUBLE) / CAST(SUM(shelf_value) AS DOUBLE) FROM fct_baskets"),
        "baskets_with_coupon_share": q("SELECT AVG(CASE WHEN has_coupon THEN 1.0 ELSE 0 END) FROM fct_baskets"),
        "n_households_with_demographics": q("SELECT COUNT(*) FROM dim_households WHERE has_demographics"),
        "n_campaigns": q("SELECT COUNT(*) FROM dim_campaigns"),
        "n_households_targeted_any": q("SELECT COUNT(DISTINCT household_id) FROM campaign_exposure"),
        "n_households_ever_redeemed": q("SELECT COUNT(DISTINCT household_id) FROM coupon_redemptions_clean"),
        "n_redemptions": q("SELECT COUNT(*) FROM coupon_redemptions_clean"),
        "cohort_sizes": {str(k)[:7]: int(v) for k, v in con.execute(
            "SELECT entry_cohort, COUNT(*) FROM dim_households GROUP BY 1 ORDER BY 1").fetchall()},
        "month1_retention_mean_excl_jan": q(
            "SELECT AVG(retention_rate) FROM cohort_retention WHERE months_since_entry = 1 AND entry_cohort > DATE '2017-01-01'"),
        "rfm_segments": dict(con.execute(
            "SELECT rfm_segment, COUNT(*) FROM household_rfm GROUP BY 1 ORDER BY 2 DESC").fetchall()),
        "survival_exclusions": dict(con.execute("SELECT reason, n FROM survival_exclusions").fetchall()),
    }
    plots.cohort_heatmap(con.execute("SELECT * FROM cohort_retention").df(), fig / "cohort_retention_heatmap.png")
    con.close()
    _dump(s, art / "sql_summary.json")
    return s


# ----------------------------------------------------------------------------------------- churn
def _cv_calibration_choice(p_valid, y_valid, seed) -> dict:
    """Pick isotonic vs Platt by 5-fold cross-fitted Brier on the validation snapshot."""
    res = {}
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for method in ("isotonic", "platt"):
        oof = np.zeros_like(p_valid)
        for tr, te in skf.split(p_valid.reshape(-1, 1), y_valid):
            oof[te] = Calibrator(method).fit(p_valid[tr], y_valid[tr]).transform(p_valid[te])
        res[method] = float(np.mean((oof - y_valid) ** 2))
    return res


def _fit_all(train: pd.DataFrame, seed: int, lgbm_params: dict | None = None):
    y = train["churned"].to_numpy()
    X = train[FEATURE_COLUMNS]
    return {
        "Recency rule": RecencyRule().fit(train, y),
        "RFM heuristic": RFMHeuristic().fit(train, y),
        "Logistic regression": make_logreg(seed).fit(X, y),
        "LightGBM": make_lgbm(seed, lgbm_params).fit(X, y),
    }


def rolling_backtest(con, cfg, lgbm_params=None) -> pd.DataFrame:
    """Expanding-window, rolling-origin backtest: for each evaluation cutoff, train on every grid snapshot whose
    90-day label window closed on or before that cutoff (>= 3 snapshots), evaluate on the cutoff snapshot."""
    ch = cfg["churn"]
    h = pd.Timedelta(days=ch["horizon_days"])
    panel = {c: build_snapshot(con, c, ch["horizon_days"], ch["lookback_active_days"]) for c in ch["grid_cutoffs"]}
    rows = []
    for ev in ch["backtest_eval_cutoffs"]:
        ev_d = pd.Timestamp(ev)
        tr_cuts = [c for c in ch["grid_cutoffs"] if pd.Timestamp(c) + h <= ev_d]
        train = pd.concat([panel[c] for c in tr_cuts], ignore_index=True)
        ev_df = panel[ev] if ev in panel else build_snapshot(con, ev, ch["horizon_days"], ch["lookback_active_days"])
        models = _fit_all(train, ch["random_state"], lgbm_params)
        for name, m in models.items():
            met = classification_metrics(ev_df["churned"], proba(m, ev_df))
            rows.append({"eval_cutoff": ev, "n_train_snapshots": len(tr_cuts), "train_last_cutoff": tr_cuts[-1],
                         "n_train_rows": len(train), "n_train_positives": int(train["churned"].sum()),
                         "model": name, "n_eval": len(ev_df), "n_eval_positives": int(ev_df["churned"].sum()), **met})
    return pd.DataFrame(rows)


def train(cfg=None) -> dict:
    cfg = cfg or load_config()
    art, fig = _dirs(cfg)
    ch, seed = cfg["churn"], cfg["churn"]["random_state"]
    con = connect(read_only=True)
    problems = validate_split_design(cfg, data_end_date(con))
    assert not problems, problems
    for c in ch["train_cutoffs"] + [ch["valid_cutoff"], ch["test_cutoff"]]:
        assert label_window_complete(con, c, ch["horizon_days"]), f"label window incomplete for {c}"
    lb = ch["lookback_active_days"]
    train_df = build_panel(con, ch["train_cutoffs"], ch["horizon_days"], lb)
    valid = build_snapshot(con, ch["valid_cutoff"], ch["horizon_days"], lb)
    test = build_snapshot(con, ch["test_cutoff"], ch["horizon_days"], lb)
    for d, n in [(train_df, "train"), (valid, "valid"), (test, "test")]:
        d.to_parquet(art / f"snapshot_{n}.parquet", index=False)

    # LightGBM hyper-parameters: small grid, chosen by validation PR-AUC.
    grid = []
    for gp in LGBM_GRID:
        m = make_lgbm(seed, gp).fit(train_df[FEATURE_COLUMNS], train_df["churned"])
        grid.append({**gp, "valid_pr_auc": classification_metrics(valid["churned"], proba(m, valid))["pr_auc"]})
    lgbm_params = {k: v for k, v in max(grid, key=lambda r: r["valid_pr_auc"]).items() if k != "valid_pr_auc"}
    pd.DataFrame(grid).to_csv(art / "lgbm_grid_valid.csv", index=False)
    models = _fit_all(train_df, seed, lgbm_params)
    yv, yt = valid["churned"].to_numpy(), test["churned"].to_numpy()
    results, test_probs, cal_info = [], {}, {}
    for name, m in models.items():
        pv, pt = proba(m, valid), proba(m, test)
        results.append({"model": name, "calibration": "none", "split": "valid", **classification_metrics(yv, pv)})
        results.append({"model": name, "calibration": "none", "split": "test", **classification_metrics(yt, pt)})
        test_probs[name] = pt
        if name in ("Logistic regression", "LightGBM"):
            cv = _cv_calibration_choice(pv, yv, seed)
            cal_info[name] = {"cv_brier_on_valid": cv}
            for method in ("isotonic", "platt"):
                cal = Calibrator(method).fit(pv, yv)
                test_probs[f"{name} + {method}"] = cal.transform(pt)
                results.append({"model": name, "calibration": method, "split": "test",
                                **classification_metrics(yt, test_probs[f"{name} + {method}"])})
            # Isotonic regression is a step function that overfits and emits exact 0/1 probabilities when the
            # calibration set has few positives; it is evaluated and reported but only eligible as the final
            # calibrator with >= min_positives_isotonic validation positives (config). Otherwise Platt vs none.
            iso_ok = int(yv.sum()) >= int(cfg["churn"]["min_positives_isotonic"])
            cv["none"] = float(np.mean((pv - yv) ** 2))
            eligible = {k: v for k, v in cv.items() if iso_ok or k != "isotonic"}
            best = min(eligible, key=eligible.get)
            cal_info[name]["isotonic_eligible"] = iso_ok
            cal_info[name]["valid_positives"] = int(yv.sum())
            cal_info[name]["chosen"] = best
    res = pd.DataFrame(results)
    res.to_csv(art / "model_metrics.csv", index=False)

    # Final model = the ML model with the best VALIDATION PR-AUC, with its CV-chosen calibrator.
    vres = res[(res.split == "valid") & res.model.isin(["Logistic regression", "LightGBM"])]
    final_name = vres.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    best_method = cal_info[final_name]["chosen"]
    final_label = final_name if best_method == "none" else f"{final_name} + {best_method}"
    final_cal = Calibrator(best_method).fit(proba(models[final_name], valid), yv)
    joblib.dump({"model": models[final_name], "model_name": final_name, "calibrator": final_cal,
                 "features": FEATURE_COLUMNS, "train_cutoffs": ch["train_cutoffs"],
                 "calibration_cutoff": ch["valid_cutoff"]}, art / "churn_model.joblib")
    test_scored = test.assign(churn_prob=test_probs[final_label], churn_prob_raw=test_probs[final_name])
    test_scored.to_parquet(art / "test_scored.parquet", index=False)
    pd.DataFrame({"household_id": test["household_id"], "churned": yt, **test_probs}).to_parquet(
        art / "test_probs.parquet", index=False)

    # Uncertainty: bootstrap CIs of the test metrics and a paired LightGBM-vs-logistic comparison.
    n_boot = ch["bootstrap_reps"]
    ci_rows = []
    for k, p in test_probs.items():
        r_lo, r_hi = bootstrap_ci(yt, p, roc_auc_score, n_boot, seed)
        p_lo, p_hi = bootstrap_ci(yt, p, average_precision_score, n_boot, seed)
        ci_rows.append({"model": k, "roc_auc": float(roc_auc_score(yt, p)), "roc_auc_lo": r_lo, "roc_auc_hi": r_hi,
                        "pr_auc": float(average_precision_score(yt, p)), "pr_auc_lo": p_lo, "pr_auc_hi": p_hi})
    pd.DataFrame(ci_rows).to_csv(art / "test_metric_ci.csv", index=False)
    paired = {
        "lightgbm_minus_logreg_roc_auc": paired_bootstrap_diff(
            yt, test_probs["LightGBM"], test_probs["Logistic regression"], roc_auc_score, n_boot, seed),
        "lightgbm_minus_logreg_pr_auc": paired_bootstrap_diff(
            yt, test_probs["LightGBM"], test_probs["Logistic regression"], average_precision_score, n_boot, seed),
        "logreg_minus_rfm_roc_auc": paired_bootstrap_diff(
            yt, test_probs["Logistic regression"], test_probs["RFM heuristic"], roc_auc_score, n_boot, seed),
        "logreg_minus_rfm_pr_auc": paired_bootstrap_diff(
            yt, test_probs["Logistic regression"], test_probs["RFM heuristic"], average_precision_score, n_boot, seed),
    }

    lifts = {k: lift_table(yt, test_probs[k]) for k in
             ["Recency rule", "RFM heuristic", "Logistic regression", final_label]}
    pd.concat([v.assign(model=k) for k, v in lifts.items()]).to_csv(art / "lift_tables.csv", index=False)
    plots.roc_pr(yt, {k: test_probs[k] for k in lifts}, fig / "roc_pr_test.png")
    plots.gains(lifts, fig / "gains_lift_test.png")
    plots.reliability(yt, {"Logistic regression (raw)": test_probs["Logistic regression"],
                           "LightGBM (raw)": test_probs["LightGBM"],
                           "LightGBM + Platt": test_probs["LightGBM + platt"],
                           "LightGBM + isotonic": test_probs["LightGBM + isotonic"]}, fig / "reliability_test.png")
    calib_in_large = {k: {"mean_pred": float(np.mean(v)), "observed": float(yt.mean())} for k, v in test_probs.items()}

    bt = rolling_backtest(con, cfg, lgbm_params)
    bt.to_csv(art / "backtest_metrics.csv", index=False)
    plots.backtest(bt, fig / "backtest_pr_auc.png")

    shap_summary = explain(models["LightGBM"], test_scored, fig, art)
    shap_lr = explain_linear(models["Logistic regression"], train_df, test_scored, fig, art)
    con.close()
    summary = {
        "n_train_rows": len(train_df), "n_valid": len(valid), "n_test": len(test),
        "n_train_positives": int(train_df["churned"].sum()), "n_valid_positives": int(yv.sum()),
        "n_test_positives": int(yt.sum()),
        "churn_rate": {"train": float(train_df["churned"].mean()), "valid": float(yv.mean()),
                       "test": float(yt.mean())},
        "churn_rate_by_train_cutoff": train_df.groupby(train_df["cutoff"].astype(str))["churned"].mean().to_dict(),
        "n_by_train_cutoff": train_df.groupby(train_df["cutoff"].astype(str)).size().to_dict(),
        "calibration": cal_info, "final_model": final_label, "lgbm_params": lgbm_params,
        "calibration_in_the_large_test": calib_in_large, "paired_bootstrap_test": paired,
        "test_metrics": res[res.split == "test"].set_index(["model", "calibration"]).drop(columns="split")
        .reset_index().to_dict(orient="records"),
        "shap_top_lgbm": shap_summary, "shap_top_logreg": shap_lr,
    }
    _dump(summary, art / "train_summary.json")
    return summary


def explain(model, test: pd.DataFrame, fig_dir: Path, art: Path, n_sample: int = 2000) -> list:
    import matplotlib.pyplot as plt
    import shap

    X = test[FEATURE_COLUMNS]
    Xs = X.sample(min(n_sample, len(X)), random_state=0)
    expl = shap.TreeExplainer(model)
    sv = expl(Xs)
    imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": np.abs(sv.values).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(art / "shap_importance.csv", index=False)
    full = expl(X)
    pd.DataFrame(full.values, columns=FEATURE_COLUMNS).assign(
        household_id=test["household_id"].to_numpy(), base_value=float(np.ravel(full.base_values)[0])
    ).to_parquet(art / "shap_values_lgbm.parquet", index=False)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("Global SHAP (LightGBM, test cutoff; log-odds of churn proxy)")
    plots.save(plt.gcf(), fig_dir / "shap_beeswarm.png")
    plt.figure()
    shap.plots.bar(sv, max_display=12, show=False)
    plots.save(plt.gcf(), fig_dir / "shap_bar.png")
    # Local explanation: highest-spend household among the top-decile risk group.
    raw = model.predict_proba(X)[:, 1]
    top = test.assign(r=raw)
    top = top[top["r"] >= np.quantile(raw, 0.9)].sort_values("monetary_total", ascending=False).iloc[0]
    idx = test.index.get_loc(top.name)
    local = expl(X.iloc[[idx]])
    plt.figure()
    shap.plots.waterfall(local[0], max_display=10, show=False)
    plt.title(f"Local SHAP: household {top['household_id']} (high spend, top-decile risk)")
    plots.save(plt.gcf(), fig_dir / "shap_local_waterfall.png")
    return imp.head(8).to_dict(orient="records")


def explain_linear(pipe, train: pd.DataFrame, test: pd.DataFrame, fig_dir: Path, art: Path) -> list:
    """Exact linear SHAP for the logistic-regression pipeline (in transformed log/standardised space)."""
    import matplotlib.pyplot as plt
    import shap

    pre = pipe[:-1]
    bg = pd.DataFrame(pre.transform(train[FEATURE_COLUMNS].sample(1000, random_state=0)), columns=FEATURE_COLUMNS)
    Xt = pd.DataFrame(pre.transform(test[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS)
    expl = shap.LinearExplainer(pipe[-1], bg)
    sv = expl(Xt)
    imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": np.abs(sv.values).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(art / "shap_importance_logreg.csv", index=False)
    pd.DataFrame(sv.values, columns=FEATURE_COLUMNS).assign(
        household_id=test["household_id"].to_numpy(), base_value=float(np.ravel(sv.base_values)[0])
    ).to_parquet(art / "shap_values_logreg.parquet", index=False)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("Global SHAP (logistic regression, test cutoff; transformed features)")
    plots.save(plt.gcf(), fig_dir / "shap_beeswarm_logreg.png")
    return imp.head(8).to_dict(orient="records")


# ----------------------------------------------------------------------------------------- CLV
def run_clv(cfg=None) -> dict:
    cfg = cfg or load_config()
    art, fig = _dirs(cfg)
    c = cfg["clv"]
    con = connect(read_only=True)
    days = con.execute("SELECT household_id, visit_date, CAST(spend AS DOUBLE) AS spend FROM fct_days").df()
    con.close()
    ch, m = holdout_validation(days, c["calibration_end"], c["observation_end"], c["penalizer"])
    ch.to_parquet(art / "clv_holdout.parquet", index=False)
    plots.clv_holdout(ch, fig / "clv_holdout_validation.png")
    _, m_short = holdout_validation(days, c["calibration_end_short"], c["observation_end"], c["penalizer"])
    m["short_holdout_13w"] = m_short
    scores = score_clv(days, cfg["churn"]["test_cutoff"], c["horizon_months"], c["discount_rate_monthly"],
                       c["penalizer"])
    scores.to_parquet(art / "clv_scores_test_cutoff.parquet", index=False)
    m["scoring_as_of"] = cfg["churn"]["test_cutoff"]
    m["scoring_params"] = {"bgf": scores.attrs["bgf_params"], "ggf": scores.attrs["ggf_params"]}
    m["clv_12m_total_scored"] = float(scores["clv_12m"].sum())
    m["clv_12m_median"] = float(scores["clv_12m"].median())
    m["n_scored_all_households"] = int(len(scores))
    _dump(m, art / "clv_summary.json")
    return m


# ----------------------------------------------------------------------------------------- targeting
def _load_scored(art, cfg) -> pd.DataFrame:
    test = pd.read_parquet(art / "test_scored.parquet")
    clv = pd.read_parquet(art / "clv_scores_test_cutoff.parquet")[
        ["household_id", "p_alive", "pred_purchases_90d", "expected_aov", "clv_12m"]]
    df = test.merge(clv, on="household_id", how="left")
    assert df["clv_12m"].notna().all()
    t = cfg["targeting"]
    return assign_segments(df, t["risk_quantiles"], t["risk_labels"], t["value_quantiles"], t["value_labels"])


def run_targeting(cfg=None) -> dict:
    cfg = cfg or load_config()
    art, fig = _dirs(cfg)
    t = cfg["targeting"]
    df = _load_scored(art, cfg)
    df["expected_net_value_base"] = expected_net_value(df["churn_prob"], df["clv_12m"], t["uplift"],
                                                       t["gross_margin"], t["contact_cost_usd"])
    df["target_base"] = df["expected_net_value_base"] > 0
    mat = segment_matrix(df)
    # OBSERVED outcomes in the 90 days after the scoring date (this is a backtest, so they are known)
    obs = df.groupby(["value_segment", "risk_segment"], observed=False).agg(
        observed_churn_rate=("churned", "mean"), observed_spend_90d=("future_spend", "sum")).reset_index()
    mat = mat.merge(obs, on=["value_segment", "risk_segment"])
    mat.to_csv(art / "targeting_matrix.csv", index=False)
    plots.targeting_matrix(mat, fig / "targeting_matrix.png", cfg["churn"]["test_cutoff"])
    sens = sensitivity_grid(df, t["sensitivity"]["uplift"], t["sensitivity"]["gross_margin"],
                            t["sensitivity"]["contact_cost_usd"])
    sens.to_csv(art / "targeting_sensitivity.csv", index=False)
    plots.sensitivity_heatmap(sens, t["gross_margin"], fig / "targeting_sensitivity.png")
    base = targeting_summary(df, t["uplift"], t["gross_margin"], t["contact_cost_usd"])
    tg = df[df["target_base"]]
    base["targeted_observed_churn_rate"] = float(tg["churned"].mean()) if len(tg) else None
    mix = tg.groupby(["value_segment", "risk_segment"], observed=True).size()
    base["targeted_segment_mix"] = {f"{v} / {r}": int(n) for (v, r), n in mix.items()}
    out_cols = ["household_id", "cutoff", "recency_days", "frequency", "monetary_total", "churn_prob", "p_alive",
                "pred_purchases_90d", "clv_12m", "risk_segment", "value_segment", "expected_net_value_base",
                "target_base", "churned", "future_spend"]
    df[out_cols].to_csv(art / f"household_scores_{cfg['churn']['test_cutoff']}.csv", index=False)
    base["spearman_clv_vs_actual_90d_spend"] = float(df[["clv_12m", "future_spend"]].corr("spearman").iloc[0, 1])
    base["spearman_bgnbd_trips_vs_actual_90d"] = float(
        df[["pred_purchases_90d", "future_trips"]].corr("spearman").iloc[0, 1])
    summary = {"base_case_ASSUMED": base, "segment_sizes": mat[["value_segment", "risk_segment", "households"]]
               .astype({"value_segment": str, "risk_segment": str}).to_dict(orient="records"),
               "sensitivity_range_net_value_usd": [float(sens.expected_net_value_usd.min()),
                                                   float(sens.expected_net_value_usd.max())],
               "sensitivity_range_n_targeted": [int(sens.n_targeted.min()), int(sens.n_targeted.max())]}
    _dump(summary, art / "targeting_summary.json")
    return summary


# ----------------------------------------------------------------------------------------- campaigns
def run_campaigns(cfg=None) -> dict:
    """Observational campaign / coupon analysis at the test cutoff (descriptive, never causal)."""
    cfg = cfg or load_config()
    art, fig = _dirs(cfg)
    ch = cfg["churn"]
    con = connect(read_only=True)
    dimc = con.execute("SELECT * FROM dim_campaigns").df()
    win = camp.window_exposure(con, ch["test_cutoff"], ch["horizon_days"])
    con.close()
    df = _load_scored(art, cfg).merge(win, on="household_id", how="left")
    df["targeted_in_window"] = df["n_window_campaigns"] > 0
    df["type_a_in_window"] = df["n_window_type_a"] > 0
    df["prior_redeemer"] = df["n_prior_redemptions"] > 0
    df["targeted_before_cutoff"] = df["n_prior_campaigns"] > 0
    tab = camp.outcome_by_flag(df, "targeted_in_window")
    tab_a = camp.outcome_by_flag(df, "type_a_in_window")
    tab_r = camp.outcome_by_flag(df, "prior_redeemer")
    tab.to_csv(art / "campaign_outcomes_targeted.csv", index=False)
    tab_a.to_csv(art / "campaign_outcomes_type_a.csv", index=False)
    tab_r.to_csv(art / "campaign_outcomes_prior_redeemer.csv", index=False)
    plots.campaign_outcomes(tab[tab["risk_segment"] != "All scored"], fig / "campaign_outcomes.png")
    balance = camp.standardized_mean_differences(df, "targeted_in_window")
    balance_r = camp.standardized_mean_differences(df, "prior_redeemer")
    balance.to_csv(art / "campaign_balance_targeted.csv", index=False)
    balance_r.to_csv(art / "campaign_balance_prior_redeemer.csv", index=False)
    dimc.to_csv(art / "campaign_summary.csv", index=False)
    camp.campaign_type_summary(dimc).to_csv(art / "campaign_type_summary.csv", index=False)
    # coupon-use gradient: churn-proxy by coupon-discount-share quartile (association)
    q = pd.qcut(df["coupon_disc_share"].rank(method="first"), 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    grad = df.assign(coupon_share_quartile=q).groupby("coupon_share_quartile", observed=True).apply(
        lambda g: pd.Series(camp._rate_row(g)), include_groups=False).reset_index()
    grad["mean_coupon_disc_share"] = df.assign(q=q).groupby("q", observed=True)["coupon_disc_share"].mean().to_numpy()
    grad.to_csv(art / "campaign_coupon_gradient.csv", index=False)
    summary = {
        "n_scored": len(df),
        "targeted_in_window_n": int(df["targeted_in_window"].sum()),
        "type_a_in_window_n": int(df["type_a_in_window"].sum()),
        "prior_redeemer_n": int(df["prior_redeemer"].sum()),
        "targeted_before_cutoff_n": int(df["targeted_before_cutoff"].sum()),
        "adjusted_difference_targeted": camp.adjusted_difference(df, "targeted_in_window"),
        "adjusted_difference_type_a": camp.adjusted_difference(df, "type_a_in_window"),
        "adjusted_difference_prior_redeemer": camp.adjusted_difference(df, "prior_redeemer"),
        "max_abs_smd_targeted": float(balance["smd"].abs().max()),
        "max_abs_smd_prior_redeemer": float(balance_r["smd"].abs().max()),
        "campaign_window": {"start": ch["test_cutoff"], "horizon_days": ch["horizon_days"]},
        "caveat": "Observational association. Targeting was not randomised and there is no holdout group.",
    }
    _dump(summary, art / "campaign_summary.json")
    return summary


# ----------------------------------------------------------------------------------------- demographics
def run_demographics(cfg=None) -> dict:
    cfg = cfg or load_config()
    art, fig = _dirs(cfg)
    con = connect(read_only=True)
    dh = con.execute("SELECT household_id, has_demographics, age, income, home_ownership, marital_status, "
                     "household_size, household_comp, kids_count FROM dim_households").df()
    n_cov = int(dh["has_demographics"].sum())
    n_all = len(dh)
    con.close()
    df = _load_scored(art, cfg).merge(dh, on="household_id", how="left")
    tab = demo.breakout(df, cfg["demographics"]["min_cell_size"])
    tab.to_csv(art / "demographics_breakout.csv", index=False)
    cov = demo.coverage_comparison(df)
    cov.to_csv(art / "demographics_coverage.csv", index=False)
    plots.demographics_profile(tab, fig / "demographics_profile.png")
    summary = {"n_households": n_all, "n_with_demographics": n_cov, "coverage_share": n_cov / n_all,
               "n_scored_with_demographics": int(df["has_demographics"].sum()),
               "n_scored": len(df),
               "n_churned_with_demographics": int(df.loc[df["has_demographics"], "churned"].sum()),
               "unreliable_cells": int((~tab["reliable"]).sum()), "cells": int(len(tab)),
               "churn_rate_with_demographics": float(df.loc[df["has_demographics"], "churned"].mean()),
               "churn_rate_without_demographics": float(df.loc[~df["has_demographics"], "churned"].mean()),
               "caveat": "Partial coverage; covered households are not guaranteed representative; cells are small; "
                         "too few churn-proxy events among covered households for group churn rates to be informative."}
    _dump(summary, art / "demographics_summary.json")
    return summary


# ----------------------------------------------------------------------------------------- scoring
def score(cutoff: str, out: str | None = None, cfg=None) -> Path:
    """Score households active before `cutoff` with the saved calibrated model + CLV fitted pre-cutoff."""
    cfg = cfg or load_config()
    art, _ = _dirs(cfg)
    bundle = joblib.load(art / "churn_model.joblib")
    con = connect(read_only=True)
    has_label = label_window_complete(con, cutoff, cfg["churn"]["horizon_days"])
    snap = build_snapshot(con, cutoff, cfg["churn"]["horizon_days"], cfg["churn"]["lookback_active_days"],
                          with_label=has_label)
    days = con.execute("SELECT household_id, visit_date, CAST(spend AS DOUBLE) AS spend FROM fct_days").df()
    con.close()
    raw = bundle["model"].predict_proba(snap[bundle["features"]])[:, 1]
    snap["churn_prob"] = bundle["calibrator"].transform(raw)
    c = cfg["clv"]
    clv = score_clv(days, cutoff, c["horizon_months"], c["discount_rate_monthly"], c["penalizer"])
    snap = snap.merge(clv[["household_id", "p_alive", "clv_12m"]], on="household_id", how="left")
    t = cfg["targeting"]
    snap = assign_segments(snap, t["risk_quantiles"], t["risk_labels"], t["value_quantiles"], t["value_labels"])
    snap["expected_net_value_assumed"] = expected_net_value(snap["churn_prob"], snap["clv_12m"], t["uplift"],
                                                            t["gross_margin"], t["contact_cost_usd"])
    path = Path(out) if out else art / f"scores_{cutoff}.csv"
    snap.to_csv(path, index=False)
    return path
