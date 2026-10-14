"""End-to-end orchestration: SQL model -> snapshots -> models -> CLV -> targeting."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import plots
from .clv import holdout_validation, score_clv
from .config import load_config, resolve
from .db import connect, export_tables, run_sql_model
from .features import FEATURE_COLUMNS, build_panel, build_snapshot, label_window_complete
from .metrics import classification_metrics, lift_table
from .models import LGBM_GRID, Calibrator, RecencyRule, RFMHeuristic, make_lgbm, make_logreg, proba
from .targeting import assign_segments, expected_net_value, segment_matrix, sensitivity_grid, targeting_summary

EXPORT_TABLES = ["dq_report", "dq_cleaning_waterfall", "fct_invoices", "dim_customers", "cohort_retention",
                 "monthly_active_customers", "customer_rfm", "survival_customers"]


def _dirs(cfg):
    art, fig = resolve(cfg["paths"]["artifacts"]), resolve(cfg["paths"]["figures"])
    art.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    return art, fig


def _dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ----------------------------------------------------------------------------------------- SQL
def run_sql(cfg=None) -> dict:
    cfg = cfg or load_config()
    art, fig = _dirs(cfg)
    con = connect()
    run_sql_model(con, cfg=cfg)
    export_tables(con, resolve("data/processed"), EXPORT_TABLES)
    s = {
        "dq_report": dict(con.execute("SELECT * FROM dq_report").fetchall()),
        "waterfall": dict(con.execute("SELECT * FROM dq_cleaning_waterfall").fetchall()),
        "n_invoices_clean": con.execute("SELECT COUNT(*) FROM fct_invoices").fetchone()[0],
        "n_customers_clean": con.execute("SELECT COUNT(*) FROM dim_customers").fetchone()[0],
        "revenue_clean_gbp": con.execute("SELECT SUM(revenue) FROM fct_invoices").fetchone()[0],
        "returns_clean_gbp": con.execute("SELECT SUM(returned_amount) FROM fct_returns").fetchone()[0],
        "date_min": str(con.execute("SELECT MIN(invoice_date) FROM fct_invoices").fetchone()[0]),
        "date_max": str(con.execute("SELECT MAX(invoice_date) FROM fct_invoices").fetchone()[0]),
        "n_countries": con.execute("SELECT COUNT(DISTINCT home_country) FROM dim_customers").fetchone()[0],
        "uk_customer_share": con.execute(
            "SELECT AVG(CASE WHEN home_country='United Kingdom' THEN 1 ELSE 0 END) FROM dim_customers").fetchone()[0],
        "one_time_buyer_share": con.execute(
            "SELECT AVG(CASE WHEN n_invoices=1 THEN 1 ELSE 0 END) FROM dim_customers").fetchone()[0],
        "month1_retention_mean": con.execute(
            "SELECT AVG(retention_rate) FROM cohort_retention WHERE months_since_acquisition=1").fetchone()[0],
        "rfm_segments": dict(con.execute(
            "SELECT rfm_segment, COUNT(*) FROM customer_rfm GROUP BY 1 ORDER BY 2 DESC").fetchall()),
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
    ch = cfg["churn"]
    all_cuts = ch["train_cutoffs"] + [ch["valid_cutoff"], ch["test_cutoff"]]
    panel = {c: build_snapshot(con, c, ch["horizon_days"], ch["lookback_active_days"]) for c in all_cuts}
    rows = []
    for ev in all_cuts[2:]:
        ev_d = pd.Timestamp(ev)
        tr_cuts = [c for c in all_cuts if pd.Timestamp(c) + pd.Timedelta(days=ch["horizon_days"]) <= ev_d]
        train = pd.concat([panel[c] for c in tr_cuts], ignore_index=True)
        models = _fit_all(train, ch["random_state"], lgbm_params)
        for name, m in models.items():
            met = classification_metrics(panel[ev]["churned"], proba(m, panel[ev]))
            rows.append({"eval_cutoff": ev, "train_cutoffs": ",".join(tr_cuts), "model": name,
                         "n_eval": len(panel[ev]), **met})
    return pd.DataFrame(rows)


def train(cfg=None) -> dict:
    cfg = cfg or load_config()
    art, fig = _dirs(cfg)
    ch, seed = cfg["churn"], cfg["churn"]["random_state"]
    con = connect(read_only=True)
    for c in ch["train_cutoffs"] + [ch["valid_cutoff"], ch["test_cutoff"]]:
        assert label_window_complete(con, c, ch["horizon_days"]), f"label window incomplete for {c}"
    train_df = build_panel(con, ch["train_cutoffs"], ch["horizon_days"], ch["lookback_active_days"])
    valid = build_snapshot(con, ch["valid_cutoff"], ch["horizon_days"], ch["lookback_active_days"])
    test = build_snapshot(con, ch["test_cutoff"], ch["horizon_days"], ch["lookback_active_days"])
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
                results.append({"model": name, "calibration": method, "split": "test",
                                **classification_metrics(yt, cal.transform(pt))})
            best = min(cv, key=cv.get)
            cal_info[name]["chosen"] = best
            test_probs[f"{name} + {best}"] = Calibrator(best).fit(pv, yv).transform(pt)
    res = pd.DataFrame(results)
    res.to_csv(art / "model_metrics.csv", index=False)

    # Final model = the ML model with the best VALIDATION PR-AUC, with its CV-chosen calibrator.
    vres = res[(res.split == "valid") & res.model.isin(["Logistic regression", "LightGBM"])]
    final_name = vres.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    best_method = cal_info[final_name]["chosen"]
    final_label = f"{final_name} + {best_method}"
    final_cal = Calibrator(best_method).fit(proba(models[final_name], valid), yv)
    joblib.dump({"model": models[final_name], "model_name": final_name, "calibrator": final_cal,
                 "features": FEATURE_COLUMNS, "train_cutoffs": ch["train_cutoffs"],
                 "calibration_cutoff": ch["valid_cutoff"]}, art / "churn_model.joblib")
    test_scored = test.assign(churn_prob=test_probs[final_label], churn_prob_raw=test_probs[final_name])
    test_scored.to_parquet(art / "test_scored.parquet", index=False)

    lifts = {k: lift_table(yt, test_probs[k]) for k in
             ["Recency rule", "RFM heuristic", "Logistic regression",
              f"LightGBM + {cal_info['LightGBM']['chosen']}"]}
    pd.concat([v.assign(model=k) for k, v in lifts.items()]).to_csv(art / "lift_tables.csv", index=False)
    plots.roc_pr(yt, {k: test_probs[k] for k in lifts}, fig / "roc_pr_test.png")
    plots.gains(lifts, fig / "gains_lift_test.png")
    lgb_lab = f"LightGBM + {cal_info['LightGBM']['chosen']}"
    lr_lab = f"Logistic regression + {cal_info['Logistic regression']['chosen']}"
    plots.reliability(yt, {"Logistic regression (raw)": test_probs["Logistic regression"],
                           lr_lab: test_probs[lr_lab], "LightGBM (raw)": test_probs["LightGBM"],
                           lgb_lab: test_probs[lgb_lab]}, fig / "reliability_test.png")
    calib_in_large = {k: {"mean_pred": float(np.mean(v)), "observed": float(yt.mean())} for k, v in test_probs.items()}

    bt = rolling_backtest(con, cfg, lgbm_params)
    bt.to_csv(art / "backtest_metrics.csv", index=False)
    plots.backtest(bt, fig / "backtest_pr_auc.png")

    shap_summary = explain(models["LightGBM"], test_scored, fig, art)
    shap_lr = explain_linear(models["Logistic regression"], train_df, test_scored, fig, art)
    con.close()
    summary = {
        "n_train_rows": len(train_df), "n_valid": len(valid), "n_test": len(test),
        "churn_rate": {"train": float(train_df["churned"].mean()), "valid": float(yv.mean()),
                       "test": float(yt.mean())},
        "churn_rate_by_train_cutoff": train_df.groupby(train_df["cutoff"].astype(str))["churned"].mean().to_dict(),
        "calibration": cal_info, "final_model": final_label, "lgbm_params": lgbm_params,
        "calibration_in_the_large_test": calib_in_large,
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
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("Global SHAP (LightGBM, test cutoff; log-odds of churn proxy)")
    plots.save(plt.gcf(), fig_dir / "shap_beeswarm.png")
    plt.figure()
    shap.plots.bar(sv, max_display=12, show=False)
    plots.save(plt.gcf(), fig_dir / "shap_bar.png")
    # Local explanation: highest-value customer among the top-decile risk group.
    raw = model.predict_proba(X)[:, 1]
    top = test.assign(r=raw)
    top = top[top["r"] >= np.quantile(raw, 0.9)].sort_values("monetary_total", ascending=False).iloc[0]
    idx = test.index.get_loc(top.name)
    local = expl(X.iloc[[idx]])
    plt.figure()
    shap.plots.waterfall(local[0], max_display=10, show=False)
    plt.title(f"Local SHAP: customer {top['customer_id']} (high spend, top-decile risk)")
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
    inv = con.execute("SELECT customer_id, invoice_date, revenue FROM fct_invoices").df()
    con.close()
    ch, m = holdout_validation(inv, c["calibration_end"], c["observation_end"], c["penalizer"])
    ch.to_parquet(art / "clv_holdout.parquet", index=False)
    plots.clv_holdout(ch, fig / "clv_holdout_validation.png")
    scores = score_clv(inv, cfg["churn"]["test_cutoff"], c["horizon_months"], c["discount_rate_monthly"],
                       c["penalizer"])
    scores.to_parquet(art / "clv_scores_test_cutoff.parquet", index=False)
    m["scoring_as_of"] = cfg["churn"]["test_cutoff"]
    m["scoring_params"] = {"bgf": scores.attrs["bgf_params"], "ggf": scores.attrs["ggf_params"]}
    m["clv_12m_total_scored"] = float(scores["clv_12m"].sum())
    m["clv_12m_median"] = float(scores["clv_12m"].median())
    _dump(m, art / "clv_summary.json")
    return m


# ----------------------------------------------------------------------------------------- targeting
def run_targeting(cfg=None) -> dict:
    cfg = cfg or load_config()
    art, fig = _dirs(cfg)
    t = cfg["targeting"]
    test = pd.read_parquet(art / "test_scored.parquet")
    clv = pd.read_parquet(art / "clv_scores_test_cutoff.parquet")[
        ["customer_id", "p_alive", "pred_purchases_90d", "expected_aov", "clv_12m"]]
    df = test.merge(clv, on="customer_id", how="left")
    assert df["clv_12m"].notna().all()
    df = assign_segments(df, t["risk_quantiles"], t["risk_labels"], t["value_quantiles"], t["value_labels"])
    df["expected_net_value_base"] = expected_net_value(df["churn_prob"], df["clv_12m"], t["uplift"],
                                                       t["gross_margin"], t["contact_cost_gbp"])
    df["target_base"] = df["expected_net_value_base"] > 0
    mat = segment_matrix(df)
    # OBSERVED outcomes in the 90 days after the scoring date (this is a backtest, so they are known)
    obs = df.groupby(["value_segment", "risk_segment"], observed=False).agg(
        observed_churn_rate=("churned", "mean"), observed_revenue_90d=("future_revenue", "sum")).reset_index()
    mat = mat.merge(obs, on=["value_segment", "risk_segment"])
    mat.to_csv(art / "targeting_matrix.csv", index=False)
    plots.targeting_matrix(mat, fig / "targeting_matrix.png")
    sens = sensitivity_grid(df, t["sensitivity"]["uplift"], t["sensitivity"]["gross_margin"],
                            t["sensitivity"]["contact_cost_gbp"])
    sens.to_csv(art / "targeting_sensitivity.csv", index=False)
    plots.sensitivity_heatmap(sens, t["gross_margin"], fig / "targeting_sensitivity.png")
    base = targeting_summary(df, t["uplift"], t["gross_margin"], t["contact_cost_gbp"])
    tg = df[df["target_base"]]
    base["targeted_observed_churn_rate"] = float(tg["churned"].mean()) if len(tg) else None
    mix = tg.groupby(["value_segment", "risk_segment"], observed=True).size()
    base["targeted_segment_mix"] = {f"{v} / {r}": int(n) for (v, r), n in mix.items()}
    out_cols = ["customer_id", "cutoff", "recency_days", "frequency", "monetary_total", "churn_prob", "p_alive",
                "pred_purchases_90d", "clv_12m", "risk_segment", "value_segment", "expected_net_value_base",
                "target_base", "churned", "future_revenue"]
    df[out_cols].to_csv(art / "customer_scores_2011-09-01.csv", index=False)
    # Is CLV ranking predictive of actual next-90d revenue? (observed)
    base["spearman_clv_vs_actual_90d_revenue"] = float(df[["clv_12m", "future_revenue"]].corr("spearman").iloc[0, 1])
    base["spearman_bgnbd_purchases_vs_actual_90d"] = float(
        df[["pred_purchases_90d", "future_invoices"]].corr("spearman").iloc[0, 1])
    summary = {"base_case_ASSUMED": base, "segment_sizes": mat[["value_segment", "risk_segment", "customers"]]
               .astype({"value_segment": str, "risk_segment": str}).to_dict(orient="records"),
               "sensitivity_range_net_value_gbp": [float(sens.expected_net_value_gbp.min()),
                                                   float(sens.expected_net_value_gbp.max())],
               "sensitivity_range_n_targeted": [int(sens.n_targeted.min()), int(sens.n_targeted.max())]}
    _dump(summary, art / "targeting_summary.json")
    return summary


# ----------------------------------------------------------------------------------------- scoring
def score(cutoff: str, out: str | None = None, cfg=None) -> Path:
    """Score customers active before `cutoff` with the saved calibrated model + CLV fitted pre-cutoff."""
    cfg = cfg or load_config()
    art, _ = _dirs(cfg)
    bundle = joblib.load(art / "churn_model.joblib")
    con = connect(read_only=True)
    has_label = label_window_complete(con, cutoff, cfg["churn"]["horizon_days"])
    snap = build_snapshot(con, cutoff, cfg["churn"]["horizon_days"], cfg["churn"]["lookback_active_days"],
                          with_label=has_label)
    inv = con.execute("SELECT customer_id, invoice_date, revenue FROM fct_invoices").df()
    con.close()
    raw = bundle["model"].predict_proba(snap[bundle["features"]])[:, 1]
    snap["churn_prob"] = bundle["calibrator"].transform(raw)
    c = cfg["clv"]
    clv = score_clv(inv, cutoff, c["horizon_months"], c["discount_rate_monthly"], c["penalizer"])
    snap = snap.merge(clv[["customer_id", "p_alive", "clv_12m"]], on="customer_id", how="left")
    t = cfg["targeting"]
    snap = assign_segments(snap, t["risk_quantiles"], t["risk_labels"], t["value_quantiles"], t["value_labels"])
    snap["expected_net_value_assumed"] = expected_net_value(snap["churn_prob"], snap["clv_12m"], t["uplift"],
                                                            t["gross_margin"], t["contact_cost_gbp"])
    path = Path(out) if out else art / f"scores_{cutoff}.csv"
    snap.to_csv(path, index=False)
    return path
