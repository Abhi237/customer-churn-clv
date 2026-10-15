import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import FEATURE_LABELS, GREEN, NAVY, RED, callout, csv, households, setup_page, shap_values, summary, usd

s = summary()
tr = s["train_summary"]
ch = s["config"]["churn"]
setup_page("Churn-proxy model", "Churn = no purchase in the 90 days after the cutoff (a behavioral proxy, not an "
           f"observed cancellation) | train {len(ch['train_cutoffs'])} cutoffs, validate {ch['valid_cutoff']}, "
           f"test {ch['test_cutoff']}")

c = st.columns(4)
c[0].metric("Training snapshots", f"{tr['n_train_rows']:,}", f"{len(ch['train_cutoffs'])} cutoffs, {tr['n_train_positives']} positives",
            delta_color="off")
c[1].metric("Validation / test households", f"{tr['n_valid']:,} / {tr['n_test']:,}",
            f"{tr['n_valid_positives']} / {tr['n_test_positives']} positives", delta_color="off")
c[2].metric("Churn-proxy rate (test)", f"{tr['churn_rate']['test']:.1%}",
            f"train {tr['churn_rate']['train']:.1%}, valid {tr['churn_rate']['valid']:.1%}", delta_color="off")
c[3].metric("Final model", tr["final_model"].replace("Logistic regression", "LogReg"),
            "best validation PR-AUC", delta_color="off")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Metrics", "ROC / PR", "Calibration", "Lift and gains", "Backtest"])
with tab1:
    m = csv("model_metrics.csv")
    split = st.segmented_control("Split", ["test", "valid"], default="test")
    show = m[m.split == (split or "test")].drop(columns=["split", "log_loss", "lift_top20", "gain_top20"])
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
        "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
        "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
        "ece": st.column_config.NumberColumn("ECE", format="%.4f"),
        "calib_slope": st.column_config.NumberColumn("Calib. slope", format="%.2f"),
        "calib_intercept": st.column_config.NumberColumn("Calib. intercept", format="%.2f"),
        "mean_pred": st.column_config.NumberColumn("Mean prediction", format="%.4f"),
        "lift_top10": st.column_config.NumberColumn("Lift top 10%", format="%.2fx"),
        "gain_top30": st.column_config.NumberColumn("Churners in top 30%", format="%.3f"),
        "base_rate": st.column_config.NumberColumn("Base rate", format="%.4f")})
    st.markdown("**Test ROC-AUC and PR-AUC with 95% bootstrap intervals (1,000 resamples)**")
    ci = csv("test_metric_ci.csv")
    st.dataframe(ci, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(c.replace("_", " "), format="%.3f") for c in ci.columns if c != "model"})
    pb = tr["paired_bootstrap_test"]
    p1, p2 = pb["lightgbm_minus_logreg_roc_auc"], pb["lightgbm_minus_logreg_pr_auc"]
    p3 = pb["logreg_minus_rfm_roc_auc"]
    callout(f"<b>Simple models are close.</b> LightGBM minus logistic regression: ROC-AUC {p1['diff']:+.3f} "
            f"({p1['lo']:+.3f} to {p1['hi']:+.3f}), PR-AUC {p2['diff']:+.3f} ({p2['lo']:+.3f} to {p2['hi']:+.3f}) "
            f"on {tr['n_test_positives']} test positives: not distinguishable. Logistic regression minus the RFM "
            f"heuristic: ROC-AUC {p3['diff']:+.3f} ({p3['lo']:+.3f} to {p3['hi']:+.3f}). "
            "Baselines: recency rule (churn rate by recency bin) and an RFM score built on weekly rates.")
    st.caption("The final model is chosen on validation PR-AUC, not on the test split. Calibrators are fit on the "
               "validation snapshot and evaluated on test.")
with tab2:
    roc, pr = csv("curve_roc.csv"), csv("curve_pr.csv")
    models = st.multiselect("Models", roc.model.unique().tolist(),
                            default=["Recency rule", "RFM heuristic", "Logistic regression", "LightGBM"])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(roc[roc.model.isin(models)], x="fpr", y="tpr", color="model",
                      labels={"fpr": "False positive rate", "tpr": "True positive rate"})
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#999"))
        fig.update_layout(title="ROC (test cutoff)", height=430)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.line(pr[pr.model.isin(models)], x="recall", y="precision", color="model",
                      labels={"recall": "Recall", "precision": "Precision"})
        fig.add_hline(y=tr["churn_rate"]["test"], line_dash="dash", line_color="#999", annotation_text="base rate")
        fig.update_layout(title="Precision-recall (test cutoff)", height=430)
        st.plotly_chart(fig, width="stretch")
with tab3:
    rel = csv("curve_reliability.csv")
    pick = st.multiselect("Models ", rel.model.unique().tolist(),
                          default=["Logistic regression", "LightGBM", tr["final_model"]])
    fig = px.line(rel[rel.model.isin(pick)], x="mean_pred", y="observed", color="model", markers=True,
                  labels={"mean_pred": "Mean predicted churn probability", "observed": "Observed churn rate"})
    top = float(max(rel.mean_pred.max(), rel.observed.max()) * 1.1)
    fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(dash="dash", color="#999"))
    fig.update_layout(title="Reliability diagram (test cutoff, quantile bins)", height=460)
    st.plotly_chart(fig, width="stretch")
    cal = tr["calibration"]["LightGBM"]
    cv = cal["cv_brier_on_valid"]
    mt = csv("model_metrics.csv").query("split == 'test' and model == 'LightGBM'").set_index("calibration")
    cil = tr["calibration_in_the_large_test"]
    callout(f"<b>Calibration did not help on test.</b> Platt scaling was selected on validation (5-fold CV Brier: none "
            f"{cv['none']:.4f}, Platt {cv['platt']:.4f}; isotonic {cv['isotonic']:.4f} was evaluated but is not eligible "
            f"with only {cal['valid_positives']} validation positives, because it overfits and emits exact zeros). "
            f"On test the raw LightGBM Brier is {mt.loc['none', 'brier']:.4f}, Platt {mt.loc['platt', 'brier']:.4f}, "
            f"isotonic {mt.loc['isotonic', 'brier']:.4f}. Mean predicted risk: raw "
            f"{cil['LightGBM']['mean_pred']:.1%}, Platt {cil['LightGBM + platt']['mean_pred']:.1%}, observed "
            f"{cil['LightGBM']['observed']:.1%}. The calibration slope of the final model is "
            f"{mt.loc['platt', 'calib_slope']:.2f} (1 is ideal). Use the scores for ranking; treat absolute "
            "probabilities as approximate.")
with tab4:
    lt = csv("lift_tables.csv")
    fig = px.line(lt, x="decile", y="cum_gain", color="model", markers=True,
                  labels={"decile": "Decile (1 = highest predicted risk)", "cum_gain": "Cumulative share of churners"})
    fig.update_layout(title="Cumulative gains (test cutoff)", height=420, yaxis_tickformat=".0%", xaxis_dtick=1)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(lt[lt.model == tr["final_model"]].drop(columns="model").round(3), hide_index=True, width="stretch")
with tab5:
    bt = csv("backtest_metrics.csv")
    met = st.segmented_control("Metric", ["roc_auc", "pr_auc", "brier"], default="roc_auc")
    fig = px.line(bt, x="eval_cutoff", y=met or "roc_auc", color="model", markers=True,
                  labels={"eval_cutoff": "Evaluation cutoff"})
    fig.update_layout(title="Rolling-origin backtest (expanding window, labels closed before each cutoff)", height=420)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(bt[["eval_cutoff", "n_train_snapshots", "model", "n_eval_positives", "roc_auc", "pr_auc", "brier"]]
                 .round(3), hide_index=True, width="stretch")
    piv = bt.pivot(index="eval_cutoff", columns="model", values="roc_auc")
    wins = int((piv["LightGBM"] >= piv["Logistic regression"]).sum())
    st.caption(f"LightGBM matches or beats logistic regression in ROC-AUC at {wins} of {len(piv)} evaluation cutoffs, "
               "by small margins; each cutoff has roughly 100 positives.")

st.subheader("What drives risk? (SHAP)")
c1, c2 = st.columns(2)
for col, (fn, title) in zip([c1, c2], [("shap_importance.csv", "LightGBM (TreeSHAP)"),
                                        ("shap_importance_logreg.csv", "Logistic regression (linear SHAP)")]):
    imp = csv(fn).head(12).iloc[::-1].copy()
    imp["label"] = imp.feature.map(FEATURE_LABELS)
    fig = px.bar(imp, x="mean_abs_shap", y="label", orientation="h",
                 labels={"mean_abs_shap": "Mean |SHAP| (log-odds)", "label": ""})
    fig.update_traces(marker_color=NAVY)
    fig.update_layout(title=f"Global importance: {title}", height=430)
    col.plotly_chart(fig, width="stretch")
st.caption("Features are rates, shares and fixed windows (at most 42 days, campaigns 90 days) so that they do not "
           "simply encode how much history a household has. Trip-frequency features overlap, so their SHAP importance is "
           "shared between them. SHAP describes the model, not causes.")

st.subheader("Household lookup: local explanation")
hh = households()
sv_choice = st.radio("Explanation model", ["Logistic regression", "LightGBM"], horizontal=True)
sv = shap_values("logreg" if sv_choice.startswith("Logistic") else "lgbm")
c1, c2 = st.columns([1, 2])
with c1:
    order = st.selectbox("Suggest households by", ["Highest churn risk", "Highest CLV at high risk", "Highest CLV"])
    if order == "Highest churn risk":
        pool = hh.sort_values("churn_prob", ascending=False)
    elif order == "Highest CLV":
        pool = hh.sort_values("clv_12m", ascending=False)
    else:
        pool = hh[hh.risk_segment == "High risk"].sort_values("clv_12m", ascending=False)
    hid = st.selectbox("Household ID", pool.household_id.head(200).tolist())
    typed = st.text_input("or type a household ID", "")
    if typed.strip() in set(hh.household_id):
        hid = typed.strip()
    elif typed.strip():
        st.warning(f"ID not in the scored population (shopped in the 90 days before {ch['test_cutoff']}).")
    r = hh.set_index("household_id").loc[hid]
    st.metric("Calibrated churn probability", f"{r.churn_prob:.1%}", r.risk_segment, delta_color="off")
    st.metric("Predicted 12-month CLV", usd(float(r.clv_12m)), r.value_segment, delta_color="off")
    st.metric("Observed next 90 days", "No purchase" if r.churned == 1 else f"{int(r.future_trips)} shopping days",
              usd(float(r.future_spend)) + " spend", delta_color="off")
with c2:
    row = sv[sv.household_id == hid].iloc[0]
    feats = [f for f in FEATURE_LABELS if f in sv.columns]
    contrib = pd.Series({f: float(row[f]) for f in feats})
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)
    rest = contrib.drop(top.index).sum()
    vals = list(top.values) + [rest]
    names = [f"{FEATURE_LABELS[f]} = {r[f]:,.2f}" for f in top.index] + ["All other features"]
    base = float(row.base_value)
    fig = go.Figure(go.Waterfall(orientation="h", y=["Average household (base)"] + names + ["This household"],
                                 x=[base] + vals + [0], measure=["absolute"] + ["relative"] * len(vals) + ["total"],
                                 increasing=dict(marker_color=RED), decreasing=dict(marker_color=GREEN),
                                 totals=dict(marker_color=NAVY),
                                 text=[f"{base:+.2f}"] + [f"{v:+.2f}" for v in vals] + [f"{base + sum(vals):+.2f}"]))
    fig.update_layout(title=f"Why is household {hid} scored this way? (log-odds; red pushes toward churn)",
                      height=520, yaxis_autorange="reversed", showlegend=False)
    st.plotly_chart(fig, width="stretch")
    p_expl = 1 / (1 + np.exp(-(base + sum(vals))))
    st.caption(f"{sv_choice} probability for this household: {p_expl:.1%}. Final model (LightGBM, Platt-calibrated): "
               f"{r.churn_prob:.1%}. SHAP values are precomputed by the pipeline.")
