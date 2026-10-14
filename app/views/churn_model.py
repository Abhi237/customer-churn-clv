import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import FEATURE_LABELS, GREEN, NAVY, RED, callout, csv, customers, gbp, setup_page, shap_values, summary

s = summary()
tr = s["train_summary"]
setup_page("Churn-proxy model", "Churn = no purchase in the 90 days after the cutoff (a behavioural proxy, not an "
           "observed cancellation) · train 4 cutoffs → validate 2011-06-01 → test 2011-09-01")

c = st.columns(4)
c[0].metric("Training snapshots", f"{tr['n_train_rows']:,}", "4 cutoffs", delta_color="off")
c[1].metric("Validation / test customers", f"{tr['n_valid']:,} / {tr['n_test']:,}")
c[2].metric("Churn rate (train → valid → test)",
            f"{tr['churn_rate']['test']:.1%}",
            f"train {tr['churn_rate']['train']:.1%} · valid {tr['churn_rate']['valid']:.1%}", delta_color="off")
c[3].metric("Selected model", tr["final_model"].replace("Logistic regression", "LogReg"),
            "chosen on validation PR-AUC", delta_color="off")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Metrics", "📈 ROC / PR", "🎯 Calibration", "🏆 Lift & gains",
                                        "🔁 Backtest"])
with tab1:
    m = csv("model_metrics.csv")
    split = st.segmented_control("Split", ["test", "valid"], default="test")
    show = m[m.split == (split or "test")].drop(columns=["split", "log_loss"])
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
        "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
        "brier": st.column_config.NumberColumn("Brier ↓", format="%.3f"),
        "ece": st.column_config.NumberColumn("ECE ↓", format="%.3f"),
        "lift_top10": st.column_config.NumberColumn("Lift top 10%", format="%.2fx"),
        "lift_top20": st.column_config.NumberColumn("Cum. lift top 20%", format="%.2fx"),
        "gain_top20": st.column_config.NumberColumn("Churners in top 20%", format="%.3f"),
        "gain_top30": st.column_config.NumberColumn("Churners in top 30%", format="%.3f"),
        "base_rate": st.column_config.NumberColumn("Base rate", format="%.3f")})
    st.caption("Baselines: recency rule (churn rate by recency bin) and RFM quintile score. Calibration is fit on "
               "the validation snapshot and evaluated on test.")
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
                          default=["Logistic regression", tr["final_model"], "LightGBM"])
    fig = px.line(rel[rel.model.isin(pick)], x="mean_pred", y="observed", color="model", markers=True,
                  labels={"mean_pred": "Mean predicted churn probability", "observed": "Observed churn rate"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#999"))
    fig.update_layout(title="Reliability diagram (test cutoff, quantile bins)", height=460,
                      xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")
    cil = tr["calibration_in_the_large_test"][tr["final_model"]]
    cv = tr["calibration"][tr["final_model"].split(" + ")[0]]["cv_brier_on_valid"]
    callout(f"<b>Calibration drift.</b> Platt was chosen on validation (CV Brier {cv['platt']:.4f} vs isotonic "
            f"{cv['isotonic']:.4f}). At the September test cutoff the final model predicts an average risk of "
            f"<b>{cil['mean_pred']:.1%}</b> vs <b>{cil['observed']:.1%}</b> observed: the Sep-Nov label window is "
            "peak season. Use scores to <b>rank</b>; re-calibrate seasonally before reading them as probabilities.")
with tab4:
    lt = csv("lift_tables.csv")
    c1, c2 = st.columns(2)
    with c1:
        g = pd.concat([pd.DataFrame({"model": m, "decile": [0], "cum_gain": [0.0]}) for m in lt.model.unique()]
                      + [lt[["model", "decile", "cum_gain"]]])
        g["pct"] = g.decile * 10
        fig = px.line(g.sort_values(["model", "pct"]), x="pct", y="cum_gain", color="model", markers=True,
                      labels={"pct": "% of customers contacted (riskiest first)", "cum_gain": "% of churners captured"})
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=1, line=dict(dash="dash", color="#999"))
        fig.update_layout(title="Cumulative gains", height=420, yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(lt, x="decile", y="lift", color="model", barmode="group",
                     labels={"decile": "Risk decile (1 = highest)", "lift": "Lift vs base rate"})
        fig.add_hline(y=1, line_color="#555")
        fig.update_layout(title="Lift by decile", height=420, xaxis_dtick=1)
        st.plotly_chart(fig, width="stretch")
with tab5:
    bt = csv("backtest_metrics.csv")
    met = st.segmented_control("Metric", ["roc_auc", "pr_auc", "brier"], default="roc_auc")
    fig = px.line(bt, x="eval_cutoff", y=met or "roc_auc", color="model", markers=True,
                  labels={"eval_cutoff": "Evaluation cutoff"})
    fig.update_layout(title="Rolling-origin backtest (retrained on snapshots whose labels closed before each cutoff)",
                      height=420)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(bt[["eval_cutoff", "model", "train_cutoffs", "n_eval", "roc_auc", "pr_auc", "brier", "base_rate"]]
                 .round(3), hide_index=True, width="stretch")

st.subheader("What drives risk? (SHAP)")
c1, c2 = st.columns(2)
for col, (fn, title) in zip([c1, c2], [("shap_importance.csv", "LightGBM (TreeSHAP)"),
                                        ("shap_importance_logreg.csv", "Logistic regression (linear SHAP)")]):
    imp = csv(fn).head(12).iloc[::-1]
    imp["label"] = imp.feature.map(FEATURE_LABELS)
    fig = px.bar(imp, x="mean_abs_shap", y="label", orientation="h",
                 labels={"mean_abs_shap": "Mean |SHAP| (log-odds)", "label": ""})
    fig.update_traces(marker_color=NAVY)
    fig.update_layout(title=f"Global importance - {title}", height=430)
    col.plotly_chart(fig, width="stretch")
st.caption("Logistic-regression SHAP is computed in the transformed (log + standardised) feature space; "
           "monetary total and frequency are correlated and partly offset each other.")

st.subheader("Customer lookup - local explanation")
cust = customers()
sv_choice = st.radio("Explanation model", ["Logistic regression (scoring model)", "LightGBM"], horizontal=True)
sv = shap_values("logreg" if sv_choice.startswith("Logistic") else "lgbm")
c1, c2 = st.columns([1, 2])
with c1:
    order = st.selectbox("Suggest customers by", ["Highest CLV at high risk", "Highest churn risk", "Highest CLV"])
    if order == "Highest churn risk":
        pool = cust.sort_values("churn_prob", ascending=False)
    elif order == "Highest CLV":
        pool = cust.sort_values("clv_12m", ascending=False)
    else:
        pool = cust[cust.risk_segment == "High risk"].sort_values("clv_12m", ascending=False)
    cid = st.selectbox("Customer ID", pool.customer_id.head(200).tolist())
    typed = st.text_input("…or type a customer ID", "")
    if typed.strip() in set(cust.customer_id):
        cid = typed.strip()
    elif typed.strip():
        st.warning("ID not in the scored population (active in the 365 days before 2011-09-01).")
    r = cust.set_index("customer_id").loc[cid]
    st.metric("Calibrated churn probability", f"{r.churn_prob:.1%}", r.risk_segment, delta_color="off")
    st.metric("Predicted 12-month CLV", gbp(float(r.clv_12m)), r.value_segment, delta_color="off")
    st.metric("Observed next 90 days", "Churned" if r.churned == 1 else f"Purchased ({int(r.future_invoices)}×)",
              gbp(float(r.future_revenue)) + " revenue", delta_color="off")
with c2:
    row = sv[sv.customer_id == cid].iloc[0]
    feats = [f for f in FEATURE_LABELS if f in sv.columns]
    contrib = pd.Series({f: float(row[f]) for f in feats})
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)
    rest = contrib.drop(top.index).sum()
    vals = list(top.values) + [rest]
    names = [f"{FEATURE_LABELS[f]} = {r[f]:,.1f}" for f in top.index] + ["All other features"]
    base = float(row.base_value)
    fig = go.Figure(go.Waterfall(orientation="h", y=["Average customer (base)"] + names + ["This customer"],
                                 x=[base] + vals + [0], measure=["absolute"] + ["relative"] * len(vals) + ["total"],
                                 increasing=dict(marker_color=RED), decreasing=dict(marker_color=GREEN),
                                 totals=dict(marker_color=NAVY),
                                 text=[f"{base:+.2f}"] + [f"{v:+.2f}" for v in vals] + [f"{base + sum(vals):+.2f}"]))
    fig.update_layout(title=f"Why is customer {cid} scored this way? (log-odds; red pushes toward churn)",
                      height=520, yaxis_autorange="reversed", showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Raw model probability {1 / (1 + np.exp(-(base + sum(vals)))):.1%} → after calibration "
               f"{r.churn_prob:.1%}. SHAP values are precomputed by the pipeline.")
