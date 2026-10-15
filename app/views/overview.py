import plotly.express as px
import streamlit as st
from core import (
    RISK_ORDER,
    TITLE,
    VALUE_ORDER,
    callout,
    csv,
    final_row,
    households,
    setup_page,
    summary,
    test_metrics,
    usd,
)

s = summary()
sq, tr, cl, tg = s["sql_summary"], s["train_summary"], s["clv_summary"], s["targeting_summary"]
setup_page("Which households deserve retention spend, and what are they worth?",
           f"{TITLE}. Out-of-time back-test at the {s['config']['churn']['test_cutoff']} scoring date; "
           "every number is computed by the pipeline.")

callout("<b>Business question.</b> A US grocery chain has a loyalty card but no subscriptions: households never "
        "cancel, they simply stop shopping. Which households are likely to be inactive for the next 90 days, how much "
        "is their future spend worth, and where could retention budget (coupons, outreach) pay back? "
        "<b>Churn here is a proxy</b> (no purchase in the 90 days after the cutoff), not an observed cancellation.")
st.write("")

m = test_metrics().set_index(["model", "calibration"])
fm = final_row()
rfm = m.loc[("RFM heuristic", "none")]
hh = households()
c = st.columns(4)
c[0].metric("Households scored", f"{tr['n_test']:,}", help="Shopped in the 90 days before the scoring date")
c[1].metric("90-day churn-proxy rate", f"{tr['churn_rate']['test']:.1%}",
            f"{tr['n_test_positives']} households", delta_color="off",
            help="Observed share with no purchase in the 90 days after the scoring date")
c[2].metric("Model ROC-AUC (test)", f"{fm.roc_auc:.3f}", f"{fm.roc_auc - rfm.roc_auc:+.3f} vs RFM baseline")
c[3].metric("Top-decile lift", f"{fm.lift_top10:.2f}x", help="Churn rate in the riskiest 10% vs base rate")
c = st.columns(4)
c[0].metric("Clean spend (2017)", usd(sq["spend_clean_usd"]), help=f"{sq['date_min']} to {sq['date_max']}")
c[1].metric("CLV holdout: shopping days", f"{cl['pred_purchases_total']:,.0f}",
            f"{cl['pred_purchases_total'] / cl['actual_purchases_total'] - 1:+.1%} vs {cl['actual_purchases_total']:,.0f} actual",
            delta_color="off")
c[2].metric("Predicted 12-month CLV", usd(float(hh.clv_12m.sum())),
            help="BG/NBD + Gamma-Gamma, discounted; sum over the scored households")
km = csv("surv_km_overall.csv").set_index("time")
c[3].metric("Lapse-free at day 90", f"{km.loc[90, 'survival']:.1%}",
            help="Kaplan-Meier from the R survival analysis (landmark design)")

st.subheader("Where the value sits")
mat = csv("targeting_matrix.csv")
left, right = st.columns([3, 2])
with left:
    kw = dict(index="value_segment", columns="risk_segment")
    piv = mat.pivot(values="total_clv_12m", **kw).reindex(index=VALUE_ORDER, columns=RISK_ORDER)
    cnt = mat.pivot(values="households", **kw).reindex(index=VALUE_ORDER, columns=RISK_ORDER)
    obs = mat.pivot(values="observed_churn_rate", **kw).reindex(index=VALUE_ORDER, columns=RISK_ORDER)
    text = [[f"{int(cnt.iloc[i, j]):,} households<br>{usd(piv.iloc[i, j])} CLV<br>observed churn {obs.iloc[i, j]:.1%}"
             for j in range(3)] for i in range(3)]
    fig = px.imshow(piv.values / 1e3, x=RISK_ORDER, y=VALUE_ORDER, color_continuous_scale="Blues",
                    labels=dict(color="CLV $K"), aspect="auto")
    fig.update_traces(text=text, texttemplate="%{text}", textfont=dict(size=12),
                      hovertemplate="%{y} / %{x}<br>%{text}<extra></extra>")
    fig.update_layout(title="Value x risk matrix (predicted 12-month CLV)", height=420,
                      xaxis_title="Churn-proxy risk tier (terciles)", yaxis_title="CLV tier (terciles)")
    st.plotly_chart(fig, width="stretch")
with right:
    hv = mat[(mat.value_segment == "High value") & (mat.risk_segment == "Low risk")].iloc[0]
    hr = mat[(mat.value_segment == "High value") & (mat.risk_segment != "Low risk")]
    lv = mat[(mat.value_segment == "Low value") & (mat.risk_segment == "High risk")].iloc[0]
    st.markdown("#### What the matrix says")
    st.markdown(f"""
1. **Protect** the {int(hv.households):,} High value / Low risk households: {hv.total_clv_12m / mat.total_clv_12m.sum():.0%}
   of predicted CLV and {hv.observed_spend_90d / mat.observed_spend_90d.sum():.0%} of next-quarter spend, with an observed
   churn-proxy rate of {hv.observed_churn_rate:.1%}. Service, not discounts.
2. **High value and at-risk is nearly empty** ({int(hr.households.sum()):,} households): high-spend households
   rarely go quiet, so there is little to rescue at the top.
3. **Most risk sits in Low value households** ({int(lv.households):,} High risk, observed churn
   {lv.observed_churn_rate:.1%}), but their small CLV means the ASSUMED ROI is thin. See the simulator.
""")
    st.caption("Observed churn per cell is the realised 90-day outcome after the back-test scoring date. "
               "ROI inputs are ASSUMED; there is no experiment in this data.")

st.subheader("Model vs baselines (out-of-time test)")
show = test_metrics().query("calibration == 'none'")[["model", "roc_auc", "pr_auc"]]
fig = px.bar(show.melt(id_vars="model", value_vars=["roc_auc", "pr_auc"]), x="model", y="value", color="variable",
             barmode="group", text_auto=".3f", labels={"value": "", "model": "", "variable": ""})
fig.update_layout(height=340, title="ROC-AUC and PR-AUC at the test cutoff")
st.plotly_chart(fig, width="stretch")
pb = tr["paired_bootstrap_test"]["lightgbm_minus_logreg_roc_auc"]
pr_ = tr["paired_bootstrap_test"]["logreg_minus_rfm_roc_auc"]
st.caption(f"Positives are rare ({tr['n_test_positives']} in test), so the two ML models are not distinguishable: "
           f"paired-bootstrap ROC-AUC difference, LightGBM minus logistic regression, {pb['diff']:+.3f} "
           f"(95% interval {pb['lo']:+.3f} to {pb['hi']:+.3f}). Logistic regression beats the RFM heuristic by "
           f"{pr_['diff']:+.3f} ({pr_['lo']:+.3f} to {pr_['hi']:+.3f}). See Churn model.")
