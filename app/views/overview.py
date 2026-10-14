import plotly.express as px
import streamlit as st
from core import RISK_ORDER, VALUE_ORDER, callout, csv, customers, gbp, setup_page, summary, test_metrics

s = summary()
sq, tr, cl, tg = s["sql_summary"], s["train_summary"], s["clv_summary"], s["targeting_summary"]
setup_page("Which customers deserve retention spend - and what are they worth?",
           "Executive summary · out-of-time back-test at the "
           f"{s['config']['churn']['test_cutoff']} scoring date · every number computed by the pipeline")

callout("<b>Business question.</b> A UK online gift wholesaler has no subscriptions: customers never cancel, "
        "they simply stop ordering. Leadership wants to know <b>who is likely to lapse in the next 90 days</b>, "
        "<b>how much their future purchases are worth</b>, and <b>where retention budget earns its keep</b>.")
st.write("")

m = test_metrics().set_index(["model", "calibration"])
final = tr["final_model"]
fm = m.loc[tuple(final.split(" + "))]
rfm = m.loc[("RFM heuristic", "none")]
c = st.columns(4)
c[0].metric("Customers scored", f"{tr['n_test']:,}", help="Active in the 365 days before the scoring date")
c[1].metric("90-day churn-proxy rate", f"{tr['churn_rate']['test']:.1%}",
            help="Observed share with no purchase in the 90 days after the scoring date")
c[2].metric("Model ROC-AUC (test)", f"{fm.roc_auc:.3f}", f"{fm.roc_auc - rfm.roc_auc:+.3f} vs RFM baseline")
c[3].metric("Top-decile lift", f"{fm.lift_top10:.2f}x", help="Churn rate in the riskiest 10% vs base rate")
c = st.columns(4)
c[0].metric("Clean revenue", gbp(sq["revenue_clean_gbp"]), help=f"{sq['date_min']} to {sq['date_max']}")
c[1].metric("CLV holdout: purchases", f"{cl['pred_purchases_total']:,.0f}",
            f"{cl['pred_purchases_total'] / cl['actual_purchases_total'] - 1:+.1%} vs {cl['actual_purchases_total']:,.0f} actual",
            delta_color="off")
c[2].metric("Predicted 12-month CLV", gbp(float(customers().clv_12m.sum())),
            help="BG/NBD + Gamma-Gamma, discounted; sum over the scored (active) customers")
c[3].metric("Repeat within 90 days", f"{csv('surv_km_overall.csv').set_index('time').loc[1, 'survival']:.0%}",
            help="Kaplan-Meier: share of new customers not lapsing after their first order")

st.subheader("Where the value sits")
mat = csv("targeting_matrix.csv")
left, right = st.columns([3, 2])
with left:
    piv = mat.pivot(index="value_segment", columns="risk_segment", values="total_clv_12m").reindex(
        index=VALUE_ORDER, columns=RISK_ORDER)
    cnt = mat.pivot(index="value_segment", columns="risk_segment", values="customers").reindex(
        index=VALUE_ORDER, columns=RISK_ORDER)
    obs = mat.pivot(index="value_segment", columns="risk_segment", values="observed_churn_rate").reindex(
        index=VALUE_ORDER, columns=RISK_ORDER)
    text = [[f"{int(cnt.iloc[i, j]):,} customers<br>{gbp(piv.iloc[i, j])} CLV<br>observed churn {obs.iloc[i, j]:.0%}"
             for j in range(3)] for i in range(3)]
    fig = px.imshow(piv.values / 1e3, x=RISK_ORDER, y=VALUE_ORDER, color_continuous_scale="Blues",
                    labels=dict(color="CLV £k"), aspect="auto")
    fig.update_traces(text=text, texttemplate="%{text}", textfont=dict(size=12), hovertemplate="%{y} / %{x}<br>%{text}<extra></extra>")
    fig.update_layout(title="Value × risk matrix (predicted 12-month CLV)", height=420,
                      xaxis_title="Churn-proxy risk tier", yaxis_title="CLV tier")
    st.plotly_chart(fig, width="stretch")
with right:
    hv = mat[(mat.value_segment == "High value") & (mat.risk_segment == "Low risk")].iloc[0]
    hr = mat[(mat.value_segment == "High value") & (mat.risk_segment != "Low risk")]
    mr = mat[(mat.value_segment == "Mid value") & (mat.risk_segment != "Low risk")]
    st.markdown("#### Key recommendation")
    st.markdown(f"""
1. **Protect** the {int(hv.customers):,} *High value / Low risk* customers - {hv.total_clv_12m / mat.total_clv_12m.sum():.0%}
   of predicted CLV and {hv.observed_revenue_90d / mat.observed_revenue_90d.sum():.0%} of next-quarter revenue.
   Service, not discounts.
2. **Personal outreach** to the {int(hr.customers.sum()):,} high-value customers at medium/high risk.
3. **Automated win-back** for {int(mr.customers.sum()):,} mid-value at-risk customers, **with a control group**
   so uplift is measured rather than assumed.
4. **Fix activation**: only {csv('surv_km_overall.csv').set_index('time').loc[1, 'survival']:.0%} of new
   customers order again within 90 days.
""")
    st.caption("Observed churn per cell is the realised 90-day outcome after the back-test scoring date.")

st.subheader("Model vs baselines (out-of-time test)")
show = test_metrics().query("calibration == 'none'")[["model", "roc_auc", "pr_auc", "brier", "lift_top10",
                                                      "gain_top30"]]
fig = px.bar(show.melt(id_vars="model", value_vars=["roc_auc", "pr_auc"]), x="model", y="value", color="variable",
             barmode="group", text_auto=".3f", labels={"value": "", "model": "", "variable": ""})
fig.update_layout(height=340, yaxis_range=[0.5, 0.85], title="ROC-AUC and PR-AUC at the test cutoff")
st.plotly_chart(fig, width="stretch")
st.caption("Logistic regression is best, but the gain over a simple RFM score is modest - see *Churn model*.")
