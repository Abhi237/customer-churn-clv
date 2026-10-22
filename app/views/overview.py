import plotly.express as px
import streamlit as st
from core import (
    PERSONA,
    QUESTION,
    RISK_ORDER,
    TITLE_METHOD,
    TITLE_PREFIX,
    VALUE_ORDER,
    callout,
    csv,
    final_row,
    households,
    page_story,
    setup_page,
    summary,
    test_metrics,
    usd,
)

s = summary()
sq, tr, cl, tg = s["sql_summary"], s["train_summary"], s["clv_summary"], s["targeting_summary"]
setup_page(TITLE_PREFIX, tagline=TITLE_METHOD, question=QUESTION)
page_story('Which households deserve retention spend next quarter, and what are they worth?',
           'Start here: the recommendations below are the decisions; every other page is the evidence behind one of them.')

callout("<b>Business question.</b> A US grocery chain has a loyalty card but no subscriptions: households never "
        "cancel, they simply stop shopping. Which households are likely to be inactive for the next 90 days, how much "
        "is their future spend worth, and where could retention budget (coupons, outreach) pay back? "
        "<b>Churn here is a proxy</b> (no purchase in the 90 days after the cutoff), not an observed cancellation.")
st.markdown(f"<p class='small'>Out-of-time back-test at the {s['config']['churn']['test_cutoff']} scoring date; "
            "every number is computed by the pipeline.</p>", unsafe_allow_html=True)

m = test_metrics().set_index(["model", "calibration"])
fm = final_row()
rfm = m.loc[("RFM heuristic", "none")]
hh = households()
tcfg = s["config"]["targeting"]
env = hh.churn_prob * tcfg["uplift"] * hh.clv_12m * tcfg["gross_margin"] - tcfg["contact_cost_usd"]
pos = env[env > 0]
clv_at_risk = float((hh.churn_prob * hh.clv_12m).sum())

st.markdown(f"**Who this is for:** a {PERSONA} deciding where retention budget goes next quarter.")
st.subheader("Headline KPIs")
c = st.columns(4)
c[0].metric("Households scored", f"{tr['n_test']:,}", help="Shopped in the 90 days before the scoring date")
c[1].metric("90-day churn-proxy rate", f"{tr['churn_rate']['test']:.1%}",
            f"{tr['n_test_positives']} households", delta_color="off",
            help="Observed share with no purchase in the 90 days after the scoring date")
c[2].metric("CLV at risk", usd(clv_at_risk), f"{clv_at_risk / float(hh.clv_12m.sum()):.1%} of predicted CLV",
            delta_color="off", help="Sum over scored households of calibrated churn probability x predicted 12-month CLV")
c[3].metric("Net value at defaults (ASSUMED)", usd(float(pos.sum())), f"{len(pos):,} households targeted",
            delta_color="off",
            help=f"Targeting rule at the ASSUMED defaults: uplift {tcfg['uplift']:.0%}, margin {tcfg['gross_margin']:.0%}, "
                 f"contact cost ${tcfg['contact_cost_usd']:.2f}. Scenario, not observed impact.")

mat0 = csv("targeting_matrix.csv")
hv0 = mat0[(mat0.value_segment == "High value") & (mat0.risk_segment == "Low risk")].iloc[0]
pr0 = tr["paired_bootstrap_test"]["logreg_minus_rfm_roc_auc"]
st.subheader("Recommendations")
st.markdown((f"""
1. **Contact only the {len(pos):,} households where expected net value is positive ({len(pos) / len(hh):.0%} of scored).**
   Why: at the ASSUMED defaults they return {usd(float(pos.sum()))} expected net value, while contacting everyone returns
   {usd(float(env.sum()))}.
2. **Protect the {int(hv0.households):,} High value / Low risk households with service, not discounts.**
   Why: they hold {hv0.total_clv_12m / mat0.total_clv_12m.sum():.0%} of predicted 12-month CLV with an observed churn-proxy
   rate of {hv0.observed_churn_rate:.1%}.
3. **Rank households with the model, not the RFM rule of thumb.**
   Why: logistic regression beats the RFM heuristic by {pr0['diff']:+.3f} ROC-AUC (95% interval {pr0['lo']:+.3f} to
   {pr0['hi']:+.3f}) on the out-of-time test.
4. **Run a holdout test before scaling coupons.**
   Why: targeted households differ from untargeted ones before any campaign, so the observed churn gap is not an effect,
   and the uplift above is ASSUMED.
""").replace("$", "\\$"))
lk = st.columns(4)
lk[0].page_link("views/targeting.py", label="1. Targeting simulator", icon=":material/tune:")
lk[1].page_link("views/clv.py", label="2. Household lifetime value", icon=":material/payments:")
lk[2].page_link("views/churn_model.py", label="3. Churn model", icon=":material/model_training:")
lk[3].page_link("views/campaigns.py", label="4. Coupons and campaigns", icon=":material/local_offer:")

st.subheader("Model and data detail")
c = st.columns(6)
c[0].metric("Model ROC-AUC (test)", f"{fm.roc_auc:.3f}", f"{fm.roc_auc - rfm.roc_auc:+.3f} vs RFM baseline")
c[1].metric("Top-decile lift", f"{fm.lift_top10:.2f}x", help="Churn rate in the riskiest 10% vs base rate")
c[2].metric("Clean spend (2017)", usd(sq["spend_clean_usd"]), help=f"{sq['date_min']} to {sq['date_max']}")
c[3].metric("CLV holdout: shopping days", f"{cl['pred_purchases_total']:,.0f}",
            f"{cl['pred_purchases_total'] / cl['actual_purchases_total'] - 1:+.1%} vs {cl['actual_purchases_total']:,.0f} actual",
            delta_color="off")
c[4].metric("Predicted 12-month CLV", usd(float(hh.clv_12m.sum())),
            help="BG/NBD + Gamma-Gamma, discounted; sum over the scored households")
km = csv("surv_km_overall.csv").set_index("time")
c[5].metric("Lapse-free at day 90", f"{km.loc[90, 'survival']:.1%}",
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
