import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import NAVY, RED, TEAL, callout, csv, setup_page, summary

s = summary()
cs = s["campaign_summary"]
sq = s["sql_summary"]
setup_page("Coupons and campaigns", "Campaign exposure, coupon redemption and a targeting analysis | observational "
           "association only, no causal claims")
callout("<b>Read this first.</b> The data lists which households each campaign was sent to and which coupons were "
        "redeemed. It does not say how households were chosen and there is no holdout group, so nothing below "
        "estimates the effect of a campaign. Targeted households are very different from untargeted ones before "
        "any campaign runs (see the balance table).")
st.write("")

c = st.columns(4)
c[0].metric("Campaigns", f"{sq['n_campaigns']}")
c[1].metric("Households targeted by any campaign", f"{sq['n_households_targeted_any']:,}",
            f"{sq['n_households_targeted_any'] / sq['n_households']:.0%} of households", delta_color="off")
c[2].metric("Coupon redemptions", f"{sq['n_redemptions']:,}")
c[3].metric("Households that ever redeemed", f"{sq['n_households_ever_redeemed']:,}",
            f"{sq['n_households_ever_redeemed'] / sq['n_households']:.0%} of households", delta_color="off")

st.subheader("Campaign types")
ct = csv("campaign_type_summary.csv")
st.dataframe(ct.rename(columns={"campaign_type": "Type", "campaigns": "Campaigns",
                                "households_targeted_total": "Household-campaign pairs", "median_targeted": "Median households per campaign",
                                "redemptions": "Redemptions", "redeeming_households_total": "Redeeming household-campaign pairs",
                                "redeeming_per_targeted": "Redeeming share of targeted"}),
             hide_index=True, width="stretch",
             column_config={"Redeeming share of targeted": st.column_config.NumberColumn(format="percent")})
st.caption("Type A campaigns reach far more households (about a thousand each) and show a higher redemption share. "
           "Types are anonymised in the source; the labels carry no further meaning.")
dc = csv("dim_campaigns.csv")
dc["start_date"] = dc["start_date"].astype(str)
fig = px.scatter(dc, x="start_date", y="redemption_rate_households", size="n_households_targeted", color="campaign_type",
                 hover_data=["campaign_id", "n_households_targeted", "n_redemptions"],
                 labels={"start_date": "Campaign start", "redemption_rate_households": "Share of targeted households redeeming"})
fig.update_layout(title="Campaigns over 2017 (bubble size = households targeted)", height=400, yaxis_tickformat=".0%")
st.plotly_chart(fig, width="stretch")

st.subheader("Churn proxy by exposure in the test window (association)")
win = cs["campaign_window"]
OPTIONS = {
    "Targeted by any campaign starting in the window": ("targeted", "targeted_in_window", "adjusted_difference_targeted"),
    "Targeted by a Type A campaign starting in the window": ("type_a", "type_a_in_window", "adjusted_difference_type_a"),
    "Redeemed a coupon before the cutoff": ("prior_redeemer", "prior_redeemer", "adjusted_difference_prior_redeemer"),
}
which = st.radio("Exposure definition", list(OPTIONS), horizontal=False)
key = OPTIONS[which]
out = csv(f"campaign_outcomes_{key[0]}.csv")
flag = key[1]
out["exposed"] = out[flag].astype(str).map({"True": "Exposed", "False": "Not exposed"})
fig = go.Figure()
for lab, col in [("Not exposed", NAVY), ("Exposed", RED)]:
    g = out[out.exposed == lab]
    fig.add_trace(go.Bar(x=g.risk_segment, y=g.churn_rate, name=lab, marker_color=col,
                         error_y=dict(type="data", symmetric=False, array=g.ci_hi - g.churn_rate,
                                      arrayminus=g.churn_rate - g.ci_lo),
                         text=[f"n={int(n)}" for n in g.n], textposition="outside"))
fig.update_layout(barmode="group", height=430, yaxis_tickformat=".0%", yaxis_title="Observed churn-proxy rate (95% Wilson CI)",
                  title=f"Window {win['start']} + {win['horizon_days']} days, by model risk tier")
st.plotly_chart(fig, width="stretch")
adj = cs[key[2]]
n_exposed = int(out[(out[flag].astype(str) == "True") & (out["risk_segment"] == "All scored")]["n"].iloc[0])
k = st.columns(3)
k[0].metric("Naive difference (exposed minus not)", f"{adj['naive_diff'] * 100:+.1f} pts")
k[1].metric("Risk-tier-standardised difference", f"{adj['tier_standardised_diff'] * 100:+.1f} pts",
            help="Within-tier differences averaged with the overall tier mix")
k[2].metric("Households exposed", f"{n_exposed:,}")
st.caption("A large naive gap that nearly disappears within model risk tiers means exposure mostly tracks pre-existing "
           "shopping intensity (campaigns reach heavier shoppers who were never likely to go quiet). It does not show that campaigns work or fail.")

st.subheader("Why the naive gap is not an effect: baseline imbalance")
bal = csv("campaign_balance_targeted.csv")
bal["label"] = bal.feature.map(lambda f: f.replace("_", " "))
fig = px.bar(bal.iloc[::-1], x="smd", y="label", orientation="h", labels={"smd": "Standardised mean difference (targeted minus not)", "label": ""})
fig.update_traces(marker_color=TEAL)
fig.add_vline(x=0.1, line_dash="dot", line_color="#999")
fig.add_vline(x=-0.1, line_dash="dot", line_color="#999")
fig.update_layout(height=420, title=f"Pre-cutoff feature balance (largest |SMD| = {cs['max_abs_smd_targeted']:.2f}; 0.1 is the usual balance threshold)")
st.plotly_chart(fig, width="stretch")

st.subheader("Coupon-discount share and the churn proxy")
cg = csv("campaign_coupon_gradient.csv")
fig = px.bar(cg, x="coupon_share_quartile", y="churn_rate", text=[f"n={int(n)}" for n in cg.n],
             labels={"coupon_share_quartile": "Quartile of coupon-discount share of shelf value (pre-cutoff)",
                     "churn_rate": "Observed churn-proxy rate"})
fig.update_traces(marker_color=NAVY, textposition="outside")
fig.update_layout(height=380, yaxis_tickformat=".0%", title="Households that use more coupons are less likely to go quiet (association)")
st.plotly_chart(fig, width="stretch")
st.caption("Coupon use is a marker of engagement, not necessarily a driver of it. The lowest quartiles include households with no coupon use "
           "at all. No causal reading.")

st.subheader("From association to a decision: what a test would need")
st.markdown("""
* Randomise eligible households to receive or not receive a campaign (or a coupon offer), stratified by risk tier and value tier.
* Measure incremental spend and incremental 90-day activity against the holdout; only then replace the ASSUMED uplift on the
  simulator page with a measured one.
* Until then every ROI number in this project is a scenario under ASSUMED parameters.
""")
