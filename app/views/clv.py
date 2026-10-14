import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import NAVY, RED, callout, csv, customers, gbp, setup_page, summary

s = summary()
cl = s["clv_summary"]
setup_page("Customer lifetime value", "BG/NBD (purchase frequency) + Gamma-Gamma (spend) via `lifetimes` · "
           f"calibration to {s['config']['clv']['calibration_end']}, {cl['holdout_days']:.0f}-day holdout")

c = st.columns(4)
c[0].metric("Holdout purchases: predicted", f"{cl['pred_purchases_total']:,.0f}",
            f"{cl['pred_purchases_total'] / cl['actual_purchases_total'] - 1:+.1%} vs actual "
            f"{cl['actual_purchases_total']:,.0f}", delta_color="off")
c[1].metric("Per-customer correlation", f"{cl['purchases_corr']:.2f}", f"MAE {cl['purchases_mae']:.2f} purchases",
            delta_color="off")
c[2].metric("Holdout spend: predicted", gbp(cl["pred_spend_total"]),
            f"{cl['pred_spend_total'] / cl['actual_spend_total'] - 1:+.1%} vs {gbp(cl['actual_spend_total'])}",
            delta_color="off")
c[3].metric("Top 10% share of actual spend", f"{cl['top10_share_of_actual_spend']:.0%}",
            help="Share of actual holdout spend from the 10% of customers with the highest predicted spend")

c1, c2 = st.columns(2)
with c1:
    f = csv("clv_holdout_by_freq.csv")
    fig = go.Figure([go.Scatter(x=f.fbin, y=f.actual, mode="lines+markers", name="Actual", line_color=NAVY),
                     go.Scatter(x=f.fbin, y=f.predicted, mode="lines+markers", name="BG/NBD predicted",
                                line=dict(color=RED, dash="dash"))])
    fig.update_layout(title="Holdout purchases by calibration repeat count", height=420,
                      xaxis_title="Repeat purchases in calibration (15 = 15+)", yaxis_title="Mean holdout purchases")
    st.plotly_chart(fig, width="stretch")
with c2:
    d = csv("clv_holdout_by_decile.csv")
    fig = go.Figure([go.Bar(x=d.decile, y=d.actual_spend, name="Actual", marker_color=NAVY),
                     go.Bar(x=d.decile, y=d.predicted_spend, name="Predicted", marker_color=RED)])
    fig.update_layout(title="Holdout spend by predicted-spend decile", barmode="group", height=420,
                      xaxis_title="Predicted decile (10 = highest)", yaxis_title="Mean holdout spend (£)",
                      xaxis_dtick=1)
    st.plotly_chart(fig, width="stretch")
callout(f"Purchase counts are forecast almost exactly; spend is under-forecast by "
        f"{1 - cl['pred_spend_total'] / cl['actual_spend_total']:.0%}, concentrated in the top decile. The holdout "
        "contains the Q4 peak and BG/NBD has no seasonality. Ranking quality (Spearman "
        f"{cl['spend_spearman']:.2f}) is what targeting relies on.")

st.subheader(f"CLV at the scoring date ({cl['scoring_as_of']})")
cust = customers()
c1, c2 = st.columns([3, 2])
with c1:
    log = st.toggle("Log scale", True)
    x = np.log10(cust.clv_12m.clip(lower=1)) if log else cust.clv_12m
    fig = px.histogram(x=x, nbins=60, labels={"x": "log10(12-month CLV £)" if log else "12-month CLV (£)"},
                       color_discrete_sequence=[NAVY])
    fig.update_layout(title="Distribution of predicted 12-month CLV (scored customers)", height=400,
                      yaxis_title="Customers", showlegend=False)
    st.plotly_chart(fig, width="stretch")
with c2:
    srt = np.sort(cust.clv_12m.to_numpy())[::-1]
    share = np.cumsum(srt) / srt.sum()
    pct = np.arange(1, len(srt) + 1) / len(srt)
    fig = go.Figure(go.Scatter(x=pct, y=share, mode="lines", line_color=NAVY, fill="tozeroy"))
    fig.update_layout(title="Concentration of predicted value", height=400, xaxis_tickformat=".0%",
                      yaxis_tickformat=".0%", xaxis_title="Share of customers (highest CLV first)",
                      yaxis_title="Share of predicted CLV")
    st.plotly_chart(fig, width="stretch")
k = st.columns(4)
k[0].metric("Median CLV (scored population)", gbp(float(cust.clv_12m.median())))
k[1].metric("Mean P(alive)", f"{cust.p_alive.mean():.1%}")
k[2].metric("Top 20% share of CLV", f"{share[int(len(srt) * 0.2) - 1]:.0%}")
rho = cust[["clv_12m", "future_revenue"]].corr("spearman").iloc[0, 1]
k[3].metric("Spearman vs actual next-90d revenue", f"{rho:.2f}")

st.subheader("Fitted parameters")
c1, c2 = st.columns(2)
c1.markdown("**BG/NBD** (scoring fit)")
c1.json(cl["scoring_params"]["bgf"])
c2.markdown("**Gamma-Gamma** (scoring fit)")
c2.json(cl["scoring_params"]["ggf"])
st.caption(f"Gamma-Gamma q = {cl['scoring_params']['ggf']['q']:.2f} < 1 means very heavy-tailed spend, so the "
           "population-mean order value is undefined; zero-repeat customers get the pooled empirical repeat order "
           "value instead. CLV = monthly expected purchases × expected order value, discounted at "
           f"{s['config']['clv']['discount_rate_monthly']:.0%}/month over {s['config']['clv']['horizon_months']} months.")
