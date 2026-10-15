import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import NAVY, RED, callout, csv, households, setup_page, summary, usd

s = summary()
cl = s["clv_summary"]
sh = cl["short_holdout_13w"]
setup_page("Household lifetime value", "BG/NBD (shopping-day frequency) + Gamma-Gamma (spend per shopping day) via `lifetimes` | "
           f"calibration to {s['config']['clv']['calibration_end']}, {cl['holdout_days']:.0f}-day holdout")

c = st.columns(4)
c[0].metric("Holdout shopping days: predicted", f"{cl['pred_purchases_total']:,.0f}",
            f"{cl['pred_purchases_total'] / cl['actual_purchases_total'] - 1:+.1%} vs actual "
            f"{cl['actual_purchases_total']:,.0f}", delta_color="off")
c[1].metric("Per-household correlation", f"{cl['purchases_corr']:.2f}", f"MAE {cl['purchases_mae']:.2f} shopping days",
            delta_color="off")
c[2].metric("Holdout spend: predicted", usd(cl["pred_spend_total"]),
            f"{cl['pred_spend_total'] / cl['actual_spend_total'] - 1:+.1%} vs {usd(cl['actual_spend_total'])}",
            delta_color="off")
c[3].metric("Top 10% share of actual spend", f"{cl['top10_share_of_actual_spend']:.0%}",
            help="Share of actual holdout spend from the 10% of households with the highest predicted spend")

c1, c2 = st.columns(2)
with c1:
    f = csv("clv_holdout_by_freq.csv")
    fig = go.Figure([go.Scatter(x=f.fbin, y=f.actual, mode="lines+markers", name="Actual", line_color=NAVY),
                     go.Scatter(x=f.fbin, y=f.predicted, mode="lines+markers", name="BG/NBD predicted",
                                line=dict(color=RED, dash="dash"))])
    fig.update_layout(title="Holdout shopping days by calibration repeat count", height=420,
                      xaxis_title="Repeat shopping days in calibration (5-day bins, 60 = 60+)",
                      yaxis_title="Mean shopping days in holdout")
    st.plotly_chart(fig, width="stretch")
with c2:
    d = csv("clv_holdout_by_decile.csv")
    fig = go.Figure([go.Bar(x=d.decile, y=d.actual_spend, name="Actual", marker_color=NAVY),
                     go.Bar(x=d.decile, y=d.predicted_spend, name="Predicted", marker_color=RED)])
    fig.update_layout(title="Holdout spend by predicted-spend decile", barmode="group", height=420,
                      xaxis_title="Predicted decile (10 = highest)", yaxis_title="Mean holdout spend (USD)",
                      xaxis_dtick=1)
    st.plotly_chart(fig, width="stretch")
callout(f"<b>Totals are close; household-level accuracy is only modestly better than a naive rate.</b> Predicted "
        f"holdout shopping days are within {abs(cl['pred_purchases_total'] / cl['actual_purchases_total'] - 1):.1%} of "
        f"actual and spend within {abs(cl['pred_spend_total'] / cl['actual_spend_total'] - 1):.1%}. But the per-household "
        f"MAE of BG/NBD ({cl['purchases_mae']:.2f}) is only slightly below carrying each household's own calibration-"
        f"period rate forward ({cl['naive_rate_mae']:.2f}); for the 13-week holdout it is {sh['purchases_mae']:.2f} vs "
        f"{sh['naive_rate_mae']:.2f}. Ranking quality (Spearman {cl['spend_spearman']:.2f} for spend) is what targeting "
        "relies on.")

st.subheader("Model diagnostics")
b, g = cl["bgf_params"], cl["ggf_params"]
k = st.columns(4)
k[0].metric("BG/NBD a + b", f"{b['a'] + b['b']:.2f}", help="a + b < 1 gives a heavy-tailed dropout distribution")
k[1].metric("Gamma-Gamma q", f"{g['q']:.2f}",
            help="q close to 2 implies a finite mean but heavy-tailed spend")
k[2].metric("Corr(frequency, spend)", f"{cl['gg_freq_spend_corr']:+.3f}",
            help="Gamma-Gamma assumes spend per trip is independent of frequency")
k[3].metric("13-week holdout: shopping days", f"{sh['pred_purchases_total']:,.0f}",
            f"{sh['pred_purchases_total'] / sh['actual_purchases_total'] - 1:+.1%} vs {sh['actual_purchases_total']:,.0f}",
            delta_color="off")

st.subheader(f"CLV at the scoring date ({cl['scoring_as_of']})")
hh = households()
c1, c2 = st.columns([3, 2])
with c1:
    log = st.toggle("Log scale", True)
    x = np.log10(hh.clv_12m.clip(lower=1)) if log else hh.clv_12m
    fig = px.histogram(x=x, nbins=60, labels={"x": "log10(12-month CLV, USD)" if log else "12-month CLV (USD)"},
                       color_discrete_sequence=[NAVY])
    fig.update_layout(title="Distribution of predicted 12-month CLV (scored households)", height=400,
                      yaxis_title="Households", showlegend=False)
    st.plotly_chart(fig, width="stretch")
with c2:
    srt = np.sort(hh.clv_12m.to_numpy())[::-1]
    share = np.cumsum(srt) / srt.sum()
    pct = np.arange(1, len(srt) + 1) / len(srt)
    fig = go.Figure(go.Scatter(x=pct, y=share, mode="lines", line_color=NAVY, fill="tozeroy"))
    fig.update_layout(title="Concentration of predicted value", height=400, xaxis_tickformat=".0%",
                      yaxis_tickformat=".0%", xaxis_title="Share of households (highest CLV first)",
                      yaxis_title="Share of predicted CLV")
    st.plotly_chart(fig, width="stretch")
k = st.columns(4)
k[0].metric("Median CLV (scored households)", usd(float(hh.clv_12m.median())))
k[1].metric("Mean P(alive)", f"{hh.p_alive.mean():.1%}")
k[2].metric("Top 20% share of CLV", f"{share[int(len(srt) * 0.2) - 1]:.0%}")
rho = hh[["clv_12m", "future_spend"]].corr("spearman").iloc[0, 1]
k[3].metric("Spearman vs actual next-90d spend", f"{rho:.2f}")

st.subheader("Fitted parameters")
c1, c2 = st.columns(2)
c1.markdown("**BG/NBD** (scoring fit)")
c1.json(cl["scoring_params"]["bgf"])
c2.markdown("**Gamma-Gamma** (scoring fit)")
c2.json(cl["scoring_params"]["ggf"])
st.caption("CLV = expected shopping days per month x expected spend per shopping day, discounted at "
           f"{s['config']['clv']['discount_rate_monthly']:.0%} per month over {s['config']['clv']['horizon_months']} months. "
           "With one year of data the 12-month horizon is an extrapolation; mean P(alive) is high because the scored "
           "households all shopped in the last 90 days.")
