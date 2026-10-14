import numpy as np
import plotly.graph_objects as go
import streamlit as st
from core import COLORWAY, NAVY, RED, callout, csv, setup_page

setup_page("Survival analysis (R)", "Time to first 90-day lapse after acquisition · Kaplan-Meier, log-rank, Cox PH "
           "with cox.zph · computed in R (survival, survminer) and rendered in Quarto")

km = csv("surv_km_curves.csv")
ov = csv("surv_km_overall.csv").set_index("time")
lr = csv("surv_logrank_tests.csv").set_index("variable")
allc = km[km.segment_var == "All"]
c = st.columns(4)
c[0].metric("Customers (acquired 2010-01+)", f"{int(allc.n_risk.iloc[0]):,}")
c[1].metric("Return within 90 days of 1st order", f"{ov.loc[1, 'survival']:.1%}")
c[2].metric("Lapse-free at day 90", f"{ov.loc[90, 'survival']:.1%}")
c[3].metric("Lapse-free at day 365", f"{ov.loc[365, 'survival']:.1%}")

names = {"All": "All customers", "first_value_tier": "First-order value tier", "country_group": "Country group",
         "acq_quarter": "Acquisition quarter"}
seg = st.segmented_control("Kaplan-Meier curves by", list(names), format_func=names.get, default="first_value_tier")
seg = seg or "first_value_tier"
d = km[km.segment_var == seg]
show_ci = st.toggle("Show 95% confidence bands", True)
fig = go.Figure()
for i, (stratum, g) in enumerate(d.groupby("stratum", sort=True)):
    col = COLORWAY[i % len(COLORWAY)]
    t = np.r_[0, g.time.to_numpy()]
    fig.add_trace(go.Scatter(x=t, y=np.r_[1, g.surv.to_numpy()], mode="lines", line_shape="hv",
                             name=f"{stratum} (n={int(g.n_risk.iloc[0]):,})", line=dict(color=col, width=2.5)))
    if show_ci:
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[1, g.upper, g.lower[::-1], 1], fill="toself",
                                 line=dict(width=0), line_shape="hv", fillcolor=col, opacity=0.15,
                                 showlegend=False, hoverinfo="skip"))
fig.update_layout(title=f"P(no 90-day lapse yet) - {names[seg]}", height=480, xaxis_title="Days since first purchase",
                  yaxis_title="Survival probability", yaxis_tickformat=".0%", xaxis_range=[0, 600])
st.plotly_chart(fig, width="stretch")
if seg in lr.index:
    t = lr.loc[seg]
    st.markdown(f"**Log-rank test:** χ² = {t.chisq:.1f}, df = {int(t.df)}, p = {t.p_value:.2g}")
if seg == "country_group":
    st.caption("EIRE and France have few customers once the left-truncated Dec-2009 cohort is excluded; read "
               "their curves with caution.")

st.subheader("Cox proportional hazards - hazard ratios")
which = st.radio("Model", ["Stratified by acquisition quarter (PH satisfied)", "Unstratified"], horizontal=True)
hr = csv("surv_cox_stratified_hazard_ratios.csv" if which.startswith("Strat") else "surv_cox_hazard_ratios.csv")
pretty = {"log_first_value": "log(1st order value)", "log_first_products": "log(1st order # products)"}
hr["label"] = hr.term.map(lambda x: pretty.get(x, x.replace("country_group", "Country: ")
                                                  .replace("acq_quarter", "Acquired in ")))
hr = hr.iloc[::-1]
fig = go.Figure(go.Scatter(x=hr.hr, y=hr.label, mode="markers",
                           marker=dict(size=11, color=[RED if v > 1 else NAVY for v in hr.hr]),
                           error_x=dict(type="data", symmetric=False, array=hr.upper95 - hr.hr,
                                        arrayminus=hr.hr - hr.lower95, color="#7f8c99"),
                           customdata=np.c_[hr.lower95, hr.upper95, hr.p_value],
                           hovertemplate="HR %{x:.3f} [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                                         "<br>p = %{customdata[2]:.3g}<extra></extra>"))
fig.add_vline(x=1, line_dash="dash", line_color="#555")
fig.update_layout(title="Hazard ratio of lapsing (95% CI; < 1 = stays active longer)", height=420,
                  xaxis_type="log", xaxis_title="Hazard ratio (log scale)")
st.plotly_chart(fig, width="stretch")

st.subheader("Proportional-hazards check (cox.zph)")
c1, c2 = st.columns(2)
z = csv("surv_cox_zph.csv")
zs = csv("surv_cox_stratified_zph.csv")
c1.markdown("**Unstratified model**")
c1.dataframe(z.round(4), hide_index=True, width="stretch")
c2.markdown("**Stratified by acquisition quarter**")
c2.dataframe(zs.round(4), hide_index=True, width="stretch")
zq = z.set_index("term")
callout(f"Acquisition quarter violates proportional hazards (p = {zq.loc['acq_quarter', 'p_value']:.3f}; global "
        f"p = {zq.loc['GLOBAL', 'p_value']:.3f}). Stratifying on it gives a separate baseline hazard per quarter and "
        f"the global test passes (p = {zs.set_index('term').loc['GLOBAL', 'p_value']:.2f}), with first-order value "
        "effects essentially unchanged. Covariates are fixed at acquisition to avoid immortal-time bias.")
