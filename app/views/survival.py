import numpy as np
import plotly.graph_objects as go
import streamlit as st
from core import COLORWAY, NAVY, RED, callout, csv, setup_page, summary

s = summary()
ex = s["sql_summary"]["survival_exclusions"]
setup_page("Survival analysis (R)", "Time to first 90-day lapse | landmark design | Kaplan-Meier, log-rank, Cox PH "
           "with cox.zph | computed in R (survival, survminer) and rendered in Quarto")

km = csv("surv_km_curves.csv")
ov = csv("surv_km_overall.csv").set_index("time")
lr = csv("surv_logrank_tests.csv").set_index("variable")
c = st.columns(4)
c[0].metric("Households analysed", f"{ex['analysed households']:,}", f"{ex['lapse events']} lapse events", delta_color="off")
c[1].metric("Lapse-free at day 30", f"{ov.loc[30, 'survival']:.1%}")
c[2].metric("Lapse-free at day 90", f"{ov.loc[90, 'survival']:.1%}")
c[3].metric("Lapse-free at day 240", f"{ov.loc[240, 'survival']:.1%}")
st.caption("Time zero is the first observed 2017 purchase plus a 28-day baseline window (the landmark). It is not "
           "acquisition: many households shopped before 2017. A lapse is the first gap over 90 days; follow-up is "
           "censored 90 days before the data end. "
           f"Excluded and counted: {ex['excluded: lapsed before the landmark']} households that lapsed before the landmark, "
           f"{ex['excluded: landmark after data_end - lapse_days (no time at risk)']} with no time at risk.")

names = {"All": "All households", "trip_tier": "Baseline trips (tertiles)", "basket_tier": "Baseline basket size (quartiles)",
         "coupon_user": "Coupon use in baseline window", "entry_group": "Entry month group"}
seg = st.segmented_control("Kaplan-Meier curves by", list(names), format_func=names.get, default="trip_tier")
seg = seg or "trip_tier"
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
fig.update_layout(title=f"P(no 90-day lapse yet): {names[seg]}", height=480,
                  xaxis_title="Days since landmark", yaxis_title="Survival probability", yaxis_tickformat=".0%",
                  xaxis_range=[0, 250])
st.plotly_chart(fig, width="stretch")
if seg in lr.index:
    t = lr.loc[seg]
    st.markdown(f"**Log-rank test:** chi-square = {t.chisq:.1f}, df = {int(t.df)}, p = {t.p_value:.2g}")
if seg == "trip_tier":
    st.caption("Baseline-trip tiers use value-based cut points (ties stay together), so tier sizes are unequal.")
if seg == "entry_group":
    st.caption("Entry group is partly a data-window artifact: households first seen after January are lighter shoppers. "
               "It is used as an adjustment variable, not read as a cohort effect.")

st.subheader("Cox proportional hazards: hazard ratios")
which = st.radio("Model", ["Stratified by entry group", "Unstratified"], horizontal=True)
hr = csv("surv_cox_stratified_hazard_ratios.csv" if which.startswith("Strat") else "surv_cox_hazard_ratios.csv")
pretty = {"log_trips": "log(1 + baseline trips)", "log_basket": "log(baseline basket spend)",
          "fresh_10pts": "Fresh share (per 10 points)", "coupon_userUsed coupons": "Used coupons (vs none)",
          "store_pattern2+ stores": "2+ stores (vs 1)", "entry_groupFeb-Mar 2017": "Entered Feb-Mar (vs Jan)",
          "entry_groupApr-Sep 2017": "Entered Apr-Sep (vs Jan)"}
hr["label"] = hr.term.map(lambda x: pretty.get(x, x))
hr = hr.iloc[::-1]
fig = go.Figure(go.Scatter(x=hr.hr, y=hr.label, mode="markers",
                           marker=dict(size=11, color=[RED if v > 1 else NAVY for v in hr.hr]),
                           error_x=dict(type="data", symmetric=False, array=hr.upper95 - hr.hr,
                                        arrayminus=hr.hr - hr.lower95, color="#7f8c99"),
                           customdata=np.c_[hr.lower95, hr.upper95, hr.p_value],
                           hovertemplate="HR %{x:.3f} [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                                         "<br>p = %{customdata[2]:.3g}<extra></extra>"))
fig.add_vline(x=1, line_dash="dash", line_color="#555")
fig.update_layout(title="Hazard ratio of lapsing (95% CI; below 1 = stays active longer)", height=420,
                  xaxis_type="log", xaxis_title="Hazard ratio (log scale)")
st.plotly_chart(fig, width="stretch")
st.caption("Associations only: coupon use, basket size and store pattern are not randomised, and baseline trip "
           "frequency is the dominant predictor of a 90-day gap by construction.")

st.subheader("Proportional-hazards check (cox.zph)")
c1, c2 = st.columns(2)
z = csv("surv_cox_zph.csv")
zs = csv("surv_cox_stratified_zph.csv")
c1.markdown("**Unstratified model**")
c1.dataframe(z.round(4), hide_index=True, width="stretch")
c2.markdown("**Stratified by entry group**")
c2.dataframe(zs.round(4), hide_index=True, width="stretch")
zq, zsq = z.set_index("term"), zs.set_index("term")
callout(f"Global proportional-hazards test: p = {zq.loc['GLOBAL', 'p_value']:.4f} (unstratified), "
        f"p = {zsq.loc['GLOBAL', 'p_value']:.3f} (stratified). Baseline trip frequency still violates proportional "
        f"hazards after stratifying (p = {zsq.loc['log_trips', 'p_value']:.4f}), so its hazard ratio is an average over "
        "follow-up, not a constant effect. Covariates are measured in the 28-day baseline window to avoid immortal-time bias.")
