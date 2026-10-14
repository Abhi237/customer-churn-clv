import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import NAVY, csv, setup_page, summary

s = summary()
setup_page("Cohorts & lifecycle", "Monthly acquisition cohorts, retention, rolling activity and RFM segments")

coh = csv("cohort_retention.csv")
coh["cohort"] = pd.to_datetime(coh["acquisition_cohort"]).dt.strftime("%Y-%m")
c1, c2 = st.columns([1, 3])
with c1:
    max_m = st.slider("Months since acquisition", 3, 24, 12)
    metric = st.radio("Show", ["Retention rate", "Active customers"], horizontal=False)
    min_size = st.number_input("Minimum cohort size", 0, 1000, 20, 10)
    m1 = coh[coh.months_since_acquisition == 1].retention_rate.mean()
    st.metric("Mean month-1 retention", f"{m1:.1%}")
    st.metric("One-time buyers", f"{s['sql_summary']['one_time_buyer_share']:.0%}")
with c2:
    d = coh[(coh.months_since_acquisition <= max_m) & (coh.months_since_acquisition > 0)
            & (coh.cohort_size >= min_size)]
    col = "retention_rate" if metric == "Retention rate" else "active_customers"
    piv = d.pivot(index="cohort", columns="months_since_acquisition", values=col)
    fig = px.imshow(piv, color_continuous_scale="Blues", aspect="auto",
                    zmax=0.5 if col == "retention_rate" else None,
                    labels=dict(x="Months since first purchase", y="Acquisition cohort",
                                color="Retention" if col == "retention_rate" else "Customers"))
    fig.update_traces(texttemplate="%{z:.0%}" if col == "retention_rate" else "%{z:.0f}", textfont_size=9)
    fig.update_layout(height=620, title="Share of each cohort purchasing in month k")
    st.plotly_chart(fig, width="stretch")
st.caption("The Dec-2009 cohort contains customers who were already active before the extract starts, so its "
           "retention is higher than true new-customer cohorts.")

st.subheader("Retention curves by cohort")
sel = st.multiselect("Cohorts", sorted(coh.cohort.unique()), default=["2010-01", "2010-03", "2010-06", "2010-09",
                                                                      "2010-12"])
d = coh[coh.cohort.isin(sel) & (coh.months_since_acquisition > 0)]
fig = px.line(d, x="months_since_acquisition", y="retention_rate", color="cohort", markers=True,
              labels={"months_since_acquisition": "Months since acquisition", "retention_rate": "Retention"})
fig.update_layout(height=380, yaxis_tickformat=".0%")
st.plotly_chart(fig, width="stretch")

st.subheader("Activity over time")
mr = csv("monthly_revenue.csv")
mac = csv("monthly_active_customers.csv")
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Bar(x=mr.month, y=mr.revenue, marker_color=NAVY, name="Revenue"))
    fig.update_layout(title="Monthly revenue (clean invoices) - note the Q4 peak", height=360,
                      yaxis_tickprefix="£", yaxis_tickformat=",.0f")
    st.plotly_chart(fig, width="stretch")
with c2:
    m = mac.melt(id_vars="month_end", var_name="window", value_name="customers")
    fig = px.line(m, x="month_end", y="customers", color="window", markers=True,
                  labels={"month_end": "Month end", "customers": "Active customers"})
    fig.update_layout(title="Rolling 30/60/90-day active customers (SQL window frames)", height=360)
    st.plotly_chart(fig, width="stretch")
st.caption("Windows at the first two month-ends are truncated because the extract starts on 2009-12-01.")

st.subheader("RFM segments (as of data end)")
rfm = csv("rfm_segments.csv").sort_values("customers", ascending=False)
c1, c2 = st.columns([2, 3])
with c1:
    fig = px.bar(rfm, x="customers", y="rfm_segment", orientation="h", text="customers",
                 labels={"rfm_segment": "", "customers": "Customers"})
    fig.update_layout(height=340, yaxis_categoryorder="total ascending")
    st.plotly_chart(fig, width="stretch")
with c2:
    st.dataframe(rfm.rename(columns={"rfm_segment": "Segment", "customers": "Customers",
                                     "avg_recency": "Avg recency (d)", "avg_frequency": "Avg invoices",
                                     "avg_monetary": "Avg spend £", "total_monetary": "Total spend £"}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.0f") for c in
                                ["Avg recency (d)", "Avg spend £", "Total spend £"]} |
                 {"Avg invoices": st.column_config.NumberColumn(format="%.1f")})
