import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import NAVY, TEAL, csv, page_story, setup_page, summary

s = summary()
sq = s["sql_summary"]
setup_page("Cohorts and shopping behavior", "Entry cohorts, rolling activity, spend mix and RFM segments | "
           "the panel starts on 2017-01-01, so cohorts are entry cohorts, not acquisition cohorts")
page_story('How quickly do households drop off after they first appear, and what do they buy?',
           'Early drop-off shows when a retention touch is most likely to matter.')

coh = csv("cohort_retention.csv")
coh["cohort"] = pd.to_datetime(coh["entry_cohort"]).dt.strftime("%Y-%m")
c1, c2 = st.columns([1, 3])
with c1:
    max_m = st.slider("Months since entry", 3, 11, 11)
    metric = st.radio("Show", ["Retention rate", "Active households"], horizontal=False)
    min_size = st.number_input("Minimum cohort size", 0, 2000, 20, 10)
    st.metric("Mean month-1 retention (cohorts after January)", f"{sq['month1_retention_mean_excl_jan']:.1%}")
    jan = coh[(coh.cohort == "2017-01") & (coh.months_since_entry == 1)].retention_rate.iloc[0]
    st.metric("January 2017 cohort, month 1", f"{jan:.1%}",
              help="Mostly households that already shopped before 2017 (tenure unknown)")
with c2:
    d = coh[(coh.months_since_entry <= max_m) & (coh.months_since_entry > 0) & (coh.cohort_size >= min_size)]
    col = "retention_rate" if metric == "Retention rate" else "active_households"
    piv = d.pivot(index="cohort", columns="months_since_entry", values=col)
    fig = px.imshow(piv, color_continuous_scale="Blues", aspect="auto",
                    labels=dict(x="Months since entry month", y="Entry cohort",
                                color="Retention" if col == "retention_rate" else "Households"))
    fig.update_traces(texttemplate="%{z:.0%}" if col == "retention_rate" else "%{z:.0f}", textfont_size=9)
    fig.update_layout(height=520, title="Share of each entry cohort with a shopping trip in month k")
    st.plotly_chart(fig, width="stretch")
sizes = sq["cohort_sizes"]
st.caption(f"The January 2017 cohort has {sizes['2017-01']:,} of {sq['n_households']:,} households: most were already "
           "shopping before the panel began, so it is a stock of established households, not new customers. "
           "Later cohorts are small and noisy.")

st.subheader("Activity over time")
ms = csv("monthly_sales.csv")
mac = csv("monthly_active_households.csv")
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Bar(x=ms.month, y=ms.spend, marker_color=NAVY, name="Spend"))
    fig.update_layout(title="Monthly spend (clean baskets)", height=360, yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    st.plotly_chart(fig, width="stretch")
with c2:
    m = mac.melt(id_vars="month_end", var_name="window", value_name="households")
    fig = px.line(m, x="month_end", y="households", color="window", markers=True,
                  labels={"month_end": "Month end", "households": "Active households"})
    fig.update_layout(title="Rolling 30/60/90-day active households (SQL window frames)", height=360)
    st.plotly_chart(fig, width="stretch")
st.caption("Rolling windows at the first month-ends are truncated because the panel starts on 2017-01-01.")

st.subheader("What households buy")
c1, c2 = st.columns(2)
with c1:
    dept = pd.Series(sq["spend_share_by_dept_group"]).drop("unknown", errors="ignore").rename("share").reset_index()
    dept.columns = ["department group", "share"]
    fig = px.bar(dept.sort_values("share"), x="share", y="department group", orientation="h", text_auto=".1%")
    fig.update_traces(marker_color=TEAL)
    fig.update_layout(title="Share of spend by department group", height=330, xaxis_tickformat=".0%",
                      xaxis_title="Share of merchandise spend", yaxis_title="")
    st.plotly_chart(fig, width="stretch")
with c2:
    ms["coupon_basket_share"] = ms.coupon_baskets / ms.baskets
    fig = px.line(ms, x="month", y="coupon_basket_share", markers=True,
                  labels={"month": "", "coupon_basket_share": "Share of baskets with a coupon"})
    fig.update_layout(title="Coupon use by month", height=330, yaxis_tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
k = st.columns(4)
k[0].metric("Median basket spend", f"${sq['median_basket_spend_usd']:.2f}")
k[1].metric("Store discounts, share of shelf value", f"{sq['retail_disc_share_of_shelf']:.1%}")
k[2].metric("Coupon discounts, share of shelf value", f"{sq['coupon_disc_share_of_shelf']:.2%}")
k[3].metric("Top 10% of households, share of spend", f"{sq['top10pct_spend_share']:.0%}")

st.subheader("RFM segments (as of data end)")
rfm = csv("rfm_segments.csv").sort_values("households", ascending=False)
c1, c2 = st.columns([2, 3])
with c1:
    fig = px.bar(rfm, x="households", y="rfm_segment", orientation="h", text="households",
                 labels={"rfm_segment": "", "households": "Households"})
    fig.update_layout(height=340, yaxis_categoryorder="total ascending")
    st.plotly_chart(fig, width="stretch")
with c2:
    st.dataframe(rfm.rename(columns={"rfm_segment": "Segment", "households": "Households",
                                     "avg_recency": "Avg recency (d)", "avg_frequency": "Avg shopping days",
                                     "avg_monetary": "Avg spend $", "total_monetary": "Total spend $"}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.0f") for c in
                                ["Avg recency (d)", "Avg spend $", "Total spend $", "Avg shopping days"]})
st.caption("RFM quintile scores computed in SQL at the end of the data; descriptive only, never a model input.")
