import plotly.express as px
import streamlit as st
from core import NAVY, TEAL, callout, csv, page_story, setup_page, summary

s = summary()
ds = s["demographics_summary"]
setup_page("Demographics breakout (partial coverage)", "Only 801 of 2,469 households have demographics | descriptive "
           "only, demographics are never model features")
page_story('Do churn risk and value differ by household type?',
           'Coverage is partial, so use this for context only, never to pick who gets an offer.')
callout(f"<b>Partial coverage.</b> {ds['n_with_demographics']:,} of {ds['n_households']:,} households "
        f"({ds['coverage_share']:.0%}) have demographic attributes. They are not a random sample: as the comparison below "
        f"shows they are much heavier shoppers, and only {ds['n_churned_with_demographics']} of "
        f"{ds['n_scored_with_demographics']} scored covered households went quiet in the test window. Group-level churn "
        "rates are therefore not informative and are not shown as rates; the breakout describes behavior and "
        "predicted value instead.")
st.write("")

st.subheader("Who is covered?")
cov = csv("demographics_coverage.csv").set_index("group").T
cov.index.name = "metric"
labels = {"n_scored": "Households scored", "n_churned": "Churn-proxy events", "churn_rate": "Observed churn-proxy rate",
          "mean_churn_prob": "Mean predicted churn probability", "median_monetary_total": "Median spend to date (USD)",
          "median_frequency": "Median shopping days to date", "mean_avg_basket_spend": "Mean basket spend (USD)",
          "mean_coupon_disc_share": "Mean coupon-discount share", "mean_clv_12m": "Mean 12-month CLV (USD)",
          "roc_auc_final_model": "Model ROC-AUC (needs 10+ events)"}
cov = cov.loc[list(labels)].rename(index=labels)
st.dataframe(cov.astype(float).round(4), width="stretch")
st.caption("Covered households shop more often, spend more per household and almost never lapse, so conclusions "
           "about demographic groups do not transfer to the 1,513 uncovered scored households.")

st.subheader("Behavior and predicted value by group")
tab = csv("demographics_breakout.csv")
dims = list(tab.dimension.unique())
dim = st.segmented_control("Dimension", dims, default=dims[0]) or dims[0]
g = tab[tab.dimension == dim].copy()
g["label"] = g.level + " (n=" + g.n_scored.astype(str) + ")"
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(g, x="label", y="mean_clv_12m", labels={"label": "", "mean_clv_12m": "Mean 12-month CLV (USD)"})
    fig.update_traces(marker_color=NAVY)
    fig.update_layout(title="Mean predicted 12-month CLV", height=380)
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = px.bar(g, x="label", y="mean_avg_basket_spend", labels={"label": "", "mean_avg_basket_spend": "Mean basket spend (USD)"})
    fig.update_traces(marker_color=TEAL)
    fig.update_layout(title="Mean basket spend", height=380)
    st.plotly_chart(fig, width="stretch")
st.dataframe(g.drop(columns=["label", "dimension"]), hide_index=True, width="stretch", column_config={
    "level": "Group", "n_scored": "Households", "n_churned": "Churn-proxy events",
    "churn_rate": st.column_config.NumberColumn("Observed churn rate", format="percent"),
    "ci_lo": st.column_config.NumberColumn("CI low", format="percent"),
    "ci_hi": st.column_config.NumberColumn("CI high", format="percent"),
    "mean_churn_prob": st.column_config.NumberColumn("Mean predicted churn", format="percent"),
    "mean_clv_12m": st.column_config.NumberColumn("Mean CLV 12m (USD)", format="%.0f"),
    "mean_monetary_total": st.column_config.NumberColumn("Mean spend to date (USD)", format="%.0f"),
    "mean_avg_basket_spend": st.column_config.NumberColumn("Mean basket (USD)", format="%.2f"),
    "mean_coupon_disc_share": st.column_config.NumberColumn("Coupon share", format="percent"),
    "mean_retail_disc_share": st.column_config.NumberColumn("Store-discount share", format="percent"),
    "mean_share_fresh": st.column_config.NumberColumn("Fresh share", format="percent"),
    "reliable": "Cell reliable (n >= 30 and >= 5 events)"})
st.caption(f"{ds['unreliable_cells']} of {ds['cells']} cells are flagged unreliable because they have fewer than 5 "
           "churn-proxy events. Differences between groups are descriptive and not tested for significance.")
