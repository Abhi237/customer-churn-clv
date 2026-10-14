"""Explore customer churn-proxy scores, CLV and value x risk segments.

Run: streamlit run app/streamlit_app.py   (after `make train`)
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "artifacts" / "customer_scores_2011-09-01.csv"

st.set_page_config(page_title="Churn & CLV explorer", layout="wide")
st.title("Customer churn-proxy & CLV explorer")
st.caption("Back-test scoring date 2011-09-01. Churn = no purchase in the next 90 days (a proxy). "
           "Expected net value uses ASSUMED uplift/cost/margin - not observed impact.")

if not SCORES.exists():
    st.error("Scores not found - run `make train` first.")
    st.stop()
df = pd.read_csv(SCORES, dtype={"customer_id": str})

c1 = c2 = c3 = st.sidebar
uplift = c1.slider("ASSUMED relative churn reduction", 0.0, 0.5, 0.10, 0.01)
margin = c2.slider("ASSUMED gross margin", 0.05, 0.8, 0.30, 0.05)
cost = c3.number_input("ASSUMED contact cost (GBP)", 0.0, 100.0, 5.0, 1.0)
df["expected_net_value"] = df["churn_prob"] * uplift * df["clv_12m"] * margin - cost
df["target"] = df["expected_net_value"] > 0

k = st.columns(4)
k[0].metric("Customers scored", f"{len(df):,}")
k[1].metric("Targeted (scenario)", f"{df.target.sum():,}")
k[2].metric("Campaign cost (scenario)", f"GBP {cost * df.target.sum():,.0f}")
k[3].metric("Expected net value (scenario)", f"GBP {df.loc[df.target, 'expected_net_value'].sum():,.0f}")

st.subheader("Value x risk matrix")
order_v = ["High value", "Mid value", "Low value"]
order_r = ["Low risk", "Medium risk", "High risk"]
pt = df.pivot_table(index="value_segment", columns="risk_segment", values="customer_id", aggfunc="count")
st.dataframe(pt.reindex(index=order_v, columns=order_r), width="stretch")
obs = df.pivot_table(index="value_segment", columns="risk_segment", values="churned", aggfunc="mean")
st.caption("Observed churn-proxy rate in the 90 days after scoring (back-test):")
st.dataframe(obs.reindex(index=order_v, columns=order_r).style.format("{:.0%}"), width="stretch")

st.subheader("Customers")
seg_v = st.multiselect("Value segment", order_v, order_v)
seg_r = st.multiselect("Risk segment", order_r, order_r)
view = df[df.value_segment.isin(seg_v) & df.risk_segment.isin(seg_r)].sort_values("expected_net_value",
                                                                                   ascending=False)
st.dataframe(view, width="stretch", height=420)
st.download_button("Download selection (CSV)", view.to_csv(index=False), "customer_selection.csv")
