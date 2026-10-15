"""Household churn and CLV on a US grocery loyalty panel - multipage portfolio app (entrypoint).

Run locally:  streamlit run app/streamlit_app.py
Reads only precomputed artifacts in app/data/ (refresh with `make app-data`).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import DATA_SOURCE, MEMO_URL, REPO_URL, summary  # noqa: E402

st.set_page_config(page_title="Household churn and CLV | US grocery loyalty panel", layout="wide",
                   initial_sidebar_state="expanded")

V = Path(__file__).resolve().parent / "views"
pages = {
    "Story": [
        st.Page(V / "overview.py", title="Executive summary", icon=":material/insights:", default=True),
        st.Page(V / "data_cleaning.py", title="Data and cleaning", icon=":material/cleaning_services:"),
        st.Page(V / "cohorts.py", title="Cohorts and shopping behavior", icon=":material/grid_on:"),
    ],
    "Models": [
        st.Page(V / "churn_model.py", title="Churn model", icon=":material/model_training:"),
        st.Page(V / "clv.py", title="Household lifetime value", icon=":material/payments:"),
        st.Page(V / "survival.py", title="Survival analysis", icon=":material/timeline:"),
    ],
    "Coupons and segments": [
        st.Page(V / "campaigns.py", title="Coupons and campaigns", icon=":material/local_offer:"),
        st.Page(V / "demographics.py", title="Demographics (partial)", icon=":material/groups:"),
    ],
    "Decide": [
        st.Page(V / "targeting.py", title="Retention targeting simulator", icon=":material/tune:"),
        st.Page(V / "methodology.py", title="Methodology and limitations", icon=":material/menu_book:"),
    ],
}
nav = st.navigation(pages)

s = summary()
sq = s["sql_summary"]
with st.sidebar:
    st.markdown("### Household churn and CLV")
    st.caption("Complete Journey (84.51) | US grocery loyalty panel | calendar year 2017")
    st.markdown(
        f"**{sq['n_households']:,}** households | **{sq['n_baskets_clean']:,}** baskets\n\n"
        f"Scoring date: **{s['config']['churn']['test_cutoff']}** (back-test)")
    st.divider()
    st.markdown("**Stack:** DuckDB SQL | scikit-learn | LightGBM | SHAP | lifetimes | R survival")
    st.markdown(f"[GitHub repository]({REPO_URL}) | [Executive memo]({MEMO_URL}) | [Data source]({DATA_SOURCE})")
    st.caption("Churn is a proxy: no purchase in the 90 days after the cutoff, not an observed cancellation. "
               "Items marked ASSUMED are scenario inputs, not observed results.")
nav.run()
