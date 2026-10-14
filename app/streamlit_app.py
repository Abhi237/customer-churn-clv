"""Customer Lifecycle, Churn & CLV - multipage portfolio app (entrypoint).

Run locally:  streamlit run app/streamlit_app.py
Reads only precomputed artifacts in app/data/ (refresh with `make app-data`).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import MEMO_URL, REPO_URL, summary  # noqa: E402

st.set_page_config(page_title="Churn & CLV | Customer Lifecycle Analytics", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

V = Path(__file__).resolve().parent / "views"
pages = {
    "Story": [
        st.Page(V / "overview.py", title="Executive summary", icon=":material/insights:", default=True),
        st.Page(V / "data_cleaning.py", title="Data & cleaning", icon=":material/cleaning_services:"),
        st.Page(V / "cohorts.py", title="Cohorts & lifecycle", icon=":material/grid_on:"),
    ],
    "Models": [
        st.Page(V / "churn_model.py", title="Churn model", icon=":material/model_training:"),
        st.Page(V / "clv.py", title="Customer lifetime value", icon=":material/payments:"),
        st.Page(V / "survival.py", title="Survival analysis", icon=":material/timeline:"),
    ],
    "Decide": [
        st.Page(V / "targeting.py", title="Retention targeting simulator", icon=":material/tune:"),
        st.Page(V / "methodology.py", title="Methodology & limitations", icon=":material/menu_book:"),
    ],
}
nav = st.navigation(pages)

s = summary()
with st.sidebar:
    st.markdown("### 📈 Customer Lifecycle, Churn & CLV")
    st.caption("UCI Online Retail II · UK gift wholesaler · Dec-2009 to Dec-2011")
    st.markdown(
        f"**{s['sql_summary']['n_customers_clean']:,}** customers · "
        f"**{s['sql_summary']['n_invoices_clean']:,}** invoices\n\n"
        f"Scoring date: **{s['config']['churn']['test_cutoff']}** (back-test)")
    st.divider()
    st.markdown("**Stack:** DuckDB SQL · scikit-learn · LightGBM · SHAP · lifetimes · R survival")
    st.markdown(f"[📂 GitHub repository]({REPO_URL}) · [📄 Executive memo]({MEMO_URL})")
    st.caption("All numbers are computed by the pipeline. Items marked ASSUMED are scenario inputs, "
               "not observed results.")
nav.run()
