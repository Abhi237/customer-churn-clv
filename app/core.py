"""Shared loaders, theme and UI helpers for the Streamlit app.

The app reads ONLY precomputed files in app/data/ (built by `make app-data`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
DOCS = Path(__file__).resolve().parents[1] / "docs"
REPO_URL = "https://github.com/Abhi237/customer-churn-clv"
MEMO_URL = f"{REPO_URL}/blob/main/docs/retention_memo.pdf"

NAVY, RED, GREEN, PURPLE, ORANGE, TEAL, GREY = ("#1f4e79", "#c0504d", "#5b9a3c", "#8064a2", "#e8892b",
                                                "#2e9bb5", "#7f8c99")
COLORWAY = [NAVY, RED, GREEN, PURPLE, ORANGE, TEAL, GREY]

pio.templates["portfolio"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#1a2433"),
    colorway=COLORWAY, paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor="#e6ebf1", zerolinecolor="#d0d7e0", linecolor="#c3ccd6"),
    yaxis=dict(gridcolor="#e6ebf1", zerolinecolor="#d0d7e0", linecolor="#c3ccd6"),
    legend=dict(bgcolor="rgba(255,255,255,0.7)", orientation="h", y=-0.2),
    margin=dict(l=50, r=20, t=50, b=50), title=dict(font=dict(size=15)),
    hoverlabel=dict(font_size=12),
))
pio.templates.default = "plotly_white+portfolio"

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1 {font-weight: 700; letter-spacing: -0.02em;}
h2, h3 {font-weight: 600; letter-spacing: -0.01em;}
[data-testid="stMetric"] {background: #f3f6fa; border: 1px solid #e1e7ef; border-radius: 10px;
  padding: 14px 16px; border-left: 4px solid #1f4e79;}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #4a5768; text-transform: uppercase;
  letter-spacing: 0.04em;}
[data-testid="stMetricValue"] {font-size: 1.65rem; font-weight: 700; color: #1f4e79;}
.assumed {background:#fff4e5; border:1px solid #f5c27a; border-radius:8px; padding:10px 14px; color:#7a4a00;}
.callout {background:#eef4fb; border-left:4px solid #1f4e79; border-radius:6px; padding:12px 16px;}
.small {font-size:0.85rem; color:#5a6676;}
</style>
"""


def setup_page(title: str, subtitle: str | None = None) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f"<p class='small'>{subtitle}</p>", unsafe_allow_html=True)


def assumed_banner(text: str) -> None:
    st.markdown(f"<div class='assumed'>⚠️ <b>ASSUMED inputs - scenario only.</b> {text}</div>",
                unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f"<div class='callout'>{text}</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def summary() -> dict:
    return json.loads((DATA / "summary.json").read_text())


@st.cache_data(show_spinner=False)
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data(show_spinner=False)
def customers() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "customers.parquet")
    df["customer_id"] = df["customer_id"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def shap_values(model: str = "logreg") -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"shap_values_{model}.parquet")
    df["customer_id"] = df["customer_id"].astype(str)
    return df


def gbp(x: float) -> str:
    if abs(x) >= 1e6:
        return f"£{x / 1e6:,.2f}m"
    if abs(x) >= 1e4:
        return f"£{x / 1e3:,.0f}k"
    return f"£{x:,.0f}"


def test_metrics() -> pd.DataFrame:
    m = csv("model_metrics.csv")
    return m[m.split == "test"].copy()


VALUE_ORDER = ["High value", "Mid value", "Low value"]
RISK_ORDER = ["Low risk", "Medium risk", "High risk"]
FEATURE_LABELS = {
    "recency_days": "Days since last purchase", "tenure_days": "Days since first purchase",
    "frequency": "Invoices (all history)", "monetary_total": "Total spend", "avg_order_value": "Avg order value",
    "invoices_30d": "Invoices, last 30d", "invoices_90d": "Invoices, last 90d", "invoices_180d": "Invoices, last 180d",
    "revenue_90d": "Revenue, last 90d", "revenue_180d": "Revenue, last 180d",
    "n_distinct_products": "Distinct products", "avg_products_per_invoice": "Products per invoice",
    "mean_interpurchase_days": "Mean gap between purchases", "std_interpurchase_days": "Std of purchase gaps",
    "active_months_12m": "Active months (last 12)", "n_returns_365d": "Returns, last 365d",
    "return_ratio_365d": "Return ratio", "recency_to_gap_ratio": "Recency / usual gap", "is_uk": "UK customer",
}
