"""Probabilistic CLV: BG/NBD (purchase frequency) + Gamma-Gamma (spend) via `lifetimes`."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data

from .db import as_date


def daily_transactions(invoices: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per customer-day (standard for BG/NBD with daily periods)."""
    d = invoices.groupby(["customer_id", "invoice_date"], as_index=False)["revenue"].sum()
    d["invoice_date"] = pd.to_datetime(d["invoice_date"])
    return d


def fit_models(summary: pd.DataFrame, penalizer: float = 0.001):
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    rep = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(rep["frequency"], rep["monetary_value"])
    return bgf, ggf


def expected_aov(ggf: GammaGammaFitter, summary: pd.DataFrame) -> pd.Series:
    """Gamma-Gamma conditional expected transaction value.

    Zero-repeat customers have no repeat-spend history. The Gamma-Gamma population mean p*v/(q-1) is
    undefined when q <= 1 (heavy-tailed wholesale spend - which is what this dataset produces), so they get
    the pooled empirical mean repeat-transaction value of the fitting period instead.
    """
    rep = (summary["frequency"] > 0) & (summary["monetary_value"] > 0)
    aov = pd.Series(np.nan, index=summary.index, dtype=float)
    aov[rep] = ggf.conditional_expected_average_profit(summary.loc[rep, "frequency"],
                                                       summary.loc[rep, "monetary_value"])
    pooled = float((summary.loc[rep, "frequency"] * summary.loc[rep, "monetary_value"]).sum()
                   / summary.loc[rep, "frequency"].sum())
    return aov.fillna(pooled)


def discounted_clv(bgf: BetaGeoFitter, summary: pd.DataFrame, aov: pd.Series, months: int = 12,
                   discount_rate_monthly: float = 0.01, days_per_month: float = 30.0) -> pd.Series:
    """Sum over months of expected purchases in that month x expected AOV, discounted monthly."""
    f, r, t = summary["frequency"], summary["recency"], summary["T"]
    clv = pd.Series(0.0, index=summary.index)
    prev = 0.0
    for m in range(1, months + 1):
        cum = bgf.conditional_expected_number_of_purchases_up_to_time(m * days_per_month, f, r, t)
        clv += (cum - prev) * aov / (1 + discount_rate_monthly) ** m
        prev = cum
    return clv


def holdout_validation(invoices: pd.DataFrame, calibration_end: str, observation_end: str,
                       penalizer: float = 0.001) -> tuple[pd.DataFrame, dict]:
    tx = daily_transactions(invoices)
    ch = calibration_and_holdout_data(
        tx, "customer_id", "invoice_date", calibration_period_end=calibration_end,
        observation_period_end=observation_end, freq="D", monetary_value_col="revenue",
    )
    summ = ch.rename(columns={"frequency_cal": "frequency", "recency_cal": "recency", "T_cal": "T",
                              "monetary_value_cal": "monetary_value"})
    bgf, ggf = fit_models(summ, penalizer)
    h = float(ch["duration_holdout"].iloc[0])
    ch["pred_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        h, ch["frequency_cal"], ch["recency_cal"], ch["T_cal"])
    rep = ch["frequency_cal"] > 0
    ch["pred_aov"] = expected_aov(ggf, summ).to_numpy()
    ch["pred_spend"] = ch["pred_purchases"] * ch["pred_aov"]
    # actual holdout spend = repeat-transactions * avg holdout value
    ch["actual_spend"] = ch["frequency_holdout"] * ch["monetary_value_holdout"]
    m = {
        "holdout_days": h,
        "n_customers": int(len(ch)),
        "actual_purchases_total": float(ch["frequency_holdout"].sum()),
        "pred_purchases_total": float(ch["pred_purchases"].sum()),
        "purchases_mae": float((ch["pred_purchases"] - ch["frequency_holdout"]).abs().mean()),
        "purchases_corr": float(np.corrcoef(ch["pred_purchases"], ch["frequency_holdout"])[0, 1]),
        "actual_spend_total": float(ch["actual_spend"].sum()),
        "pred_spend_total": float(ch["pred_spend"].sum()),
        "spend_corr": float(np.corrcoef(ch["pred_spend"], ch["actual_spend"])[0, 1]),
        "spend_spearman": float(ch[["pred_spend", "actual_spend"]].corr("spearman").iloc[0, 1]),
        "aov_corr_repeaters": float(ch.loc[rep & (ch["frequency_holdout"] > 0), ["pred_aov", "monetary_value_holdout"]]
                                    .corr().iloc[0, 1]),
    }
    # top-decile capture of actual holdout spend
    order = ch["pred_spend"].rank(ascending=False, method="first")
    top = order <= len(ch) * 0.1
    m["top10_share_of_actual_spend"] = float(ch.loc[top, "actual_spend"].sum() / ch["actual_spend"].sum())
    m["bgf_params"] = {k: float(v) for k, v in bgf.params_.items()}
    m["ggf_params"] = {k: float(v) for k, v in ggf.params_.items()}
    return ch.reset_index(), m


def score_clv(invoices: pd.DataFrame, as_of: str | date, horizon_months: int = 12,
              discount_rate_monthly: float = 0.01, penalizer: float = 0.001) -> pd.DataFrame:
    """Fit on transactions strictly before `as_of` and predict CLV for the next horizon."""
    c = pd.Timestamp(as_date(as_of))
    tx = daily_transactions(invoices)
    tx = tx[tx["invoice_date"] < c]
    summ = summary_data_from_transaction_data(
        tx, "customer_id", "invoice_date", monetary_value_col="revenue",
        observation_period_end=c - pd.Timedelta(days=1), freq="D")
    bgf, ggf = fit_models(summ, penalizer)
    out = summ.copy()
    out["p_alive"] = bgf.conditional_probability_alive(summ["frequency"], summ["recency"], summ["T"])
    out["pred_purchases_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        90, summ["frequency"], summ["recency"], summ["T"])
    aov = expected_aov(ggf, summ)
    out["expected_aov"] = aov
    out["clv_12m"] = discounted_clv(bgf, summ, aov, horizon_months, discount_rate_monthly)
    out.attrs["bgf_params"] = {k: float(v) for k, v in bgf.params_.items()}
    out.attrs["ggf_params"] = {k: float(v) for k, v in ggf.params_.items()}
    return out.reset_index()
