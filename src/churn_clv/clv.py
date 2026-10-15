"""Probabilistic CLV: BG/NBD (shopping frequency) + Gamma-Gamma (spend per shopping day) via `lifetimes`.

Time unit: days, one transaction per household-day (all baskets of the day are merged). The panel is only
12 months long, so validation uses a 26-week calibration / 26-week holdout split (and a 13-week holdout that
mirrors the scoring set-up). CLV beyond the observed 12 months is an extrapolation of the fitted model.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data
from scipy.special import hyp2f1

from .db import as_date


def daily_transactions(days: pd.DataFrame) -> pd.DataFrame:
    """One row per household-day with spend (input: household_id, visit_date, spend)."""
    d = days.groupby(["household_id", "visit_date"], as_index=False)["spend"].sum()
    d["visit_date"] = pd.to_datetime(d["visit_date"])
    d["spend"] = d["spend"].astype(float)
    return d


def expected_purchases(bgf: BetaGeoFitter, t, frequency, recency, T) -> np.ndarray:
    """BG/NBD conditional expected purchases in the next `t` days (Fader, Hardie and Lee 2005, eq. 10).

    Re-implementation of `lifetimes`' method WITHOUT the logarithm of the Gauss hypergeometric term: when
    a + b + x - 1 < 0 (a fitted a + b < 1 and zero repeat purchases) that term is negative and `lifetimes`
    returns NaN for those households. Verified equal to the library wherever the library is defined (unit test).
    """
    r, alpha, a, b = (float(bgf.params_[k]) for k in ("r", "alpha", "a", "b"))
    x = np.asarray(frequency, float)
    t = np.asarray(t, float)
    rec, T = np.asarray(recency, float), np.asarray(T, float)
    z = t / (alpha + T + t)
    c = a + b + x - 1
    hyp = hyp2f1(r + x, b + x, c, z)
    alt = hyp2f1(c - (r + x), c - (b + x), c, z) * (1 - z) ** (c - (r + x) - (b + x))
    hyp = np.where(np.isfinite(hyp), hyp, alt)
    first = c / (a - 1)
    second = 1 - hyp * ((alpha + T) / (alpha + T + t)) ** (r + x)
    denom = 1 + (x > 0) * (a / (b + x - 1)) * ((alpha + T) / (alpha + rec)) ** (r + x)
    return first * second / denom


def fit_models(summary: pd.DataFrame, penalizer: float = 0.001):
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    rep = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(rep["frequency"], rep["monetary_value"])
    return bgf, ggf


def expected_aov(ggf: GammaGammaFitter, summary: pd.DataFrame) -> pd.Series:
    """Gamma-Gamma conditional expected spend per shopping day.

    Zero-repeat households have no repeat-spend history. The Gamma-Gamma population mean p*v/(q-1) is undefined when
    q <= 1 (heavy-tailed spend), so they get the pooled empirical mean repeat-day spend of the fitting period.
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
    """Sum over months of expected shopping days in that month x expected spend per day, discounted monthly."""
    f, r, t = summary["frequency"], summary["recency"], summary["T"]
    clv = pd.Series(0.0, index=summary.index)
    prev = 0.0
    for m in range(1, months + 1):
        cum = expected_purchases(bgf, m * days_per_month, f, r, t)
        clv += (cum - prev) * aov / (1 + discount_rate_monthly) ** m
        prev = cum
    return clv


def holdout_validation(days: pd.DataFrame, calibration_end: str, observation_end: str,
                       penalizer: float = 0.001) -> tuple[pd.DataFrame, dict]:
    tx = daily_transactions(days)
    ch = calibration_and_holdout_data(
        tx, "household_id", "visit_date", calibration_period_end=calibration_end,
        observation_period_end=observation_end, freq="D", monetary_value_col="spend",
    )
    summ = ch.rename(columns={"frequency_cal": "frequency", "recency_cal": "recency", "T_cal": "T",
                              "monetary_value_cal": "monetary_value"})
    bgf, ggf = fit_models(summ, penalizer)
    h = float(ch["duration_holdout"].iloc[0])
    ch["pred_purchases"] = expected_purchases(bgf, h, ch["frequency_cal"], ch["recency_cal"], ch["T_cal"])
    rep = ch["frequency_cal"] > 0
    ch["pred_aov"] = expected_aov(ggf, summ).to_numpy()
    ch["pred_spend"] = ch["pred_purchases"] * ch["pred_aov"]
    # actual holdout spend = holdout shopping days * average holdout spend per day
    ch["actual_spend"] = ch["frequency_holdout"] * ch["monetary_value_holdout"]
    rr = summ[(summ["frequency"] > 0) & (summ["monetary_value"] > 0)]
    m = {
        "calibration_end": calibration_end,
        "observation_end": observation_end,
        "calibration_days": float(ch["T_cal"].max()),
        "holdout_days": h,
        "n_households": int(len(ch)),
        "actual_purchases_total": float(ch["frequency_holdout"].sum()),
        "pred_purchases_total": float(ch["pred_purchases"].sum()),
        "purchases_mae": float((ch["pred_purchases"] - ch["frequency_holdout"]).abs().mean()),
        "purchases_corr": float(np.corrcoef(ch["pred_purchases"], ch["frequency_holdout"])[0, 1]),
        "purchases_spearman": float(ch[["pred_purchases", "frequency_holdout"]].corr("spearman").iloc[0, 1]),
        "baseline_mae_mean_rate": float((ch["frequency_holdout"] - ch["frequency_holdout"].mean()).abs().mean()),
        "actual_spend_total": float(ch["actual_spend"].sum()),
        "pred_spend_total": float(ch["pred_spend"].sum()),
        "spend_corr": float(np.corrcoef(ch["pred_spend"], ch["actual_spend"])[0, 1]),
        "spend_spearman": float(ch[["pred_spend", "actual_spend"]].corr("spearman").iloc[0, 1]),
        "aov_corr_repeaters": float(ch.loc[rep & (ch["frequency_holdout"] > 0), ["pred_aov", "monetary_value_holdout"]]
                                    .corr().iloc[0, 1]),
        # Gamma-Gamma assumes spend per day is independent of shopping frequency: report how well that holds
        "gg_freq_spend_corr": float(np.corrcoef(rr["frequency"], rr["monetary_value"])[0, 1]),
    }
    # naive benchmark for purchases: each household's calibration-period rate extrapolated to the holdout
    naive = ch["frequency_cal"] / ch["T_cal"].clip(lower=1) * h
    m["naive_rate_mae"] = float((naive - ch["frequency_holdout"]).abs().mean())
    m["naive_rate_pred_total"] = float(naive.sum())
    order = ch["pred_spend"].rank(ascending=False, method="first")
    top = order <= len(ch) * 0.1
    m["top10_share_of_actual_spend"] = float(ch.loc[top, "actual_spend"].sum() / ch["actual_spend"].sum())
    m["bgf_params"] = {k: float(v) for k, v in bgf.params_.items()}
    m["ggf_params"] = {k: float(v) for k, v in ggf.params_.items()}
    return ch.reset_index(), m


def score_clv(days: pd.DataFrame, as_of: str | date, horizon_months: int = 12,
              discount_rate_monthly: float = 0.01, penalizer: float = 0.001) -> pd.DataFrame:
    """Fit on transactions strictly before `as_of` and predict CLV for the next horizon."""
    c = pd.Timestamp(as_date(as_of))
    tx = daily_transactions(days)
    tx = tx[tx["visit_date"] < c]
    summ = summary_data_from_transaction_data(
        tx, "household_id", "visit_date", monetary_value_col="spend",
        observation_period_end=c - pd.Timedelta(days=1), freq="D")
    bgf, ggf = fit_models(summ, penalizer)
    out = summ.copy()
    out["p_alive"] = bgf.conditional_probability_alive(summ["frequency"], summ["recency"], summ["T"])
    out["pred_purchases_90d"] = expected_purchases(bgf, 90, summ["frequency"], summ["recency"], summ["T"])
    aov = expected_aov(ggf, summ)
    out["expected_aov"] = aov
    out["clv_12m"] = discounted_clv(bgf, summ, aov, horizon_months, discount_rate_monthly)
    out.attrs["bgf_params"] = {k: float(v) for k, v in bgf.params_.items()}
    out.attrs["ggf_params"] = {k: float(v) for k, v in ggf.params_.items()}
    return out.reset_index()
