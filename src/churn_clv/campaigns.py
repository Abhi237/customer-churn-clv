"""Coupon / campaign analysis: DESCRIPTIVE and OBSERVATIONAL only.

Campaign membership lists the households each campaign targeted; nothing in the data says how they were chosen
and there is no holdout group, so nothing here can be read as a causal effect of a campaign or coupon. The module
therefore (a) describes exposure and redemption, (b) compares observed outcomes with confidence intervals, and
(c) shows how different targeted and non-targeted households already were at the cutoff (standardised mean
differences), which is exactly why the raw comparison cannot be causal.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .db import as_date
from .metrics import wilson_ci

BALANCE_FEATURES = ["trips_per_week", "spend_per_week", "avg_basket_spend", "trips_28d", "recency_days",
                    "coupon_disc_share", "retail_disc_share", "share_fresh", "redemptions_90d", "top_store_share"]


def window_exposure(con, cutoff, horizon_days: int) -> pd.DataFrame:
    """Per household: targeted by >=1 campaign that STARTS inside [cutoff, cutoff + horizon) (future exposure),
    redemptions that occur inside the same window, and pre-cutoff history (campaigns received / coupons redeemed
    before the cutoff, all time)."""
    c = as_date(cutoff)
    end = c + timedelta(days=horizon_days)
    return con.execute(f"""
        WITH e AS (
            SELECT household_id, COUNT(*) AS n_window_campaigns,
                   COUNT(*) FILTER (WHERE campaign_type = 'Type A') AS n_window_type_a
            FROM campaign_exposure WHERE start_date >= DATE '{c}' AND start_date < DATE '{end}' GROUP BY 1
        ), r AS (
            SELECT household_id,
                   COUNT(*) FILTER (WHERE redemption_date >= DATE '{c}' AND redemption_date < DATE '{end}')
                       AS n_window_redemptions,
                   COUNT(*) FILTER (WHERE redemption_date < DATE '{c}') AS n_prior_redemptions
            FROM coupon_redemptions_clean GROUP BY 1
        ), p AS (
            SELECT household_id, COUNT(*) AS n_prior_campaigns FROM campaign_exposure
            WHERE start_date < DATE '{c}' GROUP BY 1
        )
        SELECT h.household_id, COALESCE(e.n_window_campaigns, 0) AS n_window_campaigns,
               COALESCE(e.n_window_type_a, 0) AS n_window_type_a,
               COALESCE(r.n_window_redemptions, 0) AS n_window_redemptions,
               COALESCE(r.n_prior_redemptions, 0) AS n_prior_redemptions,
               COALESCE(p.n_prior_campaigns, 0) AS n_prior_campaigns
        FROM dim_households h LEFT JOIN e USING (household_id) LEFT JOIN r USING (household_id)
        LEFT JOIN p USING (household_id)""").df()


def _rate_row(g: pd.DataFrame) -> dict:
    n, k = len(g), int(g["churned"].sum())
    lo, hi = wilson_ci(k, n)
    return {"n": n, "n_churned": k, "churn_rate": k / n if n else np.nan, "ci_lo": float(lo), "ci_hi": float(hi),
            "mean_future_spend": float(g["future_spend"].mean()) if n else np.nan}


def outcome_by_flag(df: pd.DataFrame, flag: str, tier_col: str = "risk_segment") -> pd.DataFrame:
    """Observed churn-proxy rate by `flag` (bool) overall and within risk tier."""
    rows = []
    for tier, gt in [("All scored", df)] + [(t, g) for t, g in df.groupby(tier_col, observed=True)]:
        for val, g in gt.groupby(flag):
            rows.append({tier_col: tier, flag: bool(val), **_rate_row(g)})
    return pd.DataFrame(rows)


def adjusted_difference(df: pd.DataFrame, flag: str, tier_col: str = "risk_segment") -> dict:
    """Naive difference in churn rate (flagged - not) and the tier-standardised difference (weights = tier size).

    Standardising on the model's risk tier removes the part of the gap explained by predicted risk; it is still an
    association (unmeasured selection remains), not a causal effect."""
    def diff(g):
        a, b = g[g[flag]], g[~g[flag]]
        return a["churned"].mean() - b["churned"].mean() if len(a) and len(b) else np.nan
    naive = diff(df)
    tiers = [(len(g), diff(g)) for _, g in df.groupby(tier_col, observed=True)]
    w = np.array([n for n, d in tiers if not np.isnan(d)], float)
    d = np.array([d for _, d in tiers if not np.isnan(d)], float)
    return {"flag": flag, "naive_diff": float(naive), "tier_standardised_diff": float((w * d).sum() / w.sum())}


def standardized_mean_differences(df: pd.DataFrame, flag: str, features=BALANCE_FEATURES) -> pd.DataFrame:
    """Baseline (pre-cutoff) imbalance between flagged and unflagged households: SMD = diff of means / pooled SD.
    |SMD| > 0.1 is the usual rule of thumb for meaningful imbalance."""
    a, b = df[df[flag]], df[~df[flag]]
    rows = []
    for f in features:
        sd = np.sqrt((a[f].var(ddof=1) + b[f].var(ddof=1)) / 2)
        rows.append({"feature": f, "mean_flagged": float(a[f].mean()), "mean_not_flagged": float(b[f].mean()),
                     "smd": float((a[f].mean() - b[f].mean()) / sd) if sd > 0 else 0.0})
    return pd.DataFrame(rows)


def campaign_type_summary(dim_campaigns: pd.DataFrame) -> pd.DataFrame:
    g = dim_campaigns.groupby("campaign_type").agg(
        campaigns=("campaign_id", "size"), households_targeted_total=("n_households_targeted", "sum"),
        median_targeted=("n_households_targeted", "median"), redemptions=("n_redemptions", "sum"),
        redeeming_households_total=("n_redeeming_households", "sum")).reset_index()
    g["redeeming_per_targeted"] = g["redeeming_households_total"] / g["households_targeted_total"]
    return g
