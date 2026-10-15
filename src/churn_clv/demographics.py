"""Demographic breakout for the households that have demographics (PARTIAL coverage: 801 of 2,469).

Demographics are never used as model features. The breakout is descriptive; cells are small, so every row carries
the number of scored households, the number of churn-proxy events and a Wilson interval, and rows below
`min_cell_size` are flagged unreliable. Coverage itself is compared with the uncovered households because the
covered subset is not guaranteed to be representative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .metrics import wilson_ci

INCOME_BANDS = {
    "Under 15K": "Under $35K", "15-24K": "Under $35K", "25-34K": "Under $35K",
    "35-49K": "$35K-$74K", "50-74K": "$35K-$74K",
    "75-99K": "$75K-$124K", "100-124K": "$75K-$124K",
    "125-149K": "$125K+", "150-174K": "$125K+", "175-199K": "$125K+", "200-249K": "$125K+", "250K+": "$125K+",
}
INCOME_ORDER = ["Under $35K", "$35K-$74K", "$75K-$124K", "$125K+"]
AGE_ORDER = ["19-24", "25-34", "35-44", "45-54", "55-64", "65+"]
COMP_ORDER = ["1 Adult No Kids", "1 Adult Kids", "2 Adults No Kids", "2 Adults Kids"]

DIMENSIONS = {"Income": ("income_band", INCOME_ORDER), "Age": ("age", AGE_ORDER),
              "Household composition": ("household_comp", COMP_ORDER)}


def with_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["income_band"] = out["income"].map(INCOME_BANDS)
    return out


def breakout(scored: pd.DataFrame, min_cell_size: int = 30) -> pd.DataFrame:
    """One row per (dimension, level) among scored households that have demographics."""
    d = with_bands(scored[scored["has_demographics"]])
    rows = []
    for dim, (col, order) in DIMENSIONS.items():
        for lvl in order:
            g = d[d[col] == lvl]
            n, k = len(g), int(g["churned"].sum())
            if n == 0:
                continue
            lo, hi = wilson_ci(k, n)
            rows.append({
                "dimension": dim, "level": lvl, "n_scored": n, "n_churned": k, "churn_rate": k / n,
                "ci_lo": float(lo), "ci_hi": float(hi), "mean_churn_prob": float(g["churn_prob"].mean()),
                "mean_clv_12m": float(g["clv_12m"].mean()), "mean_monetary_total": float(g["monetary_total"].mean()),
                "mean_avg_basket_spend": float(g["avg_basket_spend"].mean()),
                "mean_coupon_disc_share": float(g["coupon_disc_share"].mean()),
                "mean_retail_disc_share": float(g["retail_disc_share"].mean()),
                "mean_share_fresh": float(g["share_fresh"].mean()),
                "reliable": bool(n >= min_cell_size and k >= 5),
            })
    return pd.DataFrame(rows)


def coverage_comparison(scored: pd.DataFrame) -> pd.DataFrame:
    """Covered (has demographics) vs not covered households on behaviour, outcome and model performance."""
    rows = []
    for name, g in [("With demographics", scored[scored["has_demographics"]]),
                    ("Without demographics", scored[~scored["has_demographics"]])]:
        enough_events = 10 <= g["churned"].sum() < len(g)  # AUC is NaN below 10 events
        auc = roc_auc_score(g["churned"], g["churn_prob"]) if enough_events else np.nan
        rows.append({"group": name, "n_scored": len(g), "n_churned": int(g["churned"].sum()),
                     "churn_rate": float(g["churned"].mean()), "mean_churn_prob": float(g["churn_prob"].mean()),
                     "median_monetary_total": float(g["monetary_total"].median()),
                     "median_frequency": float(g["frequency"].median()),
                     "mean_avg_basket_spend": float(g["avg_basket_spend"].mean()),
                     "mean_coupon_disc_share": float(g["coupon_disc_share"].mean()),
                     "mean_clv_12m": float(g["clv_12m"].mean()), "roc_auc_final_model": float(auc)})
    return pd.DataFrame(rows)
