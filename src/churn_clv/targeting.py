"""Value x risk segmentation and an expected-ROI targeting rule.

IMPORTANT: uplift, contact cost and margin are ASSUMPTIONS (no retention experiment exists in the
data). Outputs are scenario estimates, never observed impact.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _qcut(x: pd.Series, quantiles, labels):
    edges = np.quantile(x, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    return pd.cut(x, bins=edges, labels=labels, duplicates="drop")


def assign_segments(df: pd.DataFrame, risk_quantiles, risk_labels, value_quantiles, value_labels,
                    prob_col: str = "churn_prob", value_col: str = "clv_12m") -> pd.DataFrame:
    out = df.copy()
    out["risk_segment"] = _qcut(out[prob_col], risk_quantiles, risk_labels)
    out["value_segment"] = _qcut(out[value_col], value_quantiles, value_labels)
    return out


def expected_net_value(churn_prob, clv, uplift: float, margin: float, cost: float) -> np.ndarray:
    """Expected incremental margin of contacting a customer minus contact cost.

    E[gain] = p_churn * uplift * CLV_12m * margin - cost

    `uplift` is the ASSUMED relative reduction in churn probability caused by the treatment (so the
    expected number of customers "saved" is p_churn * uplift). CLV_12m is used as the value of a saved
    customer - an approximation (BG/NBD CLV is already probability-of-alive weighted, so this is conservative).
    """
    p = np.asarray(churn_prob, float)
    return p * uplift * np.asarray(clv, float) * margin - cost


def targeting_summary(df: pd.DataFrame, uplift: float, margin: float, cost: float) -> dict:
    env = expected_net_value(df["churn_prob"], df["clv_12m"], uplift, margin, cost)
    tgt = env > 0
    return {
        "uplift": uplift, "gross_margin": margin, "contact_cost_gbp": cost,
        "n_targeted": int(tgt.sum()), "share_targeted": float(tgt.mean()),
        "campaign_cost_gbp": float(cost * tgt.sum()),
        "expected_net_value_gbp": float(env[tgt].sum()),
        "expected_net_value_target_all_gbp": float(env.sum()),
    }


def sensitivity_grid(df: pd.DataFrame, uplifts, margins, costs) -> pd.DataFrame:
    rows = [targeting_summary(df, u, m, c) for u, m, c in itertools.product(uplifts, margins, costs)]
    return pd.DataFrame(rows)


def segment_matrix(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["value_segment", "risk_segment"], observed=False).agg(
        customers=("customer_id", "size"),
        mean_churn_prob=("churn_prob", "mean"),
        mean_clv_12m=("clv_12m", "mean"),
        total_clv_12m=("clv_12m", "sum"),
    ).reset_index()
    return g
