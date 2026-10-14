"""Evaluation metrics: discrimination, calibration, lift/gains."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def lift_table(y_true, score, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift/gains table; decile 1 = highest scores."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "s": np.asarray(score, dtype=float)})
    df = df.sample(frac=1.0, random_state=0).sort_values("s", ascending=False, kind="mergesort").reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    base = df["y"].mean()
    g = df.groupby("decile").agg(n=("y", "size"), positives=("y", "sum"), mean_score=("s", "mean"))
    g["rate"] = g["positives"] / g["n"]
    g["lift"] = g["rate"] / base if base > 0 else np.nan
    g["cum_positives"] = g["positives"].cumsum()
    g["cum_gain"] = g["cum_positives"] / df["y"].sum() if df["y"].sum() > 0 else np.nan
    g["cum_lift"] = (g["cum_positives"] / g["n"].cumsum()) / base if base > 0 else np.nan
    return g.reset_index()


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    y, p = np.asarray(y_true, float), np.asarray(prob, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classification_metrics(y_true, prob) -> dict[str, float]:
    y, p = np.asarray(y_true, float), np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
    lt = lift_table(y, p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "lift_top10": float(lt.loc[0, "lift"]),
        "lift_top20": float(lt.loc[1, "cum_lift"]),
        "gain_top20": float(lt.loc[1, "cum_gain"]),
        "gain_top30": float(lt.loc[2, "cum_gain"]),
        "base_rate": float(y.mean()),
    }
