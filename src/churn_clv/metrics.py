"""Evaluation metrics: discrimination, calibration, lift/gains."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


def calibration_slope_intercept(y_true, prob) -> tuple[float, float]:
    """Logistic recalibration y ~ a + b * logit(p): slope b = 1 and intercept a = 0 mean perfectly calibrated.

    b < 1 means predictions are too extreme (over-confident); a > 0 means risk is under-predicted on average.
    """
    y, p = np.asarray(y_true, float), np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def bootstrap_ci(y_true, prob, metric, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of metric(y, p) over resampled rows (resamples without both classes are redrawn)."""
    y, p = np.asarray(y_true, float), np.asarray(prob, float)
    rng = np.random.default_rng(seed)
    vals = []
    while len(vals) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(metric(y[idx], p[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(y_true, p_a, p_b, metric, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap of metric(y, p_a) - metric(y, p_b) on the same resampled rows."""
    y, a, b = np.asarray(y_true, float), np.asarray(p_a, float), np.asarray(p_b, float)
    rng = np.random.default_rng(seed)
    d = []
    while len(d) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            d.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    d = np.asarray(d)
    return {"diff": float(metric(y, a) - metric(y, b)), "lo": float(np.quantile(d, 0.025)),
            "hi": float(np.quantile(d, 0.975)), "share_a_better": float((d > 0).mean())}


def classification_metrics(y_true, prob) -> dict[str, float]:
    y, p = np.asarray(y_true, float), np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
    lt = lift_table(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "calib_slope": slope,
        "calib_intercept": intercept,
        "mean_pred": float(p.mean()),
        "lift_top10": float(lt.loc[0, "lift"]),
        "lift_top20": float(lt.loc[1, "cum_lift"]),
        "gain_top20": float(lt.loc[1, "cum_gain"]),
        "gain_top30": float(lt.loc[2, "cum_gain"]),
        "base_rate": float(y.mean()),
    }


def wilson_ci(k, n, z: float = 1.96):
    """Wilson score interval for a binomial proportion k / n (vectorised); returns (lo, hi)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)
