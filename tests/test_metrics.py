"""Regression tests for metric implementations (fixed inputs -> fixed outputs)."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from churn_clv.metrics import (
    bootstrap_ci,
    calibration_slope_intercept,
    classification_metrics,
    expected_calibration_error,
    lift_table,
    paired_bootstrap_diff,
    wilson_ci,
)

Y = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0] * 10)
P = np.tile(np.linspace(0.95, 0.05, 10), 10)


def test_lift_table_shape_and_gain():
    lt = lift_table(Y, P)
    assert len(lt) == 10
    assert lt["n"].sum() == 100
    assert lt["cum_gain"].iloc[-1] == pytest.approx(1.0)
    assert lt["decile"].iloc[0] == 1


def test_lift_perfect_ranking():
    y = np.r_[np.ones(10), np.zeros(90)]
    p = np.r_[np.full(10, 0.9), np.full(90, 0.1)]
    lt = lift_table(y, p)
    assert lt.loc[0, "lift"] == pytest.approx(10.0)
    assert lt.loc[0, "cum_gain"] == pytest.approx(1.0)


def test_classification_metrics_regression():
    m = classification_metrics(Y, P)
    assert m["roc_auc"] == pytest.approx(21 / 25)  # hand-computed: 21 of 25 pos-neg pairs ordered
    assert m["pr_auc"] == pytest.approx(0.885)
    assert m["base_rate"] == pytest.approx(0.5)
    assert m["brier"] == pytest.approx(np.mean((P - Y) ** 2))
    assert 0 <= m["ece"] <= 1
    assert {"calib_slope", "calib_intercept", "mean_pred"} <= set(m)


def test_ece_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01


def test_calibration_slope_of_calibrated_scores_is_one():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.9, size=100_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.05)


def test_calibration_slope_detects_overconfidence():
    rng = np.random.default_rng(2)
    true = rng.uniform(0.2, 0.6, size=50_000)
    y = (rng.uniform(size=true.size) < true).astype(int)
    logit = np.log(true / (1 - true))
    over = 1 / (1 + np.exp(-3 * (logit - logit.mean()) - logit.mean()))
    slope, _ = calibration_slope_intercept(y, over)
    assert slope < 0.6


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=600)
    y = (rng.uniform(size=600) < p).astype(int)
    lo, hi = bootstrap_ci(y, p, roc_auc_score, n_boot=200, seed=1)
    assert lo < roc_auc_score(y, p) < hi
    assert (lo, hi) == bootstrap_ci(y, p, roc_auc_score, n_boot=200, seed=1)  # deterministic


def test_paired_bootstrap_identical_scores_have_zero_difference():
    rng = np.random.default_rng(4)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < p).astype(int)
    r = paired_bootstrap_diff(y, p, p, roc_auc_score, n_boot=100, seed=1)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0


def test_paired_bootstrap_detects_better_scorer():
    rng = np.random.default_rng(5)
    y = (rng.uniform(size=2000) < 0.3).astype(int)
    good = y * 0.5 + rng.uniform(size=2000)
    bad = rng.uniform(size=2000)
    r = paired_bootstrap_diff(y, good, bad, roc_auc_score, n_boot=200, seed=1)
    assert r["diff"] > 0.2 and r["lo"] > 0


@pytest.mark.parametrize("k,n", [(0, 50), (5, 50), (50, 50), (2, 801)])
def test_wilson_interval_is_valid(k, n):
    lo, hi = wilson_ci(k, n)
    assert 0 <= lo <= k / n <= hi <= 1
    assert hi > lo
