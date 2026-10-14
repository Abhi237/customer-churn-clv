"""Regression tests for metric implementations (fixed inputs -> fixed outputs)."""
import numpy as np
import pytest

from churn_clv.metrics import classification_metrics, expected_calibration_error, lift_table

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


def test_ece_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
