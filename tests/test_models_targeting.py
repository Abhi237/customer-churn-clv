"""Unit tests for baselines, calibration, CLV and targeting logic."""
import numpy as np
import pandas as pd
import pytest
from lifetimes import BetaGeoFitter
from sklearn.metrics import roc_auc_score

from churn_clv.clv import expected_purchases, score_clv
from churn_clv.features import FEATURE_COLUMNS, build_snapshot
from churn_clv.models import Calibrator, RecencyRule, RFMHeuristic, make_lgbm, make_logreg
from churn_clv.targeting import assign_segments, expected_net_value, segment_matrix, sensitivity_grid


@pytest.fixture(scope="module")
def snap(con):
    return pd.concat([build_snapshot(con, c) for c in ["2017-02-12", "2017-03-12", "2017-04-09"]], ignore_index=True)


def test_recency_rule_monotone_on_data(snap):
    m = RecencyRule().fit(snap, snap["churned"])
    p = m.predict_proba(snap)
    assert p.min() >= 0 and p.max() <= 1
    assert np.corrcoef(snap["recency_days"], p)[0, 1] > 0


def test_rfm_heuristic_probabilities(snap):
    p = RFMHeuristic().fit(snap, snap["churned"]).predict_proba(snap)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()


@pytest.mark.parametrize("factory", [make_logreg, lambda: make_lgbm(params={"min_child_samples": 20})])
def test_models_fit_and_beat_chance(snap, factory):
    m = factory().fit(snap[FEATURE_COLUMNS], snap["churned"])
    p = m.predict_proba(snap[FEATURE_COLUMNS])[:, 1]
    assert roc_auc_score(snap["churned"], p) > 0.6


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrator_bounds_and_monotone(method):
    rng = np.random.default_rng(1)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p ** 2).astype(int)
    q = Calibrator(method).fit(p, y).transform(np.sort(p))
    assert (q >= 0).all() and (q <= 1).all()
    assert (np.diff(q) >= -1e-12).all()


def test_calibrator_none_is_identity():
    p = np.array([0.01, 0.2, 0.9])
    np.testing.assert_allclose(Calibrator("none").fit(p, np.array([0, 0, 1])).transform(p), p)


def test_isotonic_gives_exact_zeros_with_few_positives():
    """Why isotonic is not eligible with few positives: it emits probability 0 for large parts of the range."""
    rng = np.random.default_rng(3)
    p = rng.beta(1, 25, size=2300)
    y = (rng.uniform(size=p.size) < 0.04).astype(int)  # ~90 positives unrelated to score
    q = Calibrator("isotonic").fit(p, y).transform(p)
    assert q.min() >= 0
    platt = Calibrator("platt").fit(p, y).transform(p)
    assert platt.min() > 0


def test_isotonic_eligibility_threshold_is_configured(cfg):
    assert cfg["churn"]["min_positives_isotonic"] >= 500


def test_expected_net_value_formula():
    v = expected_net_value([0.5, 0.1], [1000, 100], uplift=0.1, margin=0.3, cost=5)
    assert v == pytest.approx([0.5 * 0.1 * 1000 * 0.3 - 5, 0.1 * 0.1 * 100 * 0.3 - 5])


def test_assumed_parameters_are_declared_in_config(cfg):
    t = cfg["targeting"]
    assert {"uplift", "gross_margin", "contact_cost_usd"} <= set(t)
    assert "ASSUMED" in (open("config/config.yaml").read())


def test_segments_and_sensitivity():
    df = pd.DataFrame({"household_id": range(300), "churn_prob": np.linspace(0, 1, 300),
                       "clv_12m": np.linspace(10, 3000, 300)[::-1]})
    s = assign_segments(df, [0, 1 / 3, 2 / 3, 1], ["L", "M", "H"], [0, 0.5, 0.8, 1], ["lo", "mid", "hi"])
    assert s["risk_segment"].value_counts().min() >= 99
    assert s["value_segment"].value_counts()["hi"] == pytest.approx(60, abs=1)
    g = sensitivity_grid(df, [0.1, 0.2], [0.3], [1, 10])
    assert len(g) == 4
    assert g.query("contact_cost_usd==1 and uplift==0.2").n_targeted.iloc[0] >= \
        g.query("contact_cost_usd==1 and uplift==0.1").n_targeted.iloc[0]
    assert g.query("contact_cost_usd==10 and uplift==0.1").n_targeted.iloc[0] <= \
        g.query("contact_cost_usd==1 and uplift==0.1").n_targeted.iloc[0]
    m = segment_matrix(s)
    assert m["households"].sum() == 300 and len(m) == 9


# ----------------------------------------------------------------------------------------------- CLV
@pytest.fixture(scope="module")
def days(con):
    return con.execute("SELECT household_id, visit_date, spend FROM fct_days").df()


def test_clv_scores_use_only_pre_cutoff(days):
    a = score_clv(days, "2017-07-01", horizon_months=6)
    d2 = days.copy()
    d2.loc[pd.to_datetime(d2.visit_date) >= "2017-07-01", "spend"] *= 10
    b = score_clv(d2, "2017-07-01", horizon_months=6)
    assert (a["clv_12m"] >= 0).all() and (a["expected_aov"] > 0).all()
    np.testing.assert_allclose(a["clv_12m"], b["clv_12m"])


def test_expected_purchases_matches_lifetimes_where_defined():
    bgf = BetaGeoFitter()
    bgf.params_ = pd.Series({"r": 1.4, "alpha": 9.0, "a": 0.4, "b": 1.6})
    x = np.array([1, 2, 5, 10, 25.0])
    T = np.array([180, 180, 200, 190, 210.0])
    rec = np.array([20, 80, 150, 170, 200.0])
    ref = bgf.conditional_expected_number_of_purchases_up_to_time(90, x, rec, T)
    ours = expected_purchases(bgf, 90, x, rec, T)
    np.testing.assert_allclose(ours, ref, rtol=1e-6)


@pytest.mark.filterwarnings("ignore:invalid value encountered in log")
def test_expected_purchases_finite_when_library_returns_nan():
    """a + b < 1 with zero repeat purchases makes lifetimes take the log of a negative number (NaN)."""
    bgf = BetaGeoFitter()
    bgf.params_ = pd.Series({"r": 1.4, "alpha": 9.4, "a": 0.005, "b": 0.955})
    x, rec, T = np.zeros(4), np.zeros(4), np.array([100, 150, 200, 250.0])
    ref = bgf.conditional_expected_number_of_purchases_up_to_time(90, x, rec, T)
    ours = expected_purchases(bgf, 90, x, rec, T)
    assert np.isnan(ref).any()
    assert np.isfinite(ours).all() and (ours >= 0).all()


def test_expected_purchases_increase_with_horizon():
    bgf = BetaGeoFitter()
    bgf.params_ = pd.Series({"r": 1.4, "alpha": 9.0, "a": 0.4, "b": 1.6})
    x, rec, T = np.array([5.0]), np.array([100.0]), np.array([200.0])
    assert expected_purchases(bgf, 180, x, rec, T)[0] > expected_purchases(bgf, 90, x, rec, T)[0] > 0
