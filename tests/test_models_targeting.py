"""Unit tests for baselines, calibration, CLV and targeting logic."""
import numpy as np
import pandas as pd
import pytest

from churn_clv.clv import score_clv
from churn_clv.features import FEATURE_COLUMNS, build_snapshot
from churn_clv.models import Calibrator, RecencyRule, RFMHeuristic, make_lgbm, make_logreg
from churn_clv.targeting import assign_segments, expected_net_value, sensitivity_grid


@pytest.fixture(scope="module")
def snap(con):
    return pd.concat([build_snapshot(con, c) for c in ["2010-09-01", "2010-12-01", "2011-03-01"]], ignore_index=True)


def test_recency_rule_monotone_on_data(snap):
    m = RecencyRule().fit(snap, snap["churned"])
    p = m.predict_proba(snap)
    assert p.min() >= 0 and p.max() <= 1
    assert np.corrcoef(snap["recency_days"], p)[0, 1] > 0


def test_rfm_heuristic_probabilities(snap):
    p = RFMHeuristic().fit(snap, snap["churned"]).predict_proba(snap)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()


def test_models_fit_and_beat_chance(snap):
    y = snap["churned"]
    for m in (make_logreg(), make_lgbm(params={"min_child_samples": 20})):
        m.fit(snap[FEATURE_COLUMNS], y)
        p = m.predict_proba(snap[FEATURE_COLUMNS])[:, 1]
        from sklearn.metrics import roc_auc_score
        assert roc_auc_score(y, p) > 0.6


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrator_bounds_and_monotone(method):
    rng = np.random.default_rng(1)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p ** 2).astype(int)
    c = Calibrator(method).fit(p, y)
    q = c.transform(np.sort(p))
    assert (q >= 0).all() and (q <= 1).all()
    assert (np.diff(q) >= -1e-12).all()


def test_expected_net_value_formula():
    v = expected_net_value([0.5, 0.1], [1000, 100], uplift=0.1, margin=0.3, cost=5)
    assert v == pytest.approx([0.5 * 0.1 * 1000 * 0.3 - 5, 0.1 * 0.1 * 100 * 0.3 - 5])


def test_segments_and_sensitivity():
    df = pd.DataFrame({"customer_id": range(300), "churn_prob": np.linspace(0, 1, 300),
                       "clv_12m": np.linspace(10, 3000, 300)[::-1]})
    s = assign_segments(df, [0, 1 / 3, 2 / 3, 1], ["L", "M", "H"], [0, 0.5, 0.8, 1], ["lo", "mid", "hi"])
    assert s["risk_segment"].value_counts().min() >= 99
    assert s["value_segment"].value_counts()["hi"] == pytest.approx(60, abs=1)
    g = sensitivity_grid(df, [0.1, 0.2], [0.3], [1, 10])
    assert len(g) == 4
    # more uplift never targets fewer customers; higher cost never targets more
    assert g.query("contact_cost_gbp==1 and uplift==0.2").n_targeted.iloc[0] >= \
        g.query("contact_cost_gbp==1 and uplift==0.1").n_targeted.iloc[0]
    assert g.query("contact_cost_gbp==10 and uplift==0.1").n_targeted.iloc[0] <= \
        g.query("contact_cost_gbp==1 and uplift==0.1").n_targeted.iloc[0]


def test_clv_scores_use_only_pre_cutoff(con):
    inv = con.execute("SELECT customer_id, invoice_date, revenue FROM fct_invoices").df()
    a = score_clv(inv, "2011-03-01", horizon_months=6)
    inv2 = inv.copy()
    inv2.loc[pd.to_datetime(inv2.invoice_date) >= "2011-03-01", "revenue"] *= 10
    b = score_clv(inv2, "2011-03-01", horizon_months=6)
    assert (a["clv_12m"] >= 0).all() and (a["expected_aov"] > 0).all()
    np.testing.assert_allclose(a["clv_12m"], b["clv_12m"])
