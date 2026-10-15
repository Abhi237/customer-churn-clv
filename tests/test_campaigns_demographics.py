"""Campaign exposure, observational comparison helpers and the partial-coverage demographics breakout."""
import numpy as np
import pandas as pd
import pytest

from churn_clv.campaigns import (
    BALANCE_FEATURES,
    adjusted_difference,
    campaign_type_summary,
    outcome_by_flag,
    standardized_mean_differences,
    window_exposure,
)
from churn_clv.demographics import DIMENSIONS, INCOME_BANDS, INCOME_ORDER, breakout, coverage_comparison


def test_window_exposure_counts_only_campaigns_starting_in_window(con):
    w = window_exposure(con, "2017-06-04", 90)
    assert w["household_id"].is_unique and len(w) == con.execute("SELECT COUNT(*) FROM dim_households").fetchone()[0]
    # synthetic campaigns starting in [2017-06-04, 2017-09-02): 4 (2017-07-01) and 5 (2017-09-01)
    n = con.execute("SELECT COUNT(DISTINCT household_id) FROM campaign_exposure WHERE campaign_id IN ('4', '5')"
                    ).fetchone()[0]
    assert int((w["n_window_campaigns"] > 0).sum()) == n


def test_window_exposure_prior_history_is_strictly_before_cutoff(con):
    w = window_exposure(con, "2017-06-04", 90)
    prior = con.execute("SELECT COUNT(*) FROM campaign_exposure WHERE start_date < DATE '2017-06-04'").fetchone()[0]
    assert int(w["n_prior_campaigns"].sum()) == prior


def test_window_exposure_horizon_is_respected(con):
    short = window_exposure(con, "2017-06-04", 10)
    assert short["n_window_campaigns"].sum() == 0   # nothing starts within 10 days of that cutoff


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    tier = rng.choice(["Low risk", "Medium risk", "High risk"], n, p=[0.4, 0.3, 0.3])
    base = pd.Series(tier).map({"Low risk": 0.01, "Medium risk": 0.05, "High risk": 0.2}).to_numpy()
    # exposure is concentrated among low-risk households, with NO true effect on churn
    expo = rng.uniform(size=n) < pd.Series(tier).map({"Low risk": 0.8, "Medium risk": 0.4, "High risk": 0.1}).to_numpy()
    churn = (rng.uniform(size=n) < base).astype(int)
    return pd.DataFrame({"risk_segment": tier, "flag": expo, "churned": churn, "future_spend": rng.gamma(2, 50, n)})


def test_naive_gap_is_confounded_but_tier_standardised_gap_is_not():
    df = _toy()
    adj = adjusted_difference(df, "flag")
    assert adj["naive_diff"] < -0.05                 # exposed households look much better...
    assert abs(adj["tier_standardised_diff"]) < 0.02  # ...but the gap disappears within risk tiers


def test_outcome_by_flag_structure():
    out = outcome_by_flag(_toy(), "flag")
    assert set(out["risk_segment"]) == {"All scored", "Low risk", "Medium risk", "High risk"}
    assert out["n"].sum() == 2 * 4000
    assert ((out["ci_lo"] <= out["churn_rate"]) & (out["churn_rate"] <= out["ci_hi"])).all()


def test_standardized_mean_differences_sign_and_scale():
    rng = np.random.default_rng(1)
    flag = rng.uniform(size=3000) < 0.5
    df = pd.DataFrame({f: rng.normal(size=3000) for f in BALANCE_FEATURES})
    df["flag"] = flag
    df.loc[flag, "trips_per_week"] += 1.0
    smd = standardized_mean_differences(df, "flag").set_index("feature")["smd"]
    assert smd["trips_per_week"] == pytest.approx(1.0, abs=0.1)
    assert smd.drop("trips_per_week").abs().max() < 0.15


def test_campaign_type_summary_arithmetic():
    dc = pd.DataFrame({"campaign_id": [1, 2, 3], "campaign_type": ["Type A", "Type A", "Type B"],
                       "n_households_targeted": [100, 300, 50], "n_redemptions": [10, 40, 2],
                       "n_redeeming_households": [8, 30, 2]})
    s = campaign_type_summary(dc).set_index("campaign_type")
    assert s.loc["Type A", "households_targeted_total"] == 400
    assert s.loc["Type A", "redeeming_per_targeted"] == pytest.approx(38 / 400)


# --------------------------------------------------------------------------------------- demographics
def _scored(n=600, seed=0):
    rng = np.random.default_rng(seed)
    inc = rng.choice(list(INCOME_BANDS), n)
    return pd.DataFrame({
        "household_id": range(n), "has_demographics": rng.uniform(size=n) < 0.4,
        "income": inc, "age": rng.choice(["19-24", "25-34", "35-44", "45-54", "55-64", "65+"], n),
        "household_comp": rng.choice(["1 Adult No Kids", "1 Adult Kids", "2 Adults No Kids", "2 Adults Kids"], n),
        "churned": (rng.uniform(size=n) < 0.04).astype(int), "churn_prob": rng.uniform(0, 0.1, n),
        "clv_12m": rng.gamma(2, 500, n), "monetary_total": rng.gamma(2, 400, n), "frequency": rng.integers(1, 80, n),
        "trips_per_week": rng.uniform(0, 3, n), "spend_per_week": rng.uniform(0, 100, n),
        "avg_basket_spend": rng.uniform(10, 60, n), "coupon_disc_share": rng.uniform(0, 0.02, n),
        "retail_disc_share": rng.uniform(0, 0.3, n), "share_fresh": rng.uniform(0, 0.4, n)})


def test_income_bands_cover_every_raw_level():
    assert set(INCOME_BANDS.values()) == set(INCOME_ORDER)
    assert len(INCOME_BANDS) == 12


def test_breakout_counts_only_covered_households_and_flags_small_cells():
    df = _scored()
    tab = breakout(df, min_cell_size=30)
    covered = int(df["has_demographics"].sum())
    for dim in DIMENSIONS:
        assert tab.loc[tab["dimension"] == dim, "n_scored"].sum() == covered
    assert ((tab["ci_lo"] <= tab["churn_rate"]) & (tab["churn_rate"] <= tab["ci_hi"])).all()
    assert (tab.loc[tab["n_churned"] < 5, "reliable"] == False).all()  # noqa: E712


def test_coverage_comparison_has_both_groups_and_nan_auc_with_few_events():
    df = _scored()
    df.loc[df["has_demographics"], "churned"] = 0
    cov = coverage_comparison(df).set_index("group")
    assert set(cov.index) == {"With demographics", "Without demographics"}
    assert np.isnan(cov.loc["With demographics", "roc_auc_final_model"])
    assert cov["n_scored"].sum() == len(df)


def test_demographics_are_never_model_features():
    from churn_clv.features import FEATURE_COLUMNS
    assert not {"age", "income", "household_comp", "kids_count", "home_ownership", "marital_status"} & set(FEATURE_COLUMNS)


def test_synthetic_demographics_flag_matches_raw(con, raw_tables):
    n = con.execute("SELECT COUNT(*) FROM dim_households WHERE has_demographics").fetchone()[0]
    assert n == len(raw_tables["demographics"])
