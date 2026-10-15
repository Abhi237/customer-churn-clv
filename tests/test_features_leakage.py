"""Point-in-time correctness: features never use post-cutoff rows; the label uses only the horizon window."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from churn_clv.db import run_sql_model
from churn_clv.features import DESCRIPTIVE_COLUMNS, FEATURE_COLUMNS, FEATURE_GROUPS, build_snapshot

CUTOFF = "2017-06-04"


def test_snapshot_grain_and_nulls(con):
    s = build_snapshot(con, CUTOFF)
    assert s["household_id"].is_unique
    assert not s[FEATURE_COLUMNS].isna().any().any()
    assert set(s["churned"].unique()) <= {0, 1}
    assert (s["recency_days"] >= 1).all()


def test_population_active_in_lookback(con):
    s = build_snapshot(con, CUTOFF, lookback_days=90)
    assert (s["recency_days"] <= 90).all()


def test_label_matches_definition(con):
    s = build_snapshot(con, CUTOFF, horizon_days=90)
    fut = con.execute("SELECT DISTINCT household_id FROM fct_days WHERE visit_date >= DATE '2017-06-04' "
                      "AND visit_date < DATE '2017-06-04' + INTERVAL 90 DAY").df()["household_id"]
    expected = (~s["household_id"].isin(set(fut))).astype(int)
    assert (s["churned"].to_numpy() == expected.to_numpy()).all()


def test_no_leakage_post_cutoff_rows(raw_tables, cfg):
    """Perturb every row on/after the cutoff: features must be identical, labels may change."""
    base = duckdb.connect(":memory:")
    run_sql_model(base, source=raw_tables, cfg=cfg, verbose=False)
    s1 = build_snapshot(base, CUTOFF)

    tx = raw_tables["transactions"].copy()
    post = tx["transaction_timestamp"] >= pd.Timestamp(CUTOFF)
    tx.loc[post, "quantity"] = tx.loc[post, "quantity"].abs() * 7
    tx.loc[post, "sales_value"] = tx.loc[post, "sales_value"] * 3
    rng = np.random.default_rng(0)
    tx = tx[~(post & (rng.random(len(tx)) < 0.5))]
    # also move every campaign start and redemption that is on/after the cutoff
    desc = raw_tables["campaign_descriptions"].copy()
    red = raw_tables["coupon_redemptions"].copy()
    red = red[pd.to_datetime(red["redemption_date"]) < pd.Timestamp(CUTOFF)]
    other = duckdb.connect(":memory:")
    run_sql_model(other, source={**raw_tables, "transactions": tx, "coupon_redemptions": red,
                                 "campaign_descriptions": desc}, cfg=cfg, verbose=False)
    s2 = build_snapshot(other, CUTOFF)

    pd.testing.assert_frame_equal(s1[["household_id"] + FEATURE_COLUMNS].reset_index(drop=True),
                                  s2[["household_id"] + FEATURE_COLUMNS].reset_index(drop=True))
    assert not s1["future_spend"].equals(s2["future_spend"])


@pytest.mark.parametrize("cutoff", ["2017-04-23", "2017-07-02"])
def test_features_ignore_cutoff_day(con, cutoff):
    """Rows dated exactly on the cutoff belong to the label window, not the features."""
    s = build_snapshot(con, cutoff)
    last = con.execute(f"SELECT household_id, MAX(visit_date) d FROM fct_days "
                       f"WHERE visit_date < DATE '{cutoff}' GROUP BY 1").df().set_index("household_id")["d"]
    rec = (pd.Timestamp(cutoff) - pd.to_datetime(last.loc[s["household_id"]])).dt.days.to_numpy()
    assert (rec == s["recency_days"].to_numpy()).all()


def test_redemptions_only_before_cutoff(con):
    s = build_snapshot(con, CUTOFF)
    n = con.execute("SELECT COUNT(*) FROM coupon_redemptions_clean WHERE redemption_date < DATE '2017-06-04' "
                    "AND redemption_date >= DATE '2017-06-04' - INTERVAL 90 DAY").fetchone()[0]
    assert int(s["redemptions_90d"].sum()) == n


def test_shares_and_rates_are_bounded(con):
    s = build_snapshot(con, CUTOFF)
    share_cols = [c for c in FEATURE_COLUMNS if c.startswith("share_")] + ["retail_disc_share", "coupon_disc_share",
                                                                           "coupon_basket_share", "top_store_share",
                                                                           "active_week_share"]
    assert ((s[share_cols] >= 0) & (s[share_cols] <= 1.0 + 1e-9)).all().all()
    dept = s[["share_fresh", "share_grocery", "share_fuel", "share_health_beauty"]].sum(axis=1)
    assert (dept <= 1 + 1e-5).all()
    assert (s["trips_28d"] <= s["trips_42d"]).all()


def test_no_history_growing_features_in_model_inputs():
    """Design rule: cumulative features encode calendar time in a 12-month panel and must not be model inputs."""
    assert not set(DESCRIPTIVE_COLUMNS) & set(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == sum(len(v) for v in FEATURE_GROUPS.values())


def test_feature_distribution_stable_across_cutoffs(con):
    """Shift guard: the median of fixed-window features should not trend with the amount of history."""
    early = build_snapshot(con, "2017-03-12")["trips_42d"].median()
    late = build_snapshot(con, "2017-07-30")["trips_42d"].median()
    assert abs(early - late) <= 4
