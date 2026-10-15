"""Landmark survival dataset: invariants of the SQL design used by the R analysis."""
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def sv(con):
    return con.execute("SELECT * FROM survival_households").df()


def test_grain_and_event_coding(sv):
    assert sv["household_id"].is_unique
    assert set(sv["event"].unique()) <= {0, 1}
    assert sv["event"].sum() > 0


def test_durations_positive_and_bounded_by_data(sv, con):
    assert (sv["duration_days"] >= 1).all()
    end, lapse = con.execute("SELECT data_end, lapse_days FROM params").fetchone()
    max_follow = (pd.Timestamp(end) - pd.Timedelta(days=int(lapse)) - sv["landmark_date"].min()).days + 1
    assert sv["duration_days"].max() <= max_follow


def test_landmark_is_origin_plus_baseline(sv, con):
    b = con.execute("SELECT baseline_days FROM params").fetchone()[0]
    d = (pd.to_datetime(sv["landmark_date"]) - pd.to_datetime(sv["first_purchase_date"])).dt.days
    assert (d == b).all()


def test_baseline_covariates_use_only_baseline_window(sv, con):
    """baseline_trips counts distinct shopping days in [origin, origin + baseline_days): never beyond the landmark."""
    b = con.execute("SELECT baseline_days FROM params").fetchone()[0]
    assert (sv["baseline_trips"] <= b).all() and (sv["baseline_trips"] >= 1).all()


def test_no_household_enters_after_data_end_minus_lapse(sv, con):
    end, lapse = con.execute("SELECT data_end, lapse_days FROM params").fetchone()
    assert (pd.to_datetime(sv["landmark_date"]) <= pd.Timestamp(end) - pd.Timedelta(days=int(lapse))).all()


def test_exclusions_reconcile(con):
    ex = dict(con.execute("SELECT reason, n FROM survival_exclusions").fetchall())
    total = ex["households with purchases"]
    out = ex["excluded: landmark after data_end - lapse_days (no time at risk)"] + \
        ex["excluded: lapsed before the landmark"]
    assert total - out == ex["analysed households"]


def test_event_matches_first_gap_over_lapse_days(con):
    gaps = con.execute("""
        SELECT household_id, visit_date, LEAD(visit_date) OVER (PARTITION BY household_id ORDER BY visit_date) nxt
        FROM fct_days""").df()
    gaps["gap"] = (pd.to_datetime(gaps["nxt"]) - pd.to_datetime(gaps["visit_date"])).dt.days
    lapsed = set(gaps.loc[gaps["gap"] > 90, "household_id"])
    ev = con.execute("SELECT household_id FROM household_lapse WHERE lapse_date IS NOT NULL").df()["household_id"]
    assert lapsed <= set(ev)   # every observed >90-day gap yields a lapse (plus confirmable trailing gaps)
