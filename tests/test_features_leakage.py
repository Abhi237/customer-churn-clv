"""Point-in-time correctness: features never use post-cutoff rows; label uses only the horizon window."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from churn_clv.db import run_sql_model
from churn_clv.features import FEATURE_COLUMNS, build_snapshot

CUTOFF = "2011-03-01"


def test_snapshot_grain_and_nulls(con):
    s = build_snapshot(con, CUTOFF)
    assert s["customer_id"].is_unique
    assert not s[FEATURE_COLUMNS].isna().any().any()
    assert set(s["churned"].unique()) <= {0, 1}
    assert (s["recency_days"] >= 1).all()


def test_population_active_in_lookback(con):
    s = build_snapshot(con, CUTOFF, lookback_days=365)
    assert (s["recency_days"] <= 365).all()


def test_label_matches_definition(con):
    s = build_snapshot(con, CUTOFF, horizon_days=90)
    fut = con.execute("SELECT DISTINCT customer_id FROM fct_invoices WHERE invoice_date >= DATE '2011-03-01' "
                      "AND invoice_date < DATE '2011-03-01' + INTERVAL 90 DAY").df()["customer_id"]
    expected = (~s["customer_id"].isin(set(fut))).astype(int)
    assert (s["churned"].to_numpy() == expected.to_numpy()).all()


def test_no_leakage_post_cutoff_rows(raw_df, cfg):
    """Perturb every row on/after the cutoff: features must be byte-identical, labels may change."""
    base = duckdb.connect(":memory:")
    run_sql_model(base, source=raw_df, cfg=cfg, verbose=False)
    s1 = build_snapshot(base, CUTOFF)

    pert = raw_df.copy()
    post = pert["invoice_date"] >= pd.Timestamp(CUTOFF)
    pert.loc[post, "quantity"] = pert.loc[post, "quantity"].abs() * 7
    pert.loc[post, "price"] = pert.loc[post, "price"] * 3
    rng = np.random.default_rng(0)
    drop = post & (rng.random(len(pert)) < 0.5)
    pert = pert[~drop]
    other = duckdb.connect(":memory:")
    run_sql_model(other, source=pert, cfg=cfg, verbose=False)
    s2 = build_snapshot(other, CUTOFF)

    pd.testing.assert_frame_equal(s1[["customer_id"] + FEATURE_COLUMNS].reset_index(drop=True),
                                  s2[["customer_id"] + FEATURE_COLUMNS].reset_index(drop=True))
    assert not s1["future_revenue"].equals(s2["future_revenue"])


@pytest.mark.parametrize("cutoff", ["2010-09-01", "2011-06-01"])
def test_features_ignore_cutoff_day(con, cutoff):
    """Rows dated exactly on the cutoff belong to the label window, not the features."""
    s = build_snapshot(con, cutoff)
    last = con.execute(f"SELECT customer_id, MAX(invoice_date) d FROM fct_invoices "
                       f"WHERE invoice_date < DATE '{cutoff}' GROUP BY 1").df().set_index("customer_id")["d"]
    rec = (pd.Timestamp(cutoff) - pd.to_datetime(last.loc[s["customer_id"]])).dt.days.to_numpy()
    assert (rec == s["recency_days"].to_numpy()).all()
