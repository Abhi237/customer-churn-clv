"""Data-contract tests for the SQL model: schema, nulls, grain uniqueness and explicit cleaning rules."""
import pytest

EXPECTED_SCHEMAS = {
    "fct_baskets": {"household_id", "basket_id", "store_id", "basket_date", "n_lines", "spend", "shelf_value",
                    "retail_disc", "coupon_disc", "fresh_spend", "has_coupon"},
    "fct_days": {"household_id", "visit_date", "n_baskets", "spend"},
    "dim_households": {"household_id", "first_purchase_date", "entry_cohort", "n_baskets", "lifetime_spend",
                       "has_demographics", "income", "age"},
    "cohort_retention": {"entry_cohort", "months_since_entry", "cohort_size", "active_households", "retention_rate"},
    "household_rfm": {"household_id", "recency_days", "frequency", "monetary", "r_score", "f_score", "m_score",
                      "rfm_cell", "rfm_segment"},
    "survival_households": {"household_id", "event", "duration_days", "baseline_trips", "baseline_avg_basket_spend"},
    "campaign_exposure": {"household_id", "campaign_id", "campaign_type", "start_date"},
    "dim_campaigns": {"campaign_id", "n_households_targeted", "n_redemptions", "redemption_rate_households"},
}
GRAINS = {
    "fct_baskets": ["household_id", "basket_id"],
    "fct_days": ["household_id", "visit_date"],
    "dim_households": ["household_id"],
    "cohort_retention": ["entry_cohort", "months_since_entry"],
    "household_rfm": ["household_id"],
    "survival_households": ["household_id"],
    "household_daily_rolling": ["household_id", "visit_date"],
    "dim_campaigns": ["campaign_id"],
}


def cols(con, t):
    return {r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()}


@pytest.mark.parametrize("table,expected", EXPECTED_SCHEMAS.items())
def test_schema(con, table, expected):
    assert expected <= cols(con, table)


@pytest.mark.parametrize("table,keys", GRAINS.items())
def test_grain_unique(con, table, keys):
    k = ", ".join(keys)
    dup = con.execute(f"SELECT COUNT(*) FROM (SELECT {k} FROM {table} GROUP BY {k} HAVING COUNT(*) > 1)").fetchone()[0]
    assert dup == 0


@pytest.mark.parametrize("table,col", [("fct_baskets", "household_id"), ("fct_baskets", "basket_date"),
                                       ("fct_baskets", "spend"), ("fct_days", "spend"),
                                       ("dim_households", "entry_cohort"), ("survival_households", "duration_days")])
def test_not_null(con, table, col):
    assert con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL").fetchone()[0] == 0


def test_every_basket_is_a_positive_purchase(con):
    assert con.execute("SELECT MIN(spend) FROM fct_baskets").fetchone()[0] > 0


def test_retention_bounds(con):
    lo, hi = con.execute("SELECT MIN(retention_rate), MAX(retention_rate) FROM cohort_retention").fetchone()
    assert 0 < lo and hi <= 1
    m0 = con.execute("SELECT MIN(retention_rate) FROM cohort_retention WHERE months_since_entry = 0").fetchone()
    assert m0[0] == 1.0


def test_dim_spend_reconciles(con):
    a = con.execute("SELECT SUM(lifetime_spend) FROM dim_households").fetchone()[0]
    b = con.execute("SELECT SUM(spend) FROM fct_baskets").fetchone()[0]
    assert float(a) == pytest.approx(float(b))
    d = con.execute("SELECT SUM(spend) FROM fct_days").fetchone()[0]
    assert float(d) == pytest.approx(float(b))


def test_rolling_windows_monotone(con):
    bad = con.execute("SELECT COUNT(*) FROM household_daily_rolling "
                      "WHERE NOT (trips_30d <= trips_60d AND trips_60d <= trips_90d)").fetchone()[0]
    assert bad == 0


def test_campaign_redemptions_deduplicated_and_linked(con):
    n_bad = con.execute("SELECT COUNT(*) FROM coupon_redemptions_clean r LEFT JOIN dim_households h "
                        "USING (household_id) WHERE h.household_id IS NULL").fetchone()[0]
    assert n_bad == 0


# ---- explicit cleaning rules on handcrafted edge cases (tests/fixtures/edge_transactions.csv)
def line_types(c, basket):
    return {r[0] for r in c.execute(f"SELECT line_type FROM fct_lines WHERE basket_id = '{basket}'").fetchall()}


def test_edge_empty_and_inconsistent_lines_dropped(edge_con):
    assert edge_con.execute("SELECT COUNT(*) FROM fct_lines WHERE basket_id = 'b1'").fetchone()[0] == 2  # empty gone
    assert line_types(edge_con, "b2") == set()   # zero quantity with positive sales
    assert line_types(edge_con, "b3") == set()   # zero price, no discount


def test_edge_coupon_only_kept_but_not_a_purchase(edge_con):
    assert line_types(edge_con, "b4") == {"coupon_only"}
    assert edge_con.execute("SELECT COUNT(*) FROM fct_baskets WHERE basket_id = 'b4'").fetchone()[0] == 0


def test_edge_free_item_kept_without_spend(edge_con):
    assert line_types(edge_con, "b5") == {"free_item", "merchandise"}
    spend = edge_con.execute("SELECT spend FROM fct_baskets WHERE basket_id = 'b5'").fetchone()[0]
    assert float(spend) == pytest.approx(4.0)


def test_edge_quantity_outlier_keeps_dollars_not_units(edge_con):
    r = edge_con.execute("SELECT units_clean, is_quantity_outlier, sales_value FROM fct_lines "
                         "WHERE basket_id = 'b6'").fetchone()
    assert r[0] is None and r[1] is True and float(r[2]) == pytest.approx(12.0)
    assert float(edge_con.execute("SELECT spend FROM fct_baskets WHERE basket_id = 'b6'").fetchone()[0]) == 12.0


def test_edge_high_value_line_flagged_not_dropped(edge_con):
    flagged = edge_con.execute("SELECT is_high_value_line FROM fct_lines WHERE basket_id = 'b7'").fetchone()[0]
    assert flagged is True


def test_edge_returns_kept_apart(edge_con):
    amt = edge_con.execute("SELECT returned_amount FROM fct_returns WHERE basket_id = 'b8'").fetchone()[0]
    assert float(amt) == pytest.approx(3.0)
    assert edge_con.execute("SELECT COUNT(*) FROM fct_baskets WHERE basket_id = 'b8'").fetchone()[0] == 0


def test_edge_unknown_product_gets_unknown_group(edge_con):
    g = edge_con.execute("SELECT dept_group FROM fct_lines WHERE product_id = 'p_missing'").fetchone()[0]
    assert g == "unknown"


def test_edge_shelf_value_adds_back_discounts(edge_con):
    v = edge_con.execute("SELECT shelf_value FROM fct_baskets WHERE basket_id = 'b1'").fetchone()[0]
    assert float(v) == pytest.approx(5.0 + 1.0 + 3.0)   # sales + retail discount (coupon is not added back)


def test_edge_dq_report_counts(edge_con):
    dq = dict(edge_con.execute("SELECT * FROM dq_report").fetchall())
    assert dq["total_lines"] == 13
    assert dq["empty_lines_no_value_no_discount"] == 1
    assert dq["zero_quantity_positive_sales"] == 1
    assert dq["zero_price_no_discount_lines"] == 1
    assert dq["coupon_only_lines"] == 1
    assert dq["free_item_lines"] == 1
    assert dq["negative_quantity_lines"] == 1
    assert dq["quantity_above_max_lines"] == 1
    assert dq["distinct_households"] == 2


def test_edge_waterfall_reconciles(edge_con):
    wf = dict(edge_con.execute("SELECT * FROM dq_cleaning_waterfall").fetchall())
    kept = wf["kept lines (merchandise, free items, coupon-only, returns)"]
    dropped = sum(v for k, v in wf.items() if k.startswith("dropped"))
    assert wf["raw lines"] - dropped == kept
