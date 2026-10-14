"""Data-contract tests for the SQL model: schema, nulls, grain uniqueness, cleaning rules."""
import pytest

EXPECTED_SCHEMAS = {
    "fct_invoices": {"customer_id", "invoice", "invoice_ts", "invoice_date", "country", "n_lines", "n_products",
                     "units", "revenue"},
    "dim_customers": {"customer_id", "first_purchase_date", "last_purchase_date", "home_country",
                      "acquisition_cohort", "n_invoices", "lifetime_revenue"},
    "cohort_retention": {"acquisition_cohort", "months_since_acquisition", "cohort_size", "active_customers",
                         "retention_rate"},
    "customer_rfm": {"customer_id", "recency_days", "frequency", "monetary", "r_score", "f_score", "m_score",
                     "rfm_cell", "rfm_segment"},
    "survival_customers": {"customer_id", "event", "duration_days", "first_order_value", "country_group"},
}
GRAINS = {
    "fct_invoices": ["customer_id", "invoice"],
    "dim_customers": ["customer_id"],
    "cohort_retention": ["acquisition_cohort", "months_since_acquisition"],
    "customer_rfm": ["customer_id"],
    "survival_customers": ["customer_id"],
    "customer_daily_rolling": ["customer_id", "invoice_date"],
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


@pytest.mark.parametrize("table,col", [("fct_invoices", "customer_id"), ("fct_invoices", "invoice_date"),
                                       ("fct_invoices", "revenue"), ("dim_customers", "home_country"),
                                       ("survival_customers", "duration_days")])
def test_not_null(con, table, col):
    assert con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL").fetchone()[0] == 0


def test_sales_positive(con):
    assert con.execute("SELECT MIN(revenue) FROM fct_invoices").fetchone()[0] > 0
    assert con.execute("SELECT COUNT(*) FROM fct_invoices WHERE invoice LIKE 'C%'").fetchone()[0] == 0


def test_retention_bounds(con):
    lo, hi = con.execute("SELECT MIN(retention_rate), MAX(retention_rate) FROM cohort_retention").fetchone()
    assert 0 < lo and hi <= 1
    m0 = con.execute("SELECT MIN(retention_rate) FROM cohort_retention WHERE months_since_acquisition = 0").fetchone()
    assert m0[0] == 1.0


def test_dim_revenue_reconciles(con):
    a = con.execute("SELECT SUM(lifetime_revenue) FROM dim_customers").fetchone()[0]
    b = con.execute("SELECT SUM(revenue) FROM fct_invoices").fetchone()[0]
    assert a == pytest.approx(b)


def test_rolling_windows_monotone(con):
    bad = con.execute("SELECT COUNT(*) FROM customer_daily_rolling "
                      "WHERE NOT (invoices_30d <= invoices_60d AND invoices_60d <= invoices_90d)").fetchone()[0]
    assert bad == 0


# ---- explicit cleaning rules on handcrafted edge cases
def test_edge_duplicates_removed(edge_con):
    n = edge_con.execute("SELECT COUNT(*) FROM fct_lines WHERE invoice = '900001'").fetchone()[0]
    assert n == 1  # duplicate line dropped, POST line excluded


def test_edge_exclusions(edge_con):
    codes = {r[0] for r in edge_con.execute("SELECT DISTINCT stock_code FROM fct_lines").fetchall()}
    assert not codes & {"POST", "M", "BANK CHARGES", "B"}
    invs = {r[0] for r in edge_con.execute("SELECT DISTINCT invoice FROM fct_lines").fetchall()}
    assert "900003" not in invs  # missing customer id
    assert "900004" not in invs  # zero price
    assert "900005" not in invs  # quantity outlier
    assert "900010" not in invs  # negative qty on non-cancellation invoice
    assert "C900002" in invs     # cancellation kept as return
    t = edge_con.execute("SELECT line_type FROM fct_lines WHERE invoice = 'C900002'").fetchone()[0]
    assert t == "cancellation"


def test_edge_returns(edge_con):
    amt = edge_con.execute("SELECT returned_amount FROM fct_returns WHERE invoice = 'C900002'").fetchone()[0]
    assert float(amt) == pytest.approx(5.10)


def test_dq_report_counts(edge_con):
    dq = dict(edge_con.execute("SELECT * FROM dq_report").fetchall())
    assert dq["total_rows"] == 12
    assert dq["exact_duplicate_rows"] == 1
    assert dq["cancellation_lines_invoice_C"] == 1
    assert dq["missing_customer_id"] == 2
    assert dq["abs_quantity_outlier"] == 1
