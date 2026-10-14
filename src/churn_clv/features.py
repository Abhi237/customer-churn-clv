"""Point-in-time feature and churn-proxy label construction.

Leakage rule: every feature uses ONLY rows with invoice_date < cutoff. The label uses ONLY rows in
[cutoff, cutoff + horizon). The churn label is a PROXY (no purchase within the horizon), not an
observed cancellation: this is a non-contractual retailer.
"""
from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from .db import as_date

FEATURE_COLUMNS = [
    "recency_days", "tenure_days", "frequency", "monetary_total", "avg_order_value",
    "invoices_30d", "invoices_90d", "invoices_180d", "revenue_90d", "revenue_180d",
    "n_distinct_products", "avg_products_per_invoice", "mean_interpurchase_days",
    "std_interpurchase_days", "active_months_12m", "n_returns_365d", "return_ratio_365d",
    "recency_to_gap_ratio", "is_uk",
]

FEATURE_SQL = """
WITH hist AS (                         -- strictly pre-cutoff history
    SELECT * FROM fct_invoices WHERE invoice_date < DATE '{cutoff}'
),
pop AS (                               -- population: active in the lookback window
    SELECT customer_id FROM hist GROUP BY customer_id
    HAVING MAX(invoice_date) >= DATE '{cutoff}' - INTERVAL {lookback} DAY
),
gaps AS (
    SELECT customer_id,
           invoice_date - LAG(invoice_date) OVER (PARTITION BY customer_id ORDER BY invoice_date) AS gap
    FROM (SELECT DISTINCT customer_id, invoice_date FROM hist) d
),
gap_stats AS (
    SELECT customer_id, AVG(gap) AS mean_gap, STDDEV_SAMP(gap) AS std_gap FROM gaps GROUP BY customer_id
),
prod AS (
    SELECT l.customer_id, COUNT(DISTINCT l.stock_code) AS n_distinct_products
    FROM fct_lines l WHERE l.line_type = 'sale' AND l.invoice_date < DATE '{cutoff}' GROUP BY 1
),
ret AS (
    SELECT customer_id, COUNT(*) AS n_returns, SUM(returned_amount) AS returned
    FROM fct_returns
    WHERE invoice_date < DATE '{cutoff}' AND invoice_date >= DATE '{cutoff}' - INTERVAL 365 DAY
    GROUP BY 1
),
agg AS (
    SELECT h.customer_id,
           DATE '{cutoff}' - MAX(h.invoice_date)                    AS recency_days,
           DATE '{cutoff}' - MIN(h.invoice_date)                    AS tenure_days,
           COUNT(*)                                                 AS frequency,
           SUM(h.revenue)                                           AS monetary_total,
           AVG(h.revenue)                                           AS avg_order_value,
           COUNT(*) FILTER (WHERE h.invoice_date >= DATE '{cutoff}' - INTERVAL 30 DAY)  AS invoices_30d,
           COUNT(*) FILTER (WHERE h.invoice_date >= DATE '{cutoff}' - INTERVAL 90 DAY)  AS invoices_90d,
           COUNT(*) FILTER (WHERE h.invoice_date >= DATE '{cutoff}' - INTERVAL 180 DAY) AS invoices_180d,
           COALESCE(SUM(h.revenue) FILTER (WHERE h.invoice_date >= DATE '{cutoff}' - INTERVAL 90 DAY), 0)  AS revenue_90d,
           COALESCE(SUM(h.revenue) FILTER (WHERE h.invoice_date >= DATE '{cutoff}' - INTERVAL 180 DAY), 0) AS revenue_180d,
           AVG(h.n_products)                                        AS avg_products_per_invoice,
           COUNT(DISTINCT DATE_TRUNC('month', h.invoice_date)) FILTER (
               WHERE h.invoice_date >= DATE '{cutoff}' - INTERVAL 365 DAY)                 AS active_months_12m,
           MAX(CASE WHEN h.country = 'United Kingdom' THEN 1 ELSE 0 END)                    AS is_uk
    FROM hist h JOIN pop USING (customer_id)
    GROUP BY h.customer_id
),
future AS (                             -- label window only
    SELECT customer_id, COUNT(*) AS future_invoices, SUM(revenue) AS future_revenue
    FROM fct_invoices
    WHERE invoice_date >= DATE '{cutoff}' AND invoice_date < DATE '{cutoff}' + INTERVAL {horizon} DAY
    GROUP BY 1
)
SELECT DATE '{cutoff}'                                       AS cutoff,
       a.*,
       COALESCE(p.n_distinct_products, 0)                    AS n_distinct_products,
       COALESCE(g.mean_gap, a.tenure_days)                   AS mean_interpurchase_days,
       COALESCE(g.std_gap, 0)                                AS std_interpurchase_days,
       COALESCE(r.n_returns, 0)                              AS n_returns_365d,
       COALESCE(r.returned, 0) / NULLIF(a.monetary_total, 0) AS return_ratio_365d,
       a.recency_days / GREATEST(COALESCE(g.mean_gap, a.tenure_days, 1), 1) AS recency_to_gap_ratio,
       COALESCE(f.future_invoices, 0)                        AS future_invoices,
       COALESCE(f.future_revenue, 0)                         AS future_revenue,
       CASE WHEN f.customer_id IS NULL THEN 1 ELSE 0 END     AS churned
FROM agg a
LEFT JOIN gap_stats g USING (customer_id)
LEFT JOIN prod p USING (customer_id)
LEFT JOIN ret r USING (customer_id)
LEFT JOIN future f USING (customer_id)
ORDER BY a.customer_id
"""


def build_snapshot(con: duckdb.DuckDBPyConnection, cutoff: str | date, horizon_days: int = 90,
                   lookback_days: int = 365, with_label: bool = True) -> pd.DataFrame:
    c = as_date(cutoff)
    sql = FEATURE_SQL.format(cutoff=c.isoformat(), horizon=int(horizon_days), lookback=int(lookback_days))
    df = con.execute(sql).df()
    df["return_ratio_365d"] = df["return_ratio_365d"].fillna(0.0)
    for col in FEATURE_COLUMNS:
        df[col] = df[col].astype(float)
    if not with_label:
        df = df.drop(columns=["future_invoices", "future_revenue", "churned"])
    return df


def label_window_complete(con: duckdb.DuckDBPyConnection, cutoff: str | date, horizon_days: int) -> bool:
    data_end = con.execute("SELECT MAX(invoice_date) FROM fct_invoices").fetchone()[0]
    return (as_date(cutoff).toordinal() + horizon_days - 1) <= data_end.toordinal()


def build_panel(con: duckdb.DuckDBPyConnection, cutoffs: list[str], horizon_days: int,
                lookback_days: int) -> pd.DataFrame:
    return pd.concat([build_snapshot(con, c, horizon_days, lookback_days) for c in cutoffs], ignore_index=True)
