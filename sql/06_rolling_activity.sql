-- 06_rolling_activity.sql
-- Rolling 30/60/90-day activity per customer, evaluated at every purchase date
-- (RANGE window frames over dates; DuckDB/Postgres compatible).
CREATE OR REPLACE TABLE customer_daily_rolling AS
WITH daily AS (
    SELECT customer_id, invoice_date, COUNT(*) AS n_invoices, SUM(revenue) AS revenue
    FROM fct_invoices
    GROUP BY 1, 2
)
SELECT
    customer_id,
    invoice_date,
    n_invoices,
    revenue,
    SUM(n_invoices) OVER w30 AS invoices_30d,
    SUM(n_invoices) OVER w60 AS invoices_60d,
    SUM(n_invoices) OVER w90 AS invoices_90d,
    SUM(revenue)    OVER w30 AS revenue_30d,
    SUM(revenue)    OVER w60 AS revenue_60d,
    SUM(revenue)    OVER w90 AS revenue_90d,
    invoice_date - LAG(invoice_date) OVER (PARTITION BY customer_id ORDER BY invoice_date) AS days_since_prev_purchase
FROM daily
WINDOW
    w30 AS (PARTITION BY customer_id ORDER BY invoice_date RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND CURRENT ROW),
    w60 AS (PARTITION BY customer_id ORDER BY invoice_date RANGE BETWEEN INTERVAL 59 DAY PRECEDING AND CURRENT ROW),
    w90 AS (PARTITION BY customer_id ORDER BY invoice_date RANGE BETWEEN INTERVAL 89 DAY PRECEDING AND CURRENT ROW);

-- Monthly company-level active customers (30/60/90-day windows ending each month end).
CREATE OR REPLACE TABLE monthly_active_customers AS
WITH months AS (
    SELECT (DATE_TRUNC('month', d) + INTERVAL 1 MONTH - INTERVAL 1 DAY)::DATE AS month_end
    FROM generate_series(DATE '2009-12-01', DATE '2011-11-01', INTERVAL 1 MONTH) t(d)
)
SELECT m.month_end,
       COUNT(DISTINCT CASE WHEN i.invoice_date > m.month_end - 30 THEN i.customer_id END) AS active_30d,
       COUNT(DISTINCT CASE WHEN i.invoice_date > m.month_end - 60 THEN i.customer_id END) AS active_60d,
       COUNT(DISTINCT CASE WHEN i.invoice_date > m.month_end - 90 THEN i.customer_id END) AS active_90d
FROM months m
JOIN fct_invoices i ON i.invoice_date <= m.month_end AND i.invoice_date > m.month_end - 90
GROUP BY 1
ORDER BY 1;
