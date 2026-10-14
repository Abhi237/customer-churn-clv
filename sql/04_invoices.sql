-- 04_invoices.sql
-- Invoice-level purchase fact (sales only) plus customer-level return totals.
-- Grain fct_invoices: one row per (customer_id, invoice) sale.
CREATE OR REPLACE TABLE fct_invoices AS
SELECT
    customer_id,
    invoice,
    MIN(invoice_ts)              AS invoice_ts,
    MIN(invoice_date)            AS invoice_date,
    ANY_VALUE(country)           AS country,
    COUNT(*)                     AS n_lines,
    COUNT(DISTINCT stock_code)   AS n_products,
    SUM(quantity)                AS units,
    SUM(line_amount)             AS revenue
FROM fct_lines
WHERE line_type = 'sale'
GROUP BY customer_id, invoice;

-- Grain fct_returns: one row per cancellation invoice.
CREATE OR REPLACE TABLE fct_returns AS
SELECT customer_id, invoice, MIN(invoice_date) AS invoice_date, -SUM(line_amount) AS returned_amount
FROM fct_lines
WHERE line_type = 'cancellation'
GROUP BY customer_id, invoice;

-- Grain dim_customers: one row per customer. Home country = country of the first invoice.
CREATE OR REPLACE TABLE dim_customers AS
WITH first_country AS (
    SELECT customer_id, country AS home_country
    FROM (
        SELECT customer_id, country,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_ts, invoice) AS rn
        FROM fct_invoices
    ) r
    WHERE rn = 1
)
SELECT
    i.customer_id,
    MIN(i.invoice_date)                             AS first_purchase_date,
    MAX(i.invoice_date)                             AS last_purchase_date,
    fc.home_country,
    DATE_TRUNC('month', MIN(i.invoice_date))::DATE  AS acquisition_cohort,
    COUNT(*)                                        AS n_invoices,
    SUM(i.revenue)                                  AS lifetime_revenue
FROM fct_invoices i
JOIN first_country fc USING (customer_id)
GROUP BY i.customer_id, fc.home_country;
