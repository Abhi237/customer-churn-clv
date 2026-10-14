-- 08_survival.sql
-- Time-to-inactivity dataset for survival analysis (R).
-- Event ("lapse") = first inter-purchase gap longer than params.horizon_days.
-- duration_days = active days from first purchase to the last purchase before the lapse, +1 (event = 1),
-- or to the data end minus horizon (censored, event = 0) when no lapse is observable yet.
-- Covariates are fixed at acquisition time (first invoice), avoiding immortal-time bias.
CREATE OR REPLACE TABLE survival_customers AS
WITH gaps AS (
    SELECT customer_id, invoice_date,
           LEAD(invoice_date) OVER (PARTITION BY customer_id ORDER BY invoice_date) AS next_date
    FROM (SELECT DISTINCT customer_id, invoice_date FROM fct_invoices) d
),
flagged AS (
    SELECT customer_id, invoice_date,
           CASE WHEN next_date IS NULL
                     THEN (CASE WHEN invoice_date + (SELECT horizon_days FROM params) <= (SELECT data_end FROM params) THEN 1 ELSE 0 END)
                WHEN next_date - invoice_date > (SELECT horizon_days FROM params) THEN 1
                ELSE 0 END AS lapse_after
    FROM gaps
),
first_lapse AS (
    SELECT customer_id, MIN(invoice_date) FILTER (WHERE lapse_after = 1) AS lapse_date
    FROM flagged GROUP BY customer_id
),
first_inv AS (
    SELECT customer_id, revenue AS first_order_value, n_products AS first_order_products,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_ts, invoice) AS rn
    FROM fct_invoices
)
SELECT c.customer_id,
       c.first_purchase_date,
       c.acquisition_cohort,
       c.home_country,
       CASE WHEN c.home_country IN ('United Kingdom', 'Germany', 'France', 'EIRE') THEN c.home_country
            ELSE 'Other' END AS country_group,
       f.first_order_value,
       f.first_order_products,
       NTILE(4) OVER (ORDER BY f.first_order_value) AS first_value_quartile,
       CASE WHEN l.lapse_date IS NOT NULL THEN 1 ELSE 0 END AS event,
       CASE WHEN l.lapse_date IS NOT NULL THEN l.lapse_date - c.first_purchase_date + 1
            ELSE ((SELECT data_end FROM params) - (SELECT horizon_days FROM params)) - c.first_purchase_date + 1 END AS duration_days
FROM dim_customers c
JOIN first_lapse l USING (customer_id)
JOIN first_inv f ON f.customer_id = c.customer_id AND f.rn = 1
WHERE c.first_purchase_date <= (SELECT data_end FROM params) - (SELECT horizon_days FROM params);
