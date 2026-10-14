-- 05_cohorts.sql
-- Monthly acquisition cohorts and retention matrix (share of cohort purchasing in month k after acquisition).
CREATE OR REPLACE TABLE cohort_retention AS
WITH activity AS (
    SELECT DISTINCT customer_id, DATE_TRUNC('month', invoice_date)::DATE AS activity_month
    FROM fct_invoices
),
joined AS (
    SELECT c.acquisition_cohort,
           a.customer_id,
           (DATE_PART('year', a.activity_month) - DATE_PART('year', c.acquisition_cohort)) * 12
             + DATE_PART('month', a.activity_month) - DATE_PART('month', c.acquisition_cohort) AS months_since
    FROM activity a
    JOIN dim_customers c USING (customer_id)
),
sizes AS (
    SELECT acquisition_cohort, COUNT(*) AS cohort_size FROM dim_customers GROUP BY 1
)
SELECT j.acquisition_cohort,
       CAST(j.months_since AS INTEGER)          AS months_since_acquisition,
       s.cohort_size,
       COUNT(DISTINCT j.customer_id)            AS active_customers,
       COUNT(DISTINCT j.customer_id) * 1.0 / s.cohort_size AS retention_rate
FROM joined j
JOIN sizes s USING (acquisition_cohort)
GROUP BY 1, 2, 3
ORDER BY 1, 2;
