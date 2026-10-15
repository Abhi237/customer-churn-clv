-- 06_rolling_activity.sql
-- Rolling 30/60/90-day activity per household, evaluated at every shopping day
-- (RANGE window frames over dates; DuckDB/Postgres compatible).
CREATE OR REPLACE TABLE household_daily_rolling AS
SELECT
    household_id,
    visit_date,
    n_baskets,
    spend,
    COUNT(*)   OVER w30 AS trips_30d,
    COUNT(*)   OVER w60 AS trips_60d,
    COUNT(*)   OVER w90 AS trips_90d,
    SUM(spend) OVER w30 AS spend_30d,
    SUM(spend) OVER w60 AS spend_60d,
    SUM(spend) OVER w90 AS spend_90d,
    visit_date - LAG(visit_date) OVER (PARTITION BY household_id ORDER BY visit_date) AS days_since_prev_visit
FROM fct_days
WINDOW
    w30 AS (PARTITION BY household_id ORDER BY visit_date RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND CURRENT ROW),
    w60 AS (PARTITION BY household_id ORDER BY visit_date RANGE BETWEEN INTERVAL 59 DAY PRECEDING AND CURRENT ROW),
    w90 AS (PARTITION BY household_id ORDER BY visit_date RANGE BETWEEN INTERVAL 89 DAY PRECEDING AND CURRENT ROW);

-- Panel-level active households at each month end (30/60/90-day windows ending that day).
-- Windows ending before day 30/60/90 of the panel are truncated by the 2017-01-01 start.
CREATE OR REPLACE TABLE monthly_active_households AS
WITH months AS (
    SELECT (DATE_TRUNC('month', d) + INTERVAL 1 MONTH - INTERVAL 1 DAY)::DATE AS month_end
    FROM generate_series((SELECT DATE_TRUNC('month', data_start) FROM params),
                         (SELECT DATE_TRUNC('month', data_end) FROM params), INTERVAL 1 MONTH) t(d)
)
SELECT m.month_end,
       COUNT(DISTINCT CASE WHEN f.visit_date > m.month_end - 30 THEN f.household_id END) AS active_30d,
       COUNT(DISTINCT CASE WHEN f.visit_date > m.month_end - 60 THEN f.household_id END) AS active_60d,
       COUNT(DISTINCT CASE WHEN f.visit_date > m.month_end - 90 THEN f.household_id END) AS active_90d
FROM months m
JOIN fct_days f ON f.visit_date <= m.month_end AND f.visit_date > m.month_end - 90
GROUP BY 1
ORDER BY 1;
