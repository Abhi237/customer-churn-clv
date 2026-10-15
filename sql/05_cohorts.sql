-- 05_cohorts.sql
-- ENTRY cohorts: month of the household's first OBSERVED purchase, and the share of each cohort shopping in month k
-- afterwards. The panel starts on 2017-01-01 and households are loyalty-card shoppers, so first observed purchase is
-- NOT acquisition: the January cohort is mostly pre-existing customers (left-censored), later cohorts are small.
CREATE OR REPLACE TABLE cohort_retention AS
WITH activity AS (
    SELECT DISTINCT household_id, DATE_TRUNC('month', visit_date)::DATE AS activity_month FROM fct_days
),
joined AS (
    SELECT c.entry_cohort, a.household_id,
           (DATE_PART('year', a.activity_month) - DATE_PART('year', c.entry_cohort)) * 12
             + DATE_PART('month', a.activity_month) - DATE_PART('month', c.entry_cohort) AS months_since
    FROM activity a JOIN dim_households c USING (household_id)
),
sizes AS (SELECT entry_cohort, COUNT(*) AS cohort_size FROM dim_households GROUP BY 1)
SELECT j.entry_cohort,
       CAST(j.months_since AS INTEGER)                   AS months_since_entry,
       s.cohort_size,
       COUNT(DISTINCT j.household_id)                    AS active_households,
       COUNT(DISTINCT j.household_id) * 1.0 / s.cohort_size AS retention_rate
FROM joined j JOIN sizes s USING (entry_cohort)
GROUP BY 1, 2, 3
ORDER BY 1, 2;
