-- 09_survival.sql
-- Time-to-lapse dataset for survival analysis (R). Design for a 12-month panel (see docs/methodology):
--   * Time origin = first OBSERVED purchase (the panel starts 2017-01-01, so this is not acquisition: the study
--     population is prevalent shoppers, i.e. tenure before 2017 is unknown / left-censored).
--   * Baseline covariates are measured in the first `baseline_days` after the origin (landmark design). Households
--     enter the risk set at the landmark = origin + baseline_days, so covariates never use follow-up information
--     (no immortal-time bias).
--   * Event ("lapse") = first inter-purchase gap longer than `lapse_days` (90) that starts at or after the landmark.
--     Event time = days from the landmark to the last purchase before the gap, + 1.
--   * A lapse is only confirmable if the gap fits inside the data: the last purchase d needs d + lapse_days <=
--     data_end. Everyone else is right-censored at (data_end - lapse_days) - landmark + 1.
--   * Excluded and counted: households whose landmark falls after data_end - lapse_days (no time at risk) and
--     households that already lapsed before the landmark.
CREATE OR REPLACE TABLE household_lapse AS
WITH gaps AS (
    SELECT household_id, visit_date,
           LEAD(visit_date) OVER (PARTITION BY household_id ORDER BY visit_date) AS next_date
    FROM fct_days
),
flagged AS (
    SELECT household_id, visit_date,
           CASE WHEN next_date IS NULL
                     THEN (CASE WHEN visit_date + (SELECT lapse_days FROM params) <= (SELECT data_end FROM params)
                                THEN 1 ELSE 0 END)
                WHEN next_date - visit_date > (SELECT lapse_days FROM params) THEN 1
                ELSE 0 END AS lapse_after
    FROM gaps
)
SELECT household_id, MIN(visit_date) FILTER (WHERE lapse_after = 1) AS lapse_date FROM flagged GROUP BY 1;

CREATE OR REPLACE TABLE survival_households AS
WITH origin AS (
    SELECT household_id, first_purchase_date,
           first_purchase_date + (SELECT baseline_days FROM params) AS landmark_date
    FROM dim_households
),
base AS (    -- baseline window [origin, origin + baseline_days)
    SELECT o.household_id,
           COUNT(DISTINCT b.basket_date)                                      AS baseline_trips,
           COUNT(*)                                                           AS baseline_baskets,
           SUM(b.spend) * 1.0 / COUNT(*)                                      AS baseline_avg_basket_spend,
           SUM(b.fresh_spend) * 1.0 / NULLIF(SUM(b.spend), 0)                 AS baseline_fresh_share,
           (SUM(b.coupon_disc) + SUM(b.coupon_match_disc)) * 1.0 / NULLIF(SUM(b.shelf_value), 0) AS baseline_coupon_share,
           COUNT(DISTINCT b.store_id)                                         AS baseline_n_stores
    FROM origin o JOIN fct_baskets b ON b.household_id = o.household_id
         AND b.basket_date >= o.first_purchase_date AND b.basket_date < o.landmark_date
    GROUP BY o.household_id
)
SELECT h.household_id,
       o.first_purchase_date,
       o.landmark_date,
       h.entry_cohort,
       CASE WHEN l.lapse_date IS NOT NULL THEN 1 ELSE 0 END AS event,
       CASE WHEN l.lapse_date IS NOT NULL THEN l.lapse_date - o.landmark_date + 1
            ELSE ((SELECT data_end FROM params) - (SELECT lapse_days FROM params)) - o.landmark_date + 1 END AS duration_days,
       b.baseline_trips, b.baseline_baskets, b.baseline_avg_basket_spend, b.baseline_fresh_share,
       b.baseline_coupon_share, b.baseline_n_stores,
       h.has_demographics, h.age, h.income
FROM dim_households h
JOIN origin o USING (household_id)
JOIN household_lapse l USING (household_id)
JOIN base b USING (household_id)
WHERE o.landmark_date <= (SELECT data_end FROM params) - (SELECT lapse_days FROM params)
  AND (l.lapse_date IS NULL OR l.lapse_date >= o.landmark_date);

CREATE OR REPLACE TABLE survival_exclusions AS
SELECT 'households with purchases' AS reason, COUNT(*) AS n FROM dim_households
UNION ALL SELECT 'excluded: landmark after data_end - lapse_days (no time at risk)', COUNT(*)
    FROM dim_households WHERE first_purchase_date + (SELECT baseline_days FROM params) >
                              (SELECT data_end FROM params) - (SELECT lapse_days FROM params)
UNION ALL SELECT 'excluded: lapsed before the landmark', COUNT(*) FROM dim_households h
    JOIN household_lapse l USING (household_id)
    WHERE h.first_purchase_date + (SELECT baseline_days FROM params) <= (SELECT data_end FROM params) - (SELECT lapse_days FROM params)
      AND l.lapse_date < h.first_purchase_date + (SELECT baseline_days FROM params)
UNION ALL SELECT 'analysed households', COUNT(*) FROM survival_households
UNION ALL SELECT 'lapse events', COALESCE(SUM(event), 0) FROM survival_households;
