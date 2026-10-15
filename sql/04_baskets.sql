-- 04_baskets.sql
-- Purchase-occasion facts. A BASKET is one shopping trip (basket_id); a basket counts as a purchase only if its
-- merchandise spend is > 0 (coupon-only or fully-voided baskets are not purchases). A shopping DAY merges the
-- baskets of one household on one calendar day (used for gaps, BG/NBD and the churn label).
-- Grain fct_baskets: one row per (household_id, basket_id).
CREATE OR REPLACE TABLE fct_baskets AS
SELECT
    household_id,
    basket_id,
    MIN(store_id)                                                           AS store_id,
    MIN(txn_ts)                                                             AS basket_ts,
    MIN(txn_date)                                                           AS basket_date,
    COUNT(*) FILTER (WHERE line_type IN ('merchandise', 'free_item'))       AS n_lines,
    SUM(units_clean)                                                        AS n_units,
    COUNT(DISTINCT department) FILTER (WHERE line_type IN ('merchandise', 'free_item')) AS n_departments,
    SUM(sales_value)                                                        AS spend,
    SUM(shelf_value)                                                        AS shelf_value,
    SUM(retail_disc)                                                        AS retail_disc,
    SUM(coupon_disc)                                                        AS coupon_disc,
    SUM(coupon_match_disc)                                                  AS coupon_match_disc,
    COALESCE(SUM(sales_value) FILTER (WHERE dept_group = 'fresh'), 0)       AS fresh_spend,
    COALESCE(SUM(sales_value) FILTER (WHERE dept_group = 'grocery'), 0)     AS grocery_spend,
    COALESCE(SUM(sales_value) FILTER (WHERE dept_group = 'fuel'), 0)        AS fuel_spend,
    COALESCE(SUM(sales_value) FILTER (WHERE dept_group = 'health_beauty'), 0) AS health_beauty_spend,
    COALESCE(SUM(sales_value) FILTER (WHERE is_private_label), 0)           AS private_label_spend,
    (SUM(coupon_disc) + SUM(coupon_match_disc)) > 0                         AS has_coupon
FROM fct_lines
WHERE line_type IN ('merchandise', 'free_item', 'coupon_only')
GROUP BY household_id, basket_id
HAVING SUM(sales_value) > 0;

-- Grain fct_days: one row per (household_id, visit_date).
CREATE OR REPLACE TABLE fct_days AS
SELECT household_id, basket_date AS visit_date, COUNT(*) AS n_baskets, SUM(spend) AS spend
FROM fct_baskets
GROUP BY household_id, basket_date;

-- Grain fct_returns: one row per basket with return lines (none exist in the 2017 release; the rule is still
-- implemented and unit-tested on a synthetic fixture).
CREATE OR REPLACE TABLE fct_returns AS
SELECT household_id, basket_id, MIN(txn_date) AS return_date, -SUM(sales_value) AS returned_amount
FROM fct_lines
WHERE line_type = 'return'
GROUP BY household_id, basket_id;

-- Grain dim_households: one row per household with >= 1 purchase basket.
CREATE OR REPLACE TABLE dim_households AS
WITH store_rank AS (
    SELECT household_id, store_id,
           ROW_NUMBER() OVER (PARTITION BY household_id ORDER BY COUNT(*) DESC, store_id) AS rn
    FROM fct_baskets GROUP BY household_id, store_id
),
agg AS (
    SELECT household_id,
           MIN(basket_date)                 AS first_purchase_date,
           MAX(basket_date)                 AS last_purchase_date,
           COUNT(*)                         AS n_baskets,
           COUNT(DISTINCT basket_date)      AS n_shopping_days,
           COUNT(DISTINCT store_id)         AS n_stores,
           SUM(spend)                       AS lifetime_spend
    FROM fct_baskets GROUP BY household_id
)
SELECT a.*,
       DATE_TRUNC('month', a.first_purchase_date)::DATE AS entry_cohort,
       s.store_id                                        AS top_store_id,
       (d.household_id IS NOT NULL)                      AS has_demographics,
       d.age, d.income, d.home_ownership, d.marital_status, d.household_size, d.household_comp, d.kids_count
FROM agg a
JOIN store_rank s ON s.household_id = a.household_id AND s.rn = 1
LEFT JOIN stg_demographics d ON d.household_id = a.household_id;
