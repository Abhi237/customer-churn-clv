-- 01_stage_raw.sql
-- Stage the raw Complete Journey tables (registered by the runner as views raw_*) with typed columns.
-- Money columns are DECIMAL so that sums are exact and runs are reproducible.
-- Grain stg_transactions: one row per raw product line (household x basket x product), as delivered.
CREATE OR REPLACE TABLE stg_transactions AS
SELECT
    ROW_NUMBER() OVER ()                          AS raw_line_id,
    TRIM(CAST(household_id AS VARCHAR))           AS household_id,
    TRIM(CAST(store_id AS VARCHAR))               AS store_id,
    TRIM(CAST(basket_id AS VARCHAR))              AS basket_id,
    TRIM(CAST(product_id AS VARCHAR))             AS product_id,
    CAST(quantity AS BIGINT)                      AS quantity,
    CAST(sales_value AS DECIMAL(14, 3))           AS sales_value,
    CAST(retail_disc AS DECIMAL(14, 3))           AS retail_disc,
    CAST(coupon_disc AS DECIMAL(14, 3))           AS coupon_disc,
    CAST(coupon_match_disc AS DECIMAL(14, 3))     AS coupon_match_disc,
    CAST(transaction_timestamp AS TIMESTAMP)      AS txn_ts,
    CAST(transaction_timestamp AS DATE)           AS txn_date,
    CAST(week AS INTEGER)                         AS retail_week
FROM raw_transactions;

CREATE OR REPLACE TABLE stg_products AS
SELECT product_id, department, brand, product_category
FROM (
    SELECT TRIM(CAST(product_id AS VARCHAR)) AS product_id,
           UPPER(TRIM(CAST(department AS VARCHAR)))       AS department,
           TRIM(CAST(brand AS VARCHAR))                   AS brand,
           UPPER(TRIM(CAST(product_category AS VARCHAR))) AS product_category,
           ROW_NUMBER() OVER (PARTITION BY TRIM(CAST(product_id AS VARCHAR))) AS rn
    FROM raw_products
) r
WHERE rn = 1;

CREATE OR REPLACE TABLE stg_campaign_descriptions AS
SELECT CAST(campaign_id AS VARCHAR) AS campaign_id,
       TRIM(CAST(campaign_type AS VARCHAR)) AS campaign_type,
       CAST(start_date AS DATE) AS start_date,
       CAST(end_date AS DATE)   AS end_date
FROM raw_campaign_descriptions;

CREATE OR REPLACE TABLE stg_campaigns AS     -- household x campaign: households targeted by the campaign
SELECT DISTINCT TRIM(CAST(household_id AS VARCHAR)) AS household_id, CAST(campaign_id AS VARCHAR) AS campaign_id
FROM raw_campaigns;

CREATE OR REPLACE TABLE stg_coupon_redemptions AS
SELECT TRIM(CAST(household_id AS VARCHAR)) AS household_id,
       TRIM(CAST(coupon_upc AS VARCHAR))   AS coupon_upc,
       CAST(campaign_id AS VARCHAR)        AS campaign_id,
       CAST(redemption_date AS DATE)       AS redemption_date
FROM raw_coupon_redemptions;

CREATE OR REPLACE TABLE stg_demographics AS     -- only ~1/3 of households: PARTIAL coverage
SELECT TRIM(CAST(household_id AS VARCHAR)) AS household_id,
       NULLIF(TRIM(CAST(age AS VARCHAR)), '')             AS age,
       NULLIF(TRIM(CAST(income AS VARCHAR)), '')          AS income,
       NULLIF(TRIM(CAST(home_ownership AS VARCHAR)), '')  AS home_ownership,
       NULLIF(TRIM(CAST(marital_status AS VARCHAR)), '')  AS marital_status,
       NULLIF(TRIM(CAST(household_size AS VARCHAR)), '')  AS household_size,
       NULLIF(TRIM(CAST(household_comp AS VARCHAR)), '')  AS household_comp,
       NULLIF(TRIM(CAST(kids_count AS VARCHAR)), '')      AS kids_count
FROM raw_demographics;
