-- 01_stage_raw.sql
-- Stage the raw extract (registered by the runner as view `raw_source`) with typed, snake_case columns.
-- Grain: one row per raw invoice line as delivered (duplicates and cancellations still present).
CREATE OR REPLACE TABLE stg_transactions AS
SELECT
    ROW_NUMBER() OVER ()                         AS raw_line_id,
    TRIM(CAST(invoice AS VARCHAR))               AS invoice,
    UPPER(TRIM(CAST(stockcode AS VARCHAR)))      AS stock_code,
    TRIM(CAST(description AS VARCHAR))           AS description,
    CAST(quantity AS INTEGER)                    AS quantity,
    CAST(invoice_date AS TIMESTAMP)              AS invoice_ts,
    CAST(price AS DOUBLE)                        AS unit_price,
    CAST(CAST(customer_id AS BIGINT) AS VARCHAR) AS customer_id,
    TRIM(CAST(country AS VARCHAR))               AS country,
    source_sheet
FROM raw_source;
