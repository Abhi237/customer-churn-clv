-- 03_clean_transactions.sql
-- Normalise invoice lines.
--   * drop exact duplicate lines (the two Excel sheets overlap, and the source has repeated lines)
--   * classify each line: sale | cancellation (invoice starts with 'C') | adjustment ('A' bad-debt)
--   * exclude non-product stock codes (postage, manual, bank charges, fees, samples ...)
--   * exclude lines with missing customer_id (cannot be attributed to a customer lifecycle)
--   * exclude non-positive prices and absolute quantities above params.max_abs_quantity
-- Output grain: one row per (deduplicated) customer line, sales and returns kept separately via line_type.
CREATE OR REPLACE TABLE fct_lines AS
WITH dedup AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY invoice, stock_code, description, quantity, invoice_ts, unit_price, customer_id, country
               ORDER BY raw_line_id) AS rn
    FROM stg_transactions
),
typed AS (
    SELECT *,
           CASE WHEN invoice LIKE 'C%' THEN 'cancellation'
                WHEN invoice LIKE 'A%' THEN 'adjustment'
                WHEN quantity < 0      THEN 'stock_adjustment'
                ELSE 'sale' END AS line_type
    FROM dedup
    WHERE rn = 1
)
SELECT
    raw_line_id, invoice, stock_code, description, quantity, invoice_ts,
    CAST(invoice_ts AS DATE)              AS invoice_date,
    unit_price, customer_id, country, line_type,
    CAST(ROUND(quantity * unit_price, 3) AS DECIMAL(18, 3)) AS line_amount   -- exact sums (reproducible)
FROM typed
WHERE customer_id IS NOT NULL
  AND line_type IN ('sale', 'cancellation')
  AND unit_price > 0
  AND ABS(quantity) <= (SELECT max_abs_quantity FROM params)
  AND stock_code NOT IN (SELECT code FROM non_product_codes);

-- Cleaning waterfall, for documentation.
CREATE OR REPLACE TABLE dq_cleaning_waterfall AS
SELECT 'raw rows' AS step, COUNT(*) AS n_rows FROM stg_transactions
UNION ALL SELECT 'clean lines (sales + cancellations)', COUNT(*) FROM fct_lines
UNION ALL SELECT 'clean sale lines', COUNT(*) FILTER (WHERE line_type = 'sale') FROM fct_lines
UNION ALL SELECT 'clean cancellation lines', COUNT(*) FILTER (WHERE line_type = 'cancellation') FROM fct_lines;
