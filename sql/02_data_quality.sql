-- 02_data_quality.sql
-- Data-quality profile of the staged data. One row per check; written to dq_report.
CREATE OR REPLACE TABLE dq_report AS
WITH base AS (SELECT * FROM stg_transactions),
dup AS (
    SELECT SUM(n - 1) AS dup_rows
    FROM (
        SELECT COUNT(*) AS n
        FROM base
        GROUP BY invoice, stock_code, description, quantity, invoice_ts, unit_price, customer_id, country
    ) g
)
SELECT 'total_rows' AS check_name, COUNT(*)::BIGINT AS n_rows FROM base
UNION ALL SELECT 'exact_duplicate_rows', (SELECT dup_rows FROM dup)::BIGINT
UNION ALL SELECT 'cancellation_lines_invoice_C', COUNT(*) FILTER (WHERE invoice LIKE 'C%') FROM base
UNION ALL SELECT 'adjustment_lines_invoice_A', COUNT(*) FILTER (WHERE invoice LIKE 'A%') FROM base
UNION ALL SELECT 'negative_quantity_lines', COUNT(*) FILTER (WHERE quantity < 0) FROM base
UNION ALL SELECT 'negative_qty_not_cancelled', COUNT(*) FILTER (WHERE quantity < 0 AND invoice NOT LIKE 'C%') FROM base
UNION ALL SELECT 'zero_or_negative_price', COUNT(*) FILTER (WHERE unit_price <= 0) FROM base
UNION ALL SELECT 'missing_customer_id', COUNT(*) FILTER (WHERE customer_id IS NULL) FROM base
UNION ALL SELECT 'missing_description', COUNT(*) FILTER (WHERE description IS NULL) FROM base
UNION ALL SELECT 'non_product_stock_code', COUNT(*) FILTER (WHERE stock_code IN (SELECT code FROM non_product_codes)) FROM base
UNION ALL SELECT 'abs_quantity_outlier', COUNT(*) FILTER (WHERE ABS(quantity) > (SELECT max_abs_quantity FROM params)) FROM base
UNION ALL SELECT 'distinct_customers', COUNT(DISTINCT customer_id) FROM base
UNION ALL SELECT 'distinct_invoices', COUNT(DISTINCT invoice) FROM base;
