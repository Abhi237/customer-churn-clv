-- 02_data_quality.sql
-- Data-quality profile of the staged transactions. One row per check (n_rows = lines or entities flagged).
CREATE OR REPLACE TABLE dq_report AS
WITH base AS (SELECT * FROM stg_transactions),
dup AS (
    SELECT COALESCE(SUM(n - 1), 0) AS dup_rows
    FROM (SELECT COUNT(*) AS n FROM base
          GROUP BY household_id, basket_id, product_id, quantity, sales_value, retail_disc, coupon_disc,
                   coupon_match_disc, txn_ts) g
),
basket_tot AS (SELECT household_id, basket_id, SUM(sales_value) AS s FROM base GROUP BY 1, 2)
SELECT 'total_lines' AS check_name, COUNT(*)::BIGINT AS n_rows FROM base
UNION ALL SELECT 'distinct_households',          COUNT(DISTINCT household_id) FROM base
UNION ALL SELECT 'distinct_baskets',             COUNT(DISTINCT basket_id) FROM base
UNION ALL SELECT 'exact_duplicate_lines',        (SELECT dup_rows FROM dup)::BIGINT
UNION ALL SELECT 'missing_household_or_basket',  COUNT(*) FILTER (WHERE household_id IS NULL OR basket_id IS NULL) FROM base
UNION ALL SELECT 'missing_timestamp',            COUNT(*) FILTER (WHERE txn_ts IS NULL) FROM base
UNION ALL SELECT 'negative_quantity_lines',      COUNT(*) FILTER (WHERE quantity < 0) FROM base
UNION ALL SELECT 'negative_sales_value_lines',   COUNT(*) FILTER (WHERE sales_value < 0) FROM base
UNION ALL SELECT 'zero_quantity_lines',          COUNT(*) FILTER (WHERE quantity = 0) FROM base
UNION ALL SELECT 'zero_quantity_positive_sales', COUNT(*) FILTER (WHERE quantity = 0 AND sales_value > 0) FROM base
UNION ALL SELECT 'zero_sales_value_lines',       COUNT(*) FILTER (WHERE sales_value = 0) FROM base
UNION ALL SELECT 'empty_lines_no_value_no_discount',
    COUNT(*) FILTER (WHERE quantity = 0 AND sales_value = 0 AND retail_disc = 0 AND coupon_disc = 0
                       AND coupon_match_disc = 0) FROM base
UNION ALL SELECT 'coupon_only_lines',            COUNT(*) FILTER (WHERE quantity = 0 AND sales_value = 0 AND
                       (retail_disc > 0 OR coupon_disc > 0 OR coupon_match_disc > 0)) FROM base
UNION ALL SELECT 'free_item_lines',              COUNT(*) FILTER (WHERE quantity > 0 AND sales_value = 0 AND
                       (retail_disc > 0 OR coupon_disc > 0 OR coupon_match_disc > 0)) FROM base
UNION ALL SELECT 'zero_price_no_discount_lines', COUNT(*) FILTER (WHERE quantity > 0 AND sales_value = 0 AND
                       retail_disc = 0 AND coupon_disc = 0 AND coupon_match_disc = 0) FROM base
UNION ALL SELECT 'quantity_above_max_lines',     COUNT(*) FILTER (WHERE quantity > (SELECT max_quantity FROM params)) FROM base
UNION ALL SELECT 'quantity_above_max_fuel_lines',
    COUNT(*) FILTER (WHERE b.quantity > (SELECT max_quantity FROM params) AND p.department = 'FUEL')
    FROM base b LEFT JOIN stg_products p USING (product_id)
UNION ALL SELECT 'sales_value_above_high_value_lines', COUNT(*) FILTER (WHERE sales_value > (SELECT high_value_line_usd FROM params)) FROM base
UNION ALL SELECT 'lines_without_product_match',  COUNT(*) FILTER (WHERE product_id NOT IN (SELECT product_id FROM stg_products)) FROM base
UNION ALL SELECT 'baskets_nonpositive_total',    COUNT(*) FILTER (WHERE s <= 0) FROM basket_tot
UNION ALL SELECT 'baskets_spanning_multiple_households',
    (SELECT COUNT(*) FROM (SELECT basket_id FROM base GROUP BY 1 HAVING COUNT(DISTINCT household_id) > 1) x)
UNION ALL SELECT 'baskets_spanning_multiple_days',
    (SELECT COUNT(*) FROM (SELECT basket_id FROM base GROUP BY 1 HAVING COUNT(DISTINCT txn_date) > 1) x);
