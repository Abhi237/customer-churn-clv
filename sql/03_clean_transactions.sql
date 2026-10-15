-- 03_clean_transactions.sql
-- Classify and clean product lines. Each rule is counted in dq_report / dq_cleaning_waterfall.
--   return                     negative quantity or negative sales_value: kept apart (fct_returns), never a purchase
--   empty                      quantity 0, no value, no discount                      -> dropped
--   inconsistent_zero_quantity quantity 0 but sales_value > 0                        -> dropped (cannot be a purchase)
--   zero_price_no_discount     quantity > 0, sales_value 0, no discount of any kind  -> dropped
--   coupon_only                quantity 0, sales_value 0, carries a discount          -> kept (coupon features only)
--   free_item                  quantity > 0, sales_value 0 with a discount (BOGO)     -> kept, $0 spend
--   merchandise                everything else                                       -> kept
-- quantity > max_quantity is not a unit count (mostly FUEL, sold in thousandths of a gallon): the line keeps its
-- dollar value, but units_clean is NULL so item counts are never distorted. Lines above high_value_line_usd are
-- only flagged.
CREATE OR REPLACE TABLE dept_groups AS SELECT * FROM dept_groups_df;

CREATE OR REPLACE TABLE fct_lines AS
WITH classified AS (
    SELECT t.*,
           CASE
               WHEN t.quantity < 0 OR t.sales_value < 0 THEN 'return'
               WHEN t.quantity = 0 AND t.sales_value = 0 AND t.retail_disc = 0 AND t.coupon_disc = 0
                    AND t.coupon_match_disc = 0 THEN 'empty'
               WHEN t.quantity = 0 AND t.sales_value = 0 THEN 'coupon_only'
               WHEN t.quantity = 0 AND t.sales_value > 0 THEN 'inconsistent_zero_quantity'
               WHEN t.sales_value = 0 AND t.retail_disc = 0 AND t.coupon_disc = 0
                    AND t.coupon_match_disc = 0 THEN 'zero_price_no_discount'
               WHEN t.sales_value = 0 THEN 'free_item'
               ELSE 'merchandise' END AS line_type
    FROM stg_transactions t
)
SELECT
    c.raw_line_id, c.household_id, c.store_id, c.basket_id, c.product_id,
    c.txn_ts, c.txn_date, c.retail_week, c.line_type,
    COALESCE(p.department, 'UNKNOWN')                         AS department,
    COALESCE(g.dept_group, CASE WHEN p.department IS NULL THEN 'unknown' ELSE 'other' END) AS dept_group,
    COALESCE(p.brand = 'Private', FALSE)                      AS is_private_label,
    c.quantity,
    CASE WHEN c.line_type IN ('merchandise', 'free_item')
              AND c.quantity <= (SELECT max_quantity FROM params) THEN c.quantity END AS units_clean,
    (c.quantity > (SELECT max_quantity FROM params))          AS is_quantity_outlier,
    (c.sales_value > (SELECT high_value_line_usd FROM params)) AS is_high_value_line,
    c.sales_value, c.retail_disc, c.coupon_disc, c.coupon_match_disc,
    c.sales_value + c.retail_disc + c.coupon_match_disc       AS shelf_value
FROM classified c
LEFT JOIN stg_products p USING (product_id)
LEFT JOIN dept_groups g ON g.department = p.department
WHERE c.line_type IN ('merchandise', 'free_item', 'coupon_only', 'return');

-- Cleaning waterfall (one row per step; later steps are subsets of earlier ones).
CREATE OR REPLACE TABLE dq_cleaning_waterfall AS
WITH cls AS (
    SELECT CASE
               WHEN quantity < 0 OR sales_value < 0 THEN 'return'
               WHEN quantity = 0 AND sales_value = 0 AND retail_disc = 0 AND coupon_disc = 0
                    AND coupon_match_disc = 0 THEN 'empty'
               WHEN quantity = 0 AND sales_value = 0 THEN 'coupon_only'
               WHEN quantity = 0 AND sales_value > 0 THEN 'inconsistent_zero_quantity'
               WHEN sales_value = 0 AND retail_disc = 0 AND coupon_disc = 0 AND coupon_match_disc = 0
                    THEN 'zero_price_no_discount'
               WHEN sales_value = 0 THEN 'free_item'
               ELSE 'merchandise' END AS line_type
    FROM stg_transactions
)
SELECT 'raw lines' AS step, COUNT(*)::BIGINT AS n_rows FROM cls
UNION ALL SELECT 'dropped: empty lines (no value, no discount)', COUNT(*) FROM cls WHERE line_type = 'empty'
UNION ALL SELECT 'dropped: zero quantity with positive sales', COUNT(*) FROM cls WHERE line_type = 'inconsistent_zero_quantity'
UNION ALL SELECT 'dropped: zero price, no discount', COUNT(*) FROM cls WHERE line_type = 'zero_price_no_discount'
UNION ALL SELECT 'kept lines (merchandise, free items, coupon-only, returns)', COUNT(*) FROM fct_lines
UNION ALL SELECT 'kept: merchandise lines', COUNT(*) FROM fct_lines WHERE line_type = 'merchandise'
UNION ALL SELECT 'kept: free-item lines ($0 sales, discounted)', COUNT(*) FROM fct_lines WHERE line_type = 'free_item'
UNION ALL SELECT 'kept: coupon-only lines', COUNT(*) FROM fct_lines WHERE line_type = 'coupon_only'
UNION ALL SELECT 'kept: return lines (negative quantity or value)', COUNT(*) FROM fct_lines WHERE line_type = 'return'
UNION ALL SELECT 'flagged: quantity above max (units set to NULL, dollars kept)', COUNT(*) FROM fct_lines WHERE is_quantity_outlier
UNION ALL SELECT 'flagged: line above high-value threshold (kept)', COUNT(*) FROM fct_lines WHERE is_high_value_line;
