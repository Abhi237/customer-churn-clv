# Data dictionary

All tables are built by `sql/01..08` in DuckDB (`data/processed/retail.duckdb`). Money is GBP.

## `stg_transactions` - staged raw lines (grain: raw line)
| Column | Type | Description |
|---|---|---|
| raw_line_id | BIGINT | surrogate row id (source order) |
| invoice | VARCHAR | invoice number; `C*` cancellation, `A*` adjustment |
| stock_code | VARCHAR | upper-cased product code |
| description | VARCHAR | product description |
| quantity | INTEGER | units (negative = return / adjustment) |
| invoice_ts | TIMESTAMP | invoice timestamp |
| unit_price | DOUBLE | unit price |
| customer_id | VARCHAR | customer number (NULL when missing) |
| country | VARCHAR | country |
| source_sheet | VARCHAR | Excel sheet of origin |

## `fct_lines` - cleaned lines (grain: deduplicated customer line)
`stg_transactions` minus exact duplicates, missing customer, non-positive price, `|qty| > 10,000`,
non-product codes, `A` adjustments and non-cancellation negative quantities. Adds
`invoice_date` (DATE), `line_type` (`sale` / `cancellation`), `line_amount` = quantity x unit_price.

## `fct_invoices` - purchases (grain: customer_id x invoice; sales only)
| Column | Description |
|---|---|
| customer_id, invoice | composite key |
| invoice_ts / invoice_date | timestamp / date of the invoice |
| country | invoice country |
| n_lines, n_products, units | basket size measures |
| revenue | sum of line amounts (> 0) |

## `fct_returns` - cancellations (grain: cancellation invoice)
`customer_id, invoice, invoice_date, returned_amount` (positive GBP value returned).

## `dim_customers` (grain: customer_id)
`first_purchase_date, last_purchase_date, home_country` (country of first invoice), `acquisition_cohort`
(month of first purchase), `n_invoices, lifetime_revenue`.

## `cohort_retention` (grain: acquisition_cohort x months_since_acquisition)
`cohort_size, active_customers, retention_rate` = active / cohort size.

## `customer_daily_rolling` (grain: customer_id x purchase date)
`invoices_{30,60,90}d`, `revenue_{30,60,90}d` (RANGE window frames), `days_since_prev_purchase`.

## `monthly_active_customers` (grain: month_end)
`active_30d, active_60d, active_90d` distinct purchasing customers in trailing windows.

## `customer_rfm` (grain: customer_id; as of day after last transaction)
`recency_days, frequency, monetary, r_score, f_score, m_score` (quintiles via NTILE), `rfm_cell`, `rfm_segment`.

## `survival_customers` (grain: customer_id; acquired >= 90 days before data end)
| Column | Description |
|---|---|
| event | 1 if a 90-day lapse (first inter-purchase gap > 90 days) is observed |
| duration_days | active days from first purchase to last purchase before the lapse, + 1 (or to censoring) |
| first_order_value, first_order_products, first_value_quartile | acquisition-time covariates |
| country_group | UK, Germany, France, EIRE, Other |

## Model snapshot features (`src/churn_clv/features.py`, grain: customer_id x cutoff)
Population: customers with >= 1 purchase in the 365 days before the cutoff. **All features use only rows
with `invoice_date < cutoff`.**

| Feature | Definition |
|---|---|
| recency_days | cutoff - last purchase date |
| tenure_days | cutoff - first purchase date |
| frequency | invoices before cutoff |
| monetary_total, avg_order_value | total / mean invoice revenue |
| invoices_30d / 90d / 180d | invoices in trailing windows |
| revenue_90d / 180d | revenue in trailing windows |
| n_distinct_products | distinct stock codes bought |
| avg_products_per_invoice | mean distinct products per invoice |
| mean_interpurchase_days, std_interpurchase_days | gaps between purchase days (tenure if single purchase) |
| active_months_12m | distinct months with a purchase in trailing 365 days |
| n_returns_365d, return_ratio_365d | cancellation invoices and returned value / total spend |
| recency_to_gap_ratio | recency / mean inter-purchase gap ("overdue" indicator) |
| is_uk | any UK invoice |
| **churned** (label) | 1 if **no** invoice in `[cutoff, cutoff + 90 days)` - churn **proxy** |
| future_invoices, future_revenue | label-window outcomes (evaluation only, never features) |
