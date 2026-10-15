# Data dictionary

All tables are built by `sql/01..09` in DuckDB (`data/processed/grocery.duckdb`). Money is USD and is held as
`DECIMAL` so sums are exact. Source: 84.51 Complete Journey 2.0 via the `completejourney` R package (CC0), one
calendar year (2017-01-01 to 2017-12-31), 2,469 loyalty-card households.

## Staging tables (grain: one row per raw source row)
| Table | Rows | Columns |
|---|---|---|
| `stg_transactions` | 1,469,307 | raw_line_id, household_id, store_id, basket_id, product_id, quantity, sales_value, retail_disc, coupon_disc, coupon_match_disc, txn_ts, txn_date, retail_week |
| `stg_products` | 92,331 | product_id, department, brand, product_category |
| `stg_campaign_descriptions` | 27 | campaign_id, campaign_type (A/B/C), start_date, end_date |
| `stg_campaigns` | 6,589 | household_id, campaign_id (household was targeted by the campaign) |
| `stg_coupon_redemptions` | 2,102 | household_id, coupon_upc, campaign_id, redemption_date |
| `stg_demographics` | 801 | household_id, age, income, home_ownership, marital_status, household_size, household_comp, kids_count (**partial coverage, 32% of households**) |

`sales_value` is the amount paid by the customer after all discounts. Discount columns are stored as the source
reports them (negative or zero). `shelf_value` = sales_value minus the discount columns (the pre-discount value).

## `fct_lines` - cleaned lines (grain: one transaction line)
`stg_transactions` minus empty lines, zero-quantity lines with positive sales and zero-price lines without a
discount (counts in `dq_cleaning_waterfall`). Adds `department`, `dept_group` (fresh, grocery, fuel,
health_beauty, other, unknown), `is_private_label`, `line_type` (merchandise, free item, coupon-only, return),
`units_clean` (quantity, set to NULL above `data.max_quantity`), `is_quantity_outlier`, `is_high_value_line` and
`shelf_value`.

## `fct_baskets` - purchases (grain: household x basket; 155,368 rows)
| Column | Description |
|---|---|
| household_id, basket_id | composite key |
| store_id, basket_ts, basket_date | store and timestamp of the basket |
| n_lines, n_units, n_departments | basket size and breadth |
| spend, shelf_value | amount paid and pre-discount value |
| retail_disc, coupon_disc, coupon_match_disc | discount components |
| fresh_spend, grocery_spend, fuel_spend, health_beauty_spend, private_label_spend | spend by department group |
| has_coupon | basket used a manufacturer or retailer coupon |

Baskets whose total is not positive (480) are excluded and counted; this extract has no returns, so
`fct_returns` (household_id, basket_id, return_date, returned_amount) is empty by design.

## `fct_days` (grain: household x shopping day; 127,320 rows)
household_id, visit_date, n_baskets, spend. The unit for recency, frequency and the BG/NBD model.

## `dim_households` (2,469 rows)
household_id (key), first_purchase_date, last_purchase_date, n_baskets, n_shopping_days, n_stores,
lifetime_spend, entry_cohort (month of first purchase), top_store_id, has_demographics and the demographic
columns (NULL when not covered).

## Cohorts and activity
* `cohort_retention`: entry_cohort, months_since_entry, cohort_size, active_households, retention_rate.
* `household_daily_rolling`: per shopping day, trips and spend over trailing 30/60/90 days and
  days_since_prev_visit (window frames `RANGE BETWEEN INTERVAL n DAY PRECEDING AND CURRENT ROW`).
* `monthly_active_households`: month_end, active_30d, active_60d, active_90d.
* `household_rfm`: recency_days, frequency, monetary, r/f/m quintile scores (`NTILE`), rfm_cell and rfm_segment
  (Champions, Loyal, New / Promising, Needs attention, At risk, Hibernating).

## Campaigns
* `campaign_exposure`: household x campaign with campaign_type, start_date, end_date.
* `coupon_redemptions_clean`: de-duplicated redemptions with campaign_id and date.
* `dim_campaigns`: per campaign, duration, households targeted, redeeming households, redemptions,
  redemption_rate_households.

## Survival (landmark design, consumed by R)
`survival_households` (2,278 rows): household_id, first_purchase_date, landmark_date (first purchase + 28 days),
entry_cohort, event (1 = first gap over 90 days), duration_days (from the landmark), baseline_trips,
baseline_baskets, baseline_avg_basket_spend, baseline_fresh_share, baseline_coupon_share, baseline_n_stores,
has_demographics, age, income. `survival_exclusions` records who was excluded and why;
`household_lapse` holds each household's first lapse date.

## Data quality
`dq_report` (one row per check, 22 checks) and `dq_cleaning_waterfall` (raw lines to kept lines, with flagged
rules); both are shown in the app's Data and cleaning page.

## Model snapshot features (`src/churn_clv/features.py`, one row per household x cutoff)
All use baskets strictly before the cutoff. Population: household shopped in the 90 days before the cutoff.
Label `churned` = no shopping day in `[cutoff, cutoff + 90 days)`; a **proxy**, not an observed cancellation.

| Group | Features |
|---|---|
| RFM and cadence | recency_days, trips_per_week, spend_per_week, trips_28d, trips_42d, spend_28d, spend_42d, trip_ratio_28d, mean_interpurchase_days, std_interpurchase_days, recency_to_gap_ratio, active_week_share |
| Basket behavior | avg_basket_spend, avg_units_per_basket, avg_lines_per_basket |
| Department mix (share of spend) | share_fresh, share_grocery, share_fuel, share_health_beauty, share_private_label |
| Discount and coupon use | retail_disc_share, coupon_disc_share, coupon_basket_share |
| Store loyalty | n_stores_42d, top_store_share |
| Campaign exposure and redemption | n_campaigns_90d, n_campaigns_active, n_type_a_90d, redemptions_90d, redemptions_42d |
| Descriptive only (not model inputs) | tenure_days, frequency, monetary_total (cumulative, grow with the calendar) |
| Outcome columns | future_trips, future_spend, churned (label window only) |

Rates and shares divide by `max(4, tenure_days / 7)` weeks, so a household seen for less than four weeks is not
given an inflated rate.
