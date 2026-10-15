# Data

**Source:** 84.51 **Complete Journey 2.0** study data (a US grocery retailer's loyalty-card transactions for
2,469 households over calendar year 2017), as redistributed by the R package
[`completejourney`](https://github.com/bradleyboehmke/completejourney) v1.1.1 (Boehmke, B.).
Originally published by 84.51 (the Kroger analytics subsidiary) for research and teaching.

**License:** CC0 1.0 Universal (public domain dedication,
<https://creativecommons.org/publicdomain/zero/1.0/>). No permission is required to use, modify or redistribute
the data; attribution to 84.51 and the `completejourney` package is nevertheless given here as good practice.

## Fetch

```bash
make data          # = python scripts/download_data.py
```
The package ships R objects (`.rda` / `.rds`), so conversion to CSV uses R (`R/prepare_completejourney.R`,
`Rscript` required; `data.table` is used when installed). The script downloads the pinned release files into
`data/raw/completejourney/src/` and writes one CSV per table. Use `python scripts/download_data.py --from-csv DIR`
to copy CSVs you already have, and `--force` to redo the download. Manual alternative: in R,
`install.packages("completejourney")`, then write each table from `completejourney::get_transactions()`,
`products`, `demographics`, `campaigns`, `campaign_descriptions`, `coupon_redemptions` and `coupons` with
`data.table::fwrite()`.

| Folder | Contents | Committed? |
|---|---|---|
| `data/raw/completejourney/` | source objects and CSVs | **No** (gitignored) |
| `data/processed/` | DuckDB database and model tables | **No** |

`make sql` builds `data/processed/grocery.duckdb` from the CSVs.

## Raw tables

| File | Rows | Meaning |
|---|---|---|
| `transactions.csv` | 1,469,307 | line per product in a basket: household_id, store_id, basket_id, product_id, quantity, sales_value, retail_disc, coupon_disc, coupon_match_disc, week, transaction_timestamp |
| `products.csv` | 92,331 | product_id, department, brand, product_category and more |
| `demographics.csv` | 801 | age, income, home ownership, marital status, household size, composition, kids (**only 801 of 2,469 households**) |
| `campaigns.csv` | 6,589 | which households were targeted by which of 27 campaigns |
| `campaign_descriptions.csv` | 27 | campaign type (A, B, C), start and end dates |
| `coupon_redemptions.csv` | 2,102 | household, coupon UPC, campaign, redemption date |
| `coupons.csv` | - | coupon UPC to product and campaign mapping (not used by the model) |

Known properties: one year only (2017); loyalty-card purchases only; no returns (no negative quantity or sales
value); quantity includes non-unit measures such as fuel (cleaned and counted in the SQL, see the README).

## Privacy guidance

* Household IDs are pseudonymous. Do **not** attempt to re-identify households or join to external data.
* Household-level artifacts (scores, snapshots, model files, SHAP values) are git-ignored except the compact
  scored table in `app/data/`, which holds model features and scores only (no raw transactions or demographics).
* If you adapt the pipeline to real company data, apply data minimisation, access controls and your
  organisation's retention policy, and treat churn scores as personal data.
