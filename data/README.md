# Data

**Source:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
<https://archive.ics.uci.edu/dataset/502/online+retail+ii> - DOI 10.24432/C5CG6D.

**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
You may share and adapt it with attribution (cite the dataset as above).

## Fetch

```bash
make data          # = python scripts/download_data.py
```

The script downloads the zip (~46 MB), extracts `online_retail_II.xlsx` (two sheets: 2009-2010 and
2010-2011), concatenates both sheets and writes `data/interim/online_retail_ii.parquet` (1,067,371 rows).
`make sql` then builds `data/processed/retail.duckdb` and parquet/CSV exports.

| Folder | Contents | Committed? |
|---|---|---|
| `data/raw/` | zip + xlsx as downloaded | **No** |
| `data/interim/` | parquet conversion of both sheets | **No** |
| `data/processed/` | DuckDB database and model tables | **No** |

## Raw schema

| Column | Type | Meaning |
|---|---|---|
| Invoice | str | 6-digit invoice number; prefix `C` = cancellation, `A` = bad-debt adjustment |
| StockCode | str | product code; non-product codes (POST, M, BANK CHARGES, D, ...) exist |
| Description | str | product name |
| Quantity | int | units per line (negative for cancellations / stock adjustments) |
| InvoiceDate | datetime | invoice timestamp |
| Price | float | unit price, GBP |
| Customer ID | float | 5-digit customer number; missing for ~23% of lines |
| Country | str | customer's country |

## Privacy guidance

* Customer IDs are pseudonymous numbers, but the data is still transactional personal data in spirit:
  do **not** attempt to re-identify customers, join to external data, or publish customer-level outputs.
* Customer-level artifacts (scores, snapshots, model files) are git-ignored; only aggregated results and
  figures are committed.
* If you adapt this pipeline to real company data, apply data minimisation, access controls and your
  organisation's retention policy, and treat churn scores as personal data under GDPR / DPDP Act.
