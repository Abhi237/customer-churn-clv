# Entity / data model

```mermaid
erDiagram
    stg_transactions ||--o{ fct_lines : "cleaned (empty, zero-price lines dropped)"
    stg_products ||--o{ fct_lines : "product_id -> department"
    fct_lines }o--|| fct_baskets : "grouped by household x basket"
    fct_baskets }o--|| fct_days : "grouped by household x day"
    dim_households ||--o{ fct_baskets : "household_id"
    dim_households ||--o{ fct_days : "household_id"
    dim_households ||--o{ cohort_retention : "entry_cohort"
    dim_households ||--|| household_rfm : "household_id"
    dim_households ||--o| survival_households : "household_id"
    dim_households ||--o{ campaign_exposure : "household_id"
    dim_households ||--o{ coupon_redemptions_clean : "household_id"
    dim_households ||--o{ model_snapshot : "household_id x cutoff"
    stg_demographics |o--o| dim_households : "801 of 2,469 (partial)"
    dim_campaigns ||--o{ campaign_exposure : "campaign_id"

    stg_transactions {
        bigint raw_line_id PK
        varchar household_id
        varchar basket_id
        varchar product_id
        bigint quantity
        decimal sales_value
        decimal retail_disc
        decimal coupon_disc
        timestamp txn_ts
    }
    fct_baskets {
        varchar household_id PK
        varchar basket_id PK
        varchar store_id
        date basket_date
        bigint n_lines
        decimal spend
        decimal fresh_spend
        boolean has_coupon
    }
    fct_days {
        varchar household_id PK
        date visit_date PK
        decimal spend
    }
    dim_households {
        varchar household_id PK
        date first_purchase_date
        date entry_cohort
        decimal lifetime_spend
        boolean has_demographics
    }
    household_rfm {
        varchar household_id PK
        int r_score
        int f_score
        int m_score
        varchar rfm_segment
    }
    campaign_exposure {
        varchar household_id PK
        varchar campaign_id PK
        varchar campaign_type
        date start_date
    }
    dim_campaigns {
        varchar campaign_id PK
        varchar campaign_type
        bigint n_households_targeted
        bigint n_redemptions
    }
    survival_households {
        varchar household_id PK
        int event
        bigint duration_days
        bigint baseline_trips
    }
    model_snapshot {
        varchar household_id PK
        date cutoff PK
        double trips_per_week
        double share_fresh
        int churned
    }
```

`model_snapshot` is produced in Python (`features.build_snapshot`) from `fct_baskets`, `fct_days`,
`campaign_exposure` and `coupon_redemptions_clean`, one cutoff at a time. `fct_returns` exists but is empty
because the extract has no negative quantities or sales values.
