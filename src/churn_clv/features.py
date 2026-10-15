"""Point-in-time household features and churn-proxy label construction.

Leakage rule: every feature uses ONLY rows dated strictly before the cutoff (basket_date < cutoff, campaign
start_date < cutoff, redemption_date < cutoff). The label uses ONLY rows in [cutoff, cutoff + horizon). The churn
label is a PROXY (no purchase within the horizon), not an observed cancellation: a grocery loyalty card has no
contract, households simply stop shopping (or stop swiping the card).
"""
from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pandas as pd

from .db import as_date

# Design rule (12-month panel): history grows from 42 days at the first cutoff to 273 days at the last, so any
# feature that accumulates with history (lifetime counts, total spend, tenure, longest gap, stores ever used,
# campaigns ever received) would simply encode the calendar date and would be extrapolated at the test cutoff.
# Model features are therefore RATES, SHARES or FIXED-WINDOW quantities (<= 42 days, <= 90 days for campaigns).
FEATURE_GROUPS = {
    "RFM and cadence": ["recency_days", "trips_per_week", "spend_per_week", "trips_28d", "trips_42d", "spend_28d",
                        "spend_42d", "trip_ratio_28d", "mean_interpurchase_days", "std_interpurchase_days",
                        "recency_to_gap_ratio", "active_week_share"],
    "Basket behavior": ["avg_basket_spend", "avg_units_per_basket", "avg_lines_per_basket"],
    "Department mix (share of spend)": ["share_fresh", "share_grocery", "share_fuel", "share_health_beauty",
                                        "share_private_label"],
    "Discount and coupon use": ["retail_disc_share", "coupon_disc_share", "coupon_basket_share"],
    "Store loyalty": ["n_stores_42d", "top_store_share"],
    "Campaign exposure and redemption": ["n_campaigns_90d", "n_campaigns_active", "n_type_a_90d",
                                         "redemptions_90d", "redemptions_42d"],
}
FEATURE_COLUMNS = [c for cols in FEATURE_GROUPS.values() for c in cols]
# cumulative descriptors carried in snapshots for reporting only (NOT model inputs, see the design rule above)
DESCRIPTIVE_COLUMNS = ["tenure_days", "frequency", "monetary_total"]

FEATURE_SQL = """
WITH hist AS (                         -- strictly pre-cutoff baskets
    SELECT * FROM fct_baskets WHERE basket_date < DATE '{cutoff}'
),
pop AS (                               -- population: shopped in the lookback window
    SELECT household_id FROM hist GROUP BY household_id
    HAVING MAX(basket_date) >= DATE '{cutoff}' - INTERVAL {lookback} DAY
),
days AS (SELECT household_id, visit_date, spend FROM fct_days WHERE visit_date < DATE '{cutoff}'),
gaps AS (
    SELECT household_id,
           visit_date - LAG(visit_date) OVER (PARTITION BY household_id ORDER BY visit_date) AS gap
    FROM days
),
gap_stats AS (
    SELECT household_id, AVG(gap) AS mean_gap, STDDEV_SAMP(gap) AS std_gap
    FROM gaps GROUP BY household_id
),
day_agg AS (
    SELECT household_id,
           DATE '{cutoff}' - MAX(visit_date)                                          AS recency_days,
           DATE '{cutoff}' - MIN(visit_date)                                          AS tenure_days,
           COUNT(*)                                                                   AS frequency,
           SUM(spend)                                                                 AS monetary_total,
           COUNT(*) FILTER (WHERE visit_date >= DATE '{cutoff}' - INTERVAL 28 DAY)    AS trips_28d,
           COUNT(*) FILTER (WHERE visit_date >= DATE '{cutoff}' - INTERVAL 42 DAY)    AS trips_42d,
           COALESCE(SUM(spend) FILTER (WHERE visit_date >= DATE '{cutoff}' - INTERVAL 28 DAY), 0) AS spend_28d,
           COALESCE(SUM(spend) FILTER (WHERE visit_date >= DATE '{cutoff}' - INTERVAL 42 DAY), 0) AS spend_42d,
           COUNT(DISTINCT FLOOR(DATE_DIFF('day', (SELECT data_start FROM params), visit_date) / 7)) AS active_weeks
    FROM days GROUP BY household_id
),
basket_agg AS (
    SELECT household_id,
           AVG(spend)                                                              AS avg_basket_spend,
           COALESCE(AVG(n_units), 0)                                               AS avg_units_per_basket,
           AVG(n_lines)                                                            AS avg_lines_per_basket,
           SUM(fresh_spend)         * 1.0 / NULLIF(SUM(spend), 0)                  AS share_fresh,
           SUM(grocery_spend)       * 1.0 / NULLIF(SUM(spend), 0)                  AS share_grocery,
           SUM(fuel_spend)          * 1.0 / NULLIF(SUM(spend), 0)                  AS share_fuel,
           SUM(health_beauty_spend) * 1.0 / NULLIF(SUM(spend), 0)                  AS share_health_beauty,
           SUM(private_label_spend) * 1.0 / NULLIF(SUM(spend), 0)                  AS share_private_label,
           SUM(retail_disc) * 1.0 / NULLIF(SUM(shelf_value), 0)                    AS retail_disc_share,
           (SUM(coupon_disc) + SUM(coupon_match_disc)) * 1.0 / NULLIF(SUM(shelf_value), 0) AS coupon_disc_share,
           AVG(CASE WHEN has_coupon THEN 1.0 ELSE 0.0 END)                         AS coupon_basket_share,
           COUNT(DISTINCT store_id) FILTER (WHERE basket_date >= DATE '{cutoff}' - INTERVAL 42 DAY) AS n_stores_42d
    FROM hist GROUP BY household_id
),
store_share AS (
    SELECT household_id, MAX(n) * 1.0 / SUM(n) AS top_store_share
    FROM (SELECT household_id, store_id, COUNT(*) AS n FROM hist GROUP BY 1, 2) s GROUP BY household_id
),
camp AS (
    SELECT household_id,
           COUNT(*) FILTER (WHERE start_date < DATE '{cutoff}'
                              AND start_date >= DATE '{cutoff}' - INTERVAL 90 DAY)                       AS n_campaigns_90d,
           COUNT(*) FILTER (WHERE start_date < DATE '{cutoff}' AND end_date >= DATE '{cutoff}')          AS n_campaigns_active,
           COUNT(*) FILTER (WHERE start_date < DATE '{cutoff}' AND start_date >= DATE '{cutoff}' - INTERVAL 90 DAY
                              AND campaign_type = 'Type A')                                              AS n_type_a_90d
    FROM campaign_exposure GROUP BY household_id
),
red AS (
    SELECT household_id,
           COUNT(*) FILTER (WHERE redemption_date < DATE '{cutoff}'
                              AND redemption_date >= DATE '{cutoff}' - INTERVAL 90 DAY)                  AS redemptions_90d,
           COUNT(*) FILTER (WHERE redemption_date < DATE '{cutoff}'
                              AND redemption_date >= DATE '{cutoff}' - INTERVAL 42 DAY)                  AS redemptions_42d
    FROM coupon_redemptions_clean GROUP BY household_id
),
future AS (                             -- label window only
    SELECT household_id, COUNT(*) AS future_trips, SUM(spend) AS future_spend
    FROM fct_days
    WHERE visit_date >= DATE '{cutoff}' AND visit_date < DATE '{cutoff}' + INTERVAL {horizon} DAY
    GROUP BY household_id
)
SELECT DATE '{cutoff}'                                         AS cutoff,
       d.household_id,
       d.recency_days, d.tenure_days, d.frequency, d.monetary_total,
       d.frequency / GREATEST(4.0, d.tenure_days / 7.0)        AS trips_per_week,
       d.monetary_total / GREATEST(4.0, d.tenure_days / 7.0)   AS spend_per_week,
       d.trips_28d, d.trips_42d, d.spend_28d, d.spend_42d,
       d.trips_28d * COALESCE(g.mean_gap, d.tenure_days) / 28.0 AS trip_ratio_28d,
       COALESCE(g.mean_gap, d.tenure_days)                     AS mean_interpurchase_days,
       COALESCE(g.std_gap, 0)                                  AS std_interpurchase_days,
       d.recency_days / GREATEST(COALESCE(g.mean_gap, d.tenure_days, 1), 1) AS recency_to_gap_ratio,
       LEAST(1.0, d.active_weeks / GREATEST(1.0, CEIL(DATE_DIFF('day', DATE '{cutoff}' - CAST(d.tenure_days AS INTEGER),
                                                                DATE '{cutoff}') / 7.0))) AS active_week_share,
       b.avg_basket_spend, b.avg_units_per_basket, b.avg_lines_per_basket,
       COALESCE(b.share_fresh, 0) AS share_fresh, COALESCE(b.share_grocery, 0) AS share_grocery,
       COALESCE(b.share_fuel, 0) AS share_fuel, COALESCE(b.share_health_beauty, 0) AS share_health_beauty,
       COALESCE(b.share_private_label, 0) AS share_private_label,
       COALESCE(b.retail_disc_share, 0) AS retail_disc_share, COALESCE(b.coupon_disc_share, 0) AS coupon_disc_share,
       b.coupon_basket_share, COALESCE(b.n_stores_42d, 0) AS n_stores_42d, s.top_store_share,
       COALESCE(c.n_campaigns_90d, 0) AS n_campaigns_90d, COALESCE(c.n_campaigns_active, 0) AS n_campaigns_active,
       COALESCE(c.n_type_a_90d, 0) AS n_type_a_90d,
       COALESCE(r.redemptions_90d, 0) AS redemptions_90d, COALESCE(r.redemptions_42d, 0) AS redemptions_42d,
       COALESCE(f.future_trips, 0)                             AS future_trips,
       COALESCE(f.future_spend, 0)                             AS future_spend,
       CASE WHEN f.household_id IS NULL THEN 1 ELSE 0 END      AS churned
FROM day_agg d
JOIN pop USING (household_id)
JOIN basket_agg b USING (household_id)
JOIN store_share s USING (household_id)
LEFT JOIN gap_stats g USING (household_id)
LEFT JOIN camp c USING (household_id)
LEFT JOIN red r USING (household_id)
LEFT JOIN future f USING (household_id)
ORDER BY d.household_id
"""


def build_snapshot(con: duckdb.DuckDBPyConnection, cutoff: str | date, horizon_days: int = 90,
                   lookback_days: int = 90, with_label: bool = True) -> pd.DataFrame:
    c = as_date(cutoff)
    sql = FEATURE_SQL.format(cutoff=c.isoformat(), horizon=int(horizon_days), lookback=int(lookback_days))
    df = con.execute(sql).df()
    # round away non-deterministic float noise from DuckDB's parallel aggregation (reproducible runs)
    for col in FEATURE_COLUMNS + DESCRIPTIVE_COLUMNS + ["future_spend"]:
        df[col] = df[col].astype(float).round(6)
    if not with_label:
        df = df.drop(columns=["future_trips", "future_spend", "churned"])
    return df


def data_end_date(con: duckdb.DuckDBPyConnection) -> date:
    return con.execute("SELECT MAX(visit_date) FROM fct_days").fetchone()[0]


def label_window_complete(con: duckdb.DuckDBPyConnection, cutoff: str | date, horizon_days: int) -> bool:
    return (as_date(cutoff) + timedelta(days=horizon_days - 1)) <= data_end_date(con)


def validate_split_design(cfg: dict, data_end: date) -> list[str]:
    """Return a list of design violations (empty = OK).

    Rules: (1) every label window [c, c + h) lies inside the data; (2) every train label window closes before the
    validation cutoff and the validation window closes before the test cutoff (no label leakage across splits);
    (3) cutoffs are ordered; (4) backtest cutoffs are inside the data and each has >= 3 closed training snapshots.
    """
    ch = cfg["churn"]
    h = timedelta(days=ch["horizon_days"])
    tr = [as_date(c) for c in ch["train_cutoffs"]]
    va, te = as_date(ch["valid_cutoff"]), as_date(ch["test_cutoff"])
    grid = [as_date(c) for c in ch["grid_cutoffs"]]
    problems = []
    for c in tr + [va, te] + grid:
        if c + h - timedelta(days=1) > data_end:
            problems.append(f"label window of {c} extends past data end {data_end}")
    if any(c + h > va for c in tr):
        problems.append("a train label window closes after the validation cutoff")
    if va + h > te:
        problems.append("the validation label window closes after the test cutoff")
    if not (max(tr) < va < te):
        problems.append("cutoffs are not ordered train < valid < test")
    for ev in [as_date(c) for c in ch["backtest_eval_cutoffs"]]:
        closed = [c for c in grid if c + h <= ev]
        if len(closed) < 3:
            problems.append(f"backtest cutoff {ev} has only {len(closed)} closed training snapshots")
    return problems


def build_panel(con: duckdb.DuckDBPyConnection, cutoffs: list[str], horizon_days: int,
                lookback_days: int) -> pd.DataFrame:
    return pd.concat([build_snapshot(con, c, horizon_days, lookback_days) for c in cutoffs], ignore_index=True)
