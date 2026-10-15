"""DuckDB helpers: register raw inputs, run the numbered SQL model and export tables."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from .config import ROOT, load_config, resolve

SQL_DIR = ROOT / "sql"

RAW_TABLES = ["transactions", "products", "campaigns", "campaign_descriptions", "coupon_redemptions",
              "demographics"]
# explicit reader types: ids are strings (basket ids exceed int32; ids must never be parsed as numbers)
_CSV_TYPES = {
    "transactions": {"household_id": "VARCHAR", "store_id": "VARCHAR", "basket_id": "VARCHAR",
                     "product_id": "VARCHAR", "quantity": "BIGINT", "sales_value": "DOUBLE",
                     "retail_disc": "DOUBLE", "coupon_disc": "DOUBLE", "coupon_match_disc": "DOUBLE",
                     "week": "INTEGER", "transaction_timestamp": "TIMESTAMP"},
    "products": {"product_id": "VARCHAR", "manufacturer_id": "VARCHAR"},
    "campaigns": {"household_id": "VARCHAR", "campaign_id": "VARCHAR"},
    "campaign_descriptions": {"campaign_id": "VARCHAR", "start_date": "DATE", "end_date": "DATE"},
    "coupon_redemptions": {"household_id": "VARCHAR", "coupon_upc": "VARCHAR", "campaign_id": "VARCHAR",
                           "redemption_date": "DATE"},
    "demographics": {"household_id": "VARCHAR", "age": "VARCHAR", "income": "VARCHAR",
                     "home_ownership": "VARCHAR", "marital_status": "VARCHAR", "household_size": "VARCHAR",
                     "household_comp": "VARCHAR", "kids_count": "VARCHAR"},
}


def connect(db_path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    cfg = load_config()
    path = resolve(cfg["paths"]["duckdb"]) if db_path is None else Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def dept_groups_frame(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    rows = [(x, "fresh") for x in d["fresh_departments"]] + \
           [(x, "health_beauty") for x in d["health_beauty_departments"]] + \
           [("GROCERY", "grocery"), ("FUEL", "fuel")]
    return pd.DataFrame(rows, columns=["department", "dept_group"])


def register_inputs(con: duckdb.DuckDBPyConnection, source: dict[str, pd.DataFrame] | str | Path, cfg: dict) -> None:
    """Register the raw tables (as views raw_*) and the parameter tables used by the SQL scripts.

    `source` is either a directory of raw CSVs (as written by scripts/download_data.py) or a dict of DataFrames
    keyed by raw table name (used by the tests with small synthetic fixtures).
    """
    if isinstance(source, dict):
        for name in RAW_TABLES:
            con.register(f"raw_{name}_df", source[name])
            con.execute(f"CREATE OR REPLACE VIEW raw_{name} AS SELECT * FROM raw_{name}_df")
    else:
        src = Path(source)
        for name in RAW_TABLES:
            types = ", ".join(f"'{k}': '{v}'" for k, v in _CSV_TYPES[name].items())
            con.execute(f"CREATE OR REPLACE VIEW raw_{name} AS SELECT * FROM "
                        f"read_csv('{(src / f'{name}.csv').as_posix()}', header = true, types = {{{types}}})")
    dg = dept_groups_frame(cfg)
    con.register("dept_groups_df", dg)
    data_start, data_end = con.execute(
        "SELECT CAST(MIN(transaction_timestamp) AS DATE), CAST(MAX(transaction_timestamp) AS DATE) "
        "FROM raw_transactions").fetchone()
    con.execute(
        "CREATE OR REPLACE TABLE params AS SELECT ?::INTEGER AS max_quantity, ?::DOUBLE AS high_value_line_usd, "
        "?::INTEGER AS horizon_days, ?::INTEGER AS lapse_days, ?::INTEGER AS baseline_days, "
        "?::DATE AS data_start, ?::DATE AS data_end, (?::DATE + 1) AS rfm_as_of",
        [cfg["data"]["max_quantity"], cfg["data"]["high_value_line_usd"], cfg["churn"]["horizon_days"],
         cfg["survival"]["lapse_days"], cfg["survival"]["baseline_days"], data_start, data_end, data_end],
    )


def run_sql_model(con: duckdb.DuckDBPyConnection, source: dict[str, pd.DataFrame] | str | Path | None = None,
                  cfg: dict | None = None, verbose: bool = True) -> list[str]:
    cfg = cfg or load_config()
    if source is None:
        source = resolve(cfg["paths"]["raw_dir"])
    register_inputs(con, source, cfg)
    ran = []
    for script in sorted(SQL_DIR.glob("[0-9][0-9]_*.sql")):
        con.execute(script.read_text())
        ran.append(script.name)
        if verbose:
            print(f"  ran {script.name}")
    return ran


EXPORT_CSV = {"survival_households", "cohort_retention", "dq_report", "dq_cleaning_waterfall", "dim_campaigns",
              "survival_exclusions"}


def export_tables(con: duckdb.DuckDBPyConnection, out_dir: Path, tables: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in tables:
        df = con.execute(f"SELECT * FROM {t}").df()
        df.to_parquet(out_dir / f"{t}.parquet", index=False)
        if t in EXPORT_CSV:
            df.to_csv(out_dir / f"{t}.csv", index=False)


def as_date(d: str | date) -> date:
    return d if isinstance(d, date) else date.fromisoformat(str(d))
