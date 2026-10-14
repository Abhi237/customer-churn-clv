"""DuckDB helpers: run the numbered SQL model and export tables."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from .config import ROOT, load_config, resolve

SQL_DIR = ROOT / "sql"


def connect(db_path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    cfg = load_config()
    path = resolve(cfg["paths"]["duckdb"]) if db_path is None else Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def register_inputs(con: duckdb.DuckDBPyConnection, source: pd.DataFrame | str | Path, cfg: dict) -> None:
    """Register raw source + parameter tables used by the SQL scripts."""
    if isinstance(source, pd.DataFrame):
        con.register("raw_source_df", source)
        con.execute("CREATE OR REPLACE VIEW raw_source AS SELECT * FROM raw_source_df")
    else:
        con.execute(f"CREATE OR REPLACE VIEW raw_source AS SELECT * FROM read_parquet('{Path(source).as_posix()}')")
    codes = pd.DataFrame({"code": [c.upper() for c in cfg["data"]["non_product_codes"]]})
    con.register("non_product_codes_df", codes)
    con.execute("CREATE OR REPLACE TABLE non_product_codes AS SELECT * FROM non_product_codes_df")
    data_end = con.execute("SELECT CAST(MAX(invoice_date) AS DATE) FROM raw_source").fetchone()[0]
    con.execute(
        "CREATE OR REPLACE TABLE params AS SELECT ?::INTEGER AS max_abs_quantity, ?::INTEGER AS horizon_days, "
        "?::DATE AS data_end, (?::DATE + 1) AS rfm_as_of",
        [cfg["data"]["max_abs_quantity"], cfg["churn"]["horizon_days"], data_end, data_end],
    )


def run_sql_model(con: duckdb.DuckDBPyConnection, source: pd.DataFrame | str | Path | None = None,
                  cfg: dict | None = None, verbose: bool = True) -> list[str]:
    cfg = cfg or load_config()
    if source is None:
        source = resolve(cfg["paths"]["interim_parquet"])
    register_inputs(con, source, cfg)
    ran = []
    for script in sorted(SQL_DIR.glob("[0-9][0-9]_*.sql")):
        con.execute(script.read_text())
        ran.append(script.name)
        if verbose:
            print(f"  ran {script.name}")
    return ran


def export_tables(con: duckdb.DuckDBPyConnection, out_dir: Path, tables: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in tables:
        df = con.execute(f"SELECT * FROM {t}").df()
        df.to_parquet(out_dir / f"{t}.parquet", index=False)
        if t in {"survival_customers", "cohort_retention", "dq_report", "dq_cleaning_waterfall"}:
            df.to_csv(out_dir / f"{t}.csv", index=False)


def as_date(d: str | date) -> date:
    return d if isinstance(d, date) else date.fromisoformat(str(d))
