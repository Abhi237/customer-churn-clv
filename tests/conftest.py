"""Shared fixtures: a deterministic synthetic raw extract + handcrafted edge cases, run through the SQL model."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from churn_clv.config import load_config
from churn_clv.db import run_sql_model

FIX = Path(__file__).parent / "fixtures"


def synthetic_raw(n_customers: int = 300, seed: int = 7) -> pd.DataFrame:
    """Customers with heterogeneous purchase rates over 2010-01-01 .. 2011-12-09."""
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2010-01-01"), pd.Timestamp("2011-12-09")
    span = (end - start).days
    rows, inv = [], 100000
    products = [f"2{i:04d}" for i in range(40)]
    for c in range(n_customers):
        cid = 10000 + c
        first = rng.integers(0, span - 30)
        rate = rng.gamma(1.0, 1 / 40)           # purchases per day
        life = rng.exponential(300)              # active lifetime in days
        t = first
        country = "United Kingdom" if rng.random() < 0.85 else "Germany"
        while t < min(span, first + life):
            inv += 1
            for _ in range(rng.integers(1, 5)):
                rows.append((str(inv), str(rng.choice(products)), "ITEM", int(rng.integers(1, 24)),
                             start + pd.Timedelta(days=int(t), hours=10), float(rng.choice([0.85, 1.65, 2.95, 4.95])),
                             float(cid), country, "synthetic"))
            if rng.random() < 0.05:   # occasional cancellation
                rows.append(("C" + str(inv), rows[-1][1], "ITEM", -1, start + pd.Timedelta(days=int(t) + 1, hours=9),
                             rows[-1][5], float(cid), country, "synthetic"))
            t += max(1, int(rng.exponential(1 / max(rate, 1e-3))))
    df = pd.DataFrame(rows, columns=["invoice", "stockcode", "description", "quantity", "invoice_date", "price",
                                     "customer_id", "country", "source_sheet"])
    edge = pd.read_csv(FIX / "edge_cases.csv", dtype={"invoice": str, "stockcode": str}, parse_dates=["invoice_date"])
    return pd.concat([df, edge], ignore_index=True)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_df():
    return synthetic_raw()


@pytest.fixture(scope="session")
def con(raw_df, cfg):
    c = duckdb.connect(":memory:")
    run_sql_model(c, source=raw_df, cfg=cfg, verbose=False)
    yield c
    c.close()


@pytest.fixture(scope="session")
def edge_con(cfg):
    edge = pd.read_csv(FIX / "edge_cases.csv", dtype={"invoice": str, "stockcode": str}, parse_dates=["invoice_date"])
    c = duckdb.connect(":memory:")
    run_sql_model(c, source=edge, cfg=cfg, verbose=False)
    yield c
    c.close()
