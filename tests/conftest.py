"""Shared fixtures: a deterministic synthetic Complete Journey extract + handcrafted edge cases, run through the
SQL model. Nothing here needs the real 1.4M-row raw data (CI-safe)."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from churn_clv.config import load_config
from churn_clv.db import run_sql_model

FIX = Path(__file__).parent / "fixtures"
DEPARTMENTS = [("GROCERY", "National"), ("GROCERY", "Private"), ("PRODUCE", "National"), ("MEAT", "National"),
               ("DELI", "Private"), ("DRUG GM", "National"), ("FUEL", "National"), ("PASTRY", "National")]


def synthetic_raw(n_households: int = 400, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Households with heterogeneous shopping rates and lifetimes over 2017-01-01 .. 2017-12-31."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2017-01-01")
    span = 364
    products = pd.DataFrame({
        "product_id": [str(1000 + i) for i in range(60)],
        "manufacturer_id": "1",
        "department": [DEPARTMENTS[i % len(DEPARTMENTS)][0] for i in range(60)],
        "brand": [DEPARTMENTS[i % len(DEPARTMENTS)][1] for i in range(60)],
        "product_category": "CAT", "product_type": "TYPE", "package_size": "1 EA"})
    rows, basket = [], 31_000_000_000
    for h in range(n_households):
        hid = str(1 + h)
        first = int(rng.integers(0, 120))
        rate = rng.gamma(1.5, 1 / 12)          # shopping days per day
        life = rng.exponential(220)            # active lifetime in days (lapses create churn-proxy events)
        t = first
        stores = [str(int(x)) for x in rng.choice([300, 301, 302, 303], size=2, replace=False)]
        while t < min(span, first + life):
            basket += 1
            n_lines = int(rng.integers(1, 6))
            ts = start + pd.Timedelta(days=int(t), hours=int(rng.integers(8, 21)), minutes=int(rng.integers(0, 59)))
            use_coupon = rng.random() < 0.08
            for _ in range(n_lines):
                p = products.iloc[int(rng.integers(0, 60))]
                q = int(rng.integers(1, 4))
                price = float(rng.choice([1.49, 2.99, 4.5, 7.25]))
                disc = float(rng.choice([0.0, 0.0, 0.3]))
                cd = 0.5 if use_coupon else 0.0
                rows.append((hid, rng.choice(stores), str(basket), p.product_id, q, round(q * price - disc - cd, 2),
                             disc, cd, 0.0, int(t // 7) + 1, ts))
            t += max(1, int(rng.exponential(1 / max(rate, 1e-3))))
    tx = pd.DataFrame(rows, columns=["household_id", "store_id", "basket_id", "product_id", "quantity",
                                     "sales_value", "retail_disc", "coupon_disc", "coupon_match_disc", "week",
                                     "transaction_timestamp"])
    hh = [str(1 + h) for h in range(n_households)]
    desc = pd.DataFrame({
        "campaign_id": [str(i) for i in range(1, 7)],
        "campaign_type": ["Type A", "Type B", "Type C", "Type A", "Type B", "Type A"],
        "start_date": pd.to_datetime(["2017-02-01", "2017-03-15", "2017-05-01", "2017-07-01", "2017-09-01",
                                      "2017-11-01"]).date,
        "end_date": pd.to_datetime(["2017-03-10", "2017-04-20", "2017-06-10", "2017-08-10", "2017-10-10",
                                    "2017-12-10"]).date})
    camp = pd.concat([pd.DataFrame({"campaign_id": cid, "household_id": rng.choice(hh, size=int(n_households * f),
                                                                                   replace=False)})
                      for cid, f in zip(desc.campaign_id, [0.6, 0.2, 0.1, 0.6, 0.2, 0.6])], ignore_index=True)
    red = camp.sample(frac=0.15, random_state=1).assign(coupon_upc="900")
    red = red.merge(desc[["campaign_id", "start_date"]], on="campaign_id")
    red["redemption_date"] = red["start_date"] + pd.Timedelta(days=5)
    red = red[["household_id", "coupon_upc", "campaign_id", "redemption_date"]]
    red["redemption_date"] = pd.to_datetime(red["redemption_date"]).dt.date
    ages = ["19-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    incomes = ["Under 15K", "35-49K", "75-99K", "125-149K", "250K+"]
    demo_ids = rng.choice(hh, size=int(n_households * 0.3), replace=False)
    demo = pd.DataFrame({"household_id": demo_ids, "age": rng.choice(ages, len(demo_ids)),
                         "income": rng.choice(incomes, len(demo_ids)), "home_ownership": "Homeowner",
                         "marital_status": "Married", "household_size": "2",
                         "household_comp": rng.choice(["1 Adult No Kids", "2 Adults Kids"], len(demo_ids)),
                         "kids_count": "0"})
    return {"transactions": tx, "products": products, "campaigns": camp, "campaign_descriptions": desc,
            "coupon_redemptions": red, "demographics": demo}


def edge_tables() -> dict[str, pd.DataFrame]:
    tx = pd.read_csv(FIX / "edge_transactions.csv", dtype={"household_id": str, "store_id": str, "basket_id": str,
                                                           "product_id": str}, parse_dates=["transaction_timestamp"])
    products = pd.read_csv(FIX / "edge_products.csv", dtype={"product_id": str, "manufacturer_id": str})
    empty = {
        "campaigns": pd.DataFrame({"campaign_id": pd.Series(dtype=str), "household_id": pd.Series(dtype=str)}),
        "campaign_descriptions": pd.DataFrame({"campaign_id": pd.Series(dtype=str),
                                               "campaign_type": pd.Series(dtype=str),
                                               "start_date": pd.Series(dtype="datetime64[ns]"),
                                               "end_date": pd.Series(dtype="datetime64[ns]")}),
        "coupon_redemptions": pd.DataFrame({"household_id": pd.Series(dtype=str), "coupon_upc": pd.Series(dtype=str),
                                            "campaign_id": pd.Series(dtype=str),
                                            "redemption_date": pd.Series(dtype="datetime64[ns]")}),
        "demographics": pd.DataFrame({c: pd.Series(dtype=str) for c in [
            "household_id", "age", "income", "home_ownership", "marital_status", "household_size",
            "household_comp", "kids_count"]}),
    }
    return {"transactions": tx, "products": products, **empty}


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_tables():
    return synthetic_raw()


@pytest.fixture(scope="session")
def con(raw_tables, cfg):
    c = duckdb.connect(":memory:")
    run_sql_model(c, source=raw_tables, cfg=cfg, verbose=False)
    yield c
    c.close()


@pytest.fixture(scope="session")
def edge_con(cfg):
    c = duckdb.connect(":memory:")
    run_sql_model(c, source=edge_tables(), cfg=cfg, verbose=False)
    yield c
    c.close()
