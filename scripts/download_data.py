"""Download UCI Online Retail II and convert both Excel sheets to a single parquet file.

Usage: python scripts/download_data.py [--force]
Raw data is never committed (see data/README.md).
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    p = cfg["paths"]
    zip_path, xlsx, out = ROOT / p["raw_zip"], ROOT / p["raw_xlsx"], ROOT / p["interim_parquet"]
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not xlsx.exists():
        if args.force or not zip_path.exists():
            print(f"Downloading {cfg['data']['url']} ...")
            urllib.request.urlretrieve(cfg["data"]["url"], zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(zip_path.parent)
    if out.exists() and not args.force:
        print(f"{out} exists; use --force to rebuild")
        return
    print("Reading Excel sheets (takes ~1-2 min) ...")
    sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    frames = []
    for name, df in sheets.items():
        df["source_sheet"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    df = df.rename(columns={"customer_id": "customer_id", "invoicedate": "invoice_date"})
    df["description"] = df["description"].astype("string")
    df["country"] = df["country"].astype("string")
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
