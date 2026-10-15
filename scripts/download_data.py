"""Fetch the Complete Journey grocery data (CC0) and convert it to CSV.

Source: the CRAN/GitHub `completejourney` package (https://github.com/bradleyboehmke/completejourney,
tag v1.1.1, licence CC0), which redistributes the 84.51 Complete Journey 2.0 study data
(2,469 households, calendar year 2017). The package ships R objects (.rda/.rds), so the conversion to CSV
uses R (`R/prepare_completejourney.R`, needs Rscript; `data.table` is used if installed).

Usage:
    python scripts/download_data.py            # download + convert (skips work already done)
    python scripts/download_data.py --force    # re-download and re-convert
    python scripts/download_data.py --from-csv DIR   # use CSVs you already have (copy into data/raw)

Manual alternative (no Python): in R, `install.packages("completejourney")`, then
`completejourney::get_transactions()` and write each table with `data.table::fwrite()`; see data/README.md.
Raw data is never committed (data/raw is git-ignored).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE_TABLES = ["transactions", "products", "campaigns", "campaign_descriptions", "coupon_redemptions",
               "demographics", "coupons"]


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "config/config.yaml").read_text())


def source_urls(cfg: dict) -> dict[str, str]:
    base = cfg["data"]["source_base_url"].format(ref=cfg["data"]["source_ref"])
    return {f: f"{base}/{f}" for f in cfg["data"]["source_files"]}


def expected_ok(raw_dir: Path, cfg: dict) -> bool:
    return all((raw_dir / f"{t}.csv").exists() for t in CORE_TABLES)


def count_rows(path: Path) -> int:
    with open(path, "rb") as fh:
        return sum(1 for _ in fh) - 1


def verify(raw_dir: Path, cfg: dict) -> None:
    n = count_rows(raw_dir / "transactions.csv")
    exp = cfg["data"]["expected_transaction_rows"]
    if n != exp:
        raise SystemExit(f"transactions.csv has {n:,} rows, expected {exp:,}: wrong data release?")
    print(f"OK: transactions.csv {n:,} rows; tables present: {', '.join(CORE_TABLES)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--from-csv", type=Path, default=None, help="directory with already converted CSVs")
    args = ap.parse_args()
    cfg = load_cfg()
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    src_dir = raw_dir / "src"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        for t in CORE_TABLES:
            shutil.copy(args.from_csv / f"{t}.csv", raw_dir / f"{t}.csv")
        verify(raw_dir, cfg)
        return
    if expected_ok(raw_dir, cfg) and not args.force:
        print(f"{raw_dir} already populated; use --force to rebuild")
        verify(raw_dir, cfg)
        return
    if shutil.which("Rscript") is None:
        raise SystemExit("Rscript not found. Install R, or convert the package data yourself and rerun with "
                         "--from-csv DIR (see data/README.md).")
    src_dir.mkdir(parents=True, exist_ok=True)
    for name, url in source_urls(cfg).items():
        dest = src_dir / name
        if dest.exists() and not args.force:
            continue
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    subprocess.run(["Rscript", str(ROOT / "R/prepare_completejourney.R"), str(src_dir), str(raw_dir)], check=True)
    verify(raw_dir, cfg)


if __name__ == "__main__":
    main()
