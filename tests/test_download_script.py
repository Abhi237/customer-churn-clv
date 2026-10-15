"""The reproducible data-prep entry points (no network, no R needed)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts" / "download_data.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def test_source_urls_pin_a_release_tag(cfg):
    urls = dl.source_urls(cfg)
    assert urls and all("raw.githubusercontent.com/bradleyboehmke/completejourney/" in u for u in urls.values())
    assert all(cfg["data"]["source_ref"] in u for u in urls.values())
    assert cfg["data"]["source_ref"].startswith("v")


def test_core_tables_match_sql_inputs():
    from churn_clv.db import RAW_TABLES
    assert set(RAW_TABLES) <= set(dl.CORE_TABLES)


def test_count_rows_and_verify(tmp_path, cfg):
    (tmp_path / "transactions.csv").write_text("a,b\n1,2\n3,4\n")
    assert dl.count_rows(tmp_path / "transactions.csv") == 2
    with pytest.raises(SystemExit):
        dl.verify(tmp_path, cfg)


def test_expected_ok_requires_all_tables(tmp_path, cfg):
    assert not dl.expected_ok(tmp_path, cfg)
    for t in dl.CORE_TABLES:
        (tmp_path / f"{t}.csv").write_text("x\n")
    assert dl.expected_ok(tmp_path, cfg)


def test_raw_data_is_gitignored():
    ignore = (ROOT / ".gitignore").read_text()
    assert "data/raw/" in ignore and "*.duckdb" in ignore


def test_r_prep_script_and_makefile_target_exist():
    assert (ROOT / "R" / "prepare_completejourney.R").exists()
    mk = (ROOT / "Makefile").read_text()
    assert "scripts/download_data.py" in mk and "\ndata:" in mk
