"""Repository-level checks: no emoji, no leftover retail framing in code, CI does not need raw data."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".qmd", ".R", ".sql", ".toml", ".txt"}


def repo_files():
    skip = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", "raw"}
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in TEXT_SUFFIXES and not (set(p.relative_to(ROOT).parts) & skip):
            yield p


def test_no_emoji_anywhere():
    bad = []
    for p in repo_files():
        text = p.read_text(errors="ignore")
        if any(0x1F300 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF or ord(c) == 0xFE0F for c in text):
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad


def test_no_leftover_online_retail_code():
    needles = ("invoice_date", "customer_id", "fct_invoices", "home_country", "StockCode")
    bad = []
    for p in repo_files():
        if p.suffix in {".py", ".sql", ".R", ".qmd"} and p.relative_to(ROOT).parts[0] != "tests":
            text = p.read_text(errors="ignore")
            if any(n in text for n in needles):
                bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad


def test_ci_workflow_does_not_download_raw_data():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "download_data" not in text and "make data" not in text
    assert "pytest" in text and "ruff" in text
    assert wf["jobs"]
