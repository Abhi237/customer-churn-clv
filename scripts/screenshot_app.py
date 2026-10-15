"""Capture screenshots of every app page (requires a running app on :8501 and playwright chromium)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "docs" / "app_screenshots"
PAGES = [("01_overview", ""), ("02_data_cleaning", "data_cleaning"), ("03_cohorts", "cohorts"),
         ("04_churn_model", "churn_model"), ("05_clv", "clv"), ("06_survival", "survival"),
         ("07_campaigns", "campaigns"), ("08_demographics", "demographics"), ("09_targeting", "targeting"),
         ("10_methodology", "methodology")]

OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
    for name, path in PAGES:
        pg.goto(f"{BASE}/{path}", wait_until="networkidle")
        pg.wait_for_selector("h1", timeout=60000)
        pg.wait_for_timeout(4000)
        h = pg.evaluate("() => { const m = document.querySelector('[data-testid=\"stMain\"]') ||"
                        " document.querySelector('section.main'); return m ? m.scrollHeight : 2000; }")
        pg.set_viewport_size({"width": 1440, "height": min(max(int(h) + 40, 900), 6000)})
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / f"{name}.png"))
        pg.set_viewport_size({"width": 1440, "height": 900})
        exc = pg.locator("[data-testid='stException']").count()
        print(name, h, "exceptions:", exc)
    b.close()
