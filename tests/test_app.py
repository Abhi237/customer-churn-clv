"""The Streamlit app imports and every page renders without exceptions (reads committed app/data only)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
PAGES = {"overview": "retention spend", "data_cleaning": "Data and cleaning", "cohorts": "Cohorts",
         "churn_model": "Churn-proxy model", "clv": "lifetime value", "survival": "Survival",
         "campaigns": "Coupons and campaigns", "demographics": "Demographics", "targeting": "targeting simulator",
         "methodology": "Methodology"}


def test_app_data_is_compact_and_current():
    data = APP.parent / "data"
    total = sum(f.stat().st_size for f in data.iterdir())
    assert total < 3e6
    names = {f.name for f in data.iterdir()}
    assert "households.parquet" in names and "demographics_breakout.csv" in names
    assert not {"customers.parquet", "monthly_revenue.csv", "countries.csv"} & names   # stale retail files


def test_app_data_summary_matches_dataset():
    s = json.loads((APP.parent / "data" / "summary.json").read_text())
    assert s["sql_summary"]["n_households"] == 2469
    assert "targeting" in s["config"]


@pytest.mark.parametrize("page,title", PAGES.items())
def test_page_renders(page, title):
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    if page != "overview":
        at.switch_page(f"views/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert title in at.title[0].value


def test_simulator_reacts_to_assumptions():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.switch_page("views/targeting.py").run()
    base = at.metric[0].value
    at.slider[2].set_value(50.0).run()   # much higher ASSUMED contact cost
    assert not at.exception
    assert at.metric[0].value != base


def test_simulator_labels_inputs_as_assumed():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.switch_page("views/targeting.py").run()
    assert all("ASSUMED" in s.label for s in at.slider)


def test_churn_page_household_lookup_changes_with_selection():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.switch_page("views/churn_model.py").run()
    first = [m.value for m in at.metric][-3:]
    at.selectbox[1].select_index(5).run()
    assert not at.exception
    assert [m.value for m in at.metric][-3:] != first


def test_campaign_page_switches_exposure_definition():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.switch_page("views/campaigns.py").run()
    at.radio[0].set_value("Redeemed a coupon before the cutoff").run()
    assert not at.exception


def test_app_text_has_no_emoji_and_no_gbp():
    for f in list((ROOT / "app").rglob("*.py")):
        text = f.read_text()
        assert "\u00a3" not in text, f
        assert not any(0x1F300 <= ord(ch) <= 0x1FAFF or 0x2600 <= ord(ch) <= 0x27BF for ch in text), f


FULL_TITLE = ("Grocery Household Churn & CLV - Using calibrated gradient boosting and BG/NBD models to forecast "
              "household churn and lifetime value on a US grocery loyalty panel")


def test_project_title_is_prefixed_and_consistent():
    import sys
    sys.path.insert(0, str(APP.parent))
    import core
    assert core.TITLE == FULL_TITLE
    assert (ROOT / "README.md").read_text().splitlines()[0] == f"# {FULL_TITLE}"
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert any(core.TITLE_PREFIX in m.value and core.TITLE_METHOD in m.value for m in at.markdown)
