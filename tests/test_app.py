"""The Streamlit app imports and every page renders without exceptions (reads committed app/data only)."""
from pathlib import Path

import pytest

pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = {"overview": "retention spend", "data_cleaning": "Data & cleaning", "cohorts": "Cohorts",
         "churn_model": "Churn-proxy model", "clv": "lifetime value", "survival": "Survival",
         "targeting": "targeting simulator", "methodology": "Methodology"}


def test_app_data_is_compact():
    data = APP.parent / "data"
    total = sum(f.stat().st_size for f in data.iterdir())
    assert total < 25e6
    assert not any(f.name.startswith("online_retail") for f in data.iterdir())


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
    at.slider[2].set_value(50.0).run()   # much higher assumed contact cost
    assert not at.exception
    assert at.metric[0].value != base
