# Customer Lifecycle, Churn & CLV - reproducible pipeline
PY      ?= .venv/bin/python
CHURN   ?= .venv/bin/churn
CUTOFF  ?= 2011-12-10

.PHONY: setup data sql train report test lint score app app-data screenshots all clean

setup:            ## create venv and install pinned deps + package
	python3.11 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e . --no-deps

data:             ## download UCI Online Retail II and convert to parquet
	$(PY) scripts/download_data.py

sql:              ## run numbered DuckDB SQL scripts (cleaning, DQ, cohorts, rolling, RFM, survival)
	$(CHURN) sql

train:            ## churn models + calibration + SHAP, CLV, targeting
	$(CHURN) train
	$(CHURN) clv
	$(CHURN) target

score:            ## score customers at a cutoff, e.g. make score CUTOFF=2011-12-10
	$(CHURN) score --cutoff $(CUTOFF)

report:           ## R survival analysis (Quarto -> HTML, falls back to Rscript), memo PDF, notebooks
	@if command -v quarto >/dev/null 2>&1; then cd reports && quarto render survival_report.qmd --to html; \
	 else Rscript R/survival_analysis.R; fi
	$(PY) scripts/build_memo.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_model_walkthrough.ipynb

lint:
	$(PY) -m ruff check .

test: lint        ## lint + pytest
	$(PY) -m pytest

app-data:         ## export compact aggregated artifacts for the Streamlit app (app/data/, committed)
	$(PY) scripts/export_app_data.py

app:              ## run the multipage Streamlit app on :8501
	.venv/bin/streamlit run app/streamlit_app.py --server.port 8501

screenshots:      ## capture every app page (app must be running on :8501; needs playwright chromium)
	$(PY) scripts/screenshot_app.py

all: data sql train report app-data test

clean:
	rm -rf data/processed artifacts/*.parquet reports/*_files
