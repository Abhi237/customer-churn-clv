# Household churn, CLV and campaign analytics on the Complete Journey grocery panel - reproducible pipeline
PY      ?= .venv/bin/python
CHURN   ?= .venv/bin/churn
CUTOFF  ?= 2017-10-01

.PHONY: setup data sql train clv target campaigns demographics pipeline survival report notebooks test lint score \
        app app-data screenshots all clean

setup:            ## create venv and install pinned deps + package
	python3.11 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e . --no-deps

data:             ## download the completejourney CC0 data (needs R) and convert to CSV in data/raw/completejourney
	$(PY) scripts/download_data.py

sql:              ## run numbered DuckDB SQL scripts (cleaning, DQ, baskets, cohorts, rolling, RFM, campaigns, survival)
	$(CHURN) sql

train:            ## churn models + calibration + bootstrap CIs + backtest + SHAP
	$(CHURN) train

clv:              ## BG/NBD + Gamma-Gamma holdout validation and scoring
	$(CHURN) clv

target:           ## value x risk matrix and ASSUMED-ROI targeting
	$(CHURN) target

campaigns:        ## observational coupon / campaign analysis
	$(CHURN) campaigns

demographics:     ## partial-coverage demographics breakout
	$(CHURN) demographics

pipeline: sql train clv target campaigns demographics survival   ## everything from raw CSVs to artifacts

survival:         ## R survival analysis (landmark KM / Cox); also rendered into the Quarto report by `report`
	Rscript R/survival_analysis.R

score:            ## score households at a cutoff, e.g. make score CUTOFF=2017-10-01
	$(CHURN) score --cutoff $(CUTOFF)

report:           ## Quarto survival report (HTML) and the memo PDF
	@if command -v quarto >/dev/null 2>&1; then cd reports && quarto render survival_report.qmd --to html; \
	 else echo "quarto not found: survival_report.html not rebuilt (run: Rscript R/survival_analysis.R)"; fi
	$(PY) scripts/build_memo.py

notebooks:        ## build and execute the two walkthrough notebooks
	$(PY) scripts/make_notebooks.py
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

all: data pipeline app-data report test

clean:
	rm -rf data/processed artifacts/*.parquet reports/*_files
