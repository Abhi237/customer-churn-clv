import streamlit as st
from core import DATA_SOURCE, MEMO_URL, REPO_URL, callout, setup_page, summary

s = summary()
ch = s["config"]["churn"]
tr = s["train_summary"]
sq = s["sql_summary"]
cl = s["clv_summary"]
tg = s["config"]["targeting"]
setup_page("Methodology and limitations", "How the numbers were produced, and what they can and cannot support")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Validation design")
    st.markdown(f"""
* **Train** snapshots: {', '.join(ch['train_cutoffs'])} ({tr['n_train_rows']:,} household-snapshots, {tr['n_train_positives']} positives)
* **Validation** {ch['valid_cutoff']}: LightGBM grid, calibrator choice (5-fold CV Brier), final-model selection
  ({tr['n_valid']:,} households, {tr['n_valid_positives']} positives)
* **Test** {ch['test_cutoff']}: reported once ({tr['n_test']:,} households, {tr['n_test_positives']} positives)
* Every {ch['horizon_days']}-day label window closes before the next split's cutoff (checked in code and tests)
* **Rolling-origin backtest**: expanding window at {len(ch['backtest_eval_cutoffs'])} evaluation cutoffs
* **Population**: households with at least one shopping trip in the {ch['lookback_active_days']} days before the cutoff
""")
    st.subheader("Leakage controls")
    st.markdown("""
* Features use only rows dated before the cutoff (baskets, campaign starts and coupon redemptions);
  labels use only `[cutoff, cutoff + 90 days)`
* A unit test perturbs every post-cutoff row and asserts the features are unchanged while labels change
* CLV at the scoring date is fitted on pre-cutoff transactions only
* Features are rates, shares and windows of at most 42 days (campaigns 90 days). Cumulative features such as
  tenure or total spend were removed because with a 12-month panel they encode the calendar date and shift between
  train and test
* Calibration is fitted on validation, never on test
""")
with c2:
    st.subheader("Churn is a proxy")
    callout("A loyalty card has no contract: nobody cancels. <b>Churn = no purchase in the 90 days after the "
            "cutoff.</b> A churned household may return later, or may simply have stopped using its card. "
            "The label measures inactivity, not intent or attrition.")
    st.subheader("Calibration")
    cal = tr["calibration"]["LightGBM"]
    st.markdown(f"""
Isotonic regression was evaluated but is not eligible as the final calibrator with fewer than
{ch['min_positives_isotonic']:,} validation positives ({cal['valid_positives']} here). This rule was
added after a first run in which isotonic won on cross-validated Brier but gave a poor test calibration slope and exact-zero
probabilities. The final model is {tr['final_model']}; on test its Brier score is no better than the uncalibrated
model, so calibration here mainly corrects the average level, not the ranking.
""")
    st.subheader("Assumed vs observed")
    st.markdown(f"""
* **Observed:** metrics, churn rates, CLV holdout errors, segment outcomes in the back-test window
* **ASSUMED:** uplift ({tg['uplift']:.0%}), contact cost (${tg['contact_cost_usd']:.2f}) and gross margin
  ({tg['gross_margin']:.0%}) in the targeting analysis: every ROI figure is a scenario
* Campaign analysis is observational; there is no treatment assignment record or holdout
""")

st.subheader("Limitations")
st.markdown(f"""
* **One calendar year (2017), one retailer, {sq['n_households']:,} loyalty-card households.** No season is observed twice and
  tenure before 2017 is unknown (the January cohort is a stock of established households).
* **Few positives.** About {tr['n_test_positives']} churn-proxy events per evaluation cutoff, so metrics carry wide intervals
  and model differences are within noise. Logistic regression is statistically indistinguishable from LightGBM here.
* **Partial demographics.** {sq['n_households_with_demographics']} of {sq['n_households']:,} households (32%) have demographics, they are
  heavier shoppers, and almost none churn, so demographic churn comparisons are not possible.
* **Loyalty-card data only.** Purchases made without the card are invisible, so a household that stops swiping looks churned.
* **BG/NBD** improves only slightly over a naive per-household rate at household level (MAE {cl['purchases_mae']:.2f} vs
  {cl['naive_rate_mae']:.2f}); aggregate totals are close. It has no seasonality, and 12-month CLV extrapolates beyond
  one observed year.
* **Data cleaning.** This extract contains no negative quantities or sales values, so there are no returns to model;
  {sq['dq_report']['quantity_above_max_lines']:,} lines above the quantity cap (mostly fuel) keep their dollars but not their units.
* **Targeting ROI** is an ASSUMED scenario; high-value households are rarely at risk, so expected gains are small.
""")

st.subheader("Data source and licence")
st.markdown(f"""
[Complete Journey 2.0]({DATA_SOURCE}) from **84.51** (the Kroger analytics subsidiary), distributed in the
`completejourney` R package (v1.1.1) under **CC0 1.0**. Households are anonymised and the data cover calendar year
2017. This repository's code is MIT licensed. Timestamps are naive local (America/New_York) times as shipped.
""")

st.subheader("Reproduce and read more")
c = st.columns(3)
c[0].link_button("GitHub repository", REPO_URL, width="stretch")
c[1].link_button("Executive memo (PDF)", MEMO_URL, width="stretch")
c[2].link_button("Survival report (Quarto)", f"{REPO_URL}/blob/main/reports/survival_report.html", width="stretch")
st.code("make setup && make data && make pipeline && make report && make test\n"
        "make app-data   # refresh app/data/ for this app\n"
        f"churn score --cutoff {ch['test_cutoff']}", language="bash")
st.caption("Dataset: 84.51 (2017), Complete Journey 2.0, via the completejourney R package (Boehmke, B.), CC0.")
