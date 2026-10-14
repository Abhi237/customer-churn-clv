import streamlit as st
from core import MEMO_URL, REPO_URL, callout, setup_page, summary

s = summary()
ch = s["config"]["churn"]
tr = s["train_summary"]
setup_page("Methodology & limitations", "How the numbers were produced, and what they can and cannot support")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Validation design")
    st.markdown(f"""
* **Train** snapshots: {', '.join(ch['train_cutoffs'])} ({tr['n_train_rows']:,} customer-snapshots)
* **Validation** {ch['valid_cutoff']}: LightGBM grid, calibration method (5-fold CV Brier), model selection
* **Test** {ch['test_cutoff']}: reported once ({tr['n_test']:,} customers)
* Every {ch['horizon_days']}-day label window closes before the next split's cutoff
* **Rolling-origin backtest** at four evaluation cutoffs
* **Population**: customers with ≥ 1 purchase in the {ch['lookback_active_days']} days before the cutoff
""")
    st.subheader("Leakage controls")
    st.markdown("""
* All 19 features are computed in SQL with `invoice_date < cutoff`; labels use only `[cutoff, cutoff + 90d)`
* A unit test perturbs **every** post-cutoff row (quantities, prices, 50% dropped) and asserts features are
  byte-identical while labels change
* CLV at the scoring date is fitted on pre-cutoff transactions only (also tested)
* Calibration is fitted on validation, never on test; test is touched once
""")
with c2:
    st.subheader("Churn is a proxy")
    callout("This is a non-contractual business: nobody cancels. <b>Churn = no purchase in the 90 days after the "
            "cutoff.</b> A 'churned' customer may come back later; the label measures inactivity, not intent.")
    st.subheader("Calibration drift")
    cil = tr["calibration_in_the_large_test"][tr["final_model"]]
    st.markdown(f"""
At the test cutoff the final model predicts **{cil['mean_pred']:.1%}** average risk vs **{cil['observed']:.1%}**
observed. Churn rates by cutoff range from {min(tr['churn_rate_by_train_cutoff'].values()):.1%} to
{max(tr['churn_rate_by_train_cutoff'].values()):.1%} - strongly seasonal. A same-season prior adjustment was
tested and rejected because it failed on the validation cutoff. Rank customers with the scores; recalibrate on
the latest comparable season before using absolute probabilities.
""")
    st.subheader("Assumed vs observed")
    st.markdown("""
* **Observed:** metrics, churn rates, CLV holdout errors, segment outcomes in the back-test window
* **Assumed:** uplift, contact cost, margin → every ROI figure is a scenario
* No treatment data exists, so causal uplift cannot be estimated - a randomised hold-out test is the next step
""")

st.subheader("Other limitations")
st.markdown(f"""
* One wholesaler, two years (each season seen at most twice), {s['sql_summary']['uk_customer_share']:.0%} UK customers
* {s['sql_summary']['dq_report']['missing_customer_id']:,} raw lines without a customer ID are excluded (they cannot be
  linked to a lifecycle)
* Dec-2009 customers are left-truncated (already active before the extract) - excluded from survival analysis
* BG/NBD has no seasonality; spend is under-forecast in the Q4-heavy holdout
* Logistic regression beat LightGBM; the lift over a simple RFM score is real but modest
""")

st.subheader("Reproduce & read more")
c = st.columns(3)
c[0].link_button("📂 GitHub repository", REPO_URL, width="stretch")
c[1].link_button("📄 Executive memo (PDF)", MEMO_URL, width="stretch")
c[2].link_button("📊 Survival report (Quarto)", f"{REPO_URL}/blob/main/reports/survival_report.html",
                 width="stretch")
st.code("make setup && make data && make sql && make train && make report && make test\n"
        "make app-data   # refresh app/data/ for this app\n"
        "churn score --cutoff 2011-12-10", language="bash")
st.caption("Dataset: Chen, D. (2012). Online Retail II. UCI Machine Learning Repository, CC BY 4.0. "
           "Code: MIT licence.")
