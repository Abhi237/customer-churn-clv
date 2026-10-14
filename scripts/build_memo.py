"""Generate the executive retention memo (markdown + PDF) from computed artifacts only.

Every number in the memo is read from artifacts/ produced by the pipeline; nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import markdown
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART, DOCS = ROOT / "artifacts", ROOT / "docs"


def load():
    j = {n: json.loads((ART / f"{n}.json").read_text()) for n in
         ["sql_summary", "train_summary", "clv_summary", "targeting_summary"]}
    m = pd.read_csv(ART / "model_metrics.csv")
    mat = pd.read_csv(ART / "targeting_matrix.csv")
    sens = pd.read_csv(ART / "targeting_sensitivity.csv")
    km = pd.read_csv(ART / "r_survival/km_overall.csv").set_index("time")
    kmv = pd.read_csv(ART / "r_survival/km_by_first_value_tier.csv")
    cox = pd.read_csv(ART / "r_survival/cox_stratified_hazard_ratios.csv").set_index("term")
    return j, m, mat, sens, km, kmv, cox


def gbp(x):
    return f"GBP {x / 1e6:,.2f}m" if abs(x) >= 1e6 else f"GBP {x / 1e3:,.0f}k" if abs(x) >= 1e4 else f"GBP {x:,.0f}"


def build() -> str:
    j, m, mat, sens, km, kmv, cox = load()
    s, t, c, g = j["sql_summary"], j["train_summary"], j["clv_summary"], j["targeting_summary"]
    test = m[(m.split == "test")].set_index(["model", "calibration"])
    lr, rec, rfm = test.loc[("Logistic regression", "none")], test.loc[("Recency rule", "none")], \
        test.loc[("RFM heuristic", "none")]
    final = t["final_model"]
    fm = test.loc[tuple(final.split(" + "))] if " + " in final else lr
    cil = t["calibration_in_the_large_test"][final]
    hv_lr = mat[(mat.value_segment == "High value") & (mat.risk_segment == "Low risk")].iloc[0]
    mid_med = mat[(mat.value_segment == "Mid value") & (mat.risk_segment == "Medium risk")].iloc[0]
    mid_risk = mat[(mat.value_segment == "Mid value") & (mat.risk_segment != "Low risk")]
    hv_risk = mat[(mat.value_segment == "High value") & (mat.risk_segment != "Low risk")]
    b = g["base_case_ASSUMED"]
    q1 = kmv[(kmv.time == 90) & kmv.strata.str.contains("Q1")].survival.iloc[0]
    q4 = kmv[(kmv.time == 90) & kmv.strata.str.contains("Q4")].survival.iloc[0]
    hr = cox.loc["log_first_value"]
    cons = sens[(sens.uplift == 0.05) & (sens.gross_margin == 0.3) & (sens.contact_cost_gbp == 10)].iloc[0]
    tot_obs_rev = mat.observed_revenue_90d.sum()

    md = f"""# Retention Strategy Memo - Online Retail II

**To:** Commercial leadership | **From:** Customer Analytics | **Scoring date:** {g_date(t)} (back-test)
**Basis:** UCI Online Retail II, {s['date_min']} to {s['date_max']}; every figure below is computed by the
pipeline in this repository. Items marked **ASSUMED** are scenario inputs, not observed results.

## 1. The decision
Which active customers should receive retention spend in the next quarter, and how much is their future
purchasing plausibly worth?

## 2. What the data says (observed)
* **{s['n_customers_clean']:,} identifiable customers** placed {s['n_invoices_clean']:,} clean invoices worth
  {gbp(s['revenue_clean_gbp'])}. {s['one_time_buyer_share']:.0%} bought only once.
* **Activation is the biggest leak.** Only {km.loc[1, 'survival']:.0%} of new customers made a repeat purchase
  within 90 days of their first order; {km.loc[365, 'survival']:.1%} were still active without any 90-day lapse
  after a year (Kaplan-Meier). Customers whose first order was in the top value quartile stayed active at
  90 days at {q4:.0%} vs {q1:.0%} for the bottom quartile; each 1-unit increase in log first-order value cuts the
  lapse hazard by {1 - hr.hr:.0%} (stratified Cox HR {hr.hr:.2f}, 95% CI {hr.lower95:.2f}-{hr.upper95:.2f}).
* **Churn proxy** (no purchase in the 90 days after the scoring date - a behavioural proxy, not a cancellation):
  {t['churn_rate']['test']:.1%} of the {t['n_test']:,} customers active in the prior year at {g_date(t)}.

![](figures/km_overall.png) ![](figures/roc_pr_test.png)

## 3. How well can we predict it (out-of-time test, {g_date(t)})
| Approach | ROC-AUC | PR-AUC | Lift, top 10% |
|---|---|---|---|
| Recency rule (baseline) | {rec.roc_auc:.3f} | {rec.pr_auc:.3f} | {rec.lift_top10:.2f}x |
| RFM heuristic (baseline) | {rfm.roc_auc:.3f} | {rfm.pr_auc:.3f} | {rfm.lift_top10:.2f}x |
| **{final}** (selected on validation) | **{fm.roc_auc:.3f}** | **{fm.pr_auc:.3f}** | **{fm.lift_top10:.2f}x** |

The model ranks risk better than the baselines, but the gain over a simple RFM score is modest
({fm.roc_auc - rfm.roc_auc:+.3f} ROC-AUC). **Calibration caveat:** at this pre-Christmas scoring date the model
predicted an average churn risk of {cil['mean_pred']:.0%} versus {cil['observed']:.0%} observed, consistent with
seasonal Q4 demand lowering churn (training windows mix seasons). Use the scores to *rank* customers; re-calibrate on the latest season before using
them as absolute probabilities.

**Value forecast (BG/NBD + Gamma-Gamma):** on a {c['holdout_days']:.0f}-day holdout the model predicted
{c['pred_purchases_total']:,.0f} repeat purchases vs {c['actual_purchases_total']:,.0f} actual and
{gbp(c['pred_spend_total'])} spend vs {gbp(c['actual_spend_total'])} actual (spend under-forecast by
{1 - c['pred_spend_total'] / c['actual_spend_total']:.0%}; the holdout contains the Q4 peak and the model has no
seasonality). The top 10% of customers by
predicted value delivered {c['top10_share_of_actual_spend']:.0%} of actual holdout spend.

## 4. Value x risk segments at {g_date(t)}
* **Protect - High value / Low risk:** {int(hv_lr.customers):,} customers, {gbp(hv_lr.total_clv_12m)} predicted 12-month
  CLV ({hv_lr.total_clv_12m / mat.total_clv_12m.sum():.0%} of the total). Observed next-90-day churn was only
  {hv_lr.observed_churn_rate:.0%} and they generated {hv_lr.observed_revenue_90d / tot_obs_rev:.0%} of next-quarter revenue.
  Service, not discounts.
* **Save - High value at Medium/High risk:** {int(hv_risk.customers.sum()):,} customers, {gbp(hv_risk.total_clv_12m.sum())}
  CLV; observed churn {(hv_risk.observed_churn_rate * hv_risk.customers).sum() / hv_risk.customers.sum():.0%}.
  Highest priority for personal outreach.
* **Grow - Mid value at Medium/High risk:** {int(mid_risk.customers.sum()):,} customers, {gbp(mid_risk.total_clv_12m.sum())}
  CLV (Mid/Medium alone: {int(mid_med.customers):,}). Low-cost automated win-back.
* **Low value / High risk:** {int(mat.iloc[2].customers):,} customers, mean CLV {gbp(mat.iloc[2].mean_clv_12m)}:
  email-only or no spend.

![](figures/targeting_matrix.png)

## 5. Recommended targeting rule (scenario - ASSUMED inputs)
Contact a customer when *P(churn) x uplift x CLV(12m) x margin > contact cost*.
With **ASSUMED** {b['uplift']:.0%} relative churn reduction, {b['gross_margin']:.0%} margin and GBP {b['contact_cost_gbp']:.0f}
per contact, the rule selects {b['n_targeted']:,} customers ({b['share_targeted']:.0%}), costing
{gbp(b['campaign_cost_gbp'])} with an expected net value of {gbp(b['expected_net_value_gbp'])}. Under a conservative
scenario ({cons.uplift:.0%} uplift, GBP {cons.contact_cost_gbp:.0f} contact) it selects {int(cons.n_targeted):,}
customers for {gbp(cons.expected_net_value_gbp)}. Across the {len(sens)}-scenario grid the expected net value ranges
from {gbp(g['sensitivity_range_net_value_gbp'][0])} to {gbp(g['sensitivity_range_net_value_gbp'][1])}. **These are not
observed impacts**; the probabilities feeding them are over-stated at this date (section 3).

## 6. Next steps
1. Run a randomised hold-out test (e.g. 20% control) in the Save and Grow cells to *measure* uplift; replace the
   assumption with the measured value.
2. Launch a second-order activation programme for new customers (the day-1 lapse is the largest loss).
3. Re-score monthly with `churn score --cutoff <date>` and re-calibrate on the most recent comparable season.

*Limitations: wholesale-heavy, {s['uk_customer_share']:.0%} UK customer base; 2 years of history (one Q4 per
split); {s['dq_report']['missing_customer_id']:,} raw lines lacked a customer ID and are excluded; churn is a proxy.*
"""
    return md


def g_date(t):
    import yaml

    return yaml.safe_load((ROOT / "config/config.yaml").read_text())["churn"]["test_cutoff"]


CSS = """
@page { size: A4; margin: 14mm 15mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 8.6pt; line-height: 1.32; color: #1a1a1a; }
h1 { font-size: 14pt; color: #1f4e79; margin: 0 0 4px 0; border-bottom: 2px solid #1f4e79; }
h2 { font-size: 10pt; color: #1f4e79; margin: 8px 0 3px 0; }
ul { margin: 2px 0 2px 16px; padding: 0; } li { margin-bottom: 2px; } p { margin: 3px 0; }
table { border-collapse: collapse; margin: 4px 0; } th, td { border: 1px solid #bbb; padding: 2px 7px; }
th { background: #e8eef5; }
img { width: 49%; } p:has(img) { text-align: center; }
"""


def main():
    md = build()
    (DOCS / "retention_memo.md").write_text(md)
    html = f"<html><head><style>{CSS}</style></head><body>{markdown.markdown(md, extensions=['tables', 'attr_list'])}</body></html>"
    from weasyprint import HTML

    doc = HTML(string=html, base_url=str(DOCS)).render()
    doc.write_pdf(DOCS / "retention_memo.pdf")
    print(f"memo written ({len(doc.pages)} pages): {DOCS / 'retention_memo.pdf'}")


if __name__ == "__main__":
    main()
