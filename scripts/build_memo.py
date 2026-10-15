"""Generate the executive retention memo (markdown + PDF) from computed artifacts only.

Every number in the memo is read from artifacts/ produced by the pipeline; nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import markdown
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ART, DOCS = ROOT / "artifacts", ROOT / "docs"


def load():
    names = ["sql_summary", "train_summary", "clv_summary", "targeting_summary", "campaign_summary",
             "demographics_summary"]
    j = {n: json.loads((ART / f"{n}.json").read_text()) for n in names}
    tables = {
        "metrics": pd.read_csv(ART / "model_metrics.csv"), "ci": pd.read_csv(ART / "test_metric_ci.csv"),
        "bt": pd.read_csv(ART / "backtest_metrics.csv"), "mat": pd.read_csv(ART / "targeting_matrix.csv"),
        "sens": pd.read_csv(ART / "targeting_sensitivity.csv"), "shap": pd.read_csv(ART / "shap_importance.csv"),
        "km": pd.read_csv(ART / "r_survival/km_overall.csv").set_index("time"),
        "kmt": pd.read_csv(ART / "r_survival/km_by_trip_tier.csv"),
        "cox": pd.read_csv(ART / "r_survival/cox_stratified_hazard_ratios.csv").set_index("term"),
        "cov": pd.read_csv(ART / "demographics_coverage.csv").set_index("group"),
        "ct": pd.read_csv(ART / "campaign_type_summary.csv").set_index("campaign_type"),
    }
    return j, tables, yaml.safe_load((ROOT / "config/config.yaml").read_text())


def usd(x):
    sign = "-" if x < 0 else ""
    a = abs(x)
    return sign + (f"${a / 1e6:,.2f}M" if a >= 1e6 else f"${a / 1e3:,.0f}K" if a >= 1e4 else f"${a:,.0f}")


def build() -> str:
    j, t, cfg = load()
    s, tr, cl, tg = j["sql_summary"], j["train_summary"], j["clv_summary"], j["targeting_summary"]
    cs, ds = j["campaign_summary"], j["demographics_summary"]
    ch = cfg["churn"]
    m = t["metrics"][t["metrics"].split == "test"].set_index(["model", "calibration"])
    ci = t["ci"].set_index("model")
    final = tr["final_model"]
    base, _, cal = final.partition(" + ")
    fm = m.loc[(base, cal or "none")]
    lr, lg, rfm, rec = (m.loc[(k, "none")] for k in ("Logistic regression", "LightGBM", "RFM heuristic",
                                                      "Recency rule"))
    pb = tr["paired_bootstrap_test"]
    p_ll, p_lr = pb["lightgbm_minus_logreg_roc_auc"], pb["logreg_minus_rfm_roc_auc"]
    bt = t["bt"].pivot(index="eval_cutoff", columns="model", values="roc_auc")
    wins = int((bt["LightGBM"] >= bt["Logistic regression"]).sum())
    cil = tr["calibration_in_the_large_test"]
    calib = tr["calibration"]["LightGBM"]
    mat = t["mat"]
    hv = mat[(mat.value_segment == "High value") & (mat.risk_segment == "Low risk")].iloc[0]
    hv_risk = mat[(mat.value_segment == "High value") & (mat.risk_segment != "Low risk")]
    lv_hr = mat[(mat.value_segment == "Low value") & (mat.risk_segment == "High risk")].iloc[0]
    b = tg["base_case_ASSUMED"]
    sens = t["sens"]
    km = t["km"]
    kmt = t["kmt"][t["kmt"].time == 90].set_index("strata")["survival"]
    hr = t["cox"].loc["log_trips"]
    cov = t["cov"]
    top = ", ".join(t["shap"].head(4)["feature"].str.replace("_", " "))
    adj = cs["adjusted_difference_targeted"]
    ct = t["ct"]

    return f"""# Retention Strategy Memo - US Grocery Loyalty Panel

**To:** Commercial leadership | **From:** Customer Analytics | **Scoring date:** {ch['test_cutoff']} (back-test)
**Basis:** Complete Journey 2.0 (84.51, CC0), {s['n_households']:,} loyalty-card households, {s['date_min']} to
{s['date_max']}. Every figure is computed by the pipeline in this repository. Items marked **ASSUMED** are
scenario inputs, not observed results. **Churn is a proxy**: no purchase in the {ch['horizon_days']} days after the
cutoff, not an observed cancellation.

## 1. The decision
Which active households should receive retention spend (coupons, outreach) next quarter, and how much is their
future spend plausibly worth?

## 2. What the data says (observed)
* {s['n_households']:,} households made {s['n_baskets_clean']:,} clean baskets worth {usd(s['spend_clean_usd'])} across
  {s['n_stores']} stores. The top 10% of households account for {s['top10pct_spend_share']:.0%} of spend.
* Store discounts are {s['retail_disc_share_of_shelf']:.1%} of shelf value and coupon discounts only
  {s['coupon_disc_share_of_shelf']:.2%}; {s['baskets_with_coupon_share']:.1%} of baskets use a coupon.
* Households rarely go quiet: {km.loc[90, 'survival']:.1%} remain lapse-free at day 90 (Kaplan-Meier, landmark
  design). Baseline trip frequency separates them sharply: day-90 survival is {kmt['Low (T1)']:.1%} for the lowest
  trip tier and {kmt['High (T3)']:.1%} for the highest (Cox hazard ratio per log trip {hr['hr']:.2f}, 95% CI
  {hr['lower95']:.2f} to {hr['upper95']:.2f}; the proportional-hazards test flags this effect as time-varying).

## 3. Churn-proxy model (out-of-time test, {tr['n_test']:,} households, {tr['n_test_positives']} positives)
* Prevalence is {tr['churn_rate']['test']:.1%} (train {tr['churn_rate']['train']:.1%}, validation
  {tr['churn_rate']['valid']:.1%}).
* Test ROC-AUC with 95% bootstrap intervals: recency rule {rec.roc_auc:.3f} ({ci.loc['Recency rule', 'roc_auc_lo']:.3f} to
  {ci.loc['Recency rule', 'roc_auc_hi']:.3f}), RFM heuristic {rfm.roc_auc:.3f}, logistic regression {lr.roc_auc:.3f}
  ({ci.loc['Logistic regression', 'roc_auc_lo']:.3f} to {ci.loc['Logistic regression', 'roc_auc_hi']:.3f}), LightGBM
  {lg.roc_auc:.3f} ({ci.loc['LightGBM', 'roc_auc_lo']:.3f} to {ci.loc['LightGBM', 'roc_auc_hi']:.3f}).
  PR-AUC: {rec.pr_auc:.3f}, {rfm.pr_auc:.3f}, {lr.pr_auc:.3f}, {lg.pr_auc:.3f} against a {tr['churn_rate']['test']:.3f} base rate.
* **LightGBM is not distinguishable from logistic regression**: paired-bootstrap ROC-AUC difference {p_ll['diff']:+.3f}
  ({p_ll['lo']:+.3f} to {p_ll['hi']:+.3f}). Logistic regression beats the RFM heuristic by {p_lr['diff']:+.3f}
  ({p_lr['lo']:+.3f} to {p_lr['hi']:+.3f}). In the rolling backtest LightGBM matches or beats logistic regression at
  {wins} of {len(bt)} cutoffs.
* Final model: {final} (best validation PR-AUC); top-decile lift {fm.lift_top10:.2f}x. **Calibration did not improve
  Brier on test**: raw LightGBM {lg.brier:.4f}, final {fm.brier:.4f}. Mean predicted risk is
  {cil['LightGBM']['mean_pred']:.1%} raw and {cil[final]['mean_pred']:.1%} after Platt scaling versus
  {cil['LightGBM']['observed']:.1%} observed; calibration slope {fm.calib_slope:.2f}. Isotonic was ineligible with
  {calib['valid_positives']} validation positives. Use scores for ranking.
* Main drivers (mean |SHAP|): {top}.

## 4. Lifetime value (BG/NBD + Gamma-Gamma, {cl['holdout_days']:.0f}-day holdout)
Predicted holdout shopping days are {cl['pred_purchases_total']:,.0f} versus {cl['actual_purchases_total']:,.0f} actual
({cl['pred_purchases_total'] / cl['actual_purchases_total'] - 1:+.1%}); spend {usd(cl['pred_spend_total'])} versus
{usd(cl['actual_spend_total'])} ({cl['pred_spend_total'] / cl['actual_spend_total'] - 1:+.1%}). Household-level MAE is
{cl['purchases_mae']:.2f} shopping days against {cl['naive_rate_mae']:.2f} for a naive carried-forward rate: the model is
right in aggregate and for ranking (Spearman {cl['spend_spearman']:.2f}), but only slightly better per household.

## 5. Coupons and campaigns (observational, no causal claim)
* {s['n_campaigns']} campaigns reached {s['n_households_targeted_any']:,} households; {s['n_households_ever_redeemed']:,}
  redeemed at least one coupon. Type A campaigns reach about {ct.loc['Type A', 'median_targeted']:,.0f} households each
  and have a {ct.loc['Type A', 'redeeming_per_targeted']:.1%} redeeming share of targeted, versus
  {ct.loc['Type B', 'redeeming_per_targeted']:.1%} for Type B.
* Households targeted in the test window churn less (naive gap {adj['naive_diff'] * 100:+.1f} points), but within
  model risk tiers the gap is {adj['tier_standardised_diff'] * 100:+.1f} points: targeting reached heavier shoppers
  (largest baseline standardised difference {cs['max_abs_smd_targeted']:.2f}). There is no holdout, so the data
  cannot say whether campaigns work.

## 6. Value x risk targeting (ASSUMED ROI)
* {int(hv.households):,} High value / Low risk households hold {hv.total_clv_12m / mat.total_clv_12m.sum():.0%} of predicted
  CLV (observed churn {hv.observed_churn_rate:.1%}): protect with service, not discounts. Only
  {int(hv_risk.households.sum())} households are both high value and at elevated risk.
* Most risk sits in Low value households ({int(lv_hr.households):,} High risk, observed churn
  {lv_hr.observed_churn_rate:.1%}).
* **ASSUMED** {tg['base_case_ASSUMED']['uplift']:.0%} relative churn reduction, {b['gross_margin']:.0%} margin,
  ${b['contact_cost_usd']:.2f} per contact: the rule targets {b['n_targeted']} households
  ({b['share_targeted']:.1%}) for expected net value {usd(b['expected_net_value_usd'])} on
  {usd(b['campaign_cost_usd'])} of cost; contacting everyone would be {usd(b['expected_net_value_target_all_usd'])}.
  Across the ASSUMED sensitivity grid net value ranges from {usd(sens.expected_net_value_usd.min())} to
  {usd(sens.expected_net_value_usd.max())}. These are scenarios, not forecasts.

## 7. Demographics (partial coverage)
Only {ds['n_with_demographics']} of {ds['n_households']:,} households ({ds['coverage_share']:.0%}) have demographics. They are
heavier shoppers (median spend to date {usd(cov.loc['With demographics', 'median_monetary_total'])} versus
{usd(cov.loc['Without demographics', 'median_monetary_total'])}) and only {ds['n_churned_with_demographics']} of
{ds['n_scored_with_demographics']} scored covered households went quiet, so group churn rates are not informative.

## 8. Recommendation and next step
Protect high-value households with service; do not spend discounts where risk is low. Before any ROI claim, run a
randomised holdout campaign stratified by risk and value tier and replace the ASSUMED uplift with a measured one.

## 9. Limitations
One calendar year, one retailer, {s['n_households']:,} loyalty-card households (card-less purchases are invisible);
about {tr['n_test_positives']} churn-proxy events per cutoff so model differences are within noise; the label is a
proxy; partial demographics; BG/NBD has no seasonality and 12-month CLV extrapolates beyond one observed year;
this extract contains no returns.

![Value x risk matrix](figures/targeting_matrix.png) ![Campaign outcomes](figures/campaign_outcomes.png)
"""


CSS = """
@page { size: A4; margin: 12mm 14mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 7.9pt; line-height: 1.28; color: #1a1a1a; }
h1 { font-size: 13pt; color: #1f4e79; margin: 0 0 4px 0; border-bottom: 2px solid #1f4e79; }
h2 { font-size: 9.2pt; color: #1f4e79; margin: 6px 0 2px 0; }
ul { margin: 2px 0 2px 14px; padding: 0; } li { margin-bottom: 1.5px; } p { margin: 2px 0; }
table { border-collapse: collapse; margin: 4px 0; } th, td { border: 1px solid #bbb; padding: 2px 7px; }
th { background: #e8eef5; }
img { width: 49%; } p:has(img) { text-align: center; }
"""


def main():
    md = build()
    (DOCS / "retention_memo.md").write_text(md)
    body = markdown.markdown(md, extensions=["tables", "attr_list"])
    html = f"<html><head><style>{CSS}</style></head><body>{body}</body></html>"
    from weasyprint import HTML

    doc = HTML(string=html, base_url=str(DOCS)).render()
    doc.write_pdf(DOCS / "retention_memo.pdf")
    print(f"memo written ({len(doc.pages)} pages): {DOCS / 'retention_memo.pdf'}")


if __name__ == "__main__":
    main()
