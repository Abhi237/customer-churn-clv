"""Create the two walkthrough notebooks (executed afterwards with nbconvert)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = """import os, json, duckdb, pandas as pd, numpy as np, matplotlib.pyplot as plt
os.chdir('..') if os.path.basename(os.getcwd()) == 'notebooks' else None
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 20)
con = duckdb.connect('data/processed/grocery.duckdb', read_only=True)"""

eda = nbf.v4.new_notebook()
eda.cells = [
    md("# 01 - Exploratory data analysis: Complete Journey grocery loyalty panel\nAll tables come from the DuckDB model "
       "built by `make sql` (`sql/01..09`). Raw data is not committed - run `make data` first. Source: 84.51 "
       "Complete Journey 2.0 via the `completejourney` R package, CC0."),
    code(SETUP),
    md("## Data-quality profile and cleaning waterfall"),
    code("con.execute('SELECT * FROM dq_report').df()"),
    code("con.execute('SELECT * FROM dq_cleaning_waterfall').df()"),
    md("## Grain and size of the clean model"),
    code("""con.execute('''SELECT 'fct_baskets' t, COUNT(*) n FROM fct_baskets UNION ALL
SELECT 'fct_days', COUNT(*) FROM fct_days UNION ALL SELECT 'dim_households', COUNT(*) FROM dim_households UNION ALL
SELECT 'campaign_exposure', COUNT(*) FROM campaign_exposure UNION ALL
SELECT 'coupon_redemptions_clean', COUNT(*) FROM coupon_redemptions_clean''').df()"""),
    md("## Monthly spend and active households"),
    code("""m = con.execute('''SELECT DATE_TRUNC('month', basket_date) AS month, SUM(spend) spend,
       COUNT(DISTINCT household_id) households FROM fct_baskets GROUP BY 1 ORDER BY 1''').df()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].bar(m.month, m.spend / 1e3, width=20); ax[0].set_title('Monthly spend (USD thousands)')
ax[1].plot(m.month, m.households, 'o-'); ax[1].set_title('Monthly shopping households')
plt.tight_layout(); m.tail()"""),
    code("con.execute('SELECT * FROM monthly_active_households ORDER BY month_end').df().tail(6)"),
    md("## Entry cohorts\nThe panel starts on 2017-01-01, so cohorts are entry cohorts (first observed purchase), not "
       "acquisition cohorts: the January cohort is mostly established households."),
    code("""from IPython.display import Image; Image('docs/figures/cohort_retention_heatmap.png', width=800)"""),
    code("""con.execute('''SELECT months_since_entry, AVG(retention_rate) avg_retention, COUNT(*) cohorts
FROM cohort_retention WHERE months_since_entry BETWEEN 1 AND 11 GROUP BY 1 ORDER BY 1''').df()"""),
    md("## Household concentration and RFM"),
    code("""c = con.execute('SELECT lifetime_spend FROM dim_households ORDER BY 1 DESC').df()
share = c.lifetime_spend.cumsum() / c.lifetime_spend.sum()
print(f"Top 10% of households = {c.lifetime_spend.head(int(len(c)*0.1)).sum() / c.lifetime_spend.sum():.1%} of spend")
con.execute('SELECT rfm_segment, COUNT(*) n, ROUND(AVG(monetary)) avg_monetary, ROUND(AVG(recency_days)) avg_recency '
            'FROM household_rfm GROUP BY 1 ORDER BY 2 DESC').df()"""),
    md("## Department mix, discounts and coupons"),
    code("""con.execute('''SELECT dept_group, ROUND(SUM(sales_value)) spend FROM fct_lines
WHERE line_type IN ('merchandise','free_item') GROUP BY 1 ORDER BY 2 DESC''').df()"""),
    code("""con.execute('''SELECT ROUND(SUM(retail_disc) / SUM(shelf_value), 4) retail_disc_share,
ROUND((SUM(coupon_disc)+SUM(coupon_match_disc)) / SUM(shelf_value), 4) coupon_share,
ROUND(AVG(CASE WHEN has_coupon THEN 1.0 ELSE 0 END), 4) baskets_with_coupon FROM fct_baskets''').df()"""),
    md("## Campaigns"),
    code("con.execute('SELECT * FROM dim_campaigns ORDER BY start_date').df()"),
]

mod = nbf.v4.new_notebook()
mod.cells = [
    md("# 02 - Churn-proxy and CLV model walkthrough\nReads artifacts from `make pipeline` (`churn train`, `churn clv`, "
       "`churn target`). **Churn = no purchase in the 90 days after a cutoff - a proxy, not an observed cancellation.**"),
    code(SETUP + "\nart = 'artifacts/'"),
    md("## Time-aware design\nTrain on 4 early snapshot cutoffs, choose hyper-parameters and the calibrator on "
       "2017-07-02, report on 2017-10-01. Every label window closes before the next split's cutoff."),
    code("""tr = pd.read_parquet(art + 'snapshot_train.parquet'); va = pd.read_parquet(art + 'snapshot_valid.parquet')
te = pd.read_parquet(art + 'snapshot_test.parquet')
pd.DataFrame({'rows': [len(tr), len(va), len(te)], 'churn_rate': [tr.churned.mean(), va.churned.mean(), te.churned.mean()],
              'positives': [tr.churned.sum(), va.churned.sum(), te.churned.sum()]},
             index=['train (4 cutoffs)', 'valid 2017-07-02', 'test 2017-10-01'])"""),
    code("tr.groupby('cutoff').churned.agg(['size', 'mean'])"),
    md("## Leakage check in-notebook\nFeatures at a cutoff are computed only from rows dated before it."),
    code("""from churn_clv.features import build_snapshot, FEATURE_COLUMNS
s = build_snapshot(con, '2017-10-01')
last = con.execute("SELECT household_id, MAX(visit_date) d FROM fct_days WHERE visit_date < DATE '2017-10-01' GROUP BY 1").df()
chk = s.merge(last, on='household_id')
assert ((pd.Timestamp('2017-10-01') - pd.to_datetime(chk.d)).dt.days == chk.recency_days).all()
print('recency consistent with pre-cutoff data only;', len(FEATURE_COLUMNS), 'rate/share/fixed-window features')"""),
    md("## Metrics vs baselines (test cutoff)"),
    code("pd.read_csv(art + 'model_metrics.csv').query('split==\"test\"').round(4)"),
    code("pd.read_csv(art + 'test_metric_ci.csv').round(3)"),
    code("json.load(open(art + 'train_summary.json'))['paired_bootstrap_test']"),
    code("from IPython.display import Image; Image('docs/figures/roc_pr_test.png', width=850)"),
    code("Image('docs/figures/gains_lift_test.png', width=850)"),
    md("## Calibration\nPlatt scaling was chosen on validation; isotonic is ineligible with so few positives. "
       "Calibration here changes the average level but does not improve Brier on test."),
    code("Image('docs/figures/reliability_test.png', width=850)"),
    code("json.load(open(art + 'train_summary.json'))['calibration']"),
    code("json.load(open(art + 'train_summary.json'))['calibration_in_the_large_test']"),
    md("## Rolling-origin backtest"),
    code("pd.read_csv(art + 'backtest_metrics.csv')[['eval_cutoff','n_train_snapshots','model','n_eval_positives','roc_auc','pr_auc','brier']].round(3)"),
    md("## SHAP explanations"),
    code("pd.read_csv(art + 'shap_importance.csv').head(10)"),
    code("Image('docs/figures/shap_beeswarm.png', width=700)"),
    code("Image('docs/figures/shap_local_waterfall.png', width=700)"),
    md("## CLV: BG/NBD + Gamma-Gamma holdout validation"),
    code("{k: v for k, v in json.load(open(art + 'clv_summary.json')).items() if not isinstance(v, dict)}"),
    code("Image('docs/figures/clv_holdout_validation.png', width=850)"),
    md("## Targeting matrix and ROI rule (ASSUMED parameters)"),
    code("pd.read_csv(art + 'targeting_matrix.csv').round(2)"),
    code("Image('docs/figures/targeting_sensitivity.png', width=700)"),
    md("## Campaign analysis (observational, no causal claim)"),
    code("pd.read_csv(art + 'campaign_outcomes_targeted.csv').round(3)"),
    code("Image('docs/figures/campaign_outcomes.png', width=800)"),
    md("## Demographics (partial coverage: 801 of 2,469 households)"),
    code("pd.read_csv(art + 'demographics_coverage.csv').round(3).T"),
    code("Image('docs/figures/demographics_profile.png', width=800)"),
]
nbf.write(eda, "notebooks/01_eda.ipynb")
nbf.write(mod, "notebooks/02_model_walkthrough.ipynb")
