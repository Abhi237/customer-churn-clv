"""Create the two walkthrough notebooks (executed afterwards with nbconvert)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = """import os, json, duckdb, pandas as pd, numpy as np, matplotlib.pyplot as plt
os.chdir('..') if os.path.basename(os.getcwd()) == 'notebooks' else None
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 20)
con = duckdb.connect('data/processed/retail.duckdb', read_only=True)"""

eda = nbf.v4.new_notebook()
eda.cells = [
    md("# 01 - Exploratory data analysis: UCI Online Retail II\nAll tables come from the DuckDB model built by "
       "`make sql` (`sql/01..08`). Raw data is not committed - run `make data` first."),
    code(SETUP),
    md("## Data-quality profile and cleaning waterfall"),
    code("con.execute('SELECT * FROM dq_report').df()"),
    code("con.execute('SELECT * FROM dq_cleaning_waterfall').df()"),
    md("## Grain and size of the clean model"),
    code("""con.execute('''SELECT 'fct_invoices' t, COUNT(*) n FROM fct_invoices UNION ALL
SELECT 'dim_customers', COUNT(*) FROM dim_customers UNION ALL SELECT 'fct_returns', COUNT(*) FROM fct_returns''').df()"""),
    md("## Monthly revenue and active customers"),
    code("""m = con.execute('''SELECT DATE_TRUNC('month', invoice_date) AS month, SUM(revenue) revenue,
       COUNT(DISTINCT customer_id) customers FROM fct_invoices GROUP BY 1 ORDER BY 1''').df()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].bar(m.month, m.revenue / 1e3, width=20); ax[0].set_title('Monthly revenue (GBP k)')
ax[1].plot(m.month, m.customers, 'o-'); ax[1].set_title('Monthly purchasing customers')
plt.tight_layout(); m.tail()"""),
    md("Strong Q4 seasonality (Sep-Nov) is visible - it matters for the churn proxy (see model notebook)."),
    code("con.execute('SELECT * FROM monthly_active_customers ORDER BY month_end').df().tail(6)"),
    md("## Cohort retention"),
    code("""from IPython.display import Image; Image('docs/figures/cohort_retention_heatmap.png', width=800)"""),
    code("""con.execute('''SELECT months_since_acquisition, AVG(retention_rate) avg_retention, COUNT(*) cohorts
FROM cohort_retention WHERE months_since_acquisition BETWEEN 1 AND 12 GROUP BY 1 ORDER BY 1''').df()"""),
    md("## Customer concentration and RFM"),
    code("""c = con.execute('SELECT lifetime_revenue FROM dim_customers ORDER BY 1 DESC').df()
share = c.lifetime_revenue.cumsum() / c.lifetime_revenue.sum()
print(f"Top 10% of customers = {share.iloc[int(len(c)*0.1)-1]:.1%} of revenue; top 1% = {share.iloc[int(len(c)*0.01)-1]:.1%}")
con.execute('SELECT rfm_segment, COUNT(*) n, ROUND(AVG(monetary)) avg_monetary, ROUND(AVG(recency_days)) avg_recency '
            'FROM customer_rfm GROUP BY 1 ORDER BY 2 DESC').df()"""),
    md("## Countries"),
    code("con.execute('SELECT home_country, COUNT(*) customers, ROUND(SUM(lifetime_revenue)) revenue FROM dim_customers "
         "GROUP BY 1 ORDER BY 2 DESC LIMIT 8').df()"),
]

mod = nbf.v4.new_notebook()
mod.cells = [
    md("# 02 - Churn-proxy & CLV model walkthrough\nReads artifacts from `make train` (`churn train`, `churn clv`, "
       "`churn target`). **Churn = no purchase in the 90 days after a cutoff - a proxy, not an observed cancellation.**"),
    code(SETUP + "\nart = 'artifacts/'"),
    md("## Time-aware design\nTrain on 4 snapshot cutoffs, choose hyper-parameters & calibration on 2011-06-01, "
       "report on 2011-09-01. Every label window ends before the next split's cutoff."),
    code("""tr = pd.read_parquet(art + 'snapshot_train.parquet'); va = pd.read_parquet(art + 'snapshot_valid.parquet')
te = pd.read_parquet(art + 'snapshot_test.parquet')
pd.DataFrame({'rows': [len(tr), len(va), len(te)], 'churn_rate': [tr.churned.mean(), va.churned.mean(), te.churned.mean()]},
             index=['train (4 cutoffs)', 'valid 2011-06-01', 'test 2011-09-01'])"""),
    code("tr.groupby('cutoff').churned.agg(['size', 'mean'])"),
    md("## Leakage check in-notebook\nFeatures at a cutoff are computed only from rows dated before it."),
    code("""from churn_clv.features import build_snapshot, FEATURE_COLUMNS
s = build_snapshot(con, '2011-09-01')
last = con.execute("SELECT customer_id, MAX(invoice_date) d FROM fct_invoices WHERE invoice_date < DATE '2011-09-01' GROUP BY 1").df()
chk = s.merge(last, on='customer_id')
assert ((pd.Timestamp('2011-09-01') - pd.to_datetime(chk.d)).dt.days == chk.recency_days).all(); print('recency consistent with pre-cutoff data only')"""),
    md("## Metrics vs baselines (test cutoff)"),
    code("pd.read_csv(art + 'model_metrics.csv').query('split==\"test\"').round(3)"),
    code("from IPython.display import Image; Image('docs/figures/roc_pr_test.png', width=850)"),
    code("Image('docs/figures/gains_lift_test.png', width=850)"),
    md("## Calibration"),
    code("Image('docs/figures/reliability_test.png', width=850)"),
    code("json.load(open(art + 'train_summary.json'))['calibration_in_the_large_test']"),
    md("All models over-predict churn at the Sep-2011 cutoff: the Sep-Nov label window is peak season. Rankings "
       "remain useful; absolute probabilities need seasonal re-calibration."),
    md("## Rolling-origin backtest"),
    code("pd.read_csv(art + 'backtest_metrics.csv')[['eval_cutoff','model','roc_auc','pr_auc','brier','base_rate']].round(3)"),
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
]
nbf.write(eda, "notebooks/01_eda.ipynb")
nbf.write(mod, "notebooks/02_model_walkthrough.ipynb")
