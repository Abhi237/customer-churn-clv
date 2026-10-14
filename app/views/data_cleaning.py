import plotly.graph_objects as go
import streamlit as st
from core import NAVY, RED, csv, gbp, setup_page, summary

s = summary()
sq = s["sql_summary"]
setup_page("Data & cleaning", "UCI Online Retail II (CC BY 4.0) · two Excel sheets concatenated · DuckDB SQL model")

c = st.columns(4)
c[0].metric("Raw invoice lines", f"{sq['dq_report']['total_rows']:,}")
c[1].metric("Clean purchase invoices", f"{sq['n_invoices_clean']:,}")
c[2].metric("Customers", f"{sq['n_customers_clean']:,}", f"{sq['uk_customer_share']:.0%} UK", delta_color="off")
c[3].metric("Returns kept separately", gbp(sq["returns_clean_gbp"]))

st.subheader("Pipeline")
st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4fb", color="#1f4e79",
  fontname="Helvetica", fontsize=11];
  zip [label="UCI zip / xlsx"]; pq [label="parquet\\n(both sheets)"];
  stg [label="01 stage"]; dq [label="02 data-quality\\nchecks"]; cl [label="03 clean lines\\n(dedup, exclusions)"];
  inv [label="04 invoices / returns\\n/ customers"]; co [label="05 cohorts"]; ro [label="06 rolling 30/60/90d"];
  rfm [label="07 RFM"]; sv [label="08 survival set"]; ml [label="Python: churn, CLV,\\ntargeting"];
  r [label="R: KM / Cox"];
  zip -> pq -> stg -> dq; stg -> cl -> inv; inv -> co; inv -> ro; inv -> rfm; inv -> sv -> r; inv -> ml; }
""")

st.subheader("Cleaning waterfall")
dq = csv("dq_report.csv").set_index("check_name")["n_rows"]
steps = [("Raw lines", dq["total_rows"], "absolute"),
         ("Missing customer ID", -dq["missing_customer_id"], "relative")]
wf = csv("dq_cleaning_waterfall.csv").set_index("step")["n_rows"]
other = wf["clean lines (sales + cancellations)"] - dq["total_rows"] + dq["missing_customer_id"]
steps += [("Duplicates, price ≤ 0, non-product,\nadjustments, outliers (net, after ID filter)", other, "relative"),
          ("Clean lines", wf["clean lines (sales + cancellations)"], "total")]
fig = go.Figure(go.Waterfall(x=[x[0] for x in steps], y=[x[1] for x in steps], measure=[x[2] for x in steps],
                             text=[f"{abs(x[1]):,}" for x in steps], textposition="outside",
                             decreasing=dict(marker_color=RED), totals=dict(marker_color=NAVY),
                             increasing=dict(marker_color=NAVY), connector=dict(line=dict(color="#c3ccd6"))))
fig.update_layout(height=420, title="From raw lines to modelling-grade lines", yaxis_title="Lines",
                  showlegend=False)
st.plotly_chart(fig, width="stretch")
c = st.columns(2)
c[0].metric("Clean sale lines", f"{wf['clean sale lines']:,}")
c[1].metric("Clean cancellation lines (returns)", f"{wf['clean cancellation lines']:,}")
st.caption("Checks overlap (a line can fail several), so the middle bar is the net remaining reduction after "
           "removing lines without a customer ID.")

st.subheader("Data-quality checks (raw lines flagged)")
labels = {
    "exact_duplicate_rows": ("Exact duplicate lines", "Deduplicated with ROW_NUMBER()"),
    "cancellation_lines_invoice_C": ("Cancellation lines ('C' invoices)", "Kept as returns, never purchases"),
    "adjustment_lines_invoice_A": ("Bad-debt adjustments ('A')", "Excluded"),
    "negative_quantity_lines": ("Negative quantity lines", "Cancellations kept as returns; others excluded"),
    "negative_qty_not_cancelled": ("Negative qty, not cancelled", "Stock adjustments - excluded"),
    "zero_or_negative_price": ("Price ≤ 0", "Excluded"),
    "missing_customer_id": ("Missing customer ID", "Excluded - cannot attribute to a lifecycle"),
    "missing_description": ("Missing description", "Informational"),
    "non_product_stock_code": ("Non-product stock codes", "Excluded (POST, M, BANK CHARGES, ...)"),
    "abs_quantity_outlier": (f"|quantity| > {s['config']['max_abs_quantity']:,}", "Excluded as data-entry outliers"),
}
rows = [{"Check": v[0], "Raw lines": int(dq[k]), "Share of raw": dq[k] / dq["total_rows"], "Decision": v[1]}
        for k, v in labels.items()]
st.dataframe(rows, hide_index=True, width="stretch",
             column_config={"Share of raw": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                                            max_value=0.25),
                            "Raw lines": st.column_config.NumberColumn(format="%d")})
with st.expander("Non-product stock codes excluded"):
    st.write(", ".join(s["config"]["non_product_codes"]))

st.subheader("Grain")
st.markdown(f"""
| Table | Grain | Rows |
|---|---|---|
| `fct_invoices` | customer × purchase invoice | {sq['n_invoices_clean']:,} |
| `dim_customers` | customer | {sq['n_customers_clean']:,} |
| period | {sq['date_min']} → {sq['date_max']} | {sq['n_countries']} countries |
""")
