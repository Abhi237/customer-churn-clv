import plotly.graph_objects as go
import streamlit as st
from core import DATA_SOURCE, NAVY, RED, csv, setup_page, summary, usd

s = summary()
sq = s["sql_summary"]
dqr = sq["dq_report"]
setup_page("Data and cleaning", "Complete Journey 2.0 (84.51, CC0) via the `completejourney` R package v1.1.1 | "
           "2,469 loyalty-card households, calendar year 2017 | DuckDB SQL model")

c = st.columns(4)
c[0].metric("Raw transaction lines", f"{dqr['total_lines']:,}")
c[1].metric("Clean purchase baskets", f"{sq['n_baskets_clean']:,}", f"{sq['baskets_dropped_not_purchases']} dropped", delta_color="off")
c[2].metric("Households", f"{sq['n_households']:,}", f"{sq['n_households_with_demographics']} with demographics", delta_color="off")
c[3].metric("Clean spend", usd(sq["spend_clean_usd"]), f"{sq['n_stores']} stores", delta_color="off")

st.subheader("Pipeline")
st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4fb", color="#1f4e79",
  fontname="Helvetica", fontsize=11];
  src [label="completejourney CSVs\\n(R prep script)"]; stg [label="01 stage\\n(typed views)"];
  dq [label="02 data-quality\\nchecks"]; cl [label="03 clean lines\\n(classify, flag)"];
  bk [label="04 baskets, days,\\nhouseholds"]; co [label="05 cohorts"]; ro [label="06 rolling 30/60/90d"];
  rfm [label="07 RFM"]; cp [label="08 campaigns,\\ncoupon redemptions"]; sv [label="09 survival set"];
  ml [label="Python: churn, CLV,\\ntargeting, campaigns"]; r [label="R: KM / Cox"];
  src -> stg -> dq; stg -> cl -> bk; bk -> co; bk -> ro; bk -> rfm; bk -> cp; bk -> sv -> r; bk -> ml; cp -> ml; }
""")

st.subheader("Cleaning waterfall (transaction lines)")
wf = csv("dq_cleaning_waterfall.csv").set_index("step")["n_rows"]
steps = [("Raw lines", wf["raw lines"], "absolute"),
         ("Empty lines\n(no value, no discount)", -wf["dropped: empty lines (no value, no discount)"], "relative"),
         ("Zero quantity,\npositive sales", -wf["dropped: zero quantity with positive sales"], "relative"),
         ("Zero price,\nno discount", -wf["dropped: zero price, no discount"], "relative"),
         ("Kept lines", wf["kept lines (merchandise, free items, coupon-only, returns)"], "total")]
fig = go.Figure(go.Waterfall(x=[x[0] for x in steps], y=[x[1] for x in steps], measure=[x[2] for x in steps],
                             text=[f"{abs(x[1]):,}" for x in steps], textposition="outside",
                             decreasing=dict(marker_color=RED), totals=dict(marker_color=NAVY),
                             increasing=dict(marker_color=NAVY), connector=dict(line=dict(color="#c3ccd6"))))
fig.update_layout(height=420, title="From raw lines to modelling-grade lines", yaxis_title="Lines", showlegend=False)
st.plotly_chart(fig, width="stretch")
c = st.columns(4)
c[0].metric("Merchandise lines", f"{wf['kept: merchandise lines']:,}")
c[1].metric("Free-item lines ($0, discounted)", f"{wf['kept: free-item lines ($0 sales, discounted)']:,}")
c[2].metric("Coupon-only lines", f"{wf['kept: coupon-only lines']:,}")
c[3].metric("Return lines", f"{wf['kept: return lines (negative quantity or value)']:,}",
            help="This extract contains no negative quantity or sales_value lines")
st.caption("A basket counts as a purchase only if its merchandise spend is positive; "
           f"{sq['baskets_dropped_not_purchases']} raw baskets (coupon-only or zero-spend) are not counted as shopping trips.")

st.subheader("Data-quality checks (raw lines flagged)")
total = dqr["total_lines"]
labels = [
    ("exact_duplicate_lines", "Exact duplicate lines", "None found"),
    ("negative_quantity_lines", "Negative quantity (returns)", "None in this extract; rule would keep them as returns"),
    ("negative_sales_value_lines", "Negative sales_value", "None in this extract"),
    ("zero_quantity_lines", "Zero quantity", "29 dropped (positive sales); the rest are empty or coupon-only lines, below"),
    ("zero_quantity_positive_sales", "Zero quantity with positive sales", "Dropped (inconsistent)"),
    ("empty_lines_no_value_no_discount", "Empty lines (no value, no discount)", "Dropped"),
    ("zero_price_no_discount_lines", "Zero price, no discount", "Dropped"),
    ("coupon_only_lines", "Coupon-only lines", "Kept; excluded from purchase counts"),
    ("free_item_lines", "Free items ($0 sales, discounted)", "Kept; units counted, no spend"),
    ("quantity_above_max_lines", f"Quantity above {s['config']['data']['max_quantity']:,}",
     "Units set to NULL (dollars kept); mostly fuel measured in gallons"),
    ("quantity_above_max_fuel_lines", "  of which fuel", "As above"),
    ("sales_value_above_high_value_lines", f"Line above ${s['config']['data']['high_value_line_usd']:,}", "Kept, flagged"),
    ("lines_without_product_match", "Lines without a product match", "Kept; department group = unknown"),
    ("baskets_nonpositive_total", "Baskets with non-positive total", "Not counted as purchases"),
]
rows = [{"Check": a, "Count": int(dqr[k]), "Share of raw lines": dqr[k] / total if "baskets" not in k else None,
         "Decision": d} for k, a, d in labels]
st.dataframe(rows, hide_index=True, width="stretch",
             column_config={"Share of raw lines": st.column_config.NumberColumn(format="percent"),
                            "Count": st.column_config.NumberColumn(format="%d")})
st.caption("Counts are logged by `sql/02_data_quality.sql` on every run. The 29 zero-quantity positive-sales lines and 5,859 empty "
           "lines are counted after normalising float residue (about 1e-15) in sales_value to exact DECIMAL cents.")

st.subheader("Grain")
st.markdown(f"""
| Table | Grain | Rows |
|---|---|---|
| `fct_baskets` | household x basket (shopping trip) | {sq['n_baskets_clean']:,} |
| `fct_days` | household x shopping day | {sq['n_shopping_days']:,} |
| `dim_households` | household | {sq['n_households']:,} |
| period | {sq['date_min']} to {sq['date_max']} | {sq['n_stores']} stores |
""")
st.caption(f"Source: [completejourney]({DATA_SOURCE}) v1.1.1, Complete Journey 2.0 from 84.51, released CC0. "
           "Timestamps are naive local (America/New_York) times as shipped.")
