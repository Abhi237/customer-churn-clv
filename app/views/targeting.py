import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from core import RISK_ORDER, VALUE_ORDER, assumed_banner, customers, gbp, setup_page, summary

s = summary()
t = s["config"]["targeting"]
setup_page("Retention targeting simulator", "Combine calibrated churn risk with predicted 12-month CLV and test "
           "how the economics change under different assumptions")
assumed_banner("There is no retention experiment in this dataset. Uplift, contact cost and margin below are "
               "<b>assumptions</b>; the outputs are expected values under those assumptions, never observed impact. "
               "Churn probabilities are also over-stated at this scoring date (see Churn model → Calibration).")
st.write("")
st.latex(r"\text{Expected net value} = P(\text{churn}) \times \text{uplift} \times \text{CLV}_{12m} "
         r"\times \text{margin} - \text{contact cost}")

cust = customers()
with st.container(border=True):
    c = st.columns(4)
    uplift = c[0].slider("ASSUMED relative churn reduction", 0.0, 0.5, float(t["uplift"]), 0.01, format="%.2f")
    margin = c[1].slider("ASSUMED gross margin", 0.05, 0.8, float(t["gross_margin"]), 0.05, format="%.2f")
    cost = c[2].slider("ASSUMED contact cost (£)", 0.0, 50.0, float(t["contact_cost_gbp"]), 0.5)
    budget = c[3].number_input("Optional budget cap (£, 0 = none)", 0, 200000, 0, 1000)

env = cust.churn_prob * uplift * cust.clv_12m * margin - cost
d = cust.assign(env=env).sort_values("env", ascending=False)
d["target"] = d.env > 0
if budget and cost > 0:
    d.loc[d.index[int(budget // cost):], "target"] = False
tg = d[d.target]
k = st.columns(5)
k[0].metric("Customers targeted", f"{len(tg):,}", f"{len(tg) / len(d):.0%} of scored", delta_color="off")
k[1].metric("Campaign cost", gbp(cost * len(tg)))
k[2].metric("Expected net value", gbp(float(tg.env.sum())), "scenario", delta_color="off")
k[3].metric("Expected customers saved", f"{float((tg.churn_prob * uplift).sum()):,.0f}", "scenario",
            delta_color="off")
k[4].metric("Observed churn of targeted", f"{tg.churned.mean():.0%}" if len(tg) else "-",
            "realised (back-test)", delta_color="off")

c1, c2 = st.columns([3, 2])
with c1:
    g = d.groupby(["value_segment", "risk_segment"], observed=True).agg(
        n=("customer_id", "size"), targeted=("target", "sum"), env=("env", lambda x: x[x > 0].sum()))
    g = g.reset_index()
    share = g.pivot(index="value_segment", columns="risk_segment", values="targeted").reindex(
        index=VALUE_ORDER, columns=RISK_ORDER)
    n = g.pivot(index="value_segment", columns="risk_segment", values="n").reindex(index=VALUE_ORDER,
                                                                                    columns=RISK_ORDER)
    text = [[f"{int(share.iloc[i, j]):,} / {int(n.iloc[i, j]):,} targeted" for j in range(3)] for i in range(3)]
    fig = px.imshow((share / n).values, x=RISK_ORDER, y=VALUE_ORDER, color_continuous_scale=[[0, "#fff7ec"], [1, "#f4a259"]], zmin=0,
                    zmax=1, labels=dict(color="Share targeted"), aspect="auto")
    fig.update_traces(text=text, texttemplate="%{text}", textfont=dict(size=13, color="#1a2433"))
    fig.update_layout(title="Who the rule targets (value × risk)", height=400, xaxis_title="Risk tier",
                      yaxis_title="CLV tier")
    st.plotly_chart(fig, width="stretch")
with c2:
    curve = d.reset_index(drop=True)
    curve["cum"] = curve.env.cumsum()
    curve["k"] = np.arange(1, len(curve) + 1)
    fig = px.line(curve, x="k", y="cum", labels={"k": "Customers contacted (best first)",
                                                 "cum": "Cumulative expected net value (£)"})
    fig.add_vline(x=len(tg), line_dash="dash", line_color="#c0504d", annotation_text="rule cut-off")
    fig.update_layout(title="Diminishing returns (scenario)", height=400)
    st.plotly_chart(fig, width="stretch")

st.subheader("Sensitivity: expected net value across assumptions")
ups = np.round(np.linspace(0.05, 0.30, 6), 2)
costs = [1, 2, 5, 10, 20, 30]
grid = pd.DataFrame([[float(np.clip(cust.churn_prob * u * cust.clv_12m * margin - c, 0, None).sum())
                      for c in costs] for u in ups], index=[f"{u:.0%}" for u in ups],
                    columns=[f"£{c}" for c in costs])
fig = px.imshow(grid.values / 1e3, x=grid.columns, y=grid.index, color_continuous_scale="Greens", aspect="auto",
                labels=dict(x="ASSUMED contact cost", y="ASSUMED uplift", color="£k"), origin="lower")
fig.update_traces(texttemplate="£%{z:,.1f}k")
fig.update_layout(title=f"Expected net value if the rule is applied (margin {margin:.0%}, ASSUMED)", height=420)
st.plotly_chart(fig, width="stretch")
st.caption("The committed pipeline runs the same calculation over a 48-scenario grid "
           f"(artifacts/targeting_sensitivity.csv): range {gbp(s['targeting_summary']['sensitivity_range_net_value_gbp'][0])}"
           f" to {gbp(s['targeting_summary']['sensitivity_range_net_value_gbp'][1])}.")

st.subheader("Target list")
cols = ["customer_id", "value_segment", "risk_segment", "churn_prob", "clv_12m", "env", "recency_days",
        "frequency", "monetary_total"]
st.dataframe(tg[cols].head(500), hide_index=True, width="stretch", height=360, column_config={
    "churn_prob": st.column_config.ProgressColumn("Churn prob.", format="%.2f", min_value=0, max_value=1),
    "clv_12m": st.column_config.NumberColumn("CLV 12m £", format="%.0f"),
    "env": st.column_config.NumberColumn("Exp. net value £ (ASSUMED)", format="%.1f"),
    "monetary_total": st.column_config.NumberColumn("Spend to date £", format="%.0f")})
st.download_button("Download target list (CSV)", tg[cols].to_csv(index=False), "target_list_scenario.csv",
                   type="primary")
