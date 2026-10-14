"""Figure helpers (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
PALETTE = ["#1f4e79", "#c0504d", "#9bbb59", "#8064a2", "#f79646", "#4bacc6"]


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def roc_pr(y, probs: dict[str, np.ndarray], path: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for (name, p), c in zip(probs.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=name, color=c)
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, label=name, color=c)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC curve (test cutoff)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(np.mean(y), color="k", ls="--", lw=0.8, label="base rate")
    ax[1].set(title="Precision-recall (test cutoff)", xlabel="Recall", ylabel="Precision")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    return save(fig, path)


def reliability(y, probs: dict[str, np.ndarray], path: Path, n_bins: int = 10) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    for (name, p), c in zip(probs.items(), PALETTE):
        frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax[0].plot(mean, frac, "o-", label=name, color=c, ms=4)
        ax[1].hist(p, bins=20, range=(0, 1), histtype="step", color=c, label=name)
    ax[0].set(title="Reliability diagram (test cutoff, quantile bins)", xlabel="Predicted churn probability",
              ylabel="Observed churn rate")
    ax[1].set(title="Score distribution", xlabel="Predicted probability", ylabel="Customers")
    ax[0].legend(fontsize=8)
    return save(fig, path)


def gains(lifts: dict[str, pd.DataFrame], path: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for (name, lt), c in zip(lifts.items(), PALETTE):
        x = np.r_[0, lt["decile"] / lt["decile"].max()]
        ax[0].plot(x * 100, np.r_[0, lt["cum_gain"]] * 100, "o-", ms=3, label=name, color=c)
    ax[0].plot([0, 100], [0, 100], "k--", lw=0.8, label="random")
    ax[0].set(title="Cumulative gains", xlabel="% customers contacted (highest risk first)",
              ylabel="% of churners captured")
    ax[0].legend(fontsize=8)
    names = list(lifts)
    w = 0.8 / len(names)
    for i, (name, c) in enumerate(zip(names, PALETTE)):
        lt = lifts[name]
        ax[1].bar(lt["decile"] + (i - len(names) / 2) * w + w / 2, lt["lift"], width=w, label=name, color=c)
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set(title="Lift by risk decile", xlabel="Decile (1 = highest predicted risk)", ylabel="Lift vs base rate",
              xticks=range(1, 11))
    return save(fig, path)


def cohort_heatmap(cohorts: pd.DataFrame, path: Path, max_months: int = 12) -> Path:
    m = cohorts[cohorts["months_since_acquisition"] <= max_months].pivot(
        index="acquisition_cohort", columns="months_since_acquisition", values="retention_rate")
    m.index = pd.to_datetime(m.index).strftime("%Y-%m")
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(m.values, aspect="auto", cmap="Blues", vmin=0, vmax=0.5)
    ax.set_xticks(range(m.shape[1]), m.columns)
    ax.set_yticks(range(m.shape[0]), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if not np.isnan(v) and j > 0:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.3 else "black")
    ax.set(title="Monthly acquisition cohorts: share of customers purchasing in month k",
           xlabel="Months since first purchase", ylabel="Acquisition cohort")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, format=matplotlib.ticker.PercentFormatter(1.0))
    return save(fig, path)


def clv_holdout(ch: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    b = ch.assign(fbin=np.minimum(ch["frequency_cal"], 15)).groupby("fbin")
    agg = b[["frequency_holdout", "pred_purchases"]].mean()
    ax[0].plot(agg.index, agg["frequency_holdout"], "o-", label="actual", color=PALETTE[0])
    ax[0].plot(agg.index, agg["pred_purchases"], "s--", label="BG/NBD predicted", color=PALETTE[1])
    ax[0].set(title="Holdout purchases by calibration-period repeat count",
              xlabel="Repeat purchases in calibration period (15 = 15+)", ylabel="Mean purchases in holdout")
    ax[0].legend()
    d = ch.assign(dec=pd.qcut(ch["pred_spend"].rank(method="first"), 10, labels=range(1, 11)))
    g = d.groupby("dec", observed=True)[["actual_spend", "pred_spend"]].mean()
    x = np.arange(1, 11)
    ax[1].bar(x - 0.2, g["actual_spend"], 0.4, label="actual", color=PALETTE[0])
    ax[1].bar(x + 0.2, g["pred_spend"], 0.4, label="BG/NBD x Gamma-Gamma", color=PALETTE[1])
    ax[1].set(title="Holdout spend by predicted-spend decile", xlabel="Predicted decile (10 = highest)",
              ylabel="Mean holdout spend (GBP)", xticks=x)
    ax[1].legend()
    return save(fig, path)


def targeting_matrix(mat: pd.DataFrame, path: Path) -> Path:
    cnt = mat.pivot(index="value_segment", columns="risk_segment", values="customers")
    val = mat.pivot(index="value_segment", columns="risk_segment", values="total_clv_12m")
    cnt, val = cnt.iloc[::-1], val.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(val.values / 1e3, cmap="OrRd")
    ax.set_xticks(range(cnt.shape[1]), cnt.columns)
    ax.set_yticks(range(cnt.shape[0]), cnt.index)
    for i in range(cnt.shape[0]):
        for j in range(cnt.shape[1]):
            ax.text(j, i, f"{int(cnt.values[i, j]):,} customers\nGBP {val.values[i, j] / 1e3:,.0f}k CLV",
                    ha="center", va="center", fontsize=8.5)
    ax.grid(False)
    ax.set(title="Retention targeting matrix (scored at 2011-09-01)", xlabel="Calibrated churn-proxy risk",
           ylabel="Predicted 12-month CLV tier")
    fig.colorbar(im, ax=ax, fraction=0.04, label="Total predicted 12m CLV (GBP k)")
    return save(fig, path)


def sensitivity_heatmap(sens: pd.DataFrame, margin: float, path: Path) -> Path:
    s = sens[sens["gross_margin"] == margin].pivot(index="uplift", columns="contact_cost_gbp",
                                                   values="expected_net_value_gbp")
    n = sens[sens["gross_margin"] == margin].pivot(index="uplift", columns="contact_cost_gbp", values="n_targeted")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(s.values / 1e3, cmap="Greens", origin="lower")
    ax.set_xticks(range(s.shape[1]), [f"GBP {c:g}" for c in s.columns])
    ax.set_yticks(range(s.shape[0]), [f"{u:.0%}" for u in s.index])
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            ax.text(j, i, f"GBP {s.values[i, j] / 1e3:,.1f}k\n{int(n.values[i, j]):,} targeted", ha="center",
                    va="center", fontsize=8)
    ax.grid(False)
    ax.set(title=f"SCENARIO (assumed inputs): expected net value, margin {margin:.0%}",
           xlabel="Assumed contact cost per customer", ylabel="Assumed relative churn reduction")
    fig.colorbar(im, ax=ax, fraction=0.04, label="GBP k")
    return save(fig, path)


def backtest(bt: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for (m, g), c in zip(bt.groupby("model"), PALETTE):
        ax.plot(pd.to_datetime(g["eval_cutoff"]), g["pr_auc"], "o-", label=m, color=c)
    ax.set(title="Rolling-origin backtest: PR-AUC by evaluation cutoff", ylabel="PR-AUC")
    ax.legend(fontsize=8)
    return save(fig, path)
