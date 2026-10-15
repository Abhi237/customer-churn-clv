"""Command-line interface: `churn <command>`."""
from __future__ import annotations

import json
import subprocess

import click

from . import pipeline
from .config import ROOT


def _p(obj):
    click.echo(json.dumps(obj, indent=2, default=str)[:4000])


@click.group()
def main():
    """Household churn-proxy & CLV pipeline (Complete Journey grocery loyalty panel)."""


@main.command()
def data():
    """Download the Complete Journey CSVs (R + internet needed once)."""
    subprocess.run(["python", str(ROOT / "scripts/download_data.py")], check=True)


@main.command()
def sql():
    """Run numbered DuckDB SQL scripts (cleaning, DQ, baskets, cohorts, rolling activity, RFM, campaigns, survival)."""
    _p(pipeline.run_sql())


@main.command()
def train():
    """Build point-in-time snapshots, train/calibrate/evaluate churn models, SHAP."""
    _p(pipeline.train())


@main.command()
def clv():
    """Fit BG/NBD + Gamma-Gamma with calibration/holdout validation."""
    _p(pipeline.run_clv())


@main.command()
def target():
    """Combine churn risk and CLV into a value x risk matrix and an ROI rule (assumed parameters)."""
    _p(pipeline.run_targeting())


@main.command()
def campaigns():
    """Observational campaign / coupon analysis at the test cutoff (descriptive, not causal)."""
    _p(pipeline.run_campaigns())


@main.command()
def demographics():
    """Demographic breakout for the households that have demographics (partial coverage)."""
    _p(pipeline.run_demographics())


@main.command()
@click.option("--cutoff", required=True, help="Scoring date YYYY-MM-DD; only data before it is used.")
@click.option("--out", default=None, help="Output CSV path.")
def score(cutoff, out):
    """Score households at a cutoff with the saved calibrated model."""
    click.echo(f"wrote {pipeline.score(cutoff, out)}")


@main.command(name="all")
@click.pass_context
def all_(ctx):
    """Run sql -> train -> clv -> target -> campaigns -> demographics."""
    for cmd in (sql, train, clv, target, campaigns, demographics):
        ctx.invoke(cmd)


if __name__ == "__main__":
    main()
