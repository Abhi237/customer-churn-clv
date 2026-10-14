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
    """Customer churn-proxy & CLV pipeline (UCI Online Retail II)."""


@main.command()
def data():
    """Download the raw dataset and convert to parquet."""
    subprocess.run(["python", str(ROOT / "scripts/download_data.py")], check=True)


@main.command()
def sql():
    """Run numbered DuckDB SQL scripts (cleaning, DQ, cohorts, rolling activity, RFM, survival)."""
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
@click.option("--cutoff", required=True, help="Scoring date YYYY-MM-DD; only data before it is used.")
@click.option("--out", default=None, help="Output CSV path.")
def score(cutoff, out):
    """Score customers at a cutoff with the saved calibrated model."""
    click.echo(f"wrote {pipeline.score(cutoff, out)}")


@main.command(name="all")
@click.pass_context
def all_(ctx):
    """Run sql -> train -> clv -> target."""
    for cmd in (sql, train, clv, target):
        ctx.invoke(cmd)


if __name__ == "__main__":
    main()
