"""Label-window closure and split-design validation."""
import copy
from datetime import date

import pytest

from churn_clv.features import label_window_complete, validate_split_design

DATA_END = date(2017, 12, 31)


def test_committed_config_is_valid(cfg):
    assert validate_split_design(cfg, DATA_END) == []


def test_label_windows_close_before_next_split(cfg):
    ch = cfg["churn"]
    from datetime import timedelta
    h = timedelta(days=ch["horizon_days"])
    last_train = max(date.fromisoformat(c) for c in ch["train_cutoffs"])
    assert last_train + h <= date.fromisoformat(ch["valid_cutoff"])
    assert date.fromisoformat(ch["valid_cutoff"]) + h <= date.fromisoformat(ch["test_cutoff"])


def test_overlapping_train_window_is_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["churn"]["train_cutoffs"] = bad["churn"]["train_cutoffs"] + ["2017-06-18"]  # window closes after valid
    problems = validate_split_design(bad, DATA_END)
    assert any("train label window" in p for p in problems)


def test_test_window_past_data_end_is_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["churn"]["test_cutoff"] = "2017-11-15"
    assert any("extends past data end" in p for p in validate_split_design(bad, DATA_END))


def test_unordered_cutoffs_are_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["churn"]["valid_cutoff"], bad["churn"]["test_cutoff"] = bad["churn"]["test_cutoff"], bad["churn"]["valid_cutoff"]
    assert validate_split_design(bad, DATA_END)


def test_backtest_needs_three_closed_snapshots(cfg):
    bad = copy.deepcopy(cfg)
    bad["churn"]["backtest_eval_cutoffs"] = ["2017-04-23"]
    assert any("closed training snapshots" in p for p in validate_split_design(bad, DATA_END))


def test_every_backtest_cutoff_has_closed_training_data(cfg):
    from datetime import timedelta
    ch = cfg["churn"]
    h = timedelta(days=ch["horizon_days"])
    for ev in ch["backtest_eval_cutoffs"]:
        closed = [c for c in ch["grid_cutoffs"] if date.fromisoformat(c) + h <= date.fromisoformat(ev)]
        assert len(closed) >= 3


def test_label_window_complete(con):
    assert label_window_complete(con, "2017-09-01", 90)
    assert not label_window_complete(con, "2017-12-15", 90)


@pytest.mark.parametrize("key", ["train_cutoffs", "grid_cutoffs", "backtest_eval_cutoffs"])
def test_cutoffs_sorted(cfg, key):
    c = cfg["churn"][key]
    assert c == sorted(c)
