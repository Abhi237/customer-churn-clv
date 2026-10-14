"""Baselines, logistic regression, gradient boosting and post-hoc calibration."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import FEATURE_COLUMNS

LOG_COLS = ["recency_days", "tenure_days", "frequency", "monetary_total", "avg_order_value",
            "revenue_90d", "revenue_180d", "n_distinct_products", "mean_interpurchase_days",
            "std_interpurchase_days", "recency_to_gap_ratio", "avg_products_per_invoice"]


class RecencyRule:
    """Baseline 1: churn score = recency rank, mapped to probability via train churn rate by recency bin."""

    name = "baseline_recency"

    def fit(self, X: pd.DataFrame, y):
        self.edges_ = np.unique(np.quantile(X["recency_days"], np.linspace(0, 1, 21)))
        b = np.clip(np.searchsorted(self.edges_, X["recency_days"], side="right") - 1, 0, len(self.edges_) - 2)
        s = pd.Series(np.asarray(y, float)).groupby(b).mean()
        self.rates_ = s.reindex(range(len(self.edges_) - 1)).ffill().bfill().to_numpy()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        b = np.clip(np.searchsorted(self.edges_, X["recency_days"], side="right") - 1, 0, len(self.edges_) - 2)
        return self.rates_[b]


class RFMHeuristic:
    """Baseline 2: classic RFM score (quintiles fitted on train), lower score => higher churn risk.
    Mapped to a probability by the train churn rate of each RFM score sum."""

    name = "baseline_rfm"

    def _score(self, X):
        r = 5 - np.clip(np.searchsorted(self.r_edges_, X["recency_days"], side="right") - 1, 0, 4)
        f = 1 + np.clip(np.searchsorted(self.f_edges_, X["frequency"], side="right") - 1, 0, 4)
        m = 1 + np.clip(np.searchsorted(self.m_edges_, X["monetary_total"], side="right") - 1, 0, 4)
        return r + f + m

    def fit(self, X, y):
        q = np.linspace(0, 1, 6)[:-1]
        self.r_edges_ = np.quantile(X["recency_days"], q)
        self.f_edges_ = np.quantile(X["frequency"], q)
        self.m_edges_ = np.quantile(X["monetary_total"], q)
        s = self._score(X)
        rates = pd.Series(np.asarray(y, float)).groupby(s).mean()
        self.rates_ = rates.reindex(range(3, 16)).interpolate().ffill().bfill()
        return self

    def predict_proba(self, X):
        return self.rates_.loc[self._score(X)].to_numpy()


def _log1p_cols(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in LOG_COLS:
        X[c] = np.log1p(np.clip(X[c], 0, None))
    return X


def make_logreg(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("log", FunctionTransformer(_log1p_cols)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)),
    ])


LGBM_GRID = [  # small grid; selected on the validation snapshot only (never on test)
    {"num_leaves": 15, "min_child_samples": 40, "n_estimators": 400, "learning_rate": 0.03},
    {"num_leaves": 7, "min_child_samples": 100, "n_estimators": 300, "learning_rate": 0.03},
    {"num_leaves": 4, "min_child_samples": 200, "n_estimators": 400, "learning_rate": 0.03},
    {"num_leaves": 7, "min_child_samples": 200, "n_estimators": 600, "learning_rate": 0.01},
]


def make_lgbm(random_state: int = 42, params: dict | None = None) -> lgb.LGBMClassifier:
    p = {**LGBM_GRID[2], **(params or {})}
    return lgb.LGBMClassifier(
        **p, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=random_state, verbose=-1,
    )


@dataclass
class Calibrator:
    method: str  # "isotonic" | "platt"

    def fit(self, p, y):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        else:
            self.model_ = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            return self.model_.predict(p)
        return self.model_.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def proba(model, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, (RecencyRule, RFMHeuristic)):
        return model.predict_proba(X)
    return model.predict_proba(X[FEATURE_COLUMNS])[:, 1]
