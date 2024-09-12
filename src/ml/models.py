from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


@dataclass
class MLModelWrapper:
    """
    Simple wrapper for an sklearn-like model plus metadata.

    This class is optional but convenient when you want to keep
    model parameters and reuse the same model instance.
    """

    model: BaseEstimator
    params: Dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the underlying sklearn model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary labels (0/1).
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        return self.model.predict(X)


def _time_series_split(
    dataset: pd.DataFrame,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train and test sets in chronological order.

    Parameters
    ----------
    dataset : pd.DataFrame
        Complete dataset with features and label column ``'y'``.
        Index is expected to be time-like (e.g. DatetimeIndex).
    test_ratio : float
        Fraction of observations to keep as the test set.

    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")

    dataset_sorted = dataset.sort_index()
    n = len(dataset_sorted)
    if n < 2:
        raise ValueError("Dataset must contain at least 2 rows for train/test split.")

    split_idx = int(n * (1.0 - test_ratio))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("test_ratio results in empty train or test set.")

    train_df = dataset_sorted.iloc[:split_idx]
    test_df = dataset_sorted.iloc[split_idx:]
    return train_df, test_df


def _build_model(model_type: str) -> BaseEstimator:
    """
    Build an sklearn classifier based on the requested model type.

    Parameters
    ----------
    model_type : str
        One of: ``'logistic'``, ``'logistic_regression'``,
        ``'rf'``, ``'random_forest'``.

    Returns
    -------
    sklearn.base.BaseEstimator
        Instantiated classifier.
    """
    key = model_type.lower()
    if key in {"logistic", "logistic_regression", "logit"}:
        return LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            n_jobs=None,
        )
    if key in {"rf", "random_forest", "randomforest"}:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model_type: {model_type!r}")


def train_and_evaluate(
    dataset: pd.DataFrame,
    model_type: str,
    test_ratio: float = 0.2,
) -> Dict[str, Any]:
    """
    Train a binary classifier on a direction-prediction dataset and evaluate it.

    The dataset is expected to be the output of ``build_ml_dataset``,
    i.e. it contains engineered feature columns and a binary label column
    ``'y'`` where ``1`` indicates positive future return and ``0`` otherwise.

    Parameters
    ----------
    dataset : pd.DataFrame
        Full dataset with features and label column ``'y'``.
    model_type : str
        Model identifier: ``'logistic'`` / ``'logistic_regression'`` /
        ``'rf'`` / ``'random_forest'``.
    test_ratio : float, default 0.2
        Fraction of the dataset to reserve as test set (chronological split).

    Returns
    -------
    dict
        Dictionary containing:
        - ``'model'``: fitted sklearn model
        - ``'metrics'``: dict with accuracy, precision, recall, f1, auc (if available)
        - ``'signal'``: pd.Series with values +1 (predict up) or -1 (predict down),
          indexed by test-set index
        - ``'y_test'``: pd.Series of true labels on test set
        - ``'y_pred'``: pd.Series of predicted labels on test set
    """
    if "y" not in dataset.columns:
        raise KeyError("Dataset must contain a 'y' column as the target label.")

    train_df, test_df = _time_series_split(dataset, test_ratio=test_ratio)

    feature_cols = [c for c in dataset.columns if c != "y"]

    X_train = train_df[feature_cols]
    y_train = train_df["y"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["y"].astype(int)

    model = _build_model(model_type)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # AUC (if model supports predict_proba)
    auc: float | None = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, proba))
        except Exception:
            auc = None
    metrics["auc"] = auc

    # Backtest-ready signal: +1 if predict up (label=1), otherwise -1
    signal_values = np.where(y_pred == 1, 1, -1)
    signal = pd.Series(signal_values, index=X_test.index, name="signal").astype(int)

    result: Dict[str, Any] = {
        "model": model,
        "metrics": metrics,
        "signal": signal,
        "y_test": y_test,
        "y_pred": pd.Series(y_pred, index=X_test.index, name="y_pred"),
    }
    return result

def train_meta_label_model(
    meta_dataset: pd.DataFrame,
    model_type: str = "rf",
    test_ratio: float = 0.2,
    decision_threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Train a meta-label model and return filtered execution mask on test set.

    Expected columns in meta_dataset:
    - feature columns
    - side
    - y_meta
    """
    if "y_meta" not in meta_dataset.columns:
        raise KeyError("meta_dataset must contain 'y_meta'.")
    if "side" not in meta_dataset.columns:
        raise KeyError("meta_dataset must contain 'side'.")

    ds = meta_dataset.sort_index()
    train_df, test_df = _time_series_split(ds, test_ratio=test_ratio)

    feature_cols = [c for c in ds.columns if c not in {"y_meta", "side"}]

    X_train = train_df[feature_cols]
    y_train = train_df["y_meta"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["y_meta"].astype(int)

    model = _build_model(model_type)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback if model has no proba output.
        raw_pred = model.predict(X_test)
        proba = np.asarray(raw_pred, dtype=float)

    pred = (proba >= float(decision_threshold)).astype(int)

    exec_signal = test_df["side"].where(pred == 1, 0.0)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_test, proba))
    except Exception:
        metrics["auc"] = None

    return {
        "model": model,
        "metrics": metrics,
        "execution_signal": exec_signal.astype(float),
        "proba": pd.Series(proba, index=X_test.index, name="proba"),
        "y_test": y_test,
        "y_pred": pd.Series(pred, index=X_test.index, name="y_pred"),
    }


def walk_forward_splits(
    index: pd.Index,
    train_size: int,
    test_size: int,
    step_size: int,
) -> list[tuple[pd.Index, pd.Index]]:
    """Generate chronological walk-forward train/test index splits."""
    n = len(index)
    splits: list[tuple[pd.Index, pd.Index]] = []
    start = 0
    while True:
        train_end = start + int(train_size)
        test_end = train_end + int(test_size)
        if test_end > n:
            break
        train_idx = index[start:train_end]
        test_idx = index[train_end:test_end]
        splits.append((train_idx, test_idx))
        start += int(step_size)
        if step_size <= 0:
            break
    return splits
