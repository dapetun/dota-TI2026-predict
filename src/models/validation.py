"""Shared temporal validation: walk-forward + Leave-One-TI-Out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_collection.tournaments import TI_KEYS
from src.features.sample_weights import compute_sample_weights

FitFn = Callable[[pd.DataFrame, np.ndarray, np.ndarray | None], Any]
PredictFn = Callable[[Any, pd.DataFrame], np.ndarray]


@dataclass
class FoldResult:
    """Metrics for one validation fold."""

    fold: str
    n_train: int
    n_test: int
    log_loss: float
    brier: float
    auc: float
    accuracy: float
    precision: float
    recall: float


@dataclass
class TrainResult:
    """Full training run artefacts."""

    model: Any
    feature_cols: list[str]
    walk_forward: list[FoldResult] = field(default_factory=list)
    leave_one_ti: list[FoldResult] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    model_name: str = "model"


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Binary classification metrics from win probabilities."""
    preds = (proba >= 0.5).astype(int)
    out = {
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, proba))
    else:
        out["auc"] = 0.5
    return out


def predict_proba_positive(model: Any, X: pd.DataFrame) -> np.ndarray:
    """P(radiant win) from a sklearn-like classifier."""
    return model.predict_proba(X)[:, 1]


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window temporal splits (no shuffle)."""
    n = len(df)
    fold_size = n // (n_splits + 1)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    indices = np.arange(n)
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = min(fold_size * (i + 2), n)
        if test_end <= train_end:
            break
        folds.append((indices[:train_end], indices[train_end:test_end]))
    return folds


def leave_one_ti_splits(
    df: pd.DataFrame,
    min_train: int = 30,
    min_test: int = 15,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-One-TI-Out: train on earlier matches, test on held-out TI."""
    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    for ti_key in TI_KEYS:
        test_mask = df["tournament"] == ti_key
        if int(test_mask.sum()) < min_test:
            continue
        ti_start = df.loc[test_mask, "start_time"].min()
        train_mask = df["start_time"] < ti_start
        if int(train_mask.sum()) < min_train:
            continue
        splits.append(
            (
                ti_key,
                np.where(train_mask)[0],
                np.where(test_mask)[0],
            )
        )
    return splits


def evaluate_folds(
    df: pd.DataFrame,
    folds: list[tuple[str | int, np.ndarray, np.ndarray]],
    feature_cols: list[str],
    fit_fn: FitFn,
    *,
    predict_fn: PredictFn = predict_proba_positive,
    half_life_days: float = 90.0,
) -> list[FoldResult]:
    """Run weighted model evaluation over provided folds."""
    X = df[feature_cols]
    y = df["radiant_win"].to_numpy(dtype=int)
    results: list[FoldResult] = []

    for fold_name, train_idx, test_idx in folds:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        ref = float(df.iloc[train_idx]["start_time"].max())
        weights = compute_sample_weights(
            df.iloc[train_idx],
            reference_time=ref,
            half_life_days=half_life_days,
        )
        model = fit_fn(X_train, y_train, weights)
        proba = predict_fn(model, X_test)
        m = classification_metrics(y_test, proba)
        results.append(
            FoldResult(
                fold=str(fold_name),
                n_train=len(train_idx),
                n_test=len(test_idx),
                log_loss=m["log_loss"],
                brier=m["brier"],
                auc=m["auc"],
                accuracy=m["accuracy"],
                precision=m["precision"],
                recall=m["recall"],
            )
        )
    return results


def summarize_fold_lists(
    walk_forward: list[FoldResult],
    leave_one_ti: list[FoldResult],
    feature_importance: dict[str, float] | None = None,
    title: str = "model",
) -> str:
    """Pretty-print validation summary."""
    lines = [f"=== {title}: Walk-forward ==="]
    for r in walk_forward:
        lines.append(
            f"  fold {r.fold}: n={r.n_test} LL={r.log_loss:.4f} "
            f"Brier={r.brier:.4f} AUC={r.auc:.3f} Acc={r.accuracy:.3f}"
        )
    if walk_forward:
        avg_ll = float(np.mean([r.log_loss for r in walk_forward]))
        avg_auc = float(np.mean([r.auc for r in walk_forward]))
        lines.append(f"  AVG: LL={avg_ll:.4f} AUC={avg_auc:.3f}")

    lines.append(f"=== {title}: Leave-One-TI-Out ===")
    for r in leave_one_ti:
        lines.append(
            f"  {r.fold}: n={r.n_test} LL={r.log_loss:.4f} "
            f"Brier={r.brier:.4f} AUC={r.auc:.3f} Acc={r.accuracy:.3f}"
        )
    if leave_one_ti:
        avg_ll = float(np.mean([r.log_loss for r in leave_one_ti]))
        avg_auc = float(np.mean([r.auc for r in leave_one_ti]))
        lines.append(f"  AVG: LL={avg_ll:.4f} AUC={avg_auc:.3f}")

    if feature_importance:
        lines.append(f"=== {title}: Top features ===")
        for name, imp in list(feature_importance.items())[:10]:
            lines.append(f"  {name:20s} {imp:.4f}")
    return "\n".join(lines)
