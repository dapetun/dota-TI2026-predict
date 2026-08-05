"""XGBoost training with temporal and Leave-One-TI-Out validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_collection.tournaments import TI_KEYS
from src.features.match_features import FEATURE_COLUMNS
from src.features.sample_weights import compute_sample_weights


DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


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


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
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


def _fit_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    params: dict[str, Any],
    calibrate: bool = False,
) -> Any:
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    if calibrate and len(X_train) >= 100:
        # Isotonic on a time-respecting holdout would be better; for MVP use CV.
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train, sample_weight=sample_weight)
        return calibrated
    return model


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
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
    """Leave-One-TI-Out: train on all earlier matches, test on held-out TI."""
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
    params: dict[str, Any],
    half_life_days: float = 90.0,
) -> list[FoldResult]:
    """Run weighted XGBoost evaluation over provided folds."""
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
        model = _fit_xgb(X_train, y_train, weights, params, calibrate=False)
        proba = _predict_proba(model, X_test)
        m = _metrics(y_test, proba)
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


def train_xgboost_pipeline(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    params: dict[str, Any] | None = None,
    half_life_days: float = 90.0,
    n_walk_folds: int = 5,
    calibrate_final: bool = True,
) -> TrainResult:
    """Train XGBoost with walk-forward + Leave-One-TI-Out, then fit final model."""
    feature_cols = feature_cols or FEATURE_COLUMNS
    params = {**DEFAULT_XGB_PARAMS, **(params or {})}

    df = features_df.sort_values("start_time").reset_index(drop=True)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    wf_raw = walk_forward_splits(df, n_splits=n_walk_folds)
    wf_folds = [(i + 1, tr, te) for i, (tr, te) in enumerate(wf_raw)]
    walk_results = evaluate_folds(df, wf_folds, feature_cols, params, half_life_days)

    loo_folds = leave_one_ti_splits(df)
    loo_results = evaluate_folds(df, loo_folds, feature_cols, params, half_life_days)

    X = df[feature_cols]
    y = df["radiant_win"].to_numpy(dtype=int)
    weights = compute_sample_weights(
        df,
        reference_time=float(df["start_time"].max()),
        half_life_days=half_life_days,
    )
    final_model = _fit_xgb(X, y, weights, params, calibrate=calibrate_final)

    # Importance from underlying booster when calibrated.
    base = final_model
    if hasattr(final_model, "calibrated_classifiers_"):
        # Take first fold estimator's base estimator if available.
        try:
            base = final_model.calibrated_classifiers_[0].estimator
        except Exception:
            base = final_model
    importance: dict[str, float] = {}
    if hasattr(base, "feature_importances_"):
        importance = {
            c: float(v)
            for c, v in sorted(
                zip(feature_cols, base.feature_importances_),
                key=lambda x: -x[1],
            )
        }

    return TrainResult(
        model=final_model,
        feature_cols=feature_cols,
        walk_forward=walk_results,
        leave_one_ti=loo_results,
        feature_importance=importance,
        params=params,
    )


def save_train_result(
    result: TrainResult,
    output_dir: str | Path = "outputs",
) -> Path:
    """Persist model and metrics JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": result.model, "feature_cols": result.feature_cols, "params": result.params},
        out / "model_xgb_v1.joblib",
    )
    report = {
        "params": result.params,
        "feature_importance": result.feature_importance,
        "walk_forward": [asdict(r) for r in result.walk_forward],
        "leave_one_ti": [asdict(r) for r in result.leave_one_ti],
    }
    report_path = out / "xgb_v1_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report_path


def summarize_results(result: TrainResult) -> str:
    """Pretty-print validation summary."""
    lines = ["=== Walk-forward ==="]
    for r in result.walk_forward:
        lines.append(
            f"  fold {r.fold}: n={r.n_test} LL={r.log_loss:.4f} "
            f"Brier={r.brier:.4f} AUC={r.auc:.3f} Acc={r.accuracy:.3f}"
        )
    if result.walk_forward:
        avg_ll = np.mean([r.log_loss for r in result.walk_forward])
        avg_auc = np.mean([r.auc for r in result.walk_forward])
        lines.append(f"  AVG: LL={avg_ll:.4f} AUC={avg_auc:.3f}")

    lines.append("=== Leave-One-TI-Out ===")
    for r in result.leave_one_ti:
        lines.append(
            f"  {r.fold}: n={r.n_test} LL={r.log_loss:.4f} "
            f"Brier={r.brier:.4f} AUC={r.auc:.3f} Acc={r.accuracy:.3f}"
        )
    if result.leave_one_ti:
        avg_ll = np.mean([r.log_loss for r in result.leave_one_ti])
        avg_auc = np.mean([r.auc for r in result.leave_one_ti])
        lines.append(f"  AVG: LL={avg_ll:.4f} AUC={avg_auc:.3f}")

    lines.append("=== Top features ===")
    for name, imp in list(result.feature_importance.items())[:10]:
        lines.append(f"  {name:20s} {imp:.4f}")
    return "\n".join(lines)
