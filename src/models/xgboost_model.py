"""XGBoost training with temporal and Leave-One-TI-Out validation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from src.features.match_features import FEATURE_COLUMNS
from src.features.sample_weights import RATING_HALF_LIFE_DAYS, compute_sample_weights
from src.models.validation import (
    FoldResult,
    TrainResult,
    evaluate_folds,
    leave_one_ti_splits,
    summarize_fold_lists,
    walk_forward_splits,
)

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


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    params: dict[str, Any] | None = None,
    calibrate: bool = False,
) -> Any:
    """Fit XGBClassifier, optionally with isotonic calibration."""
    params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    if calibrate and len(X_train) >= 100:
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train, sample_weight=sample_weight)
        return calibrated
    return model


def _importance_from_model(model: Any, feature_cols: list[str]) -> dict[str, float]:
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        try:
            base = model.calibrated_classifiers_[0].estimator
        except (AttributeError, IndexError):
            base = model
    if not hasattr(base, "feature_importances_"):
        return {}
    return {
        c: float(v)
        for c, v in sorted(
            zip(feature_cols, base.feature_importances_),
            key=lambda x: -x[1],
        )
    }


def train_xgboost_pipeline(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    params: dict[str, Any] | None = None,
    half_life_days: float = RATING_HALF_LIFE_DAYS,
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

    def fit_fn(X, y, w):
        return fit_xgboost(X, y, w, params=params, calibrate=False)

    wf_raw = walk_forward_splits(df, n_splits=n_walk_folds)
    wf_folds = [(i + 1, tr, te) for i, (tr, te) in enumerate(wf_raw)]
    walk_results = evaluate_folds(
        df, wf_folds, feature_cols, fit_fn, half_life_days=half_life_days
    )

    loo_folds = leave_one_ti_splits(df)
    loo_results = evaluate_folds(
        df, loo_folds, feature_cols, fit_fn, half_life_days=half_life_days
    )

    X = df[feature_cols]
    y = df["radiant_win"].to_numpy(dtype=int)
    weights = compute_sample_weights(
        df,
        reference_time=float(df["start_time"].max()),
        half_life_days=half_life_days,
    )
    final_model = fit_xgboost(X, y, weights, params=params, calibrate=calibrate_final)

    return TrainResult(
        model=final_model,
        feature_cols=feature_cols,
        walk_forward=walk_results,
        leave_one_ti=loo_results,
        feature_importance=_importance_from_model(final_model, feature_cols),
        params=params,
        model_name="xgboost",
    )


def save_train_result(
    result: TrainResult,
    output_dir: str | Path = "outputs",
    stem: str = "xgb_v1",
) -> Path:
    """Persist model and metrics JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "feature_cols": result.feature_cols,
            "params": result.params,
            "model_name": result.model_name,
        },
        out / f"model_{stem}.joblib",
    )
    report = {
        "model_name": result.model_name,
        "params": result.params,
        "feature_importance": result.feature_importance,
        "walk_forward": [asdict(r) for r in result.walk_forward],
        "leave_one_ti": [asdict(r) for r in result.leave_one_ti],
    }
    report_path = out / f"{stem}_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report_path


def summarize_results(result: TrainResult) -> str:
    """Pretty-print validation summary."""
    return summarize_fold_lists(
        result.walk_forward,
        result.leave_one_ti,
        result.feature_importance,
        title=result.model_name,
    )
