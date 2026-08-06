"""CatBoost training with the same temporal validation as XGBoost."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.features.match_features import FEATURE_COLUMNS
from src.features.sample_weights import compute_sample_weights
from src.models.validation import (
    TrainResult,
    evaluate_folds,
    leave_one_ti_splits,
    summarize_fold_lists,
    walk_forward_splits,
)

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 5,
    "l2_leaf_reg": 3.0,
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    params: dict[str, Any] | None = None,
) -> CatBoostClassifier:
    """Fit CatBoostClassifier with optional sample weights."""
    params = {**DEFAULT_CATBOOST_PARAMS, **(params or {})}
    model = CatBoostClassifier(**params)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def _importance_from_model(model: CatBoostClassifier, feature_cols: list[str]) -> dict[str, float]:
    raw = model.get_feature_importance()
    return {
        c: float(v)
        for c, v in sorted(zip(feature_cols, raw), key=lambda x: -x[1])
    }


def train_catboost_pipeline(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    params: dict[str, Any] | None = None,
    half_life_days: float = 90.0,
    n_walk_folds: int = 5,
) -> TrainResult:
    """Train CatBoost with walk-forward + Leave-One-TI-Out, then fit final model."""
    feature_cols = feature_cols or FEATURE_COLUMNS
    params = {**DEFAULT_CATBOOST_PARAMS, **(params or {})}

    df = features_df.sort_values("start_time").reset_index(drop=True)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    def fit_fn(X, y, w):
        return fit_catboost(X, y, w, params=params)

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
    final_model = fit_catboost(X, y, weights, params=params)

    return TrainResult(
        model=final_model,
        feature_cols=feature_cols,
        walk_forward=walk_results,
        leave_one_ti=loo_results,
        feature_importance=_importance_from_model(final_model, feature_cols),
        params=params,
        model_name="catboost",
    )


def save_catboost_result(
    result: TrainResult,
    output_dir: str | Path = "outputs",
    stem: str = "catboost_v1",
) -> Path:
    """Persist CatBoost model and metrics JSON."""
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
        import json

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
