"""ML models for Bo3 win prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path
import joblib
import json

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    log_loss, accuracy_score, roc_auc_score,
    brier_score_loss, classification_report,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get numeric feature columns (exclude IDs, targets, metadata)."""
    exclude = {
        "match_id", "radiant_team", "dire_team", "start_time",
        "duration", "league_id", "radiant_win",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def prepare_data(
    features_df: pd.DataFrame,
    feature_cols: list = None,
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Prepare X, y from feature matrix."""
    if feature_cols is None:
        feature_cols = get_feature_columns(features_df)

    df = features_df.dropna(subset=feature_cols + ["radiant_win"]).copy()
    X = df[feature_cols].fillna(0)
    y = df["radiant_win"].astype(int)

    return X, y, feature_cols


def train_lightgbm(X, y, params: dict = None, cv_folds: int = 5):
    """Train LightGBM model with time series CV."""
    import lightgbm as lgb

    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_log_loss")
    print(f"LightGBM CV log-loss: {-scores.mean():.4f} (+/- {scores.std():.4f})")

    model.fit(X, y)
    return model


def train_xgboost(X, y, params: dict = None, cv_folds: int = 5):
    """Train XGBoost model."""
    import xgboost as xgb

    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "use_label_encoder": False,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_log_loss")
    print(f"XGBoost CV log-loss: {-scores.mean():.4f} (+/- {scores.std():.4f})")

    model.fit(X, y)
    return model


def train_catboost(X, y, params: dict = None, cv_folds: int = 5):
    """Train CatBoost model."""
    from catboost import CatBoostClassifier

    default_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 3,
        "loss_function": "Logloss",
        "random_seed": 42,
        "verbose": 0,
    }
    if params:
        default_params.update(params)

    model = CatBoostClassifier(**default_params)

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_log_loss")
    print(f"CatBoost CV log-loss: {-scores.mean():.4f} (+/- {scores.std():.4f})")

    model.fit(X, y)
    return model


def train_random_forest(X, y, params: dict = None, cv_folds: int = 5):
    """Train Random Forest model."""
    default_params = {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = RandomForestClassifier(**default_params)

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_log_loss")
    print(f"Random Forest CV log-loss: {-scores.mean():.4f} (+/- {scores.std():.4f})")

    model.fit(X, y)
    return model


def train_logistic(X, y, cv_folds: int = 5):
    """Train calibrated Logistic Regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model = CalibratedClassifierCV(model, cv=3)

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="neg_log_loss")
    print(f"Logistic CV log-loss: {-scores.mean():.4f} (+/- {scores.std():.4f})")

    model.fit(X_scaled, y)
    return {"model": model, "scaler": scaler}


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str = "outputs",
) -> Dict[str, object]:
    """Train all models and return dict of trained models."""
    models = {}

    print("=== Training LightGBM ===")
    models["lgbm"] = train_lightgbm(X, y)

    print("\n=== Training XGBoost ===")
    models["xgb"] = train_xgboost(X, y)

    print("\n=== Training CatBoost ===")
    models["catboost"] = train_catboost(X, y)

    print("\n=== Training Random Forest ===")
    models["rf"] = train_random_forest(X, y)

    print("\n=== Training Logistic Regression ===")
    models["logistic"] = train_logistic(X, y)

    return models


def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate all models on test set."""
    results = []

    for name, model_data in models.items():
        if name == "logistic":
            model = model_data["model"]
            scaler = model_data["scaler"]
            X_eval = scaler.transform(X_test)
        else:
            model = model_data
            X_eval = X_test

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_eval)[:, 1]
        else:
            proba = model.predict(X_eval)

        preds = (proba > 0.5).astype(int)

        results.append({
            "model": name,
            "log_loss": log_loss(y_test, proba),
            "accuracy": accuracy_score(y_test, preds),
            "brier": brier_score_loss(y_test, proba),
            "auc_roc": roc_auc_score(y_test, proba),
        })

    return pd.DataFrame(results).sort_values("log_loss")


def save_models(models: dict, feature_cols: list, output_dir: str = "outputs"):
    """Save trained models and feature list."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model_file = out / f"model_{name}.joblib"
        joblib.dump(model, model_file)

    with open(out / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"Saved {len(models)} models to {output_dir}")


def load_models(output_dir: str = "outputs") -> Tuple[dict, list]:
    """Load trained models and feature list."""
    out = Path(output_dir)
    models = {}

    for model_file in out.glob("model_*.joblib"):
        name = model_file.stem.replace("model_", "")
        models[name] = joblib.load(model_file)

    with open(out / "feature_cols.json") as f:
        feature_cols = json.load(f)

    return models, feature_cols
