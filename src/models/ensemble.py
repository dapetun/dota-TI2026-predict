"""Simple probability blending for multi-model ensembles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.sample_weights import RATING_HALF_LIFE_DAYS, compute_sample_weights
from src.models.catboost_model import fit_catboost
from src.models.validation import (
    FoldResult,
    TrainResult,
    evaluate_folds,
    leave_one_ti_splits,
    summarize_fold_lists,
    walk_forward_splits,
)
from src.models.xgboost_model import fit_xgboost

# Equal blend of production-ready boosters first; extend later.
DEFAULT_WEIGHTS = {
    "xgb": 0.5,
    "catboost": 0.5,
}


def blend_probas(
    probas: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Weighted average of aligned probability vectors."""
    weights = weights or DEFAULT_WEIGHTS
    used: list[tuple[np.ndarray, float]] = []
    for name, w in weights.items():
        if name not in probas or w <= 0:
            continue
        used.append((np.asarray(probas[name], dtype=float), float(w)))
    if not used:
        raise ValueError("No probabilities available for blending")
    total = sum(w for _, w in used)
    out = np.zeros_like(used[0][0], dtype=float)
    for p, w in used:
        out += p * (w / total)
    return out


def ensemble_predict(
    models: dict[str, Any],
    X: pd.DataFrame,
    weights: dict[str, float] | None = None,
    feature_cols: list[str] | None = None,
    isotonic_calibrator: Any | None = None,
) -> np.ndarray:
    """Blend predict_proba from fitted models (sklearn-like or logistic+scaler)."""
    if feature_cols:
        X = X[feature_cols]
    weights = weights or DEFAULT_WEIGHTS
    probas: dict[str, np.ndarray] = {}

    for name in weights:
        if name not in models:
            continue
        model_data = models[name]
        if name == "logistic" and isinstance(model_data, dict):
            model = model_data["model"]
            scaler = model_data["scaler"]
            X_input = scaler.transform(X)
            probas[name] = model.predict_proba(X_input)[:, 1]
        else:
            model = model_data
            if hasattr(model, "predict_proba"):
                probas[name] = model.predict_proba(X)[:, 1]
            else:
                probas[name] = np.asarray(model.predict(X), dtype=float)

    out = blend_probas(probas, weights)
    if isotonic_calibrator is not None:
        out = np.clip(isotonic_calibrator.predict(out), 1e-6, 1.0 - 1e-6)
    return out


def _fit_blend_bundle(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    *,
    xgb_params: dict | None = None,
    cat_params: dict | None = None,
) -> dict[str, Any]:
    return {
        "xgb": fit_xgboost(X_train, y_train, sample_weight, params=xgb_params, calibrate=False),
        "catboost": fit_catboost(X_train, y_train, sample_weight, params=cat_params),
    }


def _predict_blend(bundle: dict[str, Any], X: pd.DataFrame, weights: dict[str, float] | None = None) -> np.ndarray:
    cal = bundle.get("isotonic_calibrator") if isinstance(bundle, dict) else None
    models = bundle if "xgb" in bundle else bundle
    return ensemble_predict(
        models,
        X,
        weights=weights or DEFAULT_WEIGHTS,
        isotonic_calibrator=cal,
    )


def tune_blend_weights_loo(
    df: pd.DataFrame,
    feature_cols: list[str],
    loo_folds: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    half_life_days: float = RATING_HALF_LIFE_DAYS,
    xgb_params: dict | None = None,
    cat_params: dict | None = None,
    grid: list[float] | None = None,
    calibrate: bool = False,
) -> tuple[dict[str, float], Any | None]:
    """Pick blend weights minimizing pooled LOO-TI log-loss; optional isotonic cal."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import log_loss

    grid = grid or [0.0, 0.25, 0.5, 0.75, 1.0]
    X = df[feature_cols]
    y = df["radiant_win"].to_numpy(dtype=int)
    p_xgb: list[float] = []
    p_cat: list[float] = []
    y_all: list[int] = []

    for _, train_idx, test_idx in loo_folds:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        ref = float(df.iloc[train_idx]["start_time"].max())
        sw = compute_sample_weights(
            df.iloc[train_idx], reference_time=ref, half_life_days=half_life_days
        )
        xgb = fit_xgboost(X_train, y_train, sw, params=xgb_params, calibrate=False)
        cat = fit_catboost(X_train, y_train, sw, params=cat_params)
        p_xgb.extend(xgb.predict_proba(X_test)[:, 1].tolist())
        p_cat.extend(cat.predict_proba(X_test)[:, 1].tolist())
        y_all.extend(y_test.tolist())

    y_arr = np.asarray(y_all, dtype=int)
    px = np.asarray(p_xgb, dtype=float)
    pc = np.asarray(p_cat, dtype=float)
    best_w = 0.5
    best_ll = float("inf")
    for w_x in grid:
        w_c = 1.0 - w_x
        if w_x == 0.0 and w_c == 0.0:
            continue
        blend = w_x * px + w_c * pc
        ll = float(log_loss(y_arr, blend, labels=[0, 1]))
        if ll < best_ll:
            best_ll = ll
            best_w = w_x
    weights = {"xgb": float(best_w), "catboost": float(1.0 - best_w)}
    calibrator = None
    if calibrate and len(y_arr) >= 50:
        blend = weights["xgb"] * px + weights["catboost"] * pc
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(blend, y_arr)
    return weights, calibrator


def evaluate_blend_folds(
    df: pd.DataFrame,
    folds: list[tuple[str | int, np.ndarray, np.ndarray]],
    feature_cols: list[str],
    *,
    half_life_days: float = RATING_HALF_LIFE_DAYS,
    xgb_params: dict | None = None,
    cat_params: dict | None = None,
    blend_weights: dict[str, float] | None = None,
) -> list[FoldResult]:
    """Evaluate XGB+CatBoost blend on the same folds as single models."""
    weights = blend_weights or DEFAULT_WEIGHTS

    def fit_fn(X, y, w):
        return _fit_blend_bundle(X, y, w, xgb_params=xgb_params, cat_params=cat_params)

    def predict_fn(bundle, X):
        return _predict_blend(bundle, X, weights=weights)

    return evaluate_folds(
        df,
        folds,
        feature_cols,
        fit_fn,
        predict_fn=predict_fn,
        half_life_days=half_life_days,
    )


def train_blend_pipeline(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    half_life_days: float = RATING_HALF_LIFE_DAYS,
    n_walk_folds: int = 5,
    xgb_params: dict | None = None,
    cat_params: dict | None = None,
    calibrate: bool = False,
) -> TrainResult:
    """Train final XGB+CatBoost blend and report walk-forward / LOO-TI metrics."""
    df = features_df.sort_values("start_time").reset_index(drop=True)

    wf_raw = walk_forward_splits(df, n_splits=n_walk_folds)
    wf_folds = [(i + 1, tr, te) for i, (tr, te) in enumerate(wf_raw)]
    loo_folds = leave_one_ti_splits(df)
    tuned_weights, isotonic_cal = tune_blend_weights_loo(
        df,
        feature_cols,
        loo_folds,
        half_life_days=half_life_days,
        xgb_params=xgb_params,
        cat_params=cat_params,
        calibrate=calibrate,
    )
    walk_results = evaluate_blend_folds(
        df,
        wf_folds,
        feature_cols,
        half_life_days=half_life_days,
        xgb_params=xgb_params,
        cat_params=cat_params,
        blend_weights=tuned_weights,
    )
    loo_results = evaluate_blend_folds(
        df,
        loo_folds,
        feature_cols,
        half_life_days=half_life_days,
        xgb_params=xgb_params,
        cat_params=cat_params,
        blend_weights=tuned_weights,
    )

    X = df[feature_cols]
    y = df["radiant_win"].to_numpy(dtype=int)
    weights = compute_sample_weights(
        df,
        reference_time=float(df["start_time"].max()),
        half_life_days=half_life_days,
    )
    bundle = _fit_blend_bundle(
        X, y, weights, xgb_params=xgb_params, cat_params=cat_params
    )
    if isotonic_cal is not None:
        bundle["isotonic_calibrator"] = isotonic_cal

    params: dict[str, Any] = {
        "weights": tuned_weights,
        "xgb": xgb_params,
        "catboost": cat_params,
        "calibrated": isotonic_cal is not None,
    }

    return TrainResult(
        model=bundle,
        feature_cols=feature_cols,
        walk_forward=walk_results,
        leave_one_ti=loo_results,
        feature_importance={},
        params=params,
        model_name="xgb_catboost_blend",
    )


def summarize_blend(result: TrainResult) -> str:
    """Pretty-print blend validation summary."""
    return summarize_fold_lists(
        result.walk_forward,
        result.leave_one_ti,
        title=result.model_name,
    )


def predict_match_outcome(
    models: dict,
    match_features: pd.DataFrame,
    weights: dict | None = None,
    feature_cols: list | None = None,
) -> dict:
    """Predict win probability and approximate Bo3 score distribution."""
    win_prob = ensemble_predict(models, match_features, weights, feature_cols)
    if isinstance(win_prob, np.ndarray):
        win_prob = float(win_prob[0])
    else:
        win_prob = float(win_prob)

    p_win_bo3 = win_prob**2 * (1 + 2 * (1 - win_prob))
    return {
        "game_win_prob": win_prob,
        "series_win_prob": float(p_win_bo3),
        "prob_2_0": float(win_prob**2),
        "prob_2_1": float(2 * win_prob**2 * (1 - win_prob)),
        "prob_1_2": float(2 * (1 - win_prob) ** 2 * win_prob),
        "prob_0_2": float((1 - win_prob) ** 2),
    }


def predict_all_pairs(
    models: dict,
    team_features: pd.DataFrame,
    team_ids: list,
    weights: dict | None = None,
    feature_cols: list | None = None,
) -> pd.DataFrame:
    """Predict win probabilities for all unordered team pairs."""
    results = []
    for i, team_a in enumerate(team_ids):
        for team_b in team_ids[i + 1 :]:
            feat_a = team_features[team_features["team_id"] == team_a]
            feat_b = team_features[team_features["team_id"] == team_b]
            if feat_a.empty or feat_b.empty:
                continue
            match_feat = {}
            for col in feat_a.columns:
                if col == "team_id":
                    continue
                match_feat[f"a_{col}"] = feat_a.iloc[0][col]
                match_feat[f"b_{col}"] = feat_b.iloc[0][col]
                if pd.api.types.is_numeric_dtype(type(feat_a.iloc[0][col])):
                    match_feat[f"diff_{col}"] = feat_a.iloc[0][col] - feat_b.iloc[0][col]
            df = pd.DataFrame([match_feat])
            pred = predict_match_outcome(models, df, weights, feature_cols)
            results.append({"team_a": team_a, "team_b": team_b, **pred})
    return pd.DataFrame(results)


def build_win_matrix(predictions_df: pd.DataFrame, team_ids: list) -> pd.DataFrame:
    """Build NxN series-win probability matrix from pair predictions."""
    n = len(team_ids)
    matrix = pd.DataFrame(np.ones((n, n)) * 0.5, index=team_ids, columns=team_ids)
    for _, row in predictions_df.iterrows():
        a, b = row["team_a"], row["team_b"]
        matrix.loc[a, b] = row["series_win_prob"]
        matrix.loc[b, a] = 1 - row["series_win_prob"]
    return matrix
