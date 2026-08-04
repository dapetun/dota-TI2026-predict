"""Ensemble prediction with model blending."""

import numpy as np
import pandas as pd
from typing import Dict


# Model weights (tuned via validation)
DEFAULT_WEIGHTS = {
    "lgbm": 0.30,
    "xgb": 0.25,
    "catboost": 0.25,
    "rf": 0.10,
    "logistic": 0.10,
}


def ensemble_predict(
    models: Dict[str, object],
    X: pd.DataFrame,
    weights: dict = None,
    feature_cols: list = None,
) -> np.ndarray:
    """Get weighted ensemble probability."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    if feature_cols:
        X = X[feature_cols]

    probs = []
    used_weights = []

    for name, w in weights.items():
        if name not in models:
            continue

        model_data = models[name]
        if name == "logistic":
            model = model_data["model"]
            scaler = model_data["scaler"]
            X_input = scaler.transform(X)
        else:
            model = model_data
            X_input = X

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_input)[:, 1]
        else:
            prob = model.predict(X_input)

        probs.append(prob)
        used_weights.append(w)

    if not probs:
        raise ValueError("No models available for prediction")

    # Normalize weights
    total_w = sum(used_weights)
    norm_weights = [w / total_w for w in used_weights]

    ensemble = np.zeros_like(probs[0], dtype=float)
    for prob, w in zip(probs, norm_weights):
        ensemble += prob * w

    return ensemble


def predict_match_outcome(
    models: dict,
    match_features: pd.DataFrame,
    weights: dict = None,
    feature_cols: list = None,
) -> dict:
    """Predict win probability and series score for a Bo3."""
    win_prob = ensemble_predict(models, match_features, weights, feature_cols)

    if isinstance(win_prob, np.ndarray):
        win_prob = win_prob[0]

    # Bo3 probability distribution
    p20 = (1 - win_prob) ** 2           # Lose 0-2
    p21 = 2 * win_prob * (1 - win_prob) ** 2 / (1 - (1 - win_prob) ** 2 - win_prob ** 2) if win_prob != 0.5 else 0.5
    p21_simple = 2 * win_prob * (1 - win_prob)  # Simplified approximation
    p10 = win_prob ** 2                  # Win 2-0

    # Better approximation: P(win Bo3) = p^2(1+2(1-p))
    p_win_bo3 = win_prob ** 2 * (1 + 2 * (1 - win_prob))

    return {
        "game_win_prob": float(win_prob),
        "series_win_prob": float(p_win_bo3),
        "prob_2_0": float(win_prob ** 2),
        "prob_2_1": float(2 * win_prob ** 2 * (1 - win_prob)),
        "prob_1_2": float(2 * (1 - win_prob) ** 2 * win_prob),
        "prob_0_2": float((1 - win_prob) ** 2),
    }


def predict_all_pairs(
    models: dict,
    team_features: pd.DataFrame,
    team_ids: list,
    weights: dict = None,
    feature_cols: list = None,
) -> pd.DataFrame:
    """Predict win probabilities for all team pairs."""
    results = []

    for i, team_a in enumerate(team_ids):
        for team_b in team_ids[i + 1:]:
            # Build pairwise features
            feat_a = team_features[team_features["team_id"] == team_a]
            feat_b = team_features[team_features["team_id"] == team_b]

            if feat_a.empty or feat_b.empty:
                continue

            # Create differential features
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

            results.append({
                "team_a": team_a,
                "team_b": team_b,
                **pred,
            })

    return pd.DataFrame(results)


def build_win_matrix(predictions_df: pd.DataFrame, team_ids: list) -> pd.DataFrame:
    """Build NxN win probability matrix."""
    n = len(team_ids)
    matrix = pd.DataFrame(np.ones((n, n)) * 0.5, index=team_ids, columns=team_ids)

    for _, row in predictions_df.iterrows():
        a, b = row["team_a"], row["team_b"]
        matrix.loc[a, b] = row["series_win_prob"]
        matrix.loc[b, a] = 1 - row["series_win_prob"]

    return matrix
