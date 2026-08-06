"""Compare XGBoost, CatBoost and equal blend on the same feature matrix."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.data_collection.match_details import (
    load_player_matches,
    save_player_matches,
    summarize_player_coverage,
)
from src.data_collection.match_loader import (
    load_raw_matchlists,
    save_canonical_matches,
    summarize_matches,
)
from src.features.match_features import FEATURE_COLUMNS, build_match_feature_matrix, save_features
from src.features.player_features import (
    PLAYER_FEATURE_COLUMNS,
    build_player_match_features,
    merge_team_and_player_features,
)
from src.features.chemistry_features import (
    CHEMISTRY_FEATURE_COLUMNS,
    build_chemistry_features,
    merge_chemistry_features,
)
from src.models.catboost_model import save_catboost_result, train_catboost_pipeline
from src.models.ensemble import summarize_blend, train_blend_pipeline, tune_blend_weights_loo
from src.models.validation import FoldResult, leave_one_ti_splits
from src.models.xgboost_model import save_train_result, summarize_results, train_xgboost_pipeline


def _avg_auc(folds: list[FoldResult]) -> float | None:
    if not folds:
        return None
    return float(np.mean([f.auc for f in folds]))


def _avg_ll(folds: list[FoldResult]) -> float | None:
    if not folds:
        return None
    return float(np.mean([f.log_loss for f in folds]))


def run_model_compare(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    features_dir: str = "data/features",
    output_dir: str = "outputs",
) -> dict:
    """Train XGB + CatBoost + blend; write comparison metrics."""
    print("=" * 60)
    print("TI 2026 — model compare (XGB / CatBoost / blend)")
    print("=" * 60)

    matches = load_raw_matchlists(raw_dir)
    summary = summarize_matches(matches)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if matches.empty:
        raise FileNotFoundError(f"No matchlists in {raw_dir}")
    save_canonical_matches(matches, processed_dir)

    players = load_player_matches(raw_dir)
    coverage = summarize_player_coverage(matches, players)
    print(json.dumps(coverage, indent=2))
    if not players.empty:
        save_player_matches(players, processed_dir)

    team_features = build_match_feature_matrix(matches, min_games=5)
    player_features = build_player_match_features(matches, players)
    chemistry = build_chemistry_features(matches, players)
    features = merge_team_and_player_features(team_features, player_features)
    features = merge_chemistry_features(features, chemistry)
    feature_cols = FEATURE_COLUMNS + PLAYER_FEATURE_COLUMNS + CHEMISTRY_FEATURE_COLUMNS
    feat_path = save_features(features, features_dir)
    print(f"Features: {len(features)} rows, {len(feature_cols)} cols -> {feat_path}")
    print(f"Rows with player stats: {int(features['has_player_stats'].sum())}")
    print(f"Rows with chemistry: {int(features['has_chemistry'].sum())}")

    print("\n--- XGBoost ---")
    xgb_result = train_xgboost_pipeline(features, feature_cols=feature_cols)
    print(summarize_results(xgb_result))
    xgb_path = save_train_result(xgb_result, output_dir, stem="xgb_v1")

    print("\n--- CatBoost ---")
    cat_result = train_catboost_pipeline(features, feature_cols=feature_cols)
    print(summarize_results(cat_result))
    cat_path = save_catboost_result(cat_result, output_dir, stem="catboost_v1")

    print("\n--- Blend (LOO-tuned XGB + CatBoost) ---")
    blend_result = train_blend_pipeline(features, feature_cols=feature_cols, calibrate=False)
    print(summarize_blend(blend_result))

    # Isotonic on pooled LOO (metrics only; not saved to production bundle).
    loo_folds = leave_one_ti_splits(features.sort_values("start_time").reset_index(drop=True))
    _, isotonic_cal = tune_blend_weights_loo(
        features.sort_values("start_time").reset_index(drop=True),
        feature_cols,
        loo_folds,
        calibrate=True,
    )
    calibrated_note = isotonic_cal is not None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": blend_result.model,
            "feature_cols": blend_result.feature_cols,
            "params": blend_result.params,
            "model_name": blend_result.model_name,
        },
        out / "model_blend_v1.joblib",
    )

    comparison = {
        "player_coverage": coverage,
        "n_features_rows": len(features),
        "models": {
            "xgboost": {
                "walk_forward_avg_auc": _avg_auc(xgb_result.walk_forward),
                "leave_one_ti_avg_auc": _avg_auc(xgb_result.leave_one_ti),
                "walk_forward_avg_logloss": _avg_ll(xgb_result.walk_forward),
                "leave_one_ti_avg_logloss": _avg_ll(xgb_result.leave_one_ti),
                "metrics_path": str(xgb_path),
            },
            "catboost": {
                "walk_forward_avg_auc": _avg_auc(cat_result.walk_forward),
                "leave_one_ti_avg_auc": _avg_auc(cat_result.leave_one_ti),
                "walk_forward_avg_logloss": _avg_ll(cat_result.walk_forward),
                "leave_one_ti_avg_logloss": _avg_ll(cat_result.leave_one_ti),
                "metrics_path": str(cat_path),
            },
            "blend": {
                "walk_forward_avg_auc": _avg_auc(blend_result.walk_forward),
                "leave_one_ti_avg_auc": _avg_auc(blend_result.leave_one_ti),
                "walk_forward_avg_logloss": _avg_ll(blend_result.walk_forward),
                "leave_one_ti_avg_logloss": _avg_ll(blend_result.leave_one_ti),
                "weights": blend_result.params.get("weights"),
                "isotonic_loo_evaluated": calibrated_note,
            },
        },
    }
    cmp_path = out / "model_compare.json"
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison -> {cmp_path}")
    return comparison
