"""Compare XGBoost, CatBoost and equal blend on the same feature matrix."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

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
from src.config import sample_half_life_days
from src.features.team_stitching import apply_team_stitch, build_team_stitch_map
from src.models.artifact_hash import write_sha256_sidecar
from src.ti2026.multisource import PATCH_741_START_TS, PATCH_IN_MULT
from src.models.catboost_model import save_catboost_result, train_catboost_pipeline
from src.models.ensemble import summarize_blend, train_blend_pipeline
from src.models.validation import FoldResult
from src.models.xgboost_model import save_train_result, summarize_results, train_xgboost_pipeline


def _avg_auc(folds: list[FoldResult]) -> float | None:
    if not folds:
        return None
    return float(np.mean([f.auc for f in folds]))


def _avg_ll(folds: list[FoldResult]) -> float | None:
    if not folds:
        return None
    return float(np.mean([f.log_loss for f in folds]))


def _avg_brier(folds: list[FoldResult]) -> float | None:
    if not folds:
        return None
    return float(np.mean([f.brier for f in folds]))


def run_model_compare(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    features_dir: str = "data/features",
    output_dir: str = "outputs",
    *,
    stitch_teams: bool = True,
    lan_only_chemistry: bool = False,
    half_life_days: float | None = None,
) -> dict:
    """Train XGB + CatBoost + blend; write comparison metrics."""
    if half_life_days is None:
        half_life_days = sample_half_life_days()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("=" * 60)
    logger.info("TI 2026 — model compare (XGB / CatBoost / blend) v0.3")
    logger.info("=" * 60)

    matches = load_raw_matchlists(raw_dir)
    summary = summarize_matches(matches)
    logger.info("%s", json.dumps(summary, indent=2, ensure_ascii=False))
    if matches.empty:
        raise FileNotFoundError(f"No matchlists in {raw_dir}")

    players = load_player_matches(raw_dir)
    stitch_n = 0
    if stitch_teams and not players.empty:
        stitch = build_team_stitch_map(players, matches, threshold=0.6)
        matches, players = apply_team_stitch(matches, stitch, players)
        stitch_n = sum(1 for a, b in stitch.items() if a != b)
        logger.info("Team stitch: %s team_ids remapped (Jaccard>=0.6)", stitch_n)

    save_canonical_matches(matches, processed_dir)
    coverage = summarize_player_coverage(matches, players)
    logger.info("%s", json.dumps(coverage, indent=2))
    if not players.empty:
        save_player_matches(players, processed_dir)

    team_features = build_match_feature_matrix(matches, min_games=5)
    player_features = build_player_match_features(matches, players)
    chemistry = build_chemistry_features(matches, players, lan_only=lan_only_chemistry)
    features = merge_team_and_player_features(team_features, player_features)
    features = merge_chemistry_features(features, chemistry)
    feature_cols = FEATURE_COLUMNS + PLAYER_FEATURE_COLUMNS + CHEMISTRY_FEATURE_COLUMNS
    feat_path = save_features(features, features_dir)
    logger.info("Features: %s rows, %s cols -> %s", len(features), len(feature_cols), feat_path)
    logger.info("Rows with player stats: %s", int(features["has_player_stats"].sum()))
    logger.info("Rows with chemistry: %s", int(features["has_chemistry"].sum()))
    logger.info(
        "Sample half-life: %sd · patch_741_ts=%s · patch_mult=%s",
        half_life_days,
        PATCH_741_START_TS,
        PATCH_IN_MULT,
    )

    logger.info("--- XGBoost ---")
    xgb_result = train_xgboost_pipeline(
        features, feature_cols=feature_cols, half_life_days=half_life_days
    )
    logger.info("%s", summarize_results(xgb_result))
    xgb_path = save_train_result(xgb_result, output_dir, stem="xgb_v1")

    logger.info("--- CatBoost ---")
    cat_result = train_catboost_pipeline(
        features, feature_cols=feature_cols, half_life_days=half_life_days
    )
    logger.info("%s", summarize_results(cat_result))
    cat_path = save_catboost_result(cat_result, output_dir, stem="catboost_v1")

    logger.info("--- Blend (LOO-tuned XGB + CatBoost, isotonic when calibrate=True) ---")
    blend_result = train_blend_pipeline(
        features,
        feature_cols=feature_cols,
        calibrate=True,
        half_life_days=half_life_days,
    )
    logger.info("%s", summarize_blend(blend_result))

    calibrated_note = bool(blend_result.params.get("calibrated"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    blend_path = out / "model_blend_v1.joblib"
    joblib.dump(
        {
            "model": blend_result.model,
            "feature_cols": blend_result.feature_cols,
            "params": blend_result.params,
            "model_name": blend_result.model_name,
        },
        blend_path,
    )
    blend_sha256 = write_sha256_sidecar(blend_path)

    comparison = {
        "version": "0.3.0-prod",
        "player_coverage": coverage,
        "n_features_rows": len(features),
        "n_feature_cols": len(feature_cols),
        "n_leagues": summary.get("n_tournaments"),
        "n_matches": summary.get("n_matches"),
        "team_stitch_remaps": stitch_n,
        "lan_only_chemistry": lan_only_chemistry,
        "half_life_days": half_life_days,
        "patch_741_start_ts": PATCH_741_START_TS,
        "patch_sample_mult": PATCH_IN_MULT,
        "model_blend_path": str(blend_path),
        "model_blend_sha256": blend_sha256,
        "models": {
            "xgboost": {
                "walk_forward_avg_auc": _avg_auc(xgb_result.walk_forward),
                "leave_one_ti_avg_auc": _avg_auc(xgb_result.leave_one_ti),
                "walk_forward_avg_logloss": _avg_ll(xgb_result.walk_forward),
                "leave_one_ti_avg_logloss": _avg_ll(xgb_result.leave_one_ti),
                "walk_forward_avg_brier": _avg_brier(xgb_result.walk_forward),
                "leave_one_ti_avg_brier": _avg_brier(xgb_result.leave_one_ti),
                "metrics_path": str(xgb_path),
            },
            "catboost": {
                "walk_forward_avg_auc": _avg_auc(cat_result.walk_forward),
                "leave_one_ti_avg_auc": _avg_auc(cat_result.leave_one_ti),
                "walk_forward_avg_logloss": _avg_ll(cat_result.walk_forward),
                "leave_one_ti_avg_logloss": _avg_ll(cat_result.leave_one_ti),
                "walk_forward_avg_brier": _avg_brier(cat_result.walk_forward),
                "leave_one_ti_avg_brier": _avg_brier(cat_result.leave_one_ti),
                "metrics_path": str(cat_path),
            },
            "blend": {
                "walk_forward_avg_auc": _avg_auc(blend_result.walk_forward),
                "leave_one_ti_avg_auc": _avg_auc(blend_result.leave_one_ti),
                "walk_forward_avg_logloss": _avg_ll(blend_result.walk_forward),
                "leave_one_ti_avg_logloss": _avg_ll(blend_result.leave_one_ti),
                "walk_forward_avg_brier": _avg_brier(blend_result.walk_forward),
                "leave_one_ti_avg_brier": _avg_brier(blend_result.leave_one_ti),
                "weights": blend_result.params.get("weights"),
                "isotonic_calibrated": calibrated_note,
            },
        },
    }
    cmp_path = out / "model_compare.json"
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    logger.info("Comparison -> %s", cmp_path)
    return comparison
