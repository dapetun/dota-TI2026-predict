"""Training pipeline: ETL → features → XGBoost → evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from src.data_collection.match_loader import (
    load_raw_matchlists,
    save_canonical_matches,
    summarize_matches,
)
from src.evaluation.metrics import calibration_analysis, plot_calibration, plot_feature_importance
from src.features.match_features import (
    FEATURE_COLUMNS,
    build_match_feature_matrix,
    save_features,
)
from src.models.xgboost_model import (
    save_train_result,
    summarize_results,
    train_xgboost_pipeline,
)


def run_training(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    features_dir: str = "data/features",
    output_dir: str = "outputs",
) -> dict:
    """Run ETL, feature build, XGBoost training and evaluation end-to-end."""
    print("=" * 60)
    print("TI 2026 — XGBoost training")
    print("=" * 60)

    print("\n[1/4] Loading OpenDota matchlists...")
    matches = load_raw_matchlists(raw_dir)
    summary = summarize_matches(matches)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if matches.empty:
        raise FileNotFoundError(
            f"No matchlists found in {raw_dir}. Run scripts/download_data.py --list-only first."
        )
    save_canonical_matches(matches, processed_dir)

    print("\n[2/4] Building team-level features (no leakage)...")
    features = build_match_feature_matrix(matches, min_games=5)
    feat_path = save_features(features, features_dir)
    print(
        f"  Features: {len(features)} matches, {len(FEATURE_COLUMNS)} columns -> {feat_path}"
    )
    print(f"  Date range: {features['date'].min()} .. {features['date'].max()}")
    print(f"  Radiant WR: {features['radiant_win'].mean():.3f}")

    print("\n[3/4] Training XGBoost (walk-forward + Leave-One-TI-Out)...")
    result = train_xgboost_pipeline(features)
    print(summarize_results(result))
    metrics_path = save_train_result(result, output_dir)

    print("\n[4/4] Calibration / importance plots...")
    n = len(features)
    cut = int(n * 0.8)
    holdout = features.iloc[cut:]
    X_h = holdout[result.feature_cols]
    y_h = holdout["radiant_win"].to_numpy(dtype=int)
    proba = result.model.predict_proba(X_h)[:, 1]
    plot_calibration(y_h, {"xgboost": proba}, output_dir)
    cal = calibration_analysis(y_h, proba)
    cal.to_csv(Path(output_dir) / "reliability_curve.csv", index=False)

    base = result.model
    if hasattr(result.model, "calibrated_classifiers_"):
        try:
            base = result.model.calibrated_classifiers_[0].estimator
        except Exception:
            pass
    plot_feature_importance(base, result.feature_cols, output_dir=output_dir)

    print(f"\nSaved metrics -> {metrics_path}")
    print("Training complete.")
    return {
        "match_summary": summary,
        "n_features_rows": len(features),
        "metrics_path": str(metrics_path),
    }
