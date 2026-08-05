"""Main orchestrator for TI 2026 prediction pipeline."""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Local imports
from src.data_collection.data_loader import (
    load_raw_matches, expand_players, build_team_match_results,
    save_processed_data, load_processed_data, normalize_team_name,
)
from src.features.rating_systems import EloRating, GlickoRating, build_elo_history
from src.features.feature_engineering import (
    compute_team_form, compute_hero_features, compute_player_features,
    build_match_features, add_elo_features, save_features,
)
from src.models.model_training import (
    prepare_data, train_all_models, evaluate_models, save_models, load_models,
)
from src.models.ensemble import (
    ensemble_predict, predict_match_outcome, predict_all_pairs, build_win_matrix,
)
from src.simulation.tournament_sim import (
    simulate_swiss_stage, SwissConfig,
)
from src.evaluation.metrics import (
    plot_calibration, plot_feature_importance, generate_report,
)
from src.ti2026.teams import (
    TI2026_TEAMS, get_team_ids, normalize_team_name, build_teams_df,
    SWISS_CONFIG, POWER_RANKINGS,
)


def step_collect_data(raw_dir: str = "data/raw"):
    """Step 1: Collect data from APIs."""
    print("=" * 60)
    print("STEP 1: Data Collection")
    print("=" * 60)

    # Try OpenDota first (no auth needed)
    from src.data_collection.opendota_api import download_opendota_data
    try:
        print("Downloading from OpenDota...")
        download_opendota_data(raw_dir)
    except Exception as e:
        print(f"OpenDota download failed: {e}")

    # Try STRATZ if token available
    import os
    if os.environ.get("STRATZ_TOKEN"):
        from src.data_collection.stratz_api import download_stratz_data
        try:
            print("Downloading from STRATZ...")
            download_stratz_data(output_dir=raw_dir, start_date="2024-01-01")
        except Exception as e:
            print(f"STRATZ download failed: {e}")

    print("Data collection complete.\n")


def step_process_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """Step 2: Process raw data into canonical format."""
    print("=" * 60)
    print("STEP 2: Data Processing")
    print("=" * 60)

    matches = load_raw_matches(raw_dir)
    if matches.empty:
        print("No raw data found. Run step_collect_data first or provide data manually.")
        return

    print(f"Loaded {len(matches)} raw matches")

    players = expand_players(matches)
    print(f"Expanded to {len(players)} player-match records")

    team_matches = build_team_match_results(matches, players)
    print(f"Built {len(team_matches)} team-match records")

    # Normalize team names
    if "team_id" in team_matches.columns:
        team_matches["team_id"] = team_matches["team_id"].apply(
            lambda x: normalize_team_name(str(x)) if pd.notna(x) else x
        )

    save_processed_data(team_matches, players, processed_dir)
    print("Data processing complete.\n")


def step_build_features(
    processed_dir: str = "data/processed",
    features_dir: str = "data/features",
):
    """Step 3: Build ML features."""
    print("=" * 60)
    print("STEP 3: Feature Engineering")
    print("=" * 60)

    team_matches, player_matches = load_processed_data(processed_dir)
    if team_matches.empty:
        print("No processed data. Run step_process_data first.")
        return

    print(f"Building form features...")
    form_df = compute_team_form(team_matches)

    print("Building hero features...")
    hero_features = compute_hero_features(player_matches)

    print("Building player features...")
    player_features = compute_player_features(player_matches)

    print("Building Elo history...")
    elo_df = build_elo_history(team_matches)

    print("Building match features...")
    features_df = build_match_features(
        team_matches, player_matches, form_df, hero_features, player_features
    )
    features_df = add_elo_features(features_df, elo_df)

    save_features(features_df, features_dir)
    print(f"Built {len(features_df)} match features with {len(features_df.columns)} columns")
    print("Feature engineering complete.\n")

    return features_df


def step_train_models(features_dir: str = "data/features", output_dir: str = "outputs"):
    """Step 4: Train ML models."""
    print("=" * 60)
    print("STEP 4: Model Training")
    print("=" * 60)

    features_file = Path(features_dir) / "match_features.csv"
    if not features_file.exists():
        print("No features found. Run step_build_features first.")
        return

    features_df = pd.read_csv(features_file)
    X, y, feature_cols = prepare_data(features_df)

    print(f"Training on {len(X)} samples, {len(feature_cols)} features")

    models = train_all_models(X, y, output_dir)
    results = evaluate_models(models, X, y)

    print("\n=== Model Comparison ===")
    print(results.to_string(index=False))

    save_models(models, feature_cols, output_dir)
    print("Model training complete.\n")

    return models, feature_cols


def step_generate_predictions(
    models: dict,
    feature_cols: list,
    output_dir: str = "outputs",
):
    """Step 5: Generate TI 2026 predictions."""
    print("=" * 60)
    print("STEP 5: TI 2026 Predictions")
    print("=" * 60)

    teams = get_team_ids()
    print(f"Predictions for {len(teams)} teams")

    # Build synthetic current features for TI 2026 teams
    # In production, this uses latest match data
    team_features = pd.DataFrame({"team_id": teams})

    # Add power ranking as baseline feature
    for idx, row in team_features.iterrows():
        team = row["team_id"]
        rank = POWER_RANKINGS.get(team, 16)
        team_features.at[idx, "power_ranking"] = rank

    # Generate all pairwise predictions
    predictions = predict_all_pairs(models, team_features, teams, feature_cols=feature_cols)
    win_matrix = build_win_matrix(predictions, teams)

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    win_matrix.to_csv(Path(output_dir) / "win_matrix.csv")
    predictions.to_csv(Path(output_dir) / "pairwise_predictions.csv", index=False)

    print("\n=== Win Probability Matrix ===")
    print(win_matrix.round(3).to_string())

    print("\nPredictions saved to", output_dir)
    return win_matrix


def step_simulate_tournament(
    win_matrix: pd.DataFrame,
    output_dir: str = "outputs",
    n_simulations: int = 100000,
):
    """Step 6: Monte Carlo tournament simulation."""
    print("=" * 60)
    print(f"STEP 6: Tournament Simulation ({n_simulations:,} iterations)")
    print("=" * 60)

    teams = get_team_ids()
    config = SwissConfig(**SWISS_CONFIG)

    results = simulate_swiss_stage(
        win_matrix, teams, config,
        n_simulations=n_simulations,
    )

    print("\n=== Swiss Stage Predictions ===")
    print(results.to_string(index=False))

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results.to_csv(Path(output_dir) / "swiss_predictions.csv", index=False)

    # Generate summary
    summary = {
        "tournament": "TI 2026 Group Stage",
        "format": "16-team Swiss to 4 + ER (5 of 10)",
        "n_simulations": n_simulations,
        "generated_at": datetime.now().isoformat(),
        "predictions": results.to_dict("records"),
    }

    with open(Path(output_dir) / "prediction_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")
    return results


def main():
    """Run full pipeline."""
    print("TI 2026 Group Stage Prediction Pipeline")
    print("=" * 60)
    start_time = datetime.now()

    # Iteration 1 default: real OpenDota matchlists → XGBoost
    raw_dir = Path("data/raw")
    has_matchlists = (
        any(raw_dir.glob("*_matchlist.json")) or any(raw_dir.glob("*_matches.json"))
    ) if raw_dir.exists() else False

    if "--legacy" not in sys.argv and has_matchlists:
        from src.pipeline import run_iteration1

        run_iteration1()
        elapsed = datetime.now() - start_time
        print(f"\nPipeline completed in {elapsed.total_seconds():.1f} seconds")
        return

    has_data = any(raw_dir.glob("*.csv")) if raw_dir.exists() else False

    if has_data:
        # Full pipeline with data
        step_collect_data()
        step_process_data()
        features_df = step_build_features()
        models, feature_cols = step_train_models()
        win_matrix = step_generate_predictions(models, feature_cols)
        results = step_simulate_tournament(win_matrix, n_simulations=100000)
    else:
        print("No raw data available. Running with Bradley-Terry baseline...")
        print("(Set STRATZ_TOKEN env var or place OpenDota CSVs in data/raw/)\n")

        teams = get_team_ids()
        n = len(teams)

        # Bradley-Terry model: P(A beats B) = rating_B / (rating_A + rating_B)
        # Rating derived from power ranking: rank 1 → strength 16, rank 16 → strength 1
        matrix = np.ones((n, n)) * 0.5
        strengths = {}
        for team in teams:
            rank = POWER_RANKINGS.get(team, 16)
            strengths[team] = n - rank + 1  # rank 1 → 16, rank 16 → 1

        for i, t1 in enumerate(teams):
            for j, t2 in enumerate(teams):
                if i != j:
                    matrix[i][j] = strengths[t1] / (strengths[t1] + strengths[t2])

        win_matrix = pd.DataFrame(matrix, index=teams, columns=teams)

        # Also generate synthetic match features for calibration
        rng = np.random.default_rng(42)
        synth_matches = []
        for _ in range(2000):
            t1, t2 = rng.choice(teams, 2, replace=False)
            p1 = strengths[t1] / (strengths[t1] + strengths[t2])
            r1_wr = rng.random() < p1
            rank1 = POWER_RANKINGS.get(t1, 16)
            rank2 = POWER_RANKINGS.get(t2, 16)
            synth_matches.append({
                "match_id": len(synth_matches),
                "a_power_ranking": rank1,
                "b_power_ranking": rank2,
                "a_synthetic_wr": 0.65 - (rank1 - 1) * 0.02,
                "b_synthetic_wr": 0.65 - (rank2 - 1) * 0.02,
                "diff_power_ranking": rank1 - rank2,
                "diff_synthetic_wr": (0.65 - (rank1 - 1) * 0.02) - (0.65 - (rank2 - 1) * 0.02),
                "a_elo": 1500 + (16 - rank1) * 30,
                "b_elo": 1500 + (16 - rank2) * 30,
                "diff_elo": (16 - rank1) * 30 - (16 - rank2) * 30,
                "radiant_win": r1_wr,
            })

        features_df = pd.DataFrame(synth_matches)
        features_file = Path("data/features") / "match_features.csv"
        features_file.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(features_file, index=False)

        print("Training on synthetic match data for calibration...")
        models, feature_cols = step_train_models()

        print("\n=== Win Probability Matrix (Bradley-Terry) ===")
        print(win_matrix.round(3).to_string())

        print("\n=== Power Ranking Predictions ===")
        for team in sorted(teams, key=lambda t: -strengths[t]):
            r = POWER_RANKINGS.get(team, 16)
            wr_vs_avg = np.mean([win_matrix.loc[team, t] for t in teams if t != team])
            print(f"  {team:15s} (rank {r:2d})  avg_wr={wr_vs_avg:.1%}")

        results = step_simulate_tournament(win_matrix, n_simulations=100000)

    elapsed = datetime.now() - start_time
    print(f"\nPipeline completed in {elapsed.total_seconds():.1f} seconds")


if __name__ == "__main__":
    main()
