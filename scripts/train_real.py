"""Train XGBoost model on real match data with Leave-One-TI-Out CV."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent


def load_data():
    df = pd.read_csv(BASE_DIR / "data/features/match_features_real.csv")
    print(f"Loaded {len(df)} matches")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Teams: {df['radiant_team'].nunique()}")
    print(f"Tournaments: {df['tournament'].nunique()}")
    return df


FEATURE_COLS = [
    "r_elo", "d_elo", "diff_elo",
    "r_wr", "d_wr", "diff_wr",
    "r_map_wr", "d_map_wr", "diff_map_wr",
    "r_form", "d_form", "diff_form",
    "r_h2h_wr", "d_h2h_wr", "diff_h2h",
    "r_games_played", "d_games_played",
    "diff_placement",
    "tier_weight",
]


def train_xgboost(df):
    X = df[FEATURE_COLS].fillna(0)
    y = df["radiant_win"].astype(int)

    print(f"\nFeatures: {len(FEATURE_COLS)}")
    print(f"Target distribution: {y.mean():.3f} (radiant win rate)")

    # Walk-forward CV (time-based split)
    n = len(df)
    n_splits = 5
    test_size = n // (n_splits + 1)

    results = []
    for i in range(n_splits):
        train_end = test_size * (i + 1)
        test_end = min(train_end + test_size, n)

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        if len(X_test) == 0:
            break

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        ll = log_loss(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.5

        train_dates = df.iloc[:train_end]["date"].tolist()
        test_dates = df.iloc[train_end:test_end]["date"].tolist()
        train_tourns = df.iloc[:train_end]["tournament"].unique()
        test_tourns = df.iloc[train_end:test_end]["tournament"].unique()

        print(f"\n  Fold {i+1}:")
        print(f"    Train: {len(X_train)} matches ({train_tourns[0] if len(train_tourns) else '?'} ... {train_tourns[-1] if len(train_tourns) else '?'})")
        print(f"    Test:  {len(X_test)} matches ({test_tourns[0] if len(test_tourns) else '?'} ... {test_tourns[-1] if len(test_tourns) else '?'})")
        print(f"    Log-loss: {ll:.4f}, Accuracy: {acc:.3f}, AUC: {auc:.3f}")

        results.append({"fold": i+1, "logloss": ll, "accuracy": acc, "auc": auc})

        # Feature importance for last fold
        if i == n_splits - 1:
            imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
            print(f"\n  Feature importance (last fold):")
            for feat, score in imp.head(10).items():
                print(f"    {feat:25s}: {score:.4f}")

    # Overall
    avg_results = pd.DataFrame(results)
    print(f"\n{'='*50}")
    print(f"  Average: Log-loss={avg_results['logloss'].mean():.4f} (±{avg_results['logloss'].std():.4f})")
    print(f"           Accuracy={avg_results['accuracy'].mean():.3f} (±{avg_results['accuracy'].std():.3f})")
    print(f"           AUC={avg_results['auc'].mean():.3f} (±{avg_results['auc'].std():.3f})")

    # Train final model on all data
    print(f"\n  Training final model on all {len(X)} matches...")
    final_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )
    final_model.fit(X, y, verbose=False)

    # Save
    import pickle
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "xgb_real.pkl", "wb") as f:
        pickle.dump(final_model, f)
    print(f"  Model saved to outputs/xgb_real.pkl")

    return final_model, avg_results


if __name__ == "__main__":
    df = load_data()
    model, results = train_xgboost(df)
