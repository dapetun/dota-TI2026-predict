"""Train models on enhanced dataset v2."""
import pickle, json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

BASE_DIR = Path(__file__).parent.parent


def load_data():
    df = pd.read_csv(BASE_DIR / "data" / "features" / "match_features_v2.csv")
    df = df.sort_values("start_time").reset_index(drop=True)
    return df


FEATURE_COLS = [
    "diff_elo", "abs_elo_diff", "elo_prob",
    "diff_wr", "diff_wr5", "diff_streak", "diff_peak",
    "diff_h2h", "diff_gp", "tier",
    "r_elo", "d_elo", "r_wr", "d_wr", "h2h_wr",
]


def time_series_cv(df, n_splits=5):
    """Walk-forward time series CV."""
    indices = np.arange(len(df))
    fold_size = len(df) // (n_splits + 1)
    
    folds = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = fold_size * (i + 2)
        if test_end > len(df):
            test_end = len(df)
        train_idx = indices[:train_end]
        test_idx = indices[train_end:test_end]
        folds.append((train_idx, test_idx))
    return folds


def train_and_evaluate():
    df = load_data()
    X = df[FEATURE_COLS].values
    y = df["radiant_win"].values
    
    print(f"Dataset: {len(df)} matches, {len(FEATURE_COLS)} features")
    print(f"Date: {df['date'].min()} to {df['date'].max()}")
    print(f"Win rate: {y.mean():.3f}")
    
    folds = time_series_cv(df, n_splits=5)
    
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=42,
        ),
        "Logistic": LogisticRegression(max_iter=1000, C=0.1),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        ),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n=== {name} ===")
        fold_metrics = []
        
        for fold_i, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train, y_train)
            
            y_proba = model_clone.predict_proba(X_test)[:, 1]
            y_pred = (y_proba > 0.5).astype(int)
            
            ll = log_loss(y_test, y_proba)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
            
            fold_metrics.append({"ll": ll, "acc": acc, "auc": auc})
            print(f"  Fold {fold_i+1}: LL={ll:.4f} Acc={acc:.3f} AUC={auc:.3f}")
        
        avg_ll = np.mean([m["ll"] for m in fold_metrics])
        avg_acc = np.mean([m["acc"] for m in fold_metrics])
        avg_auc = np.mean([m["auc"] for m in fold_metrics])
        
        print(f"  Average: LL={avg_ll:.4f} Acc={avg_acc:.3f} AUC={avg_auc:.3f}")
        results[name] = {"ll": avg_ll, "acc": avg_acc, "auc": avg_auc}
    
    # Train final XGBoost on all data
    print("\n=== Final XGBoost (all data) ===")
    final_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=42,
    )
    final_model.fit(X, y)
    
    # Feature importance
    fi = dict(zip(FEATURE_COLS, final_model.feature_importances_))
    fi_sorted = sorted(fi.items(), key=lambda x: -x[1])
    print("Feature importance:")
    for feat, imp in fi_sorted:
        print(f"  {feat:20s} {imp:.4f}")
    
    # Save
    out_dir = BASE_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "xgb_v2.pkl", "wb") as f:
        pickle.dump({"model": final_model, "features": FEATURE_COLS, "results": results}, f)
    print(f"\nSaved to {out_dir / 'xgb_v2.pkl'}")
    
    # Compare with v1
    print("\n=== Comparison ===")
    for name, m in results.items():
        print(f"  {name:15s}: LL={m['ll']:.4f} Acc={m['acc']:.3f} AUC={m['auc']:.3f}")
    print(f"  {'Coin flip':15s}: LL=0.6931 Acc=0.500 AUC=0.500")


if __name__ == "__main__":
    train_and_evaluate()
