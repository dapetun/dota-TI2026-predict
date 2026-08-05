"""Model evaluation and backtesting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute core probabilistic and threshold metrics."""
    preds = (y_prob >= 0.5).astype(int)
    metrics = {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }
    metrics["auc_roc"] = (
        float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    )
    return metrics



def walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_class,
    model_params: dict,
    n_splits: int = 5,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """Walk-forward time series validation."""
    n = len(X)
    test_size = int(n * test_ratio)
    fold_size = (n - test_size) // n_splits

    results = []

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_start = train_end
        test_end = min(test_start + test_size, n)

        if test_end <= test_start:
            break

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba > 0.5).astype(int)

        results.append({
            "fold": i + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "log_loss": log_loss(y_test, proba),
            "accuracy": accuracy_score(y_test, preds),
            "brier": brier_score_loss(y_test, proba),
            "auc_roc": roc_auc_score(y_test, proba),
        })

    return pd.DataFrame(results)


def calibration_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Analyze probability calibration."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_true = []
    bin_pred = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            bin_true.append(y_true[mask].mean())
            bin_pred.append(y_prob[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_true.append(0)
            bin_pred.append(bin_centers[i])
            bin_counts.append(0)

    return pd.DataFrame({
        "predicted": bin_pred,
        "actual": bin_true,
        "count": bin_counts,
    })


def plot_calibration(
    y_true: np.ndarray,
    models_proba: Dict[str, np.ndarray],
    output_dir: str = "outputs",
):
    """Plot calibration curves for multiple models."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for name, proba in models_proba.items():
        cal = calibration_analysis(y_true, proba)
        ax.plot(cal["predicted"], cal["actual"], "o-", label=name)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Win Rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output_dir) / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
    output_dir: str = "outputs",
):
    """Plot feature importance from tree-based model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "feature_importance"):
        importances = model.feature_importance()
    else:
        return

    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features")
    ax.grid(True, alpha=0.3, axis="x")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output_dir) / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def backtest_on_tournament(
    models: dict,
    features_df: pd.DataFrame,
    tournament_name: str,
    feature_cols: list = None,
) -> dict:
    """Backtest model predictions on a specific tournament."""
    if features_df.empty:
        return {"error": "No data"}

    from src.models.ensemble import ensemble_predict

    if feature_cols:
        X = features_df[feature_cols].fillna(0)
    else:
        X = features_df.select_dtypes(include=[np.number]).drop(columns=["radiant_win", "match_id"], errors="ignore").fillna(0)

    y = features_df["radiant_win"].astype(int)

    proba = ensemble_predict(models, X, feature_cols=feature_cols)

    return {
        "tournament": tournament_name,
        "n_matches": len(features_df),
        "log_loss": log_loss(y, proba),
        "accuracy": accuracy_score(y, (proba > 0.5).astype(int)),
        "brier": brier_score_loss(y, proba),
    }


def generate_report(
    results: pd.DataFrame,
    backtest_results: list,
    output_dir: str = "outputs",
):
    """Generate evaluation report."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Model Evaluation Report\n",
        "## Model Comparison\n",
        results.to_markdown(index=False),
        "\n## Walk-Forward Validation\n",
    ]

    for bt in backtest_results:
        report_lines.append(f"### {bt.get('tournament', 'Unknown')}")
        report_lines.append(f"- Log Loss: {bt.get('log_loss', 'N/A'):.4f}")
        report_lines.append(f"- Accuracy: {bt.get('accuracy', 'N/A'):.4f}")
        report_lines.append(f"- Brier Score: {bt.get('brier', 'N/A'):.4f}\n")

    with open(Path(output_dir) / "evaluation_report.md", "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to {output_dir}/evaluation_report.md")
