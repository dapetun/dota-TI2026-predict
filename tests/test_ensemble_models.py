"""Tests for CatBoost pipeline and probability blending."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.match_features import FEATURE_COLUMNS
from src.models.catboost_model import train_catboost_pipeline
from src.models.ensemble import blend_probas, ensemble_predict, train_blend_pipeline
from src.models.xgboost_model import train_xgboost_pipeline


def _synthetic_features(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    t0 = 1_700_000_000
    for i in range(n):
        row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
        score = row["diff_elo"] + 0.3 * row["diff_wr"]
        row.update(
            {
                "match_id": i,
                "start_time": t0 + i * 3600,
                "tournament": "TI11_2022" if i >= n // 2 else "TI10_2021",
                "radiant_win": int(score + rng.normal(0, 0.5) > 0),
                "tier_weight": 1.0,
                "date": pd.Timestamp("2022-01-01") + pd.Timedelta(hours=i),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_blend_probas_weights():
    a = np.array([0.2, 0.8])
    b = np.array([0.4, 0.6])
    out = blend_probas({"xgb": a, "catboost": b}, {"xgb": 0.5, "catboost": 0.5})
    np.testing.assert_allclose(out, [0.3, 0.7])


def test_catboost_and_blend_pipelines_run():
    df = _synthetic_features(90)
    # Tiny trees for speed.
    xgb = train_xgboost_pipeline(
        df,
        feature_cols=FEATURE_COLUMNS,
        params={"n_estimators": 20, "max_depth": 2},
        n_walk_folds=2,
        calibrate_final=False,
    )
    assert xgb.model is not None
    assert len(xgb.walk_forward) >= 1

    cat = train_catboost_pipeline(
        df,
        feature_cols=FEATURE_COLUMNS,
        params={"iterations": 20, "depth": 2},
        n_walk_folds=2,
    )
    assert cat.model is not None

    blend = train_blend_pipeline(
        df,
        feature_cols=FEATURE_COLUMNS,
        n_walk_folds=2,
        xgb_params={"n_estimators": 20, "max_depth": 2},
        cat_params={"iterations": 20, "depth": 2},
    )
    assert set(blend.model.keys()) == {"xgb", "catboost"}
    proba = ensemble_predict(blend.model, df[FEATURE_COLUMNS].iloc[:5])
    assert len(proba) == 5
    assert np.all((proba >= 0) & (proba <= 1))
