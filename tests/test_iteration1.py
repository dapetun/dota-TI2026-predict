"""Unit tests for iteration-1 ETL, features, weights, and validation splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_collection.match_loader import load_raw_matchlists, summarize_matches
from src.features.match_features import FEATURE_COLUMNS, build_match_feature_matrix
from src.features.sample_weights import compute_sample_weights, exponential_time_weights
from src.models.xgboost_model import leave_one_ti_splits, walk_forward_splits


@pytest.fixture()
def tiny_raw(tmp_path: Path) -> Path:
    """Minimal OpenDota matchlist fixtures."""
    raw = tmp_path / "raw"
    raw.mkdir()
    team_map = {"1": "Team Spirit", "2": "Team Liquid", "3": "Team Falcons"}
    (raw / "team_id_map.json").write_text(json.dumps(team_map), encoding="utf-8")

    # Two tournaments with chronological matches.
    ti10 = []
    t0 = 1_600_000_000
    for i in range(30):
        ti10.append(
            {
                "match_id": 1000 + i,
                "start_time": t0 + i * 86400,
                "radiant_team_id": 1 if i % 2 == 0 else 2,
                "dire_team_id": 2 if i % 2 == 0 else 3,
                "radiant_team_name": "Team Spirit" if i % 2 == 0 else "Team Liquid",
                "dire_team_name": "Team Liquid" if i % 2 == 0 else "Team Falcons",
                "radiant_win": i % 3 != 0,
                "duration": 2000,
                "radiant_score": 20,
                "dire_score": 15,
                "series_id": i // 2,
                "series_type": 1,
                "leagueid": 13256,
            }
        )
    (raw / "TI10_2021_matchlist.json").write_text(json.dumps(ti10), encoding="utf-8")

    ti11 = []
    t1 = t0 + 40 * 86400
    for i in range(25):
        ti11.append(
            {
                "match_id": 2000 + i,
                "start_time": t1 + i * 86400,
                "radiant_team_id": 1,
                "dire_team_id": 3,
                "radiant_team_name": "Team Spirit",
                "dire_team_name": "Team Falcons",
                "radiant_win": i % 2 == 0,
                "duration": 2100,
                "radiant_score": 22,
                "dire_score": 18,
                "series_id": 100 + i,
                "series_type": 1,
                "leagueid": 14268,
            }
        )
    (raw / "TI11_2022_matchlist.json").write_text(json.dumps(ti11), encoding="utf-8")
    return raw


def test_load_raw_matchlists(tiny_raw: Path):
    df = load_raw_matchlists(tiny_raw)
    assert len(df) == 55
    assert set(df["tournament"]) == {"TI10_2021", "TI11_2022"}
    assert df["start_time"].is_monotonic_increasing
    summary = summarize_matches(df)
    assert summary["n_matches"] == 55


def test_features_no_leakage_and_columns(tiny_raw: Path):
    matches = load_raw_matchlists(tiny_raw)
    feats = build_match_feature_matrix(matches, min_games=0)
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
    assert "radiant_win" in feats.columns
    # Absolute first match: both teams at initial Elo (no prior games).
    assert feats.iloc[0]["r_elo"] == pytest.approx(1500.0)
    assert feats.iloc[0]["d_elo"] == pytest.approx(1500.0)
    assert int(feats.iloc[0]["r_gp"]) == 0
    assert int(feats.iloc[0]["d_gp"]) == 0


def test_time_weights_decay():
    now = 1_700_000_000
    times = np.array([now, now - 90 * 86400, now - 180 * 86400], dtype=float)
    w = exponential_time_weights(times, now, half_life_days=90.0)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5, rel=1e-3)
    assert w[2] == pytest.approx(0.25, rel=1e-3)


def test_sample_weights_normalize():
    df = pd.DataFrame(
        {
            "start_time": [100, 200, 300],
            "tier_weight": [2.0, 1.5, 1.0],
        }
    )
    w = compute_sample_weights(df, reference_time=300, half_life_days=90.0)
    assert len(w) == 3
    assert w.mean() == pytest.approx(1.0)


def test_validation_splits(tiny_raw: Path):
    matches = load_raw_matchlists(tiny_raw)
    feats = build_match_feature_matrix(matches, min_games=0)
    wf = walk_forward_splits(feats, n_splits=3)
    assert len(wf) >= 2
    for train_idx, test_idx in wf:
        assert train_idx.max() < test_idx.min()

    loo = leave_one_ti_splits(feats, min_train=20, min_test=10)
    names = [name for name, _, _ in loo]
    assert "TI11_2022" in names
