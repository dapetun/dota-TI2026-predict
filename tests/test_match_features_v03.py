"""Tests for uncertainty / Empirical Bayes / Glicko team features."""

from __future__ import annotations

import pandas as pd

from src.features.match_features import (
    FEATURE_COLUMNS,
    build_match_feature_matrix,
    gp_uncertainty,
    shrink_elo,
    team_strength_summary,
    replay_team_states,
)
from datetime import datetime, timezone

from src.features.team_stitching import apply_team_stitch, jaccard, build_team_stitch_map
from src.ti2026.multisource import home_lan_elo_bonus, is_in_patch_window, PATCH_741_START_TS


def test_shrink_and_uncertainty():
    assert shrink_elo(1700, 0) == 1500.0
    assert abs(shrink_elo(1700, 12) - 1600.0) < 1e-6
    assert gp_uncertainty(0) == 1.0
    assert gp_uncertainty(3) < 1.0


def test_feature_matrix_has_v03_cols():
    matches = pd.DataFrame(
        [
            {
                "match_id": i,
                "start_time": 1_000_000 + i * 86400,
                "date": "2025-01-01",
                "tournament": "T",
                "tier": "major",
                "year": 2025,
                "radiant_team_id": 1,
                "dire_team_id": 2,
                "radiant_canonical": "A",
                "dire_canonical": "B",
                "radiant_win": i % 2 == 0,
                "tier_weight": 1.5,
            }
            for i in range(20)
        ]
    )
    feat = build_match_feature_matrix(matches, min_games=3)
    for col in ("min_gp", "r_elo_shrunk", "r_glicko_rd", "r_opp_avg_elo", "r_uncertainty"):
        assert col in feat.columns
        assert col in FEATURE_COLUMNS
    assert len(feat) > 0


def test_home_lan_and_patch():
    assert home_lan_elo_bonus("CN") == 30.0
    assert home_lan_elo_bonus("EU") == 0.0
    expected = int(datetime(2026, 3, 24, tzinfo=timezone.utc).timestamp())
    assert PATCH_741_START_TS == expected
    assert is_in_patch_window(PATCH_741_START_TS)
    assert not is_in_patch_window(PATCH_741_START_TS - 1)


def test_team_stitch_jaccard():
    assert jaccard({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}) == 1.0
    assert jaccard({1, 2, 3, 4, 5}, {1, 2, 3, 4, 9}) >= 0.6
    players = pd.DataFrame(
        [
            {"match_id": 1, "start_time": 100, "team_id": 10, "account_id": a}
            for a in [1, 2, 3, 4, 5]
        ]
        + [
            {"match_id": 2, "start_time": 200, "team_id": 20, "account_id": a}
            for a in [1, 2, 3, 4, 9]
        ]
    )
    stitch = build_team_stitch_map(players, threshold=0.6)
    assert stitch.get(10) == stitch.get(20)


def test_apply_team_stitch_remaps_players():
    matches = pd.DataFrame(
        [
            {
                "match_id": 1,
                "radiant_team_id": 20,
                "dire_team_id": 30,
                "radiant_win": True,
            }
        ]
    )
    players = pd.DataFrame(
        [
            {"match_id": 1, "team_id": 20, "account_id": a}
            for a in [1, 2, 3, 4, 5]
        ]
    )
    stitch = {20: 10, 10: 10}
    m2, p2 = apply_team_stitch(matches, stitch, players)
    assert int(m2.loc[0, "radiant_team_id"]) == 10
    assert set(p2["team_id"].astype(int)) == {10}


def test_strength_summary():
    matches = pd.DataFrame(
        [
            {
                "match_id": i,
                "start_time": 1_000_000 + i * 1000,
                "radiant_team_id": 1,
                "dire_team_id": 2,
                "radiant_win": True,
                "tier_weight": 1.5,
            }
            for i in range(10)
        ]
    )
    store = replay_team_states(matches)
    s = team_strength_summary(store, 1, int(matches["start_time"].max()))
    assert "mu" in s and "sigma" in s and s["gp"] == 10
