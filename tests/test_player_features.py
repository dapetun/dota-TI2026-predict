"""Tests for match-detail ETL and player feature leakage safety."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_collection.match_details import detail_to_player_rows, load_player_matches
from src.features.player_features import (
    PLAYER_FEATURE_COLUMNS,
    build_player_match_features,
    merge_team_and_player_features,
)


def _fake_detail(match_id: int, start_time: int, radiant_win: bool = True) -> dict:
    players = []
    for i in range(10):
        is_radiant = i < 5
        players.append(
            {
                "account_id": 1000 + i + match_id % 7,
                "hero_id": 1 + i,
                "isRadiant": is_radiant,
                "kills": 5 + i,
                "deaths": 2,
                "assists": 7,
                "gold_per_min": 400 + i * 10,
                "xp_per_min": 450 + i * 10,
                "hero_damage": 15000,
                "tower_damage": 2000,
                "healing": 500,
                "level": 20,
                "lane_role": (i % 5) + 1,
            }
        )
    return {
        "match_id": match_id,
        "start_time": start_time,
        "duration": 2400,
        "radiant_win": radiant_win,
        "radiant_team_id": 11,
        "dire_team_id": 22,
        "leagueid": 18324,
        "players": players,
    }


def test_detail_to_player_rows_basic():
    rows = detail_to_player_rows(_fake_detail(1, 1_700_000_000), "TI14_2025")
    assert len(rows) == 10
    assert {r["is_radiant"] for r in rows} == {True, False}
    assert all(r["tournament"] == "TI14_2025" for r in rows)


def test_load_player_matches_from_raw(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    store = {
        "1": _fake_detail(1, 1_700_000_000),
        "2": _fake_detail(2, 1_700_100_000, radiant_win=False),
    }
    (raw / "match_details.json").write_text(json.dumps(store), encoding="utf-8")
    df = load_player_matches(raw)
    assert len(df) == 20
    assert df["account_id"].nunique() > 1


def test_player_features_no_leakage_and_columns():
    matches = pd.DataFrame(
        [
            {
                "match_id": 1,
                "start_time": 100,
                "radiant_team_id": 11,
                "dire_team_id": 22,
                "radiant_win": True,
            },
            {
                "match_id": 2,
                "start_time": 200,
                "radiant_team_id": 11,
                "dire_team_id": 22,
                "radiant_win": False,
            },
        ]
    )
    players = pd.DataFrame(detail_to_player_rows(_fake_detail(1, 100), "TI14_2025"))
    players = pd.concat(
        [players, pd.DataFrame(detail_to_player_rows(_fake_detail(2, 200), "TI14_2025"))],
        ignore_index=True,
    )
    feats = build_player_match_features(matches, players)
    for col in PLAYER_FEATURE_COLUMNS:
        assert col in feats.columns
    # First match: no prior player history → games ~ 0
    assert feats.iloc[0]["r_pl_games"] == pytest.approx(0.0)
    assert feats.iloc[0]["has_player_stats"] == 1
    # Second match should see prior games for overlapping accounts
    assert feats.iloc[1]["r_pl_games"] > 0


def test_merge_team_and_player_features():
    team = pd.DataFrame(
        {
            "match_id": [1, 2],
            "diff_elo": [10.0, -5.0],
            "radiant_win": [1, 0],
        }
    )
    players = pd.DataFrame(
        {
            "match_id": [1, 2],
            "diff_pl_kda": [0.2, -0.1],
            "has_player_stats": [1, 1],
            **{c: 0.0 for c in PLAYER_FEATURE_COLUMNS if c not in ("diff_pl_kda", "has_player_stats")},
        }
    )
    # fill required wr cols
    for c in ("r_pl_wr", "d_pl_wr", "r_pl_lan_wr", "d_pl_lan_wr"):
        players[c] = 0.5
    merged = merge_team_and_player_features(team, players)
    assert "diff_pl_kda" in merged.columns
    assert len(merged) == 2
