"""Tests for team-state replay and pairwise matrix helpers."""

from __future__ import annotations

import pandas as pd

from src.features.chemistry_features import (
    compose_chemistry_pair_features,
    replay_chemistry_state,
)
from src.features.match_features import compose_pair_features, replay_team_states
from src.features.player_features import compose_player_pair_features, replay_player_states
from src.ti2026.pairwise import latest_lineup_for_team, resolve_opendota_team_ids


def _player_row(match_id, start_time, aid, radiant, team_id, won):
    return {
        "match_id": match_id,
        "start_time": start_time,
        "account_id": aid,
        "is_radiant": radiant,
        "team_id": team_id,
        "kills": 5,
        "deaths": 2,
        "assists": 7,
        "gpm": 400,
        "xpm": 450,
        "hero_damage": 15000,
        "tower_damage": 2000,
        "team_won": won,
        "duration": 2400,
        "is_lan": True,
    }


def test_replay_and_compose_pair():
    matches = pd.DataFrame(
        [
            {
                "match_id": 1,
                "start_time": 1000,
                "radiant_team_id": 10,
                "dire_team_id": 20,
                "radiant_win": True,
                "tier_weight": 1.5,
                "radiant_canonical": "Liquid",
                "dire_canonical": "Spirit",
            },
            {
                "match_id": 2,
                "start_time": 2000,
                "radiant_team_id": 10,
                "dire_team_id": 30,
                "radiant_win": False,
                "tier_weight": 1.5,
                "radiant_canonical": "Liquid",
                "dire_canonical": "Falcons",
            },
        ]
    )
    store = replay_team_states(matches)
    assert store.elo[10] != 1500.0
    row = compose_pair_features(store, 10, 20, as_of_ts=3000, tier_weight=2.0)
    assert "diff_elo" in row
    assert row["r_gp"] == 2.0
    assert row["d_gp"] == 1.0

    mapping = resolve_opendota_team_ids(matches, ["Liquid", "Spirit", "Falcons"])
    assert mapping["Liquid"] == 10
    assert mapping["Falcons"] == 30


def test_player_chem_snapshot_for_pair():
    matches = pd.DataFrame(
        [
            {
                "match_id": 1,
                "start_time": 1000,
                "radiant_team_id": 10,
                "dire_team_id": 20,
                "radiant_win": True,
                "tier_weight": 1.5,
            },
            {
                "match_id": 2,
                "start_time": 2000,
                "radiant_team_id": 10,
                "dire_team_id": 20,
                "radiant_win": True,
                "tier_weight": 1.5,
            },
        ]
    )
    players = pd.DataFrame(
        [
            _player_row(1, 1000, 1, True, 10, True),
            _player_row(1, 1000, 2, True, 10, True),
            _player_row(1, 1000, 11, False, 20, False),
            _player_row(1, 1000, 12, False, 20, False),
            _player_row(2, 2000, 1, True, 10, True),
            _player_row(2, 2000, 2, True, 10, True),
            _player_row(2, 2000, 11, False, 20, False),
            _player_row(2, 2000, 12, False, 20, False),
        ]
    )
    pstates = replay_player_states(matches, players)
    chem = replay_chemistry_state(matches, players)
    r_ids = latest_lineup_for_team(players, 10)
    d_ids = latest_lineup_for_team(players, 20)
    pf = compose_player_pair_features(r_ids, d_ids, pstates)
    cf = compose_chemistry_pair_features(r_ids, d_ids, 10, 20, chem, as_of_ts=3000)
    assert pf["has_player_stats"] == 1.0
    assert pf["r_pl_games"] > 0
    assert cf["r_chem_mean"] > 0
