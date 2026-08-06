"""Tests for roster chemistry / co-play features."""

from __future__ import annotations

import pandas as pd

from src.features.chemistry_features import (
    CHEMISTRY_FEATURE_COLUMNS,
    build_chemistry_features,
    merge_chemistry_features,
)


def _players_for_match(
    match_id: int,
    start_time: int,
    radiant: list[int],
    dire: list[int],
    radiant_team_id: int = 11,
    dire_team_id: int = 22,
) -> list[dict]:
    rows = []
    for aid in radiant:
        rows.append(
            {
                "match_id": match_id,
                "start_time": start_time,
                "account_id": aid,
                "is_radiant": True,
                "team_id": radiant_team_id,
            }
        )
    for aid in dire:
        rows.append(
            {
                "match_id": match_id,
                "start_time": start_time,
                "account_id": aid,
                "is_radiant": False,
                "team_id": dire_team_id,
            }
        )
    return rows


def test_chemistry_no_leakage_and_growth():
    radiant = [1, 2, 3, 4, 5]
    dire_a = [11, 12, 13, 14, 15]
    dire_b = [11, 12, 13, 14, 99]  # one swap

    matches = pd.DataFrame(
        [
            {
                "match_id": 1,
                "start_time": 1_000,
                "radiant_team_id": 11,
                "dire_team_id": 22,
                "radiant_win": True,
            },
            {
                "match_id": 2,
                "start_time": 2_000,
                "radiant_team_id": 11,
                "dire_team_id": 22,
                "radiant_win": True,
            },
            {
                "match_id": 3,
                "start_time": 3_000,
                "radiant_team_id": 11,
                "dire_team_id": 22,
                "radiant_win": False,
            },
        ]
    )
    players = pd.DataFrame(
        _players_for_match(1, 1_000, radiant, dire_a)
        + _players_for_match(2, 2_000, radiant, dire_a)
        + _players_for_match(3, 3_000, radiant, dire_b)
    )

    chem = build_chemistry_features(matches, players)
    assert list(chem.columns) == ["match_id", *CHEMISTRY_FEATURE_COLUMNS]
    # First meeting: no prior co-play.
    assert chem.loc[0, "r_chem_mean"] == 0.0
    assert chem.loc[0, "r_roster_jaccard"] == 0.0
    # After one shared match, C(5,2)=10 pairs each have count 1.
    assert chem.loc[1, "r_chem_mean"] == 1.0
    assert chem.loc[1, "r_chem_min"] == 1.0
    assert chem.loc[1, "r_roster_jaccard"] == 1.0
    # Third match: radiant still full continuity; dire lost one player.
    assert chem.loc[2, "r_chem_mean"] == 2.0
    assert chem.loc[2, "r_roster_jaccard"] == 1.0
    assert 0.0 < chem.loc[2, "d_roster_jaccard"] < 1.0


def test_merge_chemistry_features():
    base = pd.DataFrame({"match_id": [1, 2], "diff_elo": [0.1, -0.2]})
    chem = pd.DataFrame(
        {
            "match_id": [1],
            **{c: 1.0 for c in CHEMISTRY_FEATURE_COLUMNS},
        }
    )
    out = merge_chemistry_features(base, chem)
    assert out.loc[0, "r_chem_mean"] == 1.0
    assert out.loc[1, "r_chem_mean"] == 0.0
    assert out.loc[1, "has_chemistry"] == 0
