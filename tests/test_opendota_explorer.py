"""Unit tests for OpenDota explorer → match-detail mapping."""

from __future__ import annotations

from src.data_collection.match_details import detail_to_player_rows
from src.data_collection.opendota_api import explorer_rows_to_match_detail


def test_explorer_rows_to_match_detail_builds_players() -> None:
    rows = [
        {
            "match_id": 1,
            "account_id": 101,
            "player_slot": 0,
            "hero_id": 1,
            "kills": 2,
            "deaths": 1,
            "assists": 3,
            "gold_per_min": 400,
            "xp_per_min": 500,
            "hero_damage": 1000,
            "tower_damage": 200,
            "hero_healing": 50,
            "level": 20,
            "start_time": 100,
            "duration": 2000,
            "radiant_win": True,
            "radiant_team_id": 10,
            "dire_team_id": 20,
            "leagueid": 99,
        },
        {
            "match_id": 1,
            "account_id": 202,
            "player_slot": 128,
            "hero_id": 2,
            "kills": 1,
            "deaths": 2,
            "assists": 1,
            "gold_per_min": 300,
            "xp_per_min": 400,
            "hero_damage": 800,
            "tower_damage": 100,
            "hero_healing": 0,
            "level": 18,
            "start_time": 100,
            "duration": 2000,
            "radiant_win": True,
            "radiant_team_id": 10,
            "dire_team_id": 20,
            "leagueid": 99,
        },
    ]
    detail = explorer_rows_to_match_detail(rows, 1)
    assert detail["match_id"] == 1
    assert detail["_source"] == "opendota_explorer"
    assert len(detail["players"]) == 2
    assert detail["players"][0]["isRadiant"] is True
    assert detail["players"][1]["isRadiant"] is False

    player_rows = detail_to_player_rows(detail, "EWC_2026")
    assert len(player_rows) == 2
    assert player_rows[0]["team_id"] == 10
    assert player_rows[1]["team_id"] == 20
    assert player_rows[0]["team_won"] is True
    assert player_rows[1]["team_won"] is False
