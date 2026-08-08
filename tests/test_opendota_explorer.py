"""Unit tests for OpenDota explorer → match-detail mapping."""

from __future__ import annotations

from src.data_collection.match_details import detail_to_player_rows
from src.data_collection.opendota_api import (
    explorer_rows_to_match_detail,
    explorer_rows_to_match_details,
)


def _row(match_id: int, account_id: int, player_slot: int, **extra: object) -> dict:
    base = {
        "match_id": match_id,
        "account_id": account_id,
        "player_slot": player_slot,
        "hero_id": 1,
        "kills": 1,
        "deaths": 1,
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
    }
    base.update(extra)
    return base


def test_explorer_rows_to_match_detail_builds_players() -> None:
    rows = [
        _row(1, 101, 0, hero_id=1, kills=2, assists=3, gold_per_min=400, xp_per_min=500,
             hero_damage=1000, tower_damage=200, hero_healing=50, level=20),
        _row(1, 202, 128, hero_id=2),
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


def test_explorer_rows_to_match_details_groups_by_match() -> None:
    rows = [
        _row(1, 101, 0),
        _row(1, 202, 128),
        _row(2, 303, 0, radiant_team_id=30, dire_team_id=40),
        _row(2, 404, 128, radiant_team_id=30, dire_team_id=40),
    ]
    details = explorer_rows_to_match_details(rows)
    assert set(details) == {1, 2}
    assert len(details[1]["players"]) == 2
    assert len(details[2]["players"]) == 2
    assert details[2]["radiant_team_id"] == 30
