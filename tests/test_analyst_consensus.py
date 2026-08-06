"""Tests for analyst consensus and compendium scoring integration."""

from __future__ import annotations

from src.ti2026.analyst_consensus import (
    analyst_agreement,
    build_consensus_board,
    consensus_summary,
    load_analyst_picks,
    score_board_assignment,
)
from src.ti2026.compendium_scoring import compare_board_strategies
from src.ti2026.teams import get_team_ids


def _fake_predictions() -> list[dict]:
    teams = get_team_ids()
    return [
        {
            "id": tid,
            "name": tid,
            "short": tid,
            "qualify_pct": 50.0,
            "prob_4_0": 5.0,
            "prob_4_1": 10.0,
            "prob_advance": 30.0,
            "prob_eliminate": 25.0,
            "prob_1_4": 15.0,
            "prob_0_4": 5.0,
        }
        for tid in teams
    ]


def test_load_analyst_picks_has_eleven_full_grids():
    data = load_analyst_picks()
    assert len(data["analysts"]) == 10
    for row in data["analysts"]:
        assert len(row["board"]) == 16


def test_consensus_board_capacities():
    preds = _fake_predictions()
    board = build_consensus_board(preds)
    from src.simulation.tournament_sim import FANTASY_BOARD_SLOTS

    for slot, meta in FANTASY_BOARD_SLOTS.items():
        assert len(board[slot]) == meta["capacity"]


def test_analyst_agreement_vision_4_0():
    n = analyst_agreement("Vision", "undefeated")
    assert n >= 5


def test_compare_board_strategies_includes_analyst():
    preds = _fake_predictions()
    consensus = build_consensus_board(preds)
    cmp = compare_board_strategies(preds, extra_boards={"analyst_consensus": consensus})
    assert "analyst_consensus" in cmp
    assert cmp["analyst_consensus"]["expected_points"] > 0


def test_consensus_summary_scores():
    preds = _fake_predictions()
    summary = consensus_summary(preds)
    assert summary["n_analysts"] == 10
    assert summary["expected_points"] > 0
    scores = score_board_assignment(summary["assignment"], preds)
    assert scores["expected_points"] == summary["expected_points"]
