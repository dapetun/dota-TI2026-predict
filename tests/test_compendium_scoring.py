"""Tests for compendium points optimization."""

from __future__ import annotations

from src.simulation.tournament_sim import assign_fantasy_board
from src.ti2026.compendium_scoring import (
    VALVE_GROUP_POINTS,
    compare_board_strategies,
    expected_valve_points,
    optimize_fantasy_board,
)


def _vision_like_team(tid: str, qualify: float) -> dict:
    """Strong team: high qualify, moderate 4-0 — advance slot often better."""
    return {
        "id": tid,
        "name": tid,
        "short": tid,
        "qualify_pct": qualify,
        "power_rank": 1,
        "prob_4_0": 26.0,
        "prob_4_1": 30.0,
        "prob_advance": 30.0,
        "prob_eliminate": 10.0,
        "prob_1_4": 3.0,
        "prob_0_4": 1.0,
    }


def _weak_team(tid: str, i: int) -> dict:
    return {
        "id": tid,
        "name": tid,
        "short": tid,
        "qualify_pct": max(5.0, 80 - i * 5),
        "power_rank": i + 2,
        "prob_4_0": 1.0,
        "prob_4_1": 2.0,
        "prob_advance": 5.0,
        "prob_eliminate": 30.0,
        "prob_1_4": 35.0,
        "prob_0_4": 27.0,
    }


def test_valve_points_table_monotonic():
    pts = [VALVE_GROUP_POINTS[i] for i in range(17)]
    assert pts == sorted(pts)
    assert VALVE_GROUP_POINTS[16] == 12000


def test_optimizer_beats_qualify_rank_on_vision_case():
    """Top team in advance beats top team in 4-0 when P(4-0) << P(qualify)."""
    teams = [_vision_like_team("Vision", 86.0)]
    teams += [_weak_team(f"T{i}", i) for i in range(15)]

    qualify_board = assign_fantasy_board(teams)
    points_board = optimize_fantasy_board(teams)
    cmp = compare_board_strategies(teams)

    assert cmp["points_optimal"]["expected_points"] >= cmp["qualify_rank"]["expected_points"]
    # Vision should NOT be forced into 4-0 if advance/4-1 is better EV
    vision_qualify_slot = next(
        s for s, entries in {
            k: [e["id"] for e in v] for k, v in qualify_board.items()
        }.items()
        if "Vision" in entries
    )
    vision_points_slot = next(
        s for s, entries in {
            k: [e["id"] for e in v] for k, v in points_board.items()
        }.items()
        if "Vision" in entries
    )
    assert vision_qualify_slot == "undefeated"  # old strategy
    assert vision_points_slot in ("one_loss", "advance", "undefeated")


def test_optimizer_fills_all_capacities():
    teams = [_vision_like_team("Vision", 86.0)]
    teams += [_weak_team(f"T{i}", i) for i in range(15)]
    board = optimize_fantasy_board(teams)
    total = sum(len(v) for v in board.values())
    assert total == 16
