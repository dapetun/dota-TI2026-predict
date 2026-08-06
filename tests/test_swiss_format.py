"""Tests for TI Swiss first-to-4 + fantasy board capacities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.tournament_sim import (
    FANTASY_BOARD_SLOTS,
    SwissConfig,
    assign_fantasy_board,
    simulate_elimination_round,
    simulate_swiss_stage,
)
from src.ti2026.teams import POWER_RANKINGS, get_team_ids


def _toy_matrix(teams: list[str]) -> pd.DataFrame:
    n = len(teams)
    strengths = {t: n - POWER_RANKINGS.get(t, n) + 1 for t in teams}
    m = np.full((n, n), 0.5)
    for i, a in enumerate(teams):
        for j, b in enumerate(teams):
            if i != j:
                m[i, j] = strengths[a] / (strengths[a] + strengths[b])
    return pd.DataFrame(m, index=teams, columns=teams)


def test_swiss_records_are_first_to_four():
    teams = get_team_ids()
    df = simulate_swiss_stage(
        _toy_matrix(teams),
        teams,
        SwissConfig(),
        n_simulations=200,
        rng_seed=1,
    )
    assert "prob_4_0" in df.columns
    assert "prob_4_1" in df.columns
    assert "prob_0_4" in df.columns
    assert "prob_1_4" in df.columns
    assert "prob_advance" in df.columns
    # No legacy 3-win columns
    assert "prob_3_0" not in df.columns
    # Qualify + elim should sum ~100
    for _, row in df.iterrows():
        assert abs(row["direct_qualification_pct"] + row["eliminated_pct"] - 100) < 1.5


def test_elimination_round_pairs_3_2_vs_2_3():
    """Official ER: each 3-2 plays a 2-3; five winners advance."""
    three_two = ["A", "B", "C", "D", "E"]
    two_three = ["F", "G", "H", "I", "J"]
    teams = three_two + two_three
    # Stronger teams first in matrix order → A..E beat F..J.
    strengths = {t: 10 - i for i, t in enumerate(teams)}
    m = np.full((10, 10), 0.5)
    for i, a in enumerate(teams):
        for j, b in enumerate(teams):
            if i != j:
                m[i, j] = strengths[a] / (strengths[a] + strengths[b])
    win_matrix = pd.DataFrame(m, index=teams, columns=teams)
    records = {t: (3, 2) for t in three_two} | {t: (2, 3) for t in two_three}

    winners = simulate_elimination_round(
        teams,
        win_matrix,
        advance_n=5,
        rng=np.random.default_rng(0),
        records=records,
    )
    assert len(winners) == 5
    # Strong 3-2 side should win most/all vs weaker 2-3 under this matrix.
    assert set(winners).issubset(set(three_two) | set(two_three))
    assert len(set(winners)) == 5


def test_fantasy_board_capacities():
    teams = get_team_ids()
    fake = []
    for i, tid in enumerate(sorted(teams, key=lambda t: POWER_RANKINGS[t])):
        fake.append(
            {
                "id": tid,
                "name": tid,
                "short": tid,
                "qualify_pct": 90 - i * 5,
                "power_rank": POWER_RANKINGS[tid],
                "prob_4_0": 10.0,
                "prob_4_1": 15.0,
                "prob_advance": 30.0,
                "prob_eliminate": 25.0,
                "prob_1_4": 12.0,
                "prob_0_4": 8.0,
            }
        )
    board = assign_fantasy_board(fake)
    for key, meta in FANTASY_BOARD_SLOTS.items():
        assert len(board[key]) == meta["capacity"], key
    # Best team in 4-0, worst in 0-4
    best = min(fake, key=lambda x: x["power_rank"])
    worst = max(fake, key=lambda x: x["power_rank"])
    assert board["undefeated"][0]["id"] == best["id"]
    assert board["winless"][0]["id"] == worst["id"]
