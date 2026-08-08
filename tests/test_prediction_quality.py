"""Tests for market prior, dual Elo/MoV, hero soft prior, Swiss uncertainty, backtest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.hero_meta import load_hero_meta
from src.features.hero_soft_prior import apply_soft_prior, apply_soft_prior_matrix, team_meta_fit
from src.features.match_features import (
    margin_of_victory_k_scale,
    replay_team_states,
)
from src.features.player_signatures import (
    build_player_signature,
    signatures_from_fixture_rows,
)
from src.simulation.tournament_sim import (
    sample_strength_adjusted_matrix,
    simulate_swiss_stage,
)
from src.ti2026.expert_history import score_all_experts, score_expert_board
from src.ti2026.fusion import fuse_slot_probabilities
from src.ti2026.multisource import (
    load_market_priors,
    market_slot_prior,
    ranking_slot_prior,
)
from src.ti2026.swiss_backtest import run_swiss_backtest
from src.ti2026.teams import get_team_ids

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_META = ROOT / "tests" / "fixtures" / "hero" / "hero_meta.json"
FIXTURE_ROWS = ROOT / "tests" / "fixtures" / "hero" / "player_hero_rows.json"


def test_market_prior_loads_and_sums_to_one():
    data = load_market_priors()
    # Live Polymarket-derived slots (or seed fallback if JSON empty / seed flag).
    if data.get("seed_from_ranking") or not data.get("teams"):
        assert data.get("seeded_from_ranking") is True
        assert data.get("is_real_market") is False
        assert data.get("source") == "seeded_from_power_rankings"
    else:
        assert data.get("seeded_from_ranking") is False
        assert data.get("is_real_market") is True
        assert "polymarket" in str(data.get("source", "")).lower()
        assert len(data.get("teams") or {}) == 16
    tid = get_team_ids()[0]
    total = sum(
        market_slot_prior(tid, s, data) or 0.0
        for s in (
            "undefeated",
            "one_loss",
            "advance",
            "eliminate",
            "one_win",
            "winless",
        )
    )
    assert abs(total - 1.0) < 1e-6
    assert "bookmaker" not in json.dumps(data).lower()
    assert ranking_slot_prior(tid, "advance") > 0


def test_fusion_includes_market_and_ranking():
    preds = [
        {
            "id": tid,
            "name": tid,
            "prob_4_0": 10.0,
            "prob_4_1": 15.0,
            "prob_advance": 30.0,
            "prob_eliminate": 25.0,
            "prob_1_4": 15.0,
            "prob_0_4": 5.0,
        }
        for tid in get_team_ids()
    ]
    fused = fuse_slot_probabilities(
        preds,
        model_weight=0.5,
        market_weight=0.2,
        ranking_weight=0.1,
        expert_weight=0.0,
    )
    assert len(fused) == 16
    assert fused[0]["prob_advance"] != preds[0]["prob_advance"]


def test_fuse_soft_weights_need_not_sum_to_one():
    """Independent soft weights renormalize at blend (battlepass-style)."""
    from src.ti2026.teams import get_team_ids

    preds = [
        {
            "id": tid,
            "name": tid,
            "prob_4_0": 10.0,
            "prob_4_1": 15.0,
            "prob_advance": 30.0,
            "prob_eliminate": 25.0,
            "prob_1_4": 15.0,
            "prob_0_4": 5.0,
        }
        for tid in get_team_ids()
    ]
    # Sum of raw weights = 2.0 — must still produce valid slot probs.
    a = fuse_slot_probabilities(
        preds,
        model_weight=1.0,
        analyst_weight=1.0,
        market_weight=0.0,
        ranking_weight=0.0,
        expert_weight=0.0,
    )
    b = fuse_slot_probabilities(
        preds,
        model_weight=0.5,
        analyst_weight=0.5,
        market_weight=0.0,
        ranking_weight=0.0,
        expert_weight=0.0,
    )
    assert abs(a[0]["prob_advance"] - b[0]["prob_advance"]) < 1e-6


def test_mov_k_scale_increases_with_margin():
    close = margin_of_victory_k_scale(20, 18)
    blowout = margin_of_victory_k_scale(40, 5)
    assert blowout > close
    assert 0.75 <= close <= 1.5
    assert blowout <= 1.5
    dur = margin_of_victory_k_scale(None, None, duration=1800)
    assert dur >= 1.0


def test_dual_elo_online_vs_lan_updates():
    matches = pd.DataFrame(
        [
            {
                "match_id": 1,
                "start_time": 1_000_000,
                "radiant_team_id": 1,
                "dire_team_id": 2,
                "radiant_win": True,
                "tier_weight": 1.0,
                "is_lan": False,
                "radiant_score": 30,
                "dire_score": 10,
                "duration": 2000,
            },
            {
                "match_id": 2,
                "start_time": 1_100_000,
                "radiant_team_id": 1,
                "dire_team_id": 2,
                "radiant_win": True,
                "tier_weight": 1.5,
                "is_lan": True,
                "radiant_score": 25,
                "dire_score": 20,
                "duration": 2500,
            },
        ]
    )
    store = replay_team_states(matches)
    assert store.elo[1] > 1500
    assert store.elo_online[1] > 1500
    assert store.elo_lan[1] > 1500
    assert store.elo_online[1] != store.elo_lan[1]


def test_hero_soft_prior_logit_shift():
    meta = load_hero_meta(FIXTURE_META)
    rows = json.loads(FIXTURE_ROWS.read_text(encoding="utf-8"))
    sig = signatures_from_fixture_rows(rows["players"], meta)
    fit_a = team_meta_fit([1001], sig)
    fit_b = team_meta_fit([1002], sig)
    assert 0.3 < fit_a < 0.8
    p0 = 0.5
    p1 = apply_soft_prior(p0, fit_a, fit_b, lambda_=0.3)
    assert abs(p1 - p0) < 0.05
    assert p1 != p0

    mat = pd.DataFrame([[0.5, 0.55], [0.45, 0.5]], index=["A", "B"], columns=["A", "B"])
    out = apply_soft_prior_matrix(
        mat, {"A": [1001], "B": [1002]}, signatures=sig, lambda_=0.3
    )
    assert abs(float(out.loc["A", "B"]) - 0.55) < 0.05


def test_swiss_uncertainty_changes_distribution():
    teams = [f"T{i}" for i in range(16)]
    n = len(teams)
    strengths_bt = {t: float(n - i) for i, t in enumerate(teams)}
    mat = np.full((n, n), 0.5)
    for i, a in enumerate(teams):
        for j, b in enumerate(teams):
            if i != j:
                mat[i, j] = strengths_bt[a] / (strengths_bt[a] + strengths_bt[b])
    win_matrix = pd.DataFrame(mat, index=teams, columns=teams)
    team_strengths = {
        t: {"mu": 1400 + 10 * (n - i), "sigma": 80.0} for i, t in enumerate(teams)
    }
    fixed = simulate_swiss_stage(
        win_matrix, teams, n_simulations=400, rng_seed=1, sample_uncertainty=False
    )
    noisy = simulate_swiss_stage(
        win_matrix,
        teams,
        n_simulations=400,
        rng_seed=1,
        team_strengths=team_strengths,
        sample_uncertainty=True,
    )
    delta = (fixed["prob_4_0"] - noisy["prob_4_0"]).abs().sum()
    assert delta > 0

    adj = sample_strength_adjusted_matrix(
        win_matrix, team_strengths, np.random.default_rng(0)
    )
    assert not np.allclose(adj.values, win_matrix.values)


def test_swiss_backtest_ti14_runs():
    report = run_swiss_backtest("TI14", n_simulations=300, rng_seed=7)
    assert report["scores"]["scored_slots"] > 0
    assert "assignment" in report
    experts = score_all_experts()
    assert any(e.get("ti") == "TI14" for e in experts)
    gt = report["ground_truth"]
    partial = score_expert_board({"Spirit": "undefeated"}, gt, partial=True)
    assert partial["scored"] == 1.0
    assert partial["correct"] == 0.0


def test_build_player_signature_schema():
    meta = load_hero_meta(FIXTURE_META)
    sig = build_player_signature(
        42,
        [{"hero_id": 8, "games": 20, "win": 12}],
        meta,
    )
    assert sig["account_id"] == 42
    assert sig["heroes"][0]["hero_id"] == 8
    assert "score" in sig["heroes"][0]
