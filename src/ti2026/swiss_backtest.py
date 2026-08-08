"""Swiss MC backtest vs Liquipedia ground truth + expert board scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.tournament_sim import (
    FANTASY_BOARD_SLOTS,
    SwissConfig,
    simulate_swiss_stage,
)
from src.ti2026.compendium_scoring import (
    VALVE_GROUP_POINTS,
    _assignment_from_board,
    optimize_fantasy_board,
)
from src.ti2026.expert_history import (
    load_swiss_ground_truth,
    score_all_experts,
    score_expert_board,
)
from src.ti2026.teams import normalize_team_name


def ground_truth_assignment(ti_key: str, gt_data: dict | None = None) -> dict[str, str]:
    """Flatten GT slots for one TI into team -> slot_key."""
    gt_data = gt_data or load_swiss_ground_truth()
    tourn = (gt_data.get("tournaments") or {}).get(ti_key)
    if not tourn:
        raise KeyError(f"Unknown TI key: {ti_key}")
    out: dict[str, str] = {}
    for slot, teams in (tourn.get("slots") or {}).items():
        for name in teams:
            out[normalize_team_name(name)] = slot
            out[name] = slot
    return out


def bradley_terry_matrix(
    team_ids: list[str],
    strengths: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Toy BT win matrix from positive strengths (default: equal)."""
    n = len(team_ids)
    strengths = strengths or {t: float(n - i) for i, t in enumerate(team_ids)}
    mat = np.full((n, n), 0.5)
    for i, a in enumerate(team_ids):
        for j, b in enumerate(team_ids):
            if i == j:
                continue
            sa = max(1e-6, float(strengths.get(a, 1.0)))
            sb = max(1e-6, float(strengths.get(b, 1.0)))
            mat[i, j] = sa / (sa + sb)
    return pd.DataFrame(mat, index=team_ids, columns=team_ids)


def results_to_predictions(results: pd.DataFrame) -> list[dict]:
    """Convert simulate_swiss_stage DataFrame into prediction dicts."""
    preds: list[dict] = []
    for _, row in results.iterrows():
        tid = str(row["team"])
        qualify = float(row.get("direct_qualification_pct", row.get("qualify_pct", 0.0)))
        preds.append(
            {
                "id": tid,
                "name": tid,
                "short": tid,
                "qualify_pct": qualify,
                "power_rank": 99,
                "prob_4_0": float(row.get("prob_4_0", 0.0)),
                "prob_4_1": float(row.get("prob_4_1", 0.0)),
                "prob_advance": float(row.get("prob_advance", 0.0)),
                "prob_eliminate": float(row.get("prob_eliminate", 0.0)),
                "prob_1_4": float(row.get("prob_1_4", 0.0)),
                "prob_0_4": float(row.get("prob_0_4", 0.0)),
            }
        )
    return preds


def score_assignment_vs_gt(
    assignment: dict[str, str],
    ground_truth: dict[str, str],
) -> dict[str, float]:
    """Exact slot hits and Valve points vs GT."""
    correct = 0
    scored = 0
    for tid, slot in assignment.items():
        gt = ground_truth.get(tid) or ground_truth.get(normalize_team_name(tid))
        if gt is None:
            continue
        scored += 1
        if gt == slot:
            correct += 1
    return {
        "correct_slots": float(correct),
        "scored_slots": float(scored),
        "hit_rate": (correct / scored) if scored else 0.0,
        "valve_points": float(VALVE_GROUP_POINTS.get(correct, 0)),
    }


def run_swiss_backtest(
    ti_key: str,
    win_matrix: pd.DataFrame | None = None,
    *,
    n_simulations: int = 2000,
    rng_seed: int = 42,
    strengths: dict[str, float] | None = None,
    team_strengths: dict[str, dict] | None = None,
    sample_uncertainty: bool = False,
) -> dict:
    """MC → points-optimal board → score vs Liquipedia GT."""
    gt_data = load_swiss_ground_truth()
    gt = ground_truth_assignment(ti_key, gt_data)
    # Prefer GT team set order
    tourn = gt_data["tournaments"][ti_key]
    team_ids: list[str] = []
    seen: set[str] = set()
    for slot in FANTASY_BOARD_SLOTS:
        for name in tourn["slots"].get(slot, []):
            tid = normalize_team_name(name)
            if tid not in seen:
                seen.add(tid)
                team_ids.append(tid)

    if win_matrix is None:
        win_matrix = bradley_terry_matrix(team_ids, strengths)

    # Align matrix index
    win_matrix = win_matrix.reindex(index=team_ids, columns=team_ids).fillna(0.5)
    for t in team_ids:
        win_matrix.loc[t, t] = 0.5

    results = simulate_swiss_stage(
        win_matrix,
        team_ids,
        SwissConfig(),
        n_simulations=n_simulations,
        rng_seed=rng_seed,
        team_strengths=team_strengths,
        sample_uncertainty=sample_uncertainty,
    )
    predictions = results_to_predictions(results)
    board = optimize_fantasy_board(predictions)
    assignment = _assignment_from_board(board)
    scores = score_assignment_vs_gt(assignment, gt)
    return {
        "ti": ti_key,
        "format": tourn.get("format"),
        "n_simulations": n_simulations,
        "assignment": assignment,
        "ground_truth": {k: gt[k] for k in team_ids if k in gt},
        "scores": scores,
        "liquipedia": tourn.get("liquipedia"),
        "note": tourn.get("note"),
    }


def backtest_report(
    *,
    n_simulations: int = 2000,
) -> dict:
    """Run TI13/TI14 toy BT backtests + expert scoring summary."""
    ti_results = {}
    for ti in ("TI14", "TI13"):
        ti_results[ti] = run_swiss_backtest(ti, n_simulations=n_simulations)
    experts = score_all_experts()
    return {
        "disclaimer": (
            "Research backtest only. Market/odds signals are anonymous; "
            "author does not endorse gambling."
        ),
        "swiss_mc": ti_results,
        "experts": experts,
    }
