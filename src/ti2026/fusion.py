"""Weighted fusion of model slot probabilities and analyst prior."""

from __future__ import annotations

from src.simulation.tournament_sim import FANTASY_BOARD_SLOTS
from src.ti2026.analyst_consensus import analyst_assignments, slot_vote_counts
from src.ti2026.compendium_scoring import SLOT_PROB_KEYS, slot_probability


def analyst_slot_prior(
    team_id: str,
    slot_key: str,
    assignments: list[dict[str, str]] | None = None,
    *,
    smoothing: float = 0.5,
) -> float:
    """Smoothed P(slot) from analyst vote shares."""
    assignments = assignments or analyst_assignments()
    n = len(assignments)
    if n == 0:
        return 1.0 / len(FANTASY_BOARD_SLOTS)
    votes = slot_vote_counts(assignments, team_id)
    total = sum(votes.values()) + smoothing * len(FANTASY_BOARD_SLOTS)
    return (votes.get(slot_key, 0) + smoothing) / total


def build_analyst_slot_priors(
    team_ids: list[str],
    assignments: list[dict[str, str]] | None = None,
) -> dict[str, dict[str, float]]:
    """team_id -> slot_key -> prior probability."""
    assignments = assignments or analyst_assignments()
    return {
        tid: {
            slot: analyst_slot_prior(tid, slot, assignments)
            for slot in FANTASY_BOARD_SLOTS
        }
        for tid in team_ids
    }


def fuse_slot_probabilities(
    predictions: list[dict],
    model_weight: float = 0.7,
    assignments: list[dict[str, str]] | None = None,
) -> list[dict]:
    """Blend model P(slot) with analyst prior; returns new prediction dicts."""
    assignments = assignments or analyst_assignments()
    analyst_w = max(0.0, min(1.0, 1.0 - model_weight))
    fused: list[dict] = []
    for p in predictions:
        tid = p["id"]
        row = dict(p)
        for slot, prob_key in SLOT_PROB_KEYS.items():
            model_p = slot_probability(p, slot)
            prior_p = analyst_slot_prior(tid, slot, assignments)
            blend_p = model_weight * model_p + analyst_w * prior_p
            pct = round(blend_p * 100.0, 2)
            row[prob_key] = pct
            if "records" in row and isinstance(row["records"], dict):
                label = prob_key.replace("prob_", "").replace("_", "-")
                if prob_key == "prob_advance":
                    label = "advance"
                elif prob_key == "prob_eliminate":
                    label = "eliminate"
                elif prob_key == "prob_4_0":
                    label = "4-0"
                elif prob_key == "prob_4_1":
                    label = "4-1"
                elif prob_key == "prob_1_4":
                    label = "1-4"
                elif prob_key == "prob_0_4":
                    label = "0-4"
                row["records"][label] = pct
            if "board" in row and isinstance(row["board"], dict):
                board_key = slot
                row["board"][board_key] = pct
        fused.append(row)
    return fused


def tune_fusion_weight_loo(
    predictions: list[dict],
    *,
    grid: list[float] | None = None,
) -> tuple[float, float]:
    """Grid search model_weight maximizing E[points] on fused slot probs."""
    from src.ti2026.compendium_scoring import expected_valve_points, optimize_fantasy_board
    from src.ti2026.compendium_scoring import _assignment_from_board

    grid = grid or [0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 1.0]
    best_w = 0.7
    best_pts = -1.0
    for w in grid:
        fused = fuse_slot_probabilities(predictions, model_weight=w)
        board = optimize_fantasy_board(fused)
        assign = _assignment_from_board(board)
        pts = expected_valve_points(assign, fused)
        if pts > best_pts:
            best_pts = pts
            best_w = w
    return best_w, best_pts
