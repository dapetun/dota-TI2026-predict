"""Weighted fusion of model, analyst, market, and ranking/expert priors.

Default model_weight=0.65 balances LOO-tuned model with Sports.ru consensus.
Market / ranking / expert-history are optional research Bayesian sources.

Disclaimer: market implied probs are anonymous research signals only.
The author does not support or endorse gambling operators.
"""

from __future__ import annotations

from src.simulation.tournament_sim import FANTASY_BOARD_SLOTS
from src.ti2026.analyst_consensus import analyst_assignments, slot_vote_counts
from src.ti2026.compendium_scoring import SLOT_PROB_KEYS, slot_probability
from src.ti2026.expert_history import historical_expert_slot_prior
from src.ti2026.multisource import (
    load_market_priors,
    market_slot_prior,
    ranking_slot_prior,
)

# Sensible production default (model lean, analyst regularizes extremes).
# Soft weights are independent (need not sum to 1); fuse renormalizes at blend time.
DEFAULT_MODEL_WEIGHT: float = 0.65
DEFAULT_ANALYST_WEIGHT: float = 0.20
DEFAULT_MARKET_WEIGHT: float = 0.10
DEFAULT_RANKING_WEIGHT: float = 0.05
DEFAULT_EXPERT_WEIGHT: float = 0.0

# Precomputed UI scenarios (raw soft weights; renormalized in fuse).
FUSION_WEIGHT_SCENARIOS: dict[str, dict[str, float]] = {
    "model_heavy": {
        "model_weight": 0.80,
        "analyst_weight": 0.10,
        "market_weight": 0.05,
        "ranking_weight": 0.05,
        "expert_weight": 0.0,
    },
    "balanced": {
        "model_weight": 0.55,
        "analyst_weight": 0.15,
        "market_weight": 0.15,
        "ranking_weight": 0.10,
        "expert_weight": 0.05,
    },
    "market_lean": {
        "model_weight": 0.45,
        "analyst_weight": 0.10,
        "market_weight": 0.30,
        "ranking_weight": 0.10,
        "expert_weight": 0.05,
    },
    "analyst_lean": {
        "model_weight": 0.40,
        "analyst_weight": 0.45,
        "market_weight": 0.10,
        "ranking_weight": 0.05,
        "expert_weight": 0.0,
    },
}


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


def _renormalize_weights(
    model_w: float,
    analyst_w: float,
    market_w: float,
    ranking_w: float,
    expert_w: float,
) -> tuple[float, float, float, float, float]:
    ws = [max(0.0, float(x)) for x in (model_w, analyst_w, market_w, ranking_w, expert_w)]
    total = sum(ws)
    if total <= 0:
        return 1.0, 0.0, 0.0, 0.0, 0.0
    return tuple(w / total for w in ws)  # type: ignore[return-value]


def _record_label(prob_key: str) -> str:
    mapping = {
        "prob_advance": "advance",
        "prob_eliminate": "eliminate",
        "prob_4_0": "4-0",
        "prob_4_1": "4-1",
        "prob_1_4": "1-4",
        "prob_0_4": "0-4",
    }
    if prob_key in mapping:
        return mapping[prob_key]
    return prob_key.replace("prob_", "").replace("_", "-")


def fuse_slot_probabilities(
    predictions: list[dict],
    model_weight: float = DEFAULT_MODEL_WEIGHT,
    assignments: list[dict[str, str]] | None = None,
    *,
    analyst_weight: float | None = None,
    market_weight: float = DEFAULT_MARKET_WEIGHT,
    ranking_weight: float = DEFAULT_RANKING_WEIGHT,
    expert_weight: float = DEFAULT_EXPERT_WEIGHT,
    market_priors: dict | None = None,
    use_expert_history: bool = False,
) -> list[dict]:
    """Blend model P(slot) with analyst / market / ranking / expert-history.

    Soft weights are independent (need not sum to 1.0), like battlepass.ru:
    at blend time they are renormalized over active sources.

    If ``analyst_weight`` is ``None``, use the legacy residual
    ``max(0, 1 - model - market - ranking - expert)`` so existing callers
    (production export, LOO tune) keep the same mix. Pass an explicit value
    for battlepass-style independent soft weights.
    Missing market values fall back to ranking prior for that team/slot.
    """
    assignments = assignments or analyst_assignments()
    market_priors = market_priors if market_priors is not None else load_market_priors()

    model_w = max(0.0, float(model_weight))
    market_w = max(0.0, float(market_weight))
    ranking_w = max(0.0, float(ranking_weight))
    expert_w = max(0.0, float(expert_weight)) if use_expert_history else 0.0
    if analyst_weight is None:
        analyst_w = max(0.0, 1.0 - model_w - market_w - ranking_w - expert_w)
    else:
        analyst_w = max(0.0, float(analyst_weight))
    model_w, analyst_w, market_w, ranking_w, expert_w = _renormalize_weights(
        model_w, analyst_w, market_w, ranking_w, expert_w
    )

    fused: list[dict] = []
    for p in predictions:
        tid = p["id"]
        row = dict(p)
        for slot, prob_key in SLOT_PROB_KEYS.items():
            model_p = slot_probability(p, slot)
            prior_p = analyst_slot_prior(tid, slot, assignments)
            rank_p = ranking_slot_prior(tid, slot)
            market_p = market_slot_prior(tid, slot, market_priors)
            if market_p is None:
                market_p = rank_p
            expert_p = (
                historical_expert_slot_prior(tid, slot)
                if expert_w > 0
                else rank_p
            )
            blend_p = (
                model_w * model_p
                + analyst_w * prior_p
                + market_w * market_p
                + ranking_w * rank_p
                + expert_w * expert_p
            )
            pct = round(blend_p * 100.0, 2)
            row[prob_key] = pct
            if "records" in row and isinstance(row["records"], dict):
                row["records"][_record_label(prob_key)] = pct
            if "board" in row and isinstance(row["board"], dict):
                row["board"][slot] = pct
        fused.append(row)
    return fused


def fuse_weight_scenarios(
    predictions: list[dict],
    *,
    assignments: list[dict[str, str]] | None = None,
    scenarios: dict[str, dict[str, float]] | None = None,
) -> dict[str, list[dict]]:
    """Precompute fused prediction boards for UI weight scenarios."""
    scenarios = scenarios or FUSION_WEIGHT_SCENARIOS
    return {
        name: fuse_slot_probabilities(
            predictions,
            model_weight=float(w.get("model_weight", DEFAULT_MODEL_WEIGHT)),
            assignments=assignments,
            analyst_weight=float(
                w["analyst_weight"]
                if "analyst_weight" in w
                else DEFAULT_ANALYST_WEIGHT
            ),
            market_weight=float(w.get("market_weight", 0.0)),
            ranking_weight=float(w.get("ranking_weight", 0.0)),
            expert_weight=float(w.get("expert_weight", 0.0)),
            use_expert_history=float(w.get("expert_weight", 0.0)) > 0,
        )
        for name, w in scenarios.items()
    }


def tune_fusion_weight_loo(
    predictions: list[dict],
    *,
    grid: list[float] | None = None,
    min_model_weight: float = 0.25,
) -> tuple[float, float]:
    """In-sample grid search of ``model_weight`` maximizing E[points].

    Not true leave-one-out CV: the grid is scored on the same fused
    predictions passed in (optimistic / diagnostic only).

    ``0.0`` is excluded by default: pure-analyst boards inflate E[points]
    in-sample and would silently drop the ML model from fusion.
    """
    from src.ti2026.compendium_scoring import expected_valve_points, optimize_fantasy_board
    from src.ti2026.compendium_scoring import _assignment_from_board

    grid = grid or [0.25, 0.5, 0.65, 0.75, 0.85, 1.0]
    grid = [float(w) for w in grid if float(w) >= float(min_model_weight)]
    if not grid:
        return DEFAULT_MODEL_WEIGHT, -1.0
    best_w = DEFAULT_MODEL_WEIGHT
    best_pts = -1.0
    for w in grid:
        fused = fuse_slot_probabilities(
            predictions,
            model_weight=w,
            market_weight=0.0,
            ranking_weight=0.0,
            expert_weight=0.0,
        )
        board = optimize_fantasy_board(fused)
        assign = _assignment_from_board(board)
        pts = expected_valve_points(assign, fused)
        if pts > best_pts:
            best_pts = pts
            best_w = w
    return best_w, best_pts


def resolve_production_fusion_weight(
    tuned_weight: float | None = None,
    *,
    production_default: float = DEFAULT_MODEL_WEIGHT,
) -> float:
    """Production fusion uses the documented default, not in-sample tune.

    Tuned weights are optimistic on the same predictions and must not zero
    out the model; keep them as diagnostics only.
    """
    _ = tuned_weight  # diagnostic input; intentionally unused for production
    return float(production_default)
