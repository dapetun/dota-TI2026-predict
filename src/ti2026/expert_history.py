"""Historical expert boards → slot priors and accuracy weights.

Research-only curation from public articles. Not betting advice.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.simulation.tournament_sim import FANTASY_BOARD_SLOTS
from src.ti2026.compendium_scoring import VALVE_GROUP_POINTS
from src.ti2026.teams import normalize_team_name

DEFAULT_EXPERT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "historical" / "expert_predictions.json"
)
DEFAULT_GT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "historical"
    / "ti_swiss_ground_truth.json"
)

_SLOT_KEYS = list(FANTASY_BOARD_SLOTS.keys())


def load_expert_predictions(path: str | Path | None = None) -> dict:
    """Load curated historical + current expert boards JSON."""
    path = Path(path or DEFAULT_EXPERT_PATH)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_swiss_ground_truth(path: str | Path | None = None) -> dict:
    """Load Liquipedia-curated Swiss / group slot outcomes."""
    path = Path(path or DEFAULT_GT_PATH)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _gt_assignment(tournament: dict) -> dict[str, str]:
    """Flatten slots → team_id -> slot_key."""
    out: dict[str, str] = {}
    for slot, teams in (tournament.get("slots") or {}).items():
        for name in teams:
            tid = normalize_team_name(name)
            out[tid] = slot
            # keep raw name too for historical teams not in TI2026 map
            out[name] = slot
    return out


def score_expert_board(
    board: dict[str, str],
    ground_truth: dict[str, str],
    *,
    partial: bool = False,
) -> dict[str, float]:
    """Hit rate and Valve points vs ground-truth assignment."""
    scored = 0
    correct = 0
    for team, slot in board.items():
        tid = normalize_team_name(team)
        gt = ground_truth.get(tid) or ground_truth.get(team)
        if gt is None:
            if partial:
                continue
            scored += 1
            continue
        scored += 1
        if gt == slot:
            correct += 1
    hit_rate = (correct / scored) if scored else 0.0
    points = float(VALVE_GROUP_POINTS.get(correct, 0))
    return {
        "correct": float(correct),
        "scored": float(scored),
        "hit_rate": hit_rate,
        "valve_points": points,
        "expected_points_proxy": points,
    }


def expert_accuracy_weights(
    experts_data: dict | None = None,
    gt_data: dict | None = None,
    *,
    prior_weight: float = 1.0,
) -> dict[str, float]:
    """Weight experts by historical hit rate (additive smoothing)."""
    experts_data = experts_data or load_expert_predictions()
    gt_data = gt_data or load_swiss_ground_truth()
    tournaments = gt_data.get("tournaments") or {}
    weights: dict[str, float] = {}
    for row in experts_data.get("experts", []):
        eid = str(row.get("id", row.get("name", "")))
        ti = str(row.get("ti", ""))
        tourn = tournaments.get(ti)
        if not tourn:
            weights[eid] = prior_weight
            continue
        gt = _gt_assignment(tourn)
        stats = score_expert_board(
            row.get("board") or {},
            gt,
            partial=bool(row.get("partial")),
        )
        # skill proxy: hit_rate with Laplace-ish floor
        w = prior_weight + float(stats["correct"])
        weights[eid] = max(0.05, w)
    return weights


def historical_expert_slot_prior(
    team_id: str,
    slot_key: str,
    *,
    ti: str | None = None,
    experts_data: dict | None = None,
    gt_data: dict | None = None,
    smoothing: float = 0.5,
) -> float:
    """Weighted vote prior from historical expert boards (optionally one TI)."""
    experts_data = experts_data or load_expert_predictions()
    weights = expert_accuracy_weights(experts_data, gt_data)
    votes = {s: 0.0 for s in _SLOT_KEYS}
    for row in experts_data.get("experts", []):
        if ti and str(row.get("ti")) != ti:
            continue
        board = row.get("board") or {}
        # map aliases
        slot = None
        for name, s in board.items():
            if normalize_team_name(name) == team_id or name == team_id:
                slot = s
                break
        if slot not in votes:
            continue
        eid = str(row.get("id", row.get("name", "")))
        votes[slot] += float(weights.get(eid, 1.0))
    total = sum(votes.values()) + smoothing * len(_SLOT_KEYS)
    return (votes.get(slot_key, 0.0) + smoothing) / total


def build_expert_history_priors(
    team_ids: list[str],
    *,
    ti: str | None = None,
    experts_data: dict | None = None,
) -> dict[str, dict[str, float]]:
    """team_id -> slot -> prior from historical expert boards."""
    experts_data = experts_data or load_expert_predictions()
    return {
        tid: {
            slot: historical_expert_slot_prior(
                tid, slot, ti=ti, experts_data=experts_data
            )
            for slot in _SLOT_KEYS
        }
        for tid in team_ids
    }


def score_all_experts(
    experts_data: dict | None = None,
    gt_data: dict | None = None,
) -> list[dict]:
    """Score each curated expert board against available GT."""
    experts_data = experts_data or load_expert_predictions()
    gt_data = gt_data or load_swiss_ground_truth()
    tournaments = gt_data.get("tournaments") or {}
    out: list[dict] = []
    for row in experts_data.get("experts", []):
        ti = str(row.get("ti", ""))
        tourn = tournaments.get(ti)
        entry = {
            "id": row.get("id"),
            "name": row.get("name"),
            "ti": ti,
            "partial": bool(row.get("partial")),
        }
        if tourn:
            stats = score_expert_board(
                row.get("board") or {},
                _gt_assignment(tourn),
                partial=bool(row.get("partial")),
            )
            entry.update(stats)
        out.append(entry)
    return out
