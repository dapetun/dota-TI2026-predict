"""Analyst pick aggregation and compendium scoring."""

from __future__ import annotations

import json
from pathlib import Path

from src.simulation.tournament_sim import FANTASY_BOARD_SLOTS
from src.ti2026.compendium_scoring import (
    assignment_to_board,
    expected_correct,
    expected_valve_points,
    slot_probability,
)
from src.ti2026.teams import get_team_ids, normalize_team_name

DEFAULT_PICKS_PATH = Path(__file__).resolve().parents[2] / "docs" / "data" / "analyst_picks.json"


def load_analyst_picks(path: str | Path | None = None) -> dict:
    """Load analyst picks JSON (full grids + partial notes)."""
    path = Path(path or DEFAULT_PICKS_PATH)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_analyst_board(board: dict[str, str]) -> dict[str, str]:
    """Map team aliases to canonical ids."""
    out: dict[str, str] = {}
    for team, slot in board.items():
        tid = normalize_team_name(team)
        out[tid] = slot
    return out


def _rows_with_boards(data: dict, *, include_partial: bool) -> list[dict]:
    """Analyst / partial rows that carry a non-empty board."""
    rows = list(data.get("analysts") or [])
    if include_partial:
        rows.extend(data.get("partial") or [])
    return [row for row in rows if row.get("board")]


def analyst_assignments(
    picks_data: dict | None = None,
    *,
    include_partial: bool = True,
) -> list[dict[str, str]]:
    """List of team_id -> slot_key per analyst (partial boards may be incomplete)."""
    data = picks_data or load_analyst_picks()
    return [
        normalize_analyst_board(row.get("board") or {})
        for row in _rows_with_boards(data, include_partial=include_partial)
    ]


def full_analyst_count(picks_data: dict | None = None) -> int:
    """Number of complete Sports.ru-style grids (denominator for UI chips)."""
    data = picks_data or load_analyst_picks()
    return sum(1 for row in (data.get("analysts") or []) if row.get("board"))


def slot_vote_counts(
    assignments: list[dict[str, str]],
    team_id: str,
) -> dict[str, int]:
    """Votes per slot for one team."""
    counts = {slot: 0 for slot in FANTASY_BOARD_SLOTS}
    for assignment in assignments:
        slot = assignment.get(team_id)
        if slot in counts:
            counts[slot] += 1
    return counts


def analyst_agreement(
    team_id: str,
    slot_key: str,
    assignments: list[dict[str, str]] | None = None,
) -> int:
    """How many analysts placed team_id in slot_key."""
    assignments = assignments or analyst_assignments()
    return sum(1 for a in assignments if a.get(team_id) == slot_key)


def analyst_names_for_slot(
    team_id: str,
    slot_key: str,
    picks_data: dict | None = None,
) -> list[str]:
    """Analyst names agreeing on team slot (includes partial boards)."""
    data = picks_data or load_analyst_picks()
    names: list[str] = []
    for row in _rows_with_boards(data, include_partial=True):
        board = normalize_analyst_board(row.get("board") or {})
        if board.get(team_id) == slot_key:
            names.append(str(row.get("name", row.get("id", ""))))
    return names


def consensus_assignment(
    predictions: list[dict],
    assignments: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    """Majority-vote board; tie-break by model P(slot).

    Defaults to full grids only — partial boards lack capacity coverage.
    """
    assignments = (
        assignments
        if assignments is not None
        else analyst_assignments(include_partial=False)
    )
    n_analysts = len(assignments)
    if n_analysts == 0:
        return {p["id"]: "advance" for p in predictions}

    by_id = {p["id"]: p for p in predictions}
    team_ids = [p["id"] for p in predictions]
    caps = {k: v["capacity"] for k, v in FANTASY_BOARD_SLOTS.items()}
    remaining = dict(caps)
    assignment: dict[str, str] = {}

    # Slot-first: fill each slot with highest-vote teams not yet assigned.
    for slot in FANTASY_BOARD_SLOTS:
        need = remaining[slot]
        if need <= 0:
            continue
        candidates = [tid for tid in team_ids if tid not in assignment]
        scored: list[tuple[float, float, str]] = []
        for tid in candidates:
            votes = analyst_agreement(tid, slot, assignments)
            model_p = slot_probability(by_id[tid], slot) if tid in by_id else 0.0
            scored.append((float(votes), model_p, tid))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        for votes, _, tid in scored[:need]:
            assignment[tid] = slot
            remaining[slot] -= 1

    for tid in team_ids:
        if tid in assignment:
            continue
        best_slot = max(
            (s for s, c in remaining.items() if c > 0),
            key=lambda s: (
                analyst_agreement(tid, s, assignments),
                slot_probability(by_id[tid], s) if tid in by_id else 0.0,
            ),
            default=None,
        )
        if best_slot:
            assignment[tid] = best_slot
            remaining[best_slot] -= 1
    return assignment


def build_consensus_board(
    predictions: list[dict],
    assignments: list[dict[str, str]] | None = None,
) -> dict[str, list[dict]]:
    """Board dict from majority analyst consensus."""
    assignment = consensus_assignment(predictions, assignments)
    return assignment_to_board(predictions, assignment)


def score_board_assignment(
    assignment: dict[str, str],
    predictions: list[dict],
) -> dict[str, float]:
    """E[correct slots] and E[Valve points] for an assignment."""
    return {
        "expected_correct": round(expected_correct(assignment, predictions), 3),
        "expected_points": round(expected_valve_points(assignment, predictions), 1),
    }


def _partial_export_rows(data: dict) -> list[dict]:
    """Compact partial notes / boards for predictions.json."""
    out: list[dict] = []
    for row in data.get("partial") or []:
        entry = {
            "id": row.get("id"),
            "name": row.get("name"),
            "note": row.get("note"),
            "soft": bool(row.get("soft")),
            "source_url": row.get("source_url"),
            "updated_at": row.get("updated_at"),
        }
        board = row.get("board") or {}
        if board:
            entry["board"] = normalize_analyst_board(board)
            entry["n_slots"] = len(entry["board"])
        out.append(entry)
    return out


def consensus_summary(
    predictions: list[dict],
    picks_data: dict | None = None,
) -> dict:
    """Votes and agreement metadata for export/UI."""
    data = picks_data or load_analyst_picks()
    assignments = analyst_assignments(data, include_partial=True)
    n_full = full_analyst_count(data)
    # UI denominator stays full-grid count; partial votes still add to chips.
    n = n_full
    team_ids = get_team_ids()
    per_team: dict[str, dict] = {}
    for tid in team_ids:
        votes = slot_vote_counts(assignments, tid)
        best_slot = max(votes, key=lambda s: votes[s])
        per_team[tid] = {
            "votes": votes,
            "consensus_slot": best_slot,
            "consensus_count": votes[best_slot],
            "n_analysts": n,
        }
    # Capacity fill uses full grids only — partial boards are incomplete.
    full_assignments = analyst_assignments(data, include_partial=False)
    assignment = consensus_assignment(predictions, full_assignments)
    scores = score_board_assignment(assignment, predictions)
    partial_rows = _partial_export_rows(data)
    return {
        "n_analysts": n,
        "n_partial": len(partial_rows),
        "n_partial_boards": sum(1 for row in partial_rows if row.get("board")),
        "source": data.get("source"),
        "url": data.get("url"),
        "note": data.get("note"),
        "sources": data.get("sources") or [],
        "partial": partial_rows,
        "per_team": per_team,
        "assignment": assignment,
        "expected_correct": scores["expected_correct"],
        "expected_points": scores["expected_points"],
    }
