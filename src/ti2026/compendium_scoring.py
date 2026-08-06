"""Valve compendium group-stage scoring and board optimization."""

from __future__ import annotations

from copy import deepcopy

from src.simulation.tournament_sim import FANTASY_BOARD_SLOTS, assign_fantasy_board

# Official TI compendium: total points by number of correct slot picks (of 16).
VALVE_GROUP_POINTS: dict[int, int] = {
    0: 0,
    1: 30,
    2: 60,
    3: 120,
    4: 360,
    5: 720,
    6: 1200,
    7: 1800,
    8: 2520,
    9: 3360,
    10: 4320,
    11: 5400,
    12: 6600,
    13: 7920,
    14: 9360,
    15: 10920,
    16: 12000,
}

SLOT_PROB_KEYS: dict[str, str] = {
    "undefeated": "prob_4_0",
    "one_loss": "prob_4_1",
    "advance": "prob_advance",
    "eliminate": "prob_eliminate",
    "one_win": "prob_1_4",
    "winless": "prob_0_4",
}


def slot_probability(team: dict, slot_key: str) -> float:
    """P(team lands in slot) as fraction in [0, 1]."""
    raw = float(team.get(SLOT_PROB_KEYS[slot_key], 0.0))
    return max(0.0, min(1.0, raw / 100.0))


def assignment_to_board(
    predictions: list[dict],
    assignment: dict[str, str],
) -> dict[str, list[dict]]:
    """Build board dict from team_id -> slot_key mapping."""
    by_id = {p["id"]: p for p in predictions}
    board: dict[str, list[dict]] = {k: [] for k in FANTASY_BOARD_SLOTS}
    for tid, slot in assignment.items():
        p = by_id[tid]
        board[slot].append(_board_entry(p, slot))
    return board


def _board_entry(p: dict, slot: str) -> dict:
    prob_key = SLOT_PROB_KEYS[slot]
    return {
        "id": p["id"],
        "name": p["name"],
        "short": p.get("short", p["id"]),
        "record": FANTASY_BOARD_SLOTS[slot]["label"],
        "qualify_pct": p.get("qualify_pct", 0.0),
        "slot_pct": float(p.get(prob_key, 0.0)),
    }


def expected_correct(assignment: dict[str, str], predictions: list[dict]) -> float:
    """Expected number of exact slot hits (linear objective)."""
    by_id = {p["id"]: p for p in predictions}
    return sum(
        slot_probability(by_id[tid], slot)
        for tid, slot in assignment.items()
    )


def correct_count_distribution(
    assignment: dict[str, str],
    predictions: list[dict],
) -> dict[int, float]:
    """PMF of # correct picks assuming independent slot outcomes."""
    by_id = {p["id"]: p for p in predictions}
    dist: dict[int, float] = {0: 1.0}
    for tid, slot in assignment.items():
        p_hit = slot_probability(by_id[tid], slot)
        nxt: dict[int, float] = {}
        for k, pr in dist.items():
            nxt[k] = nxt.get(k, 0.0) + pr * (1.0 - p_hit)
            nxt[k + 1] = nxt.get(k + 1, 0.0) + pr * p_hit
        dist = nxt
    return dist


def expected_valve_points(
    assignment: dict[str, str],
    predictions: list[dict],
) -> float:
    """E[Valve compendium points] under independent slot hits."""
    dist = correct_count_distribution(assignment, predictions)
    return sum(prob * VALVE_GROUP_POINTS.get(k, 0) for k, prob in dist.items())


def _assignment_from_board(board: dict[str, list[dict]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for slot, entries in board.items():
        for e in entries:
            out[e["id"]] = slot
    return out


def _greedy_assignment(predictions: list[dict]) -> dict[str, str]:
    """Greedy fill by highest (team, slot) probability edges."""
    caps = {k: v["capacity"] for k, v in FANTASY_BOARD_SLOTS.items()}
    remaining = dict(caps)
    assignment: dict[str, str] = {}

    edges: list[tuple[float, str, str]] = []
    for p in predictions:
        for slot in FANTASY_BOARD_SLOTS:
            edges.append((slot_probability(p, slot), p["id"], slot))
    edges.sort(key=lambda x: -x[0])

    for _, tid, slot in edges:
        if tid in assignment or remaining[slot] <= 0:
            continue
        assignment[tid] = slot
        remaining[slot] -= 1

    # Fallback: any team still unassigned → best open slot
    for p in predictions:
        if p["id"] in assignment:
            continue
        best_slot = max(
            (s for s, c in remaining.items() if c > 0),
            key=lambda s: slot_probability(p, s),
            default=None,
        )
        if best_slot is None:
            break
        assignment[p["id"]] = best_slot
        remaining[best_slot] -= 1
    return assignment


def _improve_by_swaps(
    assignment: dict[str, str],
    predictions: list[dict],
    *,
    max_rounds: int = 200,
) -> dict[str, str]:
    """Hill-climb: swap teams across slots while E[points] grows."""
    best = deepcopy(assignment)
    best_score = expected_valve_points(best, predictions)
    ids = list(best.keys())

    for _ in range(max_rounds):
        improved = False
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if best[a] == best[b]:
                    continue
                trial = deepcopy(best)
                trial[a], trial[b] = trial[b], trial[a]
                score = expected_valve_points(trial, predictions)
                if score > best_score + 1e-9:
                    best = trial
                    best_score = score
                    improved = True
        if not improved:
            break
    return best


def optimize_fantasy_board(
    predictions: list[dict],
    *,
    start: str = "greedy",
) -> dict[str, list[dict]]:
    """Maximize expected Valve points; greedy init + local swaps."""
    if start == "qualify":
        init = _assignment_from_board(assign_fantasy_board(predictions))
    else:
        init = _greedy_assignment(predictions)
    optimized = _improve_by_swaps(init, predictions)
    return assignment_to_board(predictions, optimized)


def compare_board_strategies(
    predictions: list[dict],
    *,
    extra_boards: dict[str, dict[str, list[dict]]] | None = None,
) -> dict:
    """Compare qualify-rank vs points-optimal and optional extra boards."""
    qualify_board = assign_fantasy_board(predictions)
    points_board = optimize_fantasy_board(predictions)
    qa = _assignment_from_board(qualify_board)
    pa = _assignment_from_board(points_board)
    out: dict[str, dict[str, float]] = {
        "qualify_rank": {
            "expected_correct": round(expected_correct(qa, predictions), 3),
            "expected_points": round(expected_valve_points(qa, predictions), 1),
        },
        "points_optimal": {
            "expected_correct": round(expected_correct(pa, predictions), 3),
            "expected_points": round(expected_valve_points(pa, predictions), 1),
        },
    }
    if extra_boards:
        for key, board in extra_boards.items():
            assign = _assignment_from_board(board)
            out[key] = {
                "expected_correct": round(expected_correct(assign, predictions), 3),
                "expected_points": round(expected_valve_points(assign, predictions), 1),
            }
    return out
