"""Team identity stitching by roster Jaccard across brand/team_id changes."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def build_team_lineup_history(
    players: pd.DataFrame,
    matches: pd.DataFrame | None = None,
) -> dict[int, list[tuple[int, frozenset[int]]]]:
    """team_id → chronological [(start_time, frozenset account_ids)]."""
    if players is None or players.empty:
        return {}
    hist: dict[int, list[tuple[int, frozenset[int]]]] = defaultdict(list)
    # Prefer match start_time from players; fallback via matches table.
    match_ts: dict[int, int] = {}
    if matches is not None and not matches.empty:
        for _, m in matches.iterrows():
            match_ts[int(m["match_id"])] = int(m["start_time"])

    for mid, group in players.groupby("match_id"):
        mid = int(mid)
        t = int(group["start_time"].iloc[0]) if "start_time" in group.columns else match_ts.get(mid, 0)
        for team_id, side in group.groupby("team_id"):
            tid = int(team_id)
            if tid <= 0:
                continue
            ids = frozenset(int(a) for a in side["account_id"].tolist() if int(a) > 0)
            if len(ids) < 3:
                continue
            hist[tid].append((t, ids))

    for tid in hist:
        hist[tid].sort(key=lambda x: x[0])
    return dict(hist)


def jaccard(a: frozenset[int] | set[int], b: frozenset[int] | set[int]) -> float:
    """Jaccard similarity of two player sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def build_team_stitch_map(
    players: pd.DataFrame,
    matches: pd.DataFrame | None = None,
    *,
    threshold: float = 0.6,
) -> dict[int, int]:
    """Map OpenDota team_id → canonical parent team_id via roster overlap.

    Union-find over pairs of team_ids whose latest 5-man lineups share
    Jaccard ≥ threshold. Parent = smallest team_id in the component
    (stable, deterministic).
    """
    hist = build_team_lineup_history(players, matches)
    if not hist:
        return {}

    latest: dict[int, frozenset[int]] = {
        tid: lineups[-1][1] for tid, lineups in hist.items() if lineups
    }
    ids = sorted(latest.keys())
    parent: dict[int, int] = {tid: tid for tid in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if jaccard(latest[a], latest[b]) >= threshold:
                union(a, b)

    return {tid: find(tid) for tid in ids}


def apply_team_stitch(
    matches: pd.DataFrame,
    stitch: dict[int, int],
    players: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Rewrite match + player team_ids through stitch map (copies).

    Returns ``(matches, players)``. Players remapped only when provided
    so ``latest_lineup_for_team`` sees parent ids after stitch.
    """
    if matches.empty or not stitch:
        m_out = matches.copy()
        p_out = players.copy() if players is not None else None
        return m_out, p_out

    out = matches.copy()
    out["radiant_team_id"] = out["radiant_team_id"].map(
        lambda x: stitch.get(int(x), int(x))
    )
    out["dire_team_id"] = out["dire_team_id"].map(
        lambda x: stitch.get(int(x), int(x))
    )

    p_out: pd.DataFrame | None = None
    if players is not None:
        p_out = players.copy()
        if "team_id" in p_out.columns and not p_out.empty:
            p_out["team_id"] = p_out["team_id"].map(
                lambda x: stitch.get(int(x), int(x))
            )
    return out, p_out
