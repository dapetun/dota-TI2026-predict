"""Pre-match roster chemistry / co-play features (no leakage)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

CHEMISTRY_FEATURE_COLUMNS = [
    "r_chem_mean",
    "d_chem_mean",
    "diff_chem_mean",
    "r_chem_min",
    "d_chem_min",
    "diff_chem_min",
    "r_chem_90d",
    "d_chem_90d",
    "diff_chem_90d",
    "r_roster_jaccard",
    "d_roster_jaccard",
    "diff_roster_jaccard",
    "has_chemistry",
    # v0.3: joint WR of pairs + roster stability over 60d
    "r_chem_pair_wr",
    "d_chem_pair_wr",
    "diff_chem_pair_wr",
    "r_roster_stability_60d",
    "d_roster_stability_60d",
    "diff_roster_stability_60d",
]

_WINDOW_90D = 90 * 86400
_WINDOW_60D = 60 * 86400


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _lineup_ids(group: pd.DataFrame, radiant: bool) -> list[int]:
    mask = group["is_radiant"] == True if radiant else group["is_radiant"] == False  # noqa: E712
    ids = [int(a) for a in group.loc[mask, "account_id"].tolist() if int(a) > 0]
    # Stable unique order for Jaccard / pairs.
    seen: set[int] = set()
    out: list[int] = []
    for aid in ids:
        if aid not in seen:
            seen.add(aid)
            out.append(aid)
    return out


def _pair_stats(
    ids: list[int],
    pair_games: dict[tuple[int, int], int],
    pair_last: dict[tuple[int, int], int],
    now: int,
    pair_wins: dict[tuple[int, int], int] | None = None,
) -> tuple[float, float, float, float]:
    """Return mean co-play, min co-play, mean co-play within 90d, joint pair WR."""
    if len(ids) < 2:
        return 0.0, 0.0, 0.0, 0.5
    counts: list[float] = []
    recent: list[float] = []
    wrs: list[float] = []
    for a, b in combinations(ids, 2):
        key = _pair_key(a, b)
        c = float(pair_games.get(key, 0))
        counts.append(c)
        last = pair_last.get(key, 0)
        recent.append(c if last and (now - last) <= _WINDOW_90D else 0.0)
        if pair_wins is not None and c > 0:
            wrs.append(float(pair_wins.get(key, 0)) / c)
        else:
            wrs.append(0.5)
    return (
        float(np.mean(counts)),
        float(np.min(counts)),
        float(np.mean(recent)),
        float(np.mean(wrs)) if wrs else 0.5,
    )


def _jaccard(prev: set[int] | None, cur: set[int]) -> float:
    if not prev or not cur:
        return 0.0
    inter = len(prev & cur)
    union = len(prev | cur)
    return float(inter / union) if union else 0.0


def _roster_stability_60d(
    roster_hist: list[tuple[int, set[int]]],
    cur: set[int],
    now: int,
) -> float:
    """Mean Jaccard vs lineups of the same team in the last 60 days."""
    if not cur or not roster_hist:
        return 0.0
    scores = [
        _jaccard(prev, cur)
        for ts, prev in roster_hist
        if now - ts <= _WINDOW_60D and prev
    ]
    return float(np.mean(scores)) if scores else 0.0


def _update_pairs(
    ids: list[int],
    start_time: int,
    pair_games: dict[tuple[int, int], int],
    pair_last: dict[tuple[int, int], int],
    *,
    won: bool | None = None,
    pair_wins: dict[tuple[int, int], int] | None = None,
) -> None:
    for a, b in combinations(ids, 2):
        key = _pair_key(a, b)
        pair_games[key] = pair_games.get(key, 0) + 1
        pair_last[key] = start_time
        if pair_wins is not None and won is not None and won:
            pair_wins[key] = pair_wins.get(key, 0) + 1


def build_chemistry_features(
    matches: pd.DataFrame,
    players: pd.DataFrame,
    *,
    lan_only: bool = False,
) -> pd.DataFrame:
    """Build pre-match co-play / roster-continuity features per match.

    Pair and roster histories update only AFTER the match row is emitted.
    Set lan_only=True for LAN-only ablation (updates only from LAN matches).
    """
    if matches.empty:
        return pd.DataFrame(columns=["match_id", *CHEMISTRY_FEATURE_COLUMNS])

    by_match: dict[int, pd.DataFrame] = {}
    if players is not None and not players.empty:
        for mid, group in players.groupby("match_id"):
            by_match[int(mid)] = group

    pair_games: dict[tuple[int, int], int] = {}
    pair_last: dict[tuple[int, int], int] = {}
    pair_wins: dict[tuple[int, int], int] = {}
    last_roster: dict[int, set[int]] = {}
    roster_hist: dict[int, list[tuple[int, set[int]]]] = defaultdict(list)
    rows: list[dict] = []
    ordered = matches.sort_values("start_time").reset_index(drop=True)

    for m in ordered.itertuples(index=False):
        mid = int(m.match_id)
        now = int(m.start_time)
        group = by_match.get(mid)
        has = group is not None and not group.empty
        r_win = bool(getattr(m, "radiant_win", False))

        if has:
            r_ids = _lineup_ids(group, True)
            d_ids = _lineup_ids(group, False)
            r_mean, r_min, r_90, r_pwr = _pair_stats(
                r_ids, pair_games, pair_last, now, pair_wins
            )
            d_mean, d_min, d_90, d_pwr = _pair_stats(
                d_ids, pair_games, pair_last, now, pair_wins
            )
            r_team = int(getattr(m, "radiant_team_id", 0) or 0)
            d_team = int(getattr(m, "dire_team_id", 0) or 0)
            r_jac = _jaccard(last_roster.get(r_team), set(r_ids)) if r_team else 0.0
            d_jac = _jaccard(last_roster.get(d_team), set(d_ids)) if d_team else 0.0
            r_stab = (
                _roster_stability_60d(roster_hist.get(r_team, []), set(r_ids), now)
                if r_team
                else 0.0
            )
            d_stab = (
                _roster_stability_60d(roster_hist.get(d_team, []), set(d_ids), now)
                if d_team
                else 0.0
            )
        else:
            r_mean = d_mean = r_min = d_min = r_90 = d_90 = 0.0
            r_pwr = d_pwr = 0.5
            r_jac = d_jac = r_stab = d_stab = 0.0
            r_ids = d_ids = []
            r_team = d_team = 0

        rows.append(
            {
                "match_id": mid,
                "r_chem_mean": r_mean,
                "d_chem_mean": d_mean,
                "diff_chem_mean": r_mean - d_mean,
                "r_chem_min": r_min,
                "d_chem_min": d_min,
                "diff_chem_min": r_min - d_min,
                "r_chem_90d": r_90,
                "d_chem_90d": d_90,
                "diff_chem_90d": r_90 - d_90,
                "r_roster_jaccard": r_jac,
                "d_roster_jaccard": d_jac,
                "diff_roster_jaccard": r_jac - d_jac,
                "has_chemistry": 1 if has else 0,
                "r_chem_pair_wr": r_pwr,
                "d_chem_pair_wr": d_pwr,
                "diff_chem_pair_wr": r_pwr - d_pwr,
                "r_roster_stability_60d": r_stab,
                "d_roster_stability_60d": d_stab,
                "diff_roster_stability_60d": r_stab - d_stab,
            }
        )

        if has and (not lan_only or bool(getattr(m, "is_lan", False))):
            _update_pairs(
                r_ids, now, pair_games, pair_last, won=r_win, pair_wins=pair_wins
            )
            _update_pairs(
                d_ids, now, pair_games, pair_last, won=not r_win, pair_wins=pair_wins
            )
            if r_team:
                last_roster[r_team] = set(r_ids)
                roster_hist[r_team].append((now, set(r_ids)))
            if d_team:
                last_roster[d_team] = set(d_ids)
                roster_hist[d_team].append((now, set(d_ids)))

    return pd.DataFrame(rows)


@dataclass
class ChemistryState:
    """Co-play / roster continuity after replaying match history."""

    pair_games: dict[tuple[int, int], int] = field(default_factory=dict)
    pair_last: dict[tuple[int, int], int] = field(default_factory=dict)
    pair_wins: dict[tuple[int, int], int] = field(default_factory=dict)
    last_roster: dict[int, set[int]] = field(default_factory=dict)
    roster_hist: dict[int, list[tuple[int, set[int]]]] = field(default_factory=dict)


def replay_chemistry_state(
    matches: pd.DataFrame,
    players: pd.DataFrame,
    *,
    lan_only: bool = False,
) -> ChemistryState:
    """Replay history and return final chemistry state (no feature rows)."""
    state = ChemistryState()
    if matches.empty or players is None or players.empty:
        return state

    by_match: dict[int, pd.DataFrame] = {}
    for mid, group in players.groupby("match_id"):
        by_match[int(mid)] = group

    for m in matches.sort_values("start_time").itertuples(index=False):
        if lan_only and not bool(getattr(m, "is_lan", False)):
            continue
        mid = int(m.match_id)
        now = int(m.start_time)
        r_win = bool(getattr(m, "radiant_win", False))
        group = by_match.get(mid)
        if group is None or group.empty:
            continue
        r_ids = _lineup_ids(group, True)
        d_ids = _lineup_ids(group, False)
        _update_pairs(
            r_ids, now, state.pair_games, state.pair_last, won=r_win, pair_wins=state.pair_wins
        )
        _update_pairs(
            d_ids,
            now,
            state.pair_games,
            state.pair_last,
            won=not r_win,
            pair_wins=state.pair_wins,
        )
        r_team = int(getattr(m, "radiant_team_id", 0) or 0)
        d_team = int(getattr(m, "dire_team_id", 0) or 0)
        if r_team:
            state.last_roster[r_team] = set(r_ids)
            state.roster_hist.setdefault(r_team, []).append((now, set(r_ids)))
        if d_team:
            state.last_roster[d_team] = set(d_ids)
            state.roster_hist.setdefault(d_team, []).append((now, set(d_ids)))
    return state


def compose_chemistry_pair_features(
    radiant_ids: list[int],
    dire_ids: list[int],
    radiant_team_id: int,
    dire_team_id: int,
    state: ChemistryState,
    as_of_ts: int,
) -> dict[str, float]:
    """Build CHEMISTRY_FEATURE_COLUMNS for a hypothetical matchup."""
    r_mean, r_min, r_90, r_pwr = _pair_stats(
        radiant_ids, state.pair_games, state.pair_last, as_of_ts, state.pair_wins
    )
    d_mean, d_min, d_90, d_pwr = _pair_stats(
        dire_ids, state.pair_games, state.pair_last, as_of_ts, state.pair_wins
    )
    r_jac = _jaccard(state.last_roster.get(radiant_team_id), set(radiant_ids))
    d_jac = _jaccard(state.last_roster.get(dire_team_id), set(dire_ids))
    r_stab = _roster_stability_60d(
        state.roster_hist.get(radiant_team_id, []), set(radiant_ids), as_of_ts
    )
    d_stab = _roster_stability_60d(
        state.roster_hist.get(dire_team_id, []), set(dire_ids), as_of_ts
    )
    has = bool(radiant_ids and dire_ids)
    return {
        "r_chem_mean": r_mean,
        "d_chem_mean": d_mean,
        "diff_chem_mean": r_mean - d_mean,
        "r_chem_min": r_min,
        "d_chem_min": d_min,
        "diff_chem_min": r_min - d_min,
        "r_chem_90d": r_90,
        "d_chem_90d": d_90,
        "diff_chem_90d": r_90 - d_90,
        "r_roster_jaccard": r_jac,
        "d_roster_jaccard": d_jac,
        "diff_roster_jaccard": r_jac - d_jac,
        "has_chemistry": 1.0 if has else 0.0,
        "r_chem_pair_wr": r_pwr,
        "d_chem_pair_wr": d_pwr,
        "diff_chem_pair_wr": r_pwr - d_pwr,
        "r_roster_stability_60d": r_stab,
        "d_roster_stability_60d": d_stab,
        "diff_roster_stability_60d": r_stab - d_stab,
    }


def merge_chemistry_features(
    features: pd.DataFrame,
    chemistry: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join chemistry columns onto an existing feature matrix."""
    if features.empty:
        return features
    if chemistry.empty:
        out = features.copy()
        for col in CHEMISTRY_FEATURE_COLUMNS:
            out[col] = 0 if col == "has_chemistry" else 0.0
        return out
    fill = {c: 0.0 for c in CHEMISTRY_FEATURE_COLUMNS}
    fill["has_chemistry"] = 0
    return features.merge(chemistry, on="match_id", how="left").fillna(fill)
