"""Player-level rolling features aggregated to pre-match team vectors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


PLAYER_FEATURE_COLUMNS = [
    "r_pl_kda",
    "d_pl_kda",
    "diff_pl_kda",
    "r_pl_gpm",
    "d_pl_gpm",
    "diff_pl_gpm",
    "r_pl_xpm",
    "d_pl_xpm",
    "diff_pl_xpm",
    "r_pl_hdpm",
    "d_pl_hdpm",
    "diff_pl_hdpm",
    "r_pl_tdpm",
    "d_pl_tdpm",
    "diff_pl_tdpm",
    "r_pl_wr",
    "d_pl_wr",
    "diff_pl_wr",
    "r_pl_games",
    "d_pl_games",
    "diff_pl_games",
    "r_pl_lan_wr",
    "d_pl_lan_wr",
    "diff_pl_lan_wr",
    "has_player_stats",
]


@dataclass
class _PlayerState:
    """Running history for one account (updated only after a match is emitted)."""

    times: list[int]
    kda: list[float]
    gpm: list[float]
    xpm: list[float]
    hdpm: list[float]
    tdpm: list[float]
    won: list[float]
    lan_won: list[float]
    lan_mask: list[bool]

    @classmethod
    def empty(cls) -> "_PlayerState":
        return cls([], [], [], [], [], [], [], [], [])


def _kda(kills: float, deaths: float, assists: float) -> float:
    return (kills + assists) / max(deaths, 1.0)


def _mean_tail(values: list[float], n: int = 20) -> float:
    if not values:
        return 0.0
    chunk = values[-n:]
    return float(np.mean(chunk))


def _wr_tail(values: list[float], n: int = 20, default: float = 0.5) -> float:
    if not values:
        return default
    chunk = values[-n:]
    return float(np.mean(chunk))


def _lan_wr(state: _PlayerState, n: int = 20, default: float = 0.5) -> float:
    pairs = [
        (w, m)
        for w, m in zip(state.lan_won[-n:], state.lan_mask[-n:])
        if m
    ]
    if not pairs:
        return default
    return float(np.mean([w for w, _ in pairs]))


def _team_player_vector(
    account_ids: list[int],
    states: dict[int, _PlayerState],
) -> dict[str, float]:
    """Aggregate pre-match rolling stats for a lineup."""
    if not account_ids:
        return {
            "pl_kda": 0.0,
            "pl_gpm": 0.0,
            "pl_xpm": 0.0,
            "pl_hdpm": 0.0,
            "pl_tdpm": 0.0,
            "pl_wr": 0.5,
            "pl_games": 0.0,
            "pl_lan_wr": 0.5,
        }

    kdas, gpms, xpms, hdpms, tdpms, wrs, games, lan_wrs = [], [], [], [], [], [], [], []
    for aid in account_ids:
        st = states.get(aid) or _PlayerState.empty()
        kdas.append(_mean_tail(st.kda))
        gpms.append(_mean_tail(st.gpm))
        xpms.append(_mean_tail(st.xpm))
        hdpms.append(_mean_tail(st.hdpm))
        tdpms.append(_mean_tail(st.tdpm))
        wrs.append(_wr_tail(st.won))
        games.append(float(len(st.times)))
        lan_wrs.append(_lan_wr(st))

    return {
        "pl_kda": float(np.mean(kdas)),
        "pl_gpm": float(np.mean(gpms)),
        "pl_xpm": float(np.mean(xpms)),
        "pl_hdpm": float(np.mean(hdpms)),
        "pl_tdpm": float(np.mean(tdpms)),
        "pl_wr": float(np.mean(wrs)),
        "pl_games": float(np.mean(games)),
        "pl_lan_wr": float(np.mean(lan_wrs)),
    }


def _append_player_result(state: _PlayerState, row: pd.Series) -> None:
    duration = max(int(row.get("duration") or 0), 1)
    minutes = duration / 60.0
    state.times.append(int(row["start_time"]))
    state.kda.append(_kda(row["kills"], row["deaths"], row["assists"]))
    state.gpm.append(float(row["gpm"]))
    state.xpm.append(float(row["xpm"]))
    state.hdpm.append(float(row["hero_damage"]) / minutes)
    state.tdpm.append(float(row["tower_damage"]) / minutes)
    won = 1.0 if bool(row["team_won"]) else 0.0
    state.won.append(won)
    is_lan = bool(row.get("is_lan", True))
    state.lan_mask.append(is_lan)
    state.lan_won.append(won if is_lan else 0.0)


def build_player_match_features(
    matches: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Build pre-match player aggregates for each side of every match.

    Player histories are updated only AFTER the match row is emitted.
    Matches without lineup details get neutral defaults and has_player_stats=0.
    """
    if matches.empty:
        return pd.DataFrame(columns=["match_id", *PLAYER_FEATURE_COLUMNS])

    by_match: dict[int, pd.DataFrame] = {}
    if players is not None and not players.empty:
        for mid, group in players.groupby("match_id"):
            by_match[int(mid)] = group

    states: dict[int, _PlayerState] = defaultdict(_PlayerState.empty)
    rows: list[dict] = []
    ordered = matches.sort_values("start_time").reset_index(drop=True)

    for _, m in ordered.iterrows():
        mid = int(m["match_id"])
        group = by_match.get(mid)
        has_stats = group is not None and not group.empty

        if has_stats:
            radiant_ids = [
                int(a)
                for a in group.loc[group["is_radiant"] == True, "account_id"].tolist()  # noqa: E712
            ]
            dire_ids = [
                int(a)
                for a in group.loc[group["is_radiant"] == False, "account_id"].tolist()  # noqa: E712
            ]
            r_vec = _team_player_vector(radiant_ids, states)
            d_vec = _team_player_vector(dire_ids, states)
        else:
            r_vec = _team_player_vector([], states)
            d_vec = _team_player_vector([], states)

        rows.append(
            {
                "match_id": mid,
                "r_pl_kda": r_vec["pl_kda"],
                "d_pl_kda": d_vec["pl_kda"],
                "diff_pl_kda": r_vec["pl_kda"] - d_vec["pl_kda"],
                "r_pl_gpm": r_vec["pl_gpm"],
                "d_pl_gpm": d_vec["pl_gpm"],
                "diff_pl_gpm": r_vec["pl_gpm"] - d_vec["pl_gpm"],
                "r_pl_xpm": r_vec["pl_xpm"],
                "d_pl_xpm": d_vec["pl_xpm"],
                "diff_pl_xpm": r_vec["pl_xpm"] - d_vec["pl_xpm"],
                "r_pl_hdpm": r_vec["pl_hdpm"],
                "d_pl_hdpm": d_vec["pl_hdpm"],
                "diff_pl_hdpm": r_vec["pl_hdpm"] - d_vec["pl_hdpm"],
                "r_pl_tdpm": r_vec["pl_tdpm"],
                "d_pl_tdpm": d_vec["pl_tdpm"],
                "diff_pl_tdpm": r_vec["pl_tdpm"] - d_vec["pl_tdpm"],
                "r_pl_wr": r_vec["pl_wr"],
                "d_pl_wr": d_vec["pl_wr"],
                "diff_pl_wr": r_vec["pl_wr"] - d_vec["pl_wr"],
                "r_pl_games": r_vec["pl_games"],
                "d_pl_games": d_vec["pl_games"],
                "diff_pl_games": r_vec["pl_games"] - d_vec["pl_games"],
                "r_pl_lan_wr": r_vec["pl_lan_wr"],
                "d_pl_lan_wr": d_vec["pl_lan_wr"],
                "diff_pl_lan_wr": r_vec["pl_lan_wr"] - d_vec["pl_lan_wr"],
                "has_player_stats": 1 if has_stats else 0,
            }
        )

        if has_stats:
            for _, prow in group.iterrows():
                aid = int(prow["account_id"])
                _append_player_result(states[aid], prow)

    return pd.DataFrame(rows)


def merge_team_and_player_features(
    team_features: pd.DataFrame,
    player_features: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join player aggregates onto the team feature matrix."""
    if team_features.empty:
        return team_features
    if player_features.empty:
        out = team_features.copy()
        for col in PLAYER_FEATURE_COLUMNS:
            if col == "has_player_stats":
                out[col] = 0
            elif "wr" in col:
                out[col] = 0.5 if col.startswith(("r_", "d_")) else 0.0
            else:
                out[col] = 0.0
        return out
    return team_features.merge(player_features, on="match_id", how="left").fillna(
        {
            **{c: 0.0 for c in PLAYER_FEATURE_COLUMNS if c != "has_player_stats"},
            "has_player_stats": 0,
            "r_pl_wr": 0.5,
            "d_pl_wr": 0.5,
            "r_pl_lan_wr": 0.5,
            "d_pl_lan_wr": 0.5,
        }
    )
