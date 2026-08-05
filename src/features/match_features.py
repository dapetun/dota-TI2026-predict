"""Team-level match features for XGBoost (no future leakage)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# Documented feature set (see docs/FEATURES.md).
FEATURE_COLUMNS = [
    "r_elo",
    "d_elo",
    "diff_elo",
    "abs_elo_diff",
    "elo_prob",
    "r_wr",
    "d_wr",
    "diff_wr",
    "r_wr5",
    "d_wr5",
    "diff_wr5",
    "r_streak",
    "d_streak",
    "diff_streak",
    "h2h_wr",
    "diff_h2h",
    "r_gp",
    "d_gp",
    "diff_gp",
    "r_avg_tier",
    "d_avg_tier",
    "diff_tier",
    "tier_weight",
    "r_days_since",
    "d_days_since",
]


def _form_stats(
    history: list[tuple[int, bool, float]],
    decay: float = 0.95,
) -> tuple[float, float, int]:
    """Return (decayed_wr, last5_wr, win_streak) from pre-match history."""
    if not history:
        return 0.5, 0.5, 0

    recent = history[-15:]
    weights = np.array([decay ** (len(recent) - 1 - i) * recent[i][2] for i in range(len(recent))])
    wins = np.array([1.0 if h[1] else 0.0 for h in recent])
    wr = float(np.average(wins, weights=weights)) if weights.sum() > 0 else 0.5

    last5 = recent[-5:]
    wr5 = float(np.mean([1.0 if h[1] else 0.0 for h in last5]))

    streak = 0
    for _, won, _ in reversed(recent):
        if won:
            streak += 1
        else:
            break
    return wr, wr5, streak


def build_match_feature_matrix(
    matches: pd.DataFrame,
    elo_k_base: float = 20.0,
    min_games: int = 5,
) -> pd.DataFrame:
    """Build pre-match features chronologically.

    Ratings and form are updated AFTER each row is emitted, so there is no
    leakage from the current match outcome into its own features.
    """
    if matches.empty:
        return pd.DataFrame()

    df = matches.sort_values("start_time").reset_index(drop=True)

    elo: dict[int, float] = defaultdict(lambda: 1500.0)
    history: dict[int, list[tuple[int, bool, float]]] = defaultdict(list)
    last_played: dict[int, int] = {}
    h2h: dict[tuple[int, int], list[tuple[int, bool]]] = defaultdict(list)

    rows: list[dict] = []

    for _, m in df.iterrows():
        r_id = int(m["radiant_team_id"])
        d_id = int(m["dire_team_id"])
        t = int(m["start_time"])
        tier_w = float(m.get("tier_weight", 1.0))
        r_win = bool(m["radiant_win"])

        r_elo = elo[r_id]
        d_elo = elo[d_id]
        elo_prob = 1.0 / (1.0 + 10 ** ((d_elo - r_elo) / 400.0))

        r_wr, r_wr5, r_streak = _form_stats(history[r_id])
        d_wr, d_wr5, d_streak = _form_stats(history[d_id])

        pair = tuple(sorted((r_id, d_id)))
        past = h2h[pair][-10:]
        if past:
            r_wins = sum(
                1
                for hid, hw in past
                if (hid == r_id and hw) or (hid == d_id and not hw)
            )
            h2h_wr = r_wins / len(past)
        else:
            h2h_wr = 0.5

        r_gp = len(history[r_id])
        d_gp = len(history[d_id])
        r_avg_tier = (
            float(np.mean([x[2] for x in history[r_id][-10:]])) if history[r_id] else 1.0
        )
        d_avg_tier = (
            float(np.mean([x[2] for x in history[d_id][-10:]])) if history[d_id] else 1.0
        )

        r_days = (t - last_played[r_id]) / 86400.0 if r_id in last_played else 30.0
        d_days = (t - last_played[d_id]) / 86400.0 if d_id in last_played else 30.0

        rows.append(
            {
                "match_id": int(m["match_id"]),
                "start_time": t,
                "date": m["date"],
                "tournament": m["tournament"],
                "tier": m["tier"],
                "year": m.get("year"),
                "radiant_team_id": r_id,
                "dire_team_id": d_id,
                "radiant_canonical": m.get("radiant_canonical", ""),
                "dire_canonical": m.get("dire_canonical", ""),
                "r_elo": r_elo,
                "d_elo": d_elo,
                "diff_elo": r_elo - d_elo,
                "abs_elo_diff": abs(r_elo - d_elo),
                "elo_prob": elo_prob,
                "r_wr": r_wr,
                "d_wr": d_wr,
                "diff_wr": r_wr - d_wr,
                "r_wr5": r_wr5,
                "d_wr5": d_wr5,
                "diff_wr5": r_wr5 - d_wr5,
                "r_streak": r_streak,
                "d_streak": d_streak,
                "diff_streak": r_streak - d_streak,
                "h2h_wr": h2h_wr,
                "diff_h2h": h2h_wr - 0.5,
                "r_gp": r_gp,
                "d_gp": d_gp,
                "diff_gp": r_gp - d_gp,
                "r_avg_tier": r_avg_tier,
                "d_avg_tier": d_avg_tier,
                "diff_tier": r_avg_tier - d_avg_tier,
                "tier_weight": tier_w,
                "r_days_since": r_days,
                "d_days_since": d_days,
                "radiant_win": int(r_win),
            }
        )

        # Update state AFTER features are recorded.
        r_exp = elo_prob
        k = elo_k_base * (1.0 + (tier_w - 1.0) * 0.4)
        elo[r_id] = r_elo + k * ((1.0 if r_win else 0.0) - r_exp)
        elo[d_id] = d_elo + k * ((0.0 if r_win else 1.0) - (1.0 - r_exp))
        history[r_id].append((t, r_win, tier_w))
        history[d_id].append((t, not r_win, tier_w))
        h2h[pair].append((r_id, r_win))
        last_played[r_id] = t
        last_played[d_id] = t

    feat = pd.DataFrame(rows)
    if min_games > 0:
        feat = feat[(feat["r_gp"] >= min_games) & (feat["d_gp"] >= min_games)]
    return feat.reset_index(drop=True)


def save_features(df: pd.DataFrame, features_dir: str | Path = "data/features") -> Path:
    """Save feature matrix."""
    out_dir = Path(features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "match_features_xgb.csv"
    df.to_csv(out, index=False)
    return out
