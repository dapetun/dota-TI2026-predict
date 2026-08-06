"""Team-level match features for XGBoost (no future leakage).

v0.3 additions:
  - cold-start: min_gp, 1/sqrt(gp+1), Empirical Bayes Elo shrink
  - schedule strength: opp_avg_elo
  - Glicko-2 μ / RD as uncertainty (alongside classic Elo)
  - separate form half-life (~40d) vs rating continuity
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.rating_systems import GlickoRating

# Empirical Bayes shrinkage prior strength (games).
ELO_PRIOR_K: float = 12.0
ELO_PRIOR: float = 1500.0
# Form decay half-life (days) for WR rolling — short-term form.
FORM_HALF_LIFE_DAYS: float = 40.0
# Rating sample-weight half-life is documented in sample_weights (210d default).


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
    # v0.3 uncertainty / cold-start
    "min_gp",
    "r_uncertainty",
    "d_uncertainty",
    "diff_uncertainty",
    "r_elo_shrunk",
    "d_elo_shrunk",
    "diff_elo_shrunk",
    "r_opp_avg_elo",
    "d_opp_avg_elo",
    "diff_opp_avg_elo",
    "r_glicko_mu",
    "d_glicko_mu",
    "diff_glicko_mu",
    "r_glicko_rd",
    "d_glicko_rd",
    "diff_glicko_rd",
]


def _form_stats(
    history: list[tuple[int, bool, float]],
    *,
    as_of_ts: int | None = None,
    form_half_life_days: float = FORM_HALF_LIFE_DAYS,
    decay: float = 0.95,
) -> tuple[float, float, int]:
    """Return (decayed_wr, last5_wr, win_streak) from pre-match history.

    When as_of_ts is set, applies exponential age decay with form_half_life_days
    in addition to recency index decay — short-term form (~40d).
    """
    if not history:
        return 0.5, 0.5, 0

    recent = history[-15:]
    idx_w = np.array([decay ** (len(recent) - 1 - i) * recent[i][2] for i in range(len(recent))])
    if as_of_ts is not None and form_half_life_days > 0:
        age_days = np.array(
            [max(0.0, (as_of_ts - recent[i][0]) / 86400.0) for i in range(len(recent))]
        )
        time_w = np.power(0.5, age_days / form_half_life_days)
        weights = idx_w * time_w
    else:
        weights = idx_w
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


def shrink_elo(elo: float, gp: float, *, k: float = ELO_PRIOR_K, prior: float = ELO_PRIOR) -> float:
    """Empirical Bayes shrink Elo toward prior: w=gp/(gp+k)."""
    w = float(gp) / (float(gp) + k)
    return w * float(elo) + (1.0 - w) * prior


def gp_uncertainty(gp: float) -> float:
    """Cold-start proxy: 1/sqrt(gp+1)."""
    return 1.0 / np.sqrt(float(gp) + 1.0)


@dataclass
class TeamStateStore:
    """Running Elo/form/H2H/Glicko after replaying match history."""

    elo: dict[int, float] = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    history: dict[int, list[tuple[int, bool, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    last_played: dict[int, int] = field(default_factory=dict)
    h2h: dict[tuple[int, int], list[tuple[int, bool]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Sum of opponent Elo before each match (for schedule strength).
    opp_elo_sum: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    opp_elo_n: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    glicko: GlickoRating = field(default_factory=GlickoRating)


def _team_side_stats(store: TeamStateStore, team_id: int, as_of_ts: int) -> dict[str, float]:
    """Pre-match side stats for one OpenDota team_id."""
    elo = float(store.elo[team_id])
    hist = store.history[team_id]
    wr, wr5, streak = _form_stats(hist, as_of_ts=as_of_ts)
    gp = len(hist)
    avg_tier = float(np.mean([x[2] for x in hist[-10:]])) if hist else 1.0
    days = (
        (as_of_ts - store.last_played[team_id]) / 86400.0
        if team_id in store.last_played
        else 30.0
    )
    n_opp = store.opp_elo_n[team_id]
    opp_avg = float(store.opp_elo_sum[team_id] / n_opp) if n_opp > 0 else ELO_PRIOR
    g = store.glicko.get(team_id)
    return {
        "elo": elo,
        "wr": wr,
        "wr5": wr5,
        "streak": float(streak),
        "gp": float(gp),
        "avg_tier": avg_tier,
        "days_since": float(days),
        "uncertainty": gp_uncertainty(gp),
        "elo_shrunk": shrink_elo(elo, gp),
        "opp_avg_elo": opp_avg,
        "glicko_mu": float(g["mu"]),
        "glicko_rd": float(g["rd"]),
    }


def compose_pair_features(
    store: TeamStateStore,
    radiant_team_id: int,
    dire_team_id: int,
    as_of_ts: int,
    tier_weight: float = 2.0,
) -> dict[str, float]:
    """Build FEATURE_COLUMNS for a hypothetical Radiant vs Dire matchup."""
    r = _team_side_stats(store, radiant_team_id, as_of_ts)
    d = _team_side_stats(store, dire_team_id, as_of_ts)
    elo_prob = 1.0 / (1.0 + 10 ** ((d["elo_shrunk"] - r["elo_shrunk"]) / 400.0))

    pair = tuple(sorted((radiant_team_id, dire_team_id)))
    past = store.h2h[pair][-10:]
    if past:
        r_wins = sum(
            1
            for hid, hw in past
            if (hid == radiant_team_id and hw) or (hid == dire_team_id and not hw)
        )
        h2h_wr = r_wins / len(past)
    else:
        h2h_wr = 0.5

    return {
        "r_elo": r["elo"],
        "d_elo": d["elo"],
        "diff_elo": r["elo"] - d["elo"],
        "abs_elo_diff": abs(r["elo"] - d["elo"]),
        "elo_prob": elo_prob,
        "r_wr": r["wr"],
        "d_wr": d["wr"],
        "diff_wr": r["wr"] - d["wr"],
        "r_wr5": r["wr5"],
        "d_wr5": d["wr5"],
        "diff_wr5": r["wr5"] - d["wr5"],
        "r_streak": r["streak"],
        "d_streak": d["streak"],
        "diff_streak": r["streak"] - d["streak"],
        "h2h_wr": h2h_wr,
        "diff_h2h": h2h_wr - 0.5,
        "r_gp": r["gp"],
        "d_gp": d["gp"],
        "diff_gp": r["gp"] - d["gp"],
        "r_avg_tier": r["avg_tier"],
        "d_avg_tier": d["avg_tier"],
        "diff_tier": r["avg_tier"] - d["avg_tier"],
        "tier_weight": float(tier_weight),
        "r_days_since": r["days_since"],
        "d_days_since": d["days_since"],
        "min_gp": min(r["gp"], d["gp"]),
        "r_uncertainty": r["uncertainty"],
        "d_uncertainty": d["uncertainty"],
        "diff_uncertainty": r["uncertainty"] - d["uncertainty"],
        "r_elo_shrunk": r["elo_shrunk"],
        "d_elo_shrunk": d["elo_shrunk"],
        "diff_elo_shrunk": r["elo_shrunk"] - d["elo_shrunk"],
        "r_opp_avg_elo": r["opp_avg_elo"],
        "d_opp_avg_elo": d["opp_avg_elo"],
        "diff_opp_avg_elo": r["opp_avg_elo"] - d["opp_avg_elo"],
        "r_glicko_mu": r["glicko_mu"],
        "d_glicko_mu": d["glicko_mu"],
        "diff_glicko_mu": r["glicko_mu"] - d["glicko_mu"],
        "r_glicko_rd": r["glicko_rd"],
        "d_glicko_rd": d["glicko_rd"],
        "diff_glicko_rd": r["glicko_rd"] - d["glicko_rd"],
    }


def _update_after_match(
    store: TeamStateStore,
    r_id: int,
    d_id: int,
    t: int,
    tier_w: float,
    r_win: bool,
    *,
    elo_k_base: float = 20.0,
) -> None:
    """Apply Elo / Glicko / history updates after emitting features."""
    r_elo = store.elo[r_id]
    d_elo = store.elo[d_id]
    elo_prob = 1.0 / (1.0 + 10 ** ((d_elo - r_elo) / 400.0))
    k = elo_k_base * (1.0 + (tier_w - 1.0) * 0.4)
    store.elo[r_id] = r_elo + k * ((1.0 if r_win else 0.0) - elo_prob)
    store.elo[d_id] = d_elo + k * ((0.0 if r_win else 1.0) - (1.0 - elo_prob))
    store.history[r_id].append((t, r_win, tier_w))
    store.history[d_id].append((t, not r_win, tier_w))
    pair = tuple(sorted((r_id, d_id)))
    store.h2h[pair].append((r_id, r_win))
    store.last_played[r_id] = t
    store.last_played[d_id] = t
    store.opp_elo_sum[r_id] += d_elo
    store.opp_elo_n[r_id] += 1
    store.opp_elo_sum[d_id] += r_elo
    store.opp_elo_n[d_id] += 1
    if r_win:
        store.glicko.update(r_id, d_id)
    else:
        store.glicko.update(d_id, r_id)


def replay_team_states(
    matches: pd.DataFrame,
    elo_k_base: float = 20.0,
) -> TeamStateStore:
    """Replay all matches and return final team rating/form store."""
    store = TeamStateStore()
    if matches.empty:
        return store
    df = matches.sort_values("start_time").reset_index(drop=True)
    for _, m in df.iterrows():
        r_id = int(m["radiant_team_id"])
        d_id = int(m["dire_team_id"])
        t = int(m["start_time"])
        tier_w = float(m.get("tier_weight", 1.0))
        r_win = bool(m["radiant_win"])
        _update_after_match(store, r_id, d_id, t, tier_w, r_win, elo_k_base=elo_k_base)
    return store


def team_strength_summary(store: TeamStateStore, team_id: int, as_of_ts: int) -> dict[str, float]:
    """μ ± σ snapshot for UI export (Elo shrunk + Glicko RD)."""
    s = _team_side_stats(store, team_id, as_of_ts)
    # Combine gp-uncertainty scaled to Elo points with Glicko RD.
    sigma = float(np.sqrt(s["glicko_rd"] ** 2 * 0.25 + (80.0 * s["uncertainty"]) ** 2))
    return {
        "mu": s["elo_shrunk"],
        "sigma": sigma,
        "elo": s["elo"],
        "elo_shrunk": s["elo_shrunk"],
        "glicko_mu": s["glicko_mu"],
        "glicko_rd": s["glicko_rd"],
        "gp": s["gp"],
        "uncertainty": s["uncertainty"],
        "opp_avg_elo": s["opp_avg_elo"],
    }


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
    store = TeamStateStore()
    rows: list[dict] = []

    for _, m in df.iterrows():
        r_id = int(m["radiant_team_id"])
        d_id = int(m["dire_team_id"])
        t = int(m["start_time"])
        tier_w = float(m.get("tier_weight", 1.0))
        r_win = bool(m["radiant_win"])

        feats = compose_pair_features(store, r_id, d_id, t, tier_weight=tier_w)
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
                **feats,
                "radiant_win": int(r_win),
            }
        )
        _update_after_match(store, r_id, d_id, t, tier_w, r_win, elo_k_base=elo_k_base)

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
