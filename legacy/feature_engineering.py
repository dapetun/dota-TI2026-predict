"""Feature engineering pipeline for Dota 2 match prediction.

DEPRECATED legacy path. Prefer ``src.features.match_features`` (+ player /
chemistry) used by ``train_compare`` / ``export_web_data``.
"""

import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path


def compute_team_form(
    team_matches: pd.DataFrame,
    window: int = 20,
    decay: float = 0.95,
) -> pd.DataFrame:
    """Compute rolling form metrics per team."""
    records = []

    for team_id, group in team_matches.sort_values("start_time").groupby("team_id"):
        group = group.sort_values("start_time").reset_index(drop=True)

        wins = []
        kill_rates = []
        death_rates = []
        gpm_diffs = []
        xpm_diffs = []
        tower_dmg_rates = []
        durations = []

        for i, row in group.iterrows():
            # Weighted win rate
            window_slice = group.iloc[max(0, i - window) : i + 1]
            weights = np.array([decay ** (len(window_slice) - 1 - j) for j in range(len(window_slice))])
            win_flags = window_slice["team_won"].astype(float).values
            form_wr = np.average(win_flags, weights=weights) if len(win_flags) > 0 else 0.5

            # K/D/A rates
            kr = row.get("team_kills", 0) / max(row.get("duration", 1), 1) * 60
            dr = row.get("team_deaths", 0) / max(row.get("duration", 1), 1) * 60

            gpm = row.get("gpm", 0)
            xpm = row.get("xpm", 0)
            td = row.get("team_tower_damage", 0) / max(row.get("duration", 1), 1) * 60

            records.append({
                "match_id": row["match_id"],
                "team_id": team_id,
                "form_wr_10": form_wr,
                "form_wr_5": np.average(
                    group.iloc[max(0, i - 5) : i + 1]["team_won"].astype(float).values,
                    weights=np.array([decay ** (min(i, 5) - j) for j in range(min(i + 1, 5))])
                ) if i >= 0 else 0.5,
                "avg_kill_rate": kr,
                "avg_death_rate": dr,
                "avg_gpm": gpm,
                "avg_xpm": xpm,
                "avg_tower_dmg_rate": td,
                "avg_duration": row.get("duration", 0),
            })

    return pd.DataFrame(records)


def compute_hero_features(
    player_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Compute hero pool and draft features per team."""
    if player_matches.empty:
        return pd.DataFrame()

    records = []
    for team_id, group in player_matches.groupby("team_id"):
        hero_picks = group["hero_id"].value_counts()
        hero_wins = group[group["team_won"] == True]["hero_id"].value_counts()

        total = hero_picks.sum()
        unique_heroes = len(hero_picks)
        top5_wr = []

        for hero_id in hero_picks.head(10).index:
            picks = hero_picks[hero_id]
            wins = hero_wins.get(hero_id, 0)
            wr = wins / picks if picks > 0 else 0
            top5_wr.append(wr)

        records.append({
            "team_id": team_id,
            "hero_pool_size": unique_heroes,
            "avg_hero_wr_top10": np.mean(top5_wr) if top5_wr else 0.5,
            "hero_concentration": hero_picks.iloc[0] / total if len(hero_picks) > 0 else 0,
        })

    return pd.DataFrame(records)


def compute_player_features(
    player_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-team player performance aggregates."""
    if player_matches.empty:
        return pd.DataFrame()

    agg = player_matches.groupby(["team_id", "match_id"]).agg({
        "gpm": "mean",
        "xpm": "mean",
        "hero_damage": "sum",
        "tower_damage": "sum",
        "healing": "sum",
        "kills": "sum",
        "deaths": "sum",
        "assists": "sum",
        "level": "mean",
    }).reset_index()

    team_features = agg.groupby("team_id").agg({
        "gpm": "mean",
        "xpm": "mean",
        "hero_damage": "mean",
        "tower_damage": "mean",
        "healing": "mean",
        "kills": "mean",
        "deaths": "mean",
        "assists": "mean",
        "level": "mean",
    }).reset_index()

    team_features.columns = ["team_id"] + [f"avg_{c}" for c in team_features.columns if c != "team_id"]
    return team_features


def compute_meta_features(
    player_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Compute patch meta features (hero winrates, popularity)."""
    if player_matches.empty:
        return pd.DataFrame()

    hero_stats = player_matches.groupby("hero_id").agg(
        picks=("match_id", "count"),
        wins=("team_won", "sum"),
    ).reset_index()

    hero_stats["winrate"] = hero_stats["wins"] / hero_stats["picks"]
    hero_stats["popularity"] = hero_stats["picks"] / hero_stats["picks"].sum()

    return hero_stats


def build_match_features(
    team_matches: pd.DataFrame,
    player_matches: pd.DataFrame,
    form_df: pd.DataFrame,
    hero_features: pd.DataFrame,
    player_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build final feature matrix for each match from both teams' perspectives."""
    if team_matches.empty:
        return pd.DataFrame()

    features_list = []

    for match_id, group in team_matches.groupby("match_id"):
        if len(group) < 2:
            continue

        team_a = group.iloc[0]
        team_b = group.iloc[1]

        feat = {"match_id": match_id}

        # Basic info
        feat["radiant_team"] = team_a["team_id"] if team_a.get("is_radiant", True) else team_b["team_id"]
        feat["dire_team"] = team_b["team_id"] if team_a.get("is_radiant", True) else team_a["team_id"]
        feat["start_time"] = team_a.get("start_time", 0)
        feat["duration"] = team_a.get("duration", 0)
        feat["league_id"] = team_a.get("league_id", None)

        # Team A features
        for prefix, tdata in [("a", team_a), ("b", team_b)]:
            tid = int(tdata["team_id"])
            form_row = form_df[form_df["team_id"] == tid]
            hero_row = hero_features[hero_features["team_id"] == tid]
            player_row = player_features[player_features["team_id"] == tid]

            if not form_row.empty:
                for col in form_row.columns:
                    if col not in ("team_id", "match_id"):
                        feat[f"{prefix}_{col}"] = form_row.iloc[-1][col]

            if not hero_row.empty:
                for col in hero_row.columns:
                    if col != "team_id":
                        feat[f"{prefix}_{col}"] = hero_row.iloc[-1][col]

            if not player_row.empty:
                for col in player_row.columns:
                    if col != "team_id":
                        feat[f"{prefix}_{col}"] = player_row.iloc[-1][col]

        # Differential features
        for metric in ["form_wr_10", "avg_kill_rate", "avg_death_rate", "avg_gpm", "avg_xpm"]:
            a_key = f"a_{metric}"
            b_key = f"b_{metric}"
            if a_key in feat and b_key in feat:
                feat[f"diff_{metric}"] = feat[a_key] - feat[b_key]

        # Target
        feat["radiant_win"] = bool(team_a.get("team_won", False)) if team_a.get("is_radiant", True) else bool(team_b.get("team_won", False))

        features_list.append(feat)

    return pd.DataFrame(features_list)


def add_elo_features(
    features_df: pd.DataFrame,
    elo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add Elo ratings to match feature matrix."""
    if elo_df.empty:
        return features_df

    elo_lookup = elo_df.pivot_table(
        index="match_id", columns="team_id", values="elo_before", aggfunc="first"
    ).to_dict("index")

    for idx, row in features_df.iterrows():
        match_elo = elo_lookup.get(row["match_id"], {})
        radiant = row.get("radiant_team")
        dire = row.get("dire_team")

        if radiant in match_elo:
            features_df.at[idx, "a_elo"] = match_elo[radiant]
        if dire in match_elo:
            features_df.at[idx, "b_elo"] = match_elo[dire]

    if "a_elo" in features_df.columns and "b_elo" in features_df.columns:
        features_df["diff_elo"] = features_df["a_elo"] - features_df["b_elo"]

    return features_df


def save_features(features_df: pd.DataFrame, output_dir: str = "data/features"):
    """Save feature matrix."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / "match_features.csv"
    features_df.to_csv(output_file, index=False)
    print(f"Saved {len(features_df)} match features to {output_file}")
