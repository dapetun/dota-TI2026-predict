"""Unified data loader — legacy STRATZ/OpenDota CSV path.

Deprecated for production: prefer ``match_loader.load_raw_matchlists`` +
``train_compare`` / ``export_web_data``. Team-name SoT is ``ti2026.teams``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ti2026.teams import ALIAS_TO_CANONICAL, normalize_team_name

# Legacy display map (canonical ids aligned with ti2026.teams — Vision not "Team Vision").
TEAM_NAME_MAP = {
    "BetBoom Team": "BetBoom",
    "BoomBoys": "BetBoom",
    "PARIVISION": "Vision",
    "Team Vision": "Vision",
    "Tundra Esports": "1w",
    "1w": "1w",
    "Team Spirit": "Spirit",
    "Aurora Gaming": "Aurora",
    "Aurora": "Aurora",
    "Team Falcons": "Falcons",
    "Team Liquid": "Liquid",
    "Xtreme Gaming": "Xtreme",
    "Nigma Galaxy": "Nigma",
    "OG": "OG",
    "Vici Gaming": "Vici",
    "GamerLegion": "GamerLegion",
    "Team Yandex": "Yandex",
    "HULIGANI": "HULIGANI",
    "L1GA TEAM": "HULIGANI",
    "Team Resilience": "Resilience",
}

TEAM_ID_MAP: dict = {}

__all__ = [
    "TEAM_NAME_MAP",
    "TEAM_ID_MAP",
    "normalize_team_name",
    "ALIAS_TO_CANONICAL",
    "load_raw_matches",
    "expand_players",
    "build_team_match_results",
    "save_processed_data",
    "load_processed_data",
]


def load_raw_matches(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load and merge all raw match CSVs into a canonical DataFrame."""
    data_path = Path(data_dir)
    dfs = []

    for csv_file in data_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if "id" in df.columns and "didRadiantWin" in df.columns:
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")

    if not dfs:
        print("No raw match data found!")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined.drop_duplicates(subset=["id"], inplace=True)
    combined.sort_values("startDateTime", ascending=True, inplace=True)
    combined.reset_index(drop=True, inplace=True)

    return combined


def expand_players(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Expand player-level data from match records."""
    records = []

    for _, row in matches_df.iterrows():
        match_id = row["id"]
        radiant_win = row.get("didRadiantWin", False)
        duration = row.get("durationSeconds", 0)
        start_dt = row.get("startDateTime", 0)
        league_id = row.get("leagueId", None)
        radiant_team = row.get("radiantTeamId", None)
        dire_team = row.get("direTeamId", None)

        players = row.get("players", [])
        if isinstance(players, str):
            try:
                players = json.loads(players)
            except (json.JSONDecodeError, TypeError):
                continue

        if not isinstance(players, list):
            continue

        for p in players:
            if isinstance(p, str):
                try:
                    p = json.loads(p)
                except (json.JSONDecodeError, TypeError):
                    continue

            is_radiant = p.get("isRadiant", False)
            team_won = (is_radiant and radiant_win) or (not is_radiant and not radiant_win)

            records.append({
                "match_id": match_id,
                "hero_id": p.get("heroId"),
                "steam_id": p.get("steamAccountId"),
                "is_radiant": is_radiant,
                "team_id": radiant_team if is_radiant else dire_team,
                "networth": p.get("networth", 0),
                "kills": p.get("kills", 0),
                "deaths": p.get("deaths", 0),
                "assists": p.get("assists", 0),
                "gpm": p.get("goldPerMinute", 0),
                "xpm": p.get("experiencePerMinute", 0),
                "hero_damage": p.get("heroDamage", 0),
                "tower_damage": p.get("towerDamage", 0),
                "healing": p.get("healing", 0),
                "level": p.get("level", 0),
                "team_won": team_won,
                "duration": duration,
                "start_time": start_dt,
                "league_id": league_id,
            })

    return pd.DataFrame(records)


def build_team_match_results(
    matches_df: pd.DataFrame,
    player_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build team-level match results from player data."""
    if player_df.empty:
        return pd.DataFrame()

    # Aggregate per team per match
    team_agg = player_df.groupby(["match_id", "team_id"]).agg({
        "kills": "sum",
        "deaths": "sum",
        "assists": "sum",
        "networth": "sum",
        "gpm": "mean",
        "xpm": "mean",
        "hero_damage": "sum",
        "tower_damage": "sum",
        "healing": "sum",
        "team_won": "first",
        "is_radiant": "first",
        "duration": "first",
        "start_time": "first",
        "league_id": "first",
        "hero_id": lambda x: list(x),
    }).reset_index()

    team_agg.rename(columns={
        "kills": "team_kills",
        "deaths": "team_deaths",
        "assists": "team_assists",
        "networth": "team_networth",
        "hero_damage": "team_hero_damage",
        "tower_damage": "team_tower_damage",
        "healing": "team_healing",
        "hero_id": "hero_picks",
    }, inplace=True)

    return team_agg


def load_processed_data(
    processed_dir: str = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed match + player DataFrames."""
    proc_path = Path(processed_dir)
    matches_file = proc_path / "team_matches.csv"
    players_file = proc_path / "player_matches.csv"

    matches = pd.DataFrame()
    players = pd.DataFrame()

    if matches_file.exists():
        matches = pd.read_csv(matches_file)
    if players_file.exists():
        players = pd.read_csv(players_file)

    return matches, players


def save_processed_data(
    team_matches: pd.DataFrame,
    player_matches: pd.DataFrame,
    processed_dir: str = "data/processed",
):
    """Save processed DataFrames."""
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    team_matches.to_csv(proc_path / "team_matches.csv", index=False)
    player_matches.to_csv(proc_path / "player_matches.csv", index=False)
    print(f"Saved {len(team_matches)} team matches, {len(player_matches)} player matches")


if __name__ == "__main__":
    matches = load_raw_matches()
    print(f"Loaded {len(matches)} raw matches")
    if not matches.empty:
        players = expand_players(matches)
        team_matches = build_team_match_results(matches, players)
        save_processed_data(team_matches, players)
