"""Load and normalize OpenDota match details into player-match rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data_collection.tournaments import TOURNAMENTS


PLAYER_COLUMNS = [
    "match_id",
    "start_time",
    "tournament",
    "tier_weight",
    "is_lan",
    "account_id",
    "hero_id",
    "is_radiant",
    "team_id",
    "kills",
    "deaths",
    "assists",
    "gpm",
    "xpm",
    "hero_damage",
    "tower_damage",
    "healing",
    "level",
    "lane_role",
    "team_won",
    "duration",
]


def _healing_value(raw: Any) -> int:
    if isinstance(raw, dict):
        return int(sum(raw.values()))
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _iter_detail_objects(raw_dir: Path) -> Iterable[tuple[str | None, dict]]:
    """Yield (tournament_key_or_None, detail_dict) from on-disk caches."""
    # Legacy combined store: {match_id: detail}
    combined = raw_dir / "match_details.json"
    if combined.exists():
        with open(combined, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            for mid, detail in payload.items():
                if isinstance(detail, dict):
                    detail = {**detail, "match_id": detail.get("match_id") or int(mid)}
                    yield None, detail
        elif isinstance(payload, list):
            for detail in payload:
                if isinstance(detail, dict):
                    yield None, detail

    # Per-tournament detail dumps from download_data.py
    for path in sorted(raw_dir.glob("*_details.json")):
        key = path.stem.replace("_details", "")
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            continue
        for detail in payload:
            if isinstance(detail, dict):
                yield key, detail


def _tournament_meta(key: str | None, league_id: int | None) -> tuple[str, float, bool]:
    if key and key in TOURNAMENTS:
        meta = TOURNAMENTS[key]
        return meta.key, meta.tier_weight, meta.is_lan
    if league_id is not None:
        for meta in TOURNAMENTS.values():
            if meta.league_id == int(league_id):
                return meta.key, meta.tier_weight, meta.is_lan
    return key or "unknown", 1.0, True


def detail_to_player_rows(
    detail: dict,
    tournament_key: str | None = None,
) -> list[dict]:
    """Convert one OpenDota /matches/{id} payload into player rows."""
    players = detail.get("players") or []
    if not players:
        return []

    match_id = int(detail["match_id"])
    start_time = int(detail.get("start_time") or 0)
    duration = int(detail.get("duration") or 0)
    radiant_win = bool(detail.get("radiant_win"))
    radiant_team_id = detail.get("radiant_team_id")
    dire_team_id = detail.get("dire_team_id")
    league_id = detail.get("leagueid") or detail.get("league_id")
    tourn, tier_w, is_lan = _tournament_meta(tournament_key, league_id)

    rows: list[dict] = []
    for i, p in enumerate(players):
        if not isinstance(p, dict):
            continue
        is_radiant = bool(p.get("isRadiant", p.get("is_radiant", i < 5)))
        account_id = p.get("account_id")
        if account_id in (None, 0, "0"):
            continue
        team_id = radiant_team_id if is_radiant else dire_team_id
        if not team_id:
            continue
        team_won = (is_radiant and radiant_win) or ((not is_radiant) and (not radiant_win))
        deaths = max(int(p.get("deaths") or 0), 0)
        kills = int(p.get("kills") or 0)
        assists = int(p.get("assists") or 0)
        rows.append(
            {
                "match_id": match_id,
                "start_time": start_time,
                "tournament": tourn,
                "tier_weight": tier_w,
                "is_lan": is_lan,
                "account_id": int(account_id),
                "hero_id": int(p.get("hero_id") or 0),
                "is_radiant": is_radiant,
                "team_id": int(team_id),
                "kills": kills,
                "deaths": deaths,
                "assists": assists,
                "gpm": int(p.get("gold_per_min") or p.get("gpm") or 0),
                "xpm": int(p.get("xp_per_min") or p.get("xpm") or 0),
                "hero_damage": int(p.get("hero_damage") or 0),
                "tower_damage": int(p.get("tower_damage") or 0),
                "healing": _healing_value(p.get("healing")),
                "level": int(p.get("level") or 0),
                "lane_role": int(p.get("lane_role") or p.get("role") or 0),
                "team_won": bool(team_won),
                "duration": duration,
            }
        )
    return rows


def load_player_matches(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load all available match details into a player-match DataFrame."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return pd.DataFrame(columns=PLAYER_COLUMNS)

    rows: list[dict] = []
    seen_matches: set[int] = set()
    for tourn_key, detail in _iter_detail_objects(raw_path):
        mid = detail.get("match_id")
        if mid is None:
            continue
        mid = int(mid)
        if mid in seen_matches:
            continue
        player_rows = detail_to_player_rows(detail, tourn_key)
        if not player_rows:
            continue
        seen_matches.add(mid)
        rows.extend(player_rows)

    if not rows:
        return pd.DataFrame(columns=PLAYER_COLUMNS)

    df = pd.DataFrame(rows)
    df = df.sort_values(["start_time", "match_id", "is_radiant"]).reset_index(drop=True)
    return df


def summarize_player_coverage(
    matches: pd.DataFrame,
    players: pd.DataFrame,
) -> dict:
    """Coverage stats: how many matches have player rows."""
    n_matches = int(len(matches)) if matches is not None and not matches.empty else 0
    if players is None or players.empty or n_matches == 0:
        return {
            "n_matches": n_matches,
            "n_matches_with_players": 0,
            "coverage": 0.0,
            "n_player_rows": 0,
            "n_accounts": 0,
        }
    with_players = set(players["match_id"].unique())
    match_ids = set(matches["match_id"].unique())
    covered = len(match_ids & with_players)
    return {
        "n_matches": n_matches,
        "n_matches_with_players": covered,
        "coverage": round(covered / n_matches, 3) if n_matches else 0.0,
        "n_player_rows": int(len(players)),
        "n_accounts": int(players["account_id"].nunique()),
    }


def save_player_matches(
    players: pd.DataFrame,
    processed_dir: str | Path = "data/processed",
) -> Path:
    """Persist player-match table."""
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "player_matches.csv"
    players.to_csv(out, index=False)
    return out
