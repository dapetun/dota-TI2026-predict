"""Load OpenDota matchlists into a canonical match table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data_collection.tournaments import TOURNAMENTS, TournamentMeta
from src.ti2026.teams import normalize_team_name


CANONICAL_COLUMNS = [
    "match_id",
    "start_time",
    "date",
    "radiant_team_id",
    "dire_team_id",
    "radiant_team_name",
    "dire_team_name",
    "radiant_canonical",
    "dire_canonical",
    "radiant_win",
    "duration",
    "radiant_score",
    "dire_score",
    "series_id",
    "series_type",
    "league_id",
    "tournament",
    "tier",
    "tier_weight",
    "is_lan",
    "year",
]


def _load_team_id_map(raw_dir: Path) -> dict[int, str]:
    map_file = raw_dir / "team_id_map.json"
    if not map_file.exists():
        return {}
    with open(map_file, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _resolve_name(
    explicit_name: str | None,
    team_id: int | None,
    id_map: dict[int, str],
) -> str:
    if explicit_name:
        return str(explicit_name)
    if team_id is not None and int(team_id) in id_map:
        return id_map[int(team_id)]
    return ""


def _matchlist_paths(raw_dir: Path) -> list[Path]:
    """Prefer *_matchlist.json; fall back to *_matches.json."""
    lists = sorted(raw_dir.glob("*_matchlist.json"))
    if lists:
        return lists
    return sorted(raw_dir.glob("*_matches.json"))


def _tournament_key_from_path(path: Path) -> str:
    stem = path.stem
    for suffix in ("_matchlist", "_matches"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_raw_matchlists(
    raw_dir: str | Path = "data/raw",
    tournament_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load curated tournament matchlists into a canonical DataFrame.

    Expects OpenDota `/leagues/{id}/matches` JSON objects on disk.
    Does not require player-level match details.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    id_map = _load_team_id_map(raw_path)
    allowed = set(tournament_keys) if tournament_keys else set(TOURNAMENTS)
    rows: list[dict] = []

    for path in _matchlist_paths(raw_path):
        key = _tournament_key_from_path(path)
        if key not in allowed:
            continue
        meta: TournamentMeta | None = TOURNAMENTS.get(key)
        if meta is None:
            continue

        with open(path, encoding="utf-8") as f:
            matches = json.load(f)

        for m in matches:
            r_id = m.get("radiant_team_id")
            d_id = m.get("dire_team_id")
            if not r_id or not d_id or r_id == d_id:
                continue

            r_name = _resolve_name(m.get("radiant_team_name"), r_id, id_map)
            d_name = _resolve_name(m.get("dire_team_name"), d_id, id_map)
            start = int(m.get("start_time") or 0)
            if start <= 0:
                continue

            rows.append(
                {
                    "match_id": int(m["match_id"]),
                    "start_time": start,
                    "date": pd.to_datetime(start, unit="s").strftime("%Y-%m-%d"),
                    "radiant_team_id": int(r_id),
                    "dire_team_id": int(d_id),
                    "radiant_team_name": r_name,
                    "dire_team_name": d_name,
                    "radiant_canonical": normalize_team_name(r_name) if r_name else "",
                    "dire_canonical": normalize_team_name(d_name) if d_name else "",
                    "radiant_win": bool(m.get("radiant_win")),
                    "duration": int(m.get("duration") or 0),
                    "radiant_score": int(m.get("radiant_score") or 0),
                    "dire_score": int(m.get("dire_score") or 0),
                    "series_id": m.get("series_id"),
                    "series_type": int(m.get("series_type") or 1),
                    "league_id": int(m.get("leagueid") or meta.league_id),
                    "tournament": key,
                    "tier": meta.tier,
                    "tier_weight": meta.tier_weight,
                    "is_lan": meta.is_lan,
                    "year": meta.year,
                }
            )

    if not rows:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["match_id"]).sort_values("start_time")
    return df.reset_index(drop=True)


def summarize_matches(df: pd.DataFrame) -> dict:
    """Human-readable summary of the canonical match table."""
    if df.empty:
        return {"n_matches": 0}
    return {
        "n_matches": int(len(df)),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "n_tournaments": int(df["tournament"].nunique()),
        "tournaments": sorted(df["tournament"].unique().tolist()),
        "n_team_ids": int(
            pd.concat([df["radiant_team_id"], df["dire_team_id"]]).nunique()
        ),
        "radiant_winrate": float(df["radiant_win"].mean()),
        "by_tournament": df.groupby("tournament").size().to_dict(),
    }


def save_canonical_matches(
    df: pd.DataFrame,
    processed_dir: str | Path = "data/processed",
) -> Path:
    """Persist canonical matches without mutating raw files."""
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "canonical_matches.csv"
    df.to_csv(out, index=False)
    return out
