"""Global hero meta WR / pick rates (OpenDota heroStats snapshot)."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_HERO_META_PATH = Path(__file__).resolve().parents[2] / "data" / "hero" / "hero_meta.json"


def load_hero_meta(path: str | Path | None = None) -> dict:
    """Load hero_meta JSON (id → wr / games / name)."""
    path = Path(path or DEFAULT_HERO_META_PATH)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hero_winrate(meta: dict, hero_id: int, default: float = 0.5) -> float:
    """Global meta WR for hero_id."""
    heroes = meta.get("heroes") or meta
    entry = heroes.get(str(hero_id)) or heroes.get(hero_id)
    if not entry:
        return default
    return float(entry.get("winrate", entry.get("wr", default)))


def build_hero_meta_from_hero_stats(hero_stats: list[dict]) -> dict:
    """Convert OpenDota /heroStats payload into compact hero_meta."""
    heroes: dict[str, dict] = {}
    for row in hero_stats:
        hid = int(row.get("id") or row.get("hero_id") or 0)
        if hid <= 0:
            continue
        picks = float(row.get("pro_pick") or row.get("8000_pick") or row.get("pick") or 0)
        wins = float(row.get("pro_win") or row.get("8000_win") or row.get("win") or 0)
        # Prefer high-MMR pub if present
        for bracket in ("8000", "7000", "6300"):
            p = float(row.get(f"{bracket}_pick") or 0)
            w = float(row.get(f"{bracket}_win") or 0)
            if p > picks:
                picks, wins = p, w
        wr = (wins / picks) if picks > 0 else 0.5
        heroes[str(hid)] = {
            "name": row.get("localized_name") or row.get("name") or str(hid),
            "games": int(picks),
            "wins": int(wins),
            "winrate": round(wr, 4),
        }
    return {
        "source": "opendota_heroStats",
        "note": "Experimental meta snapshot for hero soft prior.",
        "heroes": heroes,
    }


def save_hero_meta(data: dict, path: str | Path | None = None) -> Path:
    """Write hero_meta JSON."""
    path = Path(path or DEFAULT_HERO_META_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
