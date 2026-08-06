"""Download match data using correct OpenDota endpoints.

Two-phase approach:
1. /leagues/{id}/matches → match list (fast, ~50ms each)
2. /matches/{match_id} → full detail with player data (slower)

Tournament registry: src.data_collection.tournaments (single source of truth).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_collection.tournaments import TOURNAMENTS, download_order

BASE_URL = "https://api.opendota.com/api"
RAW_DIR = BASE_DIR / "data" / "raw"


def download_league_match_list(league_id: int) -> list:
    """Скачать список матчей лиги OpenDota."""
    url = f"{BASE_URL}/leagues/{league_id}/matches"
    try:
        resp = requests.get(url, timeout=45)
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, list) else []
        if resp.status_code == 429:
            print("  Rate limited, waiting 60s...")
            time.sleep(60)
            return download_league_match_list(league_id)
        print(f"  Error {resp.status_code}: {resp.text[:100]}")
        return []
    except Exception as exc:  # noqa: BLE001
        print(f"  Exception: {exc}")
        return []


def download_match_detail(match_id: int, retries: int = 3) -> dict | None:
    """Скачать полный detail матча."""
    for _ in range(retries):
        try:
            resp = requests.get(f"{BASE_URL}/matches/{match_id}", timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                print(f"    Rate limited at match {match_id}, waiting 60s...")
                time.sleep(60)
            else:
                time.sleep(0.2)
        except Exception as exc:  # noqa: BLE001 — network resilience
            print(f"    Exception fetching {match_id}: {exc}")
            time.sleep(1)
    return None


def enrich_match_with_details(match: dict, detail: dict | None) -> dict:
    """Merge league match data with full match detail."""
    if not detail:
        return match

    players = detail.get("players", [])
    enriched_players = []
    for i, p in enumerate(players):
        is_radiant = i < 5
        team_id = match.get("radiant_team_id") if is_radiant else match.get("dire_team_id")
        team_won = (
            match.get("radiant_win", False) if is_radiant else not match.get("radiant_win", False)
        )

        healing = p.get("healing", 0)
        if isinstance(healing, dict):
            healing = sum(healing.values())

        enriched_players.append(
            {
                "hero_id": p.get("hero_id"),
                "account_id": p.get("account_id"),
                "is_radiant": is_radiant,
                "team_id": team_id,
                "kills": p.get("kills", 0),
                "deaths": p.get("deaths", 0),
                "assists": p.get("assists", 0),
                "gpm": p.get("gold_per_min", 0),
                "xpm": p.get("xp_per_min", 0),
                "hero_damage": p.get("hero_damage", 0),
                "tower_damage": p.get("tower_damage", 0),
                "healing": healing,
                "gold": p.get("gold", 0),
                "lane": p.get("lane"),
                "role": p.get("lane_role"),
                "level": p.get("level"),
                "team_won": team_won,
            }
        )

    match["players"] = enriched_players
    return match


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    detail_only = "--detail-only" in sys.argv
    list_only = "--list-only" in sys.argv
    specific = None
    for arg in sys.argv[1:]:
        if arg not in ("--detail-only", "--list-only") and arg in TOURNAMENTS:
            specific = arg

    tournaments = [specific] if specific else download_order()

    for key in tournaments:
        tourn = TOURNAMENTS[key]
        lid = tourn.league_id
        name = tourn.name
        print(f"\n{'=' * 50}")
        print(f"  {name} (league {lid}) [{tourn.tier} w={tourn.tier_weight}]")
        print(f"{'=' * 50}")

        list_file = RAW_DIR / f"{key}_matchlist.json"
        if not list_file.exists() and not detail_only:
            print("  Downloading match list...")
            matches = download_league_match_list(lid)
            print(f"  Found {len(matches)} matches")
            with open(list_file, "w", encoding="utf-8") as f:
                json.dump(matches, f)
            time.sleep(0.5)
        else:
            if not list_file.exists():
                print("  No matchlist on disk, skipping")
                continue
            with open(list_file, encoding="utf-8") as f:
                matches = json.load(f)
            print(f"  Already have {len(matches)} matches in list")

        if list_only:
            continue

        # Legacy per-tournament details path (prefer scripts/download_details.py).
        detail_file = RAW_DIR / f"{key}_details.json"
        if detail_file.exists():
            with open(detail_file, encoding="utf-8") as f:
                existing_details = json.load(f)
            existing_ids = {d["match_id"] for d in existing_details if d}
            print(f"  Already have {len(existing_details)} details, checking for missing...")
        else:
            existing_details = []
            existing_ids = set()

        needed = [m for m in matches if m.get("match_id") not in existing_ids]
        print(f"  Need details for {len(needed)} new matches")

        if needed:
            new_details = []
            for i, m in enumerate(needed):
                match_id = m["match_id"]
                detail = download_match_detail(match_id)
                if detail:
                    enriched = enrich_match_with_details(m, detail)
                    new_details.append(enriched)
                else:
                    m["players"] = []
                    new_details.append(m)

                if (i + 1) % 100 == 0:
                    print(
                        f"    Progress: {i + 1}/{len(needed)} "
                        f"({len([d for d in new_details if d.get('players')])} with player data)"
                    )
                    all_details = existing_details + new_details
                    with open(detail_file, "w", encoding="utf-8") as f:
                        json.dump(all_details, f)

                time.sleep(0.15)

            all_details = existing_details + new_details
            with open(detail_file, "w", encoding="utf-8") as f:
                json.dump(all_details, f)
            with_player_data = len([d for d in all_details if d.get("players")])
            print(f"  Saved {len(all_details)} details ({with_player_data} with player data)")
        else:
            print("  All details already downloaded")

    print(f"\n{'=' * 50}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'=' * 50}")
    total_matches = 0
    total_details = 0
    for key in tournaments:
        list_file = RAW_DIR / f"{key}_matchlist.json"
        detail_file = RAW_DIR / f"{key}_details.json"
        n_list = len(json.load(open(list_file, encoding="utf-8"))) if list_file.exists() else 0
        n_detail = len(json.load(open(detail_file, encoding="utf-8"))) if detail_file.exists() else 0
        print(f"  {key:35s}: {n_list:4d} matches, {n_detail:4d} details")
        total_matches += n_list
        total_details += n_detail
    print(f"  {'TOTAL':35s}: {total_matches:4d} matches, {total_details:4d} details")
    print(f"  Registry leagues: {len(TOURNAMENTS)}")


if __name__ == "__main__":
    main()
