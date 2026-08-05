"""Download match data using correct OpenDota endpoints.

Two-phase approach:
1. /leagues/{id}/matches → match list (fast, ~50ms each)
2. /matches/{match_id} → full detail with player data (slower)
"""
import requests
import json
import time
import os
import sys
from pathlib import Path

BASE_URL = "https://api.opendota.com/api"

TOURNAMENTS = {
    "TI10_2021":       {"league_id": 13256, "name": "The International 2021", "year": 2021, "tier": "ti"},
    "TI11_2022":       {"league_id": 14268, "name": "The International 2022", "year": 2022, "tier": "ti"},
    "TI12_2023":       {"league_id": 15728, "name": "The International 2023", "year": 2023, "tier": "ti"},
    "TI13_2024":       {"league_id": 16935, "name": "The International 2024", "year": 2024, "tier": "ti"},
    "TI14_2025":       {"league_id": 18324, "name": "The International 2025", "year": 2025, "tier": "ti"},
    "EWC_2026":         {"league_id": 19785, "name": "Esports World Cup 2026", "year": 2026, "tier": "major"},
    "DreamLeague_S29":  {"league_id": 19696, "name": "DreamLeague S29", "year": 2026, "tier": "major"},
    "DreamLeague_S28":  {"league_id": 19269, "name": "DreamLeague S28", "year": 2026, "tier": "major"},
    "PGL_Wallachia_S8": {"league_id": 19543, "name": "PGL Wallachia S8", "year": 2026, "tier": "major"},
    "BLAST_SLAM_VII":   {"league_id": 19101, "name": "BLAST SLAM VII", "year": 2026, "tier": "major"},
    "BLAST_SLAM_VI":    {"league_id": 19099, "name": "BLAST SLAM VI", "year": 2026, "tier": "major"},
    "ESL_Birmingham_2026": {"league_id": 19422, "name": "ESL One Birmingham 2026", "year": 2026, "tier": "major"},
}

# Download order: recent first
DOWNLOAD_ORDER = [
    "EWC_2026", "DreamLeague_S29", "PGL_Wallachia_S8", "BLAST_SLAM_VII",
    "BLAST_SLAM_VI", "ESL_Birmingham_2026", "DreamLeague_S28",
    "TI14_2025", "TI13_2024", "TI12_2023", "TI11_2022", "TI10_2021",
]


def download_league_match_list(league_id):
    """Download match list using /leagues/{id}/matches endpoint."""
    url = f"{BASE_URL}/leagues/{league_id}/matches"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            print(f"  Rate limited, waiting 60s...")
            time.sleep(60)
            return download_league_match_list(league_id)
        else:
            print(f"  Error {resp.status_code}: {resp.text[:100]}")
            return []
    except Exception as e:
        print(f"  Exception: {e}")
        return []


def download_match_detail(match_id, retries=3):
    """Download full match detail from /matches/{id}."""
    for attempt in range(retries):
        try:
            resp = requests.get(f"{BASE_URL}/matches/{match_id}", timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print(f"    Rate limited at match {match_id}, waiting 60s...")
                time.sleep(60)
            else:
                time.sleep(0.2)
        except Exception:
            time.sleep(1)
    return None


def enrich_match_with_details(match, detail):
    """Merge league match data with full match detail."""
    if not detail:
        return match

    players = detail.get("players", [])
    enriched_players = []
    for i, p in enumerate(players):
        # Players 0-4 = radiant, 5-9 = dire
        is_radiant = i < 5
        team_id = match.get("radiant_team_id") if is_radiant else match.get("dire_team_id")
        team_won = match.get("radiant_win", False) if is_radiant else not match.get("radiant_win", False)

        # Handle healing dict
        healing = p.get("healing", 0)
        if isinstance(healing, dict):
            healing = sum(healing.values())

        enriched_players.append({
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
        })

    match["players"] = enriched_players
    return match


def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse args
    detail_only = "--detail-only" in sys.argv
    list_only = "--list-only" in sys.argv
    specific = None
    for arg in sys.argv[1:]:
        if arg not in ("--detail-only", "--list-only") and arg in TOURNAMENTS:
            specific = arg

    tournaments = [specific] if specific else DOWNLOAD_ORDER

    for key in tournaments:
        tourn = TOURNAMENTS[key]
        lid = tourn["league_id"]
        name = tourn["name"]
        print(f"\n{'='*50}")
        print(f"  {name} (league {lid})")
        print(f"{'='*50}")

        # Phase 1: Download match list
        list_file = output_dir / f"{key}_matchlist.json"
        if not list_file.exists() and not detail_only:
            print(f"  Downloading match list...")
            matches = download_league_match_list(lid)
            print(f"  Found {len(matches)} matches")

            with open(list_file, "w") as f:
                json.dump(matches, f)
            time.sleep(0.5)
        else:
            with open(list_file) as f:
                matches = json.load(f)
            print(f"  Already have {len(matches)} matches in list")

        if list_only:
            continue

        # Phase 2: Download match details
        detail_file = output_dir / f"{key}_details.json"
        if detail_file.exists():
            with open(detail_file) as f:
                existing_details = json.load(f)
            existing_ids = {d["match_id"] for d in existing_details if d}
            print(f"  Already have {len(existing_details)} details, checking for missing...")
        else:
            existing_details = []
            existing_ids = set()

        # Filter to matches we need details for
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
                    # Still save match list data even without detail
                    m["players"] = []
                    new_details.append(m)

                if (i + 1) % 100 == 0:
                    print(f"    Progress: {i+1}/{len(needed)} ({len([d for d in new_details if d.get('players')])} with player data)")
                    # Save intermediate
                    all_details = existing_details + new_details
                    with open(detail_file, "w") as f:
                        json.dump(all_details, f)

                time.sleep(0.15)  # Rate limit: ~6 req/s

            all_details = existing_details + new_details
            with open(detail_file, "w") as f:
                json.dump(all_details, f)
            with_player_data = len([d for d in all_details if d.get("players")])
            print(f"  Saved {len(all_details)} details ({with_player_data} with player data)")
        else:
            print(f"  All details already downloaded")

    # Summary
    print(f"\n{'='*50}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'='*50}")
    total_matches = 0
    total_details = 0
    for key in tournaments:
        list_file = output_dir / f"{key}_matchlist.json"
        detail_file = output_dir / f"{key}_details.json"
        n_list = len(json.load(open(list_file))) if list_file.exists() else 0
        n_detail = len(json.load(open(detail_file))) if detail_file.exists() else 0
        n_with_players = len([d for d in (json.load(open(detail_file)) if detail_file.exists() else []) if d.get("players")])
        print(f"  {key:25s}: {n_list:4d} matches, {n_detail:4d} details, {n_with_players:4d} with players")
        total_matches += n_list
        total_details += n_detail
    print(f"  {'TOTAL':25s}: {total_matches:4d} matches, {total_details:4d} details")


if __name__ == "__main__":
    main()
