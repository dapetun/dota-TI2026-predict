"""Build TI 2026 roster account_id mapping from OpenDota player history."""

from __future__ import annotations

import json
from pathlib import Path

from src.data_collection.match_details import load_player_matches
from src.data_collection.match_loader import load_raw_matchlists
from src.ti2026.pairwise import resolve_opendota_team_ids, latest_lineup_for_team
from src.ti2026.teams import TI2026_TEAMS, get_team_ids

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "ti2026_rosters.json"


def build_rosters() -> dict:
    """Map TI team_id -> account_ids and nick hints from latest lineups."""
    matches = load_raw_matchlists(BASE / "data" / "raw")
    players = load_player_matches(BASE / "data" / "raw")
    team_ids = get_team_ids()
    odota = resolve_opendota_team_ids(matches, team_ids)

    teams_out: dict = {}
    for tid in team_ids:
        oid = odota.get(tid, 0)
        ids = latest_lineup_for_team(players, oid) if oid else []
        nick_map: dict[str, int] = {}
        nicks = TI2026_TEAMS[tid]["roster"]
        for nick in nicks:
            nick_map[nick] = 0
        teams_out[tid] = {
            "open_dota_team_id": oid,
            "account_ids": ids,
            "nick_to_account": nick_map,
            "roster_nicks": nicks,
        }

    payload = {
        "generated_from": "scripts/build_rosters.py",
        "note": "account_ids from latest OpenDota lineup; override nick_to_account manually if needed",
        "teams": teams_out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUT}")
    return payload


if __name__ == "__main__":
    build_rosters()
