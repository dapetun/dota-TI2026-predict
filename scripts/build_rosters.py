"""Build TI 2026 roster account_id mapping from OpenDota + known nick overrides."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.data_collection.match_details import load_player_matches
from src.data_collection.match_loader import load_raw_matchlists
from src.ti2026.pairwise import resolve_opendota_team_ids, latest_lineup_for_team
from src.ti2026.teams import TI2026_TEAMS, get_team_ids

OUT = BASE / "data" / "ti2026_rosters.json"

# Curated nick → OpenDota account_id (pro scene; override auto lineup when known).
# Sources: OpenDota / Liquipedia cross-checks. 0 = unknown (keep lineup order).
KNOWN_NICK_IDS: dict[str, dict[str, int]] = {
    "Aurora": {
        "Nightfall": 171262902,
        "Mikoto": 312436974,
        "Ws": 56351509,
        "Mira": 93817671,
        "kaori": 103735745,
    },
    "BetBoom": {
        "Kiritych~": 172099728,
        "gpk": 480412663,
        "MieRo": 165564598,
        "Save-": 317880638,
        "Kataomi`": 196878136,
    },
    "Falcons": {
        "skiter": 10366616,
        "Malr1ne": 100058342,
        "ATF": 898455820,
        "Cr1t-": 183719386,
        "Sneyking": 25907144,
    },
    "Liquid": {
        "miCKe": 16497807,
        "Nisha": 97590558,
        "Ace": 201358612,
        "Boxi": 77490514,
        "tOfu": 152962063,
    },
    "1w": {
        "Pure": 86698277,
        "bzm": 346412363,
        "33": 136829091,
        "Ari": 93618577,
        "Whitemon": 331855530,
    },
    "Xtreme": {
        "Ame": 88585077,
        "NothingToSay": 137129583,
        "Xxs": 118134220,
        "fy": 157475523,
        "xNova": 111114687,
    },
    "Yandex": {
        "watson": 154715080,
        "CHIRA_JUNIOR": 97658618,
        "DM": 126212866,
        "Saksa": 241884166,
        "Malady": 94155156,
    },
    "Spirit": {
        "Yatoro": 321580662,
        "Larl": 302214028,
        "Collapse": 106305042,
        "not me": 218231587,
        "rue": 847565596,
    },
    "Vision": {
        "Satanic": 1044002267,
        "No[o]ne": 106573901,
        "Noticed": 195108598,
        "9Class": 164199202,
        "Dukalis": 73401082,
    },
    "HULIGANI": {
        "ssnovv1": 372105535,
        "Mirage": 126598297,
        "Vazya": 228517469,
        "sayuw": 343084576,
        "RESPECT": 103110865,
    },
    "Nigma": {
        "SumaiL": 111620041,
        "lorenof": 152168157,
        "Davai": 140297552,
        "OmaR": 101356886,
        "GH": 138880576,
    },
    "Resilience": {
        "Erika": 58429537,
        "EchozZ": 193815691,
        "niu": 104758571,
        "planet": 184138153,
        "zzq": 392565237,
    },
    "Vici": {
        "shiro": 134556694,
        "Xm": 400709028,
        "Bach": 368917218,
        "XinQ": 326327879,
        "y`": 214853734,
    },
    "OG": {
        "Natsumi": 324277900,
        "Yopaj": 355168766,
        "Raven": 100594231,
        "TIMS": 132309493,
        "skem": 155494381,
    },
    "GamerLegion": {
        "Ghost": 206642367,
        "RCY": 154974246,
        "Fayde": 90423751,
        "Bignum": 191362875,
        "Speeed": 160119017,
    },
    "LGD": {
        "Yuma": 173978074,
        "TaiLung": 145957968,
        "Wisper": 150961567,
        "Thiolicor": 87013194,
        "KJ": 221666026,
    },
}


def build_rosters() -> dict:
    """Map TI team_id -> account_ids and nick→account mapping."""
    matches = load_raw_matchlists(BASE / "data" / "raw")
    players = load_player_matches(BASE / "data" / "raw")
    team_ids = get_team_ids()
    odota = resolve_opendota_team_ids(matches, team_ids)

    # Prefer existing file open_dota_team_id overrides when present.
    existing: dict = {}
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            existing = (json.load(f).get("teams") or {})

    teams_out: dict = {}
    for tid in team_ids:
        prev = existing.get(tid) or {}
        oid = int(prev.get("open_dota_team_id") or odota.get(tid, 0) or 0)
        lineup = latest_lineup_for_team(players, oid) if oid else []
        known = KNOWN_NICK_IDS.get(tid, {})
        nicks = TI2026_TEAMS[tid]["roster"]
        nick_map: dict[str, int] = {}
        for nick in nicks:
            nick_map[nick] = int(known.get(nick) or 0)
        # Prefer curated nick order when all mapped; else latest lineup.
        curated_ids = [nick_map[n] for n in nicks if nick_map[n] > 0]
        if len(curated_ids) >= 5:
            account_ids = curated_ids[:5]
        elif curated_ids:
            # Fill missing slots from lineup without duplicates.
            seen = set(curated_ids)
            account_ids = list(curated_ids)
            for aid in lineup:
                if aid not in seen:
                    account_ids.append(aid)
                    seen.add(aid)
                if len(account_ids) >= 5:
                    break
        else:
            account_ids = lineup[:5]

        teams_out[tid] = {
            "open_dota_team_id": oid,
            "account_ids": account_ids,
            "nick_to_account": nick_map,
            "roster_nicks": nicks,
        }

    payload = {
        "generated_from": "scripts/build_rosters.py",
        "note": "nick_to_account curated where known; account_ids prefer curated order",
        "teams": teams_out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    mapped = sum(
        1 for t in teams_out.values() for v in t["nick_to_account"].values() if v > 0
    )
    print(f"Wrote {OUT} ({mapped}/80 nick mappings)")
    return payload


if __name__ == "__main__":
    build_rosters()
