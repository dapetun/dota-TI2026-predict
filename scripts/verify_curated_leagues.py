"""Verify curated OpenDota league IDs and dump n_matches / date span."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.opendota.com/api"
OUT = Path(__file__).resolve().parent.parent / "data" / "league_candidates.json"

# Curated candidates (Liquipedia / Dotabuff / prior verify). Keys are tentative.
CURATED: dict[str, tuple[int, str, str, float, bool]] = {
    # key: (league_id, display, tier, weight, is_lan)
    # --- already in TOURNAMENTS (re-verify) ---
    "TI10_2021": (13256, "The International 2021", "ti", 2.0, True),
    "TI11_2022": (14268, "The International 2022", "ti", 2.0, True),
    "TI12_2023": (15728, "The International 2023", "ti", 2.0, True),
    "TI13_2024": (16935, "The International 2024", "ti", 2.0, True),
    "TI14_2025": (18324, "The International 2025", "ti", 2.0, True),
    "EWC_2026": (19785, "Esports World Cup 2026", "major", 1.5, True),
    "DreamLeague_S29": (19696, "DreamLeague S29", "major", 1.5, True),
    "DreamLeague_S28": (19269, "DreamLeague S28", "major", 1.5, True),
    "PGL_Wallachia_S8": (19543, "PGL Wallachia S8", "major", 1.5, True),
    "BLAST_SLAM_VII": (19101, "BLAST SLAM VII", "major", 1.5, True),
    "BLAST_SLAM_VI": (19099, "BLAST SLAM VI", "major", 1.5, True),
    "ESL_Birmingham_2026": (19422, "ESL One Birmingham 2026", "major", 1.5, True),
    # --- majors / LAN 2023–2025 ---
    "Riyadh_Masters_2023": (15438, "Riyadh Masters 2023", "major", 1.5, True),
    "Riyadh_Masters_2024": (16918, "Riyadh Masters 2024", "major", 1.5, True),
    "Riyadh_Masters_2025": (18390, "Riyadh Masters 2025 / EWC", "major", 1.5, True),
    "Bali_Major_2023": (15439, "Bali Major 2023", "major", 1.5, True),
    "Berlin_Major_2023": (15251, "Berlin Major 2023", "major", 1.5, True),
    "Lima_Major_2023": (15246, "Lima Major 2023", "major", 1.5, True),
    "DreamLeague_S21": (15739, "DreamLeague S21", "major", 1.5, True),
    "DreamLeague_S22": (15983, "DreamLeague S22", "major", 1.5, True),
    "DreamLeague_S23": (16483, "DreamLeague S23", "major", 1.5, True),
    "DreamLeague_S24": (16935, "DreamLeague S24", "major", 1.5, True),  # may collide TI13
    "DreamLeague_S25": (17417, "DreamLeague S25", "major", 1.5, True),
    "DreamLeague_S26": (17835, "DreamLeague S26", "major", 1.5, True),
    "DreamLeague_S27": (18300, "DreamLeague S27", "major", 1.5, True),
    "ESL_Birmingham_2024": (16484, "ESL One Birmingham 2024", "major", 1.5, True),
    "ESL_Birmingham_2025": (17891, "ESL One Birmingham 2025", "major", 1.5, True),
    "PGL_Wallachia_S1": (16485, "PGL Wallachia S1", "major", 1.5, True),
    "PGL_Wallachia_S2": (16901, "PGL Wallachia S2", "major", 1.5, True),
    "PGL_Wallachia_S3": (17418, "PGL Wallachia S3", "major", 1.5, True),
    "PGL_Wallachia_S4": (17892, "PGL Wallachia S4", "major", 1.5, True),
    "PGL_Wallachia_S5": (18325, "PGL Wallachia S5", "major", 1.5, True),
    "PGL_Wallachia_S6": (18600, "PGL Wallachia S6", "major", 1.5, True),
    "PGL_Wallachia_S7": (18950, "PGL Wallachia S7", "major", 1.5, True),
    "BLAST_SLAM_I": (17419, "BLAST Slam I", "major", 1.5, True),
    "BLAST_SLAM_II": (17893, "BLAST Slam II", "major", 1.5, True),
    "BLAST_SLAM_III": (18326, "BLAST Slam III", "major", 1.5, True),
    "BLAST_SLAM_IV": (18601, "BLAST Slam IV", "major", 1.5, True),
    "BLAST_SLAM_V": (18951, "BLAST Slam V", "major", 1.5, True),
    # --- TI quals ---
    "TI12_EU_Qual": (15740, "TI12 EU Qualifier", "qual", 0.75, False),
    "TI12_CN_Qual": (15741, "TI12 CN Qualifier", "qual", 0.75, False),
    "TI12_SEA_Qual": (15742, "TI12 SEA Qualifier", "qual", 0.75, False),
    "TI12_NA_Qual": (15743, "TI12 NA Qualifier", "qual", 0.75, False),
    "TI12_SA_Qual": (15744, "TI12 SA Qualifier", "qual", 0.75, False),
    "TI13_EU_Qual": (16936, "TI13 EU Qualifier", "qual", 0.75, False),
    "TI13_CN_Qual": (16937, "TI13 CN Qualifier", "qual", 0.75, False),
    "TI13_SEA_Qual": (16938, "TI13 SEA Qualifier", "qual", 0.75, False),
    "TI13_NA_Qual": (16939, "TI13 NA Qualifier", "qual", 0.75, False),
    "TI13_SA_Qual": (16940, "TI13 SA Qualifier", "qual", 0.75, False),
    "TI14_EU_Qual": (18330, "TI14 EU Qualifier", "qual", 0.75, False),
    "TI14_CN_Qual": (18331, "TI14 CN Qualifier", "qual", 0.75, False),
    "TI14_SEA_Qual": (18332, "TI14 SEA Qualifier", "qual", 0.75, False),
    "TI14_NA_Qual": (18333, "TI14 NA Qualifier", "qual", 0.75, False),
    "TI14_SA_Qual": (18334, "TI14 SA Qualifier", "qual", 0.75, False),
    "TI15_EU_Qual": (19720, "TI15/2026 EU Qualifier", "qual", 0.75, False),
    "TI15_CN_Qual": (19721, "TI15/2026 CN Qualifier", "qual", 0.75, False),
    "TI15_SEA_Qual": (19722, "TI15/2026 SEA Qualifier", "qual", 0.75, False),
    "TI15_NA_Qual": (19723, "TI15/2026 NA Qualifier", "qual", 0.75, False),
    "TI15_SA_Qual": (19724, "TI15/2026 SA Qualifier", "qual", 0.75, False),
    # --- online / mid-tier ---
    "BetBoom_Dacha_Belgrade_2024": (16486, "BetBoom Dacha Belgrade 2024", "online", 0.5, True),
    "FISSURE_Universe_Ep3": (17420, "FISSURE Universe Ep3", "online", 0.5, False),
    "FISSURE_Universe_Ep4": (17894, "FISSURE Universe Ep4", "online", 0.5, False),
    "Clavision_S2": (16941, "Clavision Masters S2", "online", 0.5, False),
}


def probe(league_id: int) -> dict:
    session = requests.Session()
    meta = {}
    try:
        r = session.get(f"{BASE}/leagues/{league_id}", timeout=30)
        if r.status_code == 200:
            meta = r.json()
    except Exception as exc:  # noqa: BLE001
        meta = {"error": str(exc)}
    matches: list = []
    try:
        r = session.get(f"{BASE}/leagues/{league_id}/matches", timeout=45)
        if r.status_code == 200:
            matches = r.json() if isinstance(r.json(), list) else []
        elif r.status_code == 429:
            time.sleep(65)
            r = session.get(f"{BASE}/leagues/{league_id}/matches", timeout=45)
            matches = r.json() if r.status_code == 200 and isinstance(r.json(), list) else []
    except Exception as exc:  # noqa: BLE001
        meta["match_error"] = str(exc)
    times = [int(m["start_time"]) for m in matches if m.get("start_time")]
    dmin = dmax = None
    if times:
        dmin = datetime.fromtimestamp(min(times), tz=timezone.utc).strftime("%Y-%m-%d")
        dmax = datetime.fromtimestamp(max(times), tz=timezone.utc).strftime("%Y-%m-%d")
    return {
        "api_name": meta.get("name"),
        "tier_api": meta.get("tier"),
        "n_matches": len(matches),
        "date_min": dmin,
        "date_max": dmax,
        "ok": bool(meta.get("name")) and len(matches) > 0,
    }


def main() -> None:
    rows = []
    for key, (lid, name, tier, weight, is_lan) in CURATED.items():
        print(f"Probing {key} ({lid})...")
        info = probe(lid)
        row = {
            "key": key,
            "league_id": lid,
            "planned_name": name,
            "suggested_tier": tier,
            "suggested_weight": weight,
            "is_lan": is_lan,
            **info,
        }
        rows.append(row)
        status = "OK" if info["ok"] else "MISS"
        print(
            f"  {status} api={info.get('api_name')!r} n={info['n_matches']} "
            f"{info.get('date_min')}..{info.get('date_max')}"
        )
        time.sleep(0.45)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "scripts/verify_curated_leagues.py",
        "n_ok": sum(1 for r in rows if r["ok"]),
        "candidates": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT} ok={payload['n_ok']}/{len(rows)}")


if __name__ == "__main__":
    main()
