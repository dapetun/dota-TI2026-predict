"""Discover OpenDota leagues for expanding the training corpus.

Filters: name heuristics, optional date window via matchlist probes,
minimum n_matches, and tier tags (premium/professional preferred).
Writes candidates to data/league_candidates.json for review → TOURNAMENTS.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.opendota.com/api"
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "league_candidates.json"

# Name tokens → suggested tier / weight class for curation.
NAME_RULES: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"the international|\bti\s?\d{2}\b", re.I), "ti", 2.0),
    (re.compile(r"qualifier|closed qual|open qual", re.I), "qual", 0.75),
    (re.compile(r"\bdpc\b|regional", re.I), "dpc", 0.75),
    (
        re.compile(
            r"dreamleague|pgl|wallachia|blast|esl one|riyadh|"
            r"esports world cup|\bewc\b|bali major|lima major|"
            r"stockholm major|berlin major|arlington|singapore major",
            re.I,
        ),
        "major",
        1.5,
    ),
    (
        re.compile(
            r"betboom|fissure|dacha|clavision|mesports|"
            r"universe|premier|dreamleague season",
            re.I,
        ),
        "online",
        0.5,
    ),
]

SKIP_NAME = re.compile(
    r"amateur|immortal|pub|ranked|faceit|join.?dota|open cup|"
    r"school|university|high.?school|women|female",
    re.I,
)


def _guess_meta(name: str) -> tuple[str, float]:
    """Heuristically assign tier key and sample weight."""
    for pat, tier, weight in NAME_RULES:
        if pat.search(name or ""):
            return tier, weight
    return "other", 1.0


def fetch_leagues(session: requests.Session) -> list[dict]:
    """Загрузить полный список лиг OpenDota."""
    resp = session.get(f"{BASE_URL}/leagues", timeout=90)
    resp.raise_for_status()
    return resp.json()


def fetch_matchlist(session: requests.Session, league_id: int) -> list[dict]:
    """Скачать список матчей лиги (может быть пустым)."""
    url = f"{BASE_URL}/leagues/{league_id}/matches"
    for attempt in range(4):
        resp = session.get(url, timeout=60)
        if resp.status_code == 429:
            time.sleep(60)
            continue
        if resp.status_code != 200:
            time.sleep(0.5 * (attempt + 1))
            continue
        data = resp.json()
        return data if isinstance(data, list) else []
    return []


def match_date_window(
    matches: list[dict],
) -> tuple[int | None, int | None, str | None, str | None]:
    """min/max start_time and ISO dates for a matchlist."""
    times = [int(m["start_time"]) for m in matches if m.get("start_time")]
    if not times:
        return None, None, None, None
    tmin, tmax = min(times), max(times)
    dmin = datetime.fromtimestamp(tmin, tz=timezone.utc).strftime("%Y-%m-%d")
    dmax = datetime.fromtimestamp(tmax, tz=timezone.utc).strftime("%Y-%m-%d")
    return tmin, tmax, dmin, dmax


def discover(
    *,
    min_matches: int = 40,
    date_from: str | None = "2023-01-01",
    date_to: str | None = None,
    max_probe: int | None = None,
    rate_limit: float = 0.35,
    probe: bool = True,
) -> dict:
    """Discover and optionally probe candidate leagues."""
    session = requests.Session()
    leagues = fetch_leagues(session)
    print(f"OpenDota leagues: {len(leagues)}")

    ts_from = (
        int(datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc).timestamp())
        if date_from
        else None
    )
    ts_to = (
        int(datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc).timestamp())
        if date_to
        else None
    )

    prefiltered: list[dict] = []
    for L in leagues:
        name = L.get("name") or ""
        if SKIP_NAME.search(name):
            continue
        tier_api = (L.get("tier") or "").lower()
        guess_tier, guess_w = _guess_meta(name)
        # Keep premium/professional always if name matches; else only named hits.
        named = guess_tier != "other"
        if not named and tier_api not in ("premium", "professional"):
            continue
        if not named and tier_api in ("premium", "professional"):
            # Unnamed premium — still interesting for review.
            pass
        prefiltered.append(
            {
                "league_id": int(L["leagueid"]),
                "name": name,
                "tier_api": tier_api,
                "suggested_tier": guess_tier,
                "suggested_weight": guess_w,
                "ticket": L.get("ticket"),
            }
        )

    # Prefer recent league ids and named majors/quals.
    prefiltered.sort(
        key=lambda x: (
            0 if x["suggested_tier"] in ("ti", "major", "qual", "dpc") else 1,
            -x["league_id"],
        )
    )
    if max_probe is not None:
        prefiltered = prefiltered[:max_probe]

    print(f"Prefiltered candidates: {len(prefiltered)}")
    results: list[dict] = []

    if not probe:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "min_matches": min_matches,
                "date_from": date_from,
                "date_to": date_to,
                "probe": False,
            },
            "candidates": prefiltered,
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote {OUT_PATH} ({len(prefiltered)} unprobed)")
        return payload

    for i, cand in enumerate(prefiltered):
        lid = cand["league_id"]
        matches = fetch_matchlist(session, lid)
        time.sleep(rate_limit)
        n = len(matches)
        tmin, tmax, dmin, dmax = match_date_window(matches)
        # Date filter: overlap with [date_from, date_to]
        if ts_from is not None and tmax is not None and tmax < ts_from:
            continue
        if ts_to is not None and tmin is not None and tmin > ts_to:
            continue
        if n < min_matches:
            continue
        row = {
            **cand,
            "n_matches": n,
            "date_min": dmin,
            "date_max": dmax,
            "start_time_min": tmin,
            "start_time_max": tmax,
        }
        results.append(row)
        if (i + 1) % 25 == 0 or (i + 1) == len(prefiltered):
            print(f"  probed {i + 1}/{len(prefiltered)} | kept {len(results)}")

    results.sort(key=lambda x: (-x.get("n_matches", 0), -x["league_id"]))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "min_matches": min_matches,
            "date_from": date_from,
            "date_to": date_to,
            "probe": True,
            "probed": len(prefiltered),
        },
        "n_candidates": len(results),
        "candidates": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUT_PATH} ({len(results)} candidates)")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover OpenDota leagues")
    parser.add_argument("--min-matches", type=int, default=40)
    parser.add_argument("--date-from", default="2023-01-01")
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--max-probe", type=int, default=200)
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--rate", type=float, default=0.35)
    args = parser.parse_args()
    discover(
        min_matches=args.min_matches,
        date_from=args.date_from,
        date_to=args.date_to,
        max_probe=None if args.max_probe <= 0 else args.max_probe,
        rate_limit=args.rate,
        probe=not args.no_probe,
    )


if __name__ == "__main__":
    main()
