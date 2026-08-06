"""Fetch one missing match detail into match_details.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.download_details import _atomic_write_json, _has_players
from src.data_collection.opendota_api import OpenDotaClient

BASE_DIR = Path(__file__).resolve().parent.parent
DETAILS = BASE_DIR / "data" / "raw" / "match_details.json"


def fetch_one(match_id: int) -> bool:
    """Download a single match detail if players are available."""
    existing: dict = {}
    if DETAILS.exists():
        with open(DETAILS, encoding="utf-8") as f:
            existing = json.load(f)
    key = str(match_id)
    if key in existing and _has_players(existing[key]):
        print(f"Already cached: {match_id}")
        return True

    client = OpenDotaClient(rate_limit=1.1)
    detail = client.get_match(match_id)
    if not detail or "error" in detail or not _has_players(detail):
        print(f"Failed to fetch usable detail for {match_id}")
        return False
    existing[key] = detail
    _atomic_write_json(DETAILS, existing)
    print(f"Saved match {match_id} (total {len(existing)})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch one OpenDota match detail")
    parser.add_argument("match_id", type=int, nargs="?", default=7380112773)
    args = parser.parse_args()
    fetch_one(args.match_id)


if __name__ == "__main__":
    main()
