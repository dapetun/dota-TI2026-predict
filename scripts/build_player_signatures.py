"""Build data/hero/player_signatures.json from OpenDota or fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.features.hero_meta import load_hero_meta
from src.features.player_signatures import (
    build_player_signature,
    save_player_signatures,
    signatures_from_fixture_rows,
)


def _roster_account_ids() -> list[int]:
    path = BASE / "data" / "ti2026_rosters.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    ids: list[int] = []
    for team in (data.get("teams") or {}).values():
        for aid in team.get("account_ids") or []:
            if int(aid) > 0:
                ids.append(int(aid))
    return sorted(set(ids))


def main(argv: list[str] | None = None) -> int:
    """Fetch player heroes; fixture copy requires explicit --allow-fixture."""
    parser = argparse.ArgumentParser(description="Build player_signatures.json")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=BASE / "tests" / "fixtures" / "hero" / "player_hero_rows.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BASE / "data" / "hero" / "player_signatures.json",
    )
    parser.add_argument("--fetch", action="store_true", help="Call OpenDota /players/{id}/heroes")
    parser.add_argument(
        "--allow-fixture",
        action="store_true",
        help="Allow writing test fixture into data/ (not for production soft prior)",
    )
    args = parser.parse_args(argv)

    meta_path = BASE / "data" / "hero" / "hero_meta.json"
    if not meta_path.exists():
        print(f"Missing {meta_path}; run scripts/build_hero_meta.py --fetch first")
        return 1
    meta = load_hero_meta(meta_path)

    if args.fetch:
        try:
            from src.data_collection.opendota_api import OpenDotaClient

            client = OpenDotaClient()
            players: dict[str, dict] = {}
            for aid in _roster_account_ids():
                rows = client._get(f"/players/{aid}/heroes")  # noqa: SLF001
                if not isinstance(rows, list):
                    continue
                players[str(aid)] = build_player_signature(aid, rows, meta)
            data = {
                "source": "opendota_players_heroes",
                "usable_in_production": True,
                "note": "Experimental; pub WR ≠ pro performance.",
                "players": players,
            }
            save_player_signatures(data, args.out)
            print(f"Fetched {len(players)} players -> {args.out}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"Fetch failed ({exc})")
            if not args.allow_fixture:
                print("Refusing fixture fallback without --allow-fixture")
                return 1
            print("Falling back to fixture (--allow-fixture)")

    if not args.allow_fixture and not args.fetch:
        print(
            "Refusing to write test fixture into data/. "
            "Pass --fetch (OpenDota) or --allow-fixture (tests only)."
        )
        return 1

    rows = json.loads(args.fixture.read_text(encoding="utf-8"))
    data = signatures_from_fixture_rows(rows.get("players") or {}, meta)
    data["source"] = "fixture"
    data["usable_in_production"] = False
    data["note"] = "Test fixture only — not for production soft prior."
    save_player_signatures(data, args.out)
    print(f"Wrote fixture signatures (usable_in_production=false) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
