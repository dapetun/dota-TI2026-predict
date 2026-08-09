"""Build data/hero/hero_meta.json from OpenDota heroStats (optional API)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.features.hero_meta import build_hero_meta_from_hero_stats, save_hero_meta


def main(argv: list[str] | None = None) -> int:
    """Fetch OpenDota hero meta; fixture copy requires explicit --allow-fixture."""
    parser = argparse.ArgumentParser(description="Build hero_meta.json")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=BASE / "tests" / "fixtures" / "hero" / "hero_meta.json",
        help="Offline fixture path (only with --allow-fixture)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BASE / "data" / "hero" / "hero_meta.json",
    )
    parser.add_argument("--fetch", action="store_true", help="Call OpenDota /heroStats")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Fetch attempts before giving up (default 3)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=BASE / "data" / "hero" / "hero_stats_cache.json",
        help="Cache last successful OpenDota heroStats payload",
    )
    parser.add_argument(
        "--allow-fixture",
        action="store_true",
        help="Allow writing test fixture into data/ (not for production soft prior)",
    )
    args = parser.parse_args(argv)

    if args.fetch:
        last_exc: Exception | None = None
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                from src.data_collection.opendota_api import OpenDotaClient

                client = OpenDotaClient()
                stats = client._get("/heroStats")  # noqa: SLF001 — thin public wrapper absent
                if not isinstance(stats, list):
                    raise RuntimeError("unexpected heroStats payload")
                args.cache.parent.mkdir(parents=True, exist_ok=True)
                args.cache.write_text(
                    json.dumps(stats, ensure_ascii=False),
                    encoding="utf-8",
                )
                data = build_hero_meta_from_hero_stats(stats)
                data["usable_in_production"] = True
                data["source"] = "opendota_heroStats"
                save_hero_meta(data, args.out)
                print(
                    f"Fetched {len(data.get('heroes', {}))} heroes -> {args.out} "
                    f"(attempt {attempt})"
                )
                return 0
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(f"Fetch attempt {attempt}/{args.retries} failed ({exc})")
        if args.cache.exists():
            try:
                stats = json.loads(args.cache.read_text(encoding="utf-8"))
                if isinstance(stats, list) and stats:
                    data = build_hero_meta_from_hero_stats(stats)
                    data["usable_in_production"] = True
                    data["source"] = "opendota_heroStats_cache"
                    save_hero_meta(data, args.out)
                    print(f"Used cache {args.cache} -> {args.out}")
                    return 0
            except (OSError, json.JSONDecodeError, TypeError) as cache_exc:
                print(f"Cache unusable ({cache_exc})")
        print(f"Fetch failed after retries ({last_exc})")
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

    raw = json.loads(args.fixture.read_text(encoding="utf-8"))
    raw["usable_in_production"] = False
    if "fixture" not in str(raw.get("source", "")).lower():
        raw["source"] = "fixture"
    save_hero_meta(raw, args.out)
    print(f"Wrote fixture hero_meta (usable_in_production=false) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
