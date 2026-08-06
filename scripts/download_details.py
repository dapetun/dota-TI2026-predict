"""Resume-friendly OpenDota match-detail downloader (priority: recent TI/majors)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data_collection.opendota_api import OpenDotaClient
from src.data_collection.tournaments import TOURNAMENTS

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

# Recent / high-value first — enough for a working player model quickly.
DOWNLOAD_PRIORITY = [
    "EWC_2026",
    "DreamLeague_S29",
    "DreamLeague_S28",
    "PGL_Wallachia_S8",
    "BLAST_SLAM_VII",
    "BLAST_SLAM_VI",
    "ESL_Birmingham_2026",
    "TI14_2025",
    "TI13_2024",
    "TI12_2023",
    "TI11_2022",
    "TI10_2021",
]


def _match_ids_for_tournament(key: str) -> list[int]:
    list_file = RAW_DIR / f"{key}_matchlist.json"
    alt = RAW_DIR / f"{key}_matches.json"
    path = list_file if list_file.exists() else alt
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        matches = json.load(f)
    ids: list[int] = []
    for m in matches:
        mid = m.get("match_id")
        if mid:
            ids.append(int(mid))
    return ids


def _load_existing(details_file: Path) -> dict[str, dict]:
    if not details_file.exists():
        return {}
    with open(details_file, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return {str(k): v for k, v in payload.items() if isinstance(v, dict)}
    if isinstance(payload, list):
        out: dict[str, dict] = {}
        for item in payload:
            if isinstance(item, dict) and item.get("match_id") is not None:
                out[str(item["match_id"])] = item
        return out
    return {}


def _has_players(detail: dict | None) -> bool:
    return bool(detail and isinstance(detail.get("players"), list) and detail["players"])


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically so readers never see a half-written file."""
    import os

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    # Windows may lock the target while another process reads it.
    last_err: Exception | None = None
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    # Fallback: overwrite in place (still better than crashing the downloader).
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        raise last_err or exc


def download_details(
    *,
    max_new: int | None = None,
    tournaments: list[str] | None = None,
    rate_limit: float = 1.1,
) -> dict:
    """Download missing match details into data/raw/match_details.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    details_file = RAW_DIR / "match_details.json"
    existing = _load_existing(details_file)
    client = OpenDotaClient(rate_limit=rate_limit)

    order = tournaments or [k for k in DOWNLOAD_PRIORITY if k in TOURNAMENTS]
    needed: list[int] = []
    seen: set[int] = set()
    for key in order:
        for mid in _match_ids_for_tournament(key):
            if mid in seen:
                continue
            seen.add(mid)
            cur = existing.get(str(mid))
            if _has_players(cur):
                continue
            needed.append(mid)

    if max_new is not None:
        needed = needed[: max(0, max_new)]

    print(f"Cached details: {len(existing)}")
    print(f"Need download: {len(needed)}")
    if not needed:
        return {"cached": len(existing), "downloaded": 0, "errors": 0}

    errors = 0
    downloaded = 0
    for i, mid in enumerate(needed):
        try:
            detail = client.get_match(mid)
            if detail and "error" not in detail and _has_players(detail):
                existing[str(mid)] = detail
                downloaded += 1
            else:
                errors += 1
        except Exception as exc:  # noqa: BLE001 — network resilience
            errors += 1
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg:
                print(f"  Rate limited at {mid}, sleeping 65s...")
                time.sleep(65)

        if (i + 1) % 50 == 0 or (i + 1) == len(needed):
            _atomic_write_json(details_file, existing)
            print(
                f"  Progress {i + 1}/{len(needed)} | "
                f"with_players={sum(1 for v in existing.values() if _has_players(v))} | "
                f"errors={errors}"
            )

    _atomic_write_json(details_file, existing)

    with_players = sum(1 for v in existing.values() if _has_players(v))
    print(f"Done. cached={len(existing)} with_players={with_players} new={downloaded} errors={errors}")
    return {
        "cached": len(existing),
        "with_players": with_players,
        "downloaded": downloaded,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenDota match details")
    parser.add_argument("--max", type=int, default=None, help="Max new matches to fetch")
    parser.add_argument(
        "--tournaments",
        nargs="*",
        default=None,
        help="Optional subset of tournament keys",
    )
    args = parser.parse_args()
    download_details(max_new=args.max, tournaments=args.tournaments)


if __name__ == "__main__":
    main()
