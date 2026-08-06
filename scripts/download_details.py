"""Resume-friendly OpenDota match-detail downloader (sharded + legacy monolith).

New downloads write atomic per-match shards under
``data/raw/details_shards/<tournament>/<match_id>.json``.
Existing monolith ``match_details.json`` is still read for coverage/resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_collection.match_details import (  # noqa: E402
    MONOLITH_NAME,
    atomic_write_json,
    shard_path_for_match,
)
from src.data_collection.opendota_api import OpenDotaClient  # noqa: E402
from src.data_collection.tournaments import download_order  # noqa: E402

RAW_DIR = BASE_DIR / "data" / "raw"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_details")


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


def _normalize_details_payload(payload: object) -> dict[str, dict]:
    """dict[match_id→detail] or list of details → keyed dict."""
    if isinstance(payload, dict):
        return {str(k): v for k, v in payload.items() if isinstance(v, dict)}
    if isinstance(payload, list):
        out: dict[str, dict] = {}
        for item in payload:
            if isinstance(item, dict) and item.get("match_id") is not None:
                out[str(item["match_id"])] = item
        return out
    return {}


def salvage_truncated_json_object(raw: str | bytes) -> dict | None:
    """Recover a truncated top-level JSON object up to the last complete value."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    if not text or not text.lstrip().startswith("{"):
        return None

    depth = 0
    in_string = False
    escape = False
    last_good: int | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 1 and ch == "}":
                last_good = i
            elif depth == 0 and ch == "}":
                last_good = i
                break
            elif depth < 0:
                return None
        i += 1

    if last_good is None:
        return None

    candidate = text[: last_good + 1] if depth == 0 else text[: last_good + 1] + "}"
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _load_existing(details_file: Path) -> dict[str, dict]:
    """Load monolith cache; on corrupt/truncated JSON, salvage last complete entries."""
    if not details_file.exists():
        return {}
    try:
        with open(details_file, encoding="utf-8") as f:
            payload = json.load(f)
        return _normalize_details_payload(payload)
    except json.JSONDecodeError as exc:
        logger.warning("corrupt JSON in %s (%s); attempting salvage...", details_file.name, exc)
    except OSError as exc:
        logger.warning("cannot read %s: %s; starting empty", details_file.name, exc)
        return {}

    try:
        raw = details_file.read_bytes()
    except OSError as exc:
        logger.warning("salvage read failed (%s); starting empty", exc)
        return {}

    salvaged_obj = salvage_truncated_json_object(raw)
    if salvaged_obj is None:
        logger.error("corrupt %s, salvage failed — starting empty cache", details_file.name)
        return {}

    salvaged = _normalize_details_payload(salvaged_obj)
    logger.warning("salvaged truncated %s (%s keys)", details_file.name, len(salvaged))
    bak = details_file.with_name(details_file.name + ".corrupt")
    try:
        if not bak.exists():
            details_file.replace(bak)
            logger.info("Moved corrupt file -> %s", bak.name)
        else:
            details_file.unlink(missing_ok=True)
        atomic_write_json(details_file, salvaged)
        logger.info("Rewrote repaired %s", details_file.name)
    except OSError as exc:
        logger.warning("could not persist repaired cache (%s); using in-memory salvage", exc)
    return salvaged


def _has_players(detail: dict | None) -> bool:
    return bool(detail and isinstance(detail.get("players"), list) and detail["players"])


def _shard_has_players(raw_dir: Path, match_id: int, tournament_key: str) -> bool:
    path = shard_path_for_match(raw_dir, match_id, tournament_key)
    if not path.exists():
        # Also check _unknown (league not tagged)
        alt = shard_path_for_match(raw_dir, match_id, None)
        path = alt if alt.exists() else path
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            detail = json.load(f)
        return _has_players(detail if isinstance(detail, dict) else None)
    except (OSError, json.JSONDecodeError):
        return False


def download_details(
    *,
    max_new: int | None = None,
    tournaments: list[str] | None = None,
    rate_limit: float = 1.1,
    write_monolith: bool = False,
) -> dict:
    """Download missing match details into shards (optionally sync monolith)."""
    logger.info("Loading existing details cache...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    details_file = RAW_DIR / MONOLITH_NAME
    existing = _load_existing(details_file)
    logger.info("Monolith cached details: %s", len(existing))
    client = OpenDotaClient(rate_limit=rate_limit)

    order = tournaments or download_order()
    logger.info("Scanning %s tournaments for missing match_ids...", len(order))
    needed: list[tuple[str, int]] = []
    seen: set[int] = set()
    for key in order:
        for mid in _match_ids_for_tournament(key):
            if mid in seen:
                continue
            seen.add(mid)
            if _shard_has_players(RAW_DIR, mid, key):
                continue
            cur = existing.get(str(mid))
            if _has_players(cur):
                continue
            needed.append((key, mid))

    if max_new is not None:
        needed = needed[: max(0, max_new)]

    logger.info("Need download: %s", len(needed))
    if not needed:
        with_players = sum(1 for v in existing.values() if _has_players(v))
        return {"cached": len(existing), "downloaded": 0, "errors": 0, "with_players": with_players}

    errors = 0
    downloaded = 0
    for i, (tourn_key, mid) in enumerate(needed):
        try:
            detail = client.get_match(mid)
            if detail and "error" not in detail and _has_players(detail):
                detail = {**detail, "match_id": detail.get("match_id") or mid}
                atomic_write_json(shard_path_for_match(RAW_DIR, mid, tourn_key), detail)
                existing[str(mid)] = detail
                downloaded += 1
            else:
                errors += 1
        except Exception as exc:  # noqa: BLE001 — network resilience
            errors += 1
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg:
                logger.warning("Rate limited at %s, sleeping 65s...", mid)
                time.sleep(65)
            elif "timed out" in msg or "timeout" in msg:
                logger.warning("Timeout at %s, sleeping 5s...", mid)
                time.sleep(5)
            else:
                logger.warning("Error at %s: %s", mid, exc)
                time.sleep(1)

        if (i + 1) % 10 == 0 or (i + 1) == len(needed):
            if write_monolith:
                atomic_write_json(details_file, existing)
            logger.info(
                "Progress %s/%s | new=%s | errors=%s",
                i + 1,
                len(needed),
                downloaded,
                errors,
            )

    if write_monolith:
        atomic_write_json(details_file, existing)

    with_players = sum(1 for v in existing.values() if _has_players(v))
    logger.info(
        "Done. monolith=%s with_players=%s new=%s errors=%s",
        len(existing),
        with_players,
        downloaded,
        errors,
    )
    return {
        "cached": len(existing),
        "with_players": with_players,
        "downloaded": downloaded,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenDota match details (sharded)")
    parser.add_argument("--max", type=int, default=None, help="Max new matches to fetch")
    parser.add_argument(
        "--tournaments",
        nargs="*",
        default=None,
        help="Optional subset of tournament keys",
    )
    parser.add_argument("--rate", type=float, default=1.1, help="Seconds between API calls")
    parser.add_argument(
        "--write-monolith",
        action="store_true",
        help="Also rewrite legacy match_details.json (slow/large; default off)",
    )
    args = parser.parse_args()
    download_details(
        max_new=args.max,
        tournaments=args.tournaments,
        rate_limit=args.rate,
        write_monolith=args.write_monolith,
    )


if __name__ == "__main__":
    main()
