"""Resume-friendly OpenDota match-detail downloader (priority: recent TI/majors)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_collection.opendota_api import OpenDotaClient
from src.data_collection.tournaments import TOURNAMENTS, download_order

RAW_DIR = BASE_DIR / "data" / "raw"


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
    """Recover a truncated top-level JSON object up to the last complete value.

    One brace-depth pass (string-aware), then a single json.loads of the cut.
    """
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
    """Load cache; on corrupt/truncated JSON, salvage last complete entries."""
    if not details_file.exists():
        return {}
    try:
        with open(details_file, encoding="utf-8") as f:
            payload = json.load(f)
        return _normalize_details_payload(payload)
    except json.JSONDecodeError as exc:
        print(
            f"WARNING: corrupt JSON in {details_file.name} ({exc}); attempting salvage...",
            flush=True,
        )
    except OSError as exc:
        print(f"WARNING: cannot read {details_file.name}: {exc}; starting empty", flush=True)
        return {}

    try:
        raw = details_file.read_bytes()
    except OSError as exc:
        print(f"WARNING: salvage read failed ({exc}); starting empty", flush=True)
        return {}

    salvaged_obj = salvage_truncated_json_object(raw)
    if salvaged_obj is None:
        print(
            f"ERROR: corrupt {details_file.name}, salvage failed — starting empty cache",
            flush=True,
        )
        return {}

    salvaged = _normalize_details_payload(salvaged_obj)
    print(f"WARNING: salvaged truncated {details_file.name} ({len(salvaged)} keys)", flush=True)
    bak = details_file.with_name(details_file.name + ".corrupt")
    try:
        if not bak.exists():
            details_file.replace(bak)
            print(f"Moved corrupt file -> {bak.name}", flush=True)
        else:
            details_file.unlink(missing_ok=True)
        _atomic_write_json(details_file, salvaged)
        print(f"Rewrote repaired {details_file.name}", flush=True)
    except OSError as exc:
        print(f"WARNING: could not persist repaired cache ({exc}); using in-memory salvage", flush=True)
    return salvaged


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
    print("Loading existing details cache...", flush=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    details_file = RAW_DIR / "match_details.json"
    existing = _load_existing(details_file)
    print(f"Cached details: {len(existing)}", flush=True)
    client = OpenDotaClient(rate_limit=rate_limit)

    order = tournaments or download_order()
    print(f"Scanning {len(order)} tournaments for missing match_ids...", flush=True)
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

    print(f"Need download: {len(needed)}", flush=True)
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
                print(f"  Rate limited at {mid}, sleeping 65s...", flush=True)
                time.sleep(65)
            elif "timed out" in msg or "timeout" in msg:
                print(f"  Timeout at {mid}, sleeping 5s...", flush=True)
                time.sleep(5)
            else:
                print(f"  Error at {mid}: {exc}", flush=True)
                time.sleep(1)

        if (i + 1) % 10 == 0 or (i + 1) == len(needed):
            _atomic_write_json(details_file, existing)
            print(
                f"  Progress {i + 1}/{len(needed)} | "
                f"with_players={sum(1 for v in existing.values() if _has_players(v))} | "
                f"new={downloaded} | errors={errors}",
                flush=True,
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
    parser.add_argument("--rate", type=float, default=1.1, help="Seconds between API calls")
    args = parser.parse_args()
    download_details(max_new=args.max, tournaments=args.tournaments, rate_limit=args.rate)


if __name__ == "__main__":
    main()
