"""One-shot repair for truncated data/raw/match_details.json."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from download_details import (  # noqa: E402
    _has_players,
    salvage_truncated_json_object,
)
from src.data_collection.match_details import atomic_write_json  # noqa: E402

DETAILS = BASE_DIR / "data" / "raw" / "match_details.json"


def main() -> None:
    if not DETAILS.exists():
        raise SystemExit(f"missing {DETAILS}")

    t0 = time.time()
    print(f"Reading {DETAILS} ({DETAILS.stat().st_size} bytes)...", flush=True)
    raw = DETAILS.read_bytes()
    print(f"Loaded in {time.time() - t0:.1f}s; salvaging...", flush=True)

    t1 = time.time()
    try:
        obj = json.loads(raw)
        print(f"File already valid JSON ({len(obj)} keys)", flush=True)
        return
    except json.JSONDecodeError as exc:
        print(f"JSONDecodeError: {exc}", flush=True)

    obj = salvage_truncated_json_object(raw)
    if obj is None:
        raise SystemExit("salvage failed")
    print(f"Salvaged {len(obj)} keys in {time.time() - t1:.1f}s", flush=True)

    bak = DETAILS.with_name(DETAILS.name + ".corrupt")
    if not bak.exists():
        DETAILS.replace(bak)
        print(f"Moved corrupt -> {bak.name}", flush=True)
    else:
        DETAILS.unlink(missing_ok=True)
        print(f"Removed truncated file (backup already at {bak.name})", flush=True)

    t2 = time.time()
    atomic_write_json(DETAILS, obj)
    print(f"Wrote repaired file ({DETAILS.stat().st_size} bytes) in {time.time() - t2:.1f}s", flush=True)

    with open(DETAILS, encoding="utf-8") as f:
        check = json.load(f)
    with_players = sum(1 for v in check.values() if isinstance(v, dict) and _has_players(v))
    print(
        f"OK keys={len(check)} with_players={with_players} total={time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
