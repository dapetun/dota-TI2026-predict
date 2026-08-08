"""Fetch TI2026 expert boards from battlepass.ru and merge into analyst_picks.json.

Uses author-stated ``picks`` only (not model-completed ``assignment``).
Dedupes against existing Sports.ru / Hotspawn entries by id / name aliases.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ti2026.teams import normalize_team_name

PICKS_PATH = ROOT / "docs" / "data" / "analyst_picks.json"
BP_URL = "https://battlepass.ru/ti2026/predictions"
UA = {"User-Agent": "Mozilla/5.0 (compatible; TI2026-predict/1.0)"}

SLOT_MAP = {
    "perfect": "undefeated",
    "oneLoss": "one_loss",
    "advance": "advance",
    "eliminated": "eliminate",
    "oneWin": "one_win",
    "winless": "winless",
}

# battlepass slug → canonical team id
SLUG_TO_TEAM = {
    "aurora": "Aurora",
    "boomboys": "BetBoom",
    "betboom": "BetBoom",
    "falcons": "Falcons",
    "liquid": "Liquid",
    "iron-wing": "1w",
    "xtreme": "Xtreme",
    "yandex": "Yandex",
    "spirit": "Spirit",
    "vision": "Vision",
    "huligani": "HULIGANI",
    "nigma": "Nigma",
    "resilience": "Resilience",
    "vici": "Vici",
    "og": "OG",
    "gamerlegion": "GamerLegion",
    "lgd": "LGD",
}

# Existing picks ids / name stems that must not be duplicated
DEDUP_IDS = {
    "ns",
    "sikle",
    "fishman",
    "arteezy",
    "topson",
    "nix",
    "roger",
    "rodjer",
    "reddit_sim",
    "sqreen",
    "onejey",
    "misha",
    "editorial",
    "hotspawn_lead",
    "hotspawn_sophie",
    "sophie-mccarthy",
    "hotspawn_owen",
    "owen-hotspawn",
    "hotspawn_daniel",
    "linda-guster",
}

DEDUP_NAME_STEMS = {
    "ns",
    "sikle",
    "fishman",
    "arteezy",
    "topson",
    "nix",
    "roger",
    "rodjer",
    "rodjер",
    "sqreen",
    "onejey",
    "misha",
    "sophie",
    "owen",
    "linda",
    "güster",
    "guster",
}


def fetch_html(url: str = BP_URL) -> str:
    """Download predictions page HTML."""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read().decode("utf-8", "replace")


def extract_experts_js(html: str) -> str:
    """Return the raw JS array literal for ``experts:[...]``."""
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S | re.I)
    sk = next((s for s in scripts if "__sveltekit_" in s and "experts:[" in s), None)
    if sk is None:
        raise RuntimeError("SvelteKit bootstrap with experts[] not found")
    start = sk.find("experts:[")
    i = start + len("experts:")
    depth = 0
    for j in range(i, len(sk)):
        ch = sk[j]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return sk[i : j + 1]
    raise RuntimeError("unclosed experts array")


def parse_experts_from_js(raw_array: str) -> list[dict]:
    """Parse experts without full JS eval — extract id/name/stated/picks."""
    experts: list[dict] = []
    parts = re.split(r'(?=\{id:")', raw_array)
    for chunk in parts:
        if not chunk.startswith("{id:"):
            continue
        id_m = re.match(r'\{id:"([^"]+)"', chunk)
        if not id_m:
            continue
        eid = id_m.group(1)
        name_m = re.search(r'\bname:"((?:\\.|[^"\\])*)"', chunk)
        name_en_m = re.search(r'\bnameEn:"((?:\\.|[^"\\])*)"', chunk)
        stated_m = re.search(r"\bstated:(\d+)", chunk)
        picks_m = re.search(r"\bpicks:(\{.*?\}),soft:", chunk, re.S)
        if not picks_m or not stated_m:
            continue
        picks_js = re.sub(
            r"([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:",
            r'\1"\2":',
            picks_m.group(1),
        )
        picks = json.loads(picks_js)
        url_m = re.search(
            r'source:\{title:"(?:\\.|[^"\\])*",url:(null|"((?:\\.|[^"\\])*)")',
            chunk,
        )
        source_url = None
        if url_m and url_m.group(1) != "null":
            source_url = url_m.group(2)
        quote_m = re.search(r'\bquote:(null|"((?:\\.|[^"\\])*)")', chunk)
        quote = None
        if quote_m and quote_m.group(1) != "null":
            quote = quote_m.group(2).replace(r"\"", '"')
        role_m = re.search(r'\brole:"((?:\\.|[^"\\])*)"', chunk)
        date_m = re.search(r'\bdateIso:"([^"]+)"', chunk)
        name = (name_m.group(1) if name_m else eid).replace(r"\"", '"')
        name_en = (name_en_m.group(1) if name_en_m else name).replace(r"\"", '"')
        experts.append(
            {
                "id": eid,
                "name": name,
                "nameEn": name_en,
                "stated": int(stated_m.group(1)),
                "picks": picks,
                "source": {"url": source_url},
                "quote": quote,
                "role": role_m.group(1) if role_m else None,
                "dateIso": date_m.group(1) if date_m else None,
                "soft": "soft:[{" in chunk,
            }
        )
    if not experts:
        raise RuntimeError("failed to parse any experts from battlepass payload")
    return experts


def js_literal_to_json(text: str) -> object:
    """Convert compact JS object/array literal to JSON-compatible Python."""
    s = text
    s = re.sub(r"\bvoid 0\b", "null", s)
    s = re.sub(r"\bundefined\b", "null", s)
    s = re.sub(r"([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', s)
    s = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", lambda m: json.dumps(m.group(1)), s)
    # JS allows .5; JSON needs 0.5
    s = re.sub(r"(?<![0-9])\.(\d+)", r"0.\1", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return json.loads(s)


def picks_to_board(picks: dict) -> dict[str, str]:
    """Map battlepass picks{slot:[slug]} → {canonical_team: slot_key}."""
    board: dict[str, str] = {}
    for bp_slot, slugs in (picks or {}).items():
        slot = SLOT_MAP.get(bp_slot)
        if not slot:
            continue
        for slug in slugs or []:
            tid = SLUG_TO_TEAM.get(str(slug).lower()) or normalize_team_name(str(slug))
            if tid in board:
                continue
            board[tid] = slot
    return board


def _name_stem(name: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", "", (name or "").lower())


def is_duplicate(expert: dict, existing_ids: set[str], existing_stems: set[str]) -> bool:
    """True if expert already present in analyst_picks."""
    eid = str(expert.get("id") or "").lower()
    if eid in existing_ids or eid in DEDUP_IDS:
        return True
    # strip battlepass_ prefix variants
    if eid.startswith("battlepass_") and eid[len("battlepass_") :] in existing_ids:
        return True
    name = expert.get("name") or expert.get("nameEn") or ""
    stem = _name_stem(name)
    if stem and stem in existing_stems:
        return True
    for known in DEDUP_NAME_STEMS:
        if known and known in stem:
            return True
    return False


def expert_to_record(expert: dict) -> tuple[str, dict]:
    """Build analysts[] or partial[] record from battlepass expert.

    Returns (\"full\"|\"partial\", record).
    """
    picks = expert.get("picks") or {}
    board = picks_to_board(picks)
    stated = int(expert.get("stated") or len(board))
    eid = f"battlepass_{expert.get('id')}"
    name = expert.get("name") or expert.get("nameEn") or str(expert.get("id"))
    src = expert.get("source") or {}
    source_url = src.get("url") or BP_URL
    note_parts = []
    if expert.get("role"):
        note_parts.append(str(expert["role"]))
    if expert.get("quote"):
        note_parts.append(str(expert["quote"]))
    soft = bool(expert.get("soft"))
    base = {
        "id": eid,
        "name": name,
        "source_url": source_url,
        "updated_at": str(expert.get("dateIso") or date.today().isoformat()),
        "board": board,
    }
    if note_parts:
        base["note"] = " · ".join(note_parts)
    if stated >= 16 and len(board) == 16:
        return "full", {"id": base["id"], "name": base["name"], "board": board}
    base["soft"] = soft
    base["partial"] = True
    return "partial", base


def merge_into_picks(picks: dict, experts: list[dict]) -> dict:
    """Merge new battlepass experts into analyst_picks structure."""
    analysts = list(picks.get("analysts") or [])
    partial = list(picks.get("partial") or [])
    existing_ids = {str(a.get("id") or "").lower() for a in analysts + partial}
    existing_stems = {_name_stem(a.get("name") or "") for a in analysts + partial}

    added_full = 0
    added_partial = 0
    skipped = 0
    for ex in experts:
        if is_duplicate(ex, existing_ids, existing_stems):
            skipped += 1
            continue
        kind, rec = expert_to_record(ex)
        if kind == "full":
            analysts.append(rec)
            added_full += 1
        else:
            if not rec.get("board") and not rec.get("note"):
                skipped += 1
                continue
            partial.append(rec)
            added_partial += 1
        existing_ids.add(str(rec["id"]).lower())
        existing_stems.add(_name_stem(rec.get("name") or ""))

    sources = list(picks.get("sources") or [])
    if not any(s.get("id") == "battlepass" for s in sources):
        sources.append(
            {
                "id": "battlepass",
                "url": BP_URL,
                "as_of": date.today().isoformat(),
            }
        )
    else:
        for s in sources:
            if s.get("id") == "battlepass":
                s["as_of"] = date.today().isoformat()

    picks["analysts"] = analysts
    picks["partial"] = partial
    picks["sources"] = sources
    picks["updated_at"] = date.today().isoformat()
    src_label = picks.get("source") or ""
    if "battlepass" not in src_label.lower():
        picks["source"] = (src_label + " + battlepass.ru").strip(" +")
    note = picks.get("note") or ""
    if "battlepass" not in note.lower():
        picks["note"] = (
            note
            + " battlepass.ru: новые уникальные сетки (без дублей Sports.ru/Hotspawn); "
            "только заявленные слоты автора, без model-fill."
        ).strip()
    picks["_battlepass_merge"] = {
        "added_full": added_full,
        "added_partial": added_partial,
        "skipped_dedup": skipped,
        "fetched": len(experts),
    }
    return picks


def main() -> None:
    """CLI: fetch battlepass experts and merge into analyst_picks.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print merge stats only")
    parser.add_argument(
        "--out",
        type=Path,
        default=PICKS_PATH,
        help="Path to analyst_picks.json",
    )
    args = parser.parse_args()

    html = fetch_html()
    raw = extract_experts_js(html)
    try:
        experts = js_literal_to_json(raw)
    except json.JSONDecodeError:
        experts = parse_experts_from_js(raw)
    if not isinstance(experts, list):
        raise RuntimeError("experts payload is not a list")

    picks = json.loads(args.out.read_text(encoding="utf-8"))
    merged = merge_into_picks(picks, experts)
    stats = merged.pop("_battlepass_merge", {})
    print(
        f"fetched={stats.get('fetched')} added_full={stats.get('added_full')} "
        f"added_partial={stats.get('added_partial')} skipped={stats.get('skipped_dedup')}"
    )
    if args.dry_run:
        return
    args.out.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
