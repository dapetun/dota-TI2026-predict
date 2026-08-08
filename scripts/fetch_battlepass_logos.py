"""Download all TI2026 team logos from battlepass.ru local CDN paths."""

from __future__ import annotations

import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "img" / "teams"
UA = {"User-Agent": "Mozilla/5.0 (compatible; TI2026-predict/1.0)"}

# project id → battlepass slug (and alternates)
SLUGS: dict[str, list[str]] = {
    "Aurora": ["aurora"],
    "BetBoom": ["boomboys", "betboom", "bb"],
    "Falcons": ["falcons"],
    "Liquid": ["liquid"],
    "1w": ["iron-wing", "tundra", "1win", "1w"],
    "Xtreme": ["xtreme", "xg"],
    "Yandex": ["yandex", "team-yandex", "vp"],
    "Spirit": ["spirit"],
    "Vision": ["vision", "parivision"],
    "HULIGANI": ["huligani", "l1ga"],
    "Nigma": ["nigma", "nigma-galaxy"],
    "Resilience": ["resilience"],
    "Vici": ["vici", "vg"],
    "OG": ["og"],
    "GamerLegion": ["gamerlegion", "gamer-legion"],
    "LGD": ["lgd"],
}


def fetch(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        return data if len(data) >= 400 else None
    except Exception:
        return None


def main() -> None:
    for name, slugs in SLUGS.items():
        got = False
        for slug in slugs:
            for ext in ("webp", "png", "svg"):
                url = f"https://battlepass.ru/img/ti2026/teams/{slug}.{ext}"
                data = fetch(url)
                if not data:
                    continue
                dest = OUT / f"{name}.{ext if ext != 'svg' else 'png'}"
                # always store as webp/png with matching bytes; prefer webp filename
                if ext == "svg":
                    dest = OUT / f"{name}.svg"
                else:
                    dest = OUT / f"{name}.{ext}"
                dest.write_bytes(data)
                # also mirror to .png alias if webp for simpler src
                if ext == "webp":
                    (OUT / f"{name}.png").write_bytes(data)
                print(f"OK {name} <- {url} ({len(data)})")
                got = True
                break
            if got:
                break
        if not got:
            print(f"MISS {name}")


if __name__ == "__main__":
    main()
