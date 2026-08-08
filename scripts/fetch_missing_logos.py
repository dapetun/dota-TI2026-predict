"""Scrape public pages for missing team logo URLs and download them."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "img" / "teams"
UA = {"User-Agent": "Mozilla/5.0 (compatible; TI2026-predict/1.0)"}

MISSING = {
    "Aurora": ["Aurora", "aurora"],
    "Falcons": ["Falcons", "falcons"],
    "1w": ["Tundra", "1win", "Iron", "1w"],
    "Yandex": ["Yandex", "yandex", "Virtus"],
    "Vision": ["PARIVISION", "Vision", "parivision"],
    "GamerLegion": ["GamerLegion", "Gamer_Legion", "gamerlegion"],
}

PAGES = [
    "https://battlepass.ru/ti2026/predictions",
    "https://liquipedia.net/dota2/Aurora_Gaming",
    "https://liquipedia.net/dota2/Team_Falcons",
    "https://liquipedia.net/dota2/Tundra_Esports",
    "https://liquipedia.net/dota2/Team_Yandex",
    "https://liquipedia.net/dota2/PARIVISION",
    "https://liquipedia.net/dota2/GamerLegion",
]

# Extra known CDN candidates (OpenDota / Steam / Liquipedia commons)
EXTRA = {
    "Aurora": [
        "https://liquipedia.net/commons/images/thumb/8/8a/Aurora_Gaming_allmode.png/120px-Aurora_Gaming_allmode.png",
        "https://img.abiosgaming.com/competitors/aurora-gaming-logo.png",
    ],
    "Falcons": [
        "https://liquipedia.net/commons/images/thumb/2/2f/Team_Falcons_allmode.png/120px-Team_Falcons_allmode.png",
        "https://img.abiosgaming.com/competitors/120x120/Team_Falcons_logo.png",
    ],
    "1w": [
        "https://liquipedia.net/commons/images/thumb/2/2d/Tundra_Esports_allmode.png/120px-Tundra_Esports_allmode.png",
        "https://steamcdn-a.akamaihd.net/apps/dota2/images/team_logos/8291895.png",
        "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/teams/8291895.png",
    ],
    "Yandex": [
        "https://liquipedia.net/commons/images/thumb/9/94/Team_Yandex_allmode.png/120px-Team_Yandex_allmode.png",
        "https://steamcdn-a.akamaihd.net/apps/dota2/images/team_logos/1883502.png",
    ],
    "Vision": [
        "https://liquipedia.net/commons/images/thumb/0/0e/PARIVISION_allmode.png/120px-PARIVISION_allmode.png",
        "https://steamcdn-a.akamaihd.net/apps/dota2/images/team_logos/8599101.png",
    ],
    "GamerLegion": [
        "https://liquipedia.net/commons/images/thumb/4/4a/GamerLegion_allmode.png/120px-GamerLegion_allmode.png",
        "https://img.abiosgaming.com/competitors/gamerlegion-logo.png",
    ],
}


def fetch(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = resp.read()
        return data if len(data) >= 200 else None
    except Exception:
        return None


def main() -> None:
    found: dict[str, list[str]] = {k: [] for k in MISSING}
    img_re = re.compile(r"https?://[^\"'\s>)\\]+\.(?:png|svg|webp|jpe?g)", re.I)
    rel_re = re.compile(r"(?:src|href)=[\"']([^\"']+\.(?:png|svg|webp))[\"']", re.I)

    for page in PAGES:
        data = fetch(page)
        if not data:
            print("skip page", page)
            continue
        html = data.decode("utf-8", "replace")
        urls = set(img_re.findall(html))
        for rel in rel_re.findall(html):
            if rel.startswith("//"):
                urls.add("https:" + rel)
            elif rel.startswith("/"):
                # liquipedia
                if "liquipedia" in page:
                    urls.add("https://liquipedia.net" + rel)
                elif "battlepass" in page:
                    urls.add("https://battlepass.ru" + rel)
        for name, keys in MISSING.items():
            for u in urls:
                if any(k.lower() in u.lower() for k in keys):
                    found[name].append(u)

    for name, extras in EXTRA.items():
        found[name].extend(extras)

    for name, urls in found.items():
        dest = OUT / f"{name}.png"
        # prefer larger liquipedia originals over 120px thumbs
        candidates = []
        for u in urls:
            candidates.append(u)
            if "/thumb/" in u and "/120px-" in u:
                # try full image
                parts = u.split("/thumb/")
                if len(parts) == 2:
                    rest = parts[1]
                    # commons/images/a/ab/File.png/120px-File.png -> commons/images/a/ab/File.png
                    segs = rest.split("/")
                    if len(segs) >= 2:
                        candidates.insert(0, parts[0] + "/" + "/".join(segs[:-1]))
        seen = set()
        uniq = []
        for u in candidates:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        got = False
        for u in uniq:
            blob = fetch(u)
            if not blob:
                continue
            # skip svg if we want png only — still accept
            dest.write_bytes(blob)
            print(f"OK {name} <- {u} ({len(blob)} bytes)")
            got = True
            break
        if not got:
            print(f"FAIL {name} candidates={len(uniq)}")


if __name__ == "__main__":
    main()
