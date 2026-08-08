"""Download TI2026 team logos from public esports CDNs into docs/assets/img/teams/."""

from __future__ import annotations

from pathlib import Path
import urllib.request

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "img" / "teams"
OUT.mkdir(parents=True, exist_ok=True)

# OpenDota / Steam team_id → local filename (project team id)
TEAMS: dict[str, int] = {
    "Aurora": 9823272,
    "BetBoom": 8255888,
    "Falcons": 9247354,
    "Liquid": 2163,
    "1w": 10182357,  # Tundra / Iron Wing roster brand
    "Xtreme": 726228,
    "Yandex": 9895392,
    "Spirit": 7119388,
    "Vision": 9824702,  # PARIVISION
    "HULIGANI": 8261114,  # L1GA
    "Nigma": 7554697,
    "Resilience": 8261197,
    "Vici": 8255842,
    "OG": 2586976,
    "GamerLegion": 9964962,
    "LGD": 15,
}

UA = {"User-Agent": "Mozilla/5.0 (compatible; TI2026-predict/1.0)"}
TEMPLATES = [
    "https://courier.spectral.gg/images/teams/square_{tid}.png",
    "https://courier.spectral.gg/images/teams/{tid}.png",
    "https://steamcdn-a.akamaihd.net/apps/dota2/images/team_logos/{tid}.png",
    "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/teams/{tid}.png",
]


def main() -> None:
    ok: list[str] = []
    fail: list[str] = []
    for name, tid in TEAMS.items():
        dest = OUT / f"{name}.png"
        last: Exception | None = None
        got = False
        for tmpl in TEMPLATES:
            url = tmpl.format(tid=tid)
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = resp.read()
                if len(data) < 200:
                    continue
                dest.write_bytes(data)
                print(f"OK {name} <- {url} ({len(data)} bytes)")
                ok.append(name)
                got = True
                break
            except Exception as exc:  # noqa: BLE001 — try next CDN
                last = exc
        if not got:
            print(f"FAIL {name} tid={tid}: {last}")
            fail.append(name)
    print(f"OK {len(ok)} FAIL {fail}")


if __name__ == "__main__":
    main()
