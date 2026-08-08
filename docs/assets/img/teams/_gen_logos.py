"""One-off generator for patch-style team SVG emblems."""
from pathlib import Path

TEAMS = {
    "Vision": ("#7c3aed", "V"),
    "Aurora": ("#22d3ee", "A"),
    "BetBoom": ("#ef4444", "BB"),
    "Falcons": ("#16a34a", "F"),
    "Spirit": ("#eab308", "S"),
    "Liquid": ("#38bdf8", "TL"),
    "1w": ("#a3a3a3", "1W"),
    "Xtreme": ("#f97316", "XG"),
    "Yandex": ("#fc3f1d", "YX"),
    "LGD": ("#dc2626", "LGD"),
    "Nigma": ("#6366f1", "NG"),
    "GamerLegion": ("#84cc16", "GL"),
    "OG": ("#f5f5f5", "OG"),
    "HULIGANI": ("#b45309", "HG"),
    "Resilience": ("#14b8a6", "TR"),
    "Vici": ("#e11d48", "VG"),
}


def svg_for(tid: str, color: str, mon: str) -> str:
    size = 16 if len(mon) > 2 else 20
    label = tid.upper()[:10]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" role="img" aria-label="{tid}">
  <defs>
    <radialGradient id="g" cx="50%" cy="40%" r="60%">
      <stop offset="0%" stop-color="#2a2218"/>
      <stop offset="100%" stop-color="#0e0c0a"/>
    </radialGradient>
    <linearGradient id="ring" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#e8c878"/>
      <stop offset="50%" stop-color="#a67c2e"/>
      <stop offset="100%" stop-color="#d4af57"/>
    </linearGradient>
  </defs>
  <rect width="96" height="96" rx="10" fill="url(#g)"/>
  <rect x="3" y="3" width="90" height="90" rx="8" fill="none" stroke="url(#ring)" stroke-width="2.2"/>
  <rect x="8" y="8" width="80" height="80" rx="6" fill="none" stroke="{color}" stroke-opacity="0.45" stroke-width="1.2"/>
  <circle cx="48" cy="40" r="18" fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="1.5"/>
  <text x="48" y="46" text-anchor="middle" font-family="Georgia, serif" font-size="{size}" font-weight="700" fill="#f3e6c8">{mon}</text>
  <text x="48" y="78" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="8" letter-spacing="1.5" fill="#c5a059">{label}</text>
</svg>
"""


def main() -> None:
    out = Path(__file__).resolve().parent
    for tid, (color, mon) in TEAMS.items():
        (out / f"{tid}.svg").write_text(svg_for(tid, color, mon), encoding="utf-8")
    print(f"wrote {len(TEAMS)} logos to {out}")


if __name__ == "__main__":
    main()
