"""Fetch live Polymarket TI2026 prices → Swiss slot priors JSON.

Public Gamma API (no auth). Swiss *slot* books are not listed on Polymarket;
winner Yes-prices are converted to pairwise Bradley–Terry strengths and run
through the project Swiss Monte Carlo to obtain slot probabilities.

Research signal only — not betting advice.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.tournament_sim import FANTASY_BOARD_SLOTS, simulate_swiss_stage
from src.ti2026.teams import ALIAS_TO_CANONICAL, get_team_ids

GAMMA = "https://gamma-api.polymarket.com"
WINNER_SLUG = "the-international-2026-winner-20260629212545745"
DEFAULT_OUT = ROOT / "data" / "ti2026_market_priors.json"
UA = {
    "User-Agent": "dota-TI2026-predict/1.0 (research; +https://github.com/)",
    "Accept": "application/json",
}

# Placeholders / inactive outcomes on the winner event — skip.
_SKIP_TITLES = {"a", "b", "c", "other", "another team"}

# Map Polymarket groupItemTitle → internal id when alias table misses.
_EXTRA_ALIASES = {
    "boomboys": "BetBoom",
    "1w team": "1w",
    "team vision": "Vision",
    "team yandex": "Yandex",
}


def _get_json(url: str, *, timeout: float = 45.0) -> object:
    """GET JSON from Gamma (requires UA; bare urllib often gets 403)."""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _parse_list_field(raw: object) -> list:
    if raw is None:
        return []
    if isinstance(raw, str):
        return json.loads(raw)
    return list(raw)


def resolve_team_id(label: str) -> str | None:
    """Map Polymarket outcome label to TI2026 team_id."""
    key = (label or "").strip().lower()
    if not key or key in _SKIP_TITLES:
        return None
    if key in _EXTRA_ALIASES:
        return _EXTRA_ALIASES[key]
    if key in ALIAS_TO_CANONICAL:
        return ALIAS_TO_CANONICAL[key]
    # Strip common prefixes.
    for prefix in ("team ", "the "):
        if key.startswith(prefix):
            return resolve_team_id(key[len(prefix) :])
    return ALIAS_TO_CANONICAL.get(key)


def fetch_winner_event(slug: str = WINNER_SLUG) -> dict:
    """Load winner event payload from Gamma ``/events?slug=``."""
    data = _get_json(f"{GAMMA}/events?slug={urllib.parse.quote(slug)}")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Polymarket event not found: {slug}")
    return data[0]


def extract_winner_yes_prices(event: dict) -> dict[str, dict]:
    """team_id → {label, yes_price, market_slug, active} from Yes/No markets."""
    out: dict[str, dict] = {}
    for m in event.get("markets") or []:
        if m.get("closed"):
            continue
        title = (m.get("groupItemTitle") or "").strip()
        tid = resolve_team_id(title)
        if tid is None:
            continue
        if m.get("active") is False:
            continue
        outcomes = _parse_list_field(m.get("outcomes"))
        prices = _parse_list_field(m.get("outcomePrices"))
        if not outcomes or len(outcomes) != len(prices):
            continue
        yes_idx = next(
            (i for i, o in enumerate(outcomes) if str(o).lower() == "yes"),
            None,
        )
        if yes_idx is None:
            continue
        yes = float(prices[yes_idx])
        if not (0.0 <= yes <= 1.0) or math.isnan(yes):
            continue
        out[tid] = {
            "label": title,
            "yes_price": yes,
            "market_slug": m.get("slug"),
            "question": m.get("question"),
            "active": bool(m.get("active")),
        }
    return out


def search_related_events(query: str = "The International 2026") -> list[dict]:
    """Lightweight discovery of other TI2026 Polymarket events (provenance)."""
    data = _get_json(
        f"{GAMMA}/public-search?q={urllib.parse.quote(query)}"
    )
    events = data.get("events") if isinstance(data, dict) else []
    rows = []
    for e in events or []:
        title = e.get("title") or ""
        if "international 2026" not in title.lower() and "ti 2026" not in title.lower():
            continue
        rows.append(
            {
                "title": title,
                "slug": e.get("slug"),
                "n_markets": len(e.get("markets") or []),
            }
        )
    return rows


def normalize_winner_probs(raw: dict[str, float]) -> dict[str, float]:
    """Renormalize independent Yes prices so they sum to 1 (simple de-vig)."""
    total = sum(max(0.0, float(v)) for v in raw.values())
    if total <= 0:
        raise ValueError("Winner prices sum to 0")
    return {k: max(0.0, float(v)) / total for k, v in raw.items()}


def bt_win_matrix(
    team_ids: list[str],
    strengths: dict[str, float],
    *,
    floor: float = 0.02,
    ceil: float = 0.98,
) -> pd.DataFrame:
    """Bradley–Terry P(i beats j) from positive strengths."""
    s = {t: max(1e-9, float(strengths.get(t, 1e-9))) for t in team_ids}
    mat = pd.DataFrame(0.5, index=team_ids, columns=team_ids, dtype=float)
    for a in team_ids:
        for b in team_ids:
            if a == b:
                continue
            p = s[a] / (s[a] + s[b])
            mat.loc[a, b] = float(np.clip(p, floor, ceil))
    return mat


def winner_probs_to_slot_priors(
    winner_probs: dict[str, float],
    *,
    n_sims: int = 25_000,
    seed: int = 42,
    strength_power: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Convert tournament-winner implied probs → Swiss fantasy slot probs.

    Method (documented, not silent):
    1. Normalize winner Yes-prices to a simplex.
    2. Strength s_i = p_i ** strength_power (default sqrt — dampens playoff-only
       signal when projecting to Swiss).
    3. Pairwise BT matrix → ``simulate_swiss_stage`` Monte Carlo.
    4. Map sim slot % → undefeated/one_loss/advance/eliminate/one_win/winless.
    """
    team_ids = get_team_ids()
    missing = [t for t in team_ids if t not in winner_probs]
    if missing:
        raise ValueError(f"Missing winner probs for teams: {missing}")

    p_norm = normalize_winner_probs({t: winner_probs[t] for t in team_ids})
    strengths = {t: float(p_norm[t]) ** float(strength_power) for t in team_ids}
    matrix = bt_win_matrix(team_ids, strengths)
    df = simulate_swiss_stage(
        matrix,
        team_ids,
        n_simulations=int(n_sims),
        rng_seed=int(seed),
    )

    key_map = {
        "undefeated": "prob_4_0",
        "one_loss": "prob_4_1",
        "advance": "prob_advance",
        "eliminate": "prob_eliminate",
        "one_win": "prob_1_4",
        "winless": "prob_0_4",
    }
    slots = list(FANTASY_BOARD_SLOTS.keys())
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        tid = str(row["team"])
        raw = {slot: float(row[col]) / 100.0 for slot, col in key_map.items()}
        total = sum(raw.values()) or 1.0
        out[tid] = {s: raw[s] / total for s in slots}
    return out


def build_payload(
    *,
    n_sims: int,
    seed: int,
    strength_power: float,
) -> dict:
    """Fetch Polymarket + derive slot priors payload."""
    event = fetch_winner_event()
    priced = extract_winner_yes_prices(event)
    team_ids = get_team_ids()
    missing = [t for t in team_ids if t not in priced]
    if missing:
        raise RuntimeError(
            f"Polymarket winner markets missing teams: {missing}. "
            f"Got: {sorted(priced)}"
        )

    raw_yes = {t: float(priced[t]["yes_price"]) for t in team_ids}
    p_norm = normalize_winner_probs(raw_yes)
    teams = winner_probs_to_slot_priors(
        raw_yes,
        n_sims=n_sims,
        seed=seed,
        strength_power=strength_power,
    )

    now = datetime.now(timezone.utc)
    related = search_related_events()
    swiss_markets_found = [
        r
        for r in related
        if any(
            k in (r.get("title") or "").lower()
            for k in ("swiss", "undefeated", "advance from", "group stage")
        )
    ]

    return {
        "disclaimer": (
            "Research signal only. The author does not support or endorse gambling "
            "operators. Implied probabilities are for Bayesian fusion — not betting advice."
        ),
        "source": "polymarket_gamma_api",
        "updated_at": now.strftime("%Y-%m-%d"),
        "fetched_at_utc": now.isoformat(timespec="seconds"),
        "seed_from_ranking": False,
        "is_real_market": True,
        "derivation": "derived_from_winner_odds",
        "partial": True,
        "note": (
            "No public Swiss-slot markets on Polymarket/OddsPortal as of fetch. "
            "teams{} are Swiss fantasy-slot probs derived from live tournament-WINNER "
            "Yes prices via Bradley–Terry + project Swiss Monte Carlo "
            f"(n_sims={n_sims}, strength_power={strength_power}). "
            "Winner≠Swiss strength — treat as soft market prior."
        ),
        "method": {
            "api": f"{GAMMA}/events?slug={WINNER_SLUG}",
            "docs": "https://docs.polymarket.com/",
            "winner_event_slug": WINNER_SLUG,
            "winner_event_url": (
                f"https://polymarket.com/event/{WINNER_SLUG}"
            ),
            "steps": [
                "Pull Yes outcomePrices for each active team market (skip A/B/C/Other).",
                "Normalize Yes prices to sum=1 (independent markets de-vig).",
                f"Strength s_i = p_i ** {strength_power}.",
                "Bradley–Terry P(i>j) = s_i/(s_i+s_j), clipped to [0.02, 0.98].",
                f"simulate_swiss_stage Monte Carlo n_sims={n_sims}, seed={seed}.",
                "Map prob_4_0→undefeated, prob_4_1→one_loss, …; renormalize per team.",
            ],
            "n_sims": n_sims,
            "rng_seed": seed,
            "strength_power": strength_power,
            "swiss_slot_markets_found": swiss_markets_found,
            "related_ti2026_events": related,
        },
        "how_to_enable_real_market": [
            "Run: python scripts/fetch_market_priors.py",
            "Or paste curated Swiss-slot implied probs into teams{} and set "
            "seed_from_ranking=false, is_real_market=true.",
            "Re-run scripts/export_web_data.py — market_weight stays non-zero only "
            "when is_real_market and not seeded.",
        ],
        "tournament_winner_reference": {
            "note": (
                "Live Polymarket winner Yes prices (pre-normalization). "
                "NOT Swiss slot books — used only as derivation input."
            ),
            "polymarket": {
                "url": f"https://polymarket.com/event/{WINNER_SLUG}",
                "api": f"{GAMMA}/events?slug={WINNER_SLUG}",
                "as_of": now.strftime("%Y-%m-%d"),
                "fetched_at_utc": now.isoformat(timespec="seconds"),
                "market": event.get("title") or "The International 2026: Winner",
                "volume": event.get("volume"),
                "liquidity": event.get("liquidity"),
                "raw_yes_price": {t: priced[t]["yes_price"] for t in team_ids},
                "normalized_implied": {t: round(p_norm[t], 6) for t in team_ids},
                "markets": {
                    t: {
                        "label": priced[t]["label"],
                        "slug": priced[t]["market_slug"],
                        "yes_price": priced[t]["yes_price"],
                    }
                    for t in team_ids
                },
            },
            "cyberscore_winner_odds_digest": {
                "url": "https://cyberscore.live/en/news/the-international-2026-winner-odds/",
                "as_of": "2026-07-29",
                "note": (
                    "Static article snapshot (not used for teams{} derivation). "
                    "Kept for cross-check vs Polymarket."
                ),
                "odds": {
                    "Vision": {"decimal": 4.4, "implied_pct": 22.73},
                    "Yandex": {"decimal": 5.3, "implied_pct": 18.87},
                    "BetBoom": {"decimal": 7.5, "implied_pct": 13.33},
                    "Spirit": {"decimal": 8.5, "implied_pct": 11.76},
                    "Falcons": {"decimal": 8.5, "implied_pct": 11.76},
                    "Aurora": {"decimal": 10.0, "implied_pct": 10.0},
                    "1w": {"decimal": 14.0, "implied_pct": 7.14},
                    "Liquid": {"decimal": 17.0, "implied_pct": 5.88},
                    "Nigma": {"decimal": 17.0, "implied_pct": 5.88},
                    "LGD": {"decimal": 17.0, "implied_pct": 5.88},
                    "Xtreme": {"decimal": 17.0, "implied_pct": 5.88},
                    "Vici": {"decimal": 25.0, "implied_pct": 4.0},
                    "Resilience": {"decimal": 30.0, "implied_pct": 3.33},
                    "OG": {"decimal": 50.0, "implied_pct": 2.0},
                    "GamerLegion": {"decimal": 60.0, "implied_pct": 1.67},
                    "HULIGANI": {"decimal": 100.0, "implied_pct": 1.0},
                },
            },
        },
        "teams": {
            tid: {slot: round(float(probs[slot]), 6) for slot in FANTASY_BOARD_SLOTS}
            for tid, probs in teams.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path",
    )
    parser.add_argument("--n-sims", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strength-power",
        type=float,
        default=0.5,
        help="Exponent on winner p when building BT strength (1=linear, 0.5=sqrt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only; do not write file",
    )
    args = parser.parse_args()

    payload = build_payload(
        n_sims=args.n_sims,
        seed=args.seed,
        strength_power=args.strength_power,
    )
    teams = payload["teams"]
    print(
        f"Fetched {len(teams)}/16 teams from Polymarket; "
        f"derivation={payload['derivation']}"
    )
    ref = payload["tournament_winner_reference"]["polymarket"]["normalized_implied"]
    top = sorted(ref.items(), key=lambda x: -x[1])[:5]
    print("Top winner implied:", ", ".join(f"{k}={v:.1%}" for k, v in top))
    sample = next(iter(teams.values()))
    print(
        "Sample slots:",
        {k: round(v, 3) for k, v in sample.items()},
    )

    if args.dry_run:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
