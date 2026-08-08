"""Player × hero signature pools with shrinkage to global meta."""

from __future__ import annotations

import json
from pathlib import Path

from src.features.hero_meta import hero_winrate, load_hero_meta

DEFAULT_SIGNATURES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "hero" / "player_signatures.json"
)
DEFAULT_SHRINK_K: float = 20.0
DEFAULT_TOP_K: int = 8


def load_player_signatures(path: str | Path | None = None) -> dict:
    """Load player_signatures JSON."""
    path = Path(path or DEFAULT_SIGNATURES_PATH)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def shrink_wr(games: float, wins: float, meta_wr: float, *, k: float = DEFAULT_SHRINK_K) -> float:
    """Empirical Bayes shrink player WR toward meta WR."""
    n = max(0.0, float(games))
    w_hat = (float(wins) / n) if n > 0 else meta_wr
    return (n / (n + k)) * w_hat + (k / (n + k)) * float(meta_wr)


def build_player_signature(
    account_id: int,
    hero_rows: list[dict],
    hero_meta: dict,
    *,
    top_k: int = DEFAULT_TOP_K,
    shrink_k: float = DEFAULT_SHRINK_K,
) -> dict:
    """Top-K signature heroes for one account with shrunk scores."""
    scored: list[dict] = []
    for row in hero_rows:
        hid = int(row.get("hero_id") or 0)
        if hid <= 0:
            continue
        games = float(row.get("games") or 0)
        wins = float(row.get("win") or 0)
        if games <= 0:
            continue
        meta_wr = hero_winrate(hero_meta, hid)
        score = shrink_wr(games, wins, meta_wr, k=shrink_k)
        scored.append(
            {
                "hero_id": hid,
                "games": int(games),
                "wins": int(wins),
                "wr": round(wins / games, 4),
                "meta_wr": round(meta_wr, 4),
                "score": round(score, 4),
            }
        )
    scored.sort(key=lambda x: (-x["games"], -x["score"]))
    top = scored[:top_k]
    mean_score = float(sum(h["score"] for h in top) / len(top)) if top else 0.5
    return {
        "account_id": int(account_id),
        "heroes": top,
        "sig_wr": round(mean_score, 4),
        "pool_depth": len([h for h in top if h["score"] > 0.52]),
    }


def save_player_signatures(data: dict, path: str | Path | None = None) -> Path:
    """Write player_signatures JSON."""
    path = Path(path or DEFAULT_SIGNATURES_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def signatures_from_fixture_rows(
    players: dict[str, list[dict]],
    hero_meta: dict | None = None,
) -> dict:
    """Build signatures dict from account_id → hero row list (offline fixtures)."""
    hero_meta = hero_meta or load_hero_meta()
    out: dict[str, dict] = {}
    for aid, rows in players.items():
        out[str(aid)] = build_player_signature(int(aid), rows, hero_meta)
    return {
        "source": "fixture",
        "usable_in_production": False,
        "note": "Test fixture only — not for production soft prior.",
        "players": out,
    }
