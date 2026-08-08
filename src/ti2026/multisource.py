"""Home LAN, patch-window, market/ranking priors for TI 2026.

Market implied probabilities are an anonymous research signal only.
The author does not support or endorse gambling operators.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.simulation.tournament_sim import FANTASY_BOARD_SLOTS
from src.ti2026.teams import POWER_RANKINGS, get_team_ids

# Patch 7.41 release (approximate OpenDota / Valve window).
PATCH_741_START_TS: int = 1774310400  # 2026-03-24 00:00 UTC
# Default up-weight for matches after PATCH_741_START_TS (wired into sample_weights).
PATCH_IN_MULT: float = 1.25

# Shanghai TI 2026 — home bonus for CN region teams (Elo points).
DEFAULT_HOME_LAN_ELO: float = 30.0
TI2026_HOME_REGION: str = "CN"
TI2026_VENUE: str = "Shanghai"

DEFAULT_MARKET_PRIORS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ti2026_market_priors.json"
)

# Soft ranking → slot mass: stronger ranks lean toward advance slots.
_RANK_SLOT_WEIGHTS: dict[str, list[float]] = {
    # rank band (1-best … 16-worst) → weights over FANTASY_BOARD_SLOTS order
    "top": [0.18, 0.28, 0.40, 0.10, 0.03, 0.01],
    "upper": [0.08, 0.20, 0.42, 0.22, 0.06, 0.02],
    "mid": [0.03, 0.10, 0.35, 0.35, 0.12, 0.05],
    "lower": [0.01, 0.05, 0.20, 0.40, 0.22, 0.12],
    "bottom": [0.005, 0.02, 0.10, 0.35, 0.30, 0.225],
}


@dataclass(frozen=True)
class MultiSourceConfig:
    """Конфиг мульти-источников силы (home LAN / патч / market)."""

    home_lan_elo: float = DEFAULT_HOME_LAN_ELO
    home_region: str = TI2026_HOME_REGION
    patch_start_ts: int = PATCH_741_START_TS
    market_weight: float = 0.10
    ranking_weight: float = 0.10


def home_lan_elo_bonus(
    team_region: str,
    *,
    home_region: str = TI2026_HOME_REGION,
    delta_elo: float = DEFAULT_HOME_LAN_ELO,
) -> float:
    """Вернуть Δ Elo для домашней LAN (CN на Shanghai TI)."""
    if (team_region or "").upper() == home_region.upper():
        return float(delta_elo)
    return 0.0


def is_in_patch_window(start_time: int, patch_start_ts: int = PATCH_741_START_TS) -> bool:
    """True если матч сыгран после старта целевого патча."""
    return int(start_time) >= int(patch_start_ts)


def patch_window_weight(
    start_time: int,
    *,
    patch_start_ts: int = PATCH_741_START_TS,
    in_patch_mult: float = PATCH_IN_MULT,
    out_patch_mult: float = 1.0,
) -> float:
    """Множитель sample-weight: чуть выше вес матчей текущего патча."""
    return in_patch_mult if is_in_patch_window(start_time, patch_start_ts) else out_patch_mult


def _slot_keys() -> list[str]:
    return list(FANTASY_BOARD_SLOTS.keys())


def _rank_band(rank: int) -> str:
    if rank <= 3:
        return "top"
    if rank <= 6:
        return "upper"
    if rank <= 10:
        return "mid"
    if rank <= 13:
        return "lower"
    return "bottom"


def ranking_slot_prior(
    team_id: str,
    slot_key: str,
    *,
    rankings: dict[str, int] | None = None,
) -> float:
    """Bayesian soft prior from POWER_RANKINGS (intentional source, not only fallback)."""
    rankings = rankings or POWER_RANKINGS
    slots = _slot_keys()
    if slot_key not in slots:
        return 1.0 / len(slots)
    rank = int(rankings.get(team_id, 99))
    weights = _RANK_SLOT_WEIGHTS[_rank_band(rank)]
    total = sum(weights) or 1.0
    idx = slots.index(slot_key)
    return float(weights[idx] / total)


def build_ranking_slot_priors(
    team_ids: list[str] | None = None,
    *,
    rankings: dict[str, int] | None = None,
) -> dict[str, dict[str, float]]:
    """team_id -> slot_key -> ranking prior."""
    team_ids = team_ids or get_team_ids()
    return {
        tid: {slot: ranking_slot_prior(tid, slot, rankings=rankings) for slot in _slot_keys()}
        for tid in team_ids
    }


def _normalize_slot_dict(raw: dict[str, float]) -> dict[str, float]:
    slots = _slot_keys()
    vals = {s: max(0.0, float(raw.get(s, 0.0))) for s in slots}
    total = sum(vals.values())
    if total <= 0:
        u = 1.0 / len(slots)
        return {s: u for s in slots}
    return {s: vals[s] / total for s in slots}


def load_market_priors(
    path: str | Path | None = None,
    *,
    seed_from_ranking_if_empty: bool = True,
) -> dict:
    """Load curated anonymous market implied probs; seed from ranking if empty."""
    path = Path(path or DEFAULT_MARKET_PRIORS_PATH)
    data: dict = {
        "disclaimer": (
            "Research signal only. The author does not support or endorse gambling "
            "operators. Not betting advice."
        ),
        "source": "anonymous_market",
        "teams": {},
    }
    if path.exists():
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        data.update(loaded)
        data["teams"] = dict(loaded.get("teams") or {})

    teams = data.get("teams") or {}
    need_seed = seed_from_ranking_if_empty and (
        not teams or bool(data.get("seed_from_ranking"))
    )
    if need_seed:
        seeded = build_ranking_slot_priors()
        # Keep explicit team overrides; fill missing from ranking.
        merged = dict(seeded)
        for tid, slot_probs in teams.items():
            if isinstance(slot_probs, dict) and slot_probs:
                merged[tid] = _normalize_slot_dict(slot_probs)
        data["teams"] = merged
        data["seeded_from_ranking"] = True
        # Do not advertise as live market odds when values are POWER_RANKINGS soft mass.
        data["source"] = "seeded_from_power_rankings"
        data["is_real_market"] = False
    else:
        data["teams"] = {
            tid: _normalize_slot_dict(probs)
            for tid, probs in teams.items()
            if isinstance(probs, dict)
        }
        data["seeded_from_ranking"] = False
        # Keep explicit file flag when present; otherwise infer from non-empty teams.
        if "is_real_market" in data:
            data["is_real_market"] = bool(data["is_real_market"]) and bool(data["teams"])
        else:
            data["is_real_market"] = bool(data["teams"])
        if not data.get("source"):
            data["source"] = "anonymous_market"
    return data


def market_slot_prior(
    team_id: str,
    slot_key: str,
    priors: dict | None = None,
) -> float | None:
    """Implied P(slot) from anonymous market JSON; None if team missing."""
    data = priors if priors is not None else load_market_priors()
    team = (data.get("teams") or {}).get(team_id)
    if not team:
        return None
    if slot_key not in team:
        return None
    return float(team[slot_key])


def market_slot_prior_stub(team_id: str, slot_key: str) -> float | None:
    """Backward-compatible alias for ``market_slot_prior``."""
    return market_slot_prior(team_id, slot_key)
