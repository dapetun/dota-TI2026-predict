"""Home LAN, patch-window, and multi-source context helpers for TI 2026."""

from __future__ import annotations

from dataclasses import dataclass

# Patch 7.41 release (approximate OpenDota / Valve window).
PATCH_741_START_TS: int = 1774310400  # 2026-03-24 00:00 UTC
# Default up-weight for matches after PATCH_741_START_TS (wired into sample_weights).
PATCH_IN_MULT: float = 1.25

# Shanghai TI 2026 — home bonus for CN region teams (Elo points).
DEFAULT_HOME_LAN_ELO: float = 30.0
TI2026_HOME_REGION: str = "CN"
TI2026_VENUE: str = "Shanghai"


@dataclass(frozen=True)
class MultiSourceConfig:
    """Конфиг мульти-источников силы (home LAN / патч / market stub)."""

    home_lan_elo: float = DEFAULT_HOME_LAN_ELO
    home_region: str = TI2026_HOME_REGION
    patch_start_ts: int = PATCH_741_START_TS
    # Optional market prior weight (P2 stub — unused until Polymarket wired).
    market_weight: float = 0.0


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


# P2 hook: Polymarket / market odds (not fetched yet).
def market_slot_prior_stub(_team_id: str, _slot_key: str) -> float | None:
    """Заглушка рыночного prior; None = источника нет."""
    return None
