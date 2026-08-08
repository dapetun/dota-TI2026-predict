"""Thin loader for configs/settings.yaml (optional overrides over code defaults).

Canonical numeric defaults live in code (e.g. RATING_HALF_LIFE_DAYS).
settings.yaml is documentation + optional override when present and valid.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.features.sample_weights import RATING_HALF_LIFE_DAYS

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = BASE_DIR / "configs" / "settings.yaml"

# Simulation defaults aligned with SWISS_CONFIG / export.
DEFAULT_N_SIMULATIONS: int = 50_000


@lru_cache(maxsize=4)
def load_settings(path: str | None = None) -> dict[str, Any]:
    """Load YAML settings; empty dict if missing or unreadable."""
    cfg_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"Warning: could not load settings from {cfg_path}: {exc}")
        return {}


def sample_half_life_days(settings: dict[str, Any] | None = None) -> float:
    """Half-life for training sample weights (days)."""
    cfg = settings if settings is not None else load_settings()
    data = cfg.get("data") or {}
    val = data.get("sample_weight_half_life_days")
    if val is None:
        return float(RATING_HALF_LIFE_DAYS)
    return float(val)


def n_simulations(settings: dict[str, Any] | None = None) -> int:
    """Monte Carlo sims for Swiss export."""
    cfg = settings if settings is not None else load_settings()
    sim = cfg.get("simulation") or {}
    val = sim.get("n_simulations")
    if val is None:
        return DEFAULT_N_SIMULATIONS
    return int(val)
