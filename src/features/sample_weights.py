"""Sample weighting: time decay + tournament tier + cold-start downweight.

Defaults (v0.3):
  - rating / sample half-life ≈ 210d (long continuity)
  - form features use ~40d half-life inside match_features (separate)
  - patch 7.41 window up-weight (PATCH_IN_MULT) via patch_start_ts
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ti2026.multisource import PATCH_741_START_TS, PATCH_IN_MULT

# Rating / training sample half-life (days). Form uses FORM_HALF_LIFE_DAYS elsewhere.
RATING_HALF_LIFE_DAYS: float = 210.0
# Down-weight matches where both sides are cold-start.
COLD_START_GP_THRESHOLD: int = 8
COLD_START_WEIGHT: float = 0.55


def exponential_time_weights(
    start_times: np.ndarray | pd.Series,
    reference_time: int | float,
    half_life_days: float = RATING_HALF_LIFE_DAYS,
) -> np.ndarray:
    """Weight samples by age relative to a reference timestamp.

    Half-life of 210 days keeps ~1–1.5 years of history informative for ratings.
    """
    times = np.asarray(start_times, dtype=float)
    age_days = np.maximum(0.0, (float(reference_time) - times) / 86400.0)
    return np.power(0.5, age_days / half_life_days)


def compute_sample_weights(
    df: pd.DataFrame,
    reference_time: int | float | None = None,
    half_life_days: float = RATING_HALF_LIFE_DAYS,
    tier_col: str = "tier_weight",
    *,
    cold_start_threshold: int = COLD_START_GP_THRESHOLD,
    cold_start_weight: float = COLD_START_WEIGHT,
    patch_mult: float = PATCH_IN_MULT,
    patch_start_ts: int | None = PATCH_741_START_TS,
) -> np.ndarray:
    """Combine exponential time decay with tournament-tier / patch multipliers.

    Why this scheme:
    - Esports meta drifts with patches; equal weight on old matches hurts.
    - High-tier LAN/TI results transfer better to TI group stage.
    - 210-day half-life keeps rating continuity; form is decayed separately (~40d).
    - Both-sides cold-start matches get down-weighted.
    - Matches after patch_start_ts get ``patch_mult`` (default PATCH_IN_MULT).
    """
    if df.empty:
        return np.array([])

    ref = float(reference_time) if reference_time is not None else float(df["start_time"].max())
    time_w = exponential_time_weights(df["start_time"], ref, half_life_days)
    tier_w = df[tier_col].to_numpy(dtype=float) if tier_col in df.columns else np.ones(len(df))
    weights = time_w * tier_w

    if "r_gp" in df.columns and "d_gp" in df.columns:
        cold = (df["r_gp"].to_numpy(dtype=float) < cold_start_threshold) & (
            df["d_gp"].to_numpy(dtype=float) < cold_start_threshold
        )
        weights = np.where(cold, weights * cold_start_weight, weights)

    if patch_start_ts is not None and patch_mult != 1.0 and "start_time" in df.columns:
        in_patch = df["start_time"].to_numpy(dtype=float) >= float(patch_start_ts)
        weights = np.where(in_patch, weights * patch_mult, weights)

    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1e-6)
    weights = np.maximum(weights, 1e-6)
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    return weights
