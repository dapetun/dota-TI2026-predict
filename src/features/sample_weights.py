"""Sample weighting: time decay + tournament tier."""

from __future__ import annotations

import numpy as np
import pandas as pd


def exponential_time_weights(
    start_times: np.ndarray | pd.Series,
    reference_time: int | float,
    half_life_days: float = 90.0,
) -> np.ndarray:
    """Weight samples by age relative to a reference timestamp.

    Half-life of 90 days means a match from 90 days ago gets weight 0.5
    compared to a match at the reference time (before tier multiplier).
    """
    times = np.asarray(start_times, dtype=float)
    age_days = np.maximum(0.0, (float(reference_time) - times) / 86400.0)
    # w = 0.5 ** (age / half_life) == exp(-ln2 * age / half_life)
    return np.power(0.5, age_days / half_life_days)


def compute_sample_weights(
    df: pd.DataFrame,
    reference_time: int | float | None = None,
    half_life_days: float = 90.0,
    tier_col: str = "tier_weight",
) -> np.ndarray:
    """Combine exponential time decay with tournament-tier multipliers.

    Why this scheme:
    - Esports meta drifts with patches; equal weight on old matches hurts.
    - High-tier LAN/TI results transfer better to TI group stage.
    - 90-day half-life keeps ~1 year of history informative but soft.
    """
    if df.empty:
        return np.array([])

    ref = float(reference_time) if reference_time is not None else float(df["start_time"].max())
    time_w = exponential_time_weights(df["start_time"], ref, half_life_days)
    tier_w = df[tier_col].to_numpy(dtype=float) if tier_col in df.columns else np.ones(len(df))
    weights = time_w * tier_w
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1e-6)
    weights = np.maximum(weights, 1e-6)
    # Normalize mean to 1.0 so learning rate scale stays stable.
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    return weights
