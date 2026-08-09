"""Hero soft prior: roster meta-fit → pairwise logit shift (experimental).

Default off (``USE_HERO_SOFT_PRIOR=0``). Does not require model retrain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.hero_meta import load_hero_meta
from src.features.player_signatures import load_player_signatures

DEFAULT_LAMBDA: float = 0.25


def team_meta_fit(
    account_ids: list[int],
    signatures: dict | None = None,
) -> float:
    """Mean signature score across roster accounts (0.5 if unknown)."""
    signatures = signatures or load_player_signatures()
    players = signatures.get("players") or {}
    scores: list[float] = []
    for aid in account_ids:
        entry = players.get(str(aid)) or players.get(aid)
        if not entry:
            continue
        scores.append(float(entry.get("sig_wr", 0.5)))
    if not scores:
        return 0.5
    return float(np.mean(scores))


def apply_soft_prior(
    p: float,
    fit_r: float,
    fit_d: float,
    *,
    lambda_: float = DEFAULT_LAMBDA,
) -> float:
    """Logit-shift P(win) by λ * (fit_r - fit_d)."""
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    logit = float(np.log(p / (1.0 - p)))
    logit += float(lambda_) * (float(fit_r) - float(fit_d))
    return float(1.0 / (1.0 + np.exp(-logit)))


def apply_soft_prior_matrix(
    win_matrix: pd.DataFrame,
    lineups: dict[str, list[int]],
    *,
    lambda_: float = DEFAULT_LAMBDA,
    signatures: dict | None = None,
) -> pd.DataFrame:
    """Apply soft prior to every off-diagonal cell of a win matrix.

    Skips silently if signatures are missing or marked as fixture/synthetic.
    """
    try:
        signatures = signatures or load_player_signatures()
    except FileNotFoundError:
        return win_matrix.copy()
    source = str(signatures.get("source") or "").lower()
    # Refuse test fixtures accidentally left under data/hero/.
    if "fixture" in source or signatures.get("usable_in_production") is False:
        return win_matrix.copy()
    try:
        meta = load_hero_meta()
        meta_src = str(meta.get("source") or "").lower()
        if "fixture" in meta_src or meta.get("usable_in_production") is False:
            return win_matrix.copy()
    except FileNotFoundError:
        return win_matrix.copy()
    fits = {tid: team_meta_fit(ids, signatures) for tid, ids in lineups.items()}
    out = win_matrix.copy()
    for a in out.index:
        for b in out.columns:
            if a == b:
                continue
            out.loc[a, b] = apply_soft_prior(
                float(out.loc[a, b]),
                fits.get(a, 0.5),
                fits.get(b, 0.5),
                lambda_=lambda_,
            )
    return out


def hero_artifacts_available(
    meta_path: str | Path | None = None,
    sig_path: str | Path | None = None,
) -> bool:
    """True if both hero meta and signatures exist on disk."""
    from src.features.hero_meta import DEFAULT_HERO_META_PATH
    from src.features.player_signatures import DEFAULT_SIGNATURES_PATH

    return Path(meta_path or DEFAULT_HERO_META_PATH).exists() and Path(
        sig_path or DEFAULT_SIGNATURES_PATH
    ).exists()


def team_draft_meta_fit(
    hero_ids: list[int],
    *,
    meta: dict | None = None,
) -> float:
    """Mean hero meta WR for a known 5-hero draft (0.5 if unknown)."""
    if meta is None:
        try:
            meta = load_hero_meta()
        except FileNotFoundError:
            return 0.5
    heroes = meta.get("heroes") or {}
    scores: list[float] = []
    for hid in hero_ids:
        entry = heroes.get(str(hid)) or heroes.get(hid)
        if not entry:
            continue
        wr = entry.get("wr", entry.get("winrate", entry.get("pro_wr")))
        if wr is None:
            continue
        scores.append(float(wr))
    if not scores:
        return 0.5
    return float(np.mean(scores))


def apply_known_draft_logit_shift(
    p: float,
    radiant_heroes: list[int],
    dire_heroes: list[int],
    *,
    lambda_: float = DEFAULT_LAMBDA,
    meta: dict | None = None,
) -> float:
    """Logit-shift P(radiant win) using known draft meta-fit (HERO_DRAFT level A live).

    Skips silently if hero meta is missing / fixture-only.
    """
    try:
        meta = meta or load_hero_meta()
    except FileNotFoundError:
        return float(p)
    source = str(meta.get("source") or "").lower()
    if "fixture" in source or meta.get("usable_in_production") is False:
        return float(p)
    fit_r = team_draft_meta_fit(radiant_heroes, meta=meta)
    fit_d = team_draft_meta_fit(dire_heroes, meta=meta)
    return apply_soft_prior(p, fit_r, fit_d, lambda_=lambda_)
