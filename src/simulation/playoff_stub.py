"""Thin playoff Monte Carlo for Main Event (post-Swiss top-8).

Swiss remains the production surface. Call ``simulate_playoffs`` once the
qualified eight (and optional seeds) are known. ``simulate_playoffs_stub``
stays a safe no-op for pre-ME callers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def simulate_playoffs_stub(
    *_args: Any,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Safe no-op playoff stub for pre-ME / missing top-8."""
    return {
        "implemented": False,
        "reason": "Playoff bracket MC needs post-Swiss top-8 — see simulate_playoffs()",
        "teams": [],
        "champion_probs": {},
        "reach_final_probs": {},
    }


def _p_win(matrix: pd.DataFrame, a: str, b: str) -> float:
    """P(a beats b) from win matrix; default 0.5 if missing."""
    try:
        return float(np.clip(matrix.loc[a, b], 0.02, 0.98))
    except (KeyError, TypeError, ValueError):
        return 0.5


def _bo_series(
    matrix: pd.DataFrame,
    a: str,
    b: str,
    *,
    best_of: int,
    rng: np.random.Generator,
) -> str:
    """Simulate a BoN series; return winner team id."""
    need = best_of // 2 + 1
    wa = wb = 0
    while wa < need and wb < need:
        if rng.random() < _p_win(matrix, a, b):
            wa += 1
        else:
            wb += 1
    return a if wa > wb else b


def simulate_playoffs(
    win_matrix: pd.DataFrame,
    team_ids: list[str],
    *,
    n_simulations: int = 20_000,
    rng_seed: int = 42,
    best_of: int = 3,
    final_best_of: int = 5,
) -> dict[str, Any]:
    """Thin single-elim bracket MC for 4/8/16 teams (power-of-two).

    Pairing is seed order: 1vN, 2vN-1, … (list order = seeds). Returns
    champion / reach-final frequencies. Not a full TI double-elim — enough
    for a post-Swiss research surface.
    """
    teams = [str(t) for t in team_ids]
    n = len(teams)
    if n < 2 or (n & (n - 1)) != 0:
        return {
            "implemented": False,
            "reason": f"Need power-of-two team count ≥2, got {n}",
            "teams": teams,
            "champion_probs": {},
            "reach_final_probs": {},
        }

    rng = np.random.default_rng(rng_seed)
    champ = {t: 0 for t in teams}
    finalist = {t: 0 for t in teams}

    for _ in range(int(n_simulations)):
        round_teams = list(teams)
        while len(round_teams) > 1:
            nxt: list[str] = []
            is_final = len(round_teams) == 2
            bo = final_best_of if is_final else best_of
            for i in range(0, len(round_teams), 2):
                a, b = round_teams[i], round_teams[i + 1]
                if is_final:
                    finalist[a] += 1
                    finalist[b] += 1
                winner = _bo_series(win_matrix, a, b, best_of=bo, rng=rng)
                nxt.append(winner)
            round_teams = nxt
        champ[round_teams[0]] += 1

    n_sims = float(n_simulations) or 1.0
    return {
        "implemented": True,
        "format": f"single_elim_bo{best_of}_final_bo{final_best_of}",
        "n_simulations": int(n_simulations),
        "rng_seed": int(rng_seed),
        "teams": teams,
        "champion_probs": {t: champ[t] / n_sims for t in teams},
        "reach_final_probs": {t: finalist[t] / n_sims for t in teams},
    }
