"""Elo and Glicko-2 team ratings.

``GlickoRating`` implements Mark Glickman's Glicko-2 (μ/φ/σ) with Illinois
volatility iteration. Each ``update`` is one rating period with a single game
(continuous match stream). ``advance_inactive`` inflates RD between periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

# Glicko-2 scale: rating R = SCALE * μ + 1500, RD = SCALE * φ
GLICKO2_SCALE: float = 173.7178
GLICKO2_TAU: float = 0.5
GLICKO2_EPSILON: float = 1e-6
GLICKO2_INITIAL_MU: float = 1500.0
GLICKO2_INITIAL_RD: float = 350.0
GLICKO2_INITIAL_VOL: float = 0.06
GLICKO2_MIN_RD: float = 30.0
GLICKO2_MAX_RD: float = 350.0


@dataclass
class EloRating:
    k_factor: float = 32.0
    initial: float = 1500.0
    ratings: Dict[int, float] = field(default_factory=dict)

    def get(self, team_id: int) -> float:
        return self.ratings.get(team_id, self.initial)

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, winner_id: int, loser_id: int, is_draw: bool = False):
        ra = self.get(winner_id)
        rb = self.get(loser_id)
        ea = self.expected(ra, rb)
        eb = self.expected(rb, ra)

        if is_draw:
            sa, sb = 0.5, 0.5
        else:
            sa, sb = 1.0, 0.0

        self.ratings[winner_id] = ra + self.k_factor * (sa - ea)
        self.ratings[loser_id] = rb + self.k_factor * (sb - eb)


@dataclass
class GlickoRating:
    """Canonical Glicko-2 μ/RD/volatility (Glickman)."""

    initial_mu: float = GLICKO2_INITIAL_MU
    initial_rd: float = GLICKO2_INITIAL_RD
    initial_vol: float = GLICKO2_INITIAL_VOL
    tau: float = GLICKO2_TAU
    ratings: Dict[int, dict] = field(default_factory=dict)

    def get(self, team_id: int) -> dict:
        if team_id not in self.ratings:
            self.ratings[team_id] = {
                "mu": self.initial_mu,
                "rd": self.initial_rd,
                "vol": self.initial_vol,
            }
        return self.ratings[team_id]

    def _to_glicko2(self, mu: float, rd: float) -> tuple[float, float]:
        return (mu - 1500.0) / GLICKO2_SCALE, rd / GLICKO2_SCALE

    def _from_glicko2(self, mu: float, phi: float) -> tuple[float, float]:
        return 1500.0 + GLICKO2_SCALE * mu, GLICKO2_SCALE * phi

    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / np.sqrt(1.0 + 3.0 * phi**2 / (np.pi**2))

    @staticmethod
    def _E(mu: float, mu_j: float, phi_j: float) -> float:
        return 1.0 / (1.0 + np.exp(-GlickoRating._g(phi_j) * (mu - mu_j)))

    def _update_one(
        self,
        team_id: int,
        opp_mu: float,
        opp_rd: float,
        score: float,
    ) -> None:
        """One-period Glicko-2 update vs a single opponent (score in {0, 0.5, 1})."""
        me = self.get(team_id)
        mu, phi = self._to_glicko2(me["mu"], me["rd"])
        sigma = float(me["vol"])
        mu_j, phi_j = self._to_glicko2(opp_mu, opp_rd)

        g_j = self._g(phi_j)
        e_j = self._E(mu, mu_j, phi_j)
        v = 1.0 / (g_j**2 * e_j * (1.0 - e_j) + 1e-12)
        delta = v * g_j * (score - e_j)

        a = np.log(sigma**2)
        tau = self.tau

        def f(x: float) -> float:
            ex = np.exp(x)
            num = ex * (delta**2 - phi**2 - v - ex)
            den = 2.0 * (phi**2 + v + ex) ** 2
            return num / den - (x - a) / (tau**2)

        # Illinois algorithm for volatility (Glickman §3.6)
        A = a
        if delta**2 > phi**2 + v:
            B = np.log(delta**2 - phi**2 - v)
        else:
            k = 1
            B = a - k * tau
            while f(B) < 0:
                k += 1
                B = a - k * tau
                if k > 50:
                    break

        fA, fB = f(A), f(B)
        for _ in range(50):
            if abs(B - A) <= GLICKO2_EPSILON:
                break
            C = A + (A - B) * fA / (fB - fA + 1e-18)
            fC = f(C)
            if fC * fB <= 0:
                A, fA = B, fB
            else:
                fA /= 2.0
            B, fB = C, fC

        sigma_prime = float(np.exp(A / 2.0))
        phi_star = np.sqrt(phi**2 + sigma_prime**2)
        phi_prime = 1.0 / np.sqrt(1.0 / phi_star**2 + 1.0 / v)
        mu_prime = mu + phi_prime**2 * g_j * (score - e_j)

        new_mu, new_rd = self._from_glicko2(mu_prime, phi_prime)
        me["mu"] = float(new_mu)
        me["rd"] = float(np.clip(new_rd, GLICKO2_MIN_RD, GLICKO2_MAX_RD))
        me["vol"] = sigma_prime

    def advance_inactive(self, team_id: int, n_periods: int = 1) -> None:
        """Inflate RD for inactive periods (φ' = √(φ² + σ²) per period)."""
        if n_periods <= 0:
            return
        me = self.get(team_id)
        mu, phi = self._to_glicko2(me["mu"], me["rd"])
        sigma = float(me["vol"])
        for _ in range(int(n_periods)):
            phi = np.sqrt(phi**2 + sigma**2)
        _, new_rd = self._from_glicko2(mu, phi)
        me["rd"] = float(np.clip(new_rd, GLICKO2_MIN_RD, GLICKO2_MAX_RD))

    def update(self, winner_id: int, loser_id: int) -> None:
        """Update both teams after a decisive match (winner score=1, loser=0)."""
        w = self.get(winner_id)
        l = self.get(loser_id)
        # Snapshot pre-match ratings so both sides update against the same prior.
        w_mu, w_rd = float(w["mu"]), float(w["rd"])
        l_mu, l_rd = float(l["mu"]), float(l["rd"])
        self._update_one(winner_id, l_mu, l_rd, 1.0)
        self._update_one(loser_id, w_mu, w_rd, 0.0)


def build_elo_history(
    team_matches: pd.DataFrame,
    k_factor: float = 32.0,
) -> pd.DataFrame:
    """Process all matches chronologically and return Elo history per team per match."""
    records = []
    elo2 = EloRating(k_factor=k_factor)

    for match_id, group in team_matches.sort_values("start_time").groupby("match_id"):
        if len(group) < 2:
            continue
        winners = group[group["team_won"] == True]["team_id"].values  # noqa: E712
        losers = group[group["team_won"] == False]["team_id"].values  # noqa: E712

        before = {int(row["team_id"]): elo2.get(int(row["team_id"])) for _, row in group.iterrows()}
        if len(winners) > 0 and len(losers) > 0:
            elo2.update(int(winners[0]), int(losers[0]))

        for _, row in group.iterrows():
            tid = int(row["team_id"])
            records.append(
                {
                    "match_id": match_id,
                    "team_id": tid,
                    "elo_before": before[tid],
                    "elo_after": elo2.get(tid),
                    "won": bool(row["team_won"]),
                }
            )

    return pd.DataFrame(records)
