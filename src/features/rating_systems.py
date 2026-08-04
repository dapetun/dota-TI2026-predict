"""Elo, Glicko-2, and TrueSkill rating systems for teams."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional


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
    initial_mu: float = 1500.0
    initial_rd: float = 350.0
    initial_vol: float = 0.06
    tau: float = 0.5
    ratings: Dict[int, dict] = field(default_factory=dict)

    def get(self, team_id: int) -> dict:
        if team_id not in self.ratings:
            self.ratings[team_id] = {
                "mu": self.initial_mu,
                "rd": self.initial_rd,
                "vol": self.initial_vol,
            }
        return self.ratings[team_id]

    def _g(self, rd: float) -> float:
        return 1.0 / np.sqrt(1.0 + 3.0 * (rd ** 2) / (np.pi ** 2 * 400.0 ** 2))

    def _E(self, mu: float, mu_j: float, rd_j: float) -> float:
        return 1.0 / (1.0 + np.exp(-self._g(rd_j) * (mu - mu_j) / 400.0))

    def update(self, winner_id: int, loser_id: int):
        w = self.get(winner_id)
        l = self.get(loser_id)

        g_w = self._g(w["rd"])
        g_l = self._g(l["rd"])
        E_w = self._E(w["mu"], l["mu"], l["rd"])
        E_l = self._E(l["mu"], w["mu"], w["rd"])

        v_w = 1.0 / (g_w ** 2 * E_w * (1 - E_w) + 1e-10)
        v_l = 1.0 / (g_l ** 2 * E_l * (1 - E_l) + 1e-10)

        w["mu"] += g_w * (1 - E_w) * v_w
        w["rd"] = max(150.0, w["rd"] * np.sqrt(max(1.0 - 1.0 / v_w, 0.01)))

        l["mu"] += g_l * (0 - E_l) * v_l
        l["rd"] = max(150.0, l["rd"] * np.sqrt(max(1.0 - 1.0 / v_l, 0.01)))


@dataclass
class TrueSkillRating:
    mu: float = 25.0
    sigma: float = 25.0 / 3.0
    beta: float = 25.0 / 6.0
    draw_prob: float = 0.1
    ratings: Dict[int, dict] = field(default_factory=dict)

    def get(self, team_id: int) -> dict:
        if team_id not in self.ratings:
            self.ratings[team_id] = {"mu": self.mu, "sigma": self.sigma}
        return self.ratings[team_id]

    def _v_win(self, eps: float) -> float:
        from scipy.stats import norm
        t = norm.cdf(eps)
        if t < 1e-10:
            return -eps + self.beta
        return norm.pdf(eps) / t

    def _w_win(self, eps: float, v: float) -> float:
        from scipy.stats import norm
        t = norm.cdf(eps)
        if t < 1e-10:
            return 1.0
        return v * (v + eps)

    def update(self, winner_id: int, loser_id: int):
        w = self.get(winner_id)
        l = self.get(loser_id)

        c = np.sqrt(2 * self.beta ** 2 + w["sigma"] ** 2 + l["sigma"] ** 2)
        eps = (w["mu"] - l["mu"]) / c

        v = self._v_win(eps)
        w_val = self._w_win(eps, v)

        w["mu"] += (w["sigma"] ** 2 / c) * v
        w["sigma"] *= np.sqrt(max(1.0 - w_val * (w["sigma"] ** 2 / c ** 2), 0.01))

        l["mu"] -= (l["sigma"] ** 2 / c) * v
        l["sigma"] *= np.sqrt(max(1.0 - w_val * (l["sigma"] ** 2 / c ** 2), 0.01))


def build_elo_history(
    team_matches: pd.DataFrame,
    k_factor: float = 32.0,
) -> pd.DataFrame:
    """Process all matches chronologically and return Elo history per team per match."""
    elo = EloRating(k_factor=k_factor)
    records = []

    for _, row in team_matches.sort_values("start_time").iterrows():
        team_a = int(row["team_id"])
        won_a = bool(row["team_won"])

        # Find opponent (from match_id group)
        records.append({
            "match_id": row["match_id"],
            "team_id": team_a,
            "elo_before": elo.get(team_a),
            "won": won_a,
        })

    # Process opponents in a second pass
    df = pd.DataFrame(records)
    elo2 = EloRating(k_factor=k_factor)

    for match_id, group in team_matches.sort_values("start_time").groupby("match_id"):
        if len(group) < 2:
            continue
        teams = group["team_id"].values
        winners = group[group["team_won"] == True]["team_id"].values
        losers = group[group["team_won"] == False]["team_id"].values

        if len(winners) > 0 and len(losers) > 0:
            elo2.update(int(winners[0]), int(losers[0]))

        for _, row in group.iterrows():
            df.loc[
                (df["match_id"] == row["match_id"]) & (df["team_id"] == row["team_id"]),
                "elo_after",
            ] = elo2.get(int(row["team_id"]))

    return df
