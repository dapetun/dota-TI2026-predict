"""Monte Carlo simulation for TI Swiss + Elimination Round.

TI group stage (compendium / prediction board):
- Swiss: first to 4 wins OR 4 losses, max 5 rounds, Bo3 series.
- After Swiss: 4-0 / 4-1 advance; 0-4 / 1-4 eliminated;
  remaining (typically 3-2 and 2-3) play Elimination Round.
- ER: 3-2 choose among 2-3 (5 Bo3); winners advance (5 of 10).

Fantasy board slots (16 teams):
  4-0 ×1 | 4-1 ×2 | advance ×5 | eliminate ×5 | 1-4 ×2 | 0-4 ×1
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SwissState:
    """State of a Swiss stage tournament."""

    teams: List[str]
    records: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    match_history: List[Tuple[str, str, str]] = field(default_factory=list)
    eliminated: set = field(default_factory=set)
    qualified: set = field(default_factory=set)

    def __post_init__(self):
        for team in self.teams:
            if team not in self.records:
                self.records[team] = (0, 0)


@dataclass
class SwissConfig:
    """TI Swiss + Elimination Round configuration."""

    n_rounds: int = 5
    wins_to_qualify: int = 4
    losses_to_eliminate: int = 4
    elimination_round_advance: int = 5
    # Kept for backwards-compatible kwargs from older configs.
    qualify_top_n: int = 3
    elim_bottom_n: int = 3


def _series_winner(
    team_a: str,
    team_b: str,
    win_matrix: pd.DataFrame,
    rng: np.random.Generator,
) -> str:
    """Resolve a Bo3 series from map-win probabilities in win_matrix."""
    if team_a in win_matrix.index and team_b in win_matrix.columns:
        p_a = float(win_matrix.loc[team_a, team_b])
    else:
        p_a = 0.5
    wins_a = wins_b = 0
    while wins_a < 2 and wins_b < 2:
        if rng.random() < p_a:
            wins_a += 1
        else:
            wins_b += 1
    return team_a if wins_a > wins_b else team_b


def simulate_swiss_round(
    state: SwissState,
    win_matrix: pd.DataFrame,
    rng: np.random.Generator,
    config: SwissConfig,
) -> SwissState:
    """Simulate one Swiss round with same-record pairing."""
    record_groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for team in state.teams:
        if team not in state.eliminated and team not in state.qualified:
            record_groups[state.records[team]].append(team)

    sorted_records = sorted(record_groups.keys(), key=lambda x: (-x[0], x[1]))
    paired_this_round: set[str] = set()

    for record in sorted_records:
        teams_in_group = [t for t in record_groups[record] if t not in paired_this_round]
        rng.shuffle(teams_in_group)

        while len(teams_in_group) >= 2:
            team_a = teams_in_group.pop(0)
            eligible = [t for t in teams_in_group if t not in paired_this_round]
            if not eligible:
                break

            # Avoid rematches when possible.
            played = {
                m[1] if m[0] == team_a else m[0]
                for m in state.match_history
                if team_a in (m[0], m[1])
            }
            fresh = [t for t in eligible if t not in played]
            pool = fresh or eligible
            opponent = rng.choice(pool)

            winner = _series_winner(team_a, opponent, win_matrix, rng)
            loser = opponent if winner == team_a else team_a

            w_w, w_l = state.records[winner]
            l_w, l_l = state.records[loser]
            state.records[winner] = (w_w + 1, w_l)
            state.records[loser] = (l_w, l_l + 1)
            state.match_history.append((team_a, opponent, winner))

            paired_this_round.add(team_a)
            paired_this_round.add(opponent)
            teams_in_group = [t for t in teams_in_group if t not in paired_this_round]

        # Odd leftover in a record bucket gets an implicit bye this round
        # (no auto-win recorded). Intentional MC simplification vs real Swiss.

    for team in state.teams:
        w, l = state.records[team]
        if w >= config.wins_to_qualify:
            state.qualified.add(team)
        elif l >= config.losses_to_eliminate:
            state.eliminated.add(team)

    return state


def _team_strength(team: str, win_matrix: pd.DataFrame) -> float:
    """Mean map-win probability vs other indexed opponents (MC ranking proxy)."""
    if team not in win_matrix.index:
        return 0.5
    row = win_matrix.loc[team]
    others = [c for c in win_matrix.columns if c != team]
    if not others:
        return 0.5
    return float(row[others].mean())


def simulate_elimination_round(
    candidates: List[str],
    win_matrix: pd.DataFrame,
    advance_n: int = 5,
    rng: np.random.Generator | None = None,
    records: Dict[str, Tuple[int, int]] | None = None,
) -> List[str]:
    """Elimination Round: ranked 3-2 teams each play a chosen 2-3 opponent.

    Official TI rules: best 3-2 picks any remaining 2-3, then next-best, etc.
    MC approximation: order 3-2 by strength; each picks the weakest remaining 2-3.
    Winners advance (typically 5 of 10). Falls back to random pairing if records
    are missing or the 3-2 / 2-3 split is incomplete.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if records is not None:
        three_two = [t for t in candidates if records.get(t) == (3, 2)]
        two_three = [t for t in candidates if records.get(t) == (2, 3)]
        leftover = [
            t for t in candidates if t not in three_two and t not in two_three
        ]
        if three_two and two_three and not leftover:
            three_two = sorted(
                three_two,
                key=lambda t: _team_strength(t, win_matrix),
                reverse=True,
            )
            remaining_23 = list(two_three)
            winners: list[str] = []
            for picker in three_two:
                if not remaining_23:
                    break
                remaining_23.sort(key=lambda t: _team_strength(t, win_matrix))
                opponent = remaining_23.pop(0)
                winners.append(_series_winner(picker, opponent, win_matrix, rng))
            # Extra 3-2 without a 2-3 foe (should not happen in standard 16-team Swiss).
            for picker in three_two[len(winners) :]:
                winners.append(picker)
            return winners[:advance_n]

    remaining = list(candidates)
    rng.shuffle(remaining)
    while len(remaining) > advance_n:
        next_round: list[str] = []
        i = 0
        while i < len(remaining):
            if i + 1 >= len(remaining):
                next_round.append(remaining[i])
                break
            a, b = remaining[i], remaining[i + 1]
            next_round.append(_series_winner(a, b, win_matrix, rng))
            i += 2
        remaining = next_round
    return remaining[:advance_n]


def sample_strength_adjusted_matrix(
    win_matrix: pd.DataFrame,
    team_strengths: dict[str, dict],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Resample pairwise P(win) from latent strengths ~ N(μ, σ).

    ``team_strengths[tid]`` should expose ``mu``/``sigma`` (Elo-like) or
    ``glicko_mu``/``glicko_rd``. Bradley-Terry on sampled strengths adjusts
    the base matrix multiplicatively toward the sampled logit.
    """
    teams = list(win_matrix.index)
    sampled: dict[str, float] = {}
    for tid in teams:
        s = team_strengths.get(tid) or {}
        mu = float(s.get("mu", s.get("elo_shrunk", s.get("glicko_mu", 1500.0))))
        sigma = float(s.get("sigma", s.get("glicko_rd", s.get("strength_sigma", 50.0))))
        sigma = max(1.0, sigma)
        sampled[tid] = float(mu + rng.normal(0.0, sigma))

    out = win_matrix.copy().astype(float)
    for a in teams:
        for b in teams:
            if a == b:
                continue
            # BT from sampled strengths
            sa, sb = sampled[a], sampled[b]
            p_bt = sa / (sa + sb) if (sa + sb) > 0 else 0.5
            # Also logistic on Elo-ish scale
            p_elo = 1.0 / (1.0 + 10 ** ((sb - sa) / 400.0))
            p_sample = 0.5 * (p_bt + p_elo)
            p_base = float(win_matrix.loc[a, b])
            # Blend: keep model structure, inject uncertainty
            out.loc[a, b] = float(np.clip(0.5 * p_base + 0.5 * p_sample, 0.02, 0.98))
    return out


def simulate_swiss_stage(
    win_matrix: pd.DataFrame,
    team_ids: list,
    config: SwissConfig | None = None,
    n_simulations: int = 100000,
    rng_seed: int = 42,
    *,
    team_strengths: dict[str, dict] | None = None,
    sample_uncertainty: bool = False,
) -> pd.DataFrame:
    """Run Swiss + Elimination Round Monte Carlo.

    When ``sample_uncertainty`` and ``team_strengths`` are set, each simulation
    draws latent strengths (Glicko RD / σ) and adjusts P(win) before pairing.
    """
    if config is None:
        config = SwissConfig()

    rng = np.random.default_rng(rng_seed)
    final_records: dict[str, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    slot_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    qualify_counts: dict[str, int] = defaultdict(int)
    eliminated_counts: dict[str, int] = defaultdict(int)
    er_advance_counts: dict[str, int] = defaultdict(int)
    er_elim_counts: dict[str, int] = defaultdict(int)

    for _ in range(n_simulations):
        state = SwissState(teams=list(team_ids))
        matrix = win_matrix
        if sample_uncertainty and team_strengths:
            matrix = sample_strength_adjusted_matrix(win_matrix, team_strengths, rng)

        for _round in range(config.n_rounds):
            active = [
                t
                for t in state.teams
                if t not in state.qualified and t not in state.eliminated
            ]
            if len(active) < 2:
                break
            state = simulate_swiss_round(state, matrix, rng, config)

        er_pool = [
            t
            for t in team_ids
            if t not in state.qualified and t not in state.eliminated
        ]
        er_advancers = set(
            simulate_elimination_round(
                er_pool,
                matrix,
                advance_n=config.elimination_round_advance,
                rng=rng,
                records=state.records,
            )
        ) if er_pool else set()

        for team in team_ids:
            w, l = state.records[team]
            final_records[team][(w, l)] += 1
            rec = f"{w}-{l}"

            if team in state.qualified:
                qualify_counts[team] += 1
                if rec == "4-0":
                    slot_counts[team]["4-0"] += 1
                elif rec == "4-1":
                    slot_counts[team]["4-1"] += 1
                else:
                    # Rare edge (e.g. 4-2 if config allows more rounds)
                    slot_counts[team]["advance"] += 1
            elif team in state.eliminated:
                eliminated_counts[team] += 1
                if rec == "0-4":
                    slot_counts[team]["0-4"] += 1
                elif rec == "1-4":
                    slot_counts[team]["1-4"] += 1
                else:
                    slot_counts[team]["eliminate"] += 1
            elif team in er_advancers:
                qualify_counts[team] += 1
                er_advance_counts[team] += 1
                slot_counts[team]["advance"] += 1
            else:
                eliminated_counts[team] += 1
                er_elim_counts[team] += 1
                slot_counts[team]["eliminate"] += 1

    results = []
    for team in team_ids:
        records = final_records[team]
        total = sum(records.values()) or 1
        exp_wins = sum(w * count for (w, l), count in records.items()) / total
        most_likely = max(records.items(), key=lambda x: x[1])[0]
        slots = slot_counts[team]

        def pct(key: str) -> float:
            return round(slots.get(key, 0) / total * 100, 1)

        def rec_pct(rec: tuple[int, int]) -> float:
            return round(records.get(rec, 0) / total * 100, 1)

        results.append(
            {
                "team": team,
                "expected_wins": round(exp_wins, 2),
                "most_likely_record": f"{most_likely[0]}-{most_likely[1]}",
                "direct_qualification_pct": round(qualify_counts[team] / total * 100, 1),
                "eliminated_pct": round(eliminated_counts[team] / total * 100, 1),
                "elimination_round_pct": round(
                    (er_advance_counts[team] + er_elim_counts[team]) / total * 100, 1
                ),
                "prob_4_0": pct("4-0"),
                "prob_4_1": pct("4-1"),
                "prob_advance": pct("advance"),
                "prob_eliminate": pct("eliminate"),
                "prob_1_4": pct("1-4"),
                "prob_0_4": pct("0-4"),
                # Raw Swiss terminal records (before ER remapping)
                "swiss_4_0": rec_pct((4, 0)),
                "swiss_4_1": rec_pct((4, 1)),
                "swiss_3_2": rec_pct((3, 2)),
                "swiss_2_3": rec_pct((2, 3)),
                "swiss_1_4": rec_pct((1, 4)),
                "swiss_0_4": rec_pct((0, 4)),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.sort_values("direct_qualification_pct", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    return results_df


# Fantasy / compendium board capacities for TI predictions UI.
FANTASY_BOARD_SLOTS = {
    "undefeated": {"label": "4–0", "capacity": 1, "tone": "ok"},
    "one_loss": {"label": "4–1", "capacity": 2, "tone": "ok"},
    "advance": {"label": "Проход", "capacity": 5, "tone": "warn"},
    "eliminate": {"label": "Выбывание", "capacity": 5, "tone": "warn"},
    "one_win": {"label": "1–4", "capacity": 2, "tone": "bad"},
    "winless": {"label": "0–4", "capacity": 1, "tone": "bad"},
}


def assign_fantasy_board(predictions: list[dict]) -> dict[str, list[dict]]:
    """Fill fixed-capacity prediction board from ranked qualify probabilities.

    Order: best teams → 4-0, 4-1, advance; worst → 0-4, 1-4, eliminate.
    """
    ranked = sorted(
        predictions,
        key=lambda p: (-p["qualify_pct"], p.get("power_rank", 99)),
    )
    board: dict[str, list[dict]] = {k: [] for k in FANTASY_BOARD_SLOTS}
    # Top side
    order_top = (
        [("undefeated", 1), ("one_loss", 2), ("advance", 5)]
    )
    # Bottom side (worst first into 0-4)
    order_bottom = (
        [("winless", 1), ("one_win", 2), ("eliminate", 5)]
    )

    idx = 0
    for slot, n in order_top:
        for _ in range(n):
            if idx >= len(ranked):
                break
            p = ranked[idx]
            idx += 1
            board[slot].append(_board_entry(p, slot))

    remaining = ranked[idx:]
    remaining_rev = list(reversed(remaining))
    bidx = 0
    for slot, n in order_bottom:
        for _ in range(n):
            if bidx >= len(remaining_rev):
                break
            p = remaining_rev[bidx]
            bidx += 1
            board[slot].append(_board_entry(p, slot))

    return board


def _board_entry(p: dict, slot: str) -> dict:
    prob_key = {
        "undefeated": "prob_4_0",
        "one_loss": "prob_4_1",
        "advance": "prob_advance",
        "eliminate": "prob_eliminate",
        "one_win": "prob_1_4",
        "winless": "prob_0_4",
    }[slot]
    return {
        "id": p["id"],
        "name": p["name"],
        "short": p["short"],
        "record": FANTASY_BOARD_SLOTS[slot]["label"],
        "qualify_pct": p["qualify_pct"],
        "slot_pct": float(p.get(prob_key, 0.0)),
    }
