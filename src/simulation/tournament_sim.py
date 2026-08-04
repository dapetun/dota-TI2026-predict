"""Monte Carlo tournament simulation for TI 2026 Swiss Stage."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field


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
    n_rounds: int = 5
    qualify_top_n: int = 3
    elim_bottom_n: int = 3
    elimination_round_advance: int = 5


def get_eligible_opponents(
    state: SwissState,
    team: str,
) -> List[str]:
    """Get teams that can still be paired (same record, no rematches)."""
    w, l = state.records[team]
    played = {m[1] if m[0] == team else m[0] for m in state.match_history if team in (m[0], m[1])}

    eligible = []
    for t in state.teams:
        if t == team:
            continue
        if t in state.eliminated or t in state.qualified:
            continue
        if t in played:
            continue
        tw, tl = state.records[t]
        if (tw, tl) == (w, l):
            eligible.append(t)

    return eligible


def simulate_swiss_round(
    state: SwissState,
    win_matrix: pd.DataFrame,
    rng: np.random.Generator,
) -> SwissState:
    """Simulate one round of Swiss."""
    # Group teams by record
    record_groups = defaultdict(list)
    for team in state.teams:
        if team not in state.eliminated and team not in state.qualified:
            record_groups[state.records[team]].append(team)

    # Sort records for pairing (highest first)
    sorted_records = sorted(record_groups.keys(), key=lambda x: (-x[0], x[1]))

    paired_this_round = set()

    for record in sorted_records:
        if record in paired_this_round:
            continue

        teams_in_group = [t for t in record_groups[record] if t not in paired_this_round]
        rng.shuffle(teams_in_group)

        while len(teams_in_group) >= 2:
            team_a = teams_in_group.pop(0)

            # Find best opponent
            eligible = [t for t in teams_in_group if t not in paired_this_round]

            if not eligible:
                break

            # Prefer opponents from same record group
            same_record = [t for t in eligible if state.records[t] == record]
            if same_record:
                opponent = rng.choice(same_record)
            else:
                opponent = rng.choice(eligible)

            # Simulate match
            if team_a in win_matrix.index and opponent in win_matrix.columns:
                p_a = win_matrix.loc[team_a, opponent]
            else:
                p_a = 0.5

            winner = team_a if rng.random() < p_a else opponent
            loser = opponent if winner == team_a else team_a

            w_w, w_l = state.records[winner]
            l_w, l_l = state.records[loser]
            state.records[winner] = (w_w + 1, w_l)
            state.records[loser] = (l_w, l_l + 1)
            state.match_history.append((team_a, opponent, winner))

            paired_this_round.add(team_a)
            paired_this_round.add(opponent)
            teams_in_group = [t for t in teams_in_group if t not in paired_this_round]

    # Check qualification/elimination after round
    for team in state.teams:
        w, l = state.records[team]
        if w >= 3:
            state.qualified.add(team)
        elif l >= 3:
            state.eliminated.add(team)

    return state


def simulate_swiss_stage(
    win_matrix: pd.DataFrame,
    team_ids: list,
    config: SwissConfig = None,
    n_simulations: int = 100000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Run full Swiss stage Monte Carlo simulation."""
    if config is None:
        config = SwissConfig()

    rng = np.random.default_rng(rng_seed)

    # Track results
    final_records = defaultdict(lambda: defaultdict(int))
    qualification_counts = defaultdict(int)
    elimination_round_counts = defaultdict(int)
    eliminated_counts = defaultdict(int)
    win_counts = defaultdict(lambda: defaultdict(int))

    for sim in range(n_simulations):
        state = SwissState(teams=list(team_ids))

        for round_num in range(config.n_rounds):
            state = simulate_swiss_round(state, win_matrix, rng)

        # Record final state
        for team in team_ids:
            w, l = state.records[team]
            final_records[team][(w, l)] += 1

            if team in state.qualified:
                if w == 3 and l == 0:
                    qualification_counts[team] += 1  # 3-0
                elif w == 3 and l == 1:
                    qualification_counts[team] += 1  # 3-1
                elif w >= 3:
                    qualification_counts[team] += 1
            elif team in state.eliminated:
                eliminated_counts[team] += 1
            else:
                # Made elimination round
                elimination_round_counts[team] += 1

            for w_final in range(w + 1):
                win_counts[team][w_final] += 1

    # Aggregate results
    results = []
    for team in team_ids:
        records = final_records[team]
        total = sum(records.values())

        # Calculate expected wins
        exp_wins = sum(w * count for (w, l), count in records.items()) / total

        # Most likely record
        most_likely = max(records.items(), key=lambda x: x[1])[0]

        # Direct qualification probability
        direct_qual_prob = qualification_counts[team] / total

        # Elimination round probability
        elim_round_prob = elimination_round_counts[team] / total

        # Fully eliminated probability
        elim_prob = eliminated_counts[team] / total

        results.append({
            "team": team,
            "expected_wins": round(exp_wins, 2),
            "most_likely_record": f"{most_likely[0]}-{most_likely[1]}",
            "direct_qualification_pct": round(direct_qual_prob * 100, 1),
            "elimination_round_pct": round(elim_round_prob * 100, 1),
            "eliminated_pct": round(elim_prob * 100, 1),
            "prob_3_0": round(records.get((3, 0), 0) / total * 100, 1),
            "prob_3_1": round(records.get((3, 1), 0) / total * 100, 1),
            "prob_3_2_or_2_3": round(
                (records.get((3, 2), 0) + records.get((2, 3), 0)) / total * 100, 1
            ),
            "prob_2_3": round(records.get((2, 3), 0) / total * 100, 1),
            "prob_1_3": round(records.get((1, 3), 0) / total * 100, 1),
            "prob_0_3": round(records.get((0, 3), 0) / total * 100, 1),
        })

    results_df = pd.DataFrame(results)
    results_df.sort_values("expected_wins", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    return results_df


def simulate_elimination_round(
    candidates: List[str],
    win_matrix: pd.DataFrame,
    advance_n: int = 5,
    n_simulations: int = 50000,
    rng_seed: int = 42,
) -> Dict[str, float]:
    """Simulate elimination round for middle teams."""
    rng = np.random.default_rng(rng_seed)
    advance_counts = defaultdict(int)

    for _ in range(n_simulations):
        # Random single-elimination bracket or round-robin
        # Simplified: each pair plays, top advance_n by win rate
        wins = defaultdict(int)
        matches_played = defaultdict(int)

        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                if a in win_matrix.index and b in win_matrix.columns:
                    p = win_matrix.loc[a, b]
                else:
                    p = 0.5

                if rng.random() < p:
                    wins[a] += 1
                else:
                    wins[b] += 1
                matches_played[a] += 1
                matches_played[b] += 1

        # Sort by win rate
        rates = []
        for team in candidates:
            mp = matches_played[team]
            wr = wins[team] / mp if mp > 0 else 0.5
            rates.append((team, wr))

        rates.sort(key=lambda x: -x[1])
        for team, _ in rates[:advance_n]:
            advance_counts[team] += 1

    return {team: count / n_simulations for team, count in advance_counts.items()}
