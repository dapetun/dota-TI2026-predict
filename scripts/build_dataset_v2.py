"""Fast enhanced dataset - running Elo, no O(n²) recomputation."""
import json, glob, os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
FEATURES_DIR = BASE_DIR / "data" / "features"

TOURNAMENT_TIERS = {
    "TI10_2021": 4, "TI11_2022": 4, "TI12_2023": 4, "TI13_2024": 4,
    "TI14_2025": 4, "TI2026": 4,
    "DreamLeague_S28": 3, "DreamLeague_S29": 3,
    "PGL_Wallachia_S8": 3, "PGL_Wallachia_S7": 3,
    "BLAST_SLAM_VI": 3, "BLAST_SLAM_VII": 3,
    "ESL_Birmingham_2026": 3, "EWC_2026": 3,
}


def load_matches():
    all_matches = []
    for f in sorted(glob.glob(str(RAW_DIR / "*_matches.json"))):
        tournament = Path(f).stem.replace("_matches", "")
        tier = TOURNAMENT_TIERS.get(tournament, 2)
        with open(f) as fh:
            data = json.load(fh)
            for m in data:
                m["tournament"] = tournament
                m["tier"] = tier
            all_matches.extend(data)
    return sorted(all_matches, key=lambda x: x.get("start_time", 0))


def build_features_fast():
    matches = load_matches()
    print(f"Loaded {len(matches)} matches")

    # Running state
    elo = defaultdict(lambda: 1500.0)
    match_history = defaultdict(list)  # team_id -> [(time, won, tier)]
    h2h_cache = defaultdict(list)      # (t1,t2) -> [(won_by_t1, time)]

    features_list = []
    k_base = 20

    for m in matches:
        r_id = str(m.get("radiant_team_id", ""))
        d_id = str(m.get("dire_team_id", ""))
        if not r_id or not d_id or r_id == d_id:
            continue

        t = m.get("start_time", 0)
        tier = m.get("tier", 2)
        r_elo = elo[r_id]
        d_elo = elo[d_id]

        # Expected scores
        r_exp = 1.0 / (1 + 10 ** ((d_elo - r_elo) / 400))
        d_exp = 1 - r_exp

        # Actual
        r_win = bool(m.get("radiant_win"))
        r_act = 1.0 if r_win else 0.0
        d_act = 1 - r_act

        # Form (last 15 matches, time-decayed)
        def get_form(tid):
            hist = match_history[tid][-15:]
            if not hist:
                return 0.5, 0, 0, 0.5
            # Decayed WR
            total_w, total = 0, 0
            for i, (ht, hw, htier) in enumerate(hist):
                w = 0.95 ** (len(hist) - 1 - i) * htier
                total += w
                if hw:
                    total_w += w
            wr = total_w / total if total > 0 else 0.5
            # Recent 5 WR
            r5 = hist[-5:] if len(hist) >= 5 else hist
            wr5 = sum(1 for _, hw, _ in r5 if hw) / len(r5) if r5 else 0.5
            # Streak
            streak = 0
            for _, hw, _ in reversed(hist):
                if hw:
                    streak += 1
                else:
                    break
            # Peak
            peak = max(sum(1 for _, hw, _ in hist[:i+1] if hw) / (i+1) for i in range(len(hist))) if hist else 0.5
            return wr, wr5, streak, peak

        r_wr, r_wr5, r_streak, r_peak = get_form(r_id)
        d_wr, d_wr5, d_streak, d_peak = get_form(d_id)

        # H2H
        h2h_key = tuple(sorted([r_id, d_id]))
        h2h_list = h2h_cache[h2h_key][-10:]
        if h2h_list:
            # Filter: who was on which side?
            h2h_wr = sum(1 for hid, hr_won in h2h_list if (hid == r_id and hr_won) or (hid == d_id and not hr_won)) / len(h2h_list)
        else:
            h2h_wr = 0.5

        # Games played
        r_gp = len(match_history[r_id])
        d_gp = len(match_history[d_id])

        # Tier
        r_avg_tier = np.mean([x[2] for x in match_history[r_id][-10:]]) if match_history[r_id] else 2
        d_avg_tier = np.mean([x[2] for x in match_history[d_id][-10:]]) if match_history[d_id] else 2

        # Win probability from current Elo
        elo_prob = 1 / (1 + 10 ** ((d_elo - r_elo) / 400))

        features_list.append({
            "match_id": m.get("match_id"),
            "tournament": m.get("tournament", ""),
            "start_time": t,
            "date": datetime.fromtimestamp(t).strftime("%Y-%m-%d") if t else "",
            "radiant_team_id": r_id,
            "dire_team_id": d_id,
            # Elo
            "r_elo": r_elo,
            "d_elo": d_elo,
            "diff_elo": r_elo - d_elo,
            "abs_elo_diff": abs(r_elo - d_elo),
            "elo_prob": elo_prob,
            # Form
            "r_wr": r_wr,
            "d_wr": d_wr,
            "diff_wr": r_wr - d_wr,
            "r_wr5": r_wr5,
            "d_wr5": d_wr5,
            "diff_wr5": r_wr5 - d_wr5,
            "r_streak": r_streak,
            "d_streak": d_streak,
            "diff_streak": r_streak - d_streak,
            "r_peak": r_peak,
            "d_peak": d_peak,
            "diff_peak": r_peak - d_peak,
            # H2H
            "h2h_wr": h2h_wr,
            "diff_h2h": h2h_wr - 0.5,
            # Experience
            "r_gp": r_gp,
            "d_gp": d_gp,
            "diff_gp": r_gp - d_gp,
            # Tier
            "tier": tier,
            "r_avg_tier": r_avg_tier,
            "d_avg_tier": d_avg_tier,
            "diff_tier": r_avg_tier - d_avg_tier,
            # Target
            "radiant_win": 1 if r_win else 0,
        })

        # Update running state AFTER features are computed
        k_tiered = k_base * (1 + (tier - 2) * 0.2)
        elo[r_id] += k_tiered * (r_act - r_exp)
        elo[d_id] += k_tiered * (d_act - d_exp)
        match_history[r_id].append((t, r_win, tier))
        match_history[d_id].append((t, not r_win, tier))
        h2h_cache[h2h_key].append((r_id, r_win))

    df = pd.DataFrame(features_list)
    df = df[df["r_gp"] >= 10].reset_index(drop=True)

    print(f"\nDataset: {len(df)} matches")
    print(f"Date: {df['date'].min()} to {df['date'].max()}")
    print(f"Teams: {df['radiant_team_id'].nunique()}")
    print(f"Tournaments: {df['tournament'].nunique()}")
    print(f"Win rate: {df['radiant_win'].mean():.3f}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FEATURES_DIR / "match_features_v2.csv"
    df.to_csv(out, index=False)
    print(f"Saved to {out}")
    return df


if __name__ == "__main__":
    build_features_fast()
