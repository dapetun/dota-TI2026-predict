"""Build training dataset from downloaded match data.

Phase 1: Process match lists → team-level features (fast, no API calls)
Phase 2: Download match details → player/hero features (slow, rate-limited)
Phase 3: Combine into final training dataset
"""
import json
import time
import requests
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
FEATURES_DIR = BASE_DIR / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from src.ti2026.teams import TI2026_TEAMS, ALIAS_TO_CANONICAL, normalize_team_name


# ──────────────────────────────────────────────
#  Team OpenDota IDs (from match data)
# ──────────────────────────────────────────────
# We'll discover these from the match data itself
TEAM_ID_TO_NAME = {}
TEAM_NAME_TO_ID = {}


def discover_team_ids():
    """Build team_id → name mapping from team_id_map.json."""
    global TEAM_ID_TO_NAME, TEAM_NAME_TO_ID

    map_file = RAW_DIR / "team_id_map.json"
    if map_file.exists():
        with open(map_file) as f:
            raw_map = json.load(f)
        for tid_str, name in raw_map.items():
            tid = int(tid_str)
            TEAM_ID_TO_NAME[tid] = name
            TEAM_NAME_TO_ID[name.lower()] = tid

    print(f"Loaded {len(TEAM_ID_TO_NAME)} team IDs from map")


def normalize_match_team_name(name):
    """Try to map a match team name to a TI2026 canonical name.
    Returns the canonical TI2026 name if it's a TI2026 team, otherwise the original name."""
    if not name:
        return None
    name_lower = name.lower().strip()

    # Direct alias lookup
    if name_lower in ALIAS_TO_CANONICAL:
        return ALIAS_TO_CANONICAL[name_lower]

    # Fuzzy matching
    for canonical, info in TI2026_TEAMS.items():
        if name_lower == canonical.lower():
            return canonical
        if name_lower == info["full_name"].lower():
            return canonical
        for alias in info["aliases"]:
            if name_lower == alias.lower():
                return canonical
        if name_lower in info["full_name"].lower() or info["full_name"].lower() in name_lower:
            return canonical
        for alias in info["aliases"]:
            if name_lower in alias.lower() or alias.lower() in name_lower:
                return canonical

    return name


# ──────────────────────────────────────────────
#  Phase 1: Process match lists
# ──────────────────────────────────────────────

TOURNAMENT_TIERS = {
    "TI10_2021": "ti", "TI11_2022": "ti", "TI12_2023": "ti",
    "TI13_2024": "ti", "TI14_2025": "ti",
    "EWC_2026": "major", "DreamLeague_S29": "major", "DreamLeague_S28": "major",
    "PGL_Wallachia_S8": "major", "BLAST_SLAM_VII": "major", "BLAST_SLAM_VI": "major",
    "ESL_Birmingham_2026": "major",
}

TOURNAMENT_DATES = {
    "TI10_2021": "2021-10", "TI11_2022": "2022-10", "TI12_2023": "2023-10",
    "TI13_2024": "2024-09", "TI14_2025": "2025-08",
    "EWC_2026": "2026-07", "DreamLeague_S29": "2026-06", "DreamLeague_S28": "2026-05",
    "PGL_Wallachia_S8": "2026-05", "BLAST_SLAM_VII": "2026-04", "BLAST_SLAM_VI": "2026-03",
    "ESL_Birmingham_2026": "2026-04",
}


def load_all_matchlists():
    """Load all match list files and add tournament metadata."""
    all_matches = []

    for f in sorted(RAW_DIR.glob("*_matchlist.json")):
        key = f.stem.replace("_matchlist", "")
        with open(f) as fh:
            matches = json.load(fh)

        tier = TOURNAMENT_TIERS.get(key, "other")
        date_str = TOURNAMENT_DATES.get(key, "unknown")

        for m in matches:
            m["tournament"] = key
            m["tier"] = tier
            m["tournament_date"] = date_str
            m["year"] = int(date_str[:4]) if date_str != "unknown" else 0

            # Normalize team names
            # Resolve team: prefer match data name, fallback to ID map
            r_name = m.get("radiant_team_name") or TEAM_ID_TO_NAME.get(m.get("radiant_team_id"), "")
            d_name = m.get("dire_team_name") or TEAM_ID_TO_NAME.get(m.get("dire_team_id"), "")
            m["radiant_canonical"] = normalize_match_team_name(r_name) if r_name else None
            m["dire_canonical"] = normalize_match_team_name(d_name) if d_name else None

            all_matches.append(m)

    print(f"Loaded {len(all_matches)} matches from {len(list(RAW_DIR.glob('*_matchlist.json')))} tournaments")
    return all_matches


def build_match_dataset(matches):
    """Convert match lists to a flat DataFrame with team-level features."""
    rows = []
    for m in matches:
        r_canon = m.get("radiant_canonical")
        d_canon = m.get("dire_canonical")

        # Skip matches where we can't identify both TI2026 teams
        if not r_canon or not d_canon:
            continue

        ts = m.get("start_time", 0)
        dt = datetime.fromtimestamp(ts) if ts else None

        rows.append({
            "match_id": m["match_id"],
            "timestamp": ts,
            "date": dt.strftime("%Y-%m-%d") if dt else None,
            "tournament": m["tournament"],
            "tier": m["tier"],
            "year": m["year"],
            "radiant_team": r_canon,
            "dire_team": d_canon,
            "radiant_win": m["radiant_win"],
            "duration": m.get("duration", 0),
            "radiant_score": m.get("radiant_score", 0),
            "dire_score": m.get("dire_score", 0),
            "series_id": m.get("series_id"),
            "series_type": m.get("series_type", 1),
            "radiant_team_id": m.get("radiant_team_id"),
            "dire_team_id": m.get("dire_team_id"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
#  Elo rating system
# ──────────────────────────────────────────────

class EloSystem:
    def __init__(self, k_factor=32, initial=1500, decay_per_day=0.5):
        self.k = k_factor
        self.initial = initial
        self.decay_per_day = decay_per_day
        self.ratings = {}
        self.last_update = {}

    def get_rating(self, team):
        return self.ratings.get(team, self.initial)

    def expected(self, ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def apply_decay(self, team, current_ts):
        if team in self.last_update:
            days = (current_ts - self.last_update[team]) / 86400
            decay = self.decay_per_day * days
            if self.ratings[team] > self.initial:
                self.ratings[team] = max(self.initial, self.ratings[team] - decay)
            elif self.ratings[team] < self.initial:
                self.ratings[team] = min(self.initial, self.ratings[team] + decay)

    def update(self, team_a, team_b, a_won, timestamp, tier_weight=1.0):
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)

        # Apply time decay
        self.apply_decay(team_a, timestamp)
        self.apply_decay(team_b, timestamp)
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)

        ea = self.expected(ra, rb)
        sa = 1.0 if a_won else 0.0

        # K-factor varies by tier (TI matters more)
        k = self.k * tier_weight

        self.ratings[team_a] = ra + k * (sa - ea)
        self.ratings[team_b] = rb + k * ((1 - sa) - (1 - ea))
        self.last_update[team_a] = timestamp
        self.last_update[team_b] = timestamp


# ──────────────────────────────────────────────
#  Feature engineering
# ──────────────────────────────────────────────

def compute_team_features(df):
    """Compute rolling team features using historical data."""
    elo = EloSystem(k_factor=32, initial=1500, decay_per_day=0.5)

    # Per-team stats
    team_stats = defaultdict(lambda: {
        "wins": 0, "losses": 0, "map_wins": 0, "map_losses": 0,
        "recent_results": [],  # last 20 map results (1=win, 0=loss)
        "form": 0.5,
    })

    features_list = []

    for idx, row in df.iterrows():
        radiant = row["radiant_team"]
        dire = row["dire_team"]

        # Get pre-match ratings
        r_elo_before = elo.get_rating(radiant)
        d_elo_before = elo.get_rating(dire)
        r_stats = team_stats[radiant]
        d_stats = team_stats[dire]

        # Compute features
        r_wr = r_stats["wins"] / max(1, r_stats["wins"] + r_stats["losses"])
        d_wr = d_stats["wins"] / max(1, d_stats["wins"] + d_stats["losses"])

        r_map_wr = r_stats["map_wins"] / max(1, r_stats["map_wins"] + r_stats["map_losses"])
        d_map_wr = d_stats["map_wins"] / max(1, d_stats["map_wins"] + d_stats["map_losses"])

        # Form (rolling 20 maps)
        r_form = np.mean(r_stats["recent_results"][-20:]) if r_stats["recent_results"] else 0.5
        d_form = np.mean(d_stats["recent_results"][-20:]) if d_stats["recent_results"] else 0.5

        # Tier weight
        tier_w = {"ti": 2.0, "major": 1.5, "other": 1.0}.get(row["tier"], 1.0)

        features_list.append({
            "match_id": row["match_id"],
            "timestamp": row["timestamp"],
            "date": row["date"],
            "tournament": row["tournament"],
            "tier": row["tier"],
            "year": row["year"],
            "radiant_team": radiant,
            "dire_team": dire,
            "radiant_win": row["radiant_win"],
            "duration": row["duration"],
            "radiant_score": row["radiant_score"],
            "dire_score": row["dire_score"],
            "series_id": row["series_id"],
            "series_type": row["series_type"],
            # Features
            "r_elo": r_elo_before,
            "d_elo": d_elo_before,
            "diff_elo": r_elo_before - d_elo_before,
            "r_wr": r_wr,
            "d_wr": d_wr,
            "diff_wr": r_wr - d_wr,
            "r_map_wr": r_map_wr,
            "d_map_wr": d_map_wr,
            "diff_map_wr": r_map_wr - d_map_wr,
            "r_form": r_form,
            "d_form": d_form,
            "diff_form": r_form - d_form,
            "r_games_played": r_stats["wins"] + r_stats["losses"],
            "d_games_played": d_stats["wins"] + d_stats["losses"],
            "tier_weight": tier_w,
        })

        # Update stats AFTER recording features
        won = row["radiant_win"]
        team_stats[radiant]["wins"] += int(won)
        team_stats[radiant]["losses"] += int(not won)
        team_stats[dire]["wins"] += int(not won)
        team_stats[dire]["losses"] += int(won)

        r_stats["recent_results"].append(1 if won else 0)
        d_stats["recent_results"].append(0 if won else 1)

        # Update Elo
        elo.update(radiant, dire, won, row["timestamp"], tier_weight=tier_w)

    return pd.DataFrame(features_list)


def add_h2h_features(df):
    """Add head-to-head features."""
    h2h = defaultdict(lambda: {"wins": 0, "losses": 0})

    r_list = df.to_dict("records")
    new_rows = []

    for row in r_list:
        r, d = row["radiant_team"], row["dire_team"]
        key_rd = f"{r}_vs_{d}"
        key_dr = f"{d}_vs_{r}"

        r_h2h_wins = h2h[key_rd]["wins"]
        r_h2h_losses = h2h[key_rd]["losses"]
        d_h2h_wins = h2h[key_dr]["wins"]
        d_h2h_losses = h2h[key_dr]["losses"]

        row["r_h2h_wr"] = r_h2h_wins / max(1, r_h2h_wins + r_h2h_losses)
        row["d_h2h_wr"] = d_h2h_wins / max(1, d_h2h_wins + d_h2h_losses)
        row["diff_h2h"] = row["r_h2h_wr"] - row["d_h2h_wr"]

        new_rows.append(row)

        # Update
        if row["radiant_win"]:
            h2h[key_rd]["wins"] += 1
            h2h[key_dr]["losses"] += 1
        else:
            h2h[key_rd]["losses"] += 1
            h2h[key_dr]["wins"] += 1

    return pd.DataFrame(new_rows)


def add_tournament_context(df):
    """Add tournament-level context features."""
    # Recent tournament placements
    TI2026_RESULTS = {
        "EWC_2026": {"Team Vision": 1, "BetBoom": 2, "Yandex": 3, "Vici": 4},
        "DreamLeague_S29": {"Team Vision": 1, "Aurora": 2, "Spirit": 3, "Falcons": 4},
        "PGL_Wallachia_S8": {"BetBoom": 1, "Aurora": 2, "Falcons": 3, "Liquid": 4},
        "BLAST_SLAM_VII": {"Yandex": 1, "LGD": 2, "BetBoom": 3},
    }

    # For each row, compute team's recent best placement
    def best_recent_placement(team, before_date):
        best = 99
        for tourn, results in TI2026_RESULTS.items():
            if team in results:
                best = min(best, results[team])
        return best if best < 99 else 0

    df["r_recent_placement"] = df.apply(
        lambda row: best_recent_placement(row["radiant_team"], row["date"]), axis=1)
    df["d_recent_placement"] = df.apply(
        lambda row: best_recent_placement(row["dire_team"], row["date"]), axis=1)
    df["diff_placement"] = df["d_recent_placement"] - df["r_recent_placement"]

    return df


# ──────────────────────────────────────────────
#  Phase 2: Download match details
# ──────────────────────────────────────────────

def download_match_details(match_ids, output_file, resume=True):
    """Download match details with rate limiting."""
    existing = {}
    if resume and output_file.exists():
        with open(output_file) as f:
            for item in json.load(f):
                if item:
                    existing[item["match_id"]] = item
        print(f"  Resuming: {len(existing)} already downloaded")

    needed = [mid for mid in match_ids if mid not in existing]
    print(f"  Need {len(needed)} new details")

    batch = []
    for i, mid in enumerate(needed):
        try:
            resp = requests.get(f"https://api.opendota.com/api/matches/{mid}", timeout=30)
            if resp.status_code == 200:
                detail = resp.json()
                existing[mid] = detail
                batch.append(detail)
            elif resp.status_code == 429:
                print(f"    Rate limited at {i+1}/{len(needed)}, waiting 60s...")
                time.sleep(60)
                continue
            else:
                print(f"    Error {resp.status_code} for {mid}")
        except Exception as e:
            print(f"    Exception for {mid}: {e}")

        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(needed)}")
            # Save intermediate
            with open(output_file, "w") as f:
                json.dump(list(existing.values()), f)

        time.sleep(1.1)  # Stay under 60/min

    with open(output_file, "w") as f:
        json.dump(list(existing.values()), f)

    print(f"  Total: {len(existing)} details saved")
    return existing


def extract_player_features(details):
    """Extract player-level features from match details."""
    player_matches = []

    for match in details:
        match_id = match.get("match_id")
        players = match.get("players", [])

        for i, p in enumerate(players):
            is_radiant = i < 5
            healing = p.get("healing", 0)
            if isinstance(healing, dict):
                healing = sum(healing.values())

            player_matches.append({
                "match_id": match_id,
                "account_id": p.get("account_id"),
                "hero_id": p.get("hero_id"),
                "is_radiant": is_radiant,
                "kills": p.get("kills", 0),
                "deaths": p.get("deaths", 0),
                "assists": p.get("assists", 0),
                "gpm": p.get("gold_per_min", 0),
                "xpm": p.get("xp_per_min", 0),
                "hero_damage": p.get("hero_damage", 0),
                "tower_damage": p.get("tower_damage", 0),
                "healing": healing,
                "level": p.get("level", 0),
            })

    return pd.DataFrame(player_matches)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TI 2026 Dataset Builder")
    print("=" * 60)

    # Phase 0: Discover team IDs
    print("\n[0] Discovering team IDs...")
    discover_team_ids()

    # Phase 1: Process match lists
    print("\n[1] Processing match lists...")
    matches = load_all_matchlists()
    df_match = build_match_dataset(matches)
    print(f"  {len(df_match)} matches with identifiable TI2026 teams")
    print(f"  Teams found: {sorted(df_match['radiant_team'].unique())}")

    print("\n[2] Computing Elo & team features...")
    df_feat = compute_team_features(df_match)

    print("\n[3] Adding H2H features...")
    df_feat = add_h2h_features(df_feat)

    print("\n[4] Adding tournament context...")
    df_feat = add_tournament_context(df_feat)

    # Save
    out_file = FEATURES_DIR / "match_features_real.csv"
    df_feat.to_csv(out_file, index=False)
    print(f"\n  Saved {len(df_feat)} rows to {out_file}")
    print(f"  Columns: {list(df_feat.columns)}")

    # Stats
    print(f"\n  Dataset stats:")
    print(f"    Date range: {df_feat['date'].min()} to {df_feat['date'].max()}")
    print(f"    Tournaments: {df_feat['tournament'].nunique()}")
    print(f"    Teams: {df_feat['radiant_team'].nunique()}")
    print(f"    Radiant win rate: {df_feat['radiant_win'].mean():.3f}")
    print(f"    Elo range: {df_feat['r_elo'].min():.0f} to {df_feat['r_elo'].max():.0f}")

    # Phase 2: Download match details for 2026 tournaments (optional)
    if "--with-details" in sys.argv:
        print("\n[5] Downloading match details for 2026 tournaments...")
        tourn_2026 = ["EWC_2026", "DreamLeague_S29", "DreamLeague_S28",
                       "PGL_Wallachia_S8", "BLAST_SLAM_VII", "BLAST_SLAM_VI",
                       "ESL_Birmingham_2026"]
        match_ids_2026 = list(set(df_feat.loc[df_feat["tournament"].isin(tourn_2026), "match_id"]))
        print(f"  {len(match_ids_2026)} matches to download")

        details_file = RAW_DIR / "details_2026.json"
        details = download_match_details(list(match_ids_2026), details_file)

        print("\n[6] Extracting player features...")
        df_players = extract_player_features(details)
        players_file = FEATURES_DIR / "player_matches.csv"
        df_players.to_csv(players_file, index=False)
        print(f"  Saved {len(df_players)} player-match records")

    print("\nDone!")


if __name__ == "__main__":
    main()
