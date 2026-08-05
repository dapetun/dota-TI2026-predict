"""TI 2026 teams, rosters, and tournament-specific configuration."""

import pandas as pd
from typing import Dict, List


# TI 2026 Group Stage: 16 teams
TI2026_TEAMS = {
    "Aurora": {
        "full_name": "Aurora Gaming",
        "region": "EU",
        "source": "Direct Invite",
        "roster": ["Nightfall", "Mikoto", "Ws", "Mira", "kaori"],
        "aliases": ["Aurora Gaming"],
    },
    "BetBoom": {
        "full_name": "BoomBoys",
        "region": "EU",
        "source": "Direct Invite",
        "roster": ["Kiritych~", "gpk", "MieRo", "Save-", "Kataomi`"],
        "aliases": ["BetBoom Team", "BoomBoys"],
    },
    "Falcons": {
        "full_name": "Team Falcons",
        "region": "EU",
        "source": "Direct Invite",
        "roster": ["skiter", "Malr1ne", "ATF", "Cr1t-", "Sneyking"],
        "aliases": ["Team Falcons"],
    },
    "Liquid": {
        "full_name": "Team Liquid",
        "region": "EU",
        "source": "Direct Invite",
        "roster": ["miCKe", "Nisha", "Ace", "Boxi", "tOfu"],
        "aliases": ["Team Liquid"],
    },
    "1w": {
        "full_name": "1w",
        "region": "EU",
        "source": "Direct Invite",
        "roster": ["Pure", "bzm", "33", "Ari", "Whitemon"],
        "aliases": ["Tundra Esports", "Iron Wing"],
    },
    "Xtreme": {
        "full_name": "Xtreme Gaming",
        "region": "CN",
        "source": "Direct Invite",
        "roster": ["Ame", "NothingToSay", "Xxs", "fy", "xNova"],
        "aliases": ["Xtreme Gaming"],
    },
    "Yandex": {
        "full_name": "Team Yandex",
        "region": "EU",
        "source": "Direct Invite",
        "roster": ["watson", "CHIRA_JUNIOR", "DM", "Saksa", "Malady"],
        "aliases": ["Team Yandex"],
    },
    "Spirit": {
        "full_name": "Team Spirit",
        "region": "EU",
        "source": "EU Qualifier",
        "roster": ["Yatoro", "Larl", "Collapse", "not me", "rue"],
        "aliases": ["Team Spirit"],
    },
    "Vision": {
        "full_name": "Team Vision",
        "region": "EU",
        "source": "EU Qualifier",
        "roster": ["Satanic", "No[o]ne", "Noticed", "9Class", "Dukalis"],
        "aliases": ["PARIVISION", "Team Vision"],
    },
    "HULIGANI": {
        "full_name": "HULIGANI",
        "region": "EU",
        "source": "EU Qualifier",
        "roster": ["ssnovv1", "Mirage", "Vazya", "sayuw", "RESPECT"],
        "aliases": ["L1GA TEAM"],
    },
    "Nigma": {
        "full_name": "Nigma Galaxy",
        "region": "EU",
        "source": "EU Qualifier",
        "roster": ["SumaiL", "lorenof", "Davai", "OmaR", "GH"],
        "aliases": ["Nigma Galaxy"],
    },
    "Resilience": {
        "full_name": "Team Resilience",
        "region": "CN",
        "source": "CN Qualifier",
        "roster": ["Erika", "EchozZ", "niu", "planet", "zzq"],
        "aliases": ["Team Resilience"],
    },
    "Vici": {
        "full_name": "Vici Gaming",
        "region": "CN",
        "source": "CN Qualifier",
        "roster": ["shiro", "Xm", "Bach", "XinQ", "y`"],
        "aliases": ["Vici Gaming"],
    },
    "OG": {
        "full_name": "OG",
        "region": "SEA",
        "source": "SEA Qualifier",
        "roster": ["Natsumi", "Yopaj", "Raven", "TIMS", "skem"],
        "aliases": ["OG"],
    },
    "GamerLegion": {
        "full_name": "GamerLegion",
        "region": "NA",
        "source": "NA Qualifier",
        "roster": ["Ghost", "RCY", "Fayde", "Bignum", "Speeed"],
        "aliases": ["GamerLegion"],
    },
    "LGD": {
        "full_name": "LGD Gaming",
        "region": "SA",
        "source": "SA Qualifier",
        "roster": ["Yuma", "TaiLung", "Wisper", "Thiolicor", "KJ"],
        "aliases": ["LGD Gaming"],
    },
}


# Build alias → canonical name mapping
ALIAS_TO_CANONICAL = {}
for canonical, info in TI2026_TEAMS.items():
    ALIAS_TO_CANONICAL[canonical.lower()] = canonical
    ALIAS_TO_CANONICAL[info["full_name"].lower()] = canonical
    for alias in info["aliases"]:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical


# Recent 2026 tournament results (for form/pre-tournament seeding)
TI2026_RECENT_RESULTS = {
    "EWC_2026": {
        "1st": "Vision",
        "2nd": "BetBoom",
        "3rd": "Yandex",
        "4th": "Vici",
    },
    "DreamLeague_S29": {
        "1st": "Vision",
        "2nd": "Aurora",
        "3rd": "Spirit",
        "4th": "Falcons",
    },
    "PGL_Wallachia_S8": {
        "1st": "BetBoom",
        "2nd": "Aurora",
        "3rd": "Falcons",
        "4th": "Liquid",
    },
    "BLAST_SLAM_VII": {
        "1st": "Yandex",
        "2nd": "LGD",
        "3rd": "BetBoom",
    },
}


# Power rankings (subjective baseline for missing data)
POWER_RANKINGS = {
    "Vision": 1,
    "BetBoom": 2,
    "Yandex": 3,
    "Aurora": 4,
    "Falcons": 5,
    "Liquid": 6,
    "1w": 7,
    "Xtreme": 8,
    "Spirit": 9,
    "Vici": 10,
    "OG": 11,
    "Nigma": 12,
    "GamerLegion": 13,
    "LGD": 14,
    "Resilience": 15,
    "HULIGANI": 16,
}


def normalize_team_name(name: str) -> str:
    """Map any alias to canonical team name."""
    return ALIAS_TO_CANONICAL.get(name.lower(), name)


def get_team_ids() -> List[str]:
    """Get list of all canonical team IDs."""
    return list(TI2026_TEAMS.keys())


def get_team_roster(team_id: str) -> List[str]:
    """Get roster for a team."""
    return TI2026_TEAMS.get(team_id, {}).get("roster", [])


def build_teams_df() -> pd.DataFrame:
    """Build DataFrame of all TI 2026 teams."""
    rows = []
    for team_id, info in TI2026_TEAMS.items():
        rows.append({
            "team_id": team_id,
            "full_name": info["full_name"],
            "region": info["region"],
            "source": info["source"],
            "roster": ", ".join(info["roster"]),
            "power_ranking": POWER_RANKINGS.get(team_id, 99),
        })
    return pd.DataFrame(rows)


# Swiss Stage format (TI: first to 4 wins / 4 losses, then ER)
SWISS_CONFIG = {
    "n_rounds": 5,
    "wins_to_qualify": 4,
    "losses_to_eliminate": 4,
    "elimination_round_advance": 5,  # From Elimination Round into playoffs
    "qualify_top_n": 3,              # Typical direct 4-0 + 4-1 count
    "elim_bottom_n": 3,              # Typical direct 0-4 + 1-4 count
}


# Main Event format
PLAYOFF_CONFIG = {
    "type": "double_elimination",
    "upper_bracket_rounds": 3,  # QF, SF, WF
    "lower_bracket_rounds": 5,  # R1, R2, R3, SF, WF
    "grand_final": "Bo5",
}
