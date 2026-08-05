"""Registry of tournaments used for training and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TournamentMeta:
    """Metadata for a single OpenDota league."""

    key: str
    league_id: int
    name: str
    year: int
    tier: str  # ti | major | other
    is_lan: bool = True
    # Relative sample weight multiplier (before time decay).
    tier_weight: float = 1.0


# Curated high-value events for TI-style prediction.
TOURNAMENTS: Dict[str, TournamentMeta] = {
    "TI10_2021": TournamentMeta(
        "TI10_2021", 13256, "The International 2021", 2021, "ti", True, 2.0
    ),
    "TI11_2022": TournamentMeta(
        "TI11_2022", 14268, "The International 2022", 2022, "ti", True, 2.0
    ),
    "TI12_2023": TournamentMeta(
        "TI12_2023", 15728, "The International 2023", 2023, "ti", True, 2.0
    ),
    "TI13_2024": TournamentMeta(
        "TI13_2024", 16935, "The International 2024", 2024, "ti", True, 2.0
    ),
    "TI14_2025": TournamentMeta(
        "TI14_2025", 18324, "The International 2025", 2025, "ti", True, 2.0
    ),
    "EWC_2026": TournamentMeta(
        "EWC_2026", 19785, "Esports World Cup 2026", 2026, "major", True, 1.5
    ),
    "DreamLeague_S29": TournamentMeta(
        "DreamLeague_S29", 19696, "DreamLeague S29", 2026, "major", True, 1.5
    ),
    "DreamLeague_S28": TournamentMeta(
        "DreamLeague_S28", 19269, "DreamLeague S28", 2026, "major", True, 1.5
    ),
    "PGL_Wallachia_S8": TournamentMeta(
        "PGL_Wallachia_S8", 19543, "PGL Wallachia S8", 2026, "major", True, 1.5
    ),
    "BLAST_SLAM_VII": TournamentMeta(
        "BLAST_SLAM_VII", 19101, "BLAST SLAM VII", 2026, "major", True, 1.5
    ),
    "BLAST_SLAM_VI": TournamentMeta(
        "BLAST_SLAM_VI", 19099, "BLAST SLAM VI", 2026, "major", True, 1.5
    ),
    "ESL_Birmingham_2026": TournamentMeta(
        "ESL_Birmingham_2026", 19422, "ESL One Birmingham 2026", 2026, "major", True, 1.5
    ),
}

# Chronological order of TI events for Leave-One-TI-Out.
TI_KEYS: list[str] = [
    "TI10_2021",
    "TI11_2022",
    "TI12_2023",
    "TI13_2024",
    "TI14_2025",
]

TIER_WEIGHTS: Dict[str, float] = {
    "ti": 2.0,
    "major": 1.5,
    "other": 1.0,
    "online": 0.5,
}


def get_tournament(key: str) -> TournamentMeta | None:
    """Return tournament metadata by key."""
    return TOURNAMENTS.get(key)
