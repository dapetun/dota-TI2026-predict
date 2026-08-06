"""Registry of tournaments used for training and validation.

Single source of truth for league_id, tier, and sample-weight multipliers.
Expanded in v0.3: TI + majors + regional quals + mid-tier online (2023–2026).

Tier weight ladder (before time decay):
  ti=2.0 · major=1.5 · qual/dpc=0.75 · online=0.5 · other=1.0

Leave-One-TI-Out still tests only TI_KEYS; train = all matches before TI_k start.
"""

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
    tier: str  # ti | major | qual | dpc | online | other
    is_lan: bool = True
    # Relative sample weight multiplier (before time decay).
    tier_weight: float = 1.0


def _t(
    key: str,
    league_id: int,
    name: str,
    year: int,
    tier: str,
    is_lan: bool,
    weight: float | None = None,
) -> TournamentMeta:
    """Build TournamentMeta with default weight from tier."""
    w = weight if weight is not None else TIER_WEIGHTS.get(tier, 1.0)
    return TournamentMeta(key, league_id, name, year, tier, is_lan, w)


TIER_WEIGHTS: Dict[str, float] = {
    "ti": 2.0,
    "major": 1.5,
    "qual": 0.75,
    "dpc": 0.75,
    "online": 0.5,
    "other": 1.0,
}


# Curated high-value events. League IDs cross-checked via datdota / OpenDota
# matchlists (see data/league_candidates.json + scripts/discover_leagues.py).
TOURNAMENTS: Dict[str, TournamentMeta] = {}

_ENTRIES: list[TournamentMeta] = [
    # ── The International (LAN, highest weight) ──────────────────────────
    _t("TI10_2021", 13256, "The International 2021", 2021, "ti", True),
    _t("TI11_2022", 14268, "The International 2022", 2022, "ti", True),
    _t("TI12_2023", 15728, "The International 2023", 2023, "ti", True),
    _t("TI13_2024", 16935, "The International 2024", 2024, "ti", True),
    _t("TI14_2025", 18324, "The International 2025", 2025, "ti", True),
    # ── Majors / Tier-1 LAN 2023 (Elo continuity) ────────────────────────
    _t("Lima_Major_2023", 15089, "Lima Major 2023", 2023, "major", True),
    _t("Berlin_Major_2023", 15251, "ESL One Berlin Major 2023", 2023, "major", True),
    _t("Bali_Major_2023", 15438, "The Bali Major 2023", 2023, "major", True),
    # ── Majors / Tier-1 2024 ─────────────────────────────────────────────
    _t("DreamLeague_S23", 16632, "DreamLeague S23", 2024, "major", True),
    _t("PGL_Wallachia_S1", 16669, "PGL Wallachia S1", 2024, "major", True),
    _t("ESL_Birmingham_2024", 16518, "ESL One Birmingham 2024", 2024, "major", True),
    _t("Riyadh_Masters_2024", 16881, "Riyadh Masters 2024 (EWC)", 2024, "major", True),
    _t("Elite_League_S2", 16905, "Elite League S2", 2024, "major", False),
    _t("PGL_Wallachia_S2", 17119, "PGL Wallachia S2", 2024, "major", True),
    _t("BetBoom_Dacha_Belgrade_2024", 17126, "BetBoom Dacha Belgrade 2024", 2024, "major", True),
    _t("DreamLeague_S24", 17272, "DreamLeague S24", 2024, "major", True),
    _t("BLAST_SLAM_I", 17414, "BLAST SLAM I", 2024, "major", True),
    _t("ESL_Bangkok_2024", 17509, "ESL One Bangkok 2024", 2024, "major", True),
    # ── Majors / Tier-1 2025 ─────────────────────────────────────────────
    _t("DreamLeague_S25", 17765, "DreamLeague S25", 2025, "major", True),
    _t("PGL_Wallachia_S3", 17891, "PGL Wallachia S3", 2025, "major", True),
    _t("ESL_Raleigh_2025", 17795, "ESL One Raleigh 2025", 2025, "major", True),
    _t("PGL_Wallachia_S4", 18058, "PGL Wallachia S4", 2025, "major", True),
    _t("DreamLeague_S26", 18111, "DreamLeague S26", 2025, "major", True),
    _t("PGL_Wallachia_S5", 18358, "PGL Wallachia S5", 2025, "major", True),
    _t("EWC_2025", 18375, "Esports World Cup 2025", 2025, "major", True),
    _t("Clavision_2025", 18359, "Clavision Masters 2025", 2025, "online", False),
    _t("PGL_Wallachia_S6", 18920, "PGL Wallachia S6", 2025, "major", True),
    _t("DreamLeague_S27", 18988, "DreamLeague S27", 2025, "major", True),
    # ── Majors / Tier-1 2026 (pre-TI) ────────────────────────────────────
    _t("BLAST_SLAM_VI", 19099, "BLAST SLAM VI", 2026, "major", True),
    _t("BLAST_SLAM_VII", 19101, "BLAST SLAM VII", 2026, "major", True),
    _t("DreamLeague_S28", 19269, "DreamLeague S28", 2026, "major", True),
    _t("PGL_Wallachia_S7_2026", 19435, "PGL Wallachia S7 2026", 2026, "major", True),
    _t("PGL_Wallachia_S8", 19543, "PGL Wallachia S8", 2026, "major", True),
    _t("ESL_Birmingham_2026", 19422, "ESL One Birmingham 2026", 2026, "major", True),
    _t("DreamLeague_S29", 19696, "DreamLeague S29", 2026, "major", True),
    _t("EWC_2026", 19785, "Esports World Cup 2026", 2026, "major", True),
    # ── Mid-tier / online (form signal, lower weight) ────────────────────
    _t("Elite_League_2024", 16483, "Elite League by FISSURE 2024", 2024, "online", False),
    _t("FISSURE_Universe_Ep3", 16846, "FISSURE Universe Ep3", 2024, "online", False),
    _t("Clavision_S1_2024", 16901, "Clavision S1 Snow-Ruyi 2024", 2024, "online", False),
    _t("FISSURE_Playground_1", 17588, "FISSURE PLAYGROUND 1", 2025, "online", False),
    _t("FISSURE_Universe_Ep4", 17907, "FISSURE Universe Ep4", 2025, "online", False),
    _t("FISSURE_Universe_Ep5", 18107, "FISSURE Universe Ep5", 2025, "online", False),
    _t("FISSURE_Universe_Ep6", 18433, "FISSURE Universe Ep6", 2025, "online", False),
    _t("FISSURE_Universe_Ep7", 18633, "FISSURE Universe Ep7", 2025, "online", False),
    _t("FISSURE_Playground_2", 18863, "FISSURE PLAYGROUND 2", 2025, "online", False),
    _t("FISSURE_Universe_Ep8", 19239, "FISSURE Universe Ep8", 2026, "online", False),
    # ── TI regional quals (cold-start teams: HULIGANI / Resilience / GL) ─
    _t("TI13_WEU_Qual", 16842, "TI13 WEU Qualifier", 2024, "qual", False),
    _t("TI13_SEA_Qual", 16840, "TI13 SEA Qualifier", 2024, "qual", False),
    _t("TI13_SA_Qual", 16841, "TI13 SA Qualifier", 2024, "qual", False),
    _t("TI13_EEU_Qual", 16839, "TI13 EEU Qualifier", 2024, "qual", False),
    _t("TI13_CN_Qual", 16843, "TI13 CN Qualifier", 2024, "qual", False),
    _t("TI13_NA_Qual", 16844, "TI13 NA Qualifier", 2024, "qual", False),
    _t("TI14_WEU_Qual", 18309, "TI14 WEU Qualifier", 2025, "qual", False),
    _t("TI14_SEA_Qual", 18308, "TI14 SEA Qualifier", 2025, "qual", False),
    _t("TI14_NA_Qual", 18307, "TI14 NA Qualifier", 2025, "qual", False),
    _t("TI14_CN_Qual", 18306, "TI14 CN Qualifier", 2025, "qual", False),
    _t("TI14_SA_Qual", 18305, "TI14 SA Qualifier", 2025, "qual", False),
    _t("TI14_EEU_Qual", 18304, "TI14 EEU Qualifier", 2025, "qual", False),
    # Major quals with TI2026 teams (higher coverage for Resilience/GL)
    _t("Riyadh_2024_Quals", 16740, "Riyadh Masters 2024 Qualifiers", 2024, "qual", False),
    _t("EWC_2025_Quals", 18210, "EWC 2025 Qualifiers", 2025, "qual", False),
    _t("DreamLeague_S25_Quals", 17628, "DreamLeague S25 Qualifiers", 2025, "qual", False),
    _t("DreamLeague_S26_Quals", 17874, "DreamLeague S26 Qualifiers", 2025, "qual", False),
    _t("DreamLeague_S27_Quals", 18629, "DreamLeague S27 Qualifiers", 2025, "qual", False),
    _t("DreamLeague_S28_Quals", 19089, "DreamLeague S28 Qualifiers", 2026, "qual", False),
    _t("ESL_Birmingham_2026_Quals", 19090, "ESL Birmingham 2026 Qualifiers", 2026, "qual", False),
]

for meta in _ENTRIES:
    TOURNAMENTS[meta.key] = meta

# Chronological order of TI events for Leave-One-TI-Out.
TI_KEYS: list[str] = [
    "TI10_2021",
    "TI11_2022",
    "TI12_2023",
    "TI13_2024",
    "TI14_2025",
]

# Download / details priority: TI2026-relevant → quals → majors → rest.
DOWNLOAD_PRIORITY: list[str] = [
    # Recent majors with TI2026 teams
    "EWC_2026",
    "DreamLeague_S29",
    "PGL_Wallachia_S8",
    "ESL_Birmingham_2026",
    "PGL_Wallachia_S7_2026",
    "DreamLeague_S28",
    "BLAST_SLAM_VII",
    "BLAST_SLAM_VI",
    "FISSURE_Universe_Ep8",
    # 2026 / late 2025 quals
    "ESL_Birmingham_2026_Quals",
    "DreamLeague_S28_Quals",
    "DreamLeague_S27_Quals",
    "TI14_EEU_Qual",
    "TI14_WEU_Qual",
    "TI14_CN_Qual",
    "TI14_SEA_Qual",
    "TI14_SA_Qual",
    "TI14_NA_Qual",
    "EWC_2025_Quals",
    # 2025 majors
    "DreamLeague_S27",
    "PGL_Wallachia_S6",
    "Clavision_2025",
    "EWC_2025",
    "PGL_Wallachia_S5",
    "DreamLeague_S26",
    "PGL_Wallachia_S4",
    "ESL_Raleigh_2025",
    "PGL_Wallachia_S3",
    "DreamLeague_S25",
    "FISSURE_Playground_2",
    "FISSURE_Universe_Ep7",
    "FISSURE_Universe_Ep6",
    "FISSURE_Universe_Ep5",
    "FISSURE_Universe_Ep4",
    "FISSURE_Playground_1",
    "DreamLeague_S26_Quals",
    "DreamLeague_S25_Quals",
    # TI14 itself + 2024 stack
    "TI14_2025",
    "ESL_Bangkok_2024",
    "BLAST_SLAM_I",
    "DreamLeague_S24",
    "BetBoom_Dacha_Belgrade_2024",
    "PGL_Wallachia_S2",
    "Elite_League_S2",
    "Riyadh_Masters_2024",
    "ESL_Birmingham_2024",
    "PGL_Wallachia_S1",
    "DreamLeague_S23",
    "Clavision_S1_2024",
    "FISSURE_Universe_Ep3",
    "Elite_League_2024",
    "Riyadh_2024_Quals",
    "TI13_WEU_Qual",
    "TI13_EEU_Qual",
    "TI13_CN_Qual",
    "TI13_SEA_Qual",
    "TI13_SA_Qual",
    "TI13_NA_Qual",
    "TI13_2024",
    "Bali_Major_2023",
    "Berlin_Major_2023",
    "Lima_Major_2023",
    "TI12_2023",
    "TI11_2022",
    "TI10_2021",
]


def get_tournament(key: str) -> TournamentMeta | None:
    """Вернуть метаданные турнира по ключу."""
    return TOURNAMENTS.get(key)


def download_order() -> list[str]:
    """Порядок скачивания: DOWNLOAD_PRIORITY, затем остальные ключи."""
    seen: set[str] = set()
    order: list[str] = []
    for key in DOWNLOAD_PRIORITY:
        if key in TOURNAMENTS and key not in seen:
            order.append(key)
            seen.add(key)
    for key in TOURNAMENTS:
        if key not in seen:
            order.append(key)
            seen.add(key)
    return order


def league_count() -> int:
    """Число лиг в реестре."""
    return len(TOURNAMENTS)
