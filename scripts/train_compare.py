"""CLI: compare XGBoost, CatBoost and blend on current OpenDota data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.train_compare import run_model_compare


def main() -> None:
    """Parse ablation flags and run model compare."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--half-life-days", type=float, default=None)
    p.add_argument("--min-games", type=int, default=0)
    p.add_argument("--prefer-lan", action="store_true", help="Use LAN Elo in train features")
    p.add_argument("--majors-only", action="store_true")
    p.add_argument("--lan-only-chemistry", action="store_true")
    p.add_argument("--no-stitch", action="store_true")
    args = p.parse_args()
    run_model_compare(
        half_life_days=args.half_life_days,
        min_games=args.min_games,
        prefer_lan=args.prefer_lan,
        majors_only=args.majors_only,
        lan_only_chemistry=args.lan_only_chemistry,
        stitch_teams=not args.no_stitch,
    )


if __name__ == "__main__":
    main()
