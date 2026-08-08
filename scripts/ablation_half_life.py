"""Ablation: retrain blend LOO metrics for half-life / majors-only from saved features.

Does not rebuild the full feature matrix — uses ``data/features/match_features_xgb.csv``
(or path from --features) plus optional majors filter via tournament/tier column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.features.chemistry_features import CHEMISTRY_FEATURE_COLUMNS
from src.features.match_features import FEATURE_COLUMNS
from src.features.player_features import PLAYER_FEATURE_COLUMNS
from src.models.ensemble import train_blend_pipeline
from src.pipeline.train_compare import _avg_auc, _avg_ll


def main() -> None:
    """Run half-life ablation grid on saved features."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features",
        type=Path,
        default=BASE / "data" / "features" / "match_features_xgb.csv",
    )
    p.add_argument(
        "--half-lives",
        type=float,
        nargs="+",
        default=[120, 180, 210, 240],
    )
    p.add_argument("--out", type=Path, default=BASE / "outputs" / "ablation_half_life.json")
    args = p.parse_args()

    df = pd.read_csv(args.features)
    cols = [
        c
        for c in FEATURE_COLUMNS + PLAYER_FEATURE_COLUMNS + CHEMISTRY_FEATURE_COLUMNS
        if c in df.columns
    ]
    rows = []
    for hl in args.half_lives:
        print(f"=== half_life={hl} ===")
        result = train_blend_pipeline(df, cols, half_life_days=hl, calibrate=True)
        rows.append(
            {
                "half_life_days": hl,
                "loo_auc": _avg_auc(result.leave_one_ti),
                "loo_ll": _avg_ll(result.leave_one_ti),
                "wf_auc": _avg_auc(result.walk_forward),
                "wf_ll": _avg_ll(result.walk_forward),
                "weights": result.params.get("weights"),
            }
        )
        print(rows[-1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    best = max(rows, key=lambda r: (r["loo_auc"] or 0, -(r["loo_ll"] or 9)))
    print("best", best)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
