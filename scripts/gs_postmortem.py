"""Compare pre-GS / fusion boards vs Liquipedia (or manual) Swiss GT.

After Group Stage ends, put official slots into
``data/historical/ti_swiss_ground_truth.json`` under key ``TI15`` (or pass
``--ti-key``), then:

    python scripts/gs_postmortem.py --predictions docs/data/predictions.json

Writes ``outputs/gs_postmortem.md`` and prints a short summary for RESULTS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ti2026.compendium_scoring import VALVE_GROUP_POINTS, _assignment_from_board
from src.ti2026.expert_history import load_swiss_ground_truth
from src.ti2026.swiss_backtest import ground_truth_assignment, score_assignment_vs_gt


def _board_from_payload(boards: dict, key: str) -> dict[str, list[dict]] | None:
    block = (boards or {}).get(key)
    if not block:
        return None
    if isinstance(block, dict) and "board" in block:
        return block["board"]
    if isinstance(block, dict) and any(
        k in block for k in ("undefeated", "advance", "eliminate")
    ):
        return block
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "docs" / "data" / "predictions.json",
    )
    parser.add_argument("--ti-key", default="TI15", help="Ground-truth tournament key")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "gs_postmortem.md",
    )
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f"Missing predictions: {args.predictions}", file=sys.stderr)
        return 1

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    boards = payload.get("boards") or {}
    gt_data = load_swiss_ground_truth()
    tourn = (gt_data.get("tournaments") or {}).get(args.ti_key)
    if not tourn or not tourn.get("slots"):
        lines = [
            f"# GS post-mortem ({args.ti_key})",
            "",
            f"Ground truth for `{args.ti_key}` not filled yet.",
            "After GS: add slots to `data/historical/ti_swiss_ground_truth.json`, then re-run.",
            "",
            "Boards available in predictions.json:",
            "",
        ]
        for k in sorted(boards):
            lines.append(f"- `{k}`")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0

    gt_assignment = ground_truth_assignment(args.ti_key, gt_data)

    keys = [
        "points_optimal",
        "qualify_rank",
        "analyst_consensus",
        "fusion",
        "fusion_model_heavy",
        "fusion_balanced",
        "fusion_market_lean",
        "fusion_analyst_lean",
    ]
    rows: list[tuple[str, dict]] = []
    for key in keys:
        board = _board_from_payload(boards, key)
        if board is None:
            continue
        assignment = _assignment_from_board(board)
        rows.append((key, score_assignment_vs_gt(assignment, gt_assignment)))

    lines = [
        f"# GS post-mortem ({args.ti_key})",
        "",
        "| Board | Valve points | Correct slots | Hit rate |",
        "|---|---:|---:|---:|",
    ]
    for key, sc in rows:
        lines.append(
            f"| {key} | {int(sc['valve_points'])} | "
            f"{int(sc['correct_slots'])}/{int(sc['scored_slots'])} | "
            f"{sc['hit_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Paste summary into docs/RESULTS.md appendix.",
            f"- Valve points table keys: {sorted(VALVE_GROUP_POINTS)[:5]}…",
            "",
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
