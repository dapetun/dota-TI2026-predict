"""Export prediction JSON for the static GitHub Pages frontend.

Uses an honest power-ranking Bradley-Terry baseline + Swiss Monte Carlo.
XGBoost pairwise predictions will replace this once roster features land.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.tournament_sim import SwissConfig, simulate_swiss_stage
from src.ti2026.teams import (
    POWER_RANKINGS,
    SWISS_CONFIG,
    TI2026_RECENT_RESULTS,
    TI2026_TEAMS,
    get_team_ids,
)

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DATA = BASE_DIR / "docs" / "data"
METRICS_PATH = BASE_DIR / "outputs" / "xgb_v1_metrics.json"


def build_win_matrix(teams: list[str]) -> pd.DataFrame:
    """Bradley-Terry from power ranking (rank 1 = strongest)."""
    n = len(teams)
    strengths = {t: n - POWER_RANKINGS.get(t, n) + 1 for t in teams}
    matrix = np.full((n, n), 0.5)
    for i, a in enumerate(teams):
        for j, b in enumerate(teams):
            if i == j:
                continue
            matrix[i, j] = strengths[a] / (strengths[a] + strengths[b])
    return pd.DataFrame(matrix, index=teams, columns=teams)


def load_model_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    wf = raw.get("walk_forward", [])
    loo = raw.get("leave_one_ti", [])
    return {
        "walk_forward_avg_logloss": round(float(np.mean([x["log_loss"] for x in wf])), 4) if wf else None,
        "walk_forward_avg_auc": round(float(np.mean([x["auc"] for x in wf])), 3) if wf else None,
        "leave_one_ti_avg_logloss": round(float(np.mean([x["log_loss"] for x in loo])), 4) if loo else None,
        "leave_one_ti_avg_auc": round(float(np.mean([x["auc"] for x in loo])), 3) if loo else None,
        "leave_one_ti": [
            {
                "ti": x["fold"],
                "log_loss": round(x["log_loss"], 4),
                "auc": round(x["auc"], 3),
                "n_test": x["n_test"],
            }
            for x in loo
        ],
        "top_features": list(raw.get("feature_importance", {}).keys())[:8],
    }


def team_payload(team_id: str, row: pd.Series, win_matrix: pd.DataFrame, teams: list[str]) -> dict:
    info = TI2026_TEAMS[team_id]
    avg_wr = float(
        np.mean([win_matrix.loc[team_id, t] for t in teams if t != team_id])
    )
    return {
        "id": team_id,
        "name": info["full_name"],
        "short": team_id,
        "region": info["region"],
        "source": info["source"],
        "roster": info["roster"],
        "power_rank": POWER_RANKINGS.get(team_id, 99),
        "avg_win_prob": round(avg_wr, 3),
        "expected_wins": float(row["expected_wins"]),
        "most_likely_record": row["most_likely_record"],
        "qualify_pct": float(row["direct_qualification_pct"]),
        "elim_round_pct": float(row["elimination_round_pct"]),
        "eliminated_pct": float(row["eliminated_pct"]),
        "records": {
            "3-0": float(row["prob_3_0"]),
            "3-1": float(row["prob_3_1"]),
            "3-2_or_2-3": float(row["prob_3_2_or_2_3"]),
            "2-3": float(row["prob_2_3"]),
            "1-3": float(row["prob_1_3"]),
            "0-3": float(row["prob_0_3"]),
        },
    }


def refine_records(row: pd.Series) -> dict:
    """Map simulation columns to board buckets without inventing fake splits."""
    return {
        "undefeated": float(row["prob_3_0"]),          # 3-0
        "one_loss": float(row["prob_3_1"]),            # 3-1
        "borderline": float(row["prob_3_2_or_2_3"]),   # 3-2 or 2-3
        "two_losses_out": float(row["prob_2_3"]),      # 2-3 (also in borderline)
        "one_win": float(row["prob_1_3"]),             # 1-3
        "winless": float(row["prob_0_3"]),             # 0-3
        "qualify": float(row["direct_qualification_pct"]),
        "eliminated": float(row["eliminated_pct"]),
    }


def build_matchups(win_matrix: pd.DataFrame, teams: list[str]) -> list[dict]:
    rows = []
    for i, a in enumerate(teams):
        for b in teams[i + 1 :]:
            p = float(win_matrix.loc[a, b])
            rows.append({"a": a, "b": b, "p_a": round(p, 3), "p_b": round(1 - p, 3)})
    return rows


def main(n_simulations: int = 20000) -> Path:
    teams = get_team_ids()
    win_matrix = build_win_matrix(teams)
    config = SwissConfig(**SWISS_CONFIG)
    print(f"Simulating Swiss ({n_simulations:,} runs)...")
    results = simulate_swiss_stage(
        win_matrix, teams, config, n_simulations=n_simulations, rng_seed=42
    )

    predictions = []
    for _, row in results.iterrows():
        team_id = row["team"]
        payload = team_payload(team_id, row, win_matrix, teams)
        payload["board"] = refine_records(row)
        predictions.append(payload)

    predictions.sort(key=lambda x: (-x["qualify_pct"], x["power_rank"]))

    # Place each team on the Swiss board by most likely terminal bucket
    board_slots = {
        "undefeated": [],
        "one_loss": [],
        "advancing": [],
        "borderline": [],
        "one_win": [],
        "winless": [],
    }
    for p in predictions:
        rec = p["most_likely_record"]
        entry = {
            "id": p["id"],
            "name": p["name"],
            "short": p["short"],
            "prob": p["qualify_pct"] if rec.startswith("3") else p["eliminated_pct"],
            "record": rec,
            "qualify_pct": p["qualify_pct"],
        }
        if rec == "3-0":
            board_slots["undefeated"].append(entry)
        elif rec == "3-1":
            board_slots["one_loss"].append(entry)
        elif rec == "3-2":
            board_slots["advancing"].append(entry)
        elif rec == "2-3":
            board_slots["borderline"].append(entry)
        elif rec == "1-3":
            board_slots["one_win"].append(entry)
        elif rec == "0-3":
            board_slots["winless"].append(entry)
        else:
            board_slots["borderline"].append(entry)

    payload = {
        "meta": {
            "title": "TI 2026 Swiss Predictions",
            "subtitle": "Open-source group-stage forecast",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": "baseline_power_ranking_bradley_terry",
            "model_label": "Baseline · power ranking",
            "disclaimer": (
                "Текущий UI показывает честный baseline (сила из power ranking + "
                "Monte Carlo Swiss). XGBoost v0.1 уже обучен на матчах, но пока "
                "недостаточно точен для боевых парных прогнозов (AUC ~0.56)."
            ),
            "format": "16-team Swiss, 5 rounds, Bo3",
            "n_simulations": n_simulations,
            "version": "0.1.0",
        },
        "model_metrics": load_model_metrics(),
        "recent_results": TI2026_RECENT_RESULTS,
        "board": board_slots,
        "teams": predictions,
        "matchups": build_matchups(win_matrix, teams),
    }

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    out = WEB_DATA / "predictions.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Compact teams index for future pages
    teams_out = WEB_DATA / "teams.json"
    with open(teams_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                tid: {
                    "id": tid,
                    **{k: v for k, v in info.items() if k != "aliases"},
                    "power_rank": POWER_RANKINGS.get(tid),
                }
                for tid, info in TI2026_TEAMS.items()
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {out}")
    print("Top qualify:")
    for p in predictions[:5]:
        print(f"  {p['power_rank']:2d}. {p['short']:12s} qualify={p['qualify_pct']:5.1f}%")
    return out


if __name__ == "__main__":
    main()
