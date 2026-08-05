"""Export prediction JSON for the static GitHub Pages frontend.

Uses power-ranking Bradley-Terry + TI Swiss (first to 4) + Elimination Round.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.tournament_sim import (
    FANTASY_BOARD_SLOTS,
    SwissConfig,
    assign_fantasy_board,
    simulate_swiss_stage,
)
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
        metrics: dict = {}
    else:
        with open(METRICS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        wf = raw.get("walk_forward", [])
        loo = raw.get("leave_one_ti", [])
        metrics = {
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

    try:
        from src.data_collection.match_details import (
            load_player_matches,
            summarize_player_coverage,
        )
        from src.data_collection.match_loader import load_raw_matchlists

        matches = load_raw_matchlists(BASE_DIR / "data" / "raw")
        players = load_player_matches(BASE_DIR / "data" / "raw")
        metrics["player_coverage"] = summarize_player_coverage(matches, players)
    except Exception:
        metrics["player_coverage"] = None
    return metrics


def team_payload(team_id: str, row: pd.Series, win_matrix: pd.DataFrame, teams: list[str]) -> dict:
    info = TI2026_TEAMS[team_id]
    avg_wr = float(np.mean([win_matrix.loc[team_id, t] for t in teams if t != team_id]))
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
        "prob_4_0": float(row["prob_4_0"]),
        "prob_4_1": float(row["prob_4_1"]),
        "prob_advance": float(row["prob_advance"]),
        "prob_eliminate": float(row["prob_eliminate"]),
        "prob_1_4": float(row["prob_1_4"]),
        "prob_0_4": float(row["prob_0_4"]),
        "records": {
            "4-0": float(row["prob_4_0"]),
            "4-1": float(row["prob_4_1"]),
            "advance": float(row["prob_advance"]),
            "eliminate": float(row["prob_eliminate"]),
            "1-4": float(row["prob_1_4"]),
            "0-4": float(row["prob_0_4"]),
            "swiss_3-2": float(row["swiss_3_2"]),
            "swiss_2-3": float(row["swiss_2_3"]),
        },
        "board": {
            "undefeated": float(row["prob_4_0"]),
            "one_loss": float(row["prob_4_1"]),
            "advance": float(row["prob_advance"]),
            "eliminate": float(row["prob_eliminate"]),
            "one_win": float(row["prob_1_4"]),
            "winless": float(row["prob_0_4"]),
        },
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
    config = SwissConfig(**{
        k: v
        for k, v in SWISS_CONFIG.items()
        if k in SwissConfig.__dataclass_fields__
    })
    print(
        f"Simulating TI Swiss (first to {config.wins_to_qualify}, "
        f"{config.n_rounds} rounds, ER→{config.elimination_round_advance}) "
        f"× {n_simulations:,}..."
    )
    results = simulate_swiss_stage(
        win_matrix, teams, config, n_simulations=n_simulations, rng_seed=42
    )

    predictions = [
        team_payload(row["team"], row, win_matrix, teams) for _, row in results.iterrows()
    ]
    predictions.sort(key=lambda x: (-x["qualify_pct"], x["power_rank"]))
    board = assign_fantasy_board(predictions)

    # Sanity: capacities
    for key, meta in FANTASY_BOARD_SLOTS.items():
        got = len(board[key])
        if got != meta["capacity"]:
            print(f"WARNING: board[{key}] has {got}, expected {meta['capacity']}")

    payload = {
        "meta": {
            "title": "TI 2026 Swiss Predictions",
            "subtitle": "Open-source group-stage forecast",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": "power_ranking_bradley_terry",
            "model_label": "Power ranking",
            "disclaimer": (
                "Доска Swiss построена на power ranking и Monte Carlo "
                "(Swiss до 4 побед/поражений + Elimination Round). "
                "XGBoost (team+player) обучен отдельно; парные прогнозы из модели пока не подключены."
            ),
            "format": "16-team Swiss to 4, 5 rounds Bo3 + ER (5 of 10)",
            "board_format": "4-0×1, 4-1×2, advance×5, eliminate×5, 1-4×2, 0-4×1",
            "n_simulations": n_simulations,
            "version": "0.2.0-prod",
        },
        "model_metrics": load_model_metrics(),
        "recent_results": TI2026_RECENT_RESULTS,
        "board_meta": FANTASY_BOARD_SLOTS,
        "board": board,
        "teams": predictions,
        "matchups": build_matchups(win_matrix, teams),
    }

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    out = WEB_DATA / "predictions.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

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
    print("Fantasy board:")
    for key, meta in FANTASY_BOARD_SLOTS.items():
        names = ", ".join(e["name"] for e in board[key])
        print(f"  {meta['label']:12s} ({meta['capacity']}): {names}")
    print("Top qualify:")
    for p in predictions[:5]:
        print(f"  {p['power_rank']:2d}. {p['name']:20s} qualify={p['qualify_pct']:5.1f}%")
    return out


if __name__ == "__main__":
    main()
