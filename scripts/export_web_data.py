"""Export prediction JSON for the static GitHub Pages frontend.

Prefers blend pairwise win matrix; falls back to power-ranking Bradley-Terry.
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
from src.ti2026.analyst_consensus import (
    analyst_agreement,
    analyst_names_for_slot,
    build_consensus_board,
    consensus_summary,
)
from src.ti2026.compendium_scoring import (
    compare_board_strategies,
    optimize_fantasy_board,
)
from src.ti2026.fusion import fuse_slot_probabilities, tune_fusion_weight_loo
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
COMPARE_PATH = BASE_DIR / "outputs" / "model_compare.json"
BLEND_PATH = BASE_DIR / "outputs" / "model_blend_v1.joblib"


def build_power_ranking_matrix(teams: list[str]) -> pd.DataFrame:
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


def build_export_win_matrix(teams: list[str]) -> tuple[pd.DataFrame, str, str]:
    """Return (matrix, model_key, model_label). Prefer blend pairwise."""
    if not BLEND_PATH.exists():
        return build_power_ranking_matrix(teams), "power_ranking_bradley_terry", "Power ranking"

    try:
        from src.data_collection.match_details import load_player_matches
        from src.data_collection.match_loader import load_raw_matchlists
        from src.ti2026.pairwise import build_model_win_matrix, load_blend_bundle

        matches = load_raw_matchlists(BASE_DIR / "data" / "raw")
        players = load_player_matches(BASE_DIR / "data" / "raw")
        bundle = load_blend_bundle(BLEND_PATH)
        models = bundle["model"]
        feature_cols = bundle["feature_cols"]
        blend_weights = (bundle.get("params") or {}).get("weights")
        matrix = build_model_win_matrix(
            matches,
            models,
            feature_cols,
            players=players,
            team_ids=teams,
            blend_weights=blend_weights,
        )
        # If too many teams unresolved, fall back.
        mapped = int((matrix.values != 0.5).sum() // 2)
        if mapped < 40:
            print(f"Pairwise sparse ({mapped} directed edges), falling back to power ranking")
            return (
                build_power_ranking_matrix(teams),
                "power_ranking_bradley_terry",
                "Power ranking",
            )
        return matrix, "blend_pairwise_v1", "Blend pairwise"
    except Exception as exc:  # noqa: BLE001 — export must not die
        print(f"Pairwise failed ({exc}); using power ranking")
        return build_power_ranking_matrix(teams), "power_ranking_bradley_terry", "Power ranking"


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

    if COMPARE_PATH.exists():
        try:
            with open(COMPARE_PATH, encoding="utf-8") as f:
                cmp = json.load(f)
            metrics["model_compare"] = cmp.get("models")
            blend = (cmp.get("models") or {}).get("blend") or {}
            if blend.get("leave_one_ti_avg_auc") is not None:
                metrics["blend_leave_one_ti_avg_auc"] = round(
                    float(blend["leave_one_ti_avg_auc"]), 3
                )
                metrics["blend_leave_one_ti_avg_logloss"] = round(
                    float(blend["leave_one_ti_avg_logloss"]), 4
                )
        except Exception:
            pass
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


def enrich_board_with_analysts(
    board: dict[str, list[dict]],
    slot_key: str,
    team_entry: dict,
) -> dict:
    """Add analyst agreement chips to a board entry."""
    tid = team_entry["id"]
    n = analyst_agreement(tid, slot_key)
    names = analyst_names_for_slot(tid, slot_key)
    team_entry = dict(team_entry)
    team_entry["analyst_agreement"] = n
    team_entry["analyst_names"] = names
    return team_entry


def enrich_full_board(board: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Attach N/11 analyst metadata to every team pill."""
    out: dict[str, list[dict]] = {}
    for slot, entries in board.items():
        out[slot] = [enrich_board_with_analysts(board, slot, e) for e in entries]
    return out


def build_matchups(win_matrix: pd.DataFrame, teams: list[str]) -> list[dict]:
    rows = []
    for i, a in enumerate(teams):
        for b in teams[i + 1 :]:
            p = float(win_matrix.loc[a, b])
            rows.append({"a": a, "b": b, "p_a": round(p, 3), "p_b": round(1 - p, 3)})
    return rows


def main(n_simulations: int = 20000) -> Path:
    teams = get_team_ids()
    win_matrix, model_key, model_label = build_export_win_matrix(teams)
    config = SwissConfig(**{
        k: v
        for k, v in SWISS_CONFIG.items()
        if k in SwissConfig.__dataclass_fields__
    })
    print(
        f"Simulating TI Swiss (first to {config.wins_to_qualify}, "
        f"{config.n_rounds} rounds, ER->{config.elimination_round_advance}) "
        f"x {n_simulations:,} [{model_label}]..."
    )
    results = simulate_swiss_stage(
        win_matrix, teams, config, n_simulations=n_simulations, rng_seed=42
    )

    predictions = [
        team_payload(row["team"], row, win_matrix, teams) for _, row in results.iterrows()
    ]
    predictions.sort(key=lambda x: (-x["qualify_pct"], x["power_rank"]))

    qualify_board = assign_fantasy_board(predictions)
    points_board = optimize_fantasy_board(predictions)
    consensus_board = build_consensus_board(predictions)
    fusion_weight, fusion_pts = tune_fusion_weight_loo(predictions)
    fused_predictions = fuse_slot_probabilities(predictions, model_weight=fusion_weight)
    fusion_board = optimize_fantasy_board(fused_predictions)

    board = enrich_full_board(points_board)
    boards_payload = {
        "points_optimal": enrich_full_board(points_board),
        "qualify_rank": enrich_full_board(qualify_board),
        "analyst_consensus": enrich_full_board(consensus_board),
        "fusion": enrich_full_board(fusion_board),
    }
    board_compare = compare_board_strategies(
        predictions,
        extra_boards={
            "analyst_consensus": consensus_board,
            "fusion": fusion_board,
        },
    )
    analyst_meta = consensus_summary(predictions)

    # Sanity: capacities
    for key, meta in FANTASY_BOARD_SLOTS.items():
        got = len(board[key])
        if got != meta["capacity"]:
            print(f"WARNING: board[{key}] has {got}, expected {meta['capacity']}")

    if model_key.startswith("blend"):
        disclaimer = (
            "Доска Swiss: Monte Carlo на pairwise blend (XGB+CatBoost) "
            "с team Elo/form + snapshot player/chemistry. "
            "Слоты компендиума подобраны под максимум ожидаемых очков Valve "
            "(не просто топ по шансу пройти)."
        )
    else:
        disclaimer = (
            "Доска Swiss построена на power ranking и Monte Carlo "
            "(Swiss до 4 побед/поражений + Elimination Round)."
        )

    payload = {
        "meta": {
            "title": "TI 2026 Swiss Predictions",
            "subtitle": "Open-source group-stage forecast",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model_key,
            "model_label": model_label,
            "disclaimer": disclaimer,
            "format": "16-team Swiss to 4, 5 rounds Bo3 + ER (5 of 10)",
            "board_format": "4-0×1, 4-1×2, advance×5, eliminate×5, 1-4×2, 0-4×1",
            "n_simulations": n_simulations,
            "version": "0.2.0-prod",
            "board_strategy": "points_optimal",
            "expected_compendium_points": board_compare["points_optimal"]["expected_points"],
            "expected_correct_slots": board_compare["points_optimal"]["expected_correct"],
            "board_compare": board_compare,
            "fusion_model_weight": fusion_weight,
            "fusion_expected_points": board_compare.get("fusion", {}).get("expected_points"),
        },
        "analyst": analyst_meta,
        "boards": boards_payload,
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
    print(f"\nCompendium E[points]: {board_compare['points_optimal']['expected_points']:.0f} "
          f"(qualify-rank: {board_compare['qualify_rank']['expected_points']:.0f})")
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
