"""Export prediction JSON for the static GitHub Pages frontend.

Prefers blend pairwise win matrix. Power-ranking fallback is fail-hard by
default (``--require-blend``); pass ``--allow-power-ranking`` to opt in.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import n_simulations as default_n_simulations
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
from src.ti2026.fusion import (
    FUSION_WEIGHT_SCENARIOS,
    DEFAULT_MARKET_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_RANKING_WEIGHT,
    fuse_slot_probabilities,
    fuse_weight_scenarios,
    resolve_production_fusion_weight,
    tune_fusion_weight_loo,
)
from src.ti2026.multisource import (
    DEFAULT_HOME_LAN_ELO,
    PATCH_741_START_TS,
    PATCH_IN_MULT,
    home_lan_elo_bonus,
    load_market_priors,
)
from src.ti2026.expert_history import score_all_experts
from src.ti2026.teams import (
    POWER_RANKINGS,
    SWISS_CONFIG,
    TI2026_RECENT_RESULTS,
    TI2026_TEAMS,
    get_team_ids,
)
from src.features.match_features import replay_team_states, team_strength_summary
from src.features.team_stitching import apply_team_stitch, build_team_stitch_map
from src.data_collection.match_loader import load_raw_matchlists
from src.data_collection.tournaments import TOURNAMENTS

WEB_DATA = BASE_DIR / "docs" / "data"
METRICS_PATH = BASE_DIR / "outputs" / "xgb_v1_metrics.json"
COMPARE_PATH = BASE_DIR / "outputs" / "model_compare.json"
BLEND_PATH = BASE_DIR / "outputs" / "model_blend_v1.joblib"

DEFAULT_WARN_PLAYER_COVERAGE: float = 0.50
POWER_RANKING_KEY = "power_ranking_bradley_terry"
MIN_PAIRWISE_EDGES: int = 40


class BlendRequiredError(RuntimeError):
    """Blend pairwise unavailable and power-ranking fallback was not allowed."""


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


def load_stitched_corpus() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load matchlists + players with the same roster Jaccard stitch as train."""
    matches = load_raw_matchlists(BASE_DIR / "data" / "raw")
    players: pd.DataFrame | None = None
    try:
        from src.data_collection.match_details import load_player_matches

        players = load_player_matches(BASE_DIR / "data" / "raw")
        if players is not None and not players.empty:
            stitch = build_team_stitch_map(players, matches, threshold=0.6)
            matches, players = apply_team_stitch(matches, stitch, players)
    except Exception as exc:  # noqa: BLE001 — stitch is best-effort for export
        print(f"Warning: stitch/player load failed ({exc}); continuing without players")
        players = None
    return matches, players


def _fallback_or_raise(
    teams: list[str],
    reason: str,
    *,
    allow_power_ranking: bool,
) -> tuple[pd.DataFrame, str, str]:
    msg = f"Blend unavailable ({reason})"
    if not allow_power_ranking:
        raise BlendRequiredError(
            f"{msg}. Pass --allow-power-ranking to use power-ranking fallback, "
            "or run scripts/train_compare.py first."
        )
    print(f"WARNING: {msg}; using power ranking (explicit --allow-power-ranking)")
    return build_power_ranking_matrix(teams), POWER_RANKING_KEY, "Power ranking"


def build_export_win_matrix(
    teams: list[str],
    matches: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
    *,
    allow_power_ranking: bool = False,
) -> tuple[pd.DataFrame, str, str]:
    """Return (matrix, model_key, model_label). Prefer blend pairwise.

    Without ``allow_power_ranking``, missing/sparse blend raises BlendRequiredError.
    """
    if not BLEND_PATH.exists():
        return _fallback_or_raise(
            teams, f"missing {BLEND_PATH.name}", allow_power_ranking=allow_power_ranking
        )

    try:
        from src.ti2026.pairwise import build_model_win_matrix, load_blend_bundle

        if matches is None:
            matches, players = load_stitched_corpus()
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
        mapped = int((matrix.values != 0.5).sum() // 2)
        if mapped < MIN_PAIRWISE_EDGES:
            return _fallback_or_raise(
                teams,
                f"pairwise sparse ({mapped} directed edges < {MIN_PAIRWISE_EDGES})",
                allow_power_ranking=allow_power_ranking,
            )
        return matrix, "blend_pairwise_v1", "Blend pairwise"
    except BlendRequiredError:
        raise
    except Exception as exc:  # noqa: BLE001 — map unexpected failures to policy
        return _fallback_or_raise(teams, str(exc), allow_power_ranking=allow_power_ranking)


def check_player_coverage(
    coverage: dict | None,
    *,
    warn_threshold: float = DEFAULT_WARN_PLAYER_COVERAGE,
    min_coverage: float | None = None,
) -> None:
    """Warn or fail when player-detail coverage is below thresholds."""
    if not coverage or coverage.get("coverage") is None:
        print("WARNING: player_coverage unknown (no details / load failed)")
        if min_coverage is not None:
            raise SystemExit(
                f"Player coverage required (>= {min_coverage:.0%}) but unavailable"
            )
        return
    cov = float(coverage["coverage"])
    if min_coverage is not None and cov < min_coverage:
        raise SystemExit(
            f"Player coverage {cov:.1%} < --min-player-coverage {min_coverage:.0%}"
        )
    if cov < warn_threshold:
        print(
            f"WARNING: player coverage {cov:.1%} < {warn_threshold:.0%} "
            f"({coverage.get('n_matches_with_players')}/{coverage.get('n_matches')} matches)"
        )


def _human_methodology(model_metrics: dict | None) -> str:
    """Plain-Russian methodology blurb for the model status section."""
    cov = (model_metrics or {}).get("player_coverage") or {}
    cov_frac = cov.get("coverage")
    n_with = cov.get("n_matches_with_players")
    n_tot = cov.get("n_matches")
    if cov_frac is not None and n_with is not None and n_tot:
        cov_line = (
            f"У {cov_frac:.0%} матчей корпуса ({n_with}/{n_tot}) есть составы игроков; "
            "остальные ещё без скачанных деталей."
        )
    else:
        cov_line = (
            "У части матчей есть составы игроков; остальные ещё без скачанных деталей."
        )
    return (
        "Мы берём результаты матчей турниров и оцениваем силу каждой команды.\n"
        f"{cov_line}\n"
        "Модель сравнивает две команды и говорит, кто вероятнее победит.\n"
        "Затем много раз проигрываем весь Swiss-турнир и смотрим, куда чаще попадает каждая команда.\n"
        "Качество проверяем на прошлых TI и на свежих матчах.\n"
        "Итоговый прогноз смешивает модель с мнениями аналитиков и рыночными вероятностями.\n"
        "Число μ ± σ на сайте — сила команды ± насколько мы в ней уверены.\n"
        "Технические подробности — в разделе «Как считаем» ниже."
    )


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
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: player_coverage metrics failed: {exc}")
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
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"Warning: could not parse model_compare.json: {exc}")
    return metrics


def team_payload(
    team_id: str,
    row: pd.Series,
    win_matrix: pd.DataFrame,
    teams: list[str],
    strength: dict[str, float] | None = None,
) -> dict:
    """Собрать JSON-запись команды для UI."""
    info = TI2026_TEAMS[team_id]
    avg_wr = float(np.mean([win_matrix.loc[team_id, t] for t in teams if t != team_id]))
    # Informational only for main-event later; not applied to GS displayed μ.
    home_bonus = home_lan_elo_bonus(info["region"])
    strength = strength or {}
    mu = float(strength.get("mu", 1500.0))
    sigma = float(strength.get("sigma", 100.0))
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
        "strength_mu": round(mu, 1),
        "strength_sigma": round(sigma, 1),
        "strength_label": f"{mu:.0f} ± {sigma:.0f}",
        "home_lan_elo": home_bonus,
        "gp": float(strength.get("gp", 0.0)),
        "glicko_rd": round(float(strength.get("glicko_rd", 350.0)), 1),
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


def build_slot_heatmap(predictions: list[dict]) -> dict:
    """16×6 матрица P(slot) для heatmap UI."""
    slot_keys = [
        "prob_4_0",
        "prob_4_1",
        "prob_advance",
        "prob_eliminate",
        "prob_1_4",
        "prob_0_4",
    ]
    labels = ["4-0", "4-1", "advance", "eliminate", "1-4", "0-4"]
    teams = []
    matrix = []
    for p in predictions:
        teams.append({"id": p["id"], "name": p["name"], "short": p["short"]})
        matrix.append([round(float(p.get(k, 0.0)), 2) for k in slot_keys])
    return {"slots": labels, "teams": teams, "matrix": matrix}


def build_team_strengths(matches: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Canonical team_id → μ/σ после replay Elo/Glicko."""
    from src.ti2026.pairwise import resolve_opendota_team_ids

    if matches.empty:
        return {}
    store = replay_team_states(matches)
    as_of = int(matches["start_time"].max())
    odota = resolve_opendota_team_ids(matches, get_team_ids())
    out: dict[str, dict[str, float]] = {}
    for tid, oid in odota.items():
        out[tid] = team_strength_summary(store, oid, as_of)
    return out


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
    """Attach analyst agreement metadata (N of n_analysts) to every team pill."""
    out: dict[str, list[dict]] = {}
    for slot, entries in board.items():
        out[slot] = [enrich_board_with_analysts(board, slot, e) for e in entries]
    return out


LOW_AUC_WARN_THRESHOLD: float = 0.55


def _meta_warnings(model_metrics: dict, *, is_fallback: bool) -> list[str]:
    """Build UI warnings for low AUC / fallback / coverage / uncertainty."""
    warnings: list[str] = []
    if is_fallback:
        warnings.append(
            "Power-ranking fallback: pairwise blend недоступен — не используйте для решений."
        )
    auc = model_metrics.get("blend_leave_one_ti_avg_auc")
    if auc is None:
        auc = model_metrics.get("leave_one_ti_avg_auc")
    if auc is not None and float(auc) < LOW_AUC_WARN_THRESHOLD:
        warnings.append(
            f"Низкий Leave-One-TI AUC ({float(auc):.3f} < {LOW_AUC_WARN_THRESHOLD}): "
            "прогнозы сильно неопределённы."
        )
    cov = model_metrics.get("player_coverage") or {}
    cov_frac = cov.get("coverage")
    if cov_frac is not None and float(cov_frac) < DEFAULT_WARN_PLAYER_COVERAGE:
        warnings.append(
            f"Player coverage {float(cov_frac):.0%} < {DEFAULT_WARN_PLAYER_COVERAGE:.0%} "
            f"({cov.get('n_matches_with_players')}/{cov.get('n_matches')}); "
            "цель ≥80% — докачайте scripts/download_details.py."
        )
    return warnings


def build_matchups(win_matrix: pd.DataFrame, teams: list[str]) -> list[dict]:
    rows = []
    for i, a in enumerate(teams):
        for b in teams[i + 1 :]:
            p = float(win_matrix.loc[a, b])
            rows.append({"a": a, "b": b, "p_a": round(p, 3), "p_b": round(1 - p, 3)})
    return rows


def export_predictions(
    n_simulations: int | None = None,
    *,
    allow_power_ranking: bool = False,
    min_player_coverage: float | None = None,
) -> Path:
    """Build predictions.json for the static UI."""
    if n_simulations is None:
        n_simulations = default_n_simulations()
    teams = get_team_ids()
    matches, players = load_stitched_corpus()

    model_metrics = load_model_metrics()
    check_player_coverage(
        model_metrics.get("player_coverage"),
        min_coverage=min_player_coverage,
    )

    strengths = build_team_strengths(matches)
    win_matrix, model_key, model_label = build_export_win_matrix(
        teams,
        matches=matches,
        players=players,
        allow_power_ranking=allow_power_ranking,
    )
    is_fallback = model_key == POWER_RANKING_KEY
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
    use_uncertainty = bool(strengths) and any(
        (s or {}).get("sigma") or (s or {}).get("glicko_rd") for s in strengths.values()
    )
    results = simulate_swiss_stage(
        win_matrix,
        teams,
        config,
        n_simulations=n_simulations,
        rng_seed=42,
        team_strengths=strengths if use_uncertainty else None,
        sample_uncertainty=use_uncertainty,
    )

    predictions = [
        team_payload(
            row["team"],
            row,
            win_matrix,
            teams,
            strength=strengths.get(row["team"]),
        )
        for _, row in results.iterrows()
    ]
    predictions.sort(key=lambda x: (-x["qualify_pct"], x["power_rank"]))

    qualify_board = assign_fantasy_board(predictions)
    points_board = optimize_fantasy_board(predictions)
    consensus_board = build_consensus_board(predictions)
    market_priors = load_market_priors()
    tuned_w, fusion_pts = tune_fusion_weight_loo(predictions)
    # Production SoT: documented DEFAULT_MODEL_WEIGHT (tune is in-sample diagnostic).
    fusion_weight = resolve_production_fusion_weight(tuned_w)
    fusion_market_w = DEFAULT_MARKET_WEIGHT
    fusion_ranking_w = DEFAULT_RANKING_WEIGHT
    # Seeded market == POWER_RANKINGS soft mass — do not double-count with ranking.
    if market_priors.get("seeded_from_ranking") or not market_priors.get("is_real_market", True):
        fusion_market_w = 0.0
    # Residual analyst keeps production mix when market is forced off.
    fusion_analyst_w = max(
        0.0, 1.0 - float(fusion_weight) - float(fusion_market_w) - float(fusion_ranking_w)
    )
    fused_predictions = fuse_slot_probabilities(
        predictions,
        model_weight=fusion_weight,
        analyst_weight=fusion_analyst_w,
        market_weight=fusion_market_w,
        ranking_weight=fusion_ranking_w,
        market_priors=market_priors,
    )
    fusion_board = optimize_fantasy_board(fused_predictions)

    if fusion_market_w <= 0.0:
        # Rebuild scenarios without phantom market when priors are ranking-seeded.
        scenario_preds = {
            name: fuse_slot_probabilities(
                predictions,
                model_weight=float(w.get("model_weight", DEFAULT_MODEL_WEIGHT)),
                analyst_weight=float(
                    w["analyst_weight"]
                    if "analyst_weight" in w
                    else max(
                        0.0,
                        1.0
                        - float(w.get("model_weight", DEFAULT_MODEL_WEIGHT))
                        - 0.0
                        - float(w.get("ranking_weight", 0.0))
                        - float(w.get("expert_weight", 0.0)),
                    )
                ),
                market_weight=0.0,
                ranking_weight=float(w.get("ranking_weight", 0.0)),
                expert_weight=float(w.get("expert_weight", 0.0)),
                market_priors=market_priors,
                use_expert_history=float(w.get("expert_weight", 0.0)) > 0,
            )
            for name, w in FUSION_WEIGHT_SCENARIOS.items()
        }
    else:
        scenario_preds = fuse_weight_scenarios(predictions)
    scenario_boards = {
        name: enrich_full_board(optimize_fantasy_board(preds))
        for name, preds in scenario_preds.items()
    }
    scenario_compare = {
        name: compare_board_strategies(preds)["points_optimal"]
        for name, preds in scenario_preds.items()
    }

    board = enrich_full_board(points_board)
    boards_payload = {
        "points_optimal": enrich_full_board(points_board),
        "qualify_rank": enrich_full_board(qualify_board),
        "analyst_consensus": enrich_full_board(consensus_board),
        "fusion": enrich_full_board(fusion_board),
        **{f"fusion_{k}": v for k, v in scenario_boards.items()},
    }
    board_compare = compare_board_strategies(
        predictions,
        extra_boards={
            "analyst_consensus": consensus_board,
            "fusion": fusion_board,
        },
    )
    analyst_meta = consensus_summary(predictions)
    heatmap = build_slot_heatmap(predictions)
    expert_scores = score_all_experts()

    # Sanity: capacities
    for key, meta in FANTASY_BOARD_SLOTS.items():
        got = len(board[key])
        if got != meta["capacity"]:
            print(f"WARNING: board[{key}] has {got}, expected {meta['capacity']}")

    market_disclaimer = (
        "Рыночные вероятности — только исследовательский сигнал. "
        "Автор не рекламирует букмекеров. Не для ставок и не финансовый совет."
    )
    if model_key.startswith("blend"):
        disclaimer = (
            "Это исследовательский прогноз Swiss-доски и слотов компендиума TI 2026. "
            "Считается по матчевым данным: модель + мнения аналитиков + рыночные "
            "вероятности. Неопределённость высокая — не для ставок и не финансовый совет. "
            + market_disclaimer
        )
    else:
        disclaimer = (
            "Упрощённый режим: доска по power ranking (основная модель недоступна). "
            "Исследовательский прогноз с очень высокой неопределённостью — "
            "не для ставок и не финансовый совет. "
            + market_disclaimer
        )

    meta_warnings = _meta_warnings(model_metrics, is_fallback=is_fallback)
    if market_priors.get("seeded_from_ranking") or not market_priors.get("is_real_market", True):
        meta_warnings.append(
            "Market prior seeded from POWER_RANKINGS (not live odds); "
            "fusion market_weight forced to 0 to avoid double-counting ranking."
        )
    if abs(float(tuned_w) - float(fusion_weight)) > 1e-9:
        meta_warnings.append(
            f"In-sample tune suggested model_weight={tuned_w:.2f}; "
            f"export uses production default {fusion_weight:.2f}."
        )

    payload = {
        "meta": {
            "title": "TI 2026 Swiss Predictions",
            "subtitle": "Open-source group-stage forecast",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model_key,
            "model_label": model_label,
            "is_power_ranking_fallback": is_fallback,
            "disclaimer": disclaimer,
            "market_disclaimer": market_disclaimer,
            "research_disclaimer": (
                "All fusion sources (model, analyst, anonymous market odds_implied, "
                "ranking/expert-history) are research signals only."
            ),
            "warnings": meta_warnings,
            "format": "16-team Swiss to 4, 5 rounds Bo3 + ER (5 of 10)",
            "board_format": "4-0×1, 4-1×2, advance×5, eliminate×5, 1-4×2, 0-4×1",
            "n_simulations": n_simulations,
            "sample_uncertainty": use_uncertainty,
            "version": "0.3.0-prod",
            "board_strategy": "points_optimal",
            "expected_compendium_points": board_compare["points_optimal"]["expected_points"],
            "expected_correct_slots": board_compare["points_optimal"]["expected_correct"],
            "board_compare": board_compare,
            "fusion_model_weight": fusion_weight,
            "fusion_model_weight_tuned_in_sample": tuned_w,
            "fusion_tuned_expected_points_in_sample": fusion_pts,
            "fusion_analyst_weight": fusion_analyst_w,
            "fusion_market_weight": fusion_market_w,
            "fusion_ranking_weight": fusion_ranking_w,
            "fusion_expected_points": board_compare.get("fusion", {}).get("expected_points"),
            "fusion_weight_note": (
                "Production fusion uses DEFAULT_MODEL_WEIGHT=0.65. "
                "Soft weights are independent (need not sum to 1); fuse renormalizes. "
                "tune_fusion_weight_loo is in-sample diagnostic only (not true LOO CV)."
            ),
            "fusion_weight_scenarios": FUSION_WEIGHT_SCENARIOS,
            "fusion_scenario_scores": scenario_compare,
            "market_priors_meta": {
                "source": market_priors.get("source", "anonymous_market"),
                "seeded_from_ranking": market_priors.get("seeded_from_ranking"),
                "is_real_market": market_priors.get("is_real_market"),
                "updated_at": market_priors.get("updated_at"),
                "disclaimer": market_priors.get("disclaimer"),
            },
            "expert_history_scores": expert_scores,
            "n_leagues": len(TOURNAMENTS),
            "n_maps": int(len(matches)) if matches is not None else None,
            "home_lan_elo": DEFAULT_HOME_LAN_ELO,
            "patch_741_start_ts": PATCH_741_START_TS,
            "methodology": _human_methodology(model_metrics),
            "calibration_policy": (
                "XGB pipeline: optional isotonic on final fit. "
                "Blend production: isotonic when calibrate=True in train_compare. "
                "Brier + log-loss reported in model_compare metrics."
            ),
            "swiss_bye_policy": (
                "Odd leftover in a Swiss record bucket gets an implicit bye "
                "(no auto-win). Intentional MC simplification vs real TI Swiss."
            ),
            "player_coverage_gate": {
                "warn_below": DEFAULT_WARN_PLAYER_COVERAGE,
                "target": 0.8,
                "current": (model_metrics.get("player_coverage") or {}).get("coverage"),
            },
        },
        "analyst": analyst_meta,
        "boards": boards_payload,
        "slot_heatmap": heatmap,
        "model_metrics": model_metrics,
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
                    "strength": strengths.get(tid),
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
        print(
            f"  {p['power_rank']:2d}. {p['name']:20s} "
            f"qualify={p['qualify_pct']:5.1f}% strength={p['strength_label']}"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: export web JSON; fail hard without blend unless allowed."""
    parser = argparse.ArgumentParser(description="Export predictions.json for GitHub Pages")
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=None,
        help="Monte Carlo sims (default from settings.yaml / 50000)",
    )
    parser.add_argument(
        "--require-blend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if blend pairwise unavailable (default: true)",
    )
    parser.add_argument(
        "--allow-power-ranking",
        action="store_true",
        help="Allow power-ranking fallback (implies not requiring blend)",
    )
    parser.add_argument(
        "--min-player-coverage",
        type=float,
        default=None,
        help="Fail if player coverage fraction is below this (e.g. 0.5)",
    )
    args = parser.parse_args(argv)
    allow = bool(args.allow_power_ranking) or (not args.require_blend)
    try:
        export_predictions(
            n_simulations=args.n_simulations,
            allow_power_ranking=allow,
            min_player_coverage=args.min_player_coverage,
        )
    except BlendRequiredError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
