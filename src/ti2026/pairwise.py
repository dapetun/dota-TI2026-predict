"""Pairwise win matrix for TI 2026 Swiss from trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.chemistry_features import (
    ChemistryState,
    compose_chemistry_pair_features,
    replay_chemistry_state,
)
from src.features.match_features import (
    FEATURE_COLUMNS,
    compose_pair_features,
    replay_team_states,
)
from src.features.player_features import (
    compose_player_pair_features,
    replay_player_states,
)
from src.models.ensemble import ensemble_predict
from src.ti2026.teams import get_team_ids


ROSTERS_PATH = Path(__file__).resolve().parents[2] / "data" / "ti2026_rosters.json"


def load_ti2026_roster_account_ids(team_id: str) -> list[int]:
    """Explicit TI roster account_ids if `data/ti2026_rosters.json` exists."""
    if not ROSTERS_PATH.exists():
        return []
    import json

    with open(ROSTERS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    team = (data.get("teams") or {}).get(team_id) or {}
    ids = team.get("account_ids") or []
    return [int(x) for x in ids if int(x) > 0]


def lineup_for_team(
    team_id: str,
    players: pd.DataFrame | None,
    odota_team_id: int,
) -> list[int]:
    """TI roster override, else latest lineup from match history."""
    explicit = load_ti2026_roster_account_ids(team_id)
    if explicit:
        return explicit
    return latest_lineup_for_team(players, odota_team_id)


def resolve_opendota_team_ids(
    matches: pd.DataFrame,
    team_ids: list[str] | None = None,
) -> dict[str, int]:
    """Map TI2026 ids → most recent OpenDota team_id in match history."""
    team_ids = team_ids or get_team_ids()
    mapping: dict[str, int] = {}
    if matches.empty:
        return mapping

    df = matches.sort_values("start_time")
    for tid in team_ids:
        for _, row in df.iloc[::-1].iterrows():
            if row.get("radiant_canonical") == tid:
                mapping[tid] = int(row["radiant_team_id"])
                break
            if row.get("dire_canonical") == tid:
                mapping[tid] = int(row["dire_team_id"])
                break
    return mapping


def latest_lineup_for_team(players: pd.DataFrame, odota_team_id: int) -> list[int]:
    """Most recent 5 account_ids for an OpenDota team_id."""
    if players is None or players.empty or odota_team_id <= 0:
        return []
    sub = players[players["team_id"] == odota_team_id].sort_values("start_time")
    if sub.empty:
        return []
    last_mid = int(sub.iloc[-1]["match_id"])
    group = sub[sub["match_id"] == last_mid]
    seen: set[int] = set()
    ids: list[int] = []
    for aid in group["account_id"].tolist():
        aid = int(aid)
        if aid > 0 and aid not in seen:
            seen.add(aid)
            ids.append(aid)
    return ids


def load_blend_bundle(path: str | Path = "outputs/model_blend_v1.joblib") -> dict[str, Any]:
    """Load blend joblib produced by train_compare."""
    return joblib.load(path)


def compose_full_pair_row(
    team_store,
    player_states: dict,
    chem_state: ChemistryState,
    radiant_odota_id: int,
    dire_odota_id: int,
    radiant_ids: list[int],
    dire_ids: list[int],
    as_of_ts: int,
    tier_weight: float = 2.0,
) -> dict[str, float]:
    """Team + player + chemistry features for one radiant/dire orientation."""
    row = compose_pair_features(
        team_store, radiant_odota_id, dire_odota_id, as_of_ts, tier_weight=tier_weight
    )
    row.update(compose_player_pair_features(radiant_ids, dire_ids, player_states))
    row.update(
        compose_chemistry_pair_features(
            radiant_ids,
            dire_ids,
            radiant_odota_id,
            dire_odota_id,
            chem_state,
            as_of_ts,
        )
    )
    return row


def predict_pair_proba(
    models: dict[str, Any],
    feature_cols: list[str],
    team_store,
    player_states: dict,
    chem_state: ChemistryState,
    radiant_odota_id: int,
    dire_odota_id: int,
    radiant_ids: list[int],
    dire_ids: list[int],
    as_of_ts: int,
    tier_weight: float = 2.0,
    blend_weights: dict[str, float] | None = None,
) -> float:
    """P(radiant wins map) for one orientation."""
    row = compose_full_pair_row(
        team_store,
        player_states,
        chem_state,
        radiant_odota_id,
        dire_odota_id,
        radiant_ids,
        dire_ids,
        as_of_ts,
        tier_weight,
    )
    X = pd.DataFrame([{c: float(row.get(c, 0.0)) for c in feature_cols}])
    if isinstance(models, dict) and "xgb" in models and "catboost" in models:
        cal = models.get("isotonic_calibrator")
        return float(
            ensemble_predict(
                models,
                X,
                weights=blend_weights,
                feature_cols=feature_cols,
                isotonic_calibrator=cal,
            )[0]
        )
    model = models["model"] if isinstance(models, dict) and "model" in models else models
    return float(model.predict_proba(X)[:, 1][0])


def build_model_win_matrix(
    matches: pd.DataFrame,
    models: dict[str, Any],
    feature_cols: list[str],
    players: pd.DataFrame | None = None,
    team_ids: list[str] | None = None,
    tier_weight: float = 2.0,
    blend_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """16×16 map-win matrix; P(A beats B) symmetrized over radiant/dire."""
    team_ids = team_ids or get_team_ids()
    team_store = replay_team_states(matches)
    player_states = replay_player_states(matches, players) if players is not None else {}
    chem_state = replay_chemistry_state(matches, players) if players is not None else ChemistryState()
    as_of = int(matches["start_time"].max()) if not matches.empty else 0
    odota = resolve_opendota_team_ids(matches, team_ids)
    lineups = {
        tid: lineup_for_team(tid, players, oid) if players is not None else load_ti2026_roster_account_ids(tid)
        for tid, oid in odota.items()
    }

    n = len(team_ids)
    mat = np.full((n, n), 0.5)
    for i, a in enumerate(team_ids):
        for j, b in enumerate(team_ids):
            if i == j:
                continue
            a_id = odota.get(a)
            b_id = odota.get(b)
            if a_id is None or b_id is None:
                continue
            a_ids = lineups.get(a, [])
            b_ids = lineups.get(b, [])
            p_ab = predict_pair_proba(
                models,
                feature_cols,
                team_store,
                player_states,
                chem_state,
                a_id,
                b_id,
                a_ids,
                b_ids,
                as_of,
                tier_weight,
                blend_weights,
            )
            p_ba = predict_pair_proba(
                models,
                feature_cols,
                team_store,
                player_states,
                chem_state,
                b_id,
                a_id,
                b_ids,
                a_ids,
                as_of,
                tier_weight,
                blend_weights,
            )
            mat[i, j] = 0.5 * (p_ab + (1.0 - p_ba))
    return pd.DataFrame(mat, index=team_ids, columns=team_ids)
