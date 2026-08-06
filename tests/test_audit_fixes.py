"""Regression tests for audit fixes: half-life defaults, export blend gate, OpenDota retries, normalize SoT."""

from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest

from src.data_collection.opendota_api import OpenDotaClient, OpenDotaRateLimitError
from src.features.sample_weights import RATING_HALF_LIFE_DAYS, compute_sample_weights
from src.models.artifact_hash import sha256_file, verify_artifact_sha256, write_sha256_sidecar
from src.models.catboost_model import train_catboost_pipeline
from src.models.ensemble import tune_blend_weights_loo, train_blend_pipeline
from src.models.validation import evaluate_folds
from src.models.xgboost_model import train_xgboost_pipeline
from src.ti2026.multisource import PATCH_741_START_TS, PATCH_IN_MULT
from src.ti2026.teams import normalize_team_name as teams_normalize


def test_rating_half_life_default_constant():
    assert RATING_HALF_LIFE_DAYS == 210.0


@pytest.mark.parametrize(
    "fn",
    [
        evaluate_folds,
        train_xgboost_pipeline,
        train_catboost_pipeline,
        train_blend_pipeline,
        tune_blend_weights_loo,
    ],
)
def test_half_life_defaults_use_rating_constant(fn):
    params = inspect.signature(fn).parameters
    assert "half_life_days" in params
    default = params["half_life_days"].default
    assert default == RATING_HALF_LIFE_DAYS


def test_normalize_team_name_sot_shared():
    from src.data_collection import data_loader

    assert data_loader.normalize_team_name is teams_normalize
    assert teams_normalize("PARIVISION") == "Vision"
    assert teams_normalize("BoomBoys") == "BetBoom"
    assert data_loader.normalize_team_name("PARIVISION") == "Vision"


def test_opendota_429_raises_after_max_retries():
    client = OpenDotaClient(rate_limit=0.0, max_retries=2, retry_backoff_sec=0.01)
    resp = MagicMock()
    resp.status_code = 429
    with patch.object(client.session, "get", return_value=resp) as mock_get:
        with pytest.raises(OpenDotaRateLimitError):
            client._get("/matches/1")
    assert mock_get.call_count == 3  # initial + 2 retries


def test_export_require_blend_fails_without_allow(tmp_path, monkeypatch):
    import scripts.export_web_data as export

    monkeypatch.setattr(export, "BLEND_PATH", tmp_path / "missing_blend.joblib")
    teams = ["Liquid", "Spirit"]
    with pytest.raises(export.BlendRequiredError):
        export.build_export_win_matrix(teams, allow_power_ranking=False)

    matrix, key, label = export.build_export_win_matrix(teams, allow_power_ranking=True)
    assert key == export.POWER_RANKING_KEY
    assert matrix.shape == (2, 2)


def test_check_player_coverage_warn_and_fail(capsys):
    import scripts.export_web_data as export

    export.check_player_coverage(
        {"coverage": 0.3, "n_matches_with_players": 3, "n_matches": 10},
        warn_threshold=0.5,
    )
    assert "WARNING: player coverage" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        export.check_player_coverage(
            {"coverage": 0.2, "n_matches_with_players": 2, "n_matches": 10},
            min_coverage=0.5,
        )


def test_config_half_life_from_yaml():
    from src.config import load_settings, sample_half_life_days

    load_settings.cache_clear()
    assert sample_half_life_days() == 210.0


def test_resolve_opendota_single_scan():
    from src.ti2026.pairwise import resolve_opendota_team_ids

    matches = pd.DataFrame(
        [
            {
                "start_time": 100,
                "radiant_canonical": "Liquid",
                "dire_canonical": "Spirit",
                "radiant_team_id": 1,
                "dire_team_id": 2,
            },
            {
                "start_time": 200,
                "radiant_canonical": "Liquid",
                "dire_canonical": "Falcons",
                "radiant_team_id": 11,
                "dire_team_id": 3,
            },
        ]
    )
    mapping = resolve_opendota_team_ids(matches, ["Liquid", "Spirit", "Falcons"])
    assert mapping["Liquid"] == 11  # most recent
    assert mapping["Spirit"] == 2
    assert mapping["Falcons"] == 3


def test_patch_window_wired_into_sample_weights():
    """In-patch rows get PATCH_IN_MULT before mean-normalization."""
    df = pd.DataFrame(
        {
            "start_time": [PATCH_741_START_TS - 10, PATCH_741_START_TS + 10],
            "tier_weight": [1.0, 1.0],
        }
    )
    w_off = compute_sample_weights(
        df, reference_time=PATCH_741_START_TS + 100, patch_mult=1.0, patch_start_ts=None
    )
    w_on = compute_sample_weights(
        df,
        reference_time=PATCH_741_START_TS + 100,
        patch_mult=PATCH_IN_MULT,
        patch_start_ts=PATCH_741_START_TS,
    )
    ratio_off = w_off[1] / w_off[0]
    ratio_on = w_on[1] / w_on[0]
    assert ratio_on / ratio_off == pytest.approx(PATCH_IN_MULT, rel=1e-6)


def test_artifact_sha256_warn_and_fail(tmp_path):
    path = tmp_path / "model_blend_v1.joblib"
    joblib.dump({"ok": True}, path)
    digest = write_sha256_sidecar(path)
    assert sha256_file(path) == digest
    assert verify_artifact_sha256(path, digest) == digest

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_artifact_sha256(path, "0" * 64)

    bad = tmp_path / "orphan.joblib"
    joblib.dump({"x": 1}, bad)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verify_artifact_sha256(bad)
    assert any("No SHA256" in str(w.message) for w in caught)


def test_load_blend_bundle_hash_gate(tmp_path):
    from src.ti2026.pairwise import load_blend_bundle

    path = tmp_path / "model_blend_v1.joblib"
    joblib.dump({"model": {}, "feature_cols": [], "params": {}}, path)
    write_sha256_sidecar(path)
    bundle = load_blend_bundle(path)
    assert "model" in bundle

    path.write_bytes(path.read_bytes() + b"x")  # corrupt after hash written
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_blend_bundle(path)


def test_meta_warnings_low_auc():
    import scripts.export_web_data as export

    warns = export._meta_warnings(
        {"blend_leave_one_ti_avg_auc": 0.51}, is_fallback=False
    )
    assert any("AUC" in w for w in warns)
    asserts_fallback = export._meta_warnings({}, is_fallback=True)
    assert any("fallback" in w.lower() or "Power-ranking" in w for w in asserts_fallback)


def test_curated_team_id_map_in_repo():
    from src.data_collection.match_loader import _load_team_id_map

    curated = Path("data/team_id_map.json")
    assert curated.exists()
    data = json.loads(curated.read_text(encoding="utf-8"))
    assert "2163" in data
    mapping = _load_team_id_map(Path("data/raw_missing_for_test_xyz"))
    assert mapping[2163] == "Team Liquid"


def test_tune_fusion_weight_loo_doc_mentions_in_sample():
    from src.ti2026.fusion import tune_fusion_weight_loo

    assert "in-sample" in (tune_fusion_weight_loo.__doc__ or "").lower()


def test_glicko2_update_and_inactive():
    from src.features.rating_systems import GLICKO2_INITIAL_RD, GlickoRating

    g = GlickoRating()
    g.update(1, 2)
    w, l = g.get(1), g.get(2)
    assert w["mu"] > 1500.0
    assert l["mu"] < 1500.0
    assert w["rd"] < GLICKO2_INITIAL_RD
    assert l["rd"] < GLICKO2_INITIAL_RD
    assert 0.0 < w["vol"] < 1.0

    g2 = GlickoRating()
    g2._update_one(10, 1500.0, 50.0, 1.0)
    before = g2.get(10)["rd"]
    g2.advance_inactive(10, n_periods=3)
    assert g2.get(10)["rd"] >= before


def test_detail_shards_preferred_over_monolith(tmp_path):
    from src.data_collection.match_details import (
        atomic_write_json,
        load_player_matches,
        shard_path_for_match,
    )

    raw = tmp_path / "raw"
    raw.mkdir()
    atomic_write_json(
        raw / "match_details.json",
        {"1": {"match_id": 1, "players": []}},
    )
    detail = {
        "match_id": 1,
        "start_time": 100,
        "radiant_win": True,
        "radiant_team_id": 10,
        "dire_team_id": 20,
        "players": [
            {
                "account_id": 1000 + i,
                "isRadiant": i < 5,
                "kills": 1,
                "deaths": 1,
                "assists": 1,
                "gold_per_min": 400,
                "xp_per_min": 400,
                "hero_id": 1,
            }
            for i in range(10)
        ],
    }
    atomic_write_json(shard_path_for_match(raw, 1, "TI14_2025"), detail)
    players = load_player_matches(raw)
    assert not players.empty
    assert set(players["match_id"]) == {1}


def test_playoff_stub_safe():
    from src.simulation.playoff_stub import simulate_playoffs_stub

    out = simulate_playoffs_stub(["A", "B"])
    assert out["implemented"] is False
    assert out["champion_probs"] == {}


def test_export_meta_contract_keys():
    """Smoke: export warnings include fallback + low coverage."""
    import scripts.export_web_data as export

    warns = export._meta_warnings(
        {
            "blend_leave_one_ti_avg_auc": 0.51,
            "player_coverage": {
                "coverage": 0.14,
                "n_matches_with_players": 100,
                "n_matches": 700,
            },
        },
        is_fallback=True,
    )
    assert any("fallback" in w.lower() or "Power-ranking" in w for w in warns)
    assert any("coverage" in w.lower() for w in warns)
    assert export.DEFAULT_WARN_PLAYER_COVERAGE == 0.5
