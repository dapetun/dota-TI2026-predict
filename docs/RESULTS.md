# Результаты обучения (v0.3)

Дата: 2026-08-06 · ветка `prod` · meta **0.3.0-prod**  
Официальные даты TI ([tirules](https://www.dota2.com/esports/ti15/tirules)): Group Stage **13–16 Aug** Online; Main Event **20–23 Aug** Shanghai.  
Корпус: **65 лиг**, **8709** уникальных карт (было 12 / 2134)  
Признаков: **88** · feature rows: **7285** (min_gp≥5)  
Player/chem coverage: **~13.6%** (1185/8709) — OpenDota `/matches/{id}` массово timeout (2026-08-06);  
докачка resume-friendly + shards: `python scripts/download_details.py` (цель ≥80%, gate в export).  
Team stitch: **19** team_id remapped (Jaccard≥0.6)  
Half-life: sample **210d** · form **~40d** · tier ti/major/qual/online  
Blend: XGB **0.25** / CatBoost **0.75** · isotonic только offline LOO  
Glicko: **Glicko-2** (μ/RD/σ, Illinois) в `rating_systems.GlickoRating`

## Coverage gate

| Метрика | Сейчас | Цель | Статус |
|---|---|---|---|
| Player coverage | ~13.6% | ≥80% | **OPEN** — API timeouts; shards готовы |
| Export warn | <50% | banner в UI | wired |
| Export fail | `--min-player-coverage` | optional hard gate | wired |

## Сводка (Leave-One-TI / Walk-forward)

| Модель | LOO AUC | LOO LL | WF AUC | WF LL |
|---|---|---|---|---|
| XGBoost | 0.585 | 0.767 | 0.622 | 0.720 |
| CatBoost | 0.597 | 0.730 | 0.623 | 0.692 |
| Blend (LOO-tuned) | **0.599** | **0.725** | **0.628** | **0.688** |

vs v0.2 blend: LOO AUC 0.598→0.599 (плоско), LOO LL 0.797→**0.725**, WF AUC 0.584→**0.628**.

## Leave-One-TI (blend)

| Held-out | n | LL | AUC |
|---|---|---|---|
| TI11 | 213 | 0.830 | 0.533 |
| TI12 | 141 | 0.696 | 0.590 |
| TI13 | 121 | 0.616 | **0.720** |
| TI14 | 144 | 0.759 | **0.553** |

### TI14

AUC **0.553** (было ~0.490) — выше случайного; расширенный корпус majors/quals + uncertainty/chem 2.0 стабилизировали fold.

## Calibration policy

- **XGB** `train_xgboost_pipeline`: финальный fit может включать isotonic (`calibrate_final=True`).
- **Blend production** `model_blend_v1.joblib`: **без** isotonic (`calibrate=False`).
- Isotonic на pooled LOO blend — только offline eval (`isotonic_loo_evaluated` в `model_compare.json`).

## Топ-признаки (сигнал v0.3)

CatBoost: `diff_opp_avg_elo`, `min_gp`, `chem_pair_wr`, `pl_uncertainty`, `roster_stability_60d`.  
XGBoost: `diff_elo`, `elo_prob`, `diff_opp_avg_elo`, `pl_uncertainty`.

## UI

http://localhost:8080 — Swiss на blend pairwise (или power-ranking fallback).  
Сила **μ ± σ** (Elo shrink + Glicko-2 RD) · home LAN +30 (CN) в meta, не в μ GS · heatmap 16×6 · fusion default model_weight≈0.65.  
CSP meta + escapeHtml на все `innerHTML` пути.

## Swiss bye

В MC odd leftover в record-bucket получает **implicit bye без авто-победы** (упрощение vs реальный TI Swiss). См. `tournament_sim` + `meta.swiss_bye_policy`.

## Ablation / заметки

- LOO AUC не упал >0.01 vs v0.2 при 4× корпусе; LL и WF AUC заметно лучше.
- Player coverage низкая из‑за таймаутов OpenDota details; team Elo/Glicko-2 уже на полном корпусе.
- Details storage: shards `data/raw/details_shards/<tourn>/<match_id>.json` + legacy monolith shim.
- `lan_only_chemistry` — флаг в `train_compare(..., lan_only_chemistry=True)`.
- Market prior — stub в `multisource.market_slot_prior_stub` (P2).
- Playoff — stub `simulation.playoff_stub` (Stage 6).

## Компендиум (export 0.3.0-prod)

| Стратегия | E[очки Valve] (approx) |
|---|---|
| Points-optimal | ~4080 |
| Qualify-rank | ~4052 |

Сила в UI: `strength_mu ± strength_sigma`; heatmap в `slot_heatmap`.

## Дальше

Докачать details ≥80% coverage (когда OpenDota отвечает) → retrain; playoff bracket; market prior.

CLI: `discover_leagues.py` · `download_data.py` · `download_details.py` · `train_compare.py` · `export_web_data.py` · `build_rosters.py`
