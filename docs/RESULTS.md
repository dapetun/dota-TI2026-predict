# Training results / Результаты обучения (v0.3)

**English summary:** Leave-One-TI-Out + walk-forward metrics for the XGB+CatBoost blend; player coverage **100%** (8709/8709, target ≥80%); export uses points-optimal + fusion boards. Details in Russian below / подробности ниже.

**RU · Дата:** 2026-08-08 · ветка `prod` · meta **0.3.0-prod**  
Официальные даты TI ([tirules](https://www.dota2.com/esports/ti15/tirules)): Group Stage **13–16 Aug** Online; Main Event **20–23 Aug** Shanghai.  
Корпус: **65 лиг**, **8709** уникальных карт  
Признаков: **88** · feature rows: **7396** (min_gp≥5)  
Player/chem coverage: **100%** (8709/8709) — shards `details_shards` (+ monolith shim); цель ≥80% **достигнута с запасом**.  
Team stitch: **82** team_id remapped (Jaccard≥0.6)  
Half-life: sample **210d** · form **~40d** · tier ti/major/qual/online  
Blend: XGB **0.25** / CatBoost **0.75** · isotonic в production blend (`calibrate=True`)  
Glicko: **Glicko-2** (μ/RD/σ, Illinois) в `rating_systems.GlickoRating`

## Coverage gate

| Метрика | Сейчас | Цель | Статус |
|---|---|---|---|
| Player coverage | **100%** | ≥80% | **OK** |
| Export warn | <50% | banner в UI | wired |
| Export fail | `--min-player-coverage` | optional hard gate | wired |

### Что такое player coverage (простыми словами)

**Player coverage** — доля матчей корпуса, для которых у нас есть детальный состав игроков (KDA, роли, chemistry).  
Считается в `summarize_player_coverage` (`src/data_collection/match_details.py`):

| | Число | Смысл |
|---|---|---|
| Знаменатель | **8709** | уникальные `match_id` в matchlists (корпус) |
| Числитель | **8709** | матчи с ≥1 player-row из details |
| Coverage | **100%** = 8709/8709 | цель ≥80% — достигнута; дыр нет |

Player-row появляется только если в details есть `players[]`, у игрока `account_id` ≠ 0/None и известен `team_id` стороны.

### Покрытие закрыто (было 1185 = 13.6%)

Ранее **1185** пропусков = матчи корпуса без скачанных details.  
Сейчас `download_details.py` → **Need download=0**; live coverage **8709/8709**.  
На диске **~9641** shard-файлов с players (часть orphans вне корпуса + legacy).  
**Не путать:** число keys на диске ≠ coverage. Coverage смотрит только на пересечение details ∩ 8709 matchlists.

## Сводка (Leave-One-TI / Walk-forward)

| Модель | LOO AUC | LOO LL | WF AUC | WF LL |
|---|---|---|---|---|
| XGBoost | 0.565 | 0.791 | 0.636 | 0.709 |
| CatBoost | 0.591 | 0.734 | 0.638 | 0.683 |
| Blend (LOO-tuned) | **0.586** | **0.733** | **0.642** | **0.681** |

### Как читать метрики

- **AUC** — насколько модель лучше ранжирует победителя vs проигравшего (0.5 = монетка, 1.0 = идеал). У нас ~0.58–0.64: слабый, но стабильный сигнал.
- **Log-loss (LL)** — штраф за неуверенные/некалиброванные вероятности; ниже лучше.
- **Leave-One-TI-Out (LOO)** — учимся на всём до старта TI_k, тестируем только карты этого TI. Ближе к «как сработает на TI 2026».
- **Walk-forward (WF)** — расширяющееся временное окно по календарю; проверка общей временной устойчивости.

vs предыдущий retrain (coverage 86.4%): LOO AUC 0.584→**0.586**, LOO LL 0.734→**0.733**, WF AUC 0.640→**0.642**, WF LL 0.682→**0.681**.  
Player/chem теперь на полном корпусе details — метрики чуть стабильнее.

## Leave-One-TI (blend)

| Held-out | n | LL | AUC |
|---|---|---|---|
| TI11 | 213 | 0.844 | 0.517 |
| TI12 | 141 | 0.673 | **0.639** |
| TI13 | 121 | 0.735 | 0.560 |
| TI14 | 144 | 0.681 | **0.628** |

### TI14

AUC **0.628** (было 0.583 при coverage 86.4%) — fold сильнее с полным player/chem coverage.

## Calibration policy

- **XGB** `train_xgboost_pipeline`: финальный fit может включать isotonic (`calibrate_final=True`).
- **Blend production** `model_blend_v1.joblib`: **isotonic** при `calibrate=True` в `train_compare` (default on).
- Метрики: LOO/WF **Brier + log-loss** (+ AUC) в `model_compare.json`.

## Топ-признаки (сигнал v0.3, retrain 2026-08-08)

CatBoost: `r_glicko_mu`, `diff_pl_hdpm`, `d_pl_tdpm`, `diff_opp_avg_elo`, `r_pl_tdpm`.  
XGBoost: `diff_glicko_mu`, `diff_pl_uncertainty`, `diff_pl_games`, `diff_chem_pair_wr`, `r_pl_uncertainty`.

## UI

http://localhost:8080 — Swiss на blend pairwise (или power-ranking fallback).  
Секция **«Как считаем»** — простое объяснение pipeline / coverage / μ±σ / MC / LOO·WF.  
Сила **μ ± σ** (Elo shrink + Glicko-2 RD) · home LAN +30 (CN) в meta, не в μ GS · heatmap 16×6 · fusion model⊕analyst⊕market⊕ranking.  
Слайдеры весов — precomputed scenarios в `predictions.json`.  
CSP meta + escapeHtml на все `innerHTML` пути.  
**Market disclaimer** в UI / export meta / README (без брендов БК).

## Swiss bye

В MC odd leftover в record-bucket получает **implicit bye без авто-победы** (упрощение vs реальный TI Swiss). См. `tournament_sim` + `meta.swiss_bye_policy`.  
Опционально: sample latent strength из σ/RD (`sample_uncertainty`).

## Ablation / заметки

- Player coverage **100%** (8709/8709); shards `details_shards` ~9641 с players. Не смешивать число файлов на диске с coverage.
- Details storage: shards `data/raw/details_shards/<tourn>/<match_id>.json` + legacy monolith shim.
- `lan_only_chemistry` — флаг в `train_compare(..., lan_only_chemistry=True)`.
- Market prior — `data/ti2026_market_priors.json` (anonymous; seed from POWER_RANKINGS).
- Expert history — `data/historical/expert_predictions.json`; GT Swiss — `ti_swiss_ground_truth.json` (TI14 Swiss; TI13 soft RR map).
- Hero soft prior — experimental (`USE_HERO_SOFT_PRIOR=0`); fixtures in `data/hero/`.
- Dual Elo online/LAN + MoV K — wired in `TeamStateStore` / pairwise GS `prefer_lan=False`.
- Playoff — stub `simulation.playoff_stub` (Stage 6).

## Swiss backtest

`src/ti2026/swiss_backtest.py`: toy BT → MC → points-optimal → score vs Liquipedia GT; expert hit-rate.  
См. pytest `test_swiss_backtest_ti14_runs`.

## Компендиум (export 0.3.0-prod, 2026-08-08)

| Стратегия | E[очки Valve] (approx) |
|---|---|
| Points-optimal | ~1301 |
| Qualify-rank | ~1301 |

Сила в UI: `strength_mu ± strength_sigma`; heatmap в `slot_heatmap`.  
Export: `blend_pairwise_v1`, 50 000 MC sims (default; re-export to refresh predictions.json), coverage gate current=1.0.

## Дальше

Live anonymous odds в `ti2026_market_priors.json`; `build_hero_meta.py --fetch` / `build_player_signatures.py --fetch`.

CLI: `discover_leagues.py` · `download_data.py` · `download_details.py` · `train_compare.py` · `export_web_data.py` · `build_rosters.py` · `build_hero_meta.py` · `build_player_signatures.py`
