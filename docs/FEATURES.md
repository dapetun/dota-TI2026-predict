# Feature catalog / Каталог признаков

**EN:** All features are computed **before** the current match outcome (no leakage). Tables below are bilingual in intent: Russian descriptions with English column names as in code.

**RU:** Все признаки считаются **до** исхода текущего матча (нет leakage).

## Data → features / Данные → признаки (обзор)

| Слой | Откуда | Зачем |
|---|---|---|
| Matchlists | `data/raw/*_matches.json` | Корпус карт, Elo/form без составов |
| Details (shards) | `data/raw/details_shards/…` + monolith shim | Player rows, chemistry |
| Player coverage | `summarize_player_coverage` | Доля корпуса с players; сейчас **100%** (8709/8709) |

Без details матч всё равно участвует в team Elo/form, но player/chem-фичи = 0 и `has_player_stats=0`.  
Coverage закрыто: дыр 0; не путать с числом shard-файлов на диске (~9641).

## Team features

| Признак | Источник | Как считается | Зачем | Ожидаемое влияние |
|---|---|---|---|---|
| `r_elo` / `d_elo` | OpenDota matchlists | Running Elo, K зависит от tier | Базовая сила команды | Сильное |
| `diff_elo` | derived | `r_elo - d_elo` | Главный сигнал перевеса | Очень сильное |
| `abs_elo_diff` | derived | `\|diff_elo\|` | Нелинейность «разгромов» | Среднее |
| `elo_prob` | derived | logistic по **shrunk** Elo | Калиброванный prior | Сильное |
| `r_wr` / `d_wr` | match history | Decayed WR ≤15 карт + half-life **~40d** | Среднесрочная форма | Среднее–сильное |
| `diff_wr` | derived | разница form WR | Относительная форма | Сильное |
| `r_wr5` / `d_wr5` | match history | WR последних 5 карт | Короткая форма / momentum | Среднее |
| `diff_wr5` | derived | разница short-form | Hot/cold streaks | Среднее |
| `r_streak` / `d_streak` | match history | текущая win-streak | Momentum | Слабое–среднее |
| `diff_streak` | derived | разница стриков | — | Слабое |
| `h2h_wr` | H2H cache | WR radiant в последних ≤10 очных | Matchup-специфика | Среднее (редко) |
| `diff_h2h` | derived | `h2h_wr - 0.5` | Центрированный H2H | Среднее |
| `r_gp` / `d_gp` | history length | число карт до матча | Cold-start / опыт | Слабое–среднее |
| `diff_gp` | derived | разница опыта | — | Слабое |
| `r_avg_tier` / `d_avg_tier` | history | средний tier_weight последних 10 | Уровень соперничества | Среднее |
| `diff_tier` | derived | разница avg tier | — | Слабое–среднее |
| `tier_weight` | tournament registry | ti=2.0, major=1.5, qual/dpc=0.75, online=0.5 | Контекст важности | Слабое как фича (важно как sample weight) |
| `r_days_since` / `d_days_since` | last played ts | дни простоя | Ржавчина / пауза | Слабое–среднее |
| `min_gp` | derived | `min(r_gp, d_gp)` | Совместный cold-start | Среднее |
| `r_uncertainty` / `d_*` / `diff_*` | derived | `1/sqrt(gp+1)` | Явная неопределённость | Среднее |
| `r_elo_shrunk` / `d_*` / `diff_*` | Empirical Bayes | `w=gp/(gp+k)`, k=12 → prior 1500 | Стабилизация новичков | Сильное |
| `r_opp_avg_elo` / `d_*` / `diff_*` | schedule | средний Elo соперников до матча | Сила календаря | Среднее |
| `r_glicko_mu` / `d_*` / `diff_*` | Glicko-2 (Glickman) | μ рейтинга | Альтернативная сила | Среднее–сильное |
| `r_glicko_rd` / `d_*` / `diff_*` | Glicko-2 | rating deviation (φ→RD) + volatility σ | Uncertainty μ±σ в UI | Среднее |

## Player features

Строятся по `account_id` из OpenDota match details. История игрока обновляется **после** матча.  
Player-row: элемент `players[]` с `account_id` ≠ 0/None и известным `team_id` стороны (`detail_to_player_rows`).

| Признак | Источник | Как считается | Зачем | Ожидаемое влияние |
|---|---|---|---|---|
| `r_pl_kda` … `diff_pl_lan_wr` | player details | как в v0.2 (rolling ≤20) | Индивидуальная форма | Среднее–сильное |
| `r_pl_uncertainty` / `d_*` / `diff_*` | derived | `1/sqrt(pl_games+1)` | Cold-start игрока | Слабое–среднее |
| `has_player_stats` | coverage flag | 1 если есть details | Модель видит пропуски | Служебное |

## Chemistry / co-play

| Признак | Смысл |
|---|---|
| `r_chem_mean` / `d_*` / `diff_*` | Среднее число совместных карт по парам |
| `r_chem_min` / `d_*` / `diff_*` | Самая «холодная» пара |
| `r_chem_90d` / `d_*` / `diff_*` | Mean только по парам с матчем ≤90 дней |
| `r_roster_jaccard` / `d_*` / `diff_*` | Непрерывность vs прошлый матч `team_id` |
| `r_chem_pair_wr` / `d_*` / `diff_*` | Joint WR пар в пятёрке |
| `r_roster_stability_60d` / `d_*` / `diff_*` | Mean Jaccard vs lineup за 60 дней |
| `has_chemistry` | Есть lineup details |

`lan_only=True` — ablation: обновление chemistry только с LAN-матчей (`train_compare(..., lan_only_chemistry=True)`).

## Sample weights

`w = 0.5 ** (age_days / 210) * tier_weight`, затем:

- down-weight если обе стороны `gp < 8` (×0.55);
- нормализация среднего к 1.

Разделение half-life: **rating/sample ≈ 210d**, **form features ≈ 40d**.

## Team stitching

`src/features/team_stitching.py` — Jaccard игроков ≥0.6 между `team_id` → общая Elo/form история (ребрендинг).

## Multi-source (TI export)

| Источник | Модуль | Поведение |
|---|---|---|
| Home LAN | `src/ti2026/multisource.py` | +30 Elo CN/Shanghai — meta/informational; в μ GS не добавляется |
| Patch 7.41 | `PATCH_741_START_TS` ≈ 2026-03-24 | **wired** в `compute_sample_weights` (`PATCH_IN_MULT=1.25`) |
| Fusion | `src/ti2026/fusion.py` | model + analyst + anonymous market + ranking/expert; UI scenarios |
| Market | `data/ti2026_market_priors.json` | curated `odds_implied` (research only; seeded from POWER_RANKINGS) |
| Ranking prior | `ranking_slot_prior` | intentional Bayesian soft prior from POWER_RANKINGS |
| Dual Elo | `TeamStateStore` | `elo_online` / `elo_lan`; GS pairwise `prefer_lan=False` |
| MoV K | `margin_of_victory_k_scale` | scale Elo K from kill score (fallback duration) |
| Hero soft prior | `hero_soft_prior.py` | experimental logit shift; `USE_HERO_SOFT_PRIOR=0` default |
| Swiss uncertainty | `simulate_swiss_stage` | optional sample from strength σ / Glicko RD |

## Модели

- `scripts/train.py` — XGBoost
- `scripts/train_compare.py` — XGBoost + CatBoost + LOO-tuned blend (**isotonic** when `calibrate=True`)
- Brier + log-loss в `model_compare.json`

## Pairwise / UI

`src/ti2026/pairwise.py` — 16×16 P(win); Swiss MC → P(слот); export: `strength_mu ± strength_sigma`, slot heatmap 16×6, fusion weight scenarios.

| Число в UI | Смысл |
|---|---|
| **μ ± σ** | Сила команды (Elo shrink + Glicko) и неопределённость |
| Heatmap % | Доля MC-симов, где команда попала в слот компендиума |
| Пройти / вылететь | Доля симов с advance / eliminate |
| Strategies | points-optimal, qualify-rank, analyst, fusion (разные веса источников) |
| Player coverage | 8709/8709 матчей с player details (100%, цель ≥80%) |
