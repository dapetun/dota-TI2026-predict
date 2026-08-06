# Дорожная карта TI 2026 — v0.3

Дата: 2026-08-06 · после релиза **v0.2.0**

## Контекст

Сейчас (v0.2): **12 турниров** (TI10–14 + majors 2026), **~2134** карт, 63 признака (team+player+chemistry), blend pairwise → Swiss MC, points-optimal board + analyst consensus.

[battlepass.ru](https://battlepass.ru/ti2026/predictions): **9477** карт · **93** турнира · сила с **±uncertainty** · рынок · home LAN · патч 7.41 · multi-source слайдеры · склейка брендов по ростеру.

Главный разрыв: **объём/разнообразие данных** + **явная неопределённость силы**, не «ещё одна бустинг-модель».

---

## Как сейчас работают player stats и chemistry

### Индивидуальная статистика игроков — работает

Пайплайн: OpenDota match details → `src/features/player_features.py` → матрица → обучение и pairwise.

1. По каждому `account_id` — rolling ≤20 карт: KDA, GPM, XPM, HD/TD per min, WR, LAN WR.
2. Перед матчем пятёрка → mean-вектор стороны (`r_pl_*` / `d_pl_*` / `diff_*`) + `r_pl_games` + `has_player_stats`.
3. История обновляется **после** эмиссии строки (без leakage).
4. В Swiss pairwise (`src/ti2026/pairwise.py`): lineup из `data/ti2026_rosters.json` или last lineup; состояние — `replay_player_states`.

Ограничения: только матчи из 12 лиг; нет Bayesian shrinkage к лиге; ники не склеены вручную 1:1 с account_id; нет continuity при ребрендинге (кроме alias Virtus.pro→Yandex).

### Сыгранность (chemistry) — работает

Модуль: `src/features/chemistry_features.py` (13 колонок).

1. Пары в пятёрке: `chem_mean` / `chem_min` / `chem_90d`.
2. Jaccard состава vs прошлый матч того же `team_id`.
3. `has_chemistry`; update после матча.
4. Pairwise: финальный `ChemistryState` на `max(start_time)`.
5. `lan_only=True` есть, но **не** в default `train_compare`.

Ограничения: узкий корпус; нет joint WR пар; нет «days since full-5 together».

---

## Идеи с battlepass.ru

| Идея | У нас | Приоритет |
|---|---|---|
| 9k+ карт / 93 турнира | 2k / 12 | **P0** |
| Quals ×0.5 vs majors | только TI/major | **P0** |
| Сила ± SE | только `r_gp` | **P0** |
| Half-life 240d / form 40d | единый 90d | **P1** |
| Склейка брендов по игрокам | ручные aliases | **P1** |
| Home LAN +30 Elo (Shanghai) | нет | **P1** |
| Патч 7.41 rating | нет | **P1** |
| Рынок Polymarket | нет | **P2** |
| Слайдеры источников | 4 доски + fusion | **P1** |
| Drag-drop сетка | view only | **P2** |
| Heatmap P(slot) 16×6 | JSON есть | **P1** |

---

## Фаза A — Расширить корпус (P0, 3–7 дней)

### Политика (гибрид)

1. **Train corpus**: decent турниры ~24 мес / с TI13: majors, DPC/региональные, TI quals, крупные online.
2. **Tier weights**: `ti=2.0`, `major=1.5`, `qual/dpc=0.75`, `online=0.5`.
3. **Validation**: Leave-One-TI-Out без изменений — тест только TI_k; train = всё до старта TI_k.
4. Ablation: majors-only vs expanded → RESULTS.

### Инфра

- Единый source of truth: `src/data_collection/tournaments.py`.
- `scripts/download_data.py` импортирует реестр (сейчас дублирует).
- `scripts/discover_leagues.py`: OpenDota leagues + фильтры (tier, n_matches, даты).
- Curated `data/league_candidates.json` → review → `TOURNAMENTS`.
- Цель: **≥5–8k** карт, details coverage ≥80%.

### Download

- Приоритет: TI2026 teams → quals 2026 → majors → rest.
- Метрики: `#leagues`, `#maps`, maps-per-TI2026-team.

### Готовность A

- Resilience/HULIGANI/GL ≥30–50 карт (или задокументированный cold-start).
- LOO AUC не падает >0.01 vs v0.2; E[points] не хуже без ручного тюнинга.

---

## Фаза B — Uncertainty / cold-start (P0, 3–5 дней)

### Быстрый win (признаки)

- `min_gp`, `1/sqrt(gp+1)`, Empirical Bayes Elo-shrink к prior: `w=gp/(gp+k)`.
- `r_opp_avg_elo` — сила календаря соперников.

### Средний win (рейтинг)

- Подключить **Glicko-2** из `src/features/rating_systems.py` (сейчас legacy).
- В MC: сэмпл силы из `N(μ, RD²)` или расширенный P(win).
- UI: `сила μ ± σ`.

### Weights

- Раздельные half-life: rating 180–240d, form 30–45d.
- Down-weight матчи, где обе стороны `gp < threshold`.

### Готовность B

- Корреляция `|error|` vs `1/sqrt(gp)` на hold-out.
- Экстремальные слоты (Vici 0-4 / LGD advance) стабилизируются без ручной правки.

---

## Фаза C — Player + chemistry 2.0 (P1, 4–6 дней)

1. Ручной nick→account_id для 16 команд в `ti2026_rosters.json`.
2. Team stitching: Jaccard игроков ≥0.6 → общая история.
3. Chemistry: joint WR пар; `chem_lan_only` ablation; roster stability 60d.
4. Player: patch-window WR; `pl_uncertainty=1/sqrt(pl_games+1)`.

---

## Фаза D — Multi-source (P1, 5–8 дней)

Источники: match rating · short form · patch 7.41 · home LAN · analyst prior · (optional) market.

- Weighted fusion с UI-слайдерами (precomputed scenarios в JSON или client recompute).
- Home LAN: +Δ Elo для CN в Shanghai.
- Market — P2, если данные доступны.

---

## Фаза E — UI / продукт (P1–P2)

- Heatmap 16×6 P(slot).
- `μ ± σ` силы.
- Drag-drop + live E[очки] (позже).
- Секция методики на русском.

---

## Фаза F — Валидация и v0.3.0

1. LOO AUC ≥ 0.60, LL ≤ 0.79; TI14 AUC ≥ 0.52 (stretch).
2. Ablation table в RESULTS.
3. Tag **v0.3.0** после стабилизации `prod` → `main`.

---

## Порядок, если время до драфта ограничено

1. A — данные  
2. B1 — shrinkage/gp features  
3. C1 — явные ростеры  
4. D — home LAN + patch  
5. B2 — Glicko RD  
6. E — heatmap + слайдеры  
7. Market / drag-drop — по остатку времени

## Не делать сразу

- LightGBM без gain на LOO  
- Draft embeddings до расширения корпуса  
- Слепое копирование доски battlepass  
- Per-TI blend weights на 4 фолдах  
