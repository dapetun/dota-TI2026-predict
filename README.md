# TI 2026 Swiss Predictor

Открытый ML-проект для прогноза групповой стадии The International 2026 (Dota 2).

**Официальные даты TI** ([tirules](https://www.dota2.com/esports/ti15/tirules)): Group Stage **13–16 Aug** (Online); Main Event **20–23 Aug** (Shanghai).

## Ветки

| Ветка | Назначение |
|---|---|
| `prod` | Активная разработка / эксперименты (эта ветка) |
| `main` | Стабильный снимок для Pages / тега релиза |

Не путать: свежие фичи и audit-фиксы идут в **`prod`**, затем cherry-pick / merge в `main` после стабилизации.

## Возможности

| Реализовано (v0.3) | В планах (stubs / ROADMAP) |
|---|---|
| ETL OpenDota: **65 лиг**, ~8.7k карт | Playoff bracket (`playoff_stub`) |
| Team Elo + **Glicko-2** ±uncertainty + player + chemistry | Draft / hero embeddings |
| Tier weights: ti/major/qual/online | Live auto-retrain |
| XGBoost + CatBoost + LOO-tuned blend | Crowd board / drag-drop |
| Pairwise → Swiss MC + slot heatmap 16×6 | Polymarket (`market_slot_prior_stub`) |
| Home LAN (CN Shanghai) + patch 7.41 meta | |
| Points-optimal board + analyst fusion | |
| Статичный UI (GitHub Pages) + CSP meta | |
| CI: pytest + ruff (E9/F) на push/PR | |
| Docker: `Dockerfile` (python 3.12 / pytest) | |

Текущая модель: **blend pairwise** (XGB + CatBoost, LOO weights) если есть `outputs/model_blend_v1.joblib`; иначе export требует `--allow-power-ranking`.  
Meta version: **0.3.0-prod**. См. [SECURITY.md](SECURITY.md).

## Canonical pipeline

```text
scripts/download_data.py --list-only
scripts/download_details.py          # shards via /explorer (fallback; /matches часто hang)
scripts/train_compare.py             # XGB + CatBoost + blend
scripts/build_rosters.py             # optional
scripts/export_web_data.py           # JSON для UI; по умолчанию --require-blend
```

Устаревшее вынесено в `legacy/` (pickle / LightGBM / STRATZ). Корневые stubs (`main.py`, `scripts/train_v2.py`, …) только указывают на legacy.

## Веб-интерфейс

Статика в `docs/` — для **GitHub Pages**: https://dapetun.github.io/dota-TI2026-predict/

### Локально

```powershell
python scripts/export_web_data.py --allow-power-ranking   # только если нет blend
# или (после train_compare):
python scripts/export_web_data.py
cd docs
python -m http.server 8080
```

Открой http://localhost:8080

> Через `file://` не работает: браузер блокирует `fetch` к JSON.

UI: Swiss-доска (model / qualify-rank / analyst / fusion), сила **μ ± σ**, heatmap слотов, E[очки] компендиума, chip «N/{n_analysts}» из JSON Sports.ru.  
Если meta.model = power ranking — явный **banner fallback**.  
Disclaimer: исследовательский прогноз с неопределённостью — **не для ставок**.  
Fonts: Google Fonts (Instrument Sans + Syne); CSP meta допускает fonts.googleapis.com / fonts.gstatic.com.

### Export flags

| Флаг | По умолчанию | Смысл |
|---|---|---|
| `--require-blend` / `--no-require-blend` | require=True | Без blend → exit ≠ 0 |
| `--allow-power-ranking` | off | Явный fallback на power ranking |
| `--min-player-coverage 0.5` | none | Fail если coverage ниже порога |
| `--n-simulations` | из `configs/settings.yaml` (20000) | MC sims |

## Обучение модели

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# опционально жёсткие pins:
# pip install -r requirements.lock.txt

# Discover / review leagues (optional)
python scripts/discover_leagues.py --min-matches 50 --max-probe 200

# Matchlists (реестр: src/data_collection/tournaments.py)
python scripts/download_data.py --list-only

# Details (resume-friendly shards + monolith shim)
python scripts/download_details.py

# XGB + CatBoost + blend
python scripts/train_compare.py

# Rosters nick→account_id
python scripts/build_rosters.py

# Экспорт JSON для UI (fail-hard без blend)
python scripts/export_web_data.py

# Тесты / CI локально
pytest -q

# Docker
docker build -t ti2026-predict .
docker run --rm ti2026-predict
```

## Конфиг (truth)

- **Код** — канон для half-life: `RATING_HALF_LIFE_DAYS = 210` в `src/features/sample_weights.py`.
- **`configs/settings.yaml`** — опциональные overrides через `src.config` (`safe_load`); значения синхронизированы (210d, 20000 sims).
- Имена команд — SoT: `src/ti2026/teams.py` (`normalize_team_name`).

## Данные

1. Списки матчей: TI10–14 + majors 2023–2026 + quals + mid-tier online (`data/league_candidates.json`).
2. Словарь OpenDota `team_id` → имя: curated `data/team_id_map.json` (в repo); локальный override — `data/raw/team_id_map.json` (gitignored). Используется только если в matchlist нет имён команд.
3. Ростеры TI 2026 — `src/ti2026/teams.py`, `data/ti2026_rosters.json` (nick→account curated; там же `open_dota_team_id`).
4. Признаки: Elo/Glicko-2, form (~40d), sample half-life (~210d), patch≥7.41 mult 1.25, player, chemistry, stitch Jaccard≥0.6.
5. Details: `data/raw/details_shards/<tournament>/<match_id>.json` (новые; source по умолчанию OpenDota `/explorer`, т.к. `/matches/{id}` часто hang); legacy monolith `match_details.json` читается shim-ом. Stratz deprecated stub. `OPENDOTA_API_KEY` опционален.
6. Analyst picks — `docs/data/analyst_picks.json` (Sports.ru).

**Дисклеймер:** модель даёт вероятности с высокой неопределённостью; не финансовый совет и не инструмент для ставок.

## Архитектура

```
data/raw          ← сырые OpenDota JSON (+ details_shards/)
data/processed    ← canonical_matches.csv
data/features     ← match_features_xgb.csv
legacy/           ← устаревшие entrypoints (не для прода)
src/
  config.py         ← thin YAML loader
  data_collection/  ← tournaments.py = SoT реестра лиг; OpenDota only
  features/         ← match / player / chemistry / stitching / ratings
  models/
  ti2026/           ← pairwise, compendium, fusion, multisource
  simulation/       ← Swiss MC + playoff_stub
scripts/train_compare.py
scripts/export_web_data.py
scripts/discover_leagues.py
.github/workflows/ci.yml
Dockerfile
```

## Валидация

- **Walk-forward** — расширяющееся временное окно.
- **Leave-One-TI-Out** — тест только на TI_k; train = всё до старта TI_k.
- Blend weights по LOO log-loss; isotonic — offline eval (не в production blend).

Веса: `0.5**(age/210d) * tier_weight * patch_mult` (+ cold-start down-weight).  
`patch_mult = 1.25` для матчей после `PATCH_741_START_TS` (см. `compute_sample_weights`).  
Blend joblib: SHA256 в `model_compare.json` / sidecar `*.joblib.sha256`; load fail on mismatch.

Подробнее: [docs/ROADMAP_v03.md](docs/ROADMAP_v03.md), [docs/FEATURES.md](docs/FEATURES.md), [docs/RESULTS.md](docs/RESULTS.md).

## Лицензия

[MIT](LICENSE). Данные — OpenDota (соблюдайте их ToS).
