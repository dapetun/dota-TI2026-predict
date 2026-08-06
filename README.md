# TI 2026 Swiss Predictor

Открытый ML-проект для прогноза групповой стадии The International 2026 (Dota 2).

**Официальные даты TI** ([tirules](https://www.dota2.com/esports/ti15/tirules)): Group Stage **13–16 Aug** (Online); Main Event **20–23 Aug** (Shanghai).

## Возможности

| Реализовано (v0.3) | В планах |
|---|---|
| ETL OpenDota: **65 лиг**, ~8.7k карт | Playoff bracket sim |
| Team Elo/Glicko ±uncertainty + player + chemistry | Draft / hero embeddings |
| Tier weights: ti/major/qual/online | Live auto-retrain |
| XGBoost + CatBoost + LOO-tuned blend | Crowd board / drag-drop |
| Pairwise → Swiss MC + slot heatmap 16×6 | Polymarket (hook stub) |
| Home LAN (CN Shanghai) + patch 7.41 meta | |
| Points-optimal board + analyst fusion | |
| Статичный UI (GitHub Pages) | |

Текущая модель: **blend pairwise** (XGB + CatBoost, LOO weights).  
Разработка в ветке `prod`, стабильное — в `main`. Meta version: **0.3.0-prod**.

## Веб-интерфейс

Статика в `docs/` — для **GitHub Pages**: https://dapetun.github.io/dota-TI2026-predict/

### Локально

```powershell
python scripts/export_web_data.py
cd docs
python -m http.server 8080
```

Открой http://localhost:8080

> Через `file://` не работает: браузер блокирует `fetch` к JSON.

UI: Swiss-доска (model / qualify-rank / analyst / fusion), сила **μ ± σ**, heatmap слотов, E[очки] компендиума, chip «N/11» Sports.ru.

## Обучение модели

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Discover / review leagues (optional)
python scripts/discover_leagues.py --min-matches 50 --max-probe 200

# Matchlists (реестр: src/data_collection/tournaments.py)
python scripts/download_data.py --list-only

# Details (resume-friendly, priority в DOWNLOAD_PRIORITY)
python scripts/download_details.py

# XGB + CatBoost + blend
python scripts/train_compare.py

# Rosters nick→account_id
python scripts/build_rosters.py

# Экспорт JSON для UI
python scripts/export_web_data.py

# Тесты
pytest -q
```

## Данные

1. Списки матчей: TI10–14 + majors 2023–2026 + quals + mid-tier online (`data/league_candidates.json`).
2. Словарь команд (`team_id_map.json`).
3. Ростеры TI 2026 — `src/ti2026/teams.py`, `data/ti2026_rosters.json` (nick→account curated).
4. Признаки: Elo/Glicko, form (~40d), sample half-life (~210d), player, chemistry, stitch Jaccard≥0.6.
5. Analyst picks — `docs/data/analyst_picks.json` (Sports.ru).

## Архитектура

```
data/raw          ← сырые OpenDota JSON
data/processed    ← canonical_matches.csv
data/features     ← match_features_xgb.csv
src/
  data_collection/  ← tournaments.py = SoT реестра лиг
  features/         ← match / player / chemistry / stitching / ratings
  models/
  ti2026/           ← pairwise, compendium, fusion, multisource
scripts/train_compare.py
scripts/export_web_data.py
scripts/discover_leagues.py
```

## Валидация

- **Walk-forward** — расширяющееся временное окно.
- **Leave-One-TI-Out** — тест только на TI_k; train = всё до старта TI_k.
- Blend weights по LOO log-loss; isotonic — offline eval.

Веса: `0.5**(age/210d) * tier_weight` (+ cold-start down-weight).

Подробнее: [docs/ROADMAP_v03.md](docs/ROADMAP_v03.md), [docs/FEATURES.md](docs/FEATURES.md), [docs/RESULTS.md](docs/RESULTS.md).

## Лицензия

[MIT](LICENSE). Данные — OpenDota (соблюдайте их ToS).
