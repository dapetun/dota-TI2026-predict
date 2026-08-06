# TI 2026 Swiss Predictor

Открытый ML-проект для прогноза групповой стадии The International 2026 (Dota 2).

## Возможности

| Реализовано | В планах |
|---|---|
| ETL OpenDota matchlists + match details (100% coverage) | Playoff bracket sim |
| Team Elo / form / H2H + player + chemistry (63 feat) | Draft / hero embeddings |
| XGBoost + CatBoost + LOO-tuned blend | Live auto-retrain |
| Blend pairwise → Swiss Monte Carlo UI | Crowd board |
| Points-optimal compendium board + analyst consensus | |
| Статичный UI (GitHub Pages) | |

Текущая модель: **blend pairwise** (XGB 0.25 + CatBoost 0.75, isotonic calibration).  
Разработка в ветке `prod`, стабильное — в `main`.

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

UI: Swiss-доска (переключатель model / qualify-rank / analyst consensus), E[очки] компендиума, chip «N/11» по инфлюенсерам Sports.ru.

## Обучение модели

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# При необходимости догрузить списки матчей:
python scripts/download_data.py --list-only

# Догрузить детали матчей (игроки), resume-friendly:
python scripts/download_details.py

# ETL → features → XGBoost → метрики
python scripts/train.py

# XGB + CatBoost + blend + сравнение метрик
python scripts/train_compare.py

# Экспорт JSON для UI
python scripts/export_web_data.py

# Тесты
pytest -q
```

## Данные

1. Списки матчей крупных турниров (TI10–TI14 + majors 2026).
2. Словарь команд (`team_id_map.json`).
3. Ростеры TI 2026 — `src/ti2026/teams.py`, `data/ti2026_rosters.json`.
4. Признаки матча — Elo, форма, H2H, player rolling, chemistry (без утечки).
5. Analyst picks — `docs/data/analyst_picks.json` (Sports.ru).

## Архитектура

```
data/raw          ← сырые OpenDota JSON
data/processed    ← canonical_matches.csv
data/features     ← match_features_xgb.csv
src/
  data_collection/
  features/
  models/
  evaluation/
  simulation/
  ti2026/         ← pairwise, compendium, analyst consensus, fusion
scripts/train_compare.py
scripts/export_web_data.py
```

## Валидация

- **Walk-forward** — расширяющееся временное окно.
- **Leave-One-TI-Out** — обучение на матчах до TI_k, тест на TI_k.
- Blend weights подбираются по LOO log-loss; isotonic calibration на pooled LOO.

Веса: `exp(-age / half_life_90d) * tier_weight`.

Подробнее: [docs/RESEARCH.md](docs/RESEARCH.md), [docs/FEATURES.md](docs/FEATURES.md), [docs/RESULTS.md](docs/RESULTS.md).

## Лицензия

Открытый проект. Данные — OpenDota (соблюдайте их ToS).
