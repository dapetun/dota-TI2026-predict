# TI 2026 Swiss Predictor

Открытый ML-проект для прогноза групповой стадии The International 2026 (Dota 2).

## Возможности

| Реализовано | В планах |
|---|---|
| ETL OpenDota matchlists + match details | Полное покрытие всех details |
| Team Elo / form / H2H + player rolling stats | Сыгранность (co-play) |
| Walk-forward + Leave-One-TI-Out | Парные прогнозы XGBoost в UI |
| Статичный UI (GitHub Pages) | Draft / hero embeddings |

Текущая модель: **XGBoost** (team + player признаки).  
Разработка ведётся в ветке `prod`, стабильное — в `main`.

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

## Обучение модели

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# При необходимости догрузить списки матчей:
python scripts/download_data.py --list-only

# Догрузить детали матчей (игроки), resume-friendly:
python scripts/download_details.py
# или ограничить объём: python scripts/download_details.py --max 300

# ETL → features → XGBoost → метрики
python scripts/train.py

# Тесты
pytest -q
```

## Данные

1. Списки матчей крупных турниров (TI10–TI14 + majors 2026).
2. Словарь команд (`team_id_map.json`).
3. Ростеры TI 2026 — `src/ti2026/teams.py`.
4. Признаки матча — Elo до игры, форма, H2H, вес турнира (без утечки будущего).
5. Детали игроков пока покрывают лишь часть матчей.

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
  ti2026/
scripts/train.py
scripts/export_web_data.py
```

## Валидация

- **Walk-forward** — расширяющееся временное окно.
- **Leave-One-TI-Out** — обучение на матчах до TI_k, тест на TI_k.

Веса: `exp(-age / half_life_90d) * tier_weight`.

Подробнее: [docs/RESEARCH.md](docs/RESEARCH.md), [docs/FEATURES.md](docs/FEATURES.md), [docs/RESULTS.md](docs/RESULTS.md).

## Лицензия

Открытый проект. Данные — OpenDota (соблюдайте их ToS).
