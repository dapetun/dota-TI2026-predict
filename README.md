# TI 2026 Swiss Predictor

Открытый ML-проект для прогноза групповой стадии The International 2026 (Dota 2).

## Статус: v0.1.0 (Iteration 1)

Первая production-модель: **XGBoost** на team-level признаках.

| Есть | Пока нет |
|---|---|
| ETL OpenDota matchlists | Полные player details по всем матчам |
| Elo / form / H2H / tier features | Player model, сыгранность |
| Walk-forward + Leave-One-TI-Out | Ансамбли, Monte Carlo Swiss |
| Logloss / Brier / AUC / calibration | Draft / hero embeddings |

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Данные уже лежат в data/raw (*_matchlist.json).
# При необходимости догрузить списки:
python scripts/download_data.py --list-only

# Итерация 1: ETL → features → XGBoost → метрики
python scripts/run_iteration1.py

# Тесты
pytest -q
```

## Какие данные сейчас есть

Человеческим языком:

1. **Списки матчей** крупных турниров (TI10–TI14 + majors 2026) — кто играл, кто выиграл, когда, счёт.
2. **Словарь команд** (`team_id_map.json`) — OpenDota ID → имя.
3. **Ростеры TI 2026** — зашиты вручную в `src/ti2026/teams.py`.
4. **Признаки матча** — Elo до игры, форма, H2H, вес турнира (без утечки будущего).
5. **Детали игроков** — только ~100 матчей; для MVP не используются.

Этого достаточно, чтобы оценить силу команд перед TI и проверить модель на прошлых International.

## Архитектура

```
data/raw          ← сырые OpenDota JSON (не трогаем)
data/processed    ← canonical_matches.csv
data/features     ← match_features_xgb.csv
src/
  data_collection/  match_loader, tournaments, opendota/stratz clients
  features/         match_features, sample_weights, rating_systems
  models/           xgboost_model (+ legacy ensemble/training)
  evaluation/       metrics, calibration plots
  ti2026/           teams & Swiss config
scripts/run_iteration1.py
```

Позже сюда без переписывания добавятся Elo/Glicko как отдельные фичи, CatBoost/LightGBM, симуляция Swiss.

## Валидация

- **Walk-forward** — расширяющееся временное окно (без shuffle).
- **Leave-One-TI-Out** — обучение на матчах до TI_k, тест на TI_k.

Веса обучения: `exp(-age / half_life_90d) * tier_weight`.

Подробности: [docs/RESEARCH.md](docs/RESEARCH.md), признаки: [docs/FEATURES.md](docs/FEATURES.md).

## Лицензия

Открытый hobby-проект. Данные — OpenDota (соблюдайте их ToS).
