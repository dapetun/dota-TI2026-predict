# Результаты обучения XGBoost (team + player)

Дата прогона: 2026-08-05  
Ветка: `prod`  
Сайт (локально): `docs/` → http://localhost:8080 · версия данных `0.2.0-prod`

## Данные

- **2134** матча из 12 турниров (TI10–TI14 + majors 2026)
- После фильтра `min_games>=5`: **1734** строк
- Player details: **448** матчей с игроками (**21%** покрытия), 5480 player-rows, 235 account_id
- Признаков: **50** (25 team + 25 player)

Догрузка details: `python scripts/download_details.py` (resume). Покрытие растёт фоном.

## Метрики (последний экспорт в UI)

| Метрика | Значение |
|---|---|
| Walk-forward AUC (avg) | **0.547** |
| Leave-One-TI AUC (avg) | **0.579** |
| Walk-forward LogLoss | 0.892 |
| LOO-TI LogLoss | 0.895 |

### Leave-One-TI-Out

| Held-out TI | n | LogLoss | AUC |
|---|---|---|---|
| TI11 | 213 | 0.877 | 0.560 |
| TI12 | 128 | 0.807 | 0.610 |
| TI13 | 99 | 0.722 | 0.708 |
| TI14 | 121 | 1.172 | 0.438 |

Топ важности (player уже в лидерах): `r_pl_lan_wr`, `elo_prob`, `diff_pl_lan_wr`, `diff_wr`, `diff_pl_xpm`.

## Сравнение с team-only

| Метрика | Team-only | Team+player (~21% coverage) |
|---|---|---|
| LOO-TI AUC | ~0.580 | **0.579** |
| TI14 AUC | ~0.430 | **0.438** |

При низком покрытии details прирост скромный; цель coverage ≥60–70%, затем переобучение и merge в `main` как `v0.2.0`.

## UI

Доска Swiss — power ranking + Monte Carlo (не pairwise XGBoost). В метаданных сайта: coverage players и версия `0.2.0-prod`. Имена: BoomBoys, Iron Wing, TEAM VISION.

Артефакты локально: `outputs/xgb_v1_metrics.json`, `calibration.png`, `feature_importance.png`.
