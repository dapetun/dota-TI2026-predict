# Результаты обучения (team + player + chemistry)

Дата: 2026-08-06 · ветка `prod` · coverage **100%** (2134/2134)  
Признаков: **63** · details: 2234 cached  
Pairwise Swiss: team + **player/chem snapshot** + `data/ti2026_rosters.json`  
Blend: LOO weights + **isotonic calibration (LOO eval only, не в production bundle)**

## Сводка (Leave-One-TI / Walk-forward)

| Модель | LOO AUC | LOO LL | WF AUC | WF LL |
|---|---|---|---|---|
| XGBoost | 0.594 | 0.869 | 0.568 | 0.846 |
| CatBoost | 0.594 | 0.801 | 0.585 | 0.776 |
| Blend (LOO-tuned) | **0.598** | **0.797** | 0.584 | **0.772** |

Веса blend по LOO grid: **XGB 0.25 / CatBoost 0.75**. Isotonic на pooled LOO — только для offline-оценки (`train_compare`: `isotonic_loo_evaluated`); pairwise export без calibrator.

## Leave-One-TI (blend)

| Held-out | n | LL | AUC |
|---|---|---|---|
| TI11 | 213 | 0.821 | 0.561 |
| TI12 | 128 | 0.733 | 0.611 |
| TI13 | 99 | 0.686 | **0.729** |
| TI14 | 121 | 0.948 | **0.490** |

### TI14 (слабый fold)

- AUC ~0.49 — ниже случайного на hold-out; вероятные факторы: patch/meta drift 2024, разрыв составов vs TI-стиль данных, малый n=121.
- Эксперимент **LAN-only chemistry** (`build_chemistry_features(..., lan_only=True)`) — для сравнения в `train_compare` (опционально).
- Per-TI blend weights не внедрены: 4 TI — высокий риск overfit.

## UI

http://localhost:8080 — Swiss на **blend pairwise**. Переключатель доски: points-optimal / qualify-rank / analyst consensus / fusion.  
Hero: E[очки], консенсус Sports.ru, fusion. Chip **N/10** на карточках (≥5 аналитиков).

## Компендиум: три стратегии + fusion

Battlepass.ru ([TI2026 predictions](https://battlepass.ru/ti2026/predictions)) — суперлинейная таблица Valve (16/16 → 12 000).

| Стратегия | E[верных слотов] | E[очки Valve] |
|---|---|---|
| Qualify-rank | 7.40 | **2 305** |
| Points-optimal (model) | 7.55 | **2 412** |
| Analyst consensus (Sports.ru 10 сеток) | см. export | см. `predictions.json` |
| Fusion model+analyst | см. export | см. `meta.fusion_expected_points` |

Отличия model points-optimal vs qualify-rank: OG/HULIGANI (eliminate ↔ 1–4).

### Расхождения model vs analyst consensus

| Тема | Инфлюенсеры (консенсус) | Модель (points-optimal) |
|---|---|---|
| 4–0 | Vision (6/10) | Aurora |
| 4–1 | Yandex + BetBoom | BetBoom + Vision |
| Проход | Spirit, 1w, Liquid | Spirit, Falcons, LGD, Xtreme, 1w |
| Выбывание | OG, LGD, Nigma | OG, Yandex, Liquid, Nigma, GamerLegion |
| 0–4 | Resilience (7/10) | Vici |

## Дальше

Playoff bracket, draft embeddings, live pipeline.

CLI: `train_compare.py` · `export_web_data.py` · `build_rosters.py`
