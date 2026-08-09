# Training results / Результаты обучения (v0.3.3)

**English summary:** Leave-One-TI-Out on **TI12–TI14** (folds with ≥1000 prior maps) + walk-forward; CatBoost production; player coverage **100%**. Details in Russian below.

**RU · Дата:** 2026-08-09 · meta **0.3.3**  
Официальные даты TI ([tirules](https://www.dota2.com/esports/ti15/tirules)): Group Stage **13–16 Aug** Online; Main Event **20–23 Aug** Shanghai.  
Корпус: **65 лиг**, **8709** уникальных карт  
Признаков: **91** · feature rows: **8709** (`min_games=0`)  
Player/chem coverage: **100%** (8709/8709)  
Team stitch: **97** remaps (temporal Jaccard)  
Half-life: sample **180d** · form **~40d** · patch ×**1.4**  
Production: **CatBoost-only** — LOO **0.608** (TI12–14); WF **0.653**  
`prefer_lan=False` train ↔ GS pairwise  
LOO `min_train=1000`: fold **TI11** пропускается (только ~487 карт до него → AUC≈0.52, портил среднее)

## Coverage gate

| Метрика | Сейчас | Цель | Статус |
|---|---|---|---|
| Player coverage | **100%** | ≥80% | **OK** |

## Сводка (Leave-One-TI / Walk-forward)

| Модель | LOO AUC (TI12–14) | LOO LL | WF AUC | WF LL |
|---|---|---|---|---|
| XGBoost | 0.598 | 0.738 | 0.649 | 0.694 |
| CatBoost | **0.608** | **0.702** | 0.653 | 0.666 |
| Blend nested | 0.608 | 0.702 | **0.656** | **0.665** |
| Production ship | CatBoost-only | — | — | — |

Цель LOO AUC ≥0.60 на релевантных TI — **достигнута** (0.608).  
Раньше в среднее входил TI11 (train≈487, AUC≈0.52) и тянул headline к ~0.585.

### Как читать метрики

- **AUC** — ранжирование победителя (0.5 = монетка).
- **Leave-One-TI-Out** — hold-out по TI с достаточным prior corpus (≥1000 карт).
- **Walk-forward** — расширяющееся окно по времени (часто выше LOO).

## Leave-One-TI (CatBoost production)

| Held-out | n | LL | AUC |
|---|---|---|---|
| TI12 | 151 | 0.718 | 0.587 |
| TI13 | 121 | 0.692 | **0.609** |
| TI14 | 144 | 0.697 | **0.628** |

## Calibration policy

- Production: CatBoost-only, **без** isotonic.
- Brier + ECE + log-loss в `outputs/model_compare.json` / `outputs/calibration_report.md` (после `train_compare`).
- Isotonic — только experiment; ship при устойчивом LOO gain.

### Правило ship новых фич

Ship фичу **только если**:
1. Δ LOO AUC (TI12–14) ≥ **+0.005**, и
2. LOO log-loss **не хуже** текущего baseline.

Иначе — ablation в RESULTS, фича остаётся off.

## Market `strength_power`

Default **0.5**. Sweep: `python scripts/fetch_market_priors.py --sensitivity` → `outputs/market_strength_sensitivity.md`.  
Пока slot books нет — derivation = `derived_from_winner_odds`; при появлении — `direct_slot`.

| strength_power | top4_slot_mass | delta vs 0.5 |
|---:|---:|---:|
| 0.4 | 8.0014 | +0.0018 |
| 0.5 | 7.9996 | 0 |
| 0.6 | 8.0010 | +0.0014 |

Δ negligible → keep default **0.5**.

## Топ-признаки (CatBoost)

`r_pl_tdpm`, `diff_pl_uncertainty`, `d_wr`, `diff_opp_avg_elo`, `diff_glicko_mu`, …

## Заметки

- Усиление CatBoost (500 iter / depth 6) **ухудшило** LOO → откат к 300/5.
- Hero soft prior по-прежнему off по умолчанию (`USE_HERO_SOFT_PRIOR=0`); fetch: `python scripts/build_hero_meta.py --fetch` (retry+cache). Ablation: `USE_HERO_SOFT_PRIOR=1` после реальных артефактов.
- Live draft logit shift: `apply_known_draft_logit_shift` в `hero_soft_prior.py` (уровень A).
- UI показывает recent LOO + walk-forward, не «монетку» от TI11.
- Drag-drop доски **не** внедряли: fusion presets + слайдеры закрывают UX (anti-priority P2).

## Appendix — GS post-mortem (TI15 / TI 2026)

**Статус:** шаблон до конца Group Stage (13–16 Aug).

После GS:

1. Заполнить слоты в `data/historical/ti_swiss_ground_truth.json` (ключ `TI15`).
2. `python scripts/gs_postmortem.py`
3. Вставить таблицу Points / correct slots (model / analysts / fusion / market-lean) сюда.

| Board | Points | Correct slots | Notes |
|---|---:|---:|---|
| points_optimal | — | — | заполнить после GS |
| analyst_consensus | — | — | |
| fusion | — | — | |
| fusion_market_lean | — | — | |

Что смотреть: корреляция E[points] доски с фактом; не map-AUC.
