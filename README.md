# TI 2026 Swiss Predictor / Прогноз Swiss TI 2026

**English** · **Русский**

Open-source ML project for The International 2026 **group stage** (Dota 2 Swiss) predictions.  
Открытый ML-проект для прогноза **групповой стадии** The International 2026 (швейцарская сетка Dota 2).

**Official TI dates / Официальные даты** ([tirules](https://www.dota2.com/esports/ti15/tirules)): Group Stage **13–16 Aug** (Online); Main Event **20–23 Aug** (Shanghai).

> UX inspiration (not a copy): [battlepass.ru/ti2026/predictions](https://battlepass.ru/ti2026/predictions). Official rules: [dota2.com tirules](https://www.dota2.com/esports/ti15/tirules).

## Branches / Ветки

| Branch | Purpose / Назначение |
|---|---|
| `prod` | Active development / Активная разработка |
| `main` | Stable Pages snapshot / Стабильный снимок для Pages |

## Web UI / Веб-интерфейс

Static site in `docs/` → GitHub Pages: https://dapetun.github.io/dota-TI2026-predict/

```powershell
python scripts/export_web_data.py
cd docs
python -m http.server 8080
```

Open http://localhost:8080 (`file://` will not work — `fetch` needs a server).

**UI locales / Языки UI:** `ru` / `en` / `de` / `fr` / `pt` / `es` (flagged switcher in the sidebar, saved in `localStorage`).  
**Rules / Правила:** section `#rules` on the same page ([docs/index.html](docs/index.html)); [docs/rules.html](docs/rules.html) redirects there. Informal paraphrase of [tirules](https://www.dota2.com/esports/ti15/tirules) + disclaimer (not a Valve document).

**Board strategies / Стратегии доски:** pick Model / Analysts / Mixed cards, then a friendly variant. Mixed mode has presets + optional weight sliders (snap to nearest **precomputed** fusion scenario).  
Смешанный режим: пресеты + опциональные ползунки (ближайший **заранее посчитанный** сценарий; непрерывный пересчёт без нового экспорта пока нельзя).

**Monte Carlo default / Дефолт симуляций:** `50_000` Swiss sims (`configs/settings.yaml` / `src.config.DEFAULT_N_SIMULATIONS`). Re-run `export_web_data.py` to refresh `predictions.json`.

**License on site / Лицензия на сайте:** MIT · Copyright © 2026 dapetun (footer links to [LICENSE](LICENSE)).

### Обновление перед / во время GS (SoT)

Group Stage **13–16 Aug**. Во время групп — свежий export, не retrain.

Расписание / результаты серий: `data/ti2026_swiss_results.json`  
(`phase`, `series[]` с `round` / `team_a` / `team_b` / `status` / опц. `winner`).  
Export фиксирует известные пары в Swiss MC (R1 и дальше), несыгранные серии сэмплируются из win matrix.

```powershell
# при новых парах / результатах — правим data/ti2026_swiss_results.json
python scripts/fetch_market_priors.py
python scripts/fetch_battlepass_experts.py
python scripts/export_web_data.py
# commit docs/data + push prod/main (по approve)
```

Проверка без записи: `python scripts/fetch_market_priors.py --dry-run`  
Sensitivity рынка: `python scripts/fetch_market_priors.py --sensitivity`  
После GS: `python scripts/gs_postmortem.py` → `outputs/gs_postmortem.md` → appendix в RESULTS.  
`train_compare` без новых details / смены фич **не** гонять mid-GS.

---

## Русский (подробнее)

Открытый ML-проект для прогноза групповой стадии The International 2026 (Dota 2).

### Возможности

| Реализовано (v0.3) | В планах |
|---|---|
| ETL OpenDota: **65 лиг**, ~8.7k карт | Playoff bracket |
| Team Elo + Glicko-2 ±uncertainty + player/chemistry | Draft MC |
| XGB + CatBoost blend → Swiss MC + heatmap | Crowd board |
| Points-optimal + multi-source fusion | Live retrain |
| Anonymous market prior + ranking/expert history | |
| Статичный UI (GitHub Pages) | |

Текущая модель: **CatBoost pairwise** (production `xgb=0`/`catboost=1` в `model_blend_v1.joblib`); иначе export требует `--allow-power-ranking`. Meta: **0.3.3** · LOO AUC **0.608** (TI12–14).

### Canonical pipeline

```text
scripts/download_data.py --list-only
scripts/download_details.py
scripts/train_compare.py
scripts/export_web_data.py
```

### Analyst & market sources / Аналитики и рынок

- **Analysts:** curated Sports.ru boards in `docs/data/analyst_picks.json` (~10 full compendium grids) + Hotspawn / other partial notes. Used for consensus board + fusion prior (vote shares per slot). Extra TI15 expert/power-rank soft boards live in `data/historical/expert_predictions.json`.
- **Market:** `data/ti2026_market_priors.json` — Swiss *slot* implied probs for fusion (research only). Refresh via `python scripts/fetch_market_priors.py` (Polymarket Gamma API, no key). Public books have no Swiss-slot markets yet → script derives slots from live **winner** Yes-prices (Bradley–Terry + Swiss MC), sets `is_real_market=true` / `derivation=derived_from_winner_odds`. Empty / `seed_from_ranking` still falls back to POWER_RANKINGS soft priors and export forces `market_weight=0`.
- **Fusion:** soft weights (model / analysts / market / ranking / expert history) are independent and need not sum to 1 — `fuse_slot_probabilities` renormalizes at blend time (battlepass-style). Production default `DEFAULT_MODEL_WEIGHT=0.65` with residual analysts when market is forced off. In-sample weight tune is diagnostic only. UI presets under «Смешанный».

### Player coverage

| | Value |
|---|---|
| Corpus | **8709** match_ids |
| With players | **8709** |
| Coverage | **100%** (target ≥80% ✓) |
| Missing | **0** — details downloaded for full corpus |

### Data / Данные

1. Matchlists TI10–14 + majors/quals (`data/league_candidates.json`).
2. Details shards: `data/raw/details_shards/…`
3. Rosters: `src/ti2026/teams.py`, `data/ti2026_rosters.json`
4. Analyst picks: `docs/data/analyst_picks.json`
5. Market priors: `data/ti2026_market_priors.json`
6. Historical experts: `data/historical/expert_predictions.json`

**Disclaimer:** research forecast with high uncertainty — not betting / financial advice. Market prior is anonymous research only; author does not endorse bookmakers.

### License / Лицензия

[MIT](LICENSE). Data — OpenDota (follow their ToS).

---

## English (details)

Open ML project forecasting TI 2026 group-stage Swiss outcomes and compendium slots.

**Pipeline:** matchlists → details → Elo/Glicko/player features → CatBoost (production) → ~50k Swiss Monte Carlo → slot/qualify probs → points-optimal / fusion boards → `docs/data/predictions.json`.

**Validation:** walk-forward + Leave-One-TI-Out (AUC / log-loss). See [docs/RESULTS.md](docs/RESULTS.md), [docs/FEATURES.md](docs/FEATURES.md).

**Training (short):**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py --list-only
python scripts/download_details.py --source explorer --rate 1.0 --batch 25
python scripts/train_compare.py
python scripts/export_web_data.py
pytest -q
```

More: [docs/ROADMAP_v03.md](docs/ROADMAP_v03.md).
