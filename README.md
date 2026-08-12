# Прогноз групповой стадии TI 2026

Открытый проект: вероятности исходов **швейцарской сетки** The International 2026 (Dota 2) и заполнение доски компендиума.

**Даты турнира** ([правила Valve](https://www.dota2.com/esports/ti15/tirules)): группы **13–16 августа** (онлайн); основной этап **20–23 августа** (Шанхай).

Сайт: https://dapetun.github.io/dota-TI2026-predict/

Ориентир по удобству интерфейса (не копия): [battlepass.ru/ti2026/predictions](https://battlepass.ru/ti2026/predictions).

---

# Русский

## Зачем это

Модель смотрит на прошлые матчи, считает шансы команд друг против друга, затем много раз «проигрывает» всю групповую стадию целиком. На выходе — вероятности слотов доски (4–0, 4–1, проход, вылет и т.д.) и готовые варианты заполнения.

Это **исследование**, не совет для ставок.

## Что уже есть (версия 0.3.3)

| Сделано | В планах |
|---|---|
| Данные OpenDota: **65 лиг**, ~8,7 тыс. карт | Сетка плей-офф |
| Рейтинги команд (Elo, Glicko-2) + форма игроков и «химия» состава | Моделирование драфтов |
| Обучение (CatBoost) → симуляции сетки → тепловая карта | Доска «голосов толпы» |
| Смешивание сигналов: модель, аналитики, рынок, рейтинги | Дообучение по ходу групп |
| Статичный сайт на GitHub Pages | |

Рабочая модель: **CatBoost** по парам команд. Качество на проверке «оставить один TI» (TI12–14): AUC **0,608**.

Покрытие игроками в корпусе: **8709 / 8709** карт (100%).

## Ветки

| Ветка | Зачем |
|---|---|
| `prod` | Текущая разработка |
| `main` | Стабильная версия для сайта (GitHub Pages) |

## Сайт

Файлы лежат в `docs/`. Локально:

```powershell
python scripts/export_web_data.py
cd docs
python -m http.server 8080
```

Откройте http://localhost:8080 (через `file://` не заработает — нужен простой сервер).

Языки интерфейса: русский, английский, немецкий, французский, португальский, испанский (переключатель в боковой панели).

Правила турнира — раздел `#rules` на той же странице; это вольный пересказ [официальных правил](https://www.dota2.com/esports/ti15/tirules), не документ Valve.

На доске можно выбрать стратегии: модель / аналитики / смешанный режим. В смешанном — готовые пресеты и ползунки весов (подставляется ближайший **заранее посчитанный** сценарий).

По умолчанию **50 000** прогонов сетки. Чтобы обновить цифры на сайте — снова запустите `export_web_data.py`.

## Как обновлять во время групп (13–16 августа)

Пока идут группы: обновляйте данные и пересчитывайте сайт. **Полностью переобучать модель не нужно**, если не появились новые подробные матчи в корпусе и не менялись признаки.

Известные пары и результаты серий пишите в `data/ti2026_swiss_results.json`  
(поля: этап, список серий с номером раунда, командами, статусом и при необходимости победителем).  
Симуляция **фиксирует** уже объявленные пары; исход ещё не сыгранных серий берётся из матрицы шансов модели.

```powershell
# при новых парах или результатах — правим data/ti2026_swiss_results.json
python scripts/fetch_market_priors.py
python scripts/fetch_battlepass_experts.py
python scripts/export_web_data.py
# затем коммит docs/data и push (по согласованию)
```

Проверка рынка без записи: `python scripts/fetch_market_priors.py --dry-run`  
Чувствительность рынка: `python scripts/fetch_market_priors.py --sensitivity`  
После групп: `python scripts/gs_postmortem.py` → отчёт в `outputs/gs_postmortem.md`.

## Полный цикл с нуля

```text
scripts/download_data.py --list-only
scripts/download_details.py
scripts/train_compare.py
scripts/export_web_data.py
```

Кратко по шагам:

1. Списки матчей по лигам  
2. Подробности матчей (игроки, герои)  
3. Признаки и обучение  
4. Пары 16×16 → симуляции швейцарки → доски → `docs/data/predictions.json`

## Откуда берутся «внешние» сигналы

- **Аналитики** — собранные доски Sports.ru и др. в `docs/data/analyst_picks.json`; доп. экспертные сетки — в `data/historical/expert_predictions.json`.  
- **Рынок** — `data/ti2026_market_priors.json`. Обновление: `python scripts/fetch_market_priors.py` (Polymarket, без ключа). Отдельных рынков на слоты швейцарки почти нет, поэтому скрипт **выводит** шансы слотов из котировок на победителя турнира. Если живых котировок нет — рынок в смешивании отключается.  
- **Смешивание** — веса модели, аналитиков, рынка, рейтинга и истории экспертов независимы (сумма не обязана быть 1; перед усреднением веса приводятся к норме). В проде у модели вес по умолчанию **0,65**. Подбор весов на тех же данных — только диагностика.

## Данные в репозитории

1. Списки лиг: `data/league_candidates.json`  
2. Подробности матчей: `data/raw/details_shards/…` (локально, в git обычно не кладём)  
3. Составы: `src/ti2026/teams.py`, `data/ti2026_rosters.json`  
4. Доски аналитиков: `docs/data/analyst_picks.json`  
5. Рыночные вероятности: `data/ti2026_market_priors.json`  
6. Исторические эксперты: `data/historical/expert_predictions.json`  
7. Живые пары / результаты групп: `data/ti2026_swiss_results.json`

Подробнее про признаки и метрики: [docs/FEATURES.md](docs/FEATURES.md), [docs/RESULTS.md](docs/RESULTS.md), [docs/ROADMAP_v03.md](docs/ROADMAP_v03.md).

## Лицензия

[MIT](LICENSE). Данные — OpenDota (соблюдайте их условия использования).

---

# English

## What this is

An open project that forecasts The International 2026 **group stage** (Dota 2 Swiss) and fills a compendium-style prediction board.

**Official dates** ([tirules](https://www.dota2.com/esports/ti15/tirules)): Group Stage **13–16 Aug** (online); Main Event **20–23 Aug** (Shanghai).

Live site: https://dapetun.github.io/dota-TI2026-predict/

UX reference (not a copy): [battlepass.ru/ti2026/predictions](https://battlepass.ru/ti2026/predictions).

This is a **research** forecast with high uncertainty — not betting or financial advice.

## What you get (v0.3.3)

| Done | Planned |
|---|---|
| OpenDota ETL: **65 leagues**, ~8.7k maps | Playoff bracket |
| Team Elo + Glicko-2 (±uncertainty) + player/chemistry features | Draft simulation |
| CatBoost training → Swiss simulations → heatmap | Crowd board |
| Multi-source blend: model, analysts, market, rankings | Mid-group retraining |
| Static UI on GitHub Pages | |

Production model: **CatBoost** pairwise. Leave-one-TI-out AUC (TI12–14): **0.608**.  
Player coverage in the training corpus: **8709 / 8709** (100%).

## Branches

| Branch | Role |
|---|---|
| `prod` | Active development |
| `main` | Stable snapshot for GitHub Pages |

## Web UI

Static files live in `docs/`:

```powershell
python scripts/export_web_data.py
cd docs
python -m http.server 8080
```

Open http://localhost:8080 (`file://` will not work — `fetch` needs a server).

UI languages: `ru` / `en` / `de` / `fr` / `pt` / `es` (sidebar switcher, stored in `localStorage`).

Rules: `#rules` on the same page — informal paraphrase of [tirules](https://www.dota2.com/esports/ti15/tirules), not a Valve document.

Board modes: Model / Analysts / Mixed. Mixed has presets and weight sliders (snaps to the nearest **precomputed** fusion scenario).

Default: **50,000** Swiss simulations. Re-run `export_web_data.py` to refresh `docs/data/predictions.json`.

## Updating during Group Stage (13–16 Aug)

Refresh market/experts and re-export. **Do not retrain** unless you have new match details or feature changes.

Write known pairings and series results to `data/ti2026_swiss_results.json`  
(`phase`, `series[]` with `round`, `team_a`, `team_b`, `status`, optional `winner`).  
The simulator **locks** known pairs; unfinished series are sampled from the model win matrix.

```powershell
# edit data/ti2026_swiss_results.json when pairings/results change
python scripts/fetch_market_priors.py
python scripts/fetch_battlepass_experts.py
python scripts/export_web_data.py
```

Dry-run market fetch: `python scripts/fetch_market_priors.py --dry-run`  
After GS: `python scripts/gs_postmortem.py` → `outputs/gs_postmortem.md`.

## Full pipeline from scratch

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

Flow: matchlists → details → Elo/Glicko/player features → CatBoost → ~50k Swiss Monte Carlo → slot/qualify probabilities → points-optimal / fusion boards → `docs/data/predictions.json`.

## External signals

- **Analysts:** curated boards in `docs/data/analyst_picks.json`; extra expert grids in `data/historical/expert_predictions.json`.  
- **Market:** `data/ti2026_market_priors.json` via `python scripts/fetch_market_priors.py` (Polymarket Gamma API, no key). Swiss-slot books are scarce, so slot odds are **derived** from tournament-winner prices. If there is no live market, fusion sets market weight to 0.  
- **Fusion:** independent soft weights (need not sum to 1; renormalized at blend time). Production default model weight **0.65**. In-sample weight tuning is diagnostic only.

## Data files

1. League registry: `data/league_candidates.json`  
2. Match details: `data/raw/details_shards/…` (local; usually gitignored)  
3. Rosters: `src/ti2026/teams.py`, `data/ti2026_rosters.json`  
4. Analyst picks: `docs/data/analyst_picks.json`  
5. Market priors: `data/ti2026_market_priors.json`  
6. Historical experts: `data/historical/expert_predictions.json`  
7. Live GS pairings/results: `data/ti2026_swiss_results.json`

More detail: [docs/FEATURES.md](docs/FEATURES.md), [docs/RESULTS.md](docs/RESULTS.md), [docs/ROADMAP_v03.md](docs/ROADMAP_v03.md).

## License

[MIT](LICENSE). Match data from OpenDota — follow their terms of use.
