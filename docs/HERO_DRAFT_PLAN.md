# Hero / Draft слой — дизайн (v0.3.x → v0.4)

Дата: 2026-08-08 · статус: **A soft prior + known-draft logit shift реализованы** (default off)  
Связано: [ROADMAP_v03.md](ROADMAP_v03.md), [FEATURES.md](FEATURES.md), `src/features/hero_soft_prior.py`

- Roster soft prior: `apply_soft_prior_matrix` + env `USE_HERO_SOFT_PRIOR=1`
- Live known-draft: `apply_known_draft_logit_shift(p, radiant_heroes, dire_heroes)`
- Fetch meta: `python scripts/build_hero_meta.py --fetch` (retries + cache)
- Draft MC (вариант B) — по-прежнему P2 / post-Swiss

Цель: использовать signature-героев игроков и винрейты (meta high-MMR + личный pub WR + pro history), чтобы корректировать P(win) до/после драфта — **аддитивно**, не блокируя текущий Swiss/pairwise пайплайн (фаза A corpus expansion).

---

## 1. Анализ идеи

### 1.1 Почему может помочь

| Сигнал | Что даёт |
|---|---|
| **Player hero pool** | Игрок «живёт» на 5–15 героях; сила команды ≠ средний Elo, если meta бьёт по пулу (или наоборот) |
| **Player × hero pub WR** | Личный skill ceiling на герое в ranked (часто видно до турнирного сэмпла) |
| **Global high-MMR / pro meta WR** | Baseline силы героя в патче (7.41e и т.п.) |
| **Meta fit** | Пересечение signature pool × contested meta → prior «команда готова к патчу» |
| **Matchup edges** | После известного драфта (или в MC) hero-vs-hero даёт локальный edge поверх team strength |

Текущий пайплайн (team Elo/form + player rolling + chemistry) **не видит героев** в train-признаках v0.3. Legacy `compute_hero_features` в `feature_engineering.py` есть, но в `FEATURE_COLUMNS` / pairwise не входит. В ростерах уже есть `account_id` → можно тянуть player×hero без склейки ников.

### 1.2 Почему сложно

1. **Драфт последовательный и неполный** до матча; Swiss board считается **до** TI, когда драфтов ещё нет.
2. **TI ≠ pub 7k+**: пики, баны, фасеты, роли, психология серий отличаются; D2PT сам пишет, что match history skews WR вверх.
3. **Малый pro-сэмпл** на героя у игрока (часто 3–15 карт за патч) → шум без shrinkage.
4. **Leakage**: нельзя использовать исход текущей карты / пост-драфт знание в train до `start_time`.
5. **Series-level** (Bo3): драфт map2 зависит от map1; простая avg matchup недооценивает adaptation.
6. **Роль**: WR Spectre mid ≠ carry; без position-aware агрегации сигнал ломается.

### 1.3 Когда применять

| Режим | Когда | Что считать |
|---|---|---|
| **Roster prior (pre-TI)** | Сейчас / Swiss MC / fantasy board | Soft prior: pool × meta × player-pub WR → Δ силы / logit |
| **Live per-series** | День матча, известен драфт (или mid-draft) | Matchup matrix + known 5v5; опц. remaining-pick MC |
| **Оба (рекомендация)** | v0.3.x soft → v0.4 live | Не блокируют друг друга |

---

## 2. Архитектурные варианты

### A. Soft prior only

Без симуляции драфта. По ростеру:

1. Для каждого игрока — top-K signature heroes (по games / recent).
2. Смешать **global meta WR**, **player-pub WR**, **player-pro WR** (см. §4).
3. Агрегировать в team `meta_fit`, `sig_wr`, `pool_depth` → сдвиг pairwise logit или feature-колонки в blend.

**Плюсы:** быстро, легально через OpenDota, работает до драфта, встраивается в Swiss.  
**Минусы:** не моделирует конкретный 5v5; слабый edge на live day.

### B. Draft Monte Carlo

Сэмплировать баны/пики из pick-rate priors (player + meta) → скорить 5v5 matchup matrix → усреднять P(win).

**Плюсы:** ближе к «настоящему» edge после/вместо неизвестного драфта.  
**Минусы:** сложность (ban phase, role constraints, captain style), хрупкие данные matchup, дорого валидировать; **не нужен** для текущей Swiss-доски.

### C. Hybrid (рекомендация)

- **v0.3.x:** вариант **A** как additive prior поверх текущего blend.  
- **v0.4 / live:** вариант **B** (или known-draft scorer) только когда есть драфт / match day.

**Почему C:** ROADMAP явно откладывает «draft embeddings» до расширения корпуса; Swiss сейчас — series-level без драфтов. Soft prior даёт measurable ablation за дни; MC — отдельный продуктный слой.

---

## 3. Источники данных

### 3.1 dota2protracker.com (research)

Публичные страницы (SPA, данные с 7k+ MMR + pro, enriched Imprint/Stratz):

| Тип | URL-паттерн | Полезное |
|---|---|---|
| Meta home | `/` | WR / matches / D2PT rating по ролям |
| Hero | `/hero/{name}` | WR по роли, pick trend, builds; matchups часто «Not enough Data» |
| Players | `/players` | Pro leaderboard, top heroes, pool |
| Player | `/player/{nick}` | Account id, tournament matches, top heroes, WR |

**API:** официального публичного API нет. Есть неофициальные обёртки (Parse.bot) и scrapers (Puppeteer). Страницы JS-heavy → простой HTML scrape хрупкий.

**Риски:** ToS / rate-limit / блокировки; ломкость DOM; этика (сайт чужой бизнес); зависимость от патч-сброса.  
**Вывод:** D2PT — **вторичный/калибровочный** источник meta и UX-ориентир, не primary для MVP.

### 3.2 OpenDota (предпочтительный primary)

Уже в репо (`OpenDotaClient`, match details с `hero_id`, ростеры с `account_id`).

| Endpoint | Что даёт |
|---|---|
| `GET /players/{account_id}/heroes` | **player × hero**: `games`, `win`, `last_played`, with/against — **pub career** (фильтр по date params возможен) |
| `GET /heroes/{id}/matchups` | hero-vs-hero games/wins (в основном pub) |
| `GET /heroStats` | global pub pick/win по bracket'ам |
| Match details / league matches | **player-pro** hero history из нашего корпуса |
| `GET /heroes` | id ↔ name |

**Плюсы:** ToS-дружелюбнее, стабильный JSON, уже rate-limit в клиенте, 80 account_id TI уже в `ti2026_rosters.json`.  
**Минусы:** career heroes смешивают патчи; нет «только 7k+»; matchups — не pro-only.

### 3.3 STRATZ (fallback / enrichment)

В репо есть `stratz_api.py` (`proWin` / `proPick` / `proBan`). Нужен `STRATZ_TOKEN`. Хорош для pro meta snapshot; для player×hero pub — OpenDota проще.

### 3.4 Приоритет источников

| Слой | Primary | Fallback |
|---|---|---|
| Global hero meta WR / pick | OpenDota `heroStats` (+ опц. STRATZ pro) | D2PT scrape (позже, opt-in) |
| **Player × hero pub WR** | OpenDota `/players/{id}/heroes` | D2PT player page (только если нужен high-MMR filter) |
| Player × hero pro WR | Наш `player_matches` / match details | STRATZ player matches |
| Hero matchups | OpenDota `/heroes/{id}/matchups` | D2PT matchup tabs (часто пустые) |
| Draft priors (pick rates) | Pro picks из корпуса + heroStats | D2PT contested % |

---

## 4. Трёхслойный player×hero сигнал (ключевое дополнение)

Помимо **общего** винрейта героя в high-MMR meta, смотрим **конкретного киберспортсмена** и его **личный WR на герое в паблике**.

### 4.1 Три независимых слоя

```
L1  global_meta[h, role?, patch]     — сила героя в мете
L2  player_pub[p, h]                 — личный pub WR / games (signature + deep pool)
L3  player_pro[p, h]                 — WR в pro/LAN из нашего корпуса
```

Они **не заменяют** друг друга:

- L1 без L2: «герой сильный, но игрок на нём 42% в 80 пабах».
- L2 без L1: «у игрока 60% на оффмете, который на TI забанят».
- L3: самый релевантный, но часто `n < 10` → обязательный shrinkage к L1/L2.

### 4.2 Смешивание (priors / shrinkage)

Обозначения: \(\hat{w} = wins/games\), \(n\) = games.

**Empirical Bayes к meta:**

\[
w_{\mathrm{shrunk}}(p,h) = \frac{n}{n+k}\, \hat{w}(p,h) + \frac{k}{n+k}\, w_{\mathrm{meta}}(h)
\]

Рекомендуемые стартовые \(k\):

| Слой | \(k\) | Комментарий |
|---|---|---|
| player_pub | 20–40 | Пабы шумные; smurf/boost → не доверять малому n |
| player_pro | 8–15 | Мало карт, но выше вес единицы |
| Итоговый blend | — | См. ниже |

**Итоговый score героя для игрока (MVP):**

\[
s(p,h) = \alpha\, w_{\mathrm{meta}} + \beta\, w_{\mathrm{pub,shrunk}} + \gamma\, w_{\mathrm{pro,shrunk}}
\]

Стартовые веса (тюнить на ablation): \(\alpha=0.35\), \(\beta=0.40\), \(\gamma=0.25\), при \(n_{\mathrm{pro}}\ge 15\) поднять \(\gamma\); при \(n_{\mathrm{pub}}<10\) снизить \(\beta\).

**Team soft prior:** mean top-K signatures по пятёрке (K≈5–8 на игрока), опционально взвесить contested meta pick-rate.

### 4.3 Signature vs deep pool

| Класс | Правило (стартовое) | Использование |
|---|---|---|
| Signature | top по `games` за окно патча / 180d, `n≥15` pub или `n≥5` pro | Основной meta-fit |
| Deep pool | следующие 10–20 героев с `n≥5` | Draft MC / pool depth feature |
| Situational | редкие пики | Игнор в soft prior; только live draft |

### 4.4 Риски именно player-pub слоя

| Риск | Митигация |
|---|---|
| Smurf / boosted pubs | Min MMR proxy нет в OpenDota heroes; фильтр `last_played` + half-life; down-weight если только pub без pro |
| Патч-дрейф | Окно с даты патча (или 90d); career endpoint без фильтра — **опасно** |
| Роль | Агрегировать по типичной роли из pro lineup / D2PT role tabs позже |
| Sample size | Shrinkage; `uncertainty = 1/sqrt(n+1)` как признак |
| pub ≠ pro | Низкий \(\beta\) если есть pro; калибровка на исторических TI maps |
| Party / unranked шум | OpenDota heroes ≈ ranked history; всё равно не ideal |

---

## 5. Модель данных

Каталог (предлагаемый): `data/hero/` (не мешает `data/raw` matchlists).

### 5.1 `hero_meta.json`

```json
{
  "as_of": "2026-08-06",
  "patch": "7.41e",
  "source": "opendota_heroStats",
  "heroes": [
    {
      "hero_id": 67,
      "name": "spectre",
      "pick_rate": 0.13,
      "win_rate": 0.53,
      "pro_pick": 120,
      "pro_win": 0.51,
      "by_role": {"carry": {"n": 2800, "wr": 0.54}}
    }
  ]
}
```

### 5.2 `player_signatures.json` (player × hero)

```json
{
  "as_of": "2026-08-06",
  "window_days": 180,
  "players": {
    "10366616": {
      "nick": "skiter",
      "heroes": [
        {
          "hero_id": 54,
          "pub_games": 80,
          "pub_wins": 48,
          "pub_wr": 0.60,
          "pro_games": 12,
          "pro_wins": 8,
          "pro_wr": 0.67,
          "meta_wr": 0.51,
          "score": 0.57,
          "class": "signature"
        }
      ]
    }
  }
}
```

### 5.3 `hero_matchup.parquet` / `.json`

Матрица \(124\times124\): `games`, `wins` (hero A vs B), `wr_shrunk`, `source`, `as_of`.

### 5.4 `draft_priors.json` (v0.4)

Pick/ban rates по роли × hero; опционально conditioned on player signatures.

### 5.5 Refresh cadence

| Артефакт | Частота |
|---|---|
| `hero_meta` | 1× / 1–3 дня в патче; сразу после патча |
| `player_signatures` | 1× / 2–3 дня для 80 TI account_id |
| `hero_matchup` | 1× / неделю |
| `draft_priors` | после крупных турниров / перед TI |

Кэш на диске; скрипты идемпотентны; не в hot path Swiss MC.

---

## 6. План реализации в этом репо

### 6.1 Новые модули (additive)

Предпочтительно **`src/features/`** для soft prior (рядом с player/chemistry); **`src/draft/`** — только когда появится MC.

```
src/features/hero_meta.py          # load/build hero_meta
src/features/player_signatures.py  # OpenDota heroes + pro from player_matches
src/features/hero_soft_prior.py    # compose team/pair soft features + logit shift
src/draft/__init__.py              # stub package (optional)
src/draft/matchup_matrix.py        # v0.4
src/draft/simulate.py              # v0.4 Monte Carlo
scripts/build_hero_meta.py
scripts/build_player_signatures.py
```

Не трогать структуру турниров / download корпуса (фаза A другого агента).

### 6.2 Встраивание

| Точка | Как |
|---|---|
| `pairwise.py` | После `compose_full_pair_row`: добавить soft columns **или** `logit += λ * (fit_r - fit_d)` post-model (проще для MVP, не ломает train schema) |
| `export_web_data.py` | Опц. поля `meta_fit`, `sig_wr` в team payload |
| Swiss MC | Без изменений алгоритма: только через обновлённую pairwise matrix |
| Train (позже) | Feature columns в `FEATURE_COLUMNS` + LOO ablation |

**MVP предпочтение:** post-model logit shift с λ из grid на hold-out — не требует ретрейна; ablation чистая.

### 6.3 Feature columns (если soft prior в модели)

Предлагаемые колонки (v0.3.x opt-in):

- `r_meta_fit` / `d_meta_fit` / `diff_meta_fit` — mean signature×meta score
- `r_sig_wr` / `d_sig_wr` / `diff_sig_wr` — mean shrunk player score
- `r_pool_depth` / `d_*` / `diff_*` — # heroes с score>threshold
- `r_pub_pro_gap` — mean (pub_wr − pro_wr) как риск-флаг
- `has_hero_prior` — coverage flag

### 6.4 Draft MC (псевдокод, v0.4)

```
def p_win_draft_mc(team_r, team_d, n_sims=500):
    scores = []
    for _ in range(n_sims):
        draft = sample_draft(ban_prior, pick_prior, signatures_r, signatures_d)
        # draft.radiant[5], draft.dire[5]
        m = 0.0
        for hr in draft.radiant:
            for hd in draft.dire:
                m += logit(matchup_wr_shrunk(hr, hd))
        m /= 25
        # blend with team pairwise prior
        scores.append(sigmoid(a * team_logit(team_r, team_d) + b * m))
    return mean(scores)
```

Constraints: 1 hero / role bucket; no duplicate heroes; ban list removes from pool.

### 6.5 Валидация

1. **Backtest maps** с известным драфтом (TI13/14 details): known-draft scorer vs team-only blend.  
2. **Ablation Swiss:** baseline blend vs +soft prior → LOO AUC, LL, E[points].  
3. **Calibration:** reliability diagram на map outcomes.  
4. **Leakage check:** priors только из данных строго до `start_time` матча.  
5. Не ждать улучшения Swiss board >1–2 слотов без live draft — ожидания реалистичные.

### 6.6 UI (позже)

Опциональный chip «meta fit» / «signature edge» на странице команды; не в hero viewport первого экрана until есть данные.

### 6.7 Риски, этика, scraping

- **Primary = OpenDota API** (rate ~1 req/s уже в клиенте; 80 players × 1 call ≈ 2 мин).  
- D2PT scrape — только opt-in script, robots/ToS review, backoff, no redistribute raw dump.  
- Не хранить PII сверх публичных account_id / nick.  
- Документировать, что pub WR ≠ гарантированный pro performance.

---

## 7. Фазы / milestones

| Фаза | Срок | Scope |
|---|---|---|
| **M0 MVP** | 1–3 дня | См. §8 |
| **M1** | +3–5 дней | Soft features в train + LOO ablation; patch window |
| **M2** | v0.4 | Matchup matrix + known-draft scorer для live |
| **M3** | позже | Draft MC + UI chip; опц. D2PT high-MMR overlay |

Параллельно с фазой A corpus — **не блокирует**.

---

## 8. MVP — первый PR (точный scope)

**Цель:** самый маленький полезный срез без блокировки Swiss.

### В scope

1. `scripts/build_player_signatures.py` + `src/features/player_signatures.py`  
   - Input: `data/ti2026_rosters.json` account_ids  
   - Fetch OpenDota `/players/{id}/heroes`  
   - Merge pro counts из `data/processed/player_matches.csv` (если есть)  
   - Shrinkage к `hero_meta` (даже простой global WR из heroStats)  
   - Output: `data/hero/player_signatures.json`
2. `scripts/build_hero_meta.py` → `data/hero/hero_meta.json` из OpenDota `heroStats`
3. `src/features/hero_soft_prior.py`: `team_meta_fit(account_ids) → float` и `apply_soft_prior(P, lambda_)` (logit shift 16×16)
4. Хук в `pairwise` / export **за флагом** `USE_HERO_SOFT_PRIOR=0` default off
5. Короткий тест на 2–3 account_id + schema JSON
6. Ссылка из README / RESULTS stub «hero soft prior experimental»

### Вне scope MVP

- Scraping dota2protracker  
- Draft Monte Carlo / matchup MC  
- Новые колонки в XGB/CatBoost train  
- UI chip  
- STRATZ обязателен  

### Acceptance

- Артефакты строятся для всех 16×5 account_id с coverage ≥90%  
- Swiss export с флагом off == текущий baseline bit-identical  
- С флагом on: matrix меняется плавно (|ΔP| median < 0.05 при λ≤0.3)

---

## 9. Рекомендация (summary)

| Вопрос | Ответ |
|---|---|
| Архитектура | **C (Hybrid):** soft prior сейчас, draft MC позже |
| Primary data | **OpenDota** (player heroes + heroStats + наш pro corpus) |
| D2PT | Inspiration / optional overlay, не MVP |
| Player-pub WR | **Отдельный слой L2** со shrinkage к meta; смешивать с L1 и L3 |
| Первый PR | Signatures JSON + optional logit shift в pairwise |

Документ обновлять после MVP ablation (λ, k, αβγ).
