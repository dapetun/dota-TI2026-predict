# Feature catalog

Все признаки считаются **до** исхода текущего матча (нет leakage).

| Признак | Источник | Как считается | Зачем | Ожидаемое влияние |
|---|---|---|---|---|
| `r_elo` / `d_elo` | OpenDota matchlists | Running Elo, K зависит от tier | Базовая сила команды | Сильное |
| `diff_elo` | derived | `r_elo - d_elo` | Главный сигнал перевеса | Очень сильное |
| `abs_elo_diff` | derived | `\|diff_elo\|` | Нелинейность «разгромов» | Среднее |
| `elo_prob` | derived | logistic Elo expectation | Калиброванный prior | Сильное |
| `r_wr` / `d_wr` | match history | Decayed WR последних ≤15 карт | Среднесрочная форма | Среднее–сильное |
| `diff_wr` | derived | разница form WR | Относительная форма | Сильное |
| `r_wr5` / `d_wr5` | match history | WR последних 5 карт | Короткая форма / momentum | Среднее |
| `diff_wr5` | derived | разница short-form | Hot/cold streaks | Среднее |
| `r_streak` / `d_streak` | match history | текущая win-streak | Momentum | Слабое–среднее |
| `diff_streak` | derived | разница стриков | — | Слабое |
| `h2h_wr` | H2H cache | WR radiant в последних ≤10 очных | Matchup-специфика | Среднее (редко) |
| `diff_h2h` | derived | `h2h_wr - 0.5` | Центрированный H2H | Среднее |
| `r_gp` / `d_gp` | history length | число карт до матча | Cold-start / опыт в датасете | Слабое–среднее |
| `diff_gp` | derived | разница опыта | — | Слабое |
| `r_avg_tier` / `d_avg_tier` | history | средний tier_weight последних 10 | Уровень соперничества | Среднее |
| `diff_tier` | derived | разница avg tier | — | Слабое–среднее |
| `tier_weight` | tournament registry | TI=2.0, major=1.5, … | Контекст важности матча | Слабое как фича (важно как sample weight) |
| `r_days_since` / `d_days_since` | last played ts | дни простоя | Ржавчина / пауза | Слабое–среднее |

## Player features

Строятся по `account_id` из OpenDota match details. История игрока обновляется **после** матча.

| Признак | Источник | Как считается | Зачем | Ожидаемое влияние |
|---|---|---|---|---|
| `r_pl_kda` / `d_pl_kda` | player details | mean KDA игроков состава за ≤20 прошлых карт | Индивидуальная форма | Среднее–сильное |
| `diff_pl_kda` | derived | разница KDA сторон | Перевес скилла | Сильное |
| `r_pl_gpm` / `d_pl_gpm` | player details | mean GPM состава | Экономика / фарм | Среднее |
| `diff_pl_gpm` | derived | разница GPM | — | Среднее |
| `r_pl_xpm` / `d_pl_xpm` | player details | mean XPM | Темп опыта | Среднее |
| `diff_pl_xpm` | derived | разница XPM | — | Среднее |
| `r_pl_hdpm` / `d_pl_hdpm` | player details | hero damage / min | Фраг / дамаг | Среднее |
| `diff_pl_hdpm` | derived | разница HD/min | — | Среднее |
| `r_pl_tdpm` / `d_pl_tdpm` | player details | tower damage / min | Объектный урон | Слабое–среднее |
| `diff_pl_tdpm` | derived | разница TD/min | — | Слабое |
| `r_pl_wr` / `d_pl_wr` | player details | mean WR игроков | Сила состава независимо от тега | Сильное |
| `diff_pl_wr` | derived | разница WR | — | Сильное |
| `r_pl_games` / `d_pl_games` | player details | mean число прошлых карт в корпусе | Опыт / cold-start | Слабое–среднее |
| `diff_pl_games` | derived | разница опыта | — | Слабое |
| `r_pl_lan_wr` / `d_pl_lan_wr` | player details | WR только на LAN (tier LAN) | LAN-форма к TI | Среднее |
| `diff_pl_lan_wr` | derived | разница LAN WR | — | Среднее |
| `has_player_stats` | coverage flag | 1 если есть details | Модель видит пропуски | Служебное |

## Sample weights

`w = 0.5 ** (age_days / 90) * tier_weight`, затем нормализация среднего к 1.

## Дальше

Сыгранность (co-play), draft, patch embeddings — отдельные этапы.
