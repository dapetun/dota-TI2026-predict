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

## Sample weights

`w = 0.5 ** (age_days / 90) * tier_weight`, затем нормализация среднего к 1.

## Дальше

Player KDA/GPM, сыгранность, draft, patch embeddings — отдельные этапы.
