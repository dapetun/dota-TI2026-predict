/** UI i18n: ru / en / de / fr / pt / es. Persists in localStorage. */

export const LANG_STORAGE_KEY = "ti2026_ui_lang";
export const SUPPORTED_LANGS = ["ru", "en", "de", "fr", "pt", "es"];

const LOCALE_TAGS = {
  ru: "ru-RU",
  en: "en-US",
  de: "de-DE",
  fr: "fr-FR",
  pt: "pt-BR",
  es: "es-ES",
};

/** @type {Record<string, Record<string, string>>} */
const STRINGS = {
  ru: {
    "meta.title": "TI 2026 · Прогноз Swiss",
    "meta.description": "Открытый прогноз групповой стадии The International 2026",
    "brand.kicker": "Открытый прогноз",
    "brand.title": "Прогноз Swiss",
    "nav.board": "Прогнозы",
    "nav.standings": "Команды",
    "nav.heatmap": "Карта слотов",
    "nav.matchup": "Матч",
    "nav.model": "Модель",
    "nav.how": "Как считаем",
    "nav.rules": "Правила",
    "nav.repo": "Репозиторий",
    "nav.home": "Прогноз",
    "nav.group.participate": "Участвуйте",
    "nav.group.support": "Подробнее",
    "lang.label": "Язык",
    "cta.board": "К доске",
    "hero.eyebrow": "The International 2026 · Групповая стадия",
    "hero.title": "Прогнозы",
    "hero.lead":
      "16 команд, Swiss до 4 побед или поражений. Ниже — прогнозная доска компендиума.",
    "disclaimer.label": "Важно",
    "disclaimer.loading": "Загрузка…",
    "warn.market_prior_power_rankings":
      "Рыночные вероятности сейчас взяты из нашего рейтинга команд, а не из живых коэффициентов букмекеров; вес рынка в смеси обнулён, чтобы не учитывать рейтинг дважды.",
    "warn.in_sample_tune_model_weight":
      "Подстройка на той же выборке предлагала вес модели {tuned}; в публикации используем рабочий вес {production}.",
    "warn.unknown": "Есть техническое предупреждение к этому прогнозу.",
    "board.title": "Доска Swiss",
    "board.lead":
      "Прогноз слотов компендиума — не официальная сетка Valve. Слоты: один 4-0, два 4-1, пять на проход, пять на выбывание, два 1-4, один 0-4.",
    "board.strategy_label": "Стратегия доски",
    "mode.title": "Как собрать доску",
    "mode.help": "Выберите источник прогноза. Доска ниже перестроится сразу.",
    "mode.current": "Сейчас",
    "mode.points_approx": "ожид. очки",
    "mode.group.model": "Модель",
    "mode.group.model_desc": "Считаем сами по симуляциям матчей",
    "mode.group.analysts": "Аналитики",
    "mode.group.analysts_desc": "Консенсус экспертных сеток",
    "mode.group.fusion": "Смешанный",
    "mode.group.fusion_desc": "Модель + аналитики + рынок",
    "mode.analysts_hint": "Одна доска по голосам аналитиков — без доп. настроек.",
    "strategy.points_optimal": "Максимум очков компендиума",
    "strategy.qualify_rank": "По шансу пройти дальше",
    "strategy.analyst_consensus": "Консенсус аналитиков",
    "strategy.fusion": "Рекомендуемая смесь",
    "strategy.fusion_hint": "Веса модели и аналитиков как в опубликованном прогнозе",
    "strategy.fusion_model_heavy": "Больше доверять модели",
    "strategy.fusion_balanced": "Баланс источников",
    "strategy.fusion_market_lean": "Больше доверять рынку",
    "strategy.fusion_analyst_lean": "Больше доверять аналитикам",
    "fusion.weights_label": "Веса источников",
    "fusion.weights_help":
      "Готовые смеси и ползунки. Значения независимы и не обязаны давать в сумме 100 — при расчёте доли берутся пропорционально. Ползунки переключают на ближайший заранее посчитанный вариант (живой пересчёт пока недоступен).",
    "fusion.advanced": "Тонкая настройка",
    "fusion.preset.fusion": "Рекомендуемая",
    "fusion.preset.fusion_model_heavy": "Больше модели",
    "fusion.preset.fusion_balanced": "Баланс",
    "fusion.preset.fusion_market_lean": "Больше рынка",
    "fusion.preset.fusion_analyst_lean": "Больше аналитиков",
    "fusion.weight.model": "Модель",
    "fusion.weight.analyst": "Аналитики",
    "fusion.weight.market": "Рынок (анонимно)",
    "fusion.weight.ranking": "Силовой рейтинг",
    "fusion.weight.expert": "История экспертов",
    "fusion.weight_in_mix": "в смеси",
    "fusion.weights_raw_sum":
      "Сумма выбранных весов: {sum} (не обязана быть 100). Ниже у каждого источника — доля после пропорционального пересчёта.",
    "fusion.score": "Ожидаемые очки этого сценария",
    "fusion.correct_slots": "верных слотов ≈",
    "fusion.market_fallback":
      "Рыночные вероятности — исследовательский сигнал. Автор не рекламирует букмекеров. Не для ставок.",
    "standings.title": "Шансы команд",
    "standings.lead":
      "μ ± σ — оценка силы и неопределённости. «Пройти» / «Вылететь» — доля симуляций Swiss (~50 000).",
    "standings.sort_label": "Сортировка",
    "standings.sort.qualify": "По шансу пройти",
    "standings.sort.power": "По силе",
    "standings.sort.elim": "По шансу вылететь",
    "standings.sort.alpha": "А–Я",
    "standings.search": "Найти команду…",
    "standings.th.rank": "#",
    "standings.th.team": "Команда",
    "standings.th.region": "Регион",
    "standings.th.strength": "Сила μ±σ",
    "standings.th.qualify": "Пройти",
    "standings.th.elim": "Вылететь",
    "standings.th.wins": "Ожид. побед",
    "standings.th.record": "Рекорд",
    "heatmap.title": "Карта слотов",
    "heatmap.lead":
      "Доля симуляций (%), где команда оказалась в каждом слоте компендиума (16 команд × 6 слотов).",
    "heatmap.empty": "Нет данных карты слотов",
    "heatmap.team": "Команда",
    "matchup.title": "Кто кого",
    "matchup.lead": "Парная вероятность победы (смесь парных оценок или силовой рейтинг).",
    "matchup.team_a": "Команда A",
    "matchup.team_b": "Команда B",
    "matchup.series_win": "вероятность победы серии",
    "model.title": "Состояние модели",
    "model.lead_fallback": "Смесь XGBoost + CatBoost; проверка Leave-One-TI-Out; сила μ±σ.",
    "how.title": "Как считаем",
    "how.lead_before":
      "Простыми словами — что за цифры на сайте и откуда они берутся. Детали и каталог признаков — в",
    "how.lead_and": "и",
    "how.card1.title": "1. Данные",
    "how.card1.body":
      "Списки матчей (~65 лиг, 8709 карт) + детали составов (осколки OpenDota). Без деталей нет признаков игроков и сыгранности.",
    "how.card2.title": "2. Покрытие составов",
    "how.card2.body":
      "Доля матчей корпуса, где есть состав игроков из деталей. Сейчас 100% (8709/8709) — цель ≥80% выполнена с запасом. Дыр по лигам нет: shards с players покрывают весь корпус.",
    "how.card3.title": "3. Сила команды",
    "how.card3.body":
      "Elo — текущая сила по результатам; Glicko-2 добавляет неопределённость (RD). Форма (~40 дней) — недавний процент побед. Сыгранность — насколько пятёрка сыграна вместе.",
    "how.card4.title": "4. Матч → Swiss",
    "how.card4.body":
      "Модель даёт P(победа) в паре. Затем ~50 000 раз разыгрываем весь Swiss и считаем шансы слотов и прохода.",
    "how.card5.title": "5. Метрики качества",
    "how.card5.body":
      "LOO AUC — тест на прошлых TI (ближе к TI 2026). Walk-forward AUC — устойчивость во времени. AUC ~0.5 = монетка; у смеси сейчас LOO ≈0.58, WF ≈0.64.",
    "how.card6.title": "6. Числа в интерфейсе",
    "how.card6.body":
      "μ ± σ — сила ± неопределённость. Карта слотов — % слотов. Стратегии доски: модель / проход / аналитики / смешанный (разные веса источников). Не для ставок.",
    "how.details_summary": "Чуть подробнее (для интересующихся)",
    "how.details_li1":
      "Строка игрока: в деталях есть players[], account_id ≠ 0/None и известен team_id стороны.",
    "how.details_li2":
      "Покрытие составов: 100% (8709/8709). Детали докачаны по всем лигам корпуса (DLS28/29, EWC 2026, TI14, ESL Birmingham, Wallachia S8, BLAST VI/VII и др.). Need download=0.",
    "how.details_li3":
      "Полураспад выборки 210д × tier (TI/major/qual/online); форма ~40д; патч ≥7.41 ×1.25.",
    "how.details_li4":
      "Смесь XGB+CatBoost с весами по LOO log-loss; в production может быть изотоническая калибровка.",
    "how.details_li5":
      "Нечётный остаток в корзине рекорда Swiss получает bye без авто-победы (упрощение симуляции).",
    "footer.open": "Открытый проект. Данные — OpenDota.",
    "footer.license": "Лицензия MIT",
    "projects.title": "Другие проекты",
    "projects.moex.desc": "Котировки Мосбиржи и портфель по Марковицу",
    "projects.keyboard.desc": "Помогает пройти мини-игру «Атака автоматонов» из ивента «Тёмный карнавал» в Dota",
    "col.undefeated": "4–0 · без поражений",
    "col.one_loss": "4–1 · одно поражение",
    "col.advance": "Проход",
    "col.eliminate": "Выбывание",
    "col.one_win": "1–4 · одна победа",
    "col.winless": "0–4 · без побед",
    "board.empty": "Пока пусто",
    "board.slot_p": "P(слот)",
    "chip.points": "Очки (ожид.)",
    "chip.analysts": "Аналитики",
    "chip.fusion": "Смешанный",
    "chip.analysts_fusion": "Аналитики / смешанный",
    "chip.sims": "Прогонов турнира",
    "chip.model_blend": "Прогноз модели",
    "chip.model_power": "Упрощённый рейтинг",
    "chip.model_default": "Модель",
    "format.swiss": "Swiss 16→4, 5 раундов Bo3 + доп. раунд",
    "footer.updated": "Обновлено",
    "fallback.banner":
      "Запасной режим: силовой рейтинг (не смесь парных оценок). Переобучите модель через train_compare.",
    "valve.points_title": "Очки Valve за точные слоты:",
    "metric.loo": "Проверка на прошлых TI",
    "metric.loo_hint": "Насколько хорошо модель угадывает матчи прошлых TI (0.5 = монетка)",
    "metric.corpus": "Корпус",
    "metric.corpus_hint": "лиг · списки матчей",
    "metric.corpus_hint_fallback": "матчей в данных",
    "metric.points": "Очки модели (ожид.)",
    "metric.points_hint": "Ожидаемые очки компендиума у доски модели",
    "metric.coverage": "Составы игроков",
    "metric.coverage_hint_base": "доля матчей корпуса, где есть состав из деталей",
    "metric.coverage_with": "матчей с составом",
    "metric.coverage_ok": "цель ≥80% — уже ок",
    "metric.coverage_left": "осталось ~{n}: детали ещё не скачаны (не анонимы)",
    "metric.coverage_title":
      "Покрытие = доля матчей корпуса с составом игроков из деталей. Сейчас 100%: у каждого match_id корпуса есть shard с players.",
    "metric.coverage_note":
      "Не 100%: у ~{n} матчей нет скачанных деталей. Цель ≥80%; live-цель — закрыть остаток через download_details.",
    "title.analyst_pts": "Ожидаемые очки доски по консенсусу аналитиков",
    "title.fusion_pts": "Ожидаемые очки после смешивания модели, аналитиков и рынка",
    "title.both_pts": "Ожидаемые очки: аналитики и смешанный прогноз совпадают",
    "title.strength": "Сжатый Elo ± суммарная σ",
    "error.load": "Ошибка загрузки данных: {msg}. Открой через локальный сервер или GitHub Pages.",
    "error.fetch": "Не удалось загрузить predictions.json ({status})",
    "rank.home": "дом +",
    "rules.meta_title": "Правила Swiss TI 2026 · вольный пересказ",
    "rules.eyebrow": "The International 2026",
    "rules.title": "Правила групповой стадии",
    "rules.lead":
      "Краткий вольный пересказ официальных правил Swiss для TI 2026. Это не юридический текст и не замена оригиналу Valve.",
    "rules.disclaimer_label": "Дисклеймер",
    "rules.disclaimer":
      "Страница — неофициальный пересказ для удобства. При расхождениях верны только официальные правила Valve. Автор не связан с Valve Corporation.",
    "rules.official_link": "Официальные правила на dota2.com",
    "rules.dates_title": "Даты",
    "rules.dates_body":
      "Групповая стадия: 13–16 августа (онлайн). Основной турнир: 20–23 августа, SPD Bank Oriental Sports Center, Шанхай.",
    "rules.rank_title": "Как ранжируют команды",
    "rules.rank_intro": "Порядок критериев (сверху вниз):",
    "rules.rank_1": "Число выигранных матчей",
    "rules.rank_2": "Число проигранных матчей",
    "rules.rank_3": "Сумма побед соперников, с которыми играли",
    "rules.rank_4": "Процент выигранных карт",
    "rules.rank_5": "Средний процент побед карт у соперников",
    "rules.rank_6": "Средняя длительность карты (короче — лучше)",
    "rules.rank_7": "Жребий (монетка)",
    "rules.pair_title": "Общие правила пар Swiss",
    "rules.pair_1": "Команды с одинаковым рекордом ставят друг против друга",
    "rules.pair_2": "Повторных пар избегают, когда можно",
    "rules.pair_3": "Стараются минимизировать разрыв в рейтинге между соперниками",
    "rules.round_title": "Особенности по раундам",
    "rules.round_1":
      "Раунд 1: команды делят на две группы; пары задаёт организатор внутри группы.",
    "rules.round_2": "Раунды 2–3: пары только внутри исходной группы.",
    "rules.round_3": "Раунд 4: пары только с командами из другой группы.",
    "rules.round_4":
      "Раунд 5 (матчи на выбывание): стараются максимизировать разрыв в рейтинге между соперниками.",
    "rules.elim_title": "Раунд на выбывание",
    "rules.elim_body":
      "Лучшая команда 3–2 выбирает любого из пяти соперников 2–3. Следующая по рейтингу 3–2 выбирает из оставшихся — и так далее, пока все не получат пару.",
    "rules.seed_title": "Посев на основной турнир",
    "rules.seed_body":
      "Все восемь команд, прошедшие в The International, получают посев по итоговому рейтингу Swiss.",
    "rules.pick_title": "Приоритет выбора (пик / сторона)",
    "rules.pick_bo3":
      "Bo3: жребий. Победитель выбирает пик или сторону в 1-й карте; проигравший — во 2-й. На 3-й (если есть) — новый жребий.",
    "rules.pick_bo5":
      "Bo5: 1-я и 3-я — победитель верхней сетки; 2-я и 4-я — победитель нижней; 5-я (если есть) — жребий.",
    "rules.trademark":
      "Dota и логотип Dota — товарные знаки Valve Corporation. © Valve Corporation. Все права защищены.",
  },
  en: {
    "meta.title": "TI 2026 Swiss Predictor",
    "meta.description": "Open forecast for The International 2026 group stage",
    "brand.kicker": "Open forecast",
    "brand.title": "Swiss Predictor",
    "nav.board": "Predictions",
    "nav.standings": "Teams",
    "nav.heatmap": "Slot map",
    "nav.matchup": "Matchup",
    "nav.model": "Model",
    "nav.how": "How it works",
    "nav.rules": "Rules",
    "nav.repo": "Repository",
    "nav.home": "Forecast",
    "nav.group.participate": "Participate",
    "nav.group.support": "Learn more",
    "lang.label": "Language",
    "cta.board": "To board",
    "hero.eyebrow": "The International 2026 · Group Stage",
    "hero.title": "Predictions",
    "hero.lead":
      "16 teams, Swiss to 4 wins or losses. Below — the compendium forecast board.",
    "disclaimer.label": "Note",
    "disclaimer.loading": "Loading…",
    "warn.market_prior_power_rankings":
      "Market probabilities currently come from our team ranking, not live bookmaker odds; the market share in the mix is set to zero so the ranking is not counted twice.",
    "warn.in_sample_tune_model_weight":
      "An in-sample tune suggested model weight {tuned}; the published forecast uses the production weight {production}.",
    "warn.unknown": "There is a technical warning (see export data).",
    "board.title": "Swiss board",
    "board.lead":
      "Compendium slot forecast — not Valve’s official bracket. Slots: one 4-0, two 4-1, five advancing, five eliminated, two 1-4, one 0-4.",
    "board.strategy_label": "Board strategy",
    "mode.title": "How to build the board",
    "mode.help": "Pick a forecast source. The board below updates immediately.",
    "mode.current": "Now",
    "mode.points_approx": "exp. points",
    "mode.group.model": "Model",
    "mode.group.model_desc": "Our match simulations",
    "mode.group.analysts": "Analysts",
    "mode.group.analysts_desc": "Consensus of expert grids",
    "mode.group.fusion": "Mixed",
    "mode.group.fusion_desc": "Model + analysts + market",
    "mode.analysts_hint": "One board from analyst votes — no extra settings.",
    "strategy.points_optimal": "Max compendium points",
    "strategy.qualify_rank": "By advance chance",
    "strategy.analyst_consensus": "Analyst consensus",
    "strategy.fusion": "Recommended mix",
    "strategy.fusion_hint": "Model and analyst weights as in the published forecast",
    "strategy.fusion_model_heavy": "Trust the model more",
    "strategy.fusion_balanced": "Balanced sources",
    "strategy.fusion_market_lean": "Trust the market more",
    "strategy.fusion_analyst_lean": "Trust analysts more",
    "fusion.weights_label": "Source weights",
    "fusion.weights_help":
      "Pick a ready-made mix. Sliders below snap to the nearest precomputed blend (live re-blend is not available yet). Values are independent and need not sum to 100 — at blend time shares are taken proportionally.",
    "fusion.advanced": "Fine-tune",
    "fusion.preset.fusion": "Recommended",
    "fusion.preset.fusion_model_heavy": "Model-heavy",
    "fusion.preset.fusion_balanced": "Balanced",
    "fusion.preset.fusion_market_lean": "Market lean",
    "fusion.preset.fusion_analyst_lean": "Analyst lean",
    "fusion.weight.model": "Model",
    "fusion.weight.analyst": "Analysts",
    "fusion.weight.market": "Market (anonymous)",
    "fusion.weight.ranking": "Power ranking",
    "fusion.weight.expert": "Expert history",
    "fusion.weight_in_mix": "in mix",
    "fusion.weights_raw_sum":
      "Sum of chosen weights: {sum} (need not be 100). Each source also shows its share after proportional rescale.",
    "fusion.score": "Expected points for this scenario",
    "fusion.correct_slots": "correct slots ≈",
    "fusion.market_fallback":
      "Market probabilities are a research signal. The author does not promote bookmakers. Not for betting.",
    "standings.title": "Team odds",
    "standings.lead":
      "μ ± σ — strength and uncertainty. Qualify / Eliminated — share of Swiss simulations (~50,000).",
    "standings.sort_label": "Sort",
    "standings.sort.qualify": "By qualify chance",
    "standings.sort.power": "By strength",
    "standings.sort.elim": "By elim chance",
    "standings.sort.alpha": "A–Z",
    "standings.search": "Find a team…",
    "standings.th.rank": "#",
    "standings.th.team": "Team",
    "standings.th.region": "Region",
    "standings.th.strength": "Strength μ±σ",
    "standings.th.qualify": "Qualify",
    "standings.th.elim": "Eliminated",
    "standings.th.wins": "Exp. wins",
    "standings.th.record": "Record",
    "heatmap.title": "Slot map",
    "heatmap.lead":
      "Share of sims (%) where a team landed in each compendium slot (16 teams × 6 slots).",
    "heatmap.empty": "No slot-map data",
    "heatmap.team": "Team",
    "matchup.title": "Head-to-head",
    "matchup.lead": "Pairwise win probability (pairwise blend or power ranking).",
    "matchup.team_a": "Team A",
    "matchup.team_b": "Team B",
    "matchup.series_win": "series win probability",
    "model.title": "Model status",
    "model.lead_fallback": "XGBoost + CatBoost blend; Leave-One-TI-Out; strength μ±σ.",
    "how.title": "How we compute",
    "how.lead_before": "Plain-language numbers on this site. Details and feature catalog in",
    "how.lead_and": "and",
    "how.card1.title": "1. Data",
    "how.card1.body":
      "Match lists (~65 leagues, 8709 maps) + roster details (OpenDota shards). Without details there are no player/chemistry features.",
    "how.card2.title": "2. Roster coverage",
    "how.card2.body":
      "Share of corpus matches with player rosters from details. Now 100% (8709/8709) — ≥80% target exceeded. No league gaps: player shards cover the full corpus.",
    "how.card3.title": "3. Team strength",
    "how.card3.body":
      "Elo — current strength; Glicko-2 adds uncertainty (RD). Form (~40 days) — recent win rate. Chemistry — how settled the five is.",
    "how.card4.title": "4. Match → Swiss",
    "how.card4.body":
      "The model gives P(win) for a pair. Then we simulate the full Swiss ~50,000 times and tally slot/qualify chances.",
    "how.card5.title": "5. Quality metrics",
    "how.card5.body":
      "LOO AUC — past TI test (closer to TI 2026). Walk-forward AUC — stability over time. AUC ~0.5 = coin flip; blend now LOO ≈0.58, WF ≈0.64.",
    "how.card6.title": "6. UI numbers",
    "how.card6.body":
      "μ ± σ — strength ± uncertainty. Slot map — % per slot. Board strategies: model / qualify / analysts / mixed. Not for betting.",
    "how.details_summary": "A bit more detail",
    "how.details_li1":
      "Player row: details have players[], account_id ≠ 0/None, and known side team_id.",
    "how.details_li2":
      "Roster coverage: 100% (8709/8709). Details downloaded for every corpus league (DLS28/29, EWC 2026, TI14, ESL Birmingham, Wallachia S8, BLAST VI/VII, etc.). Need download=0.",
    "how.details_li3":
      "Sample half-life 210d × tier (TI/major/qual/online); form half-life ~40d; patch ≥7.41 ×1.25.",
    "how.details_li4":
      "XGB+CatBoost blend weighted by LOO log-loss; production may use isotonic calibration.",
    "how.details_li5":
      "Odd leftover in a Swiss record bucket gets a bye without auto-win (MC simplification).",
    "footer.open": "Open project. Data — OpenDota.",
    "footer.license": "MIT License",
    "projects.title": "Other projects",
    "projects.moex.desc": "MOEX quotes and Markowitz portfolio builder",
    "projects.keyboard.desc": "Helps clear the Automatons Attack minigame from the Dark Carnival event in Dota",
    "col.undefeated": "4–0 · Undefeated",
    "col.one_loss": "4–1 · One loss",
    "col.advance": "Advancing",
    "col.eliminate": "Eliminated",
    "col.one_win": "1–4 · One win",
    "col.winless": "0–4 · Winless",
    "board.empty": "Empty for now",
    "board.slot_p": "P(slot)",
    "chip.points": "Points (exp.)",
    "chip.analysts": "Analysts",
    "chip.fusion": "Mixed",
    "chip.analysts_fusion": "Analysts / mixed",
    "chip.sims": "Tournament sims",
    "chip.model_blend": "Model forecast",
    "chip.model_power": "Simple ranking",
    "chip.model_default": "Model",
    "format.swiss": "Swiss 16→4, 5 Bo3 rounds + elim",
    "footer.updated": "Updated",
    "fallback.banner":
      "Fallback mode: power ranking (not pairwise blend). Retrain via train_compare.",
    "valve.points_title": "Valve points for exact slots:",
    "metric.loo": "Past TI check",
    "metric.loo_hint": "How well the model predicts past TI matches (0.5 = coin flip)",
    "metric.corpus": "Corpus",
    "metric.corpus_hint": "leagues · match lists",
    "metric.corpus_hint_fallback": "matches in data",
    "metric.points": "Model points (exp.)",
    "metric.points_hint": "Expected compendium points for the model board",
    "metric.coverage": "Player rosters",
    "metric.coverage_hint_base": "share of corpus matches with rosters from details",
    "metric.coverage_with": "matches with roster",
    "metric.coverage_ok": "≥80% target — met",
    "metric.coverage_left": "~{n} left: details not downloaded (not anonymous)",
    "metric.coverage_title":
      "Coverage = share of corpus matches with player rosters from details. Now 100%: every corpus match_id has a shard with players.",
    "metric.coverage_note":
      "Below 100%: ~{n} matches lack downloaded details. ≥80% target; live goal is to close the rest via download_details.",
    "title.analyst_pts": "Expected board points from analyst consensus",
    "title.fusion_pts": "Expected points after mixing model, analysts, and market",
    "title.both_pts": "Expected points: analysts and mixed forecast match",
    "title.strength": "Shrunk Elo ± combined σ",
    "error.load": "Data load error: {msg}. Open via a local server or GitHub Pages.",
    "error.fetch": "Failed to load predictions.json ({status})",
    "rank.home": "home +",
    "rules.meta_title": "TI 2026 Swiss rules · informal summary",
    "rules.eyebrow": "The International 2026",
    "rules.title": "Group Stage rules",
    "rules.lead":
      "A short informal paraphrase of the official TI 2026 Swiss rules. Not a legal text and not a substitute for Valve’s original.",
    "rules.disclaimer_label": "Disclaimer",
    "rules.disclaimer":
      "Unofficial paraphrase for convenience. If anything conflicts, Valve’s official rules win. Author is not affiliated with Valve Corporation.",
    "rules.official_link": "Official rules on dota2.com",
    "rules.dates_title": "Dates",
    "rules.dates_body":
      "Group Stage: Aug 13–16 (online). Main Event: Aug 20–23, SPD Bank Oriental Sports Center, Shanghai.",
    "rules.rank_title": "How teams are ranked",
    "rules.rank_intro": "Tiebreakers in order:",
    "rules.rank_1": "Number of matches won",
    "rules.rank_2": "Number of matches lost",
    "rules.rank_3": "Total matches won by opponents played",
    "rules.rank_4": "Percentage of games (maps) won",
    "rules.rank_5": "Average map-win % of opponents played",
    "rules.rank_6": "Average game duration (shorter is better)",
    "rules.rank_7": "Coin toss",
    "rules.pair_title": "General Swiss pairing",
    "rules.pair_1": "Same-record teams are paired against each other",
    "rules.pair_2": "Avoid repeat pairings when possible",
    "rules.pair_3": "Minimize ranking distance between opponents when possible",
    "rules.round_title": "Round-by-round modifications",
    "rules.round_1":
      "Round 1: teams split into two groups; organizer sets matchups within the group.",
    "rules.round_2": "Rounds 2–3: pair only within the initial group.",
    "rules.round_3": "Round 4: pair only against the other group.",
    "rules.round_4":
      "Round 5 (elimination matches): maximize ranking distance between opponents.",
    "rules.elim_title": "Elimination Round",
    "rules.elim_body":
      "Best 3–2 team picks any of the five 2–3 teams. Next-best 3–2 picks from the remainder — repeat until all are paired.",
    "rules.seed_title": "Seeding into The International",
    "rules.seed_body":
      "All eight teams that qualify are seeded by their final Swiss ranking.",
    "rules.pick_title": "Selection priority (pick / side)",
    "rules.pick_bo3":
      "Bo3: coin toss. Winner chooses pick or side for game 1; loser for game 2. Game 3 (if any): new coin toss.",
    "rules.pick_bo5":
      "Bo5: games 1 & 3 — upper-bracket winner; 2 & 4 — lower-bracket winner; game 5 (if any) — coin toss.",
    "rules.trademark":
      "Dota and the Dota logo are trademarks of Valve Corporation. © Valve Corporation. All rights reserved.",
  },
};

// Clone EN as base for de/fr/pt/es, then override.
function cloneEn(overrides) {
  return { ...STRINGS.en, ...overrides };
}

STRINGS.de = cloneEn({
  "meta.title": "TI 2026 Swiss-Prognose",
  "meta.description": "Offene Prognose für die Gruppenphase von The International 2026",
  "brand.kicker": "Offene Prognose",
  "brand.title": "Swiss-Prognose",
  "nav.board": "Brett",
  "nav.standings": "Chancen",
  "nav.heatmap": "Slot-Karte",
  "nav.matchup": "Duell",
  "nav.model": "Modell",
  "nav.how": "Methode",
  "nav.rules": "Regeln",
  "nav.repo": "Repository",
  "nav.home": "Prognose",
  "lang.label": "Sprache",
  "hero.eyebrow": "The International 2026 · Gruppenphase",
  "hero.title": "Wer kommt aus dem Swiss weiter?",
  "hero.lead":
    "16 Teams, Swiss bis 4 Siege/Niederlagen, dann Elimination Round. Darunter das Prognosebrett.",
  "disclaimer.label": "Wichtig",
  "disclaimer.loading": "Laden…",
  "board.title": "Swiss-Brett",
  "board.lead":
    "Prognose der Compendium-Slots — kein offizielles Valve-Bracket. Strategien mischen Modell, Analysten oder Markt.",
  "board.strategy_label": "Brett-Strategie",
  "strategy.points_optimal": "Modell · max. Punkte",
  "strategy.qualify_rank": "Modell · nach Quali-Chance",
  "strategy.analyst_consensus": "Analysten-Konsens",
  "strategy.fusion": "Empfohlene Mischung",
  "strategy.fusion_model_heavy": "Mehr dem Modell vertrauen",
  "strategy.fusion_balanced": "Ausgewogene Quellen",
  "strategy.fusion_market_lean": "Mehr dem Markt vertrauen",
  "strategy.fusion_analyst_lean": "Mehr Analysten vertrauen",
  "fusion.weights_label": "Gewichte der Quellen",
  "fusion.weights_help":
    "Fertige Mischungen und Schieberegler. Werte sind unabhängig und müssen nicht 100 ergeben — beim Berechnen werden Anteile proportional genommen. Schieberegler rasten auf dem nächsten vorberechneten Szenario ein.",
  "fusion.preset.fusion": "Empfohlen",
  "fusion.preset.fusion_model_heavy": "Mehr Modell",
  "fusion.preset.fusion_balanced": "Ausgewogen",
  "fusion.preset.fusion_market_lean": "Mehr Markt",
  "fusion.preset.fusion_analyst_lean": "Mehr Analysten",
  "fusion.weight.model": "Modell",
  "fusion.weight.analyst": "Analysten",
  "fusion.weight.market": "Markt (anonym)",
  "fusion.weight.ranking": "Power-Ranking",
  "fusion.weight.expert": "Expertenhistorie",
  "fusion.weight_in_mix": "in der Mischung",
  "fusion.weights_raw_sum":
    "Summe der gewählten Gewichte: {sum} (muss nicht 100 sein). Darunter der Anteil nach proportionaler Umskalierung.",
  "fusion.score": "Erwartete Punkte dieses Szenarios",
  "fusion.correct_slots": "richtige Slots ≈",
  "standings.title": "Team-Chancen",
  "standings.lead":
    "μ ± σ — Stärke und Unsicherheit. Quali / Elim — Anteil der Swiss-Simulationen (~50.000).",
  "standings.sort_label": "Sortierung",
  "standings.sort.qualify": "Nach Quali-Chance",
  "standings.sort.power": "Nach Stärke",
  "standings.sort.elim": "Nach Elim-Chance",
  "standings.sort.alpha": "A–Z",
  "standings.search": "Team suchen…",
  "standings.th.team": "Team",
  "standings.th.region": "Region",
  "standings.th.strength": "Stärke μ±σ",
  "standings.th.qualify": "Quali",
  "standings.th.elim": "Elim",
  "standings.th.wins": "Erw. Siege",
  "standings.th.record": "Bilanz",
  "heatmap.title": "Slot-Karte",
  "heatmap.lead": "Anteil der Sims (%), in denen ein Team in jedem Compendium-Slot landete.",
  "heatmap.empty": "Keine Slot-Karten-Daten",
  "heatmap.team": "Team",
  "matchup.title": "Wer gegen wen",
  "matchup.lead": "Paarweise Siegchance (Pairwise-Blend oder Power-Ranking).",
  "matchup.team_a": "Team A",
  "matchup.team_b": "Team B",
  "matchup.series_win": "Serie-Siegchance",
  "model.title": "Modellstatus",
  "how.title": "So rechnen wir",
  "how.lead_before": "Die Zahlen auf der Seite in einfachen Worten. Details in",
  "how.lead_and": "und",
  "footer.open": "Offenes Projekt. Daten — OpenDota.",
  "footer.license": "MIT-Lizenz",
  "projects.title": "Andere Projekte",
  "projects.moex.desc": "MOEX-Kurse und Markowitz-Portfolio",
  "projects.keyboard.desc": "Hilft beim Absolvieren des Minispiels „Automatons Attack“ aus dem Event „Dark Carnival“ in Dota",
  "col.undefeated": "4–0 · ungeschlagen",
  "col.one_loss": "4–1 · eine Niederlage",
  "col.advance": "Weiter",
  "col.eliminate": "Ausgeschieden",
  "col.one_win": "1–4 · ein Sieg",
  "col.winless": "0–4 · sieglos",
  "board.empty": "Noch leer",
  "chip.points": "Punkte (erw.)",
  "chip.analysts": "Analysten",
  "chip.fusion": "Gemischt",
  "chip.sims": "Turnier-Sims",
  "footer.updated": "Aktualisiert",
  "rules.meta_title": "TI 2026 Swiss-Regeln · informelle Zusammenfassung",
  "rules.title": "Regeln der Gruppenphase",
  "rules.lead":
    "Kurze informelle Paraphrase der offiziellen TI-2026-Swiss-Regeln. Kein Rechtstext und kein Ersatz für das Valve-Original.",
  "rules.disclaimer_label": "Haftungsausschluss",
  "rules.disclaimer":
    "Inoffizielle Paraphrase. Bei Abweichungen gelten nur die offiziellen Valve-Regeln. Autor nicht mit Valve affiliated.",
  "rules.official_link": "Offizielle Regeln auf dota2.com",
  "rules.dates_title": "Termine",
  "rules.dates_body":
    "Gruppenphase: 13.–16. Aug (online). Hauptevent: 20.–23. Aug, SPD Bank Oriental Sports Center, Shanghai.",
  "rules.rank_title": "Rangfolge der Teams",
  "rules.rank_intro": "Kriterien der Reihe nach:",
  "rules.pair_title": "Allgemeine Swiss-Paarung",
  "rules.round_title": "Runden-Besonderheiten",
  "rules.elim_title": "Elimination Round",
  "rules.seed_title": "Seeding für The International",
  "rules.pick_title": "Auswahlpriorität (Pick / Seite)",
});

STRINGS.fr = cloneEn({
  "meta.title": "TI 2026 · Pronostic Swiss",
  "meta.description": "Pronostic ouvert pour la phase de groupes de The International 2026",
  "brand.kicker": "Pronostic ouvert",
  "brand.title": "Pronostic Swiss",
  "nav.board": "Tableau",
  "nav.standings": "Chances",
  "nav.heatmap": "Carte des slots",
  "nav.matchup": "Affrontement",
  "nav.model": "Modèle",
  "nav.how": "Méthode",
  "nav.rules": "Règles",
  "nav.repo": "Dépôt",
  "nav.home": "Pronostic",
  "lang.label": "Langue",
  "hero.eyebrow": "The International 2026 · Phase de groupes",
  "hero.title": "Qui sort du Swiss ?",
  "hero.lead":
    "16 équipes, Swiss jusqu’à 4 victoires/défaites, puis tour d’élimination. Ci-dessous — le tableau de pronostic.",
  "disclaimer.label": "Important",
  "disclaimer.loading": "Chargement…",
  "board.title": "Tableau Swiss",
  "board.strategy_label": "Stratégie du tableau",
  "strategy.points_optimal": "Modèle · max points",
  "strategy.qualify_rank": "Modèle · chance de qualifier",
  "strategy.analyst_consensus": "Consensus analystes",
  "strategy.fusion": "Mélange recommandé",
  "strategy.fusion_model_heavy": "Mixte · plus modèle",
  "strategy.fusion_balanced": "Mixte · équilibré",
  "strategy.fusion_market_lean": "Mixte · plus marché",
  "strategy.fusion_analyst_lean": "Mixte · plus analystes",
  "fusion.weights_label": "Poids du pronostic mixte",
  "standings.title": "Chances des équipes",
  "standings.sort_label": "Tri",
  "standings.sort.qualify": "Par chance de qualifier",
  "standings.sort.power": "Par force",
  "standings.sort.elim": "Par chance d’élimination",
  "standings.sort.alpha": "A–Z",
  "standings.search": "Trouver une équipe…",
  "standings.th.team": "Équipe",
  "standings.th.region": "Région",
  "standings.th.strength": "Force μ±σ",
  "standings.th.qualify": "Qualifier",
  "standings.th.elim": "Éliminé",
  "standings.th.wins": "Victoires esp.",
  "standings.th.record": "Bilan",
  "heatmap.title": "Carte des slots",
  "heatmap.team": "Équipe",
  "matchup.title": "Face à face",
  "matchup.team_a": "Équipe A",
  "matchup.team_b": "Équipe B",
  "matchup.series_win": "proba de gagner la série",
  "model.title": "État du modèle",
  "how.title": "Comment on calcule",
  "how.lead_before": "Les chiffres du site en mots simples. Détails dans",
  "how.lead_and": "et",
  "footer.open": "Projet ouvert. Données — OpenDota.",
  "footer.license": "Licence MIT",
  "projects.title": "Autres projets",
  "projects.moex.desc": "Cours MOEX et portefeuille Markowitz",
  "projects.keyboard.desc": "Aide à terminer le mini-jeu « Automatons Attack » de l'événement « Dark Carnival » dans Dota",
  "col.undefeated": "4–0 · invaincu",
  "col.one_loss": "4–1 · une défaite",
  "col.advance": "Qualification",
  "col.eliminate": "Élimination",
  "col.one_win": "1–4 · une victoire",
  "col.winless": "0–4 · sans victoire",
  "board.empty": "Vide pour l’instant",
  "chip.points": "Points (esp.)",
  "chip.analysts": "Analystes",
  "chip.fusion": "Mixte",
  "chip.sims": "Sims du tournoi",
  "footer.updated": "Mis à jour",
  "rules.meta_title": "Règles Swiss TI 2026 · résumé libre",
  "rules.title": "Règles de la phase de groupes",
  "rules.lead":
    "Court résumé libre des règles officielles Swiss TI 2026. Pas un texte juridique ni un substitut à l’original Valve.",
  "rules.disclaimer_label": "Avertissement",
  "rules.disclaimer":
    "Paraphrase non officielle. En cas de conflit, seules les règles officielles Valve font foi. Auteur non affilié à Valve.",
  "rules.official_link": "Règles officielles sur dota2.com",
  "rules.dates_title": "Dates",
  "rules.dates_body":
    "Phase de groupes : 13–16 août (en ligne). Événement principal : 20–23 août, SPD Bank Oriental Sports Center, Shanghai.",
  "rules.rank_title": "Classement des équipes",
  "rules.pair_title": "Appariements Swiss généraux",
  "rules.round_title": "Particularités par tour",
  "rules.elim_title": "Tour d’élimination",
  "rules.seed_title": "Seed pour The International",
  "rules.pick_title": "Priorité de sélection (pick / côté)",
});

STRINGS.pt = cloneEn({
  "meta.title": "TI 2026 · Previsão Swiss",
  "meta.description": "Previsão aberta da fase de grupos do The International 2026",
  "brand.kicker": "Previsão aberta",
  "brand.title": "Previsão Swiss",
  "nav.board": "Painel",
  "nav.standings": "Chances",
  "nav.heatmap": "Mapa de slots",
  "nav.matchup": "Confronto",
  "nav.model": "Modelo",
  "nav.how": "Como calculamos",
  "nav.rules": "Regras",
  "nav.repo": "Repositório",
  "nav.home": "Previsão",
  "lang.label": "Idioma",
  "hero.eyebrow": "The International 2026 · Fase de grupos",
  "hero.title": "Quem avança no Swiss?",
  "hero.lead":
    "16 times, Swiss até 4 vitórias/derrotas, depois rodada de eliminação. Abaixo — o painel de previsão.",
  "disclaimer.label": "Importante",
  "disclaimer.loading": "Carregando…",
  "board.title": "Painel Swiss",
  "board.strategy_label": "Estratégia do painel",
  "strategy.points_optimal": "Modelo · máx. pontos",
  "strategy.qualify_rank": "Modelo · chance de classificar",
  "strategy.analyst_consensus": "Consenso de analistas",
  "strategy.fusion": "Mistura recomendada",
  "strategy.fusion_model_heavy": "Misto · mais modelo",
  "strategy.fusion_balanced": "Misto · equilibrado",
  "strategy.fusion_market_lean": "Misto · mais mercado",
  "strategy.fusion_analyst_lean": "Misto · mais analistas",
  "fusion.weights_label": "Pesos da previsão mista",
  "standings.title": "Chances das equipes",
  "standings.sort_label": "Ordenar",
  "standings.sort.qualify": "Por chance de classificar",
  "standings.sort.power": "Por força",
  "standings.sort.elim": "Por chance de eliminação",
  "standings.sort.alpha": "A–Z",
  "standings.search": "Buscar equipe…",
  "standings.th.team": "Equipe",
  "standings.th.region": "Região",
  "standings.th.strength": "Força μ±σ",
  "standings.th.qualify": "Classificar",
  "standings.th.elim": "Eliminado",
  "standings.th.wins": "Vitórias esp.",
  "standings.th.record": "Recorde",
  "heatmap.title": "Mapa de slots",
  "heatmap.team": "Equipe",
  "matchup.title": "Confronto",
  "matchup.team_a": "Equipe A",
  "matchup.team_b": "Equipe B",
  "matchup.series_win": "probabilidade de ganhar a série",
  "model.title": "Estado do modelo",
  "how.title": "Como calculamos",
  "how.lead_before": "Os números do site em linguagem simples. Detalhes em",
  "how.lead_and": "e",
  "footer.open": "Projeto aberto. Dados — OpenDota.",
  "footer.license": "Licença MIT",
  "projects.title": "Outros projetos",
  "projects.moex.desc": "Cotações MOEX e carteira Markowitz",
  "projects.keyboard.desc": "Ajuda a concluir o minigame Automatons Attack do evento Dark Carnival em Dota",
  "col.undefeated": "4–0 · invicto",
  "col.one_loss": "4–1 · uma derrota",
  "col.advance": "Avanço",
  "col.eliminate": "Eliminação",
  "col.one_win": "1–4 · uma vitória",
  "col.winless": "0–4 · sem vitórias",
  "board.empty": "Vazio por enquanto",
  "chip.points": "Pontos (esp.)",
  "chip.analysts": "Analistas",
  "chip.fusion": "Misto",
  "chip.sims": "Sims do torneio",
  "footer.updated": "Atualizado",
  "rules.meta_title": "Regras Swiss TI 2026 · resumo informal",
  "rules.title": "Regras da fase de grupos",
  "rules.lead":
    "Resumo informal das regras oficiais Swiss do TI 2026. Não é texto jurídico nem substitui o original da Valve.",
  "rules.disclaimer_label": "Aviso",
  "rules.disclaimer":
    "Paráfrase não oficial. Em conflito, valem só as regras oficiais da Valve. Autor não afiliado à Valve.",
  "rules.official_link": "Regras oficiais em dota2.com",
  "rules.dates_title": "Datas",
  "rules.dates_body":
    "Fase de grupos: 13–16 ago (online). Evento principal: 20–23 ago, SPD Bank Oriental Sports Center, Xangai.",
  "rules.rank_title": "Como as equipes são ranqueadas",
  "rules.pair_title": "Emparelhamento Swiss geral",
  "rules.round_title": "Modificações por rodada",
  "rules.elim_title": "Rodada de eliminação",
  "rules.seed_title": "Seed para The International",
  "rules.pick_title": "Prioridade de seleção (pick / lado)",
});

STRINGS.es = cloneEn({
  "meta.title": "TI 2026 · Pronóstico Swiss",
  "meta.description": "Pronóstico abierto de la fase de grupos de The International 2026",
  "brand.kicker": "Pronóstico abierto",
  "brand.title": "Pronóstico Swiss",
  "nav.board": "Tablero",
  "nav.standings": "Probabilidades",
  "nav.heatmap": "Mapa de slots",
  "nav.matchup": "Enfrentamiento",
  "nav.model": "Modelo",
  "nav.how": "Cómo calculamos",
  "nav.rules": "Reglas",
  "nav.repo": "Repositorio",
  "nav.home": "Pronóstico",
  "lang.label": "Idioma",
  "hero.eyebrow": "The International 2026 · Fase de grupos",
  "hero.title": "¿Quién avanza del Swiss?",
  "hero.lead":
    "16 equipos, Swiss hasta 4 victorias/derrotas, luego ronda de eliminación. Abajo — el tablero de pronóstico.",
  "disclaimer.label": "Importante",
  "disclaimer.loading": "Cargando…",
  "board.title": "Tablero Swiss",
  "board.strategy_label": "Estrategia del tablero",
  "strategy.points_optimal": "Modelo · máx. puntos",
  "strategy.qualify_rank": "Modelo · por chance de clasificar",
  "strategy.analyst_consensus": "Consenso de analistas",
  "strategy.fusion": "Mezcla recomendada",
  "strategy.fusion_model_heavy": "Mixto · más modelo",
  "strategy.fusion_balanced": "Mixto · equilibrado",
  "strategy.fusion_market_lean": "Mixto · más mercado",
  "strategy.fusion_analyst_lean": "Mixto · más analistas",
  "fusion.weights_label": "Pesos del pronóstico mixto",
  "standings.title": "Probabilidades de equipos",
  "standings.sort_label": "Ordenar",
  "standings.sort.qualify": "Por chance de clasificar",
  "standings.sort.power": "Por fuerza",
  "standings.sort.elim": "Por chance de eliminación",
  "standings.sort.alpha": "A–Z",
  "standings.search": "Buscar equipo…",
  "standings.th.team": "Equipo",
  "standings.th.region": "Región",
  "standings.th.strength": "Fuerza μ±σ",
  "standings.th.qualify": "Clasificar",
  "standings.th.elim": "Eliminado",
  "standings.th.wins": "Victorias esp.",
  "standings.th.record": "Récord",
  "heatmap.title": "Mapa de slots",
  "heatmap.team": "Equipo",
  "matchup.title": "Cara a cara",
  "matchup.team_a": "Equipo A",
  "matchup.team_b": "Equipo B",
  "matchup.series_win": "probabilidad de ganar la serie",
  "model.title": "Estado del modelo",
  "how.title": "Cómo calculamos",
  "how.lead_before": "Los números del sitio en lenguaje simple. Detalles en",
  "how.lead_and": "y",
  "footer.open": "Proyecto abierto. Datos — OpenDota.",
  "footer.license": "Licencia MIT",
  "projects.title": "Otros proyectos",
  "projects.moex.desc": "Cotizaciones MOEX y cartera Markowitz",
  "projects.keyboard.desc": "Ayuda a completar el minijuego Automatons Attack del evento Dark Carnival en Dota",
  "col.undefeated": "4–0 · invicto",
  "col.one_loss": "4–1 · una derrota",
  "col.advance": "Avance",
  "col.eliminate": "Eliminación",
  "col.one_win": "1–4 · una victoria",
  "col.winless": "0–4 · sin victorias",
  "board.empty": "Vacío por ahora",
  "chip.points": "Puntos (esp.)",
  "chip.analysts": "Analistas",
  "chip.fusion": "Mixto",
  "chip.sims": "Sims del torneo",
  "footer.updated": "Actualizado",
  "rules.meta_title": "Reglas Swiss TI 2026 · resumen informal",
  "rules.title": "Reglas de la fase de grupos",
  "rules.lead":
    "Resumen informal de las reglas oficiales Swiss del TI 2026. No es texto legal ni sustituye el original de Valve.",
  "rules.disclaimer_label": "Aviso",
  "rules.disclaimer":
    "Paráfrasis no oficial. Si hay conflicto, priman las reglas oficiales de Valve. Autor no afiliado a Valve.",
  "rules.official_link": "Reglas oficiales en dota2.com",
  "rules.dates_title": "Fechas",
  "rules.dates_body":
    "Fase de grupos: 13–16 ago (online). Evento principal: 20–23 ago, SPD Bank Oriental Sports Center, Shanghái.",
  "rules.rank_title": "Cómo se clasifican los equipos",
  "rules.pair_title": "Emparejamiento Swiss general",
  "rules.round_title": "Modificaciones por ronda",
  "rules.elim_title": "Ronda de eliminación",
  "rules.seed_title": "Seed para The International",
  "rules.pick_title": "Prioridad de selección (pick / lado)",
});

/** @type {string} */
let currentLang = "ru";
/** @type {Set<(lang: string) => void>} */
const listeners = new Set();

function detectLang() {
  const saved = localStorage.getItem(LANG_STORAGE_KEY);
  if (saved && SUPPORTED_LANGS.includes(saved)) return saved;
  const nav = (navigator.language || "en").slice(0, 2).toLowerCase();
  if (SUPPORTED_LANGS.includes(nav)) return nav;
  return "ru";
}

/** Current UI language code. */
export function getLang() {
  return currentLang;
}

/** BCP-47 tag for Number/Date formatting. */
export function localeTag() {
  return LOCALE_TAGS[currentLang] || "en-US";
}

/**
 * Translate a key; optional `{name}` placeholders via vars.
 * @param {string} key
 * @param {Record<string, string|number>} [vars]
 */
export function t(key, vars) {
  const dict = STRINGS[currentLang] || STRINGS.en;
  let s = dict[key] ?? STRINGS.en[key] ?? key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      s = s.replaceAll(`{${k}}`, String(v));
    }
  }
  return s;
}

/** Apply data-i18n* attributes under root. */
export function applyDom(root = document) {
  root.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (key) el.textContent = t(key);
  });
  root.querySelectorAll("[data-i18n-html]").forEach((el) => {
    const key = el.getAttribute("data-i18n-html");
    if (key) el.innerHTML = t(key);
  });
  root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const key = el.getAttribute("data-i18n-placeholder");
    if (key && "placeholder" in el) el.placeholder = t(key);
  });
  root.querySelectorAll("[data-i18n-title]").forEach((el) => {
    const key = el.getAttribute("data-i18n-title");
    if (key) el.title = t(key);
  });
  root.querySelectorAll("[data-i18n-aria]").forEach((el) => {
    const key = el.getAttribute("data-i18n-aria");
    if (key) el.setAttribute("aria-label", t(key));
  });
  const titleEl = document.querySelector("title[data-i18n]");
  if (titleEl) {
    const key = titleEl.getAttribute("data-i18n");
    if (key) document.title = t(key);
  }
  const desc = document.querySelector('meta[name="description"][data-i18n]');
  if (desc) {
    const key = desc.getAttribute("data-i18n");
    if (key) desc.setAttribute("content", t(key));
  }
  document.documentElement.lang = currentLang;
}

/**
 * @param {string} lang
 * @param {{ persist?: boolean }} [opts]
 */
export function setLang(lang, opts = {}) {
  const next = SUPPORTED_LANGS.includes(lang) ? lang : "en";
  currentLang = next;
  if (opts.persist !== false) localStorage.setItem(LANG_STORAGE_KEY, next);
  applyDom();
  syncLangSwitcher(next);
  listeners.forEach((fn) => fn(next));
}

/** @param {(lang: string) => void} fn */
export function onLangChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Native language names + flag image paths (not emoji — Windows select issue). */
export const LANG_OPTIONS = [
  { code: "ru", label: "Русский", flag: "assets/img/flags/ru.svg" },
  { code: "en", label: "English", flag: "assets/img/flags/en.svg" },
  { code: "de", label: "Deutsch", flag: "assets/img/flags/de.svg" },
  { code: "fr", label: "Français", flag: "assets/img/flags/fr.svg" },
  { code: "pt", label: "Português", flag: "assets/img/flags/pt.svg" },
  { code: "es", label: "Español", flag: "assets/img/flags/es.svg" },
];

const WARNING_RULES = [
  {
    id: "market_prior_power_rankings",
    match: /Market prior seeded from POWER_RANKINGS|market_weight forced to 0/i,
  },
  {
    id: "in_sample_tune_model_weight",
    match: /In-sample tune suggested model_weight[= ]*([0-9.]+).*production default[= ]*([0-9.]+)/i,
  },
];

/**
 * Map export warning English text → localized UI copy.
 * @param {string} raw
 */
export function localizeWarning(raw) {
  const text = String(raw || "").trim();
  if (!text) return "";
  for (const rule of WARNING_RULES) {
    const m = text.match(rule.match);
    if (!m) continue;
    if (rule.id === "in_sample_tune_model_weight") {
      return t("warn.in_sample_tune_model_weight", {
        tuned: m[1] || "0.25",
        production: m[2] || "0.65",
      });
    }
    return t(`warn.${rule.id}`);
  }
  // Don't dump raw English into non-EN UI
  if (currentLang === "en") return text;
  return t("warn.unknown");
}

function syncLangSwitcher(lang) {
  const root = document.getElementById("lang-switcher");
  if (!root) return;
  const opt = LANG_OPTIONS.find((o) => o.code === lang) || LANG_OPTIONS[0];
  const flag = document.getElementById("lang-switch-flag");
  const text = document.getElementById("lang-switch-text");
  const btn = document.getElementById("lang-switch-btn");
  if (flag) {
    flag.src = opt.flag;
    flag.alt = "";
  }
  if (text) text.textContent = opt.label;
  if (btn) btn.setAttribute("aria-label", `${t("lang.label")}: ${opt.label}`);
  root.querySelectorAll(".lang-switch-option").forEach((el) => {
    el.setAttribute("aria-selected", el.getAttribute("data-lang") === lang ? "true" : "false");
  });
}

function closeLangSwitcher() {
  const root = document.getElementById("lang-switcher");
  const btn = document.getElementById("lang-switch-btn");
  const list = document.getElementById("lang-switch-list");
  if (!root || !btn || !list) return;
  root.classList.remove("is-open");
  btn.setAttribute("aria-expanded", "false");
  list.hidden = true;
}

function openLangSwitcher() {
  const root = document.getElementById("lang-switcher");
  const btn = document.getElementById("lang-switch-btn");
  const list = document.getElementById("lang-switch-list");
  if (!root || !btn || !list) return;
  root.classList.add("is-open");
  btn.setAttribute("aria-expanded", "true");
  list.hidden = false;
}

/** Mount custom language dropdown (SVG flags) if present. */
export function initI18n() {
  currentLang = detectLang();
  const root = document.getElementById("lang-switcher");
  const btn = document.getElementById("lang-switch-btn");
  const list = document.getElementById("lang-switch-list");
  if (root && btn && list) {
    list.innerHTML = LANG_OPTIONS.map((o) => {
      const selected = o.code === currentLang ? "true" : "false";
      return `<li role="option">
        <button type="button" class="lang-switch-option" data-lang="${o.code}" aria-selected="${selected}">
          <img class="lang-flag" src="${o.flag}" alt="" width="20" height="14" decoding="async" />
          <span>${o.label}</span>
        </button>
      </li>`;
    }).join("");

    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (list.hidden) openLangSwitcher();
      else closeLangSwitcher();
    });
    list.querySelectorAll("[data-lang]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const code = el.getAttribute("data-lang");
        closeLangSwitcher();
        if (code) setLang(code);
      });
    });
    document.addEventListener("click", (e) => {
      if (!root.contains(/** @type {Node} */ (e.target))) closeLangSwitcher();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeLangSwitcher();
    });
    syncLangSwitcher(currentLang);
  }
  applyDom();
}
