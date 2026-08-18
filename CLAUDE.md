# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Python-инструмент для прогнозирования биржевых котировок (Московская биржа, MOEX).

**Два параллельных направления:**

**Направление A (фазы 1-6) — LP-пайплайн:** LP-декомпозиция ряда на att (Local Projective filter) + noise; прогноз att через LWR с каскадным поиском соседей и acc_ang. Стек: **logtrend → LP-фильтр → att-прогноз (LWR + p-aligned каскад + acc_ang λ=0.01)**.

**Направление B (фаза 7) — Событийная фрактальность зигзага:** крупные события зигзага (T_big) строятся из меньших (T_frac). Прогноз: поиск k-NN в пуле мелких событий → аппроксимация следующего крупного. ⚠️ Это НЕ аттрактор динамических систем — FNN/D₂/DET к зигзагу не применимы как теоретическое основание. Метод пересекается с нелинейной динамикой только в концепции поиска соседа.

Владелец имеет опыт написания торговых плагинов на MQL5.

## Ключевые методические решения

- **Нормализация — logtrend (causal OLS)**: `ratio = close / exp(a + b·t)`, коэффициенты вычисляются через инкрементальные суммы O(N). Нет разгрева, нет параметра окна. Даёт −2.4% MAPE vs SMA(1000) на 8 тикерах 1d (скр.41).
  - **⚠️ Все новые скрипты используют logtrend.** Скрипты phase3 (30-40) работали на SMA(1000) — их числа не смешивать с новыми без перепроверки.
- **Filter bank (causal Butterworth sosfilt)**: разложение dratio на 6 компонент C0-C5 (периоды ~2-4, 4-8, 8-16, 16-52, 52-103, 103+ баров). Аттрактор вскрывается в C2-C5. **C2-C5 нельзя разделять** — разделение ломает фазовые отношения аттрактора. Поэтому перешли на LP-фильтр (Wn=0.125, граница C1/C2), который сохраняет C2-C5 как единый сигнал att.
- **LWR (Locally Weighted Regression)**: МНК с Гауссовыми весами по расстоянию; p=20, ξ=63 (правило: ξ ≥ 3(p+1)); правило p ≥ T_min/4.
- **Реконструкция цены**: `dratio_hat → ratio_hat = ratio[origin] + cumsum(dratio_hat) → price = ratio_hat × logtrend[origin]`. Всегда однократный cumsum.
- **Источник данных** — [MOEX ISS API](https://iss.moex.com/iss/reference/) (бесплатный, задержка 15 мин).

## Исследовательская база (три завершённые фазы)

### Фаза 1 — val_mape gate (`research/phase1_hurst_gate/`, скр. 01-07)
Hurst не предсказывает качество прогноза (r≈0). Сильный предиктор — val_mape: порог 0.0043, разница 2.2× между хорошими и плохими прогнозами. Per-ticker пороги и взвешивание отклонены — глобальный порог оптимален.

### Фаза 2 — хаотичность и встраивание (`research/phase2_chaos_embed/`, скр. 08-29)
Δratio стохастичен (FNN не убывает), ratio имеет аттрактор при p≈4. DET — слабый гейт (r≈0.13). EMD (28-29) — нестабильна. Лучший результат фазы: GMM/PCA-кластеризация (26) — незначимо. Итог: встраивание и фазовое пространство не улучшают LA напрямую.

### Фаза 3 — filter bank и нормализация (`research/phase3_filterbank/`, скр. 30-41)
- **Filter bank** (скр. 30-34): causal Butterworth на dratio, C3-C5, p=20 → **−22% MAPE** от raw dratio (baseline 0.027 → 0.021).
- **AR-whitening / d²ratio** (скр. 35-38): Hamilton(h=1) ≡ d²ratio + однокр. cumsum → −33% суммарно, но за счёт мартингала (предсказывает «ноль»), не паттерн-матчинга.
- **Нормализация** (скр. 39-41): z-score (+47× slow std), per-comp norm (657× усиление C5) — провалились. **logtrend causal OLS → −2.4%, p=0.0000** ⭐ — принят как стандарт.
- Накопленный результат: logtrend + filter bank C3-C5 + LWR ≈ **−24% от raw dratio baseline**.

## Итог фазы 4 и парадигма фазы 5

### Фаза 4 — компонентный анализ (`research/phase4_components/`, скр. 42-80)
- Скр. 42-74: диагностика C0-C5, LP-аттрактор vs filter bank, KF2D, SSA.
- Скр. 75-80: **Local Projective filter** (m=9 d=3 k=30 n=3) — стандарт att-фильтра. **acc_ang** (d_pos + 0.01·d_ang(acc)) → −10.8% rMAE на 8 тикерах 1d.
- Итог: LP att + каскадный LWR + acc_ang = −10.8% vs pos_only baseline.

### Фаза 5 — исследование att (`research/phase5_attractor/`, скр. 81-99) — ЗАВЕРШЕНА
- global_acc_ang, adaptive λ, temporal decay — **отклонены**
- Oracle per-origin p_fit: потолок **−53%** rMAE — аттрактор многомасштабен
- Ensemble 1/LOO (p-aligned каскад): **−12.41%** — лучший реализуемый метод
- Слепое применение теорформул (TwoNN, DET, Такенс) не работает как предиктор качества

### Фаза 6 — адаптивные методы (`research/phase6_attractor/`, скр. 100+) — ТЕКУЩАЯ (направление A)
- app3: LP офлайн + p-aligned каскад + global_blend; утечка LP выявлена
- LB-гейт d≥LB: 34.7% улучшение при m=3d — первый рабочий per-origin дискриминатор
- Стартовые настройки: n_iter=2, d=11, m=33, p_fit=33 (эксп.101)
- Открытое направление: rolling LP (устранение нонкаузальности), per-origin p_fit

### Фаза 7 — событийная фрактальность (`research/phase7_fractality/`) — ТЕКУЩАЯ (направление B)
**Ключевая концепция:** большие зигзаг-события строятся из меньших (иерархическая фрактальная структура). Референс: `research/reference/zigzag_forecast_ref.py` (4-way ансамбль, rMAE=0.3911).
- Установлено (2026-06-28): пул ТОЛЬКО из T_frac событий бьёт baseline при T_frac ≈ 0.85 × T_big
- T_BIG=4%: оптимум T_frac=3%, rMAE=0.402 (−3.4% vs baseline LWR); T_BIG=2%: T_frac=1.7%, rMAE=0.394 (−6.9%)
- Оптимальный ratio пула: ~1.4–1.7× (не 7× как T/7-правило из комбинированного пула)
- Новые скрипты в `research/phase7_fractality/`

## Протокол проведения экспериментов

### Шаг 1 — Согласование перед запуском

Перед написанием кода любого нового исследовательского скрипта:

1. **Согласовать ограничения и условия** — явно перечислить все ключевые параметры (диапазоны переменных, сетки значений, пороги, метрики) и спросить у пользователя, правильно ли они выбраны. Не принимать значения «по умолчанию» без явного обсуждения.

2. **Разъяснить концепции и обозначения** — объяснить, что означает каждая ключевая переменная эксперимента. Пользователь не всегда отслеживает детали реализации, поэтому аббревиатуры и технические термины расшифровывать при введении.

3. **Провести критику постановки и согласовать** — перед запуском написать самостоятельный критический разбор: какие предположения сделаны, где может быть смещение, что остаётся непроверенным. Показать критику пользователю и получить явное согласие на запуск.

> **Почему это важно:** скр.88 и скр.94 были запущены с P_REG_GRID=[4..16], тогда как TwoNN давал d∈[32..79] → p≥65..159. Oracle обрезан на 16, snap уничтожал информацию, вывод «r≈0 — не артефакт» оказался преждевременным.

### Шаг 2 — Структура каталога эксперимента

Каждый эксперимент размещается в отдельном подкаталоге внутри `research/phaseN_*/`:

```
research/phaseN_<тема>/
  <NN>_<название>/          ← подкаталог эксперимента
    README.md               ← полная документация
    <NN>_<скрипт>.py        ← исследовательские скрипты
    results/                ← логи, csv, json с выходными данными
    figures/                ← рисунки
    data/                   ← рабочие входные данные (если нужны локально)
```

Глобальный `research/figures/` для новых экспериментов не используется — все файлы хранятся внутри подкаталога.

### Шаг 3 — Обязательная структура README.md

Каждый `README.md` содержит восемь разделов:

1. **Гипотезы** — чётко сформулированные проверяемые утверждения.
2. **Концепции, терминология, обозначения** — расшифровка всех аббревиатур и переменных, используемых в скриптах и результатах.
3. **Методология и протокол** — как проводится эксперимент, walk-forward схема, метрики оценки.
4. **Параметры** — полный список значений, с которыми скрипт запущен (сетки, пороги, гиперпараметры).
5. **Критика постановки** — что могло быть измерено неточно, какие предположения сделаны, где возможно смещение.
6. **Результаты** — таблицы, ссылки на рисунки и файлы в `results/`.
7. **Анализ и выводы** — интерпретация результатов, подтверждены или отклонены гипотезы.
8. **Дальнейшие направления** — что исследовать следующим, альтернативные гипотезы, открытые вопросы.

### Шаг 4 — Требования к выходным данным

Выходные данные в `results/` должны:
- содержать все условия запуска (параметры, дату, тикеры, интервал);
- обеспечивать полноту: все промежуточные метрики, не только итоговые;
- быть пригодны для повторного анализа и проверки косвенно связанных гипотез без перезапуска скрипта.

### Шаг 5 — Docker для долгих вычислений

Если ожидаемое время выполнения скрипта превышает **20 минут** — создать самодостаточный Docker-контейнер в каталоге эксперимента.

Требования к контейнеру:
1. При запуске немедленно выводить идентификатор эксперимента и версию образа.
2. Запускать скрипт дважды автоматически:
   - **Тестовый прогон** — 2 тикера, 5 origins, сокращённая сетка параметров. Завершается за 1-2 минуты. Подтверждает, что скрипт не сломан.
   - **Рабочий прогон** — полные данные.
3. Выходные данные писать на хост **онлайн** (flush после каждого тикера/origin), чтобы получить частичные результаты при краше.
4. Bash-скрипты запуска не нужны. В `README.md` эксперимента указать путь и команду запуска (`docker build` + `docker run`).

## Принципы модульности

Поддерживать чистое разделение ответственности — как на бэкенде, так и на фронтенде. При добавлении новой функциональности:

- **Не складывать всё в существующий файл**, если логика образует отдельную область (новый источник данных, новый алгоритм, новая группа UI-функций).
- **Выносить в отдельный модуль**, когда: файл превышает ~300–400 строк, появляется повторяющаяся тематика, или новая фича слабо связана с текущим модулем.
- На **бэкенде**: новые роуты — в `sma/api/routes/`, новые алгоритмы — в `sma/core/forecast/`, новые метрики — в `sma/core/metrics/`.
- На **фронтенде**: JS использует ES-модули (`type="module"`). Новые смысловые блоки выносить в отдельные `.js`-файлы в `sma/ui/`. Функции, вызываемые из HTML-атрибутов (`onclick`, `onchange`), регистрировать через `window.xxx = fn` в `app.js`.

## Структура репозитория

```
data/
  candles/          — кэш свечей (TICKER/interval.json)

docs/
  moex-api/         — справочник MOEX ISS API
  *.pdf             — лекции и учебные материалы

prototype/          — Streamlit-прототип; используется для экспериментов с новыми методами прогнозирования
  forcaster/        — оригинальный пакет прототипа
  scripts/          — бенчмарки и walk-forward анализ

research/           — исследовательские скрипты по фазам
  phase1_hurst_gate/   — скр. 01-07: val_mape gate, Hurst
  phase2_chaos_embed/  — скр. 08-29: хаотичность, FNN, PE, EMD, кластеризация
  phase3_filterbank/   — скр. 30-41: filter bank, AR-whitening, logtrend (SMA-нормализация!)
  phase4_components/   — скр. 42-80: компонентный анализ, LP-аттрактор, acc_ang
  phase5_attractor/    — скр. 81+: исследование аттрактора att (00_summary.md — сводка)
  figures/             — графики
  results/             — числовые результаты

sma/                — основное приложение (FastAPI + HTML/JS)
  api/              — FastAPI приложение
    app.py          — точка входа; lifespan: init_db + TaskManager
    deps.py         — DB_PATH, get_db(), get_task_manager()
    task_manager.py — очередь задач (forecast/candle_fetch/calibration/pretest), WebSocket broadcast
    routes/
      instruments.py      — GET/POST /instruments (added_via manual|pool), /search, /board
      candles.py           — POST /candles/fetch, GET /candles
      forecasts.py         — POST /forecasts (202), GET /forecasts/{id}, GET /forecasts,
                              GET /forecasts/zigzag, POST /forecasts/{id}/geometry
      forecast_settings.py — GET /forecast-settings, POST /calibrate, POST /pretest,
                              POST /{id}/activate, DELETE /{id}
      display_presets.py   — GET/POST /display-presets, DELETE /{id}
      tasks.py              — GET /tasks/{id}, WS /tasks/{id}/ws
      series.py             — POST /series/spectrogram (STFT-анализатор вкладки «Анализ»)
      settings.py            — GET/POST /settings (глобальные тюнинги: moex_pool_workers,
                                calibration_workers — левая панель «Настройки приложения»)
  core/             — алгоритмическое ядро
    db.py           — async SQLite (aiosqlite): instruments, candles, forecasts,
                      forecast_settings, display_presets, tasks
    models.py       — Pydantic-модели общего назначения (CandleBar)
    candle_fetch.py — resolve_fetch_plan (full/incremental), queue_pool_candle_fetch
    forecast/       — модули алгоритмов
      normalize.py         — normalize(candles) -> DataFrame[...,trend,ratio]; trend — каузальный
                              logtrend OLS (см. «Ключевые методические решения»)
      pool_selection.py      — автоотбор пула калибровки: категория → кандидаты MOEX →
                                ранжирование по ликвидности → топ-N (resolve_pool_candidates)
      band_lambda.py         — движок band_lambda: build_zigzag (4-tuple, extreme+confirm dates),
                                bar/pivot-native ranks, forecast_live_band, pool_values_and_weights
      band_lambda_calibrator.py — многопроходный координатный спуск по λ (calibrate_combined_multipass),
                                    ProcessPoolExecutor entry points (init_worker_env, calibrate_one_target)
    analysis/         — анализаторы вкладки «Анализ» (расширяемый набор, свой файл на анализатор)
      spectrogram.py         — compute_spectrogram: STFT каузального Δratio (scipy.signal.spectrogram),
                                filter_bank_cutoffs (частоты среза C0-C5 для reference-линий)
  data/             — источники данных
    moex/
      candles.py     — download_candles(), save_candles(), DATA_SOURCE="moex"
      securities.py  — search_securities(), list_board_securities(), get_security_history_range()
      __main__.py    — CLI: python -m sma.data.moex SBER --intervals 1d
  ui/               — фронтенд (HTML + Plotly.js + ES-модули)
    index.html
    style.css
    app.js          — точка входа: свечи, инструменты, вкладки, init, window.*
    state.js        — глобальный объект S, константы (INTERVAL_SECONDS, …)
    api.js          — fetch-обёртка api(), статусбар (setStatus/setBusy/setIdle)
    chart.js        — рендеринг Plotly: renderChart, renderBandForecast, зигзаг-трейс,
                      buildBandZoneShapes (editable rect'ы шаг1/шаг2), buildSubpanelTraces
                      (Heatmap спектрограммы на yaxis2), initChartEvents (клик по свече/маркеру)
    analysis.js     — вкладка «Анализ»: анализаторы осцилляторной панели (сейчас — спектрограмма
                      Δratio: calculateSpectrogram, applySpectrogramContrast, toggleSpectrogramDisplay)
    forecast.js     — вкладка «Прогноз»: T-селектор, калибровка/претест, submitForecast,
                      WS, история, драг-геометрия полос, пресеты отображения

tests/
  benchmark_workers.py — Docker-бенчмарк количества воркеров ProcessPoolExecutor
  Dockerfile
  run.bat

run.sh              — запуск приложения (uvicorn)
requirements.txt
```

## Окружение

venv находится вне общей папки (vboxsf не поддерживает симлинки):

```bash
# создать (один раз)
python3 -m venv /home/kali/.venvs/sma
/home/kali/.venvs/sma/bin/pip install -r requirements.txt

# активировать
source /home/kali/.venvs/sma/bin/activate
```

## Команды

```bash
# Запустить приложение (из корня проекта)
./run.sh
# или вручную:
uvicorn sma.api.app:app --reload --host 0.0.0.0 --port 8000

# CLI: скачать историю SBER интервал 1d
python -m sma.data.moex SBER --intervals 1d

# CLI: несколько тикеров/интервалов
python -m sma.data.moex SBER GAZP --intervals 1h 10m

# CLI: помощь
python -m sma.data.moex --help
```

Данные сохраняются в `data/candles/{TICKER}/{interval}.json`.

## Архитектура

### База данных (SQLite через aiosqlite)

Таблицы: `instruments`, `candles`, `forecasts`, `forecast_settings`, `display_presets`, `tasks`, `app_settings`.

- `instruments` — тикер + источник данных (`data_source`) + тип актива (`asset_type`) + `added_via`
  (`manual` — добавлен пользователем, `pool` — автоотобран калибровкой; `GET /instruments` по умолчанию
  отдаёт только `manual`, `include_pool=true` — все)
- `candles` — `ON CONFLICT DO UPDATE` (сохраняет row ID → FK прогнозов не ломаются при обновлении)
- `forecasts` — одна обобщённая таблица с дискриминатором `model_type`; `origin_candle_id` — жёсткая
  привязка к бару **подтверждения** пивота (каузальная точка); `result_json` иммутабелен после записи;
  `zone_geometry_json` — единственное редактируемое поле (растяжка полос мышью)
- `forecast_settings` — результат калибровки (λ, m, θ, состав пула) на
  (instrument_id, interval, model_type, t_query, pool_key); несколько `pool_key` могут сосуществовать
  для одного T (сравнение составов пула), `is_active` — какой из них используется вживую
- `display_presets` — именованные визуальные настройки (уровни/прозрачность), не привязаны к прогнозу
- `tasks` — очередь задач: pending → running → done/error; `kind` = forecast|candle_fetch|calibration|pretest
- `app_settings` — singleton-строка (id=1) глобальных тюнингов (параллельность MOEX-запросов при
  отборе пула, число процессов калибровки; 0 = авто); левая панель «Настройки приложения»

### TaskManager

Один asyncio-воркер обрабатывает задачи последовательно. Прогноз запускается в `run_in_executor` (ProcessPoolExecutor). Прогресс пробрасывается из потока в event loop через `loop.call_soon_threadsafe`. Подписчики получают события через `asyncio.Queue`.

### Воркеры ProcessPoolExecutor

Количество воркеров: `max(1, os.cpu_count() // 4)` — numpy-операции memory-bandwidth-bound; более агрессивные значения приводят к деградации производительности (проверено бенчмарком на 16-ядерной машине).

### Прогноз band_lambda: точка отсчёта (origin) и каузальность

Origin — **пивот зигзага целевого T**, не произвольный бар. Пивот на графике рисуется в дату
**экстремума** (`extreme_date` — где реально была цена), но origin прогноза — дата **подтверждения**
(`confirm_date`, единственная каузально верная точка; см. `band_lambda.build_zigzag`).

Прогноз — не точка, а **полоса неопределённости** на двух шагах: h=1 (уход) и h=2 (уход+возврат),
читается напрямую из взвешенных квантилей причинно обрезанного пула (не цепочка точечных прогнозов).
`forecast_live_band(..., origin_index=None)` берёт последний доступный пивот целевого тикера
(«живой» прогноз на сегодня); для прогноза «как было в прошлом» вызывающий код (`task_manager.
_run_forecast`) обрезает ДАННЫЕ (не индекс) до `cutoff_date` через `mask_ticker_data` — это
единственная точка контроля каузальности, применяется и к целевому инструменту, и к каждому тикеру
пула (утечка данных пула после origin — самая частая причина ложно оптимистичных чисел, см. память
`feedback-causality-enforcement`).

Поле прогноза (шаг1/шаг2) растягивается по горизонтали мышью (`editable:true` Plotly-фигуры) —
геометрия пишется в `forecasts.zone_geometry_json` через `POST /forecasts/{id}/geometry`, отдельно
от иммутабельного `result_json`.

### Фронтенд: ES-модули

`index.html` загружает `app.js` как `type="module"`. Граф зависимостей:

```
state.js    ← (ничего)
api.js      ← (ничего)
chart.js    ← state.js
analysis.js ← state.js, api.js, chart.js
forecast.js ← state.js, api.js, chart.js
app.js      ← все выше
```

Функции для HTML-обработчиков регистрируются в `app.js` через `window.xxx = fn`. Привязка `plotly_click` отложена до первого `renderChart` (`initChartEvents` + `_bindClickIfNeeded`), чтобы Plotly успел инициализировать элемент.

### Новый источник данных

Создать `sma/data/<source>/` по образцу `sma/data/moex/`:
- `candles.py` с константой `DATA_SOURCE = "<source>"`
- `__init__.py`, `__main__.py`

API-справочник MOEX: `docs/moex-api/README.md`
