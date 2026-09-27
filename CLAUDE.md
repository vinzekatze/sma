# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Python-инструмент для прогнозирования биржевых котировок (Московская биржа, MOEX):
FastAPI-бэкенд + HTML/JS-фронтенд (Plotly.js).

Это **прод-репозиторий** — только работающее приложение. Вся исследовательская история
(какие методы перебирались, почему выбран именно этот алгоритм, экспериментальные
протоколы и журнал по фазам) живёт в отдельном репозитории
**[sma-research](https://github.com/vinzekatze/sma-research)**. Если нужно понять
*происхождение* конкретного алгоритма — искать там (докстринги в `sma/core/` часто
ссылаются на исходные research/prototype-скрипты по имени файла).

Владелец имеет опыт написания торговых плагинов на MQL5.

## Ключевые методические решения (что именно делает прогноз)

- **Нормализация — logtrend (causal OLS)**: `ratio = close / exp(a + b·t)`, коэффициенты
  вычисляются через инкрементальные суммы O(N). Нет разгрева, нет параметра окна.
- **LP-фильтр (Local Projective filter)**: разложение `dratio` на att (детерминированная
  компонента) + noise; параметры — см. `sma/core/forecast/`.
- **LWR (Locally Weighted Regression)**: МНК с Гауссовыми весами по расстоянию — основной
  метод локальной аппроксимации соседей.
- **Реконструкция цены**: `dratio_hat → ratio_hat = ratio[origin] + cumsum(dratio_hat) →
  price = ratio_hat × logtrend[origin]`. Всегда однократный cumsum.
- **Прогноз — не точка, а полоса неопределённости**, читается из взвешенных квантилей
  причинно обрезанного пула соседей (см. «Прогноз band_lambda: точка отсчёта и
  каузальность» ниже) — не цепочка точечных прогнозов.
- **Источник данных** — [MOEX ISS API](https://iss.moex.com/iss/reference/) (бесплатный,
  задержка 15 мин).

> ⚠️ **Каузальность — жёсткое требование.** Любой новый алгоритм прогноза обязан
> использовать только данные, доступные на момент `origin` (дата подтверждения пивота
> для целевого инструмента, и для каждого тикера пула). Утечка данных пула после origin —
> самая частая причина ложно оптимистичных чисел. Единственная точка контроля —
> `mask_ticker_data`, обрезающая ДАННЫЕ (не индекс) до `cutoff_date`.

## Принципы модульности

Поддерживать чистое разделение ответственности — как на бэкенде, так и на фронтенде. При
добавлении новой функциональности:

- **Не складывать всё в существующий файл**, если логика образует отдельную область (новый
  источник данных, новый алгоритм, новая группа UI-функций).
- **Выносить в отдельный модуль**, когда: файл превышает ~300–400 строк, появляется
  повторяющаяся тематика, или новая фича слабо связана с текущим модулем.
- На **бэкенде**: новые роуты — в `sma/api/routes/`, новые алгоритмы — в
  `sma/core/forecast/`, новые метрики — в `sma/core/metrics/`.
- На **фронтенде**: JS использует ES-модули (`type="module"`). Новые смысловые блоки
  выносить в отдельные `.js`-файлы в `sma/ui/`. Функции, вызываемые из HTML-атрибутов
  (`onclick`, `onchange`), регистрировать через `window.xxx = fn` в `app.js`.

## Структура репозитория

```
docs/
  moex-api/         — справочник MOEX ISS API

sma/                — приложение (FastAPI + HTML/JS)
  api/              — FastAPI приложение
    app.py          — точка входа; lifespan: init_db + TaskManager
    deps.py         — DB_PATH, get_db(), get_task_manager()
    task_manager.py — очередь задач (forecast/candle_fetch/pool_resolve/
                       range_forecast_calibration), WebSocket broadcast
    routes/
      instruments.py      — GET/POST /instruments (added_via manual|pool), /search, /board
      candles.py           — POST /candles/fetch, GET /candles
      forecasts.py         — POST /forecasts (202), GET /forecasts/{id}, GET /forecasts,
                              GET /forecasts/zigzag, POST /forecasts/{id}/geometry
      forecast_settings.py — GET /forecast-settings/defaults, GET /forecast-settings/pool,
                              POST .../pool и POST .../resolve-pool (асинхронные, 202+task_id),
                              GET .../resolve-pool-cache
      display_presets.py   — GET/POST /display-presets, DELETE /{id}
      tasks.py              — GET /tasks/{id}, POST /{id}/cancel|resume, POST /resume-all,
                                DELETE /{id}, WS /tasks/{id}/ws
      series.py             — POST /series/spectrogram (STFT-анализатор вкладки «Анализ»)
      settings.py            — GET/POST /settings (глобальные тюнинги)
  core/             — алгоритмическое ядро
    db.py           — async SQLite (aiosqlite): instruments, candles, forecasts,
                      band_lambda_pool, display_presets, tasks
    models.py       — Pydantic-модели общего назначения (CandleBar)
    candle_fetch.py — resolve_fetch_plan (full/incremental), queue_pool_candle_fetch
    forecast/         — модули алгоритмов (normalize.py, band_lambda.py, pool_selection.py,
                        simplex_ensemble.py, range_forecast.py, risk_corridor.py,
                        regime_mixture_potential.py, pivot_time_band.py и др.)
    analysis/         — анализаторы вкладки «Анализ» (свой файл на анализатор)
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
                      buildBandZoneShapes, buildSubpanelTraces, initChartEvents
    analysis.js     — вкладка «Анализ»
    forecast.js     — вкладка «Прогноз»: submitForecast, WS, история, драг-геометрия полос
    band_pool_modal.js — модалка «Пул»

run.sh              — запуск приложения (uvicorn)
requirements.txt
pyproject.toml      — делает пакет sma editable-installable (нужно для sma-research)
Dockerfile
docker-compose.yml
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

# Docker
docker build -t sma .
docker compose up

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

Таблицы: `instruments`, `candles`, `forecasts`, `band_lambda_pool`, `display_presets`, `tasks`, `app_settings`,
`pool_resolution_cache`, `forecast_defaults`.

- `instruments` — тикер + источник данных (`data_source`) + тип актива (`asset_type`) + `added_via`
  (`manual` — добавлен пользователем, `pool` — автоотобран при резолве пула; `GET /instruments` по
  умолчанию отдаёт только `manual`, `include_pool=true` — все)
- `candles` — `ON CONFLICT DO UPDATE` (сохраняет row ID → FK прогнозов не ломаются при обновлении)
- `forecasts` — одна обобщённая таблица с дискриминатором `model_type`; `origin_candle_id` — жёсткая
  привязка к бару **подтверждения** пивота (каузальная точка); `result_json` иммутабелен после записи;
  `zone_geometry_json` — единственное редактируемое поле (растяжка полос мышью)
- `band_lambda_pool` — ОДИН состав пула на (instrument_id, interval) — не по T; T/m/θ — свободные
  параметры каждого запроса прогноза
- `display_presets` — именованные визуальные настройки (уровни/прозрачность/уровень доверия), не
  привязаны к прогнозу
- `tasks` — очередь задач: pending → running → done/error; `kind` = forecast|candle_fetch|
  pool_resolve|range_forecast_calibration. Строго последовательная (один воркер) — зависшая
  задача блокирует ВСЮ очередь; на старте `_reconcile_after_restart` помечает зависший `running`
  как `interrupted`, а `pending` с `cancel_requested=1` — сразу как `cancelled`
- `app_settings` — singleton-строка (id=1) глобальных тюнингов (параллельность MOEX-запросов при
  отборе пула, число процессов калибровки range_forecast/simplex_ensemble; 0 = авто); левая панель
  «Настройки приложения»

### TaskManager

Один asyncio-воркер обрабатывает задачи последовательно. Прогноз запускается в `run_in_executor`
(ProcessPoolExecutor). Прогресс пробрасывается из потока в event loop через
`loop.call_soon_threadsafe`. Подписчики получают события через `asyncio.Queue`.

### Воркеры ProcessPoolExecutor

Количество воркеров: `max(1, os.cpu_count() // 4)` — numpy-операции memory-bandwidth-bound; более
агрессивные значения приводят к деградации производительности (проверено бенчмарком на
16-ядерной машине — методика и код бенчмарка в `sma-research/tests/`).

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
пула (утечка данных пула после origin — самая частая причина ложно оптимистичных чисел).

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
band_pool_modal.js ← state.js, api.js, forecast.js (refreshes the sidebar pool summary on save)
app.js      ← все выше
```

Функции для HTML-обработчиков регистрируются в `app.js` через `window.xxx = fn`. Привязка
`plotly_click` отложена до первого `renderChart` (`initChartEvents` + `_bindClickIfNeeded`), чтобы
Plotly успел инициализировать элемент.

### Новый источник данных

Создать `sma/data/<source>/` по образцу `sma/data/moex/`:
- `candles.py` с константой `DATA_SOURCE = "<source>"`
- `__init__.py`, `__main__.py`

API-справочник MOEX: `docs/moex-api/README.md`
