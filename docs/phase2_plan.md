# Phase 2 — план переработки

> Статус: **к реализации**.

---

## Мотивация

Текущий прототип (`forcaster/`) выполнил свою роль: отработали методы, нашли
рабочий стек (LWR авто-p + PCA + дисперсионный бэнд). Два ограничения:

1. **Streamlit** пересчитывает весь скрипт на каждое взаимодействие; тяжёлые
   вычисления (перебор тысяч пар (p, pca_k)) выполняются в один поток.
2. **Нет планировщика задач** — сбор данных, расчёты и замеры точности нельзя
   поставить в очередь и запустить в фоне.

---

## Архитектура

### Принцип: воркеры + очередь, не микросервисы

Полноценные микросервисы (отдельные HTTP-сервисы, общающиеся по сети) для
инструмента одного разработчика — лишний overhead. Вместо этого:

- **Один FastAPI-процесс** обслуживает HTTP + WebSocket
- **Очередь задач** (asyncio.Queue или Redis-backed при необходимости) принимает
  задания от UI
- **Воркеры** (пул процессов ProcessPoolExecutor) разбирают задачи независимо
- **WebSocket** гонит статус и промежуточные результаты в UI в реальном времени

MOEX-модуль остаётся чистым Python-модулем (не отдельным сервисом), но с чёткой
границей: `core/data/moex.py` → при необходимости вынести в отдельный процесс
или сервис позже без переписывания логики.

```
UI (браузер)
    │  WebSocket / HTTP
    ▼
FastAPI (api/)
    │  asyncio.Queue
    ▼
TaskManager (api/tasks.py)
    │  ProcessPoolExecutor
    ├── Worker: ForecastTask   → core/forecast/
    ├── Worker: DataFetchTask  → core/data/moex.py
    └── Worker: AccuracyTask   → core/forecast/ + core/metrics/
```

### Типы задач планировщика

| Задача | Описание |
|---|---|
| `DataFetch` | Скачать / обновить свечи по тикеру и интервалу |
| `Forecast` | Запустить LWR/LA1 авто-p для тикера, сохранить результат |
| `AccuracyWalkForward` | Walk-forward по историческим данным, замерить MAPE |
| *(расширяемо)* | Сканирование по списку тикеров и т.д. |

---

## Структура репозитория

```
forcaster/              ← прототип (не трогаем, источник для подглядывания)
sma/                    ← новая реализация
  core/                 — вычислительное ядро, без UI-зависимостей
    data/
      moex.py           — MOEX ISS API
    forecast/
      embedding.py      — матрица задержек, last_vector
      normalize.py      — ratio = close / SMA
      lwr.py            — LWR авто-p (основной)
      la1.py            — LA1 авто-p (контрольный)
      hurst.py          — зоны Херста
      ma_trend.py       — линейный тренд по N барам MA
    metrics/
      accuracy.py       — MAPE, walk-forward evaluation
    runner.py           — параллельный перебор (p, pca_k) через ProcessPoolExecutor
  api/
    main.py             — FastAPI app, WebSocket endpoint
    tasks.py            — TaskManager, очередь, типы задач
    store.py            — хранилище результатов (in-memory / SQLite)
  ui/
    static/             — HTML + Plotly.js + минимум vanilla JS
    templates/          — Jinja2 шаблоны (опционально)
```

---

## Что берём из прототипа

| Компонент | Статус | Примечание |
|---|---|---|
| LWR — авто p | ✅ основной | переносим как есть |
| LA1 — авто p | ✅ контрольный | переносим как есть |
| Дисперсионный бэнд (±1σ) | ✅ | из тех же результатов |
| PCA-метрика | ✅ всегда включена | убираем как чекбокс — это не опция, а часть метода |
| Скользящая средняя + автоопределение окна | ✅ | |
| Зоны Херста | ✅ | |
| Линейный тренд MA | ✅ | |
| MOEX ISS API | ✅ | |

## Что не берём

| Компонент | Причина |
|---|---|
| `norm_vecs` | PCA покрывает; избыточно |
| Режимный / фазовый фильтр | на практике ухудшает результат |
| Фильтр по дням недели | тестировали — негативный эффект |
| Huber-регрессия | медленно + LWR-взвешенность решает ту же задачу мягче |
| LA1 одиночный / ансамбль | не нужны при наличии авто-p |

---

## Параллелизм

Перебор пар (p, pca_k) — задачи полностью независимы → ProcessPoolExecutor.
При p_max=70: ~2300 задач. На 8+ ядрах — секунды вместо минут.
С параллелизмом p_max можно поднять до 100–150 (открытый вопрос).

```python
# core/runner.py — концепция
def run_auto_p(params: AutoPParams, progress_cb=None) -> list[CandResult]:
    tasks = [(p, k) for p in range(2, params.p_max + 1) for k in range(2, p)]
    results = []
    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(eval_one, params, p, k): (p, k) for p, k in tasks}
        for i, f in enumerate(as_completed(futures)):
            results.append(f.result())
            if progress_cb:
                progress_cb(i + 1, len(tasks))   # → WebSocket
    return sorted(results, key=lambda r: r.mape)
```

---

## UI

FastAPI backend + Plotly.js + vanilla JS.

- Никакого Python UI-фреймворка между кодом и браузером
- Прогресс задач — WebSocket, обновляет progress bar без перезагрузки страницы
- Графики — Plotly.js (тот же рендер, что в прототипе, но без Streamlit-обёртки)
- Планировщик — простая таблица активных/завершённых задач, кнопки запуска

---

## Решённые вопросы

| Вопрос | Решение |
|---|---|
| Хранилище | **SQLite** — персистентно, нет лишних зависимостей. Свечи, результаты прогнозов, история задач. |
| p_max | **100** — при параллелизме реально, при 70 уже хорошие результаты, запас есть. |
| Несколько тикеров | Активный просмотр **одного тикера** за раз. Результаты прогнозов сохраняются в БД — можно вернуться в любой момент. |
| Обновление данных | **Только вручную** — через задачу планировщика или кнопку. Без автообновления при открытии (не агрить MOEX). |

## Деплой

Запуск на хосте: `python -m sma.api` + systemd-юнит для автостарта.
Docker не нужен сейчас, но код писать так, чтобы завернуть было легко:
никаких абсолютных путей, конфиг через переменные окружения / `.env`-файл,
данные в отдельной директории (не внутри пакета).

---

## План реализации

Принцип: снизу вверх. Каждый слой тестируется до перехода к следующему.
`core/` не знает про `api/`, `api/` не знает про `ui/`.

### Этап 1 — вычислительное ядро `core/`

Самодостаточный Python-пакет. Никаких FastAPI / Streamlit импортов.

1. **`core/models.py`** — Pydantic-модели для параметров и результатов:
   `ForecastParams`, `CandResult`, `AutoPResult`, `CandleBar`

2. **`core/data/moex.py`** — перенос из прототипа:
   `download_candles()`, `save_candles()`, retry + пагинация

3. **`core/forecast/embedding.py`**, **`normalize.py`** — перенос как есть

4. **`core/forecast/lwr.py`**, **`la1.py`** — перенос, убрать все
   параметры которые не берём (`use_huber`, `norm_vecs`, `regime_mask`)

5. **`core/forecast/hurst.py`**, **`ma_trend.py`** — перенос как есть

6. **`core/runner.py`** — параллельный перебор (p, pca_k) через
   `ProcessPoolExecutor`, `progress_cb` для последующей передачи в WebSocket

7. **`core/metrics/accuracy.py`** — walk-forward MAPE по истории,
   используется задачей `AccuracyWalkForward`

> Проверка: `core/` должен работать из чистого Python-скрипта без запуска сервера.

---

### Этап 2 — хранилище `core/db.py`

SQLite через `aiosqlite` (async-совместимо с FastAPI).

Таблицы:
- `candles (ticker, interval, ts, open, high, low, close, volume)`
- `forecasts (id, ticker, interval, origin_ts, created_at, params_json, result_json)`
- `tasks (id, type, status, created_at, finished_at, payload_json, error)`

> Проверка: миграция применяется, данные пишутся и читаются.

---

### Этап 3 — API и планировщик `api/`

3. **`api/tasks.py`** — `TaskManager`:
   - `asyncio.Queue` для входящих задач
   - `ProcessPoolExecutor` для CPU-задач (forecast, accuracy)
   - `asyncio.create_task` для IO-задач (data fetch)
   - статус задач пишется в `tasks`-таблицу

4. **`api/ws.py`** — WebSocket: подписка на прогресс задачи по `task_id`,
   сервер пушит `{progress, status, partial_result}` по мере выполнения

5. **`api/routes.py`** — HTTP эндпоинты:
   - `POST /tasks` — поставить задачу в очередь, вернуть `task_id`
   - `GET /tasks` — список задач (активные + история)
   - `GET /forecasts` — сохранённые результаты прогнозов
   - `GET /candles/{ticker}/{interval}` — свечи из БД
   - `WS /ws/{task_id}` — прогресс задачи

6. **`api/main.py`** — сборка приложения, lifespan (старт TaskManager при запуске)

> Проверка: через curl/httpie задача ставится в очередь, WebSocket отдаёт прогресс,
> результат появляется в БД.

---

### Этап 4 — фронтенд `ui/`

Минимум зависимостей: HTML + Plotly.js + vanilla JS. Никакого React/Vue.

7. **Планировщик задач** (`/tasks`):
   - форма: тип задачи, тикер, интервал, параметры
   - таблица задач с live-статусом через WebSocket
   - progress bar пока задача выполняется

8. **График прогноза** (`/forecast/{id}`):
   - свечи + MA + зоны Херста
   - лучший прогноз (жёлтый), средняя (фуксия), бэнд ±1σ
   - зона валидации

9. **Список сохранённых прогнозов** (`/`):
   - таблица: тикер, дата, MAPE лучшего, параметры
   - ссылка на график

> Проверка: полный цикл через браузер — поставил задачу, дождался, открыл результат.

---

### Порядок зависимостей

```
core/models     ──► core/data, core/forecast, core/metrics
core/runner     ──► core/forecast
core/db         ──► core/models
api/tasks       ──► core/runner, core/db
api/routes      ──► api/tasks, core/db
ui/             ──► api/routes, api/ws
```
