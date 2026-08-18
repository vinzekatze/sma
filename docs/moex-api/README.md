# MOEX ISS API — Справочник

Базовый URL: `https://iss.moex.com/iss`  
Официальная документация: https://www.moex.com/a2193  
Аутентификация не требуется. Задержка данных — 15 минут.

---

## Свечи (OHLCV)

```
GET /engines/{engine}/markets/{market}/securities/{secid}/candles.json
```

Для акций: `engine=stock`, `market=shares`. Борд (TQBR) выбирается автоматически.

### Параметры запроса

| Параметр   | Тип    | Описание |
|------------|--------|----------|
| `from`     | string | Начало диапазона `YYYY-MM-DD` |
| `till`     | string | Конец диапазона `YYYY-MM-DD` |
| `interval` | int    | Код таймфрейма (см. таблицу ниже) |
| `start`    | int    | Смещение для пагинации (по умолчанию 0) |
| `iss.meta` | string | `off` — не возвращать блок метаданных |
| `iss.only` | string | `candles` — вернуть только блок свечей |

### Поддерживаемые таймфреймы (interval)

| Код | Период    |
|-----|-----------|
| 1   | 1 минута  |
| 10  | 10 минут  |
| 60  | 1 час     |
| 24  | 1 день    |
| 7   | 1 неделя  |
| 31  | 1 месяц   |

> **15m и 30m нативно не поддерживаются.**  
> `forcaster/data/moex.py` строит их ресемплингом:  
> `30m` ← агрегация `10m` × 3  
> `15m` ← агрегация `1m` × 15 (заметно больше запросов)

### Пагинация

Максимум **500 записей** на страницу. Для следующей страницы: `start += 500`.  
Признак последней страницы: количество строк в ответе < 500.

### Поля ответа (`candles.columns`)

| Поле     | Тип    | Описание |
|----------|--------|----------|
| `open`   | float  | Цена открытия |
| `close`  | float  | Цена закрытия |
| `high`   | float  | Максимум периода |
| `low`    | float  | Минимум периода |
| `value`  | float  | Оборот в рублях |
| `volume` | float  | Объём в лотах |
| `begin`  | string | Открытие свечи `YYYY-MM-DD HH:MM:SS` |
| `end`    | string | Закрытие свечи `YYYY-MM-DD HH:MM:SS` |

### Пример запроса

```
GET https://iss.moex.com/iss/engines/stock/markets/shares/securities/SBER/candles.json
    ?from=2024-01-01&till=2024-01-31&interval=60&start=0&iss.meta=off&iss.only=candles
```

---

## Поиск тикера по названию

```
GET /securities.json?q={query}&iss.meta=off
```

Полезные поля ответа: `secid`, `shortname`, `name`, `engine`, `market`, `type`.

## Информация об инструменте

```
GET /securities/{secid}.json?iss.meta=off
```

## Список акций основного борда (TQBR)

```
GET /engines/stock/markets/shares/boards/TQBR/securities.json?iss.meta=off
```

---

## Сессии ММВБ (МСК)

| Сессия            | Время       |
|-------------------|-------------|
| Основная          | 10:00–18:50 |
| Вечерняя (не все) | 19:00–23:50 |

Свечи за нерабочие периоды в ответе отсутствуют (не нулевые, а просто пропущены).
