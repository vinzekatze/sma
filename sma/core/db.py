"""
Async SQLite storage layer (aiosqlite).

Schema overview
---------------
instruments      — one row per (ticker, data_source) pair; carries asset_type
                    so the same code works for stocks, futures, currencies,
                    etc. added_via distinguishes tickers the user added
                    manually from ones auto-selected into a calibration pool.
candles          — OHLCV bars; FK → instruments; ON CONFLICT updates in-place
                    so forecast FKs remain valid when bars are refreshed.
forecasts        — one row per forecast, discriminated by model_type; anchored
                    to the origin candle (bar of CONFIRMATION, the causally
                    valid point) via FK. result_json is immutable once
                    written; zone_geometry_json is the only editable field
                    (visual drag-resize of the band shapes on the chart).
forecast_settings — a calibration result (λ weights etc.) for one
                    (instrument, interval, model_type, t_query, pool_key).
                    Several pool_key rows can coexist for the same T so
                    different pool compositions can be compared side by
                    side; is_active marks which one live forecasts use.
display_presets  — named visual settings (band levels/opacity), independent
                    of any specific forecast — reused across models.
forecast_defaults — last-used UI params for (instrument, interval, model_type),
                    auto-saved server-side after every successful forecast of
                    that model_type and auto-loaded to prefill the form next
                    time — currently only simplex_ensemble writes this;
                    band_lambda has its own, older "active settings" concept
                    (forecast_settings.is_active) and doesn't use this table.
analysis_settings — same idea as forecast_defaults but for the Анализ tab's
                    analyzers (analyzer_type='spectrogram' currently) —
                    client-saved (POST /series/analysis-settings, called
                    right after a successful compute) rather than server-
                    saved, because analyzer params include pure-display
                    fields (contrast/logY) the backend never sees.
tasks            — background job queue (pending → running → done/error).
app_settings     — singleton row (id=1) of global tunables (MOEX request
                    concurrency, calibration worker count) editable from the
                    left panel — see sma/api/routes/settings.py.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

# ── schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS instruments (
    id          INTEGER PRIMARY KEY,
    ticker      TEXT    NOT NULL,
    data_source TEXT    NOT NULL,
    asset_type  TEXT    NOT NULL DEFAULT 'stock',
    full_name   TEXT,
    UNIQUE(ticker, data_source)
);

CREATE TABLE IF NOT EXISTS candles (
    id            INTEGER PRIMARY KEY,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    interval      TEXT    NOT NULL,
    begin         TEXT    NOT NULL,
    open          REAL    NOT NULL,
    high          REAL    NOT NULL,
    low           REAL    NOT NULL,
    close         REAL    NOT NULL,
    volume        REAL    NOT NULL,
    fetched_at    TEXT    NOT NULL,
    UNIQUE(instrument_id, interval, begin)
);
CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON candles(instrument_id, interval, begin);

CREATE TABLE IF NOT EXISTS forecasts (
    id                 INTEGER PRIMARY KEY,
    instrument_id      INTEGER NOT NULL REFERENCES instruments(id),
    interval           TEXT    NOT NULL,
    model_type         TEXT    NOT NULL,
    origin_candle_id   INTEGER NOT NULL REFERENCES candles(id),
    created_at         TEXT    NOT NULL,
    params_json        TEXT    NOT NULL,
    result_json        TEXT    NOT NULL,
    zone_geometry_json TEXT,
    is_stale           INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_forecasts_lookup
    ON forecasts(instrument_id, interval, created_at DESC);

CREATE TABLE IF NOT EXISTS forecast_settings (
    id                    INTEGER PRIMARY KEY,
    instrument_id         INTEGER NOT NULL REFERENCES instruments(id),
    interval              TEXT    NOT NULL,
    model_type            TEXT    NOT NULL,
    t_query               REAL    NOT NULL,
    min_bars              INTEGER NOT NULL DEFAULT 5,
    lambda_json           TEXT    NOT NULL,
    m                     INTEGER NOT NULL DEFAULT 6,
    theta                 REAL    NOT NULL DEFAULT 0,
    pool_config_json      TEXT    NOT NULL,
    pool_key              TEXT    NOT NULL,
    is_active             INTEGER NOT NULL DEFAULT 1,
    calibration_meta_json TEXT    NOT NULL,
    UNIQUE(instrument_id, interval, model_type, t_query, pool_key)
);
CREATE INDEX IF NOT EXISTS idx_forecast_settings_lookup
    ON forecast_settings(instrument_id, interval, model_type);

CREATE TABLE IF NOT EXISTS display_presets (
    id          INTEGER PRIMARY KEY,
    model_type  TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    levels_json TEXT    NOT NULL,
    opacity     REAL    NOT NULL,
    is_default  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL,
    UNIQUE(model_type, name)
);

CREATE TABLE IF NOT EXISTS forecast_defaults (
    id            INTEGER PRIMARY KEY,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id),
    interval      TEXT    NOT NULL,
    model_type    TEXT    NOT NULL,
    params_json   TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    UNIQUE(instrument_id, interval, model_type)
);

CREATE TABLE IF NOT EXISTS analysis_settings (
    id            INTEGER PRIMARY KEY,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id),
    interval      TEXT    NOT NULL,
    analyzer_type TEXT    NOT NULL,
    params_json   TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    UNIQUE(instrument_id, interval, analyzer_type)
);

CREATE TABLE IF NOT EXISTS tasks (
    id             INTEGER PRIMARY KEY,
    created_at     TEXT    NOT NULL,
    status         TEXT    NOT NULL DEFAULT 'pending',
    instrument_id  INTEGER NOT NULL REFERENCES instruments(id),
    interval       TEXT    NOT NULL,
    origin_ts      TEXT    NOT NULL,
    params_json    TEXT    NOT NULL,
    forecast_id    INTEGER REFERENCES forecasts(id),
    error          TEXT,
    progress_done  INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, created_at);

CREATE TABLE IF NOT EXISTS app_settings (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    moex_pool_workers   INTEGER NOT NULL DEFAULT 16,
    calibration_workers INTEGER NOT NULL DEFAULT 0,
    updated_at          TEXT    NOT NULL DEFAULT ''
);
"""


# ── connection helper ─────────────────────────────────────────────────────────

@asynccontextmanager
async def open_db(path: str | Path):
    """Yield an aiosqlite connection with WAL mode and FK enforcement."""
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode = WAL")
        await db.execute("PRAGMA foreign_keys = ON")
        yield db


_MIGRATIONS = [
    "ALTER TABLE instruments      ADD COLUMN engine            TEXT    NOT NULL DEFAULT 'stock'",
    "ALTER TABLE instruments      ADD COLUMN market            TEXT    NOT NULL DEFAULT 'shares'",
    "ALTER TABLE instruments      ADD COLUMN board             TEXT",
    "ALTER TABLE instruments      ADD COLUMN is_favorite       INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE instruments      ADD COLUMN added_via          TEXT    NOT NULL DEFAULT 'manual'",
    "ALTER TABLE tasks            ADD COLUMN kind               TEXT    NOT NULL DEFAULT 'forecast'",
    "ALTER TABLE tasks            ADD COLUMN label              TEXT    NOT NULL DEFAULT ''",
    "ALTER TABLE tasks            ADD COLUMN cancel_requested   INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE tasks            ADD COLUMN updated_at         TEXT    NOT NULL DEFAULT ''",
]


async def _drop_stale_forecasts_table(db: aiosqlite.Connection) -> None:
    """
    The band_lambda migration replaces `forecasts` with an incompatible
    schema (model_type discriminator, no more ma_window/hurst_at_origin/
    mean_price_last columns). Old LA/LWR forecasts are not migrated — this
    was an explicit, approved decision (docs/plans/band_forecast_migration_
    plan.md, section 3): the old model was never reliable, so there is
    nothing worth carrying forward. Detect the old shape by a column that
    only it has and drop it before the new CREATE TABLE IF NOT EXISTS runs.
    """
    cursor = await db.execute("PRAGMA table_info(forecasts)")
    cols = {row[1] for row in await cursor.fetchall()}
    if cols and "model_type" not in cols:
        await db.execute("DROP TABLE forecasts")
        await db.commit()


async def _drop_series_settings_table(db: aiosqlite.Connection) -> None:
    """
    series_settings backed the Hurst-zones/optimal-MA analysis-tab features,
    removed 2026-08-08 in favor of the spectrogram analyzer (see
    sma/core/analysis/spectrogram.py) — nothing computes or reads
    optimal_ma/hurst_* any more (the old LA/LWR forecast engine that used to
    consume hurst_look_back is long gone too, since the band_lambda
    migration). Drop unconditionally; there is nothing that reads this
    table's contents to preserve.
    """
    await db.execute("DROP TABLE IF EXISTS series_settings")
    await db.commit()


async def init_db(path: str | Path) -> None:
    """Create tables if they don't exist yet; run safe column migrations."""
    async with aiosqlite.connect(path) as db:
        await _drop_stale_forecasts_table(db)
        await _drop_series_settings_table(db)
        await db.executescript(_SCHEMA)
        await db.commit()
        for sql in _MIGRATIONS:
            try:
                await db.execute(sql)
                await db.commit()
            except aiosqlite.OperationalError:
                pass  # column already exists


# ── instruments ───────────────────────────────────────────────────────────────

async def upsert_instrument(
    db: aiosqlite.Connection,
    ticker: str,
    data_source: str,
    asset_type: str | None = None,
    full_name: str | None = None,
    engine: str | None = None,
    market: str | None = None,
    board: str | None = None,
    added_via: str | None = None,
) -> int:
    """
    Insert or update instrument record; return its id.

    asset_type/engine/market/board are only overwritten when explicitly
    given (non-None); omitting them on an update (e.g. an incremental
    candles refresh that only knows the ticker) preserves whatever was
    stored when the instrument was first added.

    added_via distinguishes tickers the user added themselves ('manual')
    from ones auto-selected into a calibration pool ('pool') — see
    docs/plans/band_forecast_migration_plan.md 5.2a. On conflict:
      - added_via='manual' always wins (a user finding a pool ticker via
        normal search should "claim" it, moving it into "Мои тикеры").
      - added_via='pool' never downgrades an existing 'manual' row.
      - added_via=None (e.g. a plain incremental candle refresh that
        doesn't know or care about ownership) leaves it untouched.
    """
    await db.execute(
        """
        INSERT INTO instruments (ticker, data_source, asset_type, full_name, engine, market, board, added_via)
        VALUES (?, ?, COALESCE(?, 'stock'), ?, COALESCE(?, 'stock'), COALESCE(?, 'shares'), ?, COALESCE(?, 'manual'))
        ON CONFLICT(ticker, data_source) DO UPDATE SET
            asset_type = COALESCE(?, asset_type),
            full_name  = COALESCE(excluded.full_name, full_name),
            engine     = COALESCE(?, engine),
            market     = COALESCE(?, market),
            board      = COALESCE(excluded.board, board),
            added_via  = CASE WHEN ? = 'manual' THEN 'manual' ELSE added_via END
        """,
        (ticker, data_source, asset_type, full_name, engine, market, board, added_via,
         asset_type, engine, market, added_via),
    )
    await db.commit()
    cursor = await db.execute(
        "SELECT id FROM instruments WHERE ticker=? AND data_source=?",
        (ticker, data_source),
    )
    row = await cursor.fetchone()
    return row["id"]


async def get_instrument(
    db: aiosqlite.Connection,
    ticker: str,
    data_source: str,
) -> dict | None:
    cursor = await db.execute(
        "SELECT * FROM instruments WHERE ticker=? AND data_source=?",
        (ticker, data_source),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_instrument_by_id(
    db: aiosqlite.Connection,
    instrument_id: int,
) -> dict | None:
    cursor = await db.execute(
        "SELECT * FROM instruments WHERE id=?", (instrument_id,)
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def list_instruments(
    db: aiosqlite.Connection,
    include_pool: bool = False,
) -> list[dict]:
    """By default only 'manual' instruments (what the user actually tracks) —
    pool-selected tickers from calibration stay out of "Мои тикеры" unless
    include_pool=True (see docs/plans/band_forecast_migration_plan.md 5.2a)."""
    q = "SELECT * FROM instruments"
    if not include_pool:
        q += " WHERE added_via = 'manual'"
    q += " ORDER BY ticker, data_source"
    cursor = await db.execute(q)
    return [dict(r) for r in await cursor.fetchall()]


async def set_favorite(
    db: aiosqlite.Connection,
    instrument_id: int,
    value: bool,
) -> None:
    await db.execute(
        "UPDATE instruments SET is_favorite=? WHERE id=?",
        (1 if value else 0, instrument_id),
    )
    await db.commit()


async def delete_instrument(
    db: aiosqlite.Connection,
    instrument_id: int,
) -> bool:
    """
    Cascading delete of an instrument and everything anchored to it.
    candles drop automatically via ON DELETE CASCADE; tasks and forecasts
    don't have that FK option, so they're removed explicitly first
    (foreign_keys=ON would otherwise reject the instrument delete).
    """
    await db.execute("DELETE FROM tasks WHERE instrument_id=?", (instrument_id,))
    await db.execute("DELETE FROM forecasts WHERE instrument_id=?", (instrument_id,))
    cursor = await db.execute("DELETE FROM instruments WHERE id=?", (instrument_id,))
    await db.commit()
    return cursor.rowcount > 0


# ── candles ───────────────────────────────────────────────────────────────────

async def upsert_candles(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    candles: list[dict],
) -> int:
    """
    Bulk-insert candles; existing rows are updated in-place (id preserved so
    forecast FKs remain valid).  Returns number of rows touched.
    """
    now = _utcnow()
    # currency/metal instruments (engine=currency) quote FX-style pairs with no
    # lot concept — MOEX returns volume=None for them, unlike stocks/futures.
    rows = [
        (
            instrument_id,
            interval,
            c["begin"],
            float(c["open"]),
            float(c["high"]),
            float(c["low"]),
            float(c["close"]),
            float(c["volume"]) if c["volume"] is not None else 0.0,
            now,
        )
        for c in candles
    ]
    await db.executemany(
        """
        INSERT INTO candles
            (instrument_id, interval, begin, open, high, low, close, volume, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(instrument_id, interval, begin) DO UPDATE SET
            open       = excluded.open,
            high       = excluded.high,
            low        = excluded.low,
            close      = excluded.close,
            volume     = excluded.volume,
            fetched_at = excluded.fetched_at
        """,
        rows,
    )
    await db.commit()
    return len(rows)


async def get_candles(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    since: str | None = None,
    until: str | None = None,
) -> list[dict]:
    """Return candles ordered by begin asc, optionally filtered by date range."""
    q = "SELECT * FROM candles WHERE instrument_id=? AND interval=?"
    args: list[Any] = [instrument_id, interval]
    if since:
        q += " AND begin >= ?"; args.append(since)
    if until:
        q += " AND begin <= ?"; args.append(until)
    q += " ORDER BY begin ASC"
    cursor = await db.execute(q, args)
    return [dict(r) for r in await cursor.fetchall()]


async def get_candle_by_id(
    db: aiosqlite.Connection,
    candle_id: int,
) -> dict | None:
    cursor = await db.execute("SELECT * FROM candles WHERE id=?", (candle_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_candle_by_ts(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    ts: str,
) -> dict | None:
    cursor = await db.execute(
        "SELECT * FROM candles WHERE instrument_id=? AND interval=? AND begin=?",
        (instrument_id, interval, ts),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def list_instrument_coverage(
    db: aiosqlite.Connection,
    instrument_id: int,
) -> list[dict]:
    """Per-interval candle coverage (first/last bar, count) for one instrument."""
    cursor = await db.execute(
        """
        SELECT interval, MIN(begin) AS first, MAX(begin) AS last, COUNT(*) AS n
        FROM candles
        WHERE instrument_id=?
        GROUP BY interval
        ORDER BY interval
        """,
        (instrument_id,),
    )
    return [dict(r) for r in await cursor.fetchall()]


# ── forecasts ─────────────────────────────────────────────────────────────────

async def save_forecast(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    model_type: str,
    origin_candle_id: int,
    params: dict,   # {t_query, lambda, m, theta, min_bars, forecast_settings_id}
    result: dict,   # {origin_date, origin_price, origin_direction, steps: {...}} — immutable
) -> int:
    """Persist a completed forecast; return its id."""
    cursor = await db.execute(
        """
        INSERT INTO forecasts (
            instrument_id, interval, model_type, origin_candle_id,
            created_at, params_json, result_json, zone_geometry_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            instrument_id, interval, model_type, origin_candle_id,
            _utcnow(),
            json.dumps(params, ensure_ascii=False),
            json.dumps(result, ensure_ascii=False),
        ),
    )
    await db.commit()
    return cursor.lastrowid


async def get_forecast(
    db: aiosqlite.Connection,
    forecast_id: int,
) -> dict | None:
    cursor = await db.execute(
        "SELECT * FROM forecasts WHERE id=?", (forecast_id,)
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json"))
    d["result"] = json.loads(d.pop("result_json"))
    geometry = d.pop("zone_geometry_json")
    d["zone_geometry"] = json.loads(geometry) if geometry else None
    return d


async def list_forecasts(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    model_type: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return recent forecasts without unpacking the full result blob (for
    list views) — origin_date/origin_price/origin_direction are pulled out
    of result_json since every model_type's result carries those three keys
    (see docs/plans/band_forecast_migration_plan.md 5.1). origin_extreme_ts
    is pulled out too: origin_price is the pivot's EXTREME price, not the
    confirmation bar's price, so a marker plotted at (origin_ts, origin_price)
    lines up with nothing on the actual chart — see band_lambda.py:
    pool_values_and_weights docstring and project feedback 2026-08-07."""
    q = """
        SELECT id, created_at, model_type, is_stale,
               json_extract(result_json, '$.origin_date')         AS origin_ts,
               json_extract(result_json, '$.origin_extreme_date') AS origin_extreme_ts,
               json_extract(result_json, '$.origin_price')        AS origin_price,
               json_extract(result_json, '$.origin_direction')    AS origin_direction
        FROM forecasts
        WHERE instrument_id=? AND interval=?
    """
    args: list[Any] = [instrument_id, interval]
    if model_type:
        q += " AND model_type=?"; args.append(model_type)
    q += " ORDER BY json_extract(result_json, '$.origin_date') DESC, created_at DESC LIMIT ?"
    args.append(limit)
    cursor = await db.execute(q, args)
    return [dict(r) for r in await cursor.fetchall()]


async def delete_forecast(
    db: aiosqlite.Connection,
    forecast_id: int,
) -> bool:
    """Delete a forecast by id. Returns True if a row was deleted."""
    await db.execute("UPDATE tasks SET forecast_id = NULL WHERE forecast_id = ?", (forecast_id,))
    cursor = await db.execute("DELETE FROM forecasts WHERE id=?", (forecast_id,))
    await db.commit()
    return cursor.rowcount > 0


async def update_forecast_geometry(
    db: aiosqlite.Connection,
    forecast_id: int,
    geometry: dict,
) -> bool:
    """Upsert the (only editable) zone_geometry_json — the visual drag-resize
    override of the band shapes, kept separate from the immutable result."""
    cursor = await db.execute(
        "UPDATE forecasts SET zone_geometry_json=? WHERE id=?",
        (json.dumps(geometry, ensure_ascii=False), forecast_id),
    )
    await db.commit()
    return cursor.rowcount > 0


async def refresh_stale_flags(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
) -> int:
    """
    Mark forecasts as stale when the origin candle's close has diverged from
    what it was when the forecast was saved (close_at_origin in result_json —
    NOT origin_price: that's the zigzag pivot's EXTREME price, a different
    bar entirely from origin_candle_id, the CONFIRMATION bar — comparing
    against it would flag every forecast stale immediately regardless of any
    real change. See band_lambda.py:pool_values_and_weights docstring and
    project feedback 2026-08-07.). Returns number of rows updated.
    """
    cursor = await db.execute(
        """
        UPDATE forecasts
        SET is_stale = 1
        WHERE instrument_id = ?
          AND interval = ?
          AND is_stale = 0
          AND ABS(
              json_extract(result_json, '$.close_at_origin') - (
                  SELECT c.close FROM candles c
                  WHERE c.id = forecasts.origin_candle_id
              )
          ) > 1e-9
        """,
        (instrument_id, interval),
    )
    await db.commit()
    return cursor.rowcount


# ── forecast settings (calibration results) ─────────────────────────────────

async def upsert_forecast_settings(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    model_type: str,
    t_query: float,
    min_bars: int,
    lambda_json: dict,
    m: int,
    theta: float,
    pool_config: dict,
    pool_key: str,
    calibration_meta: dict,
) -> int:
    """
    Insert or update one calibration row, keyed by (instrument, interval,
    model_type, t_query, pool_key) — several pool_key rows can coexist for
    the same T (see docs/plans/band_forecast_migration_plan.md 5.1). A brand
    new (instrument, interval, model_type, t_query) combo activates its
    first row automatically; subsequent pool_key variants for the same T are
    inserted inactive so an existing live forecast isn't silently redirected
    — the user switches via activate_forecast_settings().
    """
    cursor = await db.execute(
        """SELECT COUNT(*) AS n FROM forecast_settings
           WHERE instrument_id=? AND interval=? AND model_type=? AND t_query=?""",
        (instrument_id, interval, model_type, t_query),
    )
    is_new_group = (await cursor.fetchone())["n"] == 0

    await db.execute(
        """
        INSERT INTO forecast_settings (
            instrument_id, interval, model_type, t_query, min_bars,
            lambda_json, m, theta, pool_config_json, pool_key, is_active,
            calibration_meta_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(instrument_id, interval, model_type, t_query, pool_key) DO UPDATE SET
            min_bars              = excluded.min_bars,
            lambda_json           = excluded.lambda_json,
            m                     = excluded.m,
            theta                 = excluded.theta,
            pool_config_json      = excluded.pool_config_json,
            calibration_meta_json = excluded.calibration_meta_json
        """,
        (
            instrument_id, interval, model_type, t_query, min_bars,
            json.dumps(lambda_json, ensure_ascii=False), m, theta,
            json.dumps(pool_config, ensure_ascii=False), pool_key,
            1 if is_new_group else 0,
            json.dumps(calibration_meta, ensure_ascii=False),
        ),
    )
    await db.commit()
    cursor = await db.execute(
        """SELECT id FROM forecast_settings
           WHERE instrument_id=? AND interval=? AND model_type=? AND t_query=? AND pool_key=?""",
        (instrument_id, interval, model_type, t_query, pool_key),
    )
    return (await cursor.fetchone())["id"]


def _forecast_settings_row(row: dict) -> dict:
    d = dict(row)
    d["lambda"] = json.loads(d.pop("lambda_json"))
    d["pool_config"] = json.loads(d.pop("pool_config_json"))
    d["calibration_meta"] = json.loads(d.pop("calibration_meta_json"))
    d["is_active"] = bool(d["is_active"])
    return d


async def get_forecast_settings(
    db: aiosqlite.Connection,
    settings_id: int,
) -> dict | None:
    cursor = await db.execute("SELECT * FROM forecast_settings WHERE id=?", (settings_id,))
    row = await cursor.fetchone()
    return _forecast_settings_row(dict(row)) if row else None


async def get_active_forecast_settings(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    model_type: str,
    t_query: float,
) -> dict | None:
    cursor = await db.execute(
        """SELECT * FROM forecast_settings
           WHERE instrument_id=? AND interval=? AND model_type=? AND t_query=? AND is_active=1""",
        (instrument_id, interval, model_type, t_query),
    )
    row = await cursor.fetchone()
    return _forecast_settings_row(dict(row)) if row else None


async def list_forecast_settings(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    model_type: str,
) -> list[dict]:
    """All calibrated T / pool_key combinations — powers the quick T-selector
    (grouped by t_query on the frontend: active row shown, others collapsed)."""
    cursor = await db.execute(
        """SELECT * FROM forecast_settings
           WHERE instrument_id=? AND interval=? AND model_type=?
           ORDER BY t_query, is_active DESC""",
        (instrument_id, interval, model_type),
    )
    return [_forecast_settings_row(dict(r)) for r in await cursor.fetchall()]


async def activate_forecast_settings(
    db: aiosqlite.Connection,
    settings_id: int,
) -> bool:
    """Make settings_id the active row for its (instrument, interval,
    model_type, t_query) group; deactivate every sibling pool_key."""
    row = await get_forecast_settings(db, settings_id)
    if row is None:
        return False
    await db.execute(
        """UPDATE forecast_settings SET is_active=0
           WHERE instrument_id=? AND interval=? AND model_type=? AND t_query=?""",
        (row["instrument_id"], row["interval"], row["model_type"], row["t_query"]),
    )
    await db.execute("UPDATE forecast_settings SET is_active=1 WHERE id=?", (settings_id,))
    await db.commit()
    return True


async def delete_forecast_settings(
    db: aiosqlite.Connection,
    settings_id: int,
) -> bool:
    cursor = await db.execute("DELETE FROM forecast_settings WHERE id=?", (settings_id,))
    await db.commit()
    return cursor.rowcount > 0


# ── forecast defaults (last-used params per ticker, see table docstring) ────

async def get_forecast_defaults(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    model_type: str,
) -> dict | None:
    cursor = await db.execute(
        """SELECT * FROM forecast_defaults
           WHERE instrument_id=? AND interval=? AND model_type=?""",
        (instrument_id, interval, model_type),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json"))
    return d


async def upsert_forecast_defaults(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    model_type: str,
    params: dict,
) -> int:
    """Overwrite the stored params for this (instrument, interval, model_type)
    with the ones just used — called server-side right after a successful
    save_forecast, so this only ever reflects a forecast that actually
    completed (never a submitted-but-failed one)."""
    now = _utcnow()
    cursor = await db.execute(
        """INSERT INTO forecast_defaults (instrument_id, interval, model_type, params_json, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(instrument_id, interval, model_type)
           DO UPDATE SET params_json=excluded.params_json, updated_at=excluded.updated_at""",
        (instrument_id, interval, model_type, json.dumps(params, ensure_ascii=False), now),
    )
    await db.commit()
    return cursor.lastrowid


# ── analysis settings (last-used params per ticker, Анализ tab) ─────────────

async def get_analysis_settings(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    analyzer_type: str,
) -> dict | None:
    cursor = await db.execute(
        """SELECT * FROM analysis_settings
           WHERE instrument_id=? AND interval=? AND analyzer_type=?""",
        (instrument_id, interval, analyzer_type),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json"))
    return d


async def upsert_analysis_settings(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    analyzer_type: str,
    params: dict,
) -> int:
    now = _utcnow()
    cursor = await db.execute(
        """INSERT INTO analysis_settings (instrument_id, interval, analyzer_type, params_json, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(instrument_id, interval, analyzer_type)
           DO UPDATE SET params_json=excluded.params_json, updated_at=excluded.updated_at""",
        (instrument_id, interval, analyzer_type, json.dumps(params, ensure_ascii=False), now),
    )
    await db.commit()
    return cursor.lastrowid


# ── display presets ──────────────────────────────────────────────────────────

async def list_display_presets(
    db: aiosqlite.Connection,
    model_type: str,
) -> list[dict]:
    cursor = await db.execute(
        "SELECT * FROM display_presets WHERE model_type=? ORDER BY name",
        (model_type,),
    )
    rows = [dict(r) for r in await cursor.fetchall()]
    for r in rows:
        r["levels"] = json.loads(r.pop("levels_json"))
        r["is_default"] = bool(r["is_default"])
    return rows


async def save_display_preset(
    db: aiosqlite.Connection,
    model_type: str,
    name: str,
    levels: list[float],
    opacity: float,
    is_default: bool = False,
) -> int:
    """Insert or update by (model_type, name). At most one default preset per
    model_type — setting is_default clears it on every other preset of the
    same model_type first."""
    if is_default:
        await db.execute(
            "UPDATE display_presets SET is_default=0 WHERE model_type=?",
            (model_type,),
        )
    await db.execute(
        """
        INSERT INTO display_presets (model_type, name, levels_json, opacity, is_default, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_type, name) DO UPDATE SET
            levels_json = excluded.levels_json,
            opacity     = excluded.opacity,
            is_default  = excluded.is_default
        """,
        (model_type, name, json.dumps(levels), opacity, 1 if is_default else 0, _utcnow()),
    )
    await db.commit()
    cursor = await db.execute(
        "SELECT id FROM display_presets WHERE model_type=? AND name=?",
        (model_type, name),
    )
    return (await cursor.fetchone())["id"]


async def delete_display_preset(
    db: aiosqlite.Connection,
    preset_id: int,
) -> bool:
    cursor = await db.execute("DELETE FROM display_presets WHERE id=?", (preset_id,))
    await db.commit()
    return cursor.rowcount > 0


# ── tasks ─────────────────────────────────────────────────────────────────────

async def create_task(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    origin_ts: str,
    params: Any,
    kind: str = "forecast",
    label: str = "",
) -> int:
    """
    origin_ts is required by the column (NOT NULL) but only meaningful for
    kind='forecast' — pass "" for kinds that have no single point-in-time
    (candle_fetch, calibration, ...).
    """
    params_d = params.model_dump() if hasattr(params, "model_dump") else dict(params)
    now = _utcnow()
    cursor = await db.execute(
        """
        INSERT INTO tasks
            (created_at, updated_at, status, kind, label,
             instrument_id, interval, origin_ts, params_json)
        VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?)
        """,
        (now, now, kind, label, instrument_id, interval, origin_ts,
         json.dumps(params_d, ensure_ascii=False)),
    )
    await db.commit()
    return cursor.lastrowid


async def update_task(
    db: aiosqlite.Connection,
    task_id: int,
    *,
    status: str | None = None,
    progress_done: int | None = None,
    progress_total: int | None = None,
    forecast_id: int | None = None,
    error: str | None = None,
    cancel_requested: int | None = None,
) -> None:
    """updated_at is bumped on every call, whether or not other fields change."""
    fields: list[str] = ["updated_at = ?"]
    args:   list[Any] = [_utcnow()]

    if status           is not None: fields.append("status = ?");           args.append(status)
    if progress_done    is not None: fields.append("progress_done = ?");    args.append(progress_done)
    if progress_total   is not None: fields.append("progress_total = ?");   args.append(progress_total)
    if forecast_id      is not None: fields.append("forecast_id = ?");      args.append(forecast_id)
    if error            is not None: fields.append("error = ?");            args.append(error)
    if cancel_requested is not None: fields.append("cancel_requested = ?"); args.append(cancel_requested)

    args.append(task_id)
    await db.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?", args)
    await db.commit()


async def get_task(
    db: aiosqlite.Connection,
    task_id: int,
) -> dict | None:
    cursor = await db.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json"))
    return d


async def list_tasks(
    db: aiosqlite.Connection,
    status: str | None = None,
    kind: str | None = None,
    limit: int = 100,
) -> list[dict]:
    q = "SELECT * FROM tasks WHERE 1=1"
    args: list[Any] = []
    if status:
        q += " AND status = ?"; args.append(status)
    if kind:
        q += " AND kind = ?"; args.append(kind)
    # id DESC tie-break: updated_at has second resolution, and a batch of
    # tasks (e.g. pool candle_fetch during calibration) routinely finishes
    # within the same second — without a tie-break their relative order is
    # undefined and can visibly shuffle between polls. Same reasoning as
    # prune_old_done_tasks' ORDER BY below.
    q += " ORDER BY updated_at DESC, id DESC LIMIT ?"
    args.append(limit)

    cursor = await db.execute(q, args)
    rows = await cursor.fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["params"] = json.loads(d.pop("params_json"))
        result.append(d)
    return result


async def prune_old_done_tasks(db: aiosqlite.Connection, keep: int = 10) -> int:
    """
    Deletes 'done' task rows beyond the most recent `keep` (by updated_at,
    id as tie-break — updated_at has only second resolution, so a batch of
    tasks finishing within the same second, e.g. a pool calibration's
    candle_fetch tasks, would otherwise sort ambiguously and could evict the
    wrong rows; id DESC is a stable, deterministic proxy for "more recent"
    among ties since ids only ever increase). The task queue is a transient
    work log, not a history store — whatever a task actually produced
    already lives in its own table (forecasts, forecast_settings, candles);
    a pretest's summary was never persisted anywhere besides the task's own
    terminal WS message in the first place. So dropping old completed task
    rows loses nothing. Only 'done' rows are touched — error/cancelled/
    interrupted rows are left alone (still useful for debugging what went
    wrong, and don't accumulate the way routine successful runs do).
    Returns number of rows deleted.
    """
    cursor = await db.execute(
        """
        DELETE FROM tasks
        WHERE status = 'done'
          AND id NOT IN (
              SELECT id FROM tasks WHERE status = 'done'
              ORDER BY updated_at DESC, id DESC LIMIT ?
          )
        """,
        (keep,),
    )
    await db.commit()
    return cursor.rowcount


# ── app settings ─────────────────────────────────────────────────────────────
# Singleton row (id=1) of global tunables, editable from the left panel (see
# sma/api/routes/settings.py). calibration_workers=0 means "auto" — the
# task_manager's own max(1, cpu_count // 4) formula, kept as the fallback so
# an unconfigured/fresh install behaves exactly as before this setting
# existed.
DEFAULT_MOEX_POOL_WORKERS = 16
DEFAULT_CALIBRATION_WORKERS = 0  # 0 = auto


async def get_app_settings(db: aiosqlite.Connection) -> dict:
    cursor = await db.execute("SELECT moex_pool_workers, calibration_workers FROM app_settings WHERE id = 1")
    row = await cursor.fetchone()
    if row is None:
        return {
            "moex_pool_workers": DEFAULT_MOEX_POOL_WORKERS,
            "calibration_workers": DEFAULT_CALIBRATION_WORKERS,
        }
    return {"moex_pool_workers": row["moex_pool_workers"], "calibration_workers": row["calibration_workers"]}


async def save_app_settings(db: aiosqlite.Connection, moex_pool_workers: int, calibration_workers: int) -> dict:
    await db.execute(
        """
        INSERT INTO app_settings (id, moex_pool_workers, calibration_workers, updated_at)
        VALUES (1, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            moex_pool_workers   = excluded.moex_pool_workers,
            calibration_workers = excluded.calibration_workers,
            updated_at          = excluded.updated_at
        """,
        (moex_pool_workers, calibration_workers, datetime.now(timezone.utc).isoformat()),
    )
    await db.commit()
    return await get_app_settings(db)


# ── internal helpers ──────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
