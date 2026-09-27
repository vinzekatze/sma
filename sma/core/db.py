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
band_lambda_pool — the ONE pool composition for a given (instrument,
                    interval) — categories/n/resolved tickers. T is a free
                    live parameter on every forecast request (no longer a
                    saved dimension — λ-calibration and per-T saved rows
                    were removed 2026-09-12, see memory
                    project_phase7_calibration_removed_final), so only the
                    pool itself still needs persisting.
display_presets  — named visual settings (band levels/opacity), independent
                    of any specific forecast — reused across models.
forecast_defaults — last-used UI params for (instrument, interval, model_type),
                    auto-saved server-side after every successful forecast of
                    that model_type and auto-loaded to prefill the form next
                    time — simplex_ensemble and band_lambda (t_query/m/theta)
                    both write this now.
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
pool_resolution_cache — TTL cache of resolve_pool_candidates()'s result,
                    keyed by pool_key (categories+n) — avoids re-hitting MOEX
                    (category search + per-candidate liquidity ranking) on
                    every incidental pool resolution; the settings modal's
                    explicit "Обновить по категориям" button bypasses this
                    (force=True) since that's the user's deliberate refresh.
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

CREATE TABLE IF NOT EXISTS band_lambda_pool (
    instrument_id     INTEGER NOT NULL REFERENCES instruments(id),
    interval          TEXT    NOT NULL,
    pool_config_json  TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL,
    PRIMARY KEY (instrument_id, interval)
);

CREATE TABLE IF NOT EXISTS range_forecast_settings (
    id                    INTEGER PRIMARY KEY,
    instrument_id         INTEGER NOT NULL REFERENCES instruments(id),
    interval              TEXT    NOT NULL,
    h_steps               INTEGER NOT NULL,
    p                     INTEGER NOT NULL,
    theiler               INTEGER NOT NULL,
    params_json           TEXT    NOT NULL,
    feature_order_json    TEXT    NOT NULL,
    q_read_by_level_json  TEXT    NOT NULL,
    calibration_meta_json TEXT    NOT NULL,
    is_active             INTEGER NOT NULL DEFAULT 1,
    UNIQUE(instrument_id, interval, h_steps, p, theiler)
);
CREATE INDEX IF NOT EXISTS idx_range_forecast_settings_lookup
    ON range_forecast_settings(instrument_id, interval);

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

CREATE TABLE IF NOT EXISTS pool_resolution_cache (
    pool_key        TEXT    PRIMARY KEY,
    candidates_json TEXT    NOT NULL,
    computed_at     TEXT    NOT NULL
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
    "ALTER TABLE app_settings     ADD COLUMN color_profile_json TEXT",
    "ALTER TABLE app_settings     ADD COLUMN chart_window_bars  INTEGER NOT NULL DEFAULT 1000",
    "ALTER TABLE display_presets  ADD COLUMN trade_level_pct    REAL    NOT NULL DEFAULT 70",
    "ALTER TABLE display_presets  ADD COLUMN show_zones         INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE display_presets  ADD COLUMN show_trade_level   INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE display_presets  ADD COLUMN trim_zone1         INTEGER NOT NULL DEFAULT 0",
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


async def _drop_forecast_settings_table(db: aiosqlite.Connection) -> None:
    """
    forecast_settings held one row per (instrument, interval, model_type,
    t_query, pool_key) — band_lambda's per-T saved calibration result.
    Removed 2026-09-12 alongside λ-calibration itself and the "pick a
    pre-saved T" UI: T is now a free live parameter on every forecast
    request, and only the pool composition still needs persisting (see
    band_lambda_pool above). Drop unconditionally; nothing reads this
    table's contents any more.
    """
    await db.execute("DROP TABLE IF EXISTS forecast_settings")
    await db.commit()


async def init_db(path: str | Path) -> None:
    """Create tables if they don't exist yet; run safe column migrations."""
    async with aiosqlite.connect(path) as db:
        await _drop_stale_forecasts_table(db)
        await _drop_series_settings_table(db)
        await _drop_forecast_settings_table(db)
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
    limit: int | None = None,
) -> list[dict]:
    """Return candles ordered by begin asc, optionally filtered by date range.

    `limit` takes the LAST `limit` rows matching since/until (windowed chart
    loading, sma/ui/candle_window.js) — implemented as ORDER BY begin DESC
    LIMIT ? then reversed back to asc, since a plain ASC+LIMIT would instead
    take the OLDEST rows in range, not the most recent."""
    q = "SELECT * FROM candles WHERE instrument_id=? AND interval=?"
    args: list[Any] = [instrument_id, interval]
    if since:
        q += " AND begin >= ?"; args.append(since)
    if until:
        q += " AND begin <= ?"; args.append(until)
    if limit:
        q += " ORDER BY begin DESC LIMIT ?"; args.append(limit)
        cursor = await db.execute(q, args)
        return [dict(r) for r in reversed(await cursor.fetchall())]
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


async def get_last_fetched_at(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
) -> str | None:
    """
    Most recent `fetched_at` among this instrument+interval's candles — i.e.
    "when did we last successfully hit MOEX for this data", used by
    candle_fetch.queue_pool_candle_fetch to skip re-queuing a fetch that
    just happened (see FRESHNESS_TTL_SECONDS there).
    """
    cursor = await db.execute(
        "SELECT MAX(fetched_at) FROM candles WHERE instrument_id=? AND interval=?",
        (instrument_id, interval),
    )
    row = await cursor.fetchone()
    return row[0] if row else None


# ── pool resolution cache ─────────────────────────────────────────────────────
# Caches resolve_pool_candidates()'s expensive MOEX category-search +
# liquidity-ranking pass (sma/core/forecast/pool_selection.py) by pool_key
# (categories+n) — see _resolve_and_upsert_pool in
# sma/api/routes/forecast_settings.py for the freshness check that decides
# whether to use this or recompute.

async def get_pool_resolution_cache(
    db: aiosqlite.Connection,
    pool_key: str,
) -> dict | None:
    cursor = await db.execute(
        "SELECT candidates_json, computed_at FROM pool_resolution_cache WHERE pool_key=?",
        (pool_key,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {"candidates": json.loads(row["candidates_json"]), "computed_at": row["computed_at"]}


async def upsert_pool_resolution_cache(
    db: aiosqlite.Connection,
    pool_key: str,
    candidates: list[dict],
) -> None:
    await db.execute(
        """
        INSERT INTO pool_resolution_cache (pool_key, candidates_json, computed_at)
        VALUES (?, ?, ?)
        ON CONFLICT(pool_key) DO UPDATE SET
            candidates_json = excluded.candidates_json,
            computed_at     = excluded.computed_at
        """,
        (pool_key, json.dumps(candidates, ensure_ascii=False), _utcnow()),
    )
    await db.commit()


# ── forecasts ─────────────────────────────────────────────────────────────────

async def save_forecast(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    model_type: str,
    origin_candle_id: int,
    params: dict,   # band_lambda: {t_query, m, theta, min_bars}; simplex_ensemble: its own param set
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
        SELECT id, created_at, model_type,
               json_extract(result_json, '$.origin_date')         AS origin_ts,
               json_extract(result_json, '$.origin_extreme_date') AS origin_extreme_ts,
               json_extract(result_json, '$.origin_price')        AS origin_price,
               json_extract(result_json, '$.origin_direction')    AS origin_direction,
               json_extract(params_json, '$.t_query')             AS t_query
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


# ── band_lambda pool config ─────────────────────────────────────────────────
# Replaces the old per-T "forecast_settings" (calibration result) rows —
# removed 2026-09-12 along with λ-calibration itself (see memory
# project_phase7_calibration_removed_final): T is now a free live parameter
# on every forecast request (see sma/api/routes/forecasts.py), not something
# that needs its own saved row, so only the POOL COMPOSITION (which doesn't
# vary by T) still needs persisting, one row per (instrument, interval).

async def get_band_lambda_pool(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
) -> dict | None:
    cursor = await db.execute(
        "SELECT pool_config_json FROM band_lambda_pool WHERE instrument_id=? AND interval=?",
        (instrument_id, interval),
    )
    row = await cursor.fetchone()
    return json.loads(row["pool_config_json"]) if row else None


async def upsert_band_lambda_pool(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    pool_config: dict,
) -> None:
    await db.execute(
        """
        INSERT INTO band_lambda_pool (instrument_id, interval, pool_config_json, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(instrument_id, interval) DO UPDATE SET
            pool_config_json = excluded.pool_config_json,
            updated_at       = excluded.updated_at
        """,
        (instrument_id, interval, json.dumps(pool_config, ensure_ascii=False), _utcnow()),
    )
    await db.commit()


# ── range forecast settings (calibration results — see docs/plans/
#    app16_range_forecast_migration_plan.md §3: no pool, no zone_geometry —
#    UNIQUE(instrument, interval, h_steps, p, theiler) means exactly one row
#    per combo, always active; is_active kept for future-proofing only) ────

def _range_forecast_settings_row(row: dict) -> dict:
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json"))
    d["feature_order"] = json.loads(d.pop("feature_order_json"))
    d["q_read_by_level"] = json.loads(d.pop("q_read_by_level_json"))
    d["calibration_meta"] = json.loads(d.pop("calibration_meta_json"))
    d["is_active"] = bool(d["is_active"])
    return d


async def upsert_range_forecast_settings(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    h_steps: int,
    p: int,
    theiler: int,
    params: dict,
    feature_order: list,
    q_read_by_level: dict,
    calibration_meta: dict,
) -> int:
    """Insert or replace the ONE calibration row for (instrument, interval,
    h_steps, p, theiler) — re-calibrating the same combo overwrites it."""
    await db.execute(
        """
        INSERT INTO range_forecast_settings (
            instrument_id, interval, h_steps, p, theiler,
            params_json, feature_order_json, q_read_by_level_json,
            calibration_meta_json, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(instrument_id, interval, h_steps, p, theiler) DO UPDATE SET
            params_json           = excluded.params_json,
            feature_order_json    = excluded.feature_order_json,
            q_read_by_level_json  = excluded.q_read_by_level_json,
            calibration_meta_json = excluded.calibration_meta_json
        """,
        (
            instrument_id, interval, h_steps, p, theiler,
            json.dumps(params, ensure_ascii=False), json.dumps(feature_order, ensure_ascii=False),
            json.dumps(q_read_by_level, ensure_ascii=False), json.dumps(calibration_meta, ensure_ascii=False),
        ),
    )
    await db.commit()
    cursor = await db.execute(
        """SELECT id FROM range_forecast_settings
           WHERE instrument_id=? AND interval=? AND h_steps=? AND p=? AND theiler=?""",
        (instrument_id, interval, h_steps, p, theiler),
    )
    return (await cursor.fetchone())["id"]


async def get_range_forecast_settings(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
    h_steps: int,
    p: int,
    theiler: int,
) -> dict | None:
    cursor = await db.execute(
        """SELECT * FROM range_forecast_settings
           WHERE instrument_id=? AND interval=? AND h_steps=? AND p=? AND theiler=? AND is_active=1""",
        (instrument_id, interval, h_steps, p, theiler),
    )
    row = await cursor.fetchone()
    return _range_forecast_settings_row(dict(row)) if row else None


async def list_range_forecast_settings(
    db: aiosqlite.Connection,
    instrument_id: int,
    interval: str,
) -> list[dict]:
    """All calibrated (h_steps, p, theiler) combinations for this ticker —
    powers the settings list UI ("уже калибровано под H=5/p=8/theiler=5...")."""
    cursor = await db.execute(
        """SELECT * FROM range_forecast_settings
           WHERE instrument_id=? AND interval=?
           ORDER BY h_steps, p, theiler""",
        (instrument_id, interval),
    )
    return [_range_forecast_settings_row(dict(r)) for r in await cursor.fetchall()]


async def delete_range_forecast_settings(
    db: aiosqlite.Connection,
    settings_id: int,
) -> bool:
    cursor = await db.execute("DELETE FROM range_forecast_settings WHERE id=?", (settings_id,))
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
        r["show_zones"] = bool(r["show_zones"])
        r["show_trade_level"] = bool(r["show_trade_level"])
        r["trim_zone1"] = bool(r["trim_zone1"])
    return rows


async def save_display_preset(
    db: aiosqlite.Connection,
    model_type: str,
    name: str,
    levels: list[float],
    opacity: float,
    is_default: bool = False,
    trade_level_pct: float = 70,
    show_zones: bool = True,
    show_trade_level: bool = True,
    trim_zone1: bool = False,
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
        INSERT INTO display_presets (
            model_type, name, levels_json, opacity, is_default, created_at,
            trade_level_pct, show_zones, show_trade_level, trim_zone1
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_type, name) DO UPDATE SET
            levels_json      = excluded.levels_json,
            opacity          = excluded.opacity,
            is_default       = excluded.is_default,
            trade_level_pct  = excluded.trade_level_pct,
            show_zones       = excluded.show_zones,
            show_trade_level = excluded.show_trade_level,
            trim_zone1       = excluded.trim_zone1
        """,
        (
            model_type, name, json.dumps(levels), opacity, 1 if is_default else 0, _utcnow(),
            trade_level_pct, 1 if show_zones else 0, 1 if show_trade_level else 0, 1 if trim_zone1 else 0,
        ),
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


async def delete_task(db: aiosqlite.Connection, task_id: int) -> bool:
    """
    Manual delete for one task row — terminal states only (see
    sma/api/routes/tasks.py, which rejects pending/running before calling
    this). Unlike prune_old_done_tasks (auto, 'done' only, keeps the last
    N), cancelled/interrupted/error rows are never auto-pruned — without
    this they'd accumulate forever (reported 2026-09-12: a cancelled task
    had no way to be removed from the list).
    """
    cursor = await db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    await db.commit()
    return cursor.rowcount > 0


async def prune_old_done_tasks(db: aiosqlite.Connection, keep: int = 10) -> int:
    """
    Deletes 'done' task rows beyond the most recent `keep` (by updated_at,
    id as tie-break — updated_at has only second resolution, so a batch of
    tasks finishing within the same second, e.g. a pool save's batch of
    candle_fetch tasks, would otherwise sort ambiguously and could evict the
    wrong rows; id DESC is a stable, deterministic proxy for "more recent"
    among ties since ids only ever increase). The task queue is a transient
    work log, not a history store — whatever a task actually produced
    already lives in its own table (forecasts, band_lambda_pool, candles);
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
DEFAULT_CHART_WINDOW_BARS = 1000  # sma/ui/candle_window.js — bars kept loaded on the main chart at once; 1000 confirmed noticeably smoother in Firefox than the initial 3000

# Colour profile (docs/plans/frontend_improvements_plan.md §1.1a, expanded
# 2026-08-25 to cover every tool-meaningful color in the app — project
# feedback: "можно расширить и позволить вообще все используемые
# инструментами цвета настраивать") — fixed, named roles (not one entry per
# user-created object: one price-level color for ALL levels, one palette
# cycled by index for ALL MA/zigzag_tool series, etc.). Values here are
# exactly the hardcoded colors each frontend module used before this existed
# (or before its role was added) — so a fresh install renders identically to
# before the profile existed, only now overridable from the left panel's
# "Цветовой профиль" section (sma/ui/settings.js — see COLOR_ROLES there for
# the single source of truth on label/grouping; keys here and there MUST
# match). "Chrome" colors (backgrounds, grid lines, borders, legend/tooltip
# styling) are deliberately NOT roles — only colors that carry analytical
# meaning (direction, series identity, model identity) are.
DEFAULT_COLOR_PROFILE: dict[str, Any] = {
    # ── Основной график ──
    "price_level": "#ff0000",
    "next_origin_marker": "#58a6ff",
    "trend_ruler_line": "#1f77b4",
    "trend_ruler_origin": "#bc8cff",
    "trend_ruler_accel_up": "#2ca02c",
    "trend_ruler_accel_down": "#d62728",
    "trend_ruler_accel_line": "#7f7f7f",
    "ma_palette": ["#f0883e", "#a5d6ff", "#d2a8ff", "#7ee787", "#ffa198", "#79c0ff"],
    "zigzag_tool_palette": ["#d29922", "#f0883e", "#a5d6ff", "#7ee787", "#ffa198", "#d2a8ff"],
    "forecast_zigzag": "#d29922",
    "band_zone_up": "#3fb950",
    "band_zone_down": "#f85149",
    "candle_up": "#3fb950",
    "candle_down": "#f85149",
    "forecast_marker_selected": "#ffd600",
    "forecast_marker_pinned_band_lambda": "#58a6ff",
    "forecast_marker_pinned_simplex_ensemble": "#f0883e",
    "forecast_marker_pinned_regime_mixture_potential": "#d2a8ff",
    "simplex_origin_lines": "#64b4ff",
    "simplex_mean_band": "#ffd600",
    "range_forecast_line": "#ffd600",
    "risk_corridor_close": "#ffffff",
    "risk_corridor_high": "#26a69a",
    "risk_corridor_low": "#ef5350",
    # ── Осциллятор ──
    "spectrogram_colorscale": "Viridis",
    "spectrogram_cutoff_line": "#ff5050",
    "variance_slope_up": "#2ca02c",
    "variance_slope_down": "#d62728",
    "variance_var": "#9467bd",
    "volume_up": "#3fb950",
    "volume_down": "#f85149",
}


def _parse_color_profile(raw: str | None) -> dict:
    """Merge stored JSON over the defaults so a NEW role added later (e.g. a
    future forecast model) always has a value even for rows saved before it
    existed — never crashes/omits a key just because an old row predates it."""
    stored = {}
    if raw:
        try:
            stored = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            stored = {}
    return {**DEFAULT_COLOR_PROFILE, **stored}


async def get_app_settings(db: aiosqlite.Connection) -> dict:
    cursor = await db.execute(
        "SELECT moex_pool_workers, calibration_workers, chart_window_bars, color_profile_json FROM app_settings WHERE id = 1"
    )
    row = await cursor.fetchone()
    if row is None:
        return {
            "moex_pool_workers": DEFAULT_MOEX_POOL_WORKERS,
            "calibration_workers": DEFAULT_CALIBRATION_WORKERS,
            "chart_window_bars": DEFAULT_CHART_WINDOW_BARS,
            "color_profile": dict(DEFAULT_COLOR_PROFILE),
        }
    return {
        "moex_pool_workers": row["moex_pool_workers"],
        "calibration_workers": row["calibration_workers"],
        "chart_window_bars": row["chart_window_bars"],
        "color_profile": _parse_color_profile(row["color_profile_json"]),
    }


async def save_app_settings(
    db: aiosqlite.Connection,
    moex_pool_workers: int,
    calibration_workers: int,
    chart_window_bars: int,
    color_profile: dict | None = None,
) -> dict:
    # color_profile=None means "leave whatever is already stored alone" —
    # save_app_settings is also called from the moex/calibration-workers
    # form (settings.js), which never sends a color profile and must not
    # blow the stored one away with defaults.
    if color_profile is None:
        existing = await get_app_settings(db)
        color_profile = existing["color_profile"]
    profile_json = json.dumps({**DEFAULT_COLOR_PROFILE, **color_profile})
    await db.execute(
        """
        INSERT INTO app_settings (id, moex_pool_workers, calibration_workers, chart_window_bars, color_profile_json, updated_at)
        VALUES (1, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            moex_pool_workers   = excluded.moex_pool_workers,
            calibration_workers = excluded.calibration_workers,
            chart_window_bars   = excluded.chart_window_bars,
            color_profile_json  = excluded.color_profile_json,
            updated_at          = excluded.updated_at
        """,
        (moex_pool_workers, calibration_workers, chart_window_bars, profile_json, datetime.now(timezone.utc).isoformat()),
    )
    await db.commit()
    return await get_app_settings(db)


# ── internal helpers ──────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
