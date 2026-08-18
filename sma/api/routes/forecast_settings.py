from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import aiosqlite

from sma.core.db import (
    get_instrument_by_id,
    upsert_instrument,
    create_task,
    list_forecast_settings,
    activate_forecast_settings,
    delete_forecast_settings,
    get_app_settings,
    get_forecast_defaults,
)
from sma.core.candle_fetch import queue_pool_candle_fetch
from sma.core.forecast.pool_selection import resolve_pool_candidates, compute_pool_key
from sma.api.deps import get_db, get_task_manager
from sma.api.task_manager import TaskManager

router = APIRouter()


class PoolSpec(BaseModel):
    categories: list[str]
    n: int = 25


class CalibrationTarget(BaseModel):
    t_query: float
    min_bars: int = 5


class CalibrateRequest(BaseModel):
    instrument_id: int
    interval: str
    model_type: str = "band_lambda"
    targets: list[CalibrationTarget]
    m: int = 6
    theta: float = 0.0
    pool: PoolSpec


class PretestRequest(BaseModel):
    instrument_id: int
    interval: str
    pool: PoolSpec
    t_query: float
    min_bars: int = 5
    level: int = 1   # 1 = instant (no download), 2 = real download + build_zigzag


class TaskResponse(BaseModel):
    task_id: int


async def _resolve_and_upsert_pool(db, categories: list[str], n: int) -> tuple[list[dict], dict]:
    """
    resolve_pool_candidates hits MOEX ISS synchronously (blocking requests
    calls) — run off the event loop. Each candidate is upserted as
    added_via='pool' (see docs/plans/band_forecast_migration_plan.md 5.2a) —
    this never downgrades an existing 'manual' instrument, and existing
    'pool'/'manual' rows are reused as-is (upsert, not insert-only).

    Returns (instrument_rows, pool_config) where pool_config is ready to
    store verbatim in forecast_settings.pool_config_json / task params.
    """
    app_settings = await get_app_settings(db)
    loop = asyncio.get_running_loop()
    candidates = await loop.run_in_executor(
        None, resolve_pool_candidates, categories, n, app_settings["moex_pool_workers"]
    )

    rows = []
    for c in candidates:
        iid = await upsert_instrument(
            db, c["secid"], "moex", c["asset_type"], c.get("name") or c.get("shortname"),
            engine=c["engine"], market=c["market"], board=c.get("board"),
            added_via="pool",
        )
        rows.append(await get_instrument_by_id(db, iid))

    pool_config = {
        "categories": categories, "n": n,
        "resolved_instrument_ids": [r["id"] for r in rows],
        "pool_key": compute_pool_key(categories, n),
    }
    return rows, pool_config


@router.get("")
async def get_forecast_settings_list(
    instrument_id: int = Query(...),
    interval: str = Query(...),
    model_type: str = Query("band_lambda"),
    db: aiosqlite.Connection = Depends(get_db),
):
    """All calibrated (T, pool_key) rows — powers the quick T-selector; if a
    T has multiple pool_key variants, the frontend shows the is_active one
    by default and the rest behind a "N more" toggle."""
    return await list_forecast_settings(db, instrument_id, interval, model_type)


@router.get("/defaults")
async def get_defaults(
    instrument_id: int = Query(...),
    interval: str = Query(...),
    model_type: str = Query(...),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Last-used UI params for this (instrument, interval, model_type) — see
    db.py forecast_defaults table docstring. Currently only written by
    simplex_ensemble (server-side, after a forecast actually completes);
    band_lambda has no writer for this table and this will just return null
    for it. `{"params": null}` (not 404) when nothing is saved yet — the
    frontend falls back to hardcoded defaults in that case.
    """
    row = await get_forecast_defaults(db, instrument_id, interval, model_type)
    return {"params": row["params"] if row else None}


@router.post("/calibrate", response_model=TaskResponse, status_code=202)
async def calibrate(
    body: CalibrateRequest,
    db: aiosqlite.Connection = Depends(get_db),
    tm: TaskManager = Depends(get_task_manager),
):
    """
    Resolves the pool NOW (category -> candidates -> liquidity ranking,
    see pool_selection.py) — cheap, no λ computed here — then queues one
    candle_fetch task per pool ticker, followed by the calibration task
    itself. Returns the calibration task's id; the candle_fetch tasks are
    independent queue entries (see GET /tasks), not reported here.
    """
    instr = await get_instrument_by_id(db, body.instrument_id)
    if instr is None:
        raise HTTPException(404, f"Instrument {body.instrument_id} not found")

    pool_rows, pool_config = await _resolve_and_upsert_pool(db, body.pool.categories, body.pool.n)
    if not pool_rows:
        raise HTTPException(422, f"Пул пуст — ни один кандидат не прошёл порог ликвидности для категорий {body.pool.categories}")

    await queue_pool_candle_fetch(db, tm, pool_rows, body.interval)

    ts = ", ".join(f"{t.t_query*100:.0f}%" for t in body.targets)
    label = f"Калибровка {instr['ticker']} [{body.interval}] T={ts}"
    task_id = await create_task(
        db, body.instrument_id, body.interval, "",
        {
            "model_type": body.model_type,
            "targets": [t.model_dump() for t in body.targets],
            "m": body.m, "theta": body.theta,
            "pool": pool_config,
        },
        kind="calibration", label=label,
    )
    await tm.submit(task_id)
    return TaskResponse(task_id=task_id)


@router.post("/pretest", response_model=TaskResponse, status_code=202)
async def pretest(
    body: PretestRequest,
    db: aiosqlite.Connection = Depends(get_db),
    tm: TaskManager = Depends(get_task_manager),
):
    """
    Diagnostic-only — does NOT write forecast_settings. Level 1 doesn't
    even touch `instruments` (no candle download, just a history-range
    check per candidate); level 2 downloads real data (added_via='pool',
    reused by a later /calibrate on the same pool) and counts actual zigzag
    events, skipping the expensive λ optimization (see docs/plans/
    band_forecast_migration_plan.md 5.2b). Result arrives only via the
    task's WS/`done` event (summary field) — it is not persisted anywhere
    else, by design (this is meant to inform an immediate go/no-go decision).
    """
    instr = await get_instrument_by_id(db, body.instrument_id)
    if instr is None:
        raise HTTPException(404, f"Instrument {body.instrument_id} not found")

    if body.level == 1:
        app_settings = await get_app_settings(db)
        loop = asyncio.get_running_loop()
        candidates = await loop.run_in_executor(
            None, resolve_pool_candidates, body.pool.categories, body.pool.n, app_settings["moex_pool_workers"]
        )
        params = {
            "level": 1,
            "candidates": [
                {"secid": c["secid"], "engine": c["engine"], "market": c["market"], "board": c.get("board")}
                for c in candidates
            ],
        }
        label = f"Претест (ур.1) {instr['ticker']} — {'+'.join(body.pool.categories)}"
    elif body.level == 2:
        pool_rows, pool_config = await _resolve_and_upsert_pool(db, body.pool.categories, body.pool.n)
        if not pool_rows:
            raise HTTPException(422, f"Пул пуст для категорий {body.pool.categories}")
        await queue_pool_candle_fetch(db, tm, pool_rows, body.interval)
        params = {"level": 2, "pool": pool_config, "t_query": body.t_query, "min_bars": body.min_bars}
        label = f"Претест (ур.2) {instr['ticker']} T={body.t_query*100:.0f}% — {'+'.join(body.pool.categories)}"
    else:
        raise HTTPException(422, f"level must be 1 or 2, got {body.level}")

    task_id = await create_task(
        db, body.instrument_id, body.interval, "", params,
        kind="pretest", label=label,
    )
    await tm.submit(task_id)
    return TaskResponse(task_id=task_id)


@router.post("/{settings_id}/activate")
async def activate(
    settings_id: int,
    db: aiosqlite.Connection = Depends(get_db),
):
    ok = await activate_forecast_settings(db, settings_id)
    if not ok:
        raise HTTPException(404, "forecast_settings not found")
    return {"ok": True}


@router.delete("/{settings_id}", status_code=204)
async def remove(
    settings_id: int,
    db: aiosqlite.Connection = Depends(get_db),
):
    ok = await delete_forecast_settings(db, settings_id)
    if not ok:
        raise HTTPException(404, "forecast_settings not found")
