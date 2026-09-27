from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import aiosqlite

from sma.core.db import (
    get_instrument_by_id,
    get_candles,
    get_candle_by_id,
    create_task,
    delete_forecast,
    get_forecast,
    list_forecasts,
    update_forecast_geometry,
    get_band_lambda_pool,
)
from sma.core.candle_fetch import queue_pool_candle_fetch
from sma.core.forecast import band_lambda as bl
from sma.api.deps import get_db, get_task_manager
from sma.api.task_manager import TaskManager

router = APIRouter()


class ForecastRequest(BaseModel):
    instrument_id: int
    interval: str
    model_type: str = "band_lambda"
    t_query: float | None = None                 # band_lambda only — free live parameter, no calibration/saved-T lookup any more
    m: int | None = None                          # band_lambda only
    theta: float | None = None                    # band_lambda only
    min_bars: int | None = None                   # band_lambda only
    origin_candle_id: int | None = None          # None = live ("as of now")
    params: dict | None = None                   # simplex_ensemble/regime_mixture_potential only — see
    # sma/core/forecast/simplex_ensemble.py:forecast_ensemble for the accepted
    # keys (window/horizon/xy_x/xy_y/xi_add/blend_alpha/n_levels/
    # p_cascade_max/bars/pca_p_range/pca_thr_range/use_lp_corr), or
    # sma/core/forecast/regime_mixture_potential.py:forecast_regime_mixture_
    # potential (horizon/n_forecasts/rewind_step/theta/warmup/theiler_window/
    # bars/n_lookback/lookback_step/n_sim/seed/mix_n_resample/bin_height_pct/
    # coverage_pct). No settings row for either model — params travel
    # straight in the task, same as band_lambda now does.


class ForecastTaskResponse(BaseModel):
    task_id: int


class GeometryRequest(BaseModel):
    zone_geometry: dict


@router.post("", response_model=ForecastTaskResponse, status_code=202)
async def request_forecast(
    body: ForecastRequest,
    db: aiosqlite.Connection = Depends(get_db),
    tm: TaskManager = Depends(get_task_manager),
):
    """
    Create a background forecast task. Poll GET /tasks/{task_id} or connect
    to WS /tasks/{task_id}/ws for progress/result (forecast_id).

    band_lambda's pool is whatever was last saved via POST /forecast-
    settings/pool for this (instrument, interval) — T/m/theta are free live
    parameters on every request instead (no more per-T saved row, see
    memory project_phase7_calibration_removed_final). Pool DATA is still
    refreshed here: one freshness-aware candle_fetch task per pool ticker is
    queued ahead of the forecast task itself (sma/core/candle_fetch.py).
    """
    instr = await get_instrument_by_id(db, body.instrument_id)
    if instr is None:
        raise HTTPException(404, f"Instrument {body.instrument_id} not found")

    if body.model_type == "simplex_ensemble":
        origin_ts = ""
        if body.origin_candle_id is not None:
            origin_candle = await get_candle_by_id(db, body.origin_candle_id)
            if origin_candle is None or origin_candle["instrument_id"] != body.instrument_id:
                raise HTTPException(404, "origin_candle_id not found for this instrument")
            origin_ts = origin_candle["begin"]

        # No pool (single ticker) — just make sure the target's own candles
        # are current, same freshness guarantee band_lambda gives its pool.
        await queue_pool_candle_fetch(db, tm, [instr], body.interval)

        label = f"Ансамбль Simplex {instr['ticker']} [{body.interval}]"
        task_id = await create_task(
            db, body.instrument_id, body.interval, origin_ts,
            {"model_type": "simplex_ensemble", **(body.params or {})},
            kind="forecast", label=label,
        )
        await tm.submit(task_id)
        return ForecastTaskResponse(task_id=task_id)

    if body.model_type == "regime_mixture_potential":
        origin_ts = ""
        if body.origin_candle_id is not None:
            origin_candle = await get_candle_by_id(db, body.origin_candle_id)
            if origin_candle is None or origin_candle["instrument_id"] != body.instrument_id:
                raise HTTPException(404, "origin_candle_id not found for this instrument")
            origin_ts = origin_candle["begin"]

        # No pool (single ticker) — same freshness guarantee as simplex_ensemble.
        await queue_pool_candle_fetch(db, tm, [instr], body.interval)

        label = f"Потенциал {instr['ticker']} [{body.interval}]"
        task_id = await create_task(
            db, body.instrument_id, body.interval, origin_ts,
            {"model_type": "regime_mixture_potential", **(body.params or {})},
            kind="forecast", label=label,
        )
        await tm.submit(task_id)
        return ForecastTaskResponse(task_id=task_id)

    if body.t_query is None:
        raise HTTPException(400, "t_query is required for band_lambda")

    pool_config = await get_band_lambda_pool(db, body.instrument_id, body.interval)
    if pool_config is None:
        raise HTTPException(400, "Пул не настроен для этого инструмента — откройте «Пул…» в «Параметры»")

    origin_ts = ""
    if body.origin_candle_id is not None:
        origin_candle = await get_candle_by_id(db, body.origin_candle_id)
        if origin_candle is None or origin_candle["instrument_id"] != body.instrument_id:
            raise HTTPException(404, "origin_candle_id not found for this instrument")
        origin_ts = origin_candle["begin"]

    resolved_ids = pool_config["resolved_instrument_ids"]
    pool_rows = [r for iid in resolved_ids if (r := await get_instrument_by_id(db, iid)) is not None]
    await queue_pool_candle_fetch(db, tm, pool_rows, body.interval)

    m = body.m if body.m is not None else bl.DEFAULT_M
    theta = body.theta if body.theta is not None else bl.DEFAULT_THETA
    min_bars = body.min_bars if body.min_bars is not None else bl.DEFAULT_MIN_BARS

    label = f"Прогноз {instr['ticker']} [{body.interval}] T={body.t_query*100:.0f}%"
    task_id = await create_task(
        db, body.instrument_id, body.interval, origin_ts,
        {"t_query": body.t_query, "m": m, "theta": theta, "min_bars": min_bars, "pool": pool_config},
        kind="forecast", label=label,
    )
    await tm.submit(task_id)
    return ForecastTaskResponse(task_id=task_id)


@router.get("/zigzag")
async def get_zigzag(
    instrument_id: int = Query(...),
    interval: str = Query(...),
    t_query: float = Query(...),
    min_bars: int = Query(5, ge=0),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Every pivot of the target's own zigzag at (t_query, min_bars) — powers
    the chart's zigzag overlay (docs/plans/band_forecast_migration_plan.md
    5.1: "Зигзаг для отображения... пересчитывается на лету из
    params_json.t_query/min_bars при каждом рендере"). Pivots are drawn at
    extreme_date (where the reversal actually printed); the forecast origin
    marker uses confirm_date instead — see build_zigzag's docstring for why
    those differ and which one is causally valid.
    """
    candles = await get_candles(db, instrument_id, interval)
    if not candles:
        raise HTTPException(404, "No candles available")

    def _run():
        import numpy as np
        from sma.core.forecast.band_lambda import build_zigzag
        from sma.core.forecast.pivot_time_band import zigzag_direction_stats

        log_highs = np.log(np.array([c["high"] for c in candles], dtype=np.float64))
        log_lows = np.log(np.array([c["low"] for c in candles], dtype=np.float64))
        dates = np.array([c["begin"] for c in candles])
        prices, extreme_dates, confirm_dates, directions = build_zigzag(log_highs, log_lows, dates, t_query, min_bars)
        pivots = [
            {
                "extreme_date": str(extreme_dates[i]),
                "confirm_date": str(confirm_dates[i]),
                "price": float(np.exp(prices[i])),
                "direction": int(directions[i]),
            }
            for i in range(len(prices))
        ]
        stats = zigzag_direction_stats(prices, dates, confirm_dates, directions)
        return pivots, stats

    loop = asyncio.get_running_loop()
    pivots, stats = await loop.run_in_executor(None, _run)
    return {"pivots": pivots, "stats": stats}


@router.get("/{forecast_id}")
async def get_forecast_result(
    forecast_id: int,
    db: aiosqlite.Connection = Depends(get_db),
):
    row = await get_forecast(db, forecast_id)
    if row is None:
        raise HTTPException(404, "Forecast not found")
    return row


@router.delete("/{forecast_id}", status_code=204)
async def remove_forecast(
    forecast_id: int,
    db: aiosqlite.Connection = Depends(get_db),
):
    deleted = await delete_forecast(db, forecast_id)
    if not deleted:
        raise HTTPException(404, "Forecast not found")


@router.post("/{forecast_id}/geometry")
async def set_forecast_geometry(
    forecast_id: int,
    body: GeometryRequest,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Upsert the drag-resized band shape geometry (the ONLY editable part of a
    forecast — see docs/plans/band_forecast_migration_plan.md 5.1). No
    PATCH semantics anywhere else in this codebase, so this follows the
    existing upsert-via-POST convention.
    """
    ok = await update_forecast_geometry(db, forecast_id, body.zone_geometry)
    if not ok:
        raise HTTPException(404, "Forecast not found")
    return {"ok": True}


@router.get("")
async def list_forecast_summaries(
    instrument_id: int = Query(...),
    interval: str = Query(...),
    model_type: str | None = Query(None),
    limit: int = Query(20, ge=1, le=200),
    db: aiosqlite.Connection = Depends(get_db),
):
    return await list_forecasts(db, instrument_id, interval, model_type=model_type, limit=limit)
