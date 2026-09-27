"""
range_forecast — прогноз диапазона (min(low)..max(high)) за h=1..H шагов
вперёд. Портировано из prototype/forcaster/ui/app16-range-forecast.py, см.
docs/plans/app16_range_forecast_migration_plan.md.

Калибровка (θ+5λ+read-квантиль, 30-40с на дневных данных) — ЧЕРЕЗ
TaskManager (kind='range_forecast_calibration'), не синхронный HTTP-запрос.
Сам прогноз — синхронный роут (миллисекунды, один поиск соседей + дешёвые
квантили — тот же класс стоимости, что у POST /series/spectrogram).
"""
from __future__ import annotations

import asyncio

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import aiosqlite

from sma.core.db import (
    get_instrument_by_id,
    get_candles,
    get_candle_by_ts,
    create_task,
    list_range_forecast_settings,
    get_range_forecast_settings,
    delete_range_forecast_settings,
)
from sma.core.forecast.range_forecast import (
    RANK_FEATURES, MAX_CANDLES_DEFAULT, dratio_from_close, rolling_cummin_cummax,
    predict_zones_multistep,
)
from sma.core.forecast.band_lambda import compute_bar_rank_dict
from sma.api.deps import get_db, get_task_manager
from sma.api.task_manager import TaskManager

router = APIRouter()

DEFAULT_LEVELS = (50, 75, 90)


class CalibrateRequest(BaseModel):
    instrument_id: int
    interval: str
    h_steps: int = Field(5, ge=1, le=60)
    p: int = Field(8, ge=3, le=150)
    theiler: int = Field(5, ge=1, le=150)
    levels: list[int] = Field(default_factory=lambda: list(DEFAULT_LEVELS))
    max_candles: int = Field(MAX_CANDLES_DEFAULT, ge=0)  # 0 = вся история, без обрезки


class TaskResponse(BaseModel):
    task_id: int


class LiveRequest(BaseModel):
    instrument_id: int
    interval: str
    origin_ts: str | None = None  # None = последний доступный бар
    h_steps: int = Field(5, ge=1, le=60)
    p: int = Field(8, ge=3, le=150)
    theiler: int = Field(5, ge=1, le=150)
    levels: list[int] = Field(default_factory=lambda: list(DEFAULT_LEVELS))
    mode: str = Field("calibrated", pattern="^(calibrated|default)$")
    max_candles: int = Field(MAX_CANDLES_DEFAULT, ge=0)  # 0 = вся история, без обрезки


@router.get("/settings")
async def get_settings_list(
    instrument_id: int,
    interval: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    """Все откалиброванные (h_steps, p, theiler) комбинации для этого
    тикера — список для UI («уже откалибровано под H=5/p=8/theiler=5,
    converged=true, final_ratio=0.56»)."""
    return await list_range_forecast_settings(db, instrument_id, interval)


@router.post("/calibrate", response_model=TaskResponse, status_code=202)
async def calibrate(
    body: CalibrateRequest,
    db: aiosqlite.Connection = Depends(get_db),
    tm: TaskManager = Depends(get_task_manager),
):
    """Ставит в очередь ОДНУ задачу калибровки (нет пула, нет фан-аута по
    нескольким T — один тикер, одна комбинация h_steps/p/theiler за раз)."""
    instr = await get_instrument_by_id(db, body.instrument_id)
    if instr is None:
        raise HTTPException(404, f"Instrument {body.instrument_id} not found")

    label = f"Калибровка диапазона {instr['ticker']} [{body.interval}] H={body.h_steps}/p={body.p}/theiler={body.theiler}"
    task_id = await create_task(
        db, body.instrument_id, body.interval, "",
        {"h_steps": body.h_steps, "p": body.p, "theiler": body.theiler, "levels": body.levels,
         "max_candles": body.max_candles},
        kind="range_forecast_calibration", label=label,
    )
    await tm.submit(task_id)
    return TaskResponse(task_id=task_id)


@router.delete("/settings/{settings_id}", status_code=204)
async def delete_settings(settings_id: int, db: aiosqlite.Connection = Depends(get_db)):
    ok = await delete_range_forecast_settings(db, settings_id)
    if not ok:
        raise HTTPException(404, "Settings row not found")


@router.post("/live")
async def live(
    body: LiveRequest,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Синхронный live-прогноз — один поиск соседей + дешёвые квантильные
    считывания, миллисекунды на дневных данных. mode='calibrated' без
    активной калибровки для (h_steps, p, theiler) → {"calibrated": false,
    ...} с прогнозом на дефолтных θ=0/λ=0 — фронт явно подсвечивает баннер
    «нужна калибровка», НЕ подставляет молча (см. план §8 п.1). mode=
    'default' всегда считает с θ=0/λ=0 и номинальным read-квантилем.
    """
    instr = await get_instrument_by_id(db, body.instrument_id)
    if instr is None:
        raise HTTPException(404, f"Instrument {body.instrument_id} not found")

    candles = await get_candles(db, body.instrument_id, body.interval, until=body.origin_ts)
    if len(candles) < 200:
        raise HTTPException(400, f"Недостаточно данных ({len(candles)} баров)")

    settings_row = None
    if body.mode == "calibrated":
        settings_row = await get_range_forecast_settings(
            db, body.instrument_id, body.interval, body.h_steps, body.p, body.theiler
        )

    calibrated = settings_row is not None
    params = settings_row["params"] if settings_row else {}
    q_read_by_level = (
        {int(k): v for k, v in settings_row["q_read_by_level"].items()} if settings_row else None
    )
    theta = params.get("theta", 0.0)
    lambdas = {f: params.get(f, 0.0) for f in RANK_FEATURES}

    # Обрезаем СПИСОК свечей (не только производные массивы) до последних
    # max_candles баров — иначе индекс origin (посчитанный от обрезанной
    # длины) рассинхронизируется с candles[origin] ниже. max_candles<=0 —
    # вся история (параметр запроса, см. LiveRequest — НЕ молчаливая
    # константа, borrow simplex_ensemble.bars: "точек в библиотеке"/"все
    # точки библиотеки").
    if body.max_candles > 0 and len(candles) > body.max_candles:
        candles = candles[-body.max_candles:]
    close = np.array([c["close"] for c in candles], dtype=np.float64)
    high = np.array([c["high"] for c in candles], dtype=np.float64)
    low = np.array([c["low"] for c in candles], dtype=np.float64)
    volume = np.array([c["volume"] for c in candles], dtype=np.float64)
    n_total = len(close)
    if n_total < 200:
        raise HTTPException(400, f"Недостаточно данных после обрезки ({n_total} баров)")

    loop = asyncio.get_event_loop()

    def _compute():
        dratio = dratio_from_close(close)
        cml, cmh = rolling_cummin_cummax(low, high, body.h_steps)
        rank_dict = None
        if any(lambdas.values()):
            rank_dict = compute_bar_rank_dict(
                np.log(np.maximum(high, 1e-10)), np.log(np.maximum(low, 1e-10)), volume
            )
        origin = n_total - 1
        if origin < body.p + body.theiler + 20:
            return None, None
        res = predict_zones_multistep(
            dratio, close, cml, cmh, origin, body.p, body.theiler, theta,
            tuple(body.levels), body.h_steps, lambdas, rank_dict, q_read_by_level,
        )
        return res, origin

    res, origin = await loop.run_in_executor(None, _compute)
    if res is None:
        raise HTTPException(400, "Пул соседей слишком мал — уменьшите p/theiler или horizon")
    zones_by_step, n_neighbors = res

    origin_candle = candles[origin]
    return {
        "calibrated": calibrated,
        "params": params,
        "origin_date": origin_candle["begin"],
        "close_at_origin": origin_candle["close"],
        "n_neighbors": n_neighbors,
        "zones_by_step": {str(h): {str(lv): list(bounds) for lv, bounds in lv_zones.items()}
                           for h, lv_zones in zones_by_step.items()},
    }
