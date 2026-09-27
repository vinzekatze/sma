"""
risk_corridor — риск-корридор High/Low + полоса Close, портировано из
prototype/forcaster/ui/app27-risk-corridor.py, см. docs/plans/
app27_risk_corridor_migration_plan.md.

В отличие от band_lambda/simplex_ensemble (async, forecasts-таблица) и даже
range_forecast (калибровка через TaskManager) — у этого инструмента НЕТ
персистентности и НЕТ калибровки вообще: параметры глобальны, валидированы
честным temporal walk-forward и устойчиво обобщаются на новые тикеры (эксп.06
фазы 19). Единственный роут — синхронный live-прогноз, run_in_executor
(чистый numpy-расчёт вне event loop, тот же паттерн, что и POST /range-
forecast/live и POST /series/spectrogram).
"""
from __future__ import annotations

import asyncio

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import aiosqlite

from sma.core.db import get_instrument_by_id, get_candles
from sma.core.forecast.risk_corridor import (
    compute_risk_corridor,
    DEFAULT_P_FIT, DEFAULT_BLEND_ALPHA, DEFAULT_THETA, DEFAULT_N_SIM,
    DEFAULT_H, DEFAULT_COVERAGE_PCT, DEFAULT_SEED, MAX_CANDLES_DEFAULT,
)
from sma.api.deps import get_db

router = APIRouter()


class LiveRequest(BaseModel):
    instrument_id: int
    interval: str
    origin_ts: str | None = None  # None = последний доступный бар
    h: int = Field(DEFAULT_H, ge=1, le=60)
    p_fit: int = Field(DEFAULT_P_FIT, ge=5, le=150)
    blend_alpha: float = Field(DEFAULT_BLEND_ALPHA, ge=0.0, le=1.0)
    theta: float = Field(DEFAULT_THETA, gt=0.0, le=500.0)
    n_sim: int = Field(DEFAULT_N_SIM, ge=500, le=30000)
    coverage_pct: float = Field(DEFAULT_COVERAGE_PCT, ge=50.0, le=100.0)
    seed: int = DEFAULT_SEED
    max_candles: int = Field(MAX_CANDLES_DEFAULT, ge=0)  # 0 = вся история


@router.post("/live")
async def live(
    body: LiveRequest,
    db: aiosqlite.Connection = Depends(get_db),
):
    """Синхронный live-прогноз — один поиск соседей + MC-симуляция + дешёвые
    квантильные считывания (~50мс на n_sim=15000/h=20, см. план §1). Ответ —
    уже готовые числа по каждому шагу h=1..H, фронт не пересчитывает квантили."""
    instr = await get_instrument_by_id(db, body.instrument_id)
    if instr is None:
        raise HTTPException(404, f"Instrument {body.instrument_id} not found")

    candles = await get_candles(db, body.instrument_id, body.interval, until=body.origin_ts)
    if len(candles) < 200:
        raise HTTPException(400, f"Недостаточно данных ({len(candles)} баров)")

    # Обрезаем СПИСОК свечей (не только производные массивы) до последних
    # max_candles баров — иначе origin_candle=candles[-1] ниже рассинхронизируется
    # с close[-1] (см. тот же приём в range_forecast.py:live).
    if body.max_candles > 0 and len(candles) > body.max_candles:
        candles = candles[-body.max_candles:]
    close = np.array([c["close"] for c in candles], dtype=np.float64)
    high = np.array([c["high"] for c in candles], dtype=np.float64)
    low = np.array([c["low"] for c in candles], dtype=np.float64)
    n_total = len(close)
    if n_total < 200:
        raise HTTPException(400, f"Недостаточно данных после обрезки ({n_total} баров)")
    if np.any(close <= 0) or np.any(high <= 0) or np.any(low <= 0):
        raise HTTPException(400, "Обнаружены неположительные цены")

    loop = asyncio.get_event_loop()

    def _compute():
        return compute_risk_corridor(
            close, high, low, body.h, body.p_fit, body.blend_alpha,
            body.theta, body.n_sim, body.coverage_pct, body.seed,
        )

    res = await loop.run_in_executor(None, _compute)
    if res is None:
        raise HTTPException(400, "Пул соседей слишком мал — уменьшите p_fit/theiler или horizon")

    origin_candle = candles[-1]
    return {
        "origin_date": origin_candle["begin"],
        "close_at_origin": origin_candle["close"],
        "n_neighbors": res["n_neighbors"],
        "coverage_pct": body.coverage_pct,
        "steps": {str(h): bounds for h, bounds in res["steps"].items()},
    }
