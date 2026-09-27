from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import aiosqlite

from sma.core.db import (
    get_instrument,
    upsert_instrument,
    get_candles,
    create_task,
)
from sma.core.candle_fetch import resolve_fetch_plan
from sma.api.deps import get_db, get_task_manager
from sma.api.task_manager import TaskManager

router = APIRouter()


class FetchRequest(BaseModel):
    ticker: str
    data_source: str = "moex"
    asset_type: Optional[str] = None
    interval: str = "1d"
    date_from: Optional[str] = None
    engine: Optional[str] = None
    market: Optional[str] = None
    full_refresh: bool = False


class FetchTaskResponse(BaseModel):
    task_id: int


@router.post("/fetch", response_model=FetchTaskResponse, status_code=202)
async def fetch_candles(
    body: FetchRequest,
    db: aiosqlite.Connection = Depends(get_db),
    tm: TaskManager = Depends(get_task_manager),
):
    """
    Queue a candle download/refresh as a background task (kind='candle_fetch') —
    runs through the same sequential TaskManager queue as everything else, not
    synchronously inside this request. Poll GET /tasks/{task_id} or connect to
    WS /tasks/{task_id}/ws for progress/result (candles_saved, mode).

    Unless date_from/full_refresh is given explicitly, only the tail missing
    since the last stored candle is downloaded (upsert_candles overwrites the
    last stored day too, in case it was a partial/forming bar) — the actual
    full/incremental decision is re-resolved fresh when the task runs, this
    request only uses it to compose a human-readable label.
    """
    if body.data_source != "moex":
        raise HTTPException(422, f"Unsupported data_source {body.data_source!r}")

    instrument_id = await upsert_instrument(
        db, body.ticker, body.data_source, body.asset_type,
        engine=body.engine, market=body.market,
    )

    plan = await resolve_fetch_plan(
        db, instrument_id, body.interval, body.date_from, body.full_refresh,
    )
    label = (
        f"Загрузка {body.ticker} [{body.interval}]" if plan["mode"] == "full"
        else f"Дозагрузка {body.ticker} [{body.interval}]"
    )

    task_id = await create_task(
        db, instrument_id, body.interval, "",
        {
            "ticker": body.ticker,
            "data_source": body.data_source,
            "date_from": body.date_from,
            "full_refresh": body.full_refresh,
        },
        kind="candle_fetch",
        label=label,
    )
    await tm.submit(task_id)
    return FetchTaskResponse(task_id=task_id)


@router.get("")
async def list_candles(
    ticker: str = Query(...),
    data_source: str = Query("moex"),
    interval: str = Query(...),
    since: Optional[str] = Query(None),
    until: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, description="last N bars matching since/until — sma/ui/candle_window.js"),
    db: aiosqlite.Connection = Depends(get_db),
):
    instr = await get_instrument(db, ticker, data_source)
    if instr is None:
        raise HTTPException(404, f"Instrument {ticker}/{data_source} not found")
    rows = await get_candles(db, instr["id"], interval, since=since, until=until, limit=limit)
    return rows
