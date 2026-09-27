from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import aiosqlite

from sma.core.db import (
    get_instrument_by_id,
    upsert_instrument,
    get_forecast_defaults,
    get_band_lambda_pool,
    get_pool_resolution_cache,
    create_task,
)
from sma.core.forecast.pool_selection import compute_pool_key
from sma.api.deps import get_db, get_task_manager
from sma.api.task_manager import TaskManager

router = APIRouter()


class PoolSpec(BaseModel):
    categories: list[str] = []
    n: int = 25
    # When set, BYPASSES category-based MOEX resolution entirely and uses
    # exactly this list (still upserted/refreshed like any pool ticker) —
    # the pool modal's manually-edited pool list. categories/n are still
    # stored (as the "recommended" starting point the edit began from) but
    # resolved_instrument_ids becomes this list verbatim.
    manual_instrument_ids: list[int] | None = None


class PoolSaveRequest(BaseModel):
    instrument_id: int
    interval: str
    model_type: str = "band_lambda"
    pool: PoolSpec


class PoolPreviewRequest(BaseModel):
    instrument_id: int
    interval: str
    pool: PoolSpec


class TaskResponse(BaseModel):
    task_id: int


@router.get("/defaults")
async def get_defaults(
    instrument_id: int = Query(...),
    interval: str = Query(...),
    model_type: str = Query(...),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Last-used UI params for this (instrument, interval, model_type) — see
    db.py forecast_defaults table docstring. Written server-side after every
    successful forecast of that model_type (simplex_ensemble and, since
    T/m/theta became free live parameters instead of a saved settings row,
    band_lambda too). `{"params": null}` (not 404) when nothing is saved
    yet — the frontend falls back to hardcoded defaults in that case.
    """
    row = await get_forecast_defaults(db, instrument_id, interval, model_type)
    return {"params": row["params"] if row else None}


@router.get("/pool")
async def get_pool(
    instrument_id: int = Query(...),
    interval: str = Query(...),
    model_type: str = Query("band_lambda"),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    The one saved pool composition for this (instrument, interval) — powers
    the pool modal's prefill and the "Пул" summary in the sidebar.
    `{"pool": null}` when nothing has been saved yet (POST /forecasts will
    reject a band_lambda request in that case with a clear message).
    """
    pool_config = await get_band_lambda_pool(db, instrument_id, interval)
    return {"pool": pool_config}


@router.get("/resolve-pool-cache")
async def get_resolve_pool_cache(
    categories: str = Query(..., description="comma-separated"),
    n: int = Query(...),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Reads pool_resolution_cache directly (no MOEX call, no task) — lets the
    pool modal recover a preview resolve's result after reopening (closing
    the modal mid-resolve, or simply missing the one-shot WS 'done' message,
    used to mean starting over from scratch, reported 2026-09-12: "все равно
    приходится заново ждать в окне"). band_pool_modal.js calls this when it
    finds a 'done' pool_resolve(save=false) task for the current
    (instrument, interval) on open. `{"resolved_tickers": null}` if nothing
    cached yet for this categories+n combo.
    """
    cached = await get_pool_resolution_cache(db, compute_pool_key(categories.split(","), n))
    if cached is None:
        return {"resolved_tickers": None}

    # Cache stores raw MOEX candidate dicts, not resolved instrument ids —
    # upsert (cheap, local-only, same as task_manager.py:
    # _resolve_and_upsert_pool's cache-hit branch) to get real ids the
    # frontend can pass back as manual_instrument_ids on save.
    resolved_tickers = []
    for c in cached["candidates"]:
        iid = await upsert_instrument(
            db, c["secid"], "moex", c["asset_type"], c.get("name") or c.get("shortname"),
            engine=c["engine"], market=c["market"], board=c.get("board"),
            added_via="pool",
        )
        resolved_tickers.append({"id": iid, "ticker": c["secid"]})
    return {"resolved_tickers": resolved_tickers}


@router.post("/pool", response_model=TaskResponse, status_code=202)
async def save_pool(
    body: PoolSaveRequest,
    db: aiosqlite.Connection = Depends(get_db),
    tm: TaskManager = Depends(get_task_manager),
):
    """
    Queues a task that resolves (cache-aware, see task_manager.py:
    POOL_CACHE_TTL_HOURS) and saves the ONE pool composition for this
    (instrument, interval) — no more T-keying: since λ-calibration was
    removed from prod (2026-09-12, see memory
    project_phase7_calibration_removed_final), the pool no longer varies by
    T, so one row replaces what used to be one forecast_settings row per T.

    Async (202 + task_id, poll/WS like any other task) rather than the
    synchronous response this used to return — resolve_pool_candidates hits
    MOEX for potentially 100+ candidates and was found to occasionally run
    long enough to make the request feel hung with zero visibility/
    cancel/timeout (reported 2026-09-12); as a task it's at least visible
    in the queue and consistent with every other MOEX-touching operation.
    The manual_instrument_ids path (no MOEX call, purely local) still goes
    through the same task for a uniform frontend contract — it just
    finishes near-instantly.
    """
    instr = await get_instrument_by_id(db, body.instrument_id)
    if instr is None:
        raise HTTPException(404, f"Instrument {body.instrument_id} not found")

    task_id = await create_task(
        db, body.instrument_id, body.interval, "",
        {"model_type": body.model_type, "pool": body.pool.model_dump(), "save": True},
        kind="pool_resolve", label=f"Пул {instr['ticker']} [{body.interval}]",
    )
    await tm.submit(task_id)
    return TaskResponse(task_id=task_id)


@router.post("/resolve-pool", response_model=TaskResponse, status_code=202)
async def resolve_pool(
    body: PoolPreviewRequest,
    db: aiosqlite.Connection = Depends(get_db),
    tm: TaskManager = Depends(get_task_manager),
):
    """
    Preview-only: queues a task that resolves categories+n to a ticker list
    WITHOUT saving anything — powers the pool modal's "what the algorithm
    recommends" list, which the user then edits before actually saving via
    POST /pool. Still upserts newly-seen instruments (same idempotent side
    effect POST /pool already has) — that part can't be previewed away,
    it's just how pool candidates get a DB row.

    Always bypasses pool_resolution_cache (this button IS the user's
    deliberate "recompute now" action, unlike POST /pool's incidental,
    cache-eligible resolution) and refreshes the cache with what it finds.
    Async for the same reason POST /pool is — see its docstring.
    """
    if body.pool.manual_instrument_ids is not None:
        raise HTTPException(422, "resolve-pool ожидает categories+n, а не manual_instrument_ids")
    instr = await get_instrument_by_id(db, body.instrument_id)
    if instr is None:
        raise HTTPException(404, f"Instrument {body.instrument_id} not found")

    task_id = await create_task(
        db, body.instrument_id, body.interval, "",
        {"pool": body.pool.model_dump(), "save": False},
        kind="pool_resolve", label=f"Резолв пула {instr['ticker']} [{body.interval}]",
    )
    await tm.submit(task_id)
    return TaskResponse(task_id=task_id)
