"""
Shared logic for deciding a candle download plan (full vs incremental).

Used both by the /candles/fetch route (to compose a human-readable task
label before queuing) and by the candle_fetch task engine itself (to
actually run the download) — one source of truth for the full/incremental
decision, resolved fresh each time rather than cached across the two uses.
"""

from __future__ import annotations

from sma.core.db import get_instrument_by_id, list_instrument_coverage, create_task


async def queue_pool_candle_fetch(db, tm, instrument_rows: list[dict], interval: str) -> list[int]:
    """
    Queue one incremental candle_fetch task per instrument row (already
    upserted by the caller — this does NOT touch `instruments`). Used by
    the band_lambda calibrate/forecast/pretest routes to make sure every
    pool ticker's data is current before the model reads it (see
    docs/plans/band_forecast_migration_plan.md 5.2: "Дозагрузка пула —
    отдельные задачи candle_fetch в очереди перед основной задачей").
    Mirrors the exact task shape POST /candles/fetch creates — same
    full/incremental resolution happens fresh inside the task itself.

    Returns the created task ids (already submitted to `tm`) — callers
    queue these BEFORE the task that actually needs the data, relying on
    the queue's strict FIFO ordering (no dependency mechanism needed).
    """
    task_ids: list[int] = []
    for row in instrument_rows:
        plan = await resolve_fetch_plan(db, row["id"], interval, None, False)
        label = (
            f"Загрузка {row['ticker']} [{interval}]" if plan["mode"] == "full"
            else f"Дозагрузка {row['ticker']} [{interval}]"
        )
        task_id = await create_task(
            db, row["id"], interval, "",
            {"ticker": row["ticker"], "data_source": row["data_source"], "date_from": None, "full_refresh": False},
            kind="candle_fetch", label=label,
        )
        await tm.submit(task_id)
        task_ids.append(task_id)
    return task_ids


async def resolve_fetch_plan(
    db,
    instrument_id: int,
    interval: str,
    date_from: str | None,
    full_refresh: bool,
) -> dict:
    """
    Returns {"engine": str, "market": str, "mode": "full"|"incremental",
    "date_from": str (optional)} — kwargs ready to pass to download_candles,
    plus "mode" for display/reporting.
    """
    instrument = await get_instrument_by_id(db, instrument_id)
    plan: dict = {
        "engine": instrument["engine"],
        "market": instrument["market"],
        "mode": "full",
    }
    if date_from:
        plan["date_from"] = date_from
    elif not full_refresh:
        coverage = await list_instrument_coverage(db, instrument_id)
        cov = next((c for c in coverage if c["interval"] == interval), None)
        if cov:
            plan["date_from"] = cov["last"][:10]
            plan["mode"] = "incremental"
    return plan
