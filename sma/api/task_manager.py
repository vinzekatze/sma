"""
Background task queue — persistent (SQLite-backed), strictly sequential
(one task at a time), cooperative cancellation, manual resume after a
process crash.

Each task maps to one row in the `tasks` DB table. Kinds are dispatched by
`task["kind"]` inside `_run_task`; each kind's engine runs inline, wrapping
blocking work (numpy/requests) in loop.run_in_executor(None, ...) so the
event loop stays responsive.

Progress events flow:
  task thread  →  call_soon_threadsafe  →  asyncio event loop
  →  _broadcast  →  subscriber Queues  →  WebSocket handlers

Design notes (see docs/plans/task_manager_plan.md for the full rationale):
- Queue is a flat FIFO of task_ids — TaskManager has no concept of task
  composition/dependencies. Callers that need "download pool tickers, then
  calibrate" just submit the download tasks first, in order; the queue's
  own FIFO ordering is the only sequencing mechanism.
- Cancellation is cooperative: `is_cancel_requested()` is checked by an
  engine at its own natural checkpoints (if it has any). Single-shot
  engines (forecast, candle_fetch) have no internal checkpoint, so a
  cancel requested while they're already running only takes effect if it
  arrives before they start.
- Resuming reuses the same task_id (status → 'pending' → resubmitted).
  TaskManager doesn't serialize/restore any execution state itself — each
  engine is responsible for being idempotent (skip what's already done,
  determined by looking at its own persisted checkpoints in the DB).
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path

import numpy as np

from sma.core.db import (
    open_db,
    get_app_settings,
    get_task,
    update_task,
    list_tasks,
    get_instrument_by_id,
    get_candles,
    get_candle_by_ts,
    save_forecast,
    upsert_candles,
    refresh_stale_flags,
    get_forecast_settings,
    upsert_forecast_settings,
    upsert_forecast_defaults,
    prune_old_done_tasks,
)
from sma.core.candle_fetch import resolve_fetch_plan

RESUMABLE_STATUSES = ("pending", "cancelled", "interrupted", "error")
KEEP_DONE_TASKS = 10  # see db.py:prune_old_done_tasks — the queue is a work log, not a history store


class TaskManager:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._active: set[int] = set()      # task_ids currently queued or running — dedupes submit()
        self._cancelled: set[int] = set()   # task_ids with a pending cancel request
        # task_id → list of per-subscriber queues
        self._subs: dict[int, list[asyncio.Queue]] = {}
        self._worker_task: asyncio.Task | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self._reconcile_after_restart()
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="task-manager-worker"
        )

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker_task

    async def _reconcile_after_restart(self) -> None:
        """
        Tasks stuck 'running' from a previous process lifetime — the queue
        was only in-memory, so nobody is actually executing them. Mark them
        'interrupted' (distinct from 'cancelled' — this wasn't the user's
        choice). Deliberately NOT auto-resubmitted (nor are still-'pending'
        ones) — the user resumes explicitly via resume()/resume_all().
        """
        async with open_db(self._db_path) as db:
            stale = await list_tasks(db, status="running")
            for t in stale:
                await update_task(db, t["id"], status="interrupted")

    # ── public API ────────────────────────────────────────────────────────────

    async def submit(self, task_id: int) -> None:
        """Enqueue task_id for execution (no-op if already queued/running)."""
        if task_id in self._active:
            return
        self._active.add(task_id)
        await self._queue.put(task_id)

    async def request_cancel(self, task_id: int) -> None:
        """Cooperative — takes effect at the task's own next checkpoint (if
        it hasn't started yet, immediately; single-shot engines may finish
        before ever checking)."""
        self._cancelled.add(task_id)
        async with open_db(self._db_path) as db:
            await update_task(db, task_id, cancel_requested=1)

    def is_cancel_requested(self, task_id: int) -> bool:
        return task_id in self._cancelled

    async def resume(self, task_id: int) -> bool:
        """Re-queue a task that is cancelled/interrupted/error. Returns
        False if the task doesn't exist or isn't in a resumable state."""
        async with open_db(self._db_path) as db:
            task = await get_task(db, task_id)
            if task is None or task["status"] not in RESUMABLE_STATUSES:
                return False
            await update_task(db, task_id, status="pending", cancel_requested=0)
        self._cancelled.discard(task_id)
        await self.submit(task_id)
        return True

    async def resume_all(self) -> list[int]:
        """«Запустить всё» — resumes every pending/cancelled/interrupted/
        error task, oldest first. Still processed strictly one at a time
        (the worker loop is unchanged) — this just refills the queue."""
        async with open_db(self._db_path) as db:
            candidates: list[dict] = []
            for status in RESUMABLE_STATUSES:
                candidates.extend(await list_tasks(db, status=status))
            candidates.sort(key=lambda t: t["created_at"])
            for t in candidates:
                if t["status"] != "pending":
                    await update_task(db, t["id"], status="pending", cancel_requested=0)
                self._cancelled.discard(t["id"])

        for t in candidates:
            await self.submit(t["id"])
        return [t["id"] for t in candidates]

    def subscribe(self, task_id: int) -> asyncio.Queue:
        """Return a queue that receives progress/status dicts for task_id."""
        q: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(task_id, []).append(q)
        return q

    def unsubscribe(self, task_id: int, q: asyncio.Queue) -> None:
        subs = self._subs.get(task_id, [])
        with suppress(ValueError):
            subs.remove(q)

    # ── internals ─────────────────────────────────────────────────────────────

    def _broadcast(self, task_id: int, msg: dict) -> None:
        """Called from any thread; safe because it only uses put_nowait."""
        for q in list(self._subs.get(task_id, [])):
            q.put_nowait(msg)

    def _report_progress(self, task_id: int, loop: asyncio.AbstractEventLoop, done: int, total: int) -> None:
        """
        Callable from ANY thread (executor workers) — broadcasts over WS (for
        an actively-connected submit flow, unchanged behavior) AND persists
        progress_done/progress_total to the tasks row, so the left-panel task
        manager (sma/ui/tasks.js:miniProgress, polled via GET /tasks) shows
        real progress too — previously these columns were only ever written
        as the final 0 defaults, never updated mid-run, so the task list's
        progress bars were always empty until done. Fire-and-forget: schedules
        the DB write on the event loop via run_coroutine_threadsafe without
        blocking the calling thread.
        """
        self._broadcast(task_id, {"status": "running", "done": done, "total": total})
        asyncio.run_coroutine_threadsafe(self._persist_progress(task_id, done, total), loop)

    async def _persist_progress(self, task_id: int, done: int, total: int) -> None:
        async with open_db(self._db_path) as db:
            await update_task(db, task_id, progress_done=done, progress_total=total)

    async def _worker_loop(self) -> None:
        while True:
            task_id = await self._queue.get()
            try:
                await self._run_task(task_id)
            except Exception as exc:
                async with open_db(self._db_path) as db:
                    await update_task(db, task_id, status="error", error=str(exc))
                self._broadcast(task_id, {"status": "error", "error": str(exc)})
            finally:
                self._active.discard(task_id)
                self._cancelled.discard(task_id)

    async def _run_task(self, task_id: int) -> None:
        async with open_db(self._db_path) as db:
            task = await get_task(db, task_id)
            if task is None:
                return

            if self.is_cancel_requested(task_id):
                await update_task(db, task_id, status="cancelled")
                self._broadcast(task_id, {"status": "cancelled"})
                return

            await update_task(db, task_id, status="running")

        self._broadcast(task_id, {"status": "running", "done": 0, "total": 0})

        kind = task["kind"]
        if kind == "forecast":
            await self._run_forecast(task_id, task)
        elif kind == "candle_fetch":
            await self._run_candle_fetch(task_id, task)
        elif kind == "calibration":
            await self._run_calibration(task_id, task)
        elif kind == "pretest":
            await self._run_pretest(task_id, task)
        else:
            raise ValueError(f"Unknown task kind: {kind!r}")

        # Whatever this task produced already lives in its own table (see
        # prune_old_done_tasks) — keep the queue itself from growing
        # unbounded, e.g. from the dozens of candle_fetch tasks one pool
        # calibration queues. Runs after every kind, harmless if this
        # particular task ended up cancelled rather than done (is_cancel_
        # requested already returned early above, so reaching here means
        # the kind handler itself set a terminal status).
        async with open_db(self._db_path) as db:
            await prune_old_done_tasks(db, keep=KEEP_DONE_TASKS)

    # ── shared: load target + pool candle data, compute bar-native ranks ────

    async def _load_pool_ticker_data(self, instrument_id: int, resolved_ids: list[int], interval: str):
        """Target + pool instrument rows -> {ticker: (lh, ll, dates, rank_dict)}.
        DB I/O happens here (async); rank computation is CPU-bound, run in an
        executor by the caller — this method only assembles the raw arrays
        plus the deduplicated instrument row list."""
        from sma.core.forecast import band_lambda as bl

        async with open_db(self._db_path) as db:
            target_instr = await get_instrument_by_id(db, instrument_id)
            if target_instr is None:
                raise ValueError(f"Instrument {instrument_id} not found")
            seen = {target_instr["id"]}
            instrument_rows = [target_instr]
            for iid in resolved_ids:
                if iid in seen:
                    continue
                seen.add(iid)
                row = await get_instrument_by_id(db, iid)
                if row is not None:
                    instrument_rows.append(row)
            raw_arrays = await bl.load_raw_ticker_arrays(db, instrument_rows, interval)

        return target_instr, raw_arrays

    # ── kind: forecast — dispatch by model_type ──────────────────────────────

    async def _run_forecast(self, task_id: int, task: dict) -> None:
        model_type = task["params"].get("model_type", "band_lambda")
        if model_type == "simplex_ensemble":
            await self._run_forecast_simplex(task_id, task)
        else:
            await self._run_forecast_band_lambda(task_id, task)

    async def _run_forecast_band_lambda(self, task_id: int, task: dict) -> None:
        from sma.core.forecast import band_lambda as bl

        loop = asyncio.get_running_loop()
        p = task["params"]

        async with open_db(self._db_path) as db:
            settings = await get_forecast_settings(db, p["forecast_settings_id"])
        if settings is None:
            raise ValueError(f"forecast_settings {p['forecast_settings_id']} not found")

        resolved_ids = settings["pool_config"]["resolved_instrument_ids"]
        target_instr, raw_arrays = await self._load_pool_ticker_data(
            task["instrument_id"], resolved_ids, task["interval"]
        )
        if target_instr["ticker"] not in raw_arrays:
            raise ValueError(
                f"Недостаточно данных по {target_instr['ticker']} [{task['interval']}]"
            )

        # origin_ts == "" -> live ("as of now"); otherwise a specific bar's
        # begin timestamp -> truncate EVERY ticker (target + pool) to it, so
        # a historical replay can't see anything past its own origin (see
        # band_lambda.mask_ticker_data docstring for why this is safe/exact).
        origin_ts = task["origin_ts"] or None

        def _compute():
            ticker_data = bl.build_ticker_data(raw_arrays)
            if origin_ts:
                ticker_data = bl.mask_ticker_data(ticker_data, origin_ts)
                if target_instr["ticker"] not in ticker_data:
                    raise ValueError("Insufficient data before the requested origin")
            return bl.forecast_live_band(
                target_instr["ticker"], settings["t_query"], settings["lambda"], ticker_data,
                origin_index=None, m=settings["m"], min_bars=settings["min_bars"], theta=settings["theta"],
            )

        result = await loop.run_in_executor(None, _compute)
        if result is None:
            raise ValueError("Недостаточно пивотов/пула для прогноза на этот origin")

        params_out = {
            "t_query": settings["t_query"], "lambda": settings["lambda"],
            "m": settings["m"], "theta": settings["theta"], "min_bars": settings["min_bars"],
            "forecast_settings_id": settings["id"],
        }

        async with open_db(self._db_path) as db:
            origin_c = await get_candle_by_ts(
                db, task["instrument_id"], task["interval"], result["origin_date"]
            )
            # Snapshot the confirmation bar's close for refresh_stale_flags —
            # origin_price is the pivot's EXTREME price (a different bar
            # entirely), comparing that against origin_candle_id's live close
            # would flag every forecast stale immediately.
            result["close_at_origin"] = origin_c["close"]
            forecast_id = await save_forecast(
                db, task["instrument_id"], task["interval"], "band_lambda",
                origin_c["id"], params_out, result,
            )
            await update_task(db, task_id, status="done", forecast_id=forecast_id)

        self._broadcast(
            task_id, {"status": "done", "done": 1, "total": 1, "forecast_id": forecast_id}
        )

    # ── kind: forecast (simplex_ensemble) ────────────────────────────────────

    async def _run_forecast_simplex(self, task_id: int, task: dict) -> None:
        """
        No forecast_settings row for this model — params travel straight in
        the task (see routes/forecasts.py:request_forecast). No pool, single
        ticker: causality is enforced purely by `until=origin_ts` at load
        time (get_candles), same principle as band_lambda's mask_ticker_data
        but simpler (no pool to truncate). See simplex_ensemble.py module
        docstring for why origin_index isn't a separate parameter — origin is
        always the last loaded bar.
        """
        from sma.core.forecast import simplex_ensemble as se

        loop = asyncio.get_running_loop()
        p = task["params"]

        async with open_db(self._db_path) as db:
            candles = await get_candles(
                db, task["instrument_id"], task["interval"],
                until=(task["origin_ts"] or None),
            )
        if len(candles) < 20:
            raise ValueError(f"Недостаточно данных [{task['interval']}] для прогноза на этот origin")

        times = np.array([c["begin"] for c in candles])
        close = np.array([c["close"] for c in candles], dtype=np.float64)

        pca_p_range = tuple(p.get("pca_p_range", se.DEFAULT_PCA_P_RANGE))
        pca_thr1, pca_thr2 = p.get("pca_thr_range", [se.DEFAULT_PCA_THR1, se.DEFAULT_PCA_THR2])

        def _progress(done: int, total: int) -> None:
            self._report_progress(task_id, loop, done, total)

        def _compute():
            return se.forecast_ensemble(
                times, close,
                window=p.get("window", se.DEFAULT_WINDOW),
                horizon=p.get("horizon", se.DEFAULT_HORIZON),
                xy_x=p.get("xy_x", se.DEFAULT_XY_X),
                xy_y=p.get("xy_y", se.DEFAULT_XY_Y),
                xi_add=p.get("xi_add", se.DEFAULT_XI_ADD),
                blend_alpha=p.get("blend_alpha", se.DEFAULT_BLEND_ALPHA),
                n_levels=p.get("n_levels", se.DEFAULT_N_LEVELS),
                p_cascade_max=p.get("p_cascade_max", se.DEFAULT_P_CASCADE_MAX),
                bars=p.get("bars", se.DEFAULT_BARS),
                pca_p_range=pca_p_range,
                pca_thr1=pca_thr1, pca_thr2=pca_thr2,
                use_lp_corr=p.get("use_lp_corr", se.DEFAULT_USE_LP_CORR),
                progress_cb=_progress,
            )

        result = await loop.run_in_executor(None, _compute)
        if "error" in result:
            raise ValueError(result["error"])

        params_out = {k: v for k, v in p.items() if k != "model_type"}

        async with open_db(self._db_path) as db:
            origin_c = await get_candle_by_ts(
                db, task["instrument_id"], task["interval"], result["origin_date"]
            )
            forecast_id = await save_forecast(
                db, task["instrument_id"], task["interval"], "simplex_ensemble",
                origin_c["id"], params_out, result,
            )
            # "Last used settings" prefill (see db.py forecast_defaults docstring)
            # — only ever written after a forecast that actually completed.
            await upsert_forecast_defaults(
                db, task["instrument_id"], task["interval"], "simplex_ensemble", params_out,
            )
            await update_task(db, task_id, status="done", forecast_id=forecast_id)

        self._broadcast(
            task_id, {"status": "done", "done": 1, "total": 1, "forecast_id": forecast_id}
        )

    # ── kind: calibration (band_lambda) ──────────────────────────────────────

    async def _run_calibration(self, task_id: int, task: dict) -> None:
        import os
        from concurrent.futures import ProcessPoolExecutor
        from datetime import datetime, timezone
        from sma.core.forecast import band_lambda as bl
        from sma.core.forecast import band_lambda_calibrator as calib

        loop = asyncio.get_running_loop()
        p = task["params"]
        pool_config = p["pool"]
        resolved_ids = pool_config["resolved_instrument_ids"]

        target_instr, raw_arrays = await self._load_pool_ticker_data(
            task["instrument_id"], resolved_ids, task["interval"]
        )
        if target_instr["ticker"] not in raw_arrays:
            raise ValueError(
                f"Недостаточно данных по тикеру {target_instr['ticker']} [{task['interval']}] "
                "— дозагрузите через менеджер тикеров или пересоздайте задачу калибровки"
            )

        ticker_data = await loop.run_in_executor(None, bl.build_ticker_data, raw_arrays)

        targets = p["targets"]           # [{"t_query": .., "min_bars": ..}, ...]
        m = p.get("m", bl.DEFAULT_M)
        theta = p.get("theta", bl.DEFAULT_THETA)
        total = len(targets)

        async with open_db(self._db_path) as db:
            app_settings = await get_app_settings(db)
        configured_workers = app_settings["calibration_workers"]
        # 0 = auto — see docs/plans/band_forecast_migration_plan.md 5.2 and
        # feedback memory blas_oversubscription_multiprocessing (numpy ops
        # are memory-bandwidth bound, so cpu_count itself over-subscribes).
        max_workers = configured_workers if configured_workers > 0 else max(1, (os.cpu_count() or 4) // 4)
        executor = ProcessPoolExecutor(max_workers=max_workers, initializer=calib.init_worker_env)
        try:
            submitted = [
                (spec, executor.submit(
                    calib.calibrate_one_target, target_instr["ticker"], spec["t_query"],
                    spec.get("min_bars", bl.DEFAULT_MIN_BARS), m, ticker_data,
                ))
                for spec in targets
            ]
            done = 0
            for spec, fut in submitted:
                if self.is_cancel_requested(task_id):
                    for _, f in submitted:
                        f.cancel()
                    async with open_db(self._db_path) as db:
                        await update_task(db, task_id, status="cancelled")
                    self._broadcast(task_id, {"status": "cancelled"})
                    return

                result = await asyncio.wrap_future(fut)
                min_bars = spec.get("min_bars", bl.DEFAULT_MIN_BARS)
                if not result.get("skipped"):
                    width_rel = (
                        result["final_width"] / result["final_baseline_width"]
                        if result["final_width"] and result["final_baseline_width"] else None
                    )
                    async with open_db(self._db_path) as db:
                        await upsert_forecast_settings(
                            db, task["instrument_id"], task["interval"], "band_lambda",
                            spec["t_query"], min_bars, result["lambdas"], m, theta,
                            pool_config, pool_config["pool_key"],
                            {
                                "n_passes": result["n_passes"],
                                "converged": result["n_passes"] < calib.DEFAULT_MAX_PASSES,
                                "pinball_rel": result["final_rel"],
                                "width_rel": width_rel,
                                "n_holdout": result["n_holdout"],
                                "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            },
                        )
                done += 1
                self._broadcast(task_id, {"status": "running", "done": done, "total": total})
                async with open_db(self._db_path) as db:
                    await update_task(db, task_id, progress_done=done, progress_total=total)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        async with open_db(self._db_path) as db:
            await update_task(db, task_id, status="done")
        self._broadcast(task_id, {"status": "done", "done": total, "total": total})

    # ── kind: pretest (band_lambda) ──────────────────────────────────────────

    async def _run_pretest(self, task_id: int, task: dict) -> None:
        from datetime import date
        from sma.core.forecast import band_lambda as bl

        loop = asyncio.get_running_loop()
        p = task["params"]
        level = p["level"]

        if level == 1:
            from sma.data.moex import get_security_history_range

            candidates = p["candidates"]  # [{"secid","engine","market","board"}, ...] — no DB rows needed

            def _run_level1():
                rows = []
                for c in candidates:
                    try:
                        boards = get_security_history_range(c["secid"])
                    except Exception:
                        continue
                    board_row = next(
                        (b for b in boards if str(b.get("boardid", "")).upper() == str(c.get("board") or "").upper()),
                        None,
                    ) or next((b for b in boards if b.get("is_primary") in (1, True, "1")), None) \
                      or (boards[0] if boards else None)
                    if not board_row:
                        continue
                    hf, ht = board_row.get("history_from"), board_row.get("history_till")
                    if not hf or not ht:
                        continue
                    try:
                        years = (date.fromisoformat(str(ht)[:10]) - date.fromisoformat(str(hf)[:10])).days / 365.25
                    except ValueError:
                        continue
                    rows.append({"secid": c["secid"], "years": round(years, 2)})
                return rows

            rows = await loop.run_in_executor(None, _run_level1)
            summary = {
                "level": 1, "n_candidates": len(candidates), "n_resolved": len(rows),
                "total_years": round(sum(r["years"] for r in rows), 1),
                "avg_years": round(sum(r["years"] for r in rows) / len(rows), 2) if rows else 0.0,
                "per_ticker": rows,
            }
        else:
            pool_config = p["pool"]
            resolved_ids = pool_config["resolved_instrument_ids"]
            t_query = p["t_query"]
            min_bars = p.get("min_bars", bl.DEFAULT_MIN_BARS)

            _target_instr, raw_arrays = await self._load_pool_ticker_data(
                task["instrument_id"], resolved_ids, task["interval"]
            )

            def _run_level2():
                rows = []
                for ticker, (lh, ll, dates, _vol) in raw_arrays.items():
                    pivots, _ext, _conf, _dirs = bl.build_zigzag(lh, ll, dates, t_query, min_bars)
                    rows.append({"ticker": ticker, "n_events": len(pivots), "n_bars": len(lh)})
                return rows

            rows = await loop.run_in_executor(None, _run_level2)
            summary = {
                "level": 2, "t_query": t_query, "min_bars": min_bars,
                "n_tickers": len(rows),
                "total_events": sum(r["n_events"] for r in rows),
                "avg_events": round(sum(r["n_events"] for r in rows) / len(rows), 1) if rows else 0.0,
                "per_ticker": rows,
            }

        async with open_db(self._db_path) as db:
            await update_task(db, task_id, status="done")
        self._broadcast(task_id, {"status": "done", "done": 1, "total": 1, "summary": summary})

    # ── kind: candle_fetch ───────────────────────────────────────────────────

    async def _run_candle_fetch(self, task_id: int, task: dict) -> None:
        from sma.data.moex import download_candles, INTERVALS

        loop = asyncio.get_running_loop()
        p = task["params"]
        ticker = p["ticker"]
        data_source = p.get("data_source", "moex")
        interval = task["interval"]

        if data_source != "moex":
            raise ValueError(f"Unsupported data_source {data_source!r}")
        if interval not in INTERVALS:
            raise ValueError(f"Unknown interval {interval!r} for moex")

        async with open_db(self._db_path) as db:
            plan = await resolve_fetch_plan(
                db, task["instrument_id"], interval,
                p.get("date_from"), p.get("full_refresh", False),
            )

        kwargs = {k: v for k, v in plan.items() if k != "mode"}
        candles = await loop.run_in_executor(
            None,
            lambda: download_candles(ticker, interval, show_progress=False, **kwargs),
        )
        if not candles:
            raise ValueError(f"No data returned for {ticker} [{interval}]")

        async with open_db(self._db_path) as db:
            n = await upsert_candles(db, task["instrument_id"], interval, candles)
            await refresh_stale_flags(db, task["instrument_id"], interval)
            await update_task(db, task_id, status="done")

        self._broadcast(
            task_id,
            {"status": "done", "done": 1, "total": 1,
             "candles_saved": n, "mode": plan["mode"]},
        )
