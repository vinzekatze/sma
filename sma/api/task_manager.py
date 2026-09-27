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

from datetime import datetime, timezone

from sma.core.db import (
    open_db,
    get_app_settings,
    get_task,
    update_task,
    list_tasks,
    get_instrument_by_id,
    upsert_instrument,
    get_candles,
    get_candle_by_ts,
    save_forecast,
    upsert_candles,
    upsert_forecast_defaults,
    upsert_range_forecast_settings,
    get_band_lambda_pool,
    upsert_band_lambda_pool,
    get_pool_resolution_cache,
    upsert_pool_resolution_cache,
    prune_old_done_tasks,
)
from sma.core.candle_fetch import resolve_fetch_plan, queue_pool_candle_fetch
from sma.core.forecast.pool_selection import resolve_pool_candidates, compute_pool_key

RESUMABLE_STATUSES = ("pending", "cancelled", "interrupted", "error")
KEEP_DONE_TASKS = 10  # see db.py:prune_old_done_tasks — the queue is a work log, not a history store

POOL_CACHE_TTL_HOURS = 24  # liquidity ranking uses a 30-CALENDAR-day trailing
# window (pool_selection.LIQUIDITY_WINDOW_DAYS) — daily granularity is already
# more than enough, so incidental re-resolutions (every pool save) reuse the
# cached candidate list instead of re-hitting MOEX (category search +
# per-candidate liquidity download) every time. The pool modal's "Обновить
# по категориям" button (POST /resolve-pool) is the user's deliberate
# refresh action and always bypasses this (force=True).

POOL_RESOLVE_TIMEOUT_SEC = 180  # defensive backstop — resolve_pool_candidates
# already bounds each MOEX request at 30s (sma/data/moex/candles.py) and runs
# candidates concurrently, so this should never legitimately fire; it exists
# so a genuinely stuck resolve (e.g. DNS hang bypassing requests' own
# timeout) fails the task cleanly instead of blocking every task queued
# behind it forever — TaskManager is strictly single-worker/sequential (see
# module docstring), so a hung task stalls candle_fetch/forecast too, not
# just itself (reported 2026-09-12).


def _pool_cache_is_fresh(computed_at: str) -> bool:
    ts = datetime.fromisoformat(computed_at)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    return age_hours < POOL_CACHE_TTL_HOURS


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

        Separately: a 'pending' task can carry cancel_requested=1 with
        nobody left to act on it — request_cancel() only ever updates the
        DB flag + an in-memory set; the actual pending->cancelled
        transition normally happens inside _run_task right before a task
        starts running. If the process restarts (e.g. dev server's
        --reload firing on a source edit) while that task is still sitting
        in the now-discarded in-memory queue, the cancel request is
        orphaned forever — the task shows 'pending' (looks like it's still
        queued) even though the user unambiguously asked to stop it
        (reported 2026-09-12). Since cancel_requested=1 already IS that
        unambiguous signal regardless of which process lifetime set it,
        resolve it here instead of waiting for a queue that will never
        process it.
        """
        async with open_db(self._db_path) as db:
            stale = await list_tasks(db, status="running", limit=1000)
            for t in stale:
                await update_task(db, t["id"], status="interrupted")

            orphaned_cancels = await list_tasks(db, status="pending", limit=1000)
            for t in orphaned_cancels:
                if t["cancel_requested"]:
                    await update_task(db, t["id"], status="cancelled")

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
        elif kind == "pool_resolve":
            await self._run_pool_resolve(task_id, task)
        elif kind == "range_forecast_calibration":
            await self._run_range_forecast_calibration(task_id, task)
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
        elif model_type == "regime_mixture_potential":
            await self._run_forecast_regime_mixture_potential(task_id, task)
        else:
            await self._run_forecast_band_lambda(task_id, task)

    async def _run_forecast_band_lambda(self, task_id: int, task: dict) -> None:
        """
        T/m/theta travel straight in the task params (like simplex_ensemble's
        params dict) — no more forecast_settings row lookup, see
        sma/api/routes/forecasts.py:request_forecast. λ is always the
        uniform-pool zero vector (calibration removed from prod, see memory
        project_phase7_calibration_removed_final); pool composition is
        whatever was last saved via POST /forecast-settings/pool, resolved by
        that route and passed through here unchanged.
        """
        from sma.core.forecast import band_lambda as bl

        loop = asyncio.get_running_loop()
        p = task["params"]
        pool_config = p["pool"]
        t_query, m, theta, min_bars = p["t_query"], p["m"], p["theta"], p["min_bars"]
        zero_lambda = {f: 0.0 for f in bl.FEATURE_ORDER}

        resolved_ids = pool_config["resolved_instrument_ids"]
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
                target_instr["ticker"], t_query, zero_lambda, ticker_data,
                origin_index=None, m=m, min_bars=min_bars, theta=theta,
            )

        result = await loop.run_in_executor(None, _compute)
        if result is None:
            raise ValueError("Недостаточно пивотов/пула для прогноза на этот origin")

        params_out = {"t_query": t_query, "m": m, "theta": theta, "min_bars": min_bars}

        async with open_db(self._db_path) as db:
            origin_c = await get_candle_by_ts(
                db, task["instrument_id"], task["interval"], result["origin_date"]
            )
            forecast_id = await save_forecast(
                db, task["instrument_id"], task["interval"], "band_lambda",
                origin_c["id"], params_out, result,
            )
            # "Last used settings" prefill (see db.py forecast_defaults
            # docstring) — same mechanism simplex_ensemble already uses;
            # band_lambda has no other place T/m/theta live now.
            await upsert_forecast_defaults(
                db, task["instrument_id"], task["interval"], "band_lambda", params_out,
            )
            await update_task(db, task_id, status="done", forecast_id=forecast_id)

        self._broadcast(
            task_id, {"status": "done", "done": 1, "total": 1, "forecast_id": forecast_id}
        )

    # ── kind: forecast (simplex_ensemble) ────────────────────────────────────

    async def _run_forecast_simplex(self, task_id: int, task: dict) -> None:
        """
        Params travel straight in the task (see routes/forecasts.py:
        request_forecast), same as band_lambda now does. No pool, single
        ticker: causality is enforced purely by `until=origin_ts` at load
        time (get_candles), same principle as band_lambda's mask_ticker_data
        but simpler (no pool to truncate). See simplex_ensemble.py module
        docstring for why origin_index isn't a separate parameter — origin is
        always the last loaded bar.
        """
        import os
        from sma.core.forecast import simplex_ensemble as se

        loop = asyncio.get_running_loop()
        p = task["params"]

        async with open_db(self._db_path) as db:
            candles = await get_candles(
                db, task["instrument_id"], task["interval"],
                until=(task["origin_ts"] or None),
            )
            # Same knob as _run_range_forecast_calibration below —
            # "Потоков вычислений (CPU)" in Настройки приложения covers every
            # CPU-bound ProcessPoolExecutor task, not just one of them (see
            # its own tooltip in index.html).
            app_settings = await get_app_settings(db)
        configured_workers = app_settings["calibration_workers"]
        max_workers = configured_workers if configured_workers > 0 else max(1, (os.cpu_count() or 4) // 4)

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
                max_workers=max_workers,
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

    # ── kind: forecast (regime_mixture_potential) ────────────────────────────

    async def _run_forecast_regime_mixture_potential(self, task_id: int, task: dict) -> None:
        """
        Params travel straight in the task (see routes/forecasts.py:
        request_forecast), same as simplex_ensemble/band_lambda. No pool,
        single ticker: causality enforced purely by `until=origin_ts` at
        load time, same principle as _run_forecast_simplex.

        Премотка (n_forecasts независимых origin) фанится ВНУТРИ
        regime_mixture_potential.forecast_regime_mixture_potential через
        joblib/loky (см. модуль — НЕ голый ProcessPoolExecutor, sklearn.
        GaussianMixture требует spawn-семантику loky, иначе реальный hang
        при fork()). max_workers — тот же общий тюнинг «Потоков вычислений
        (CPU)», что и у simplex/range_forecast_calibration.
        """
        import os
        from sma.core.forecast import regime_mixture_potential as rmp

        loop = asyncio.get_running_loop()
        p = task["params"]

        async with open_db(self._db_path) as db:
            candles = await get_candles(
                db, task["instrument_id"], task["interval"],
                until=(task["origin_ts"] or None),
            )
            app_settings = await get_app_settings(db)
        configured_workers = app_settings["calibration_workers"]
        max_workers = configured_workers if configured_workers > 0 else max(1, (os.cpu_count() or 4) // 4)

        if len(candles) < 20:
            raise ValueError(f"Недостаточно данных [{task['interval']}] для прогноза на этот origin")

        times = np.array([c["begin"] for c in candles])
        close = np.array([c["close"] for c in candles], dtype=np.float64)

        def _progress(done: int, total: int) -> None:
            self._report_progress(task_id, loop, done, total)

        def _compute():
            return rmp.forecast_regime_mixture_potential(
                times, close,
                horizon=p.get("horizon", rmp.DEFAULT_HORIZON),
                n_forecasts=p.get("n_forecasts", rmp.DEFAULT_N_FORECASTS),
                rewind_step=p.get("rewind_step", rmp.DEFAULT_REWIND_STEP),
                theta=p.get("theta", rmp.DEFAULT_THETA),
                warmup=p.get("warmup", rmp.DEFAULT_WARMUP),
                theiler_window=p.get("theiler_window", rmp.DEFAULT_THEILER_WINDOW),
                bars=p.get("bars", rmp.DEFAULT_BARS),
                n_lookback=p.get("n_lookback", rmp.DEFAULT_N_LOOKBACK),
                lookback_step=p.get("lookback_step", rmp.DEFAULT_LOOKBACK_STEP),
                n_sim=p.get("n_sim", rmp.DEFAULT_N_SIM),
                seed=p.get("seed", rmp.DEFAULT_SEED),
                mix_n_resample=p.get("mix_n_resample", rmp.DEFAULT_MIX_N_RESAMPLE),
                bin_height_pct=p.get("bin_height_pct", rmp.DEFAULT_BIN_HEIGHT_PCT),
                coverage_pct=p.get("coverage_pct", rmp.DEFAULT_COVERAGE_PCT),
                progress_cb=_progress,
                max_workers=max_workers,
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
                db, task["instrument_id"], task["interval"], "regime_mixture_potential",
                origin_c["id"], params_out, result,
            )
            await upsert_forecast_defaults(
                db, task["instrument_id"], task["interval"], "regime_mixture_potential", params_out,
            )
            await update_task(db, task_id, status="done", forecast_id=forecast_id)

        self._broadcast(
            task_id, {"status": "done", "done": 1, "total": 1, "forecast_id": forecast_id}
        )

    # ── kind: range_forecast_calibration ─────────────────────────────────────

    async def _run_range_forecast_calibration(self, task_id: int, task: dict) -> None:
        """
        θ+5λ+read-квантиль координатный спуск для ОДНОЙ (h_steps, p, theiler)
        комбинации одного тикера — нет пула, нет фан-аута по нескольким T,
        поэтому один блокирующий вызов через run_in_executor (тот же
        паттерн, что _run_forecast_simplex), а не ProcessPoolExecutor-
        фан-аут (band_lambda's λ-calibration used to fan out this way
        before it was removed from prod, see band_lambda_calibrator.py's
        module docstring).
        """
        from datetime import datetime, timezone
        from sma.core.forecast import range_forecast_calibrator as rfc

        loop = asyncio.get_running_loop()
        p = task["params"]
        h_steps, p_lags, theiler = p["h_steps"], p["p"], p["theiler"]
        levels = tuple(p.get("levels", (50, 75, 90)))
        max_candles = p.get("max_candles", rfc.MAX_CANDLES_DEFAULT)

        async with open_db(self._db_path) as db:
            candles = await get_candles(db, task["instrument_id"], task["interval"])
        if len(candles) < 200:
            raise ValueError(f"Недостаточно данных [{task['interval']}] для калибровки")

        close = np.array([c["close"] for c in candles], dtype=np.float64)
        high = np.array([c["high"] for c in candles], dtype=np.float64)
        low = np.array([c["low"] for c in candles], dtype=np.float64)
        volume = np.array([c["volume"] for c in candles], dtype=np.float64)

        def _progress(done: int, total: int) -> None:
            self._report_progress(task_id, loop, done, total)

        def _compute():
            return rfc.calibrate_combined_multipass(
                close, high, low, H=h_steps, p=p_lags, theiler=theiler,
                volumes=volume, levels=levels, max_candles=max_candles, progress_cb=_progress,
            )

        result = await loop.run_in_executor(None, _compute)
        if result.get("skipped"):
            raise ValueError(f"Калибровка не удалась: {result.get('reason', 'недостаточно истории')}")

        async with open_db(self._db_path) as db:
            await upsert_range_forecast_settings(
                db, task["instrument_id"], task["interval"], h_steps, p_lags, theiler,
                result["params"], list(result["params"].keys()),
                {str(lv): qr for lv, qr in result["q_read_by_level"].items()},
                {
                    "n_passes": result["n_passes"], "converged": result["converged"],
                    "final_ratio": result["final_ratio"], "final_baseline_ratio": result["final_baseline_ratio"],
                    "final_rel": result["final_rel"], "n_holdout": result["n_holdout"],
                    "levels": list(result["levels"]), "widest_level": result["widest_level"],
                    "max_candles": max_candles,
                    "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
            await update_task(db, task_id, status="done")
        self._broadcast(task_id, {"status": "done", "done": 1, "total": 1})

    # ── kind: pool_resolve (band_lambda pool modal) ──────────────────────────

    async def _resolve_and_upsert_pool(self, db, pool: dict, force: bool = False) -> tuple[list[dict], dict]:
        """
        resolve_pool_candidates hits MOEX ISS synchronously (blocking
        requests calls) — run off the event loop (with a defensive overall
        timeout, POOL_RESOLVE_TIMEOUT_SEC), and only when
        pool_resolution_cache is missing/stale/bypassed (force=True). Each
        candidate is upserted as added_via='pool' (see docs/plans/
        band_forecast_migration_plan.md 5.2a) — this never downgrades an
        existing 'manual' instrument, and existing 'pool'/'manual' rows are
        reused as-is (upsert, not insert-only) — that upsert pass always
        runs, cache hit or not, since it's cheap (local DB only).

        pool["manual_instrument_ids"], when set, skips MOEX category
        resolution entirely — the pool modal's manually-edited pool list.
        categories/n are kept in pool_config only as the "recommended
        starting point" label; resolved_instrument_ids is exactly the
        manual list in that case.

        Returns (instrument_rows, pool_config) where pool_config is ready to
        store verbatim in band_lambda_pool.pool_config_json.
        resolved_tickers ({id, ticker} pairs) rides along so the frontend
        never needs a second round trip just to LABEL the resolved ids.
        """
        categories = pool.get("categories", [])
        n = pool.get("n", 25)
        manual_instrument_ids = pool.get("manual_instrument_ids")

        if manual_instrument_ids is not None:
            rows = []
            for iid in manual_instrument_ids:
                row = await get_instrument_by_id(db, iid)
                if row is not None:
                    rows.append(row)
        else:
            pool_key = compute_pool_key(categories, n)
            cached = None if force else await get_pool_resolution_cache(db, pool_key)
            if cached is not None and _pool_cache_is_fresh(cached["computed_at"]):
                candidates = cached["candidates"]
            else:
                app_settings = await get_app_settings(db)
                loop = asyncio.get_running_loop()
                try:
                    candidates = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, resolve_pool_candidates, categories, n, app_settings["moex_pool_workers"]
                        ),
                        timeout=POOL_RESOLVE_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    raise ValueError(
                        f"Резолвинг пула превысил {POOL_RESOLVE_TIMEOUT_SEC}с (MOEX недоступен/слишком медленный) — попробуйте ещё раз"
                    )
                await upsert_pool_resolution_cache(db, pool_key, candidates)
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
            "resolved_tickers": [{"id": r["id"], "ticker": r["ticker"]} for r in rows],
            "pool_key": compute_pool_key(categories, n),
        }
        return rows, pool_config

    async def _run_pool_resolve(self, task_id: int, task: dict) -> None:
        """
        Backs both POST /forecast-settings/pool (save=True — also persists
        to band_lambda_pool and queues a freshness-aware candle_fetch per
        pool ticker) and POST /forecast-settings/resolve-pool (save=False —
        preview only). Moved off the request/response cycle 2026-09-12: it
        used to run synchronously inside the route handler, which could
        make the request hang for as long as MOEX took with zero
        visibility/cancel/timeout — see POOL_RESOLVE_TIMEOUT_SEC above.
        """
        p = task["params"]
        instrument_id, interval = task["instrument_id"], task["interval"]

        async with open_db(self._db_path) as db:
            pool_rows, pool_config = await self._resolve_and_upsert_pool(db, p["pool"], force=not p["save"])
            if not pool_rows:
                raise ValueError(
                    f"Пул пуст — ни один кандидат не прошёл порог ликвидности для категорий {p['pool'].get('categories')}"
                )
            if p["save"]:
                await queue_pool_candle_fetch(db, self, pool_rows, interval)
                await upsert_band_lambda_pool(db, instrument_id, interval, pool_config)
            await update_task(db, task_id, status="done")

        self._broadcast(task_id, {"status": "done", "done": 1, "total": 1, "pool": pool_config})

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
            await update_task(db, task_id, status="done")

        self._broadcast(
            task_id,
            {"status": "done", "done": 1, "total": 1,
             "candles_saved": n, "mode": plan["mode"]},
        )
