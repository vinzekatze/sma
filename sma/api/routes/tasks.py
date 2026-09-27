from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
import aiosqlite

from sma.core.db import get_task, list_tasks, delete_task
from sma.api.deps import get_db, get_task_manager
from sma.api.task_manager import TaskManager

router = APIRouter()


@router.get("")
async def get_tasks(
    status: Optional[str] = Query(None),
    kind: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: aiosqlite.Connection = Depends(get_db),
):
    """List tasks (newest-updated first). Powers the task manager UI."""
    return await list_tasks(db, status=status, kind=kind, limit=limit)


@router.get("/{task_id}")
async def get_task_status(
    task_id: int,
    db: aiosqlite.Connection = Depends(get_db),
):
    row = await get_task(db, task_id)
    if row is None:
        raise HTTPException(404, "Task not found")
    return row


@router.post("/resume-all")
async def resume_all_tasks(tm: TaskManager = Depends(get_task_manager)):
    """«Запустить всё» — resumes every pending/cancelled/interrupted/error task."""
    task_ids = await tm.resume_all()
    return {"resumed": task_ids}


@router.post("/{task_id}/cancel", status_code=204)
async def cancel_task(
    task_id: int,
    db: aiosqlite.Connection = Depends(get_db),
    tm: TaskManager = Depends(get_task_manager),
):
    row = await get_task(db, task_id)
    if row is None:
        raise HTTPException(404, "Task not found")
    if row["status"] not in ("pending", "running"):
        raise HTTPException(422, f"Cannot cancel a task in status {row['status']!r}")
    await tm.request_cancel(task_id)


@router.post("/{task_id}/resume")
async def resume_task(
    task_id: int,
    tm: TaskManager = Depends(get_task_manager),
):
    ok = await tm.resume(task_id)
    if not ok:
        raise HTTPException(422, "Task not found or not in a resumable state")
    return {"task_id": task_id}


@router.delete("/{task_id}", status_code=204)
async def delete_task_route(
    task_id: int,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Removes one task row — terminal states only (pending/running are
    actively managed by TaskManager; cancel first). 'done' rows already
    age out on their own (see prune_old_done_tasks) but an explicit delete
    is harmless there too; cancelled/interrupted/error rows never
    auto-prune, so this is the only way to clear them.
    """
    row = await get_task(db, task_id)
    if row is None:
        raise HTTPException(404, "Task not found")
    if row["status"] in ("pending", "running"):
        raise HTTPException(422, f"Cannot delete a task in status {row['status']!r} — cancel it first")
    await delete_task(db, task_id)


@router.websocket("/{task_id}/ws")
async def task_progress_ws(
    task_id: int,
    ws: WebSocket,
    tm: TaskManager = Depends(get_task_manager),
):
    """
    Stream progress events for a running task.

    Each message is a JSON object:
      {"status": "running",   "done": int, "total": int}
      {"status": "done",      "done": 1,   "total": 1, "forecast_id": int}
      {"status": "error",     "error": str}
      {"status": "cancelled"}

    Connection closes automatically after a terminal event
    (done/error/cancelled).
    """
    await ws.accept()
    q = tm.subscribe(task_id)
    try:
        while True:
            msg = await q.get()
            await ws.send_json(msg)
            if msg.get("status") in ("done", "error", "cancelled"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        tm.unsubscribe(task_id, q)
