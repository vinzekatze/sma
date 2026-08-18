"""FastAPI dependency injection helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite

from sma.core.db import open_db
from sma.api.task_manager import TaskManager

DB_PATH: Path = Path(os.getenv("SMA_DB_PATH", "sma.db"))

# set by app.py lifespan before any request arrives
_task_manager: TaskManager | None = None


async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with open_db(DB_PATH) as db:
        yield db


def get_task_manager() -> TaskManager:
    assert _task_manager is not None, "TaskManager not initialised"
    return _task_manager
