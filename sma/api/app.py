"""
FastAPI application entry point.

Run:
    uvicorn sma.api.app:app --reload --host 0.0.0.0 --port 8000

Environment:
    SMA_DB_PATH  — path to SQLite database file (default: sma.db)
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

import sma.api.deps as _deps
from sma.core.db import init_db
from sma.api.task_manager import TaskManager
from sma.api.routes import instruments, candles, forecasts, forecast_settings, display_presets, tasks, series, settings

_UI_DIR = Path(__file__).parent.parent / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(_deps.DB_PATH)
    tm = TaskManager(_deps.DB_PATH)
    await tm.start()
    _deps._task_manager = tm
    yield
    await tm.stop()


app = FastAPI(
    title="SMA Forecast API",
    description="Time-series forecasting for exchange instruments (MOEX and others).",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(instruments.router,       prefix="/instruments",       tags=["instruments"])
app.include_router(candles.router,           prefix="/candles",           tags=["candles"])
app.include_router(forecasts.router,         prefix="/forecasts",         tags=["forecasts"])
app.include_router(forecast_settings.router, prefix="/forecast-settings", tags=["forecast-settings"])
app.include_router(display_presets.router,   prefix="/display-presets",   tags=["display-presets"])
app.include_router(tasks.router,             prefix="/tasks",             tags=["tasks"])
app.include_router(series.router,            prefix="/series",            tags=["series"])
app.include_router(settings.router,          prefix="/settings",          tags=["settings"])


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/ui/")


app.mount("/ui", StaticFiles(directory=_UI_DIR, html=True), name="ui")
