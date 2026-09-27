from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
import aiosqlite

from sma.core.db import get_app_settings, save_app_settings
from sma.api.deps import get_db

router = APIRouter()


class AppSettingsIn(BaseModel):
    moex_pool_workers: int = Field(16, ge=1, le=64)
    calibration_workers: int = Field(0, ge=0, le=64, description="0 = auto (cpu_count // 4)")
    chart_window_bars: int = Field(1000, ge=200, le=20000, description="bars kept loaded on the main chart at once — sma/ui/candle_window.js")
    # None = leave the stored color profile untouched — the moex/calibration-
    # workers form (settings.js) never sends this; the color-profile form
    # (settings.js, separate section) always sends the full dict it read+
    # edited. See sma/core/db.py:save_app_settings/DEFAULT_COLOR_PROFILE for
    # the fixed role keys (docs/plans/frontend_improvements_plan.md §1.1a).
    color_profile: dict | None = None


@router.get("")
async def get_settings(db: aiosqlite.Connection = Depends(get_db)):
    return await get_app_settings(db)


@router.post("")
async def set_settings(body: AppSettingsIn, db: aiosqlite.Connection = Depends(get_db)):
    return await save_app_settings(db, body.moex_pool_workers, body.calibration_workers, body.chart_window_bars, body.color_profile)
