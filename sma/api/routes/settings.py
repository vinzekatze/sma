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


@router.get("")
async def get_settings(db: aiosqlite.Connection = Depends(get_db)):
    return await get_app_settings(db)


@router.post("")
async def set_settings(body: AppSettingsIn, db: aiosqlite.Connection = Depends(get_db)):
    return await save_app_settings(db, body.moex_pool_workers, body.calibration_workers)
