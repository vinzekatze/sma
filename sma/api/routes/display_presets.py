from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import aiosqlite

from sma.core.db import list_display_presets, save_display_preset, delete_display_preset
from sma.api.deps import get_db

router = APIRouter()


class DisplayPresetIn(BaseModel):
    model_type: str
    name: str
    levels: list[float]
    opacity: float = 0.5
    is_default: bool = False
    trade_level_pct: float = 70
    show_zones: bool = True
    show_trade_level: bool = True
    trim_zone1: bool = False


@router.get("")
async def get_display_presets(
    model_type: str = Query(...),
    db: aiosqlite.Connection = Depends(get_db),
):
    return await list_display_presets(db, model_type)


@router.post("")
async def create_display_preset(
    body: DisplayPresetIn,
    db: aiosqlite.Connection = Depends(get_db),
):
    """Upsert by (model_type, name) — saving the same name again overwrites
    it rather than creating a duplicate."""
    preset_id = await save_display_preset(
        db, body.model_type, body.name, body.levels, body.opacity, body.is_default,
        body.trade_level_pct, body.show_zones, body.show_trade_level, body.trim_zone1,
    )
    return {"id": preset_id}


@router.delete("/{preset_id}", status_code=204)
async def remove_display_preset(
    preset_id: int,
    db: aiosqlite.Connection = Depends(get_db),
):
    ok = await delete_display_preset(db, preset_id)
    if not ok:
        raise HTTPException(404, "Preset not found")
