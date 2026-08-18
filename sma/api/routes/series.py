import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from sma.core.db import get_candles, get_instrument, get_analysis_settings, upsert_analysis_settings
from sma.core.analysis.spectrogram import compute_spectrogram
from sma.core.analysis.trend_variance import compute_trend_variance
from sma.api.deps import get_db

router = APIRouter()


class SpectrogramRequest(BaseModel):
    ticker: str
    data_source: str = "moex"
    interval: str = "1d"
    depth_bars: int = Field(500, ge=0, description="0 = full history")
    nperseg: int = Field(64, ge=4)
    overlap_pct: float = Field(75.0, ge=0, lt=100)
    fmin: float = Field(0.0, ge=0)
    fmax: float = Field(0.5, gt=0, le=0.5)


class TrendVarianceRequest(BaseModel):
    ticker: str
    data_source: str = "moex"
    interval: str = "1d"
    window: int = Field(200, ge=3)
    bands: list[float] = Field(default_factory=lambda: [1.0])
    show_extension: bool = False
    n_future: int = Field(50, ge=1)
    show_oscillator: bool = False
    show_accel_fan: bool = False
    m_accel: int = Field(50, ge=3)
    n_accel: int = Field(50, ge=1)


class AnalysisSettingsRequest(BaseModel):
    instrument_id: int
    interval: str
    analyzer_type: str
    params: dict


@router.post("/spectrogram")
async def spectrogram(req: SpectrogramRequest, db=Depends(get_db)):
    """
    STFT of causal Δratio — see sma/core/analysis/spectrogram.py. Synchronous
    (not routed through TaskManager): bounded by depth_bars, scipy.signal.
    spectrogram runs in well under a second even on multi-thousand-bar
    series, same cost class as the old /analyze endpoint this replaced.
    """
    inst = await get_instrument(db, req.ticker, req.data_source)
    if not inst:
        raise HTTPException(404, "Instrument not found — fetch candles first")

    candles = await get_candles(db, inst["id"], req.interval)
    if len(candles) < 2:
        raise HTTPException(400, f"Too few candles ({len(candles)})")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: compute_spectrogram(
                candles,
                depth_bars=req.depth_bars,
                nperseg=req.nperseg,
                overlap_pct=req.overlap_pct,
                fmin=req.fmin,
                fmax=req.fmax,
            ),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return result


@router.post("/trend-variance")
async def trend_variance(req: TrendVarianceRequest, db=Depends(get_db)):
    """
    Rolling OLS trend + residual-variance bands (+ optional oscillator /
    extension / acceleration fan) — see sma/core/analysis/trend_variance.py.
    Synchronous, same cost class as /spectrogram (O(window) for the main
    trend, O(n) only when show_oscillator=True — see that module's
    docstring for why the oscillator is gated).
    """
    inst = await get_instrument(db, req.ticker, req.data_source)
    if not inst:
        raise HTTPException(404, "Instrument not found — fetch candles first")

    candles = await get_candles(db, inst["id"], req.interval)
    if len(candles) < 3:
        raise HTTPException(400, f"Too few candles ({len(candles)})")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: compute_trend_variance(
                candles,
                window=req.window,
                bands=req.bands,
                show_extension=req.show_extension,
                n_future=req.n_future,
                show_oscillator=req.show_oscillator,
                show_accel_fan=req.show_accel_fan,
                m_accel=req.m_accel,
                n_accel=req.n_accel,
            ),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return result


@router.get("/analysis-settings")
async def get_analysis_settings_route(
    instrument_id: int = Query(...),
    interval: str = Query(...),
    analyzer_type: str = Query(...),
    db=Depends(get_db),
):
    """Last-used params for this (instrument, interval, analyzer_type) — see
    db.py analysis_settings table docstring. `{"params": null}` (not 404)
    when nothing is saved yet."""
    row = await get_analysis_settings(db, instrument_id, interval, analyzer_type)
    return {"params": row["params"] if row else None}


@router.post("/analysis-settings")
async def save_analysis_settings_route(body: AnalysisSettingsRequest, db=Depends(get_db)):
    """
    Client-saved (unlike forecast_defaults, which task_manager writes
    server-side after a completed forecast) — analyzer params include
    pure-display fields (e.g. spectrogram's contrast/logY) the backend never
    computes with, so the frontend is the only place that has the full set
    to persist. Called right after a successful analyzer run.
    """
    await upsert_analysis_settings(db, body.instrument_id, body.interval, body.analyzer_type, body.params)
    return {"ok": True}
