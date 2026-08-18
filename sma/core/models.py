from __future__ import annotations

from pydantic import BaseModel


class CandleBar(BaseModel):
    begin: str
    open: float
    high: float
    low: float
    close: float
    volume: float
