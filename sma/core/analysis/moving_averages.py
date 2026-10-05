"""Simple / exponential / weighted moving averages over the full candle history.

Port of the former client-side math in sma/ui/moving_averages.js (SMA, EMA
seeded with the SMA of the first `period` points, WMA with linear weights).
Moved to the backend so the averages cover every stored bar, not just the
window currently loaded in the browser — the chart's candle limit is a
display-only setting now (see sma/ui/chart.js:displayIndexRange).

Output keeps `None` for the warm-up bars (same as the JS `null` placeholders)
so the chart's connectgaps:false leaves them empty.
"""
from __future__ import annotations

SERIES_TYPES = ("sma", "ema", "wma")


def sma(closes: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    total = 0.0
    for i, c in enumerate(closes):
        total += c
        if i >= period:
            total -= closes[i - period]
        if i >= period - 1:
            out[i] = total / period
    return out


def wma(closes: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    denom = period * (period + 1) / 2
    for i in range(period - 1, len(closes)):
        acc = 0.0
        for j in range(period):
            acc += closes[i - period + 1 + j] * (j + 1)
        out[i] = acc / denom
    return out


def ema(closes: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    alpha = 2 / (period + 1)
    prev = None
    for i in range(len(closes)):
        if i < period - 1:
            continue
        if i == period - 1:
            prev = sum(closes[: i + 1]) / period  # seed = SMA of the first `period` points
        else:
            prev = closes[i] * alpha + prev * (1 - alpha)
        out[i] = prev
    return out


def compute_series(closes: list[float], kind: str, period: int) -> list[float | None]:
    if kind == "ema":
        return ema(closes, period)
    if kind == "wma":
        return wma(closes, period)
    return sma(closes, period)


def compute_moving_averages(
    candles: list[dict],
    series: list[dict],
) -> dict[str, dict]:
    """`series`: [{"id", "type", "period"}]. Series whose period is < 2 or
    longer than the history are skipped (the client draws nothing for them)."""
    closes = [float(c["close"]) for c in candles]
    times = [c["begin"] for c in candles]
    out: dict[str, dict] = {}
    for s in series:
        period = int(s["period"])
        if period < 2 or period > len(closes):
            continue
        out[str(s["id"])] = {
            "times": times,
            "values": compute_series(closes, s.get("type", "sma"), period),
        }
    return out
