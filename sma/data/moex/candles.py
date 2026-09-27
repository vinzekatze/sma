"""
MOEX ISS API — historical candles downloader.

CLI:
    python -m sma.data.moex SBER --intervals 1h 10m
    python -m sma.data.moex SBER GAZP --intervals 1h --out ./data/raw
    python -m sma.data.moex SBER --intervals 1h --from 2020-01-01

Embedded:
    from sma.data.moex import download_candles
    candles = download_candles("SBER", interval="1h")
    candles = download_candles("SBER", interval="10m", date_from="2020-01-01")
"""

from __future__ import annotations

import json
import time
import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


# data_source identifier used when persisting to the DB
DATA_SOURCE   = "moex"

BASE_URL      = "https://iss.moex.com/iss"
HISTORY_START = "2000-01-01"
PAGE_SIZE     = 500
REQUEST_DELAY = 0.3
MAX_RETRIES   = 3

# Codes accepted by the MOEX ISS /candles endpoint.
# Ref: docs/moex-api/README.md
INTERVALS: dict[str, int] = {
    "1m":  1,
    "10m": 10,
    "1h":  60,
    "1d":  24,
    "1w":  7,
    "1mo": 31,
}

# Default engine/market for ordinary exchange-traded shares (board TQBR).
# Override for futures (engine=futures, market=forts) or currencies, etc.
DEFAULT_ENGINE = "stock"
DEFAULT_MARKET = "shares"


# ── public API ────────────────────────────────────────────────────────────────

def download_candles(
    secid: str,
    interval: str,
    date_from: str = HISTORY_START,
    date_till: Optional[str] = None,
    engine: str = DEFAULT_ENGINE,
    market: str = DEFAULT_MARKET,
    show_progress: bool = True,
) -> list[dict]:
    """
    Download all historical OHLCV candles for a MOEX instrument.

    Args:
        secid:         MOEX ticker, e.g. "SBER", "GAZP", "Si-6.25"
        interval:      One of INTERVALS keys: 1m, 10m, 1h, 1d, 1w, 1mo
        date_from:     Start of range "YYYY-MM-DD". Default: full history from 2000.
        date_till:     End of range "YYYY-MM-DD". Default: today.
        engine:        MOEX ISS engine. Shares: "stock", futures: "futures".
        market:        MOEX ISS market. Shares: "shares", futures: "forts".
        show_progress: Show tqdm progress bar on stderr.

    Returns:
        List of dicts with keys: open, close, high, low, value, volume, begin, end.
    """
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {list(INTERVALS)}, got {interval!r}")
    if date_till is None:
        date_till = date.today().isoformat()

    return _fetch_all_pages(
        secid, INTERVALS[interval], interval,
        date_from, date_till, engine, market, show_progress,
    )


def save_candles(candles: list[dict], path: str | Path) -> None:
    """Write candles to *path* as pretty-printed JSON. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(candles, fh, ensure_ascii=False, indent=2)


# ── internals ────────────────────────────────────────────────────────────────

def _fetch_all_pages(
    secid: str,
    interval_code: int,
    interval_label: str,
    date_from: str,
    date_till: str,
    engine: str,
    market: str,
    show_progress: bool,
) -> list[dict]:
    url = (
        f"{BASE_URL}/engines/{engine}/markets/{market}"
        f"/securities/{secid}/candles.json"
    )
    params: dict = {
        "from":     date_from,
        "till":     date_till,
        "interval": interval_code,
        "iss.meta": "off",
        "iss.only": "candles",
        "start":    0,
    }

    all_candles: list[dict] = []
    bar = tqdm(
        desc=f"{secid} [{interval_label}]",
        unit="page",
        disable=not show_progress,
    )
    try:
        while True:
            batch = _get_page(url, params)
            if not batch:
                break
            all_candles.extend(batch)
            bar.update(1)
            bar.set_postfix_str(f"{len(all_candles):,} candles | last: {batch[-1]['begin']}")
            if len(batch) < PAGE_SIZE:
                break
            params["start"] += PAGE_SIZE
            time.sleep(REQUEST_DELAY)
    finally:
        bar.close()

    return all_candles


def _get_page(url: str, params: dict, attempt: int = 0) -> list[dict]:
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        block = resp.json()["candles"]
        return [dict(zip(block["columns"], row)) for row in block["data"]]
    except (requests.RequestException, KeyError, ValueError) as exc:
        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)
            return _get_page(url, params, attempt + 1)
        raise RuntimeError(
            f"Failed to fetch {url} after {MAX_RETRIES} retries"
        ) from exc


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    interval_list = ", ".join(INTERVALS)
    p = argparse.ArgumentParser(
        prog="python -m sma.data.moex",
        description="Download historical MOEX candles to JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Intervals: {interval_list}

Examples:
  python -m sma.data.moex SBER --intervals 1h 10m
  python -m sma.data.moex SBER GAZP LKOH --intervals 1h --out ./data/raw
  python -m sma.data.moex SBER --intervals 1h --from 2020-01-01
  python -m sma.data.moex Si-6.25 --intervals 1h --engine futures --market forts
""",
    )
    p.add_argument("tickers", nargs="+", metavar="TICKER",
                   help="One or more MOEX tickers, e.g. SBER GAZP Si-6.25")
    p.add_argument(
        "--intervals", nargs="+", default=["1h"],
        choices=list(INTERVALS), metavar="INTERVAL",
        help=f"Candle intervals ({interval_list}). Default: 1h",
    )
    p.add_argument("--from", dest="date_from", default=HISTORY_START,
                   metavar="DATE", help="Start date YYYY-MM-DD (default: %(default)s)")
    p.add_argument("--till", dest="date_till", default=None,
                   metavar="DATE", help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--out", default="./data/candles",
                   help="Output root directory (default: %(default)s)")
    p.add_argument("--engine", default=DEFAULT_ENGINE,
                   help="MOEX ISS engine (default: %(default)s)")
    p.add_argument("--market", default=DEFAULT_MARKET,
                   help="MOEX ISS market (default: %(default)s)")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    out_root = Path(args.out)

    for ticker in args.tickers:
        for interval in args.intervals:
            candles = download_candles(
                ticker, interval,
                date_from=args.date_from,
                date_till=args.date_till,
                engine=args.engine,
                market=args.market,
            )
            if not candles:
                print(f"[warn] no data returned: {ticker} [{interval}]", file=sys.stderr)
                continue
            out_path = out_root / ticker / f"{interval}.json"
            save_candles(candles, out_path)
            print(f"saved {len(candles):,} candles -> {out_path}")


if __name__ == "__main__":
    main()
