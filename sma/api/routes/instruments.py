from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import aiosqlite

from sma.core.db import (
    upsert_instrument,
    get_instrument,
    get_instrument_by_id,
    list_instruments,
    set_favorite,
    delete_instrument,
    list_instrument_coverage,
)
from sma.api.deps import get_db

router = APIRouter()

# UI-facing search categories -> underlying MOEX security group(s) (see
# sma/data/moex/securities.py:_GROUP_MAP for how each group resolves to
# engine/market/asset_type). "fund" merges two MOEX groups behind one label
# since ETFs and paev funds share asset_type="fund". "TQBR" isn't listed here
# — it's served by /board (a complete single-shot board listing, better than
# the paginated/capped group search for that specific, well-bounded case).
# A liquidity-ranked "top" category isn't exposed by this MOEX endpoint at
# all (would need a separate marketdata+sort query) — not implemented.
_SEARCH_CATEGORIES: dict[str, list[str]] = {
    "bond":     ["stock_bonds"],
    "fund":     ["stock_etf", "stock_ppif"],
    "index":    ["stock_index"],
    "currency": ["currency_selt"],
    "metal":    ["currency_metal"],
    "futures":  ["futures_forts"],
    "option":   ["futures_options"],
}


class InstrumentIn(BaseModel):
    ticker: str
    data_source: str = "moex"
    asset_type: str = "stock"
    full_name: str | None = None
    engine: str = "stock"
    market: str = "shares"
    board: str | None = None


class InstrumentOut(BaseModel):
    id: int
    ticker: str
    data_source: str
    asset_type: str
    full_name: str | None
    engine: str
    market: str
    board: str | None
    is_favorite: bool
    added_via: str


class InstrumentPatch(BaseModel):
    is_favorite: bool | None = None
    full_name: str | None = None


class CoverageOut(BaseModel):
    interval: str
    first: str
    last: str
    n: int


class SecurityOut(BaseModel):
    secid: str
    shortname: str | None
    name: str | None
    engine: str
    market: str
    asset_type: str
    board: str | None
    already_added: bool


def _to_out(row: dict) -> dict:
    return {**row, "is_favorite": bool(row["is_favorite"])}


@router.get("", response_model=list[InstrumentOut])
async def get_instruments(
    include_pool: bool = Query(False, description="Include auto-selected pool tickers (see added_via)"),
    db: aiosqlite.Connection = Depends(get_db),
):
    return [_to_out(r) for r in await list_instruments(db, include_pool=include_pool)]


@router.get("/search", response_model=list[SecurityOut])
async def search_instruments(
    q: str = Query(""),
    category: str | None = Query(None),
    db: aiosqlite.Connection = Depends(get_db),
):
    from sma.data.moex import search_securities

    groups = _SEARCH_CATEGORIES.get(category) if category else None
    if groups is None and not q:
        raise HTTPException(422, "q is required when no category is selected")

    def _run() -> list[dict]:
        if groups is None:
            return search_securities(q)
        # Merge multiple MOEX groups behind one category (e.g. "fund"),
        # de-duplicating by secid in case a security somehow appears twice.
        seen: set[str] = set()
        out: list[dict] = []
        for g in groups:
            for row in search_securities(q, group=g):
                if row["secid"] not in seen:
                    seen.add(row["secid"])
                    out.append(row)
        return out

    loop = asyncio.get_running_loop()
    hits = await loop.run_in_executor(None, _run)
    added = {row["ticker"] for row in await list_instruments(db, include_pool=True) if row["data_source"] == "moex"}
    return [{**hit, "already_added": hit["secid"] in added} for hit in hits]


@router.get("/board", response_model=list[SecurityOut])
async def board_instruments(
    engine: str = Query("stock"),
    market: str = Query("shares"),
    board: str = Query("TQBR"),
    db: aiosqlite.Connection = Depends(get_db),
):
    from sma.data.moex import list_board_securities

    loop = asyncio.get_running_loop()
    hits = await loop.run_in_executor(
        None, lambda: list_board_securities(engine, market, board)
    )
    added = {row["ticker"] for row in await list_instruments(db, include_pool=True) if row["data_source"] == "moex"}
    return [{**hit, "already_added": hit["secid"] in added} for hit in hits]


@router.post("", response_model=InstrumentOut, status_code=201)
async def create_instrument(body: InstrumentIn, db: aiosqlite.Connection = Depends(get_db)):
    iid = await upsert_instrument(
        db, body.ticker, body.data_source, body.asset_type, body.full_name,
        engine=body.engine, market=body.market, board=body.board,
    )
    row = await get_instrument_by_id(db, iid)
    return _to_out(row)


@router.get("/{instrument_id}", response_model=InstrumentOut)
async def get_instrument_by_id_route(
    instrument_id: int, db: aiosqlite.Connection = Depends(get_db)
):
    row = await get_instrument_by_id(db, instrument_id)
    if row is None:
        raise HTTPException(404, "Instrument not found")
    return _to_out(row)


@router.patch("/{instrument_id}", response_model=InstrumentOut)
async def patch_instrument(
    instrument_id: int, body: InstrumentPatch, db: aiosqlite.Connection = Depends(get_db)
):
    row = await get_instrument_by_id(db, instrument_id)
    if row is None:
        raise HTTPException(404, "Instrument not found")

    if body.is_favorite is not None:
        await set_favorite(db, instrument_id, body.is_favorite)
    if body.full_name is not None:
        await upsert_instrument(db, row["ticker"], row["data_source"], full_name=body.full_name)

    row = await get_instrument_by_id(db, instrument_id)
    return _to_out(row)


@router.delete("/{instrument_id}", status_code=204)
async def remove_instrument(instrument_id: int, db: aiosqlite.Connection = Depends(get_db)):
    ok = await delete_instrument(db, instrument_id)
    if not ok:
        raise HTTPException(404, "Instrument not found")


@router.get("/{instrument_id}/coverage", response_model=list[CoverageOut])
async def get_coverage(instrument_id: int, db: aiosqlite.Connection = Depends(get_db)):
    row = await get_instrument_by_id(db, instrument_id)
    if row is None:
        raise HTTPException(404, "Instrument not found")
    return await list_instrument_coverage(db, instrument_id)
