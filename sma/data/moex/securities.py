"""
MOEX ISS API — instrument discovery (search + board listing).

Embedded:
    from sma.data.moex import search_securities, list_board_securities
    hits = search_securities("сбер")
    tqbr = list_board_securities()  # engine=stock, market=shares, board=TQBR
"""

from __future__ import annotations

import requests

from .candles import BASE_URL

REQUEST_TIMEOUT = 15

# /securities.json search results don't carry engine/market columns — MOEX only
# exposes those per-board via /securities/{secid}.json. The result's `group`
# field is a stable proxy for the (engine, market, asset_type) triple, so
# results are filtered/enriched via this map instead of an extra round-trip
# per hit. Each entry below was confirmed against the real MOEX candles
# endpoint (not just /securities.json) before inclusion.
#
# Deliberately excluded despite existing as MOEX groups:
#   currency_futures — engine/market resolves fine (currency/selt) but every
#     active instrument checked returned zero candles (thin forward market).
#   stock_eurobond   — primary board is a repo market, not a price series.
#   stock_foreign_shares, currency_indices/otcindices, stock_qnv/gcc/deposit/
#     mortgage — niche/fixing/repo instruments, no confirmed working example.
_GROUP_MAP: dict[str, tuple[str, str, str]] = {
    "stock_shares":    ("stock", "shares", "stock"),
    "stock_dr":        ("stock", "shares", "stock"),
    "stock_etf":       ("stock", "shares", "fund"),
    "stock_ppif":      ("stock", "shares", "fund"),
    "stock_bonds":     ("stock", "bonds", "bond"),
    "stock_index":     ("stock", "index", "index"),
    "currency_selt":   ("currency", "selt", "currency"),
    "currency_metal":  ("currency", "selt", "metal"),
    "futures_forts":   ("futures", "forts", "futures"),
    "futures_options": ("futures", "options", "option"),
}


def search_securities(query: str, group: str | None = None) -> list[dict]:
    """
    Search MOEX instruments by ticker or name, restricted to groups in
    _GROUP_MAP. Each row is normalized to {secid, shortname, name, engine,
    market, asset_type, board}.

    group: restrict to one MOEX security group (e.g. "stock_bonds", any key
    of _GROUP_MAP). Passed through as group_by=group&group_by_filter=<group>,
    which lets MOEX return every match in that group even with query="" —
    this is what powers the search category filter (empty text + category
    picked = "show everything in this category").
    """
    params: dict = {"q": query, "iss.meta": "off"}
    if group:
        params["group_by"] = "group"
        params["group_by_filter"] = group
    resp = requests.get(
        f"{BASE_URL}/securities.json",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    block = resp.json()["securities"]
    rows = [dict(zip(block["columns"], row)) for row in block["data"]]

    out = []
    for row in rows:
        mapping = _GROUP_MAP.get(row.get("group"))
        if mapping is None:
            continue
        engine, market, asset_type = mapping
        out.append({
            "secid":      row.get("secid"),
            "shortname":  row.get("shortname"),
            "name":       row.get("name"),
            "engine":     engine,
            "market":     market,
            "asset_type": asset_type,
            "board":      row.get("primary_boardid"),
        })
    return out


def list_board_securities(
    engine: str = "stock",
    market: str = "shares",
    board: str = "TQBR",
    asset_type: str = "stock",
) -> list[dict]:
    """
    List all securities traded on a given board (e.g. TQBR — main shares
    board). Normalized to the same shape as search_securities().
    """
    resp = requests.get(
        f"{BASE_URL}/engines/{engine}/markets/{market}/boards/{board}/securities.json",
        params={"iss.meta": "off"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    block = resp.json()["securities"]
    rows = [dict(zip(block["columns"], row)) for row in block["data"]]

    return [
        {
            "secid":      row.get("SECID"),
            "shortname":  row.get("SHORTNAME"),
            "name":       row.get("SECNAME"),
            "engine":     engine,
            "market":     market,
            "asset_type": asset_type,
            "board":      row.get("BOARDID", board),
        }
        for row in rows
    ]


def get_security_history_range(secid: str) -> list[dict]:
    """
    Per-security board coverage (engine/market/board + history_from/
    history_till), from /securities/{secid}.json. Cheap (no candle data) —
    used as pretest level 1 (docs/plans/band_forecast_migration_plan.md 5.2b):
    a quick "how much history exists at all" check before committing to a
    real download. Returns the raw `boards` rows (list of dicts keyed by
    column name).
    """
    resp = requests.get(
        f"{BASE_URL}/securities/{secid}.json",
        params={"iss.meta": "off", "iss.only": "boards"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    block = resp.json()["boards"]
    return [dict(zip(block["columns"], row)) for row in block["data"]]
