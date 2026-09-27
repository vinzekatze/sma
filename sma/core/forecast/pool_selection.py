"""
Auto-selects a calibration/pretest pool for band_lambda: category(ies) ->
candidate securities -> liquidity ranking (per-engine metric) -> top N.

See docs/plans/band_forecast_migration_plan.md 5.2a for the research behind
the category map / thresholds / defaults below (measured 2026-07-31 — a
snapshot, not a constant; re-measure if numbers drift far from that table).

resolve_pool_candidates() does NOT touch the DB and does NOT persist
anything — it is a pure MOEX-ISS-facing function, called from
POST /forecast-settings/pool and POST /forecast-settings/resolve-pool (see
sma/api/routes/forecast_settings.py:_resolve_and_upsert_pool, which wraps
it with a TTL cache — pool_resolution_cache) — one source of truth, per
the plan.

Liquidity ranking note: the plan's research mentions a cheap "few requests"
resolution using board-level marketdata (VALTODAY/VOLTODAY). That MOEX ISS
response shape could not be re-verified in this environment (no outbound
network access when this module was written) and a single day's session
total is a noisier signal than a trailing window anyway. This implementation
instead reuses download_candles() — the one candle-fetching code path
already exercised throughout this app — over a short trailing window, and
gets its speed from concurrency (ThreadPoolExecutor) rather than from a
single bulk call. Trade-off: more HTTP calls, but every one of them hits an
endpoint already proven correct here.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

from sma.data.moex import download_candles, search_securities

# UI category keys -> MOEX security groups. Mirrors
# sma/api/routes/instruments.py:_SEARCH_CATEGORIES with two differences
# required specifically for pool selection (plan 5.2a):
#   - "fund" here means stock_ppif ONLY (ETFs are practically dead — 0/54
#     passed the liquidity threshold in the 2026-07-31 snapshot, all frozen
#     foreign ETFs). The general ticker-search "fund" category is
#     unaffected — browsing a dead ETF there is harmless.
#   - "option" is not offered at all — structurally unsuited as a pool
#     source (priced by strike/Greeks, not comparable to a spot zigzag).
POOL_CATEGORIES: dict[str, list[str]] = {
    "stock":    ["stock_shares", "stock_dr"],
    "bond":     ["stock_bonds"],
    "fund":     ["stock_ppif"],
    "currency": ["currency_selt"],
    "metal":    ["currency_metal"],
    "futures":  ["futures_forts"],
}

DEFAULT_N = 25   # research_large_scale_pooled_17 (эксп.17 Stage 1b): 25→45 tickers
                 # didn't help — see plan 5.2a point 7.
LIQUIDITY_WINDOW_DAYS = 30     # calendar days requested (~20 trading days)
MIN_COVERAGE_BARS = 15         # of the ~20 trading days expected in that window
DEFAULT_MAX_WORKERS = 16       # concurrent download_candles calls — fallback when the
                                # caller doesn't pass app_settings.moex_pool_workers
                                # (see sma/api/routes/settings.py)


def compute_pool_key(categories: list[str], n: int) -> str:
    """
    Deterministic string from a pool config — same categories (any order) +
    n always produce the same key. Used as pool_resolution_cache's cache
    key, and stored in pool_config purely as a human-readable label (e.g.
    "stock:25", "metal+stock:25") — band_lambda_pool itself is keyed on
    (instrument_id, interval) only, one pool per instrument now.
    """
    return "+".join(sorted(categories)) + f":{n}"


def resolve_pool_candidates(categories: list[str], n: int = DEFAULT_N, max_workers: int = DEFAULT_MAX_WORKERS) -> list[dict]:
    """
    categories -> ranked, liquidity-thresholded top-N instrument dicts
    ({secid, shortname, name, engine, market, asset_type, board}).

    Returns min(n, passed_threshold) — never pads with low-liquidity filler
    (plan 5.2a point 3). Unions + de-dups by secid across categories when
    more than one is given (e.g. "metal"+"stock" to make up for metal's
    shallow depth).

    max_workers caps concurrent MOEX requests during liquidity ranking —
    see app_settings.moex_pool_workers (sma/api/routes/settings.py), an
    app-level control for users on constrained/shared connections.
    """
    seen: set[str] = set()
    candidates: list[dict] = []
    for cat in categories:
        groups = POOL_CATEGORIES.get(cat)
        if groups is None:
            raise ValueError(f"Unknown pool category {cat!r}, expected one of {list(POOL_CATEGORIES)}")
        for group in groups:
            for row in search_securities("", group=group):
                if row["secid"] not in seen:
                    seen.add(row["secid"])
                    candidates.append(row)

    ranked = _rank_by_liquidity(candidates, max_workers)
    return ranked[:n]


def _rank_by_liquidity(candidates: list[dict], max_workers: int = DEFAULT_MAX_WORKERS) -> list[dict]:
    """Score every candidate by trailing-window liquidity (value for stock
    engine, volume for futures, bar-coverage for currency/metal — see module
    docstring and plan 5.2a point 2) and return those passing the coverage
    threshold, best first."""
    if not candidates:
        return []

    date_from = (date.today() - timedelta(days=LIQUIDITY_WINDOW_DAYS)).isoformat()
    scored: list[tuple[float, dict]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_liquidity_score, c, date_from): c
            for c in candidates
        }
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                score = fut.result()
            except Exception:
                continue
            if score is not None:
                scored.append((score, c))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [row for _, row in scored]


def _fetch_liquidity_score(candidate: dict, date_from: str) -> float | None:
    """Returns None if the candidate doesn't pass the coverage threshold."""
    try:
        bars = download_candles(
            candidate["secid"], "1d",
            date_from=date_from,
            engine=candidate["engine"], market=candidate["market"],
            show_progress=False,
        )
    except Exception:
        return None
    if len(bars) < MIN_COVERAGE_BARS:
        return None

    engine = candidate["engine"]
    if engine == "currency":
        # value/volume are always None for FX-style quotes (no lot concept,
        # see sma/core/db.py:upsert_candles) — bar-coverage itself is the score.
        return float(len(bars))
    field = "volume" if engine == "futures" else "value"
    total = sum(float(b[field]) for b in bars if b.get(field) is not None)
    return total if total > 0 else None
