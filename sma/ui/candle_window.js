// candle_window.js — keeps only a bounded window of candles loaded/rendered
// on the main chart (project feedback: tickers with a long intraday history
// load their ENTIRE history at once otherwise — ~250k bars measured on a
// 10m-interval ticker — which Firefox renders far worse than Chrome; see
// docs/plans note in the session this was written). app.js:loadCandles
// already requests only the last S.chartWindowBars bars (a global app_setting,
// "Настройки приложения" → «Свечей на графике», settings.js) on ticker/
// interval load; this module's only job is fetching an older chunk and
// PREPENDING it to S.candles when the user pans left close enough to the
// currently loaded window's edge — no trimming from the right in this first
// version, so a session can only grow the loaded window backward as far as
// the user actually pans, never shrink it.

import { S } from './state.js';
import { api, setStatus } from './api.js';
import { renderChart, getCurrentRanges, registerRelayoutHook } from './chart.js';

let _fetching = false;

// "Close enough to the left edge" — within half a screen's worth of bars
// from the oldest loaded candle, so the fetch lands before the user actually
// pans into a wall of nothing (network round trip has time to land while
// they're still looking at loaded data).
function _shouldFetchOlder() {
  if (_fetching || S.candlesFullyLoadedBack || S.candles.length < 2) return false;
  const { x } = getCurrentRanges();
  if (!x) return false;
  const visibleMs = new Date(x[1]).getTime() - new Date(x[0]).getTime();
  const oldestLoadedMs = new Date(S.candles[0].begin).getTime();
  return new Date(x[0]).getTime() <= oldestLoadedMs + visibleMs * 0.5;
}

async function _fetchOlderChunk() {
  if (!S.instrumentId) return;
  _fetching = true;
  const oldest = S.candles[0];
  try {
    // until is inclusive (sma/core/db.py:get_candles) — asking for
    // chartWindowBars+1 and dropping the row matching the current oldest
    // bar's id sidesteps having to reason about an off-by-one there.
    const rows = await api('GET',
      `/candles?ticker=${S.ticker}&data_source=${S.dataSource}&interval=${S.interval}` +
      `&until=${oldest.begin}&limit=${S.chartWindowBars + 1}`);
    const older = rows.filter(r => r.id !== oldest.id);
    if (rows.length <= S.chartWindowBars) S.candlesFullyLoadedBack = true;
    if (older.length) {
      S.candles = [...older, ...S.candles];
      renderChart({ preserveRange: true }); // same visible x-range — nothing visually jumps, there's just more to pan into now
    }
  } catch (e) {
    setStatus(e.message, 'err');
  } finally {
    _fetching = false;
  }
}

export function initCandleWindowing() {
  registerRelayoutHook(() => {
    if (_shouldFetchOlder()) _fetchOlderChunk();
  });
}
