import { S } from './state.js';
import { renderChart, displayCandles } from './chart.js';
import { registerTool } from './tools.js';
import { saveToolDisplayDefaults } from './settings.js';

// ── volume analyzer (Осциллятор tab) ──────────────────────────────────────
// `volume` is already present on every candle from GET /candles
// (sma/core/db.py's candles table has had the column all along) —
// everything here is a pure client-side derived view of S.candles, no
// backend endpoint of its own either mode.
//
// Two modes (project request 2026-10-04):
//   'plain'   — standard red/green volume bars, color follows the SAME rule
//                the candlesticks themselves use (close >= open -> green,
//                else red), not a separate up/down-vs-previous-close
//                convention, so the two panels read consistently at a
//                glance. The ORIGINAL (and, until now, only) mode here.
//   'buysell' — splits each bar's volume into buy/sell via a CLV (Chaikin)
//                heuristic proxy: buy_fraction = (close-low)/(high-low)
//                (0.5 on a doji/flat bar, high==low), buy_volume =
//                volume·buy_fraction, sell_volume = volume·(1-buy_fraction).
//                BOTH drawn at once per bar (project feedback 2026-10-04:
//                a single net-delta bar hid which side actually drove a
//                low-conviction bar) — buy as a positive bar, sell as a
//                negative one, same x. Ported from sma-research эксп.120
//                (120_volume_profile_buysell), see memory
//                project_anomaly_analyzer_buysell_profile.
//
// Mode is the only setting (nothing else to configure) — a GLOBAL standing
// preference (project request 2026-10-04), not per-ticker: no origin/window
// concept here at all to split it from, unlike variance_oscillator.js. See
// settings.js:saveToolDisplayDefaults / sma/core/db.py:DEFAULT_TOOL_DISPLAY.

function redraw() {
  renderChart({ preserveRange: true });
}

const VOLUME_OSC_HINTS = {
  plain: 'Цвет бара — как у свечи (close ≥ open — зелёный, иначе красный).',
  buysell: 'Эвристика (CLV): объём делится на покупки/продажи по тому, ' +
    'ближе close к high или к low бара.',
};

function updateVolumeOscHint() {
  const el = document.getElementById('vol-osc-mode-hint');
  if (el) el.textContent = VOLUME_OSC_HINTS[S.volumeOscSettings.mode] ?? VOLUME_OSC_HINTS.plain;
}

function applyVolumeOscSettings(p) {
  if (!p) return;
  document.getElementById('vol-osc-mode').value = p.mode;
  S.volumeOscSettings = { mode: p.mode };
  updateVolumeOscHint();
}

// Called once at startup AND on every ticker switch (idempotent, same
// "harmless re-apply" pattern trend_ruler's showOnChart uses) — reads the
// GLOBAL slice, no network call of its own (S.toolDisplayDefaults is
// already populated by settings.js:loadAppSettings by the time any ticker
// can be selected).
export function loadVolumeOscDefaults() {
  applyVolumeOscSettings(S.toolDisplayDefaults.volume_osc);
}

// Pure display switch — both modes are derived from candles already in
// memory, so this never fetches anything, just recomputes on redraw.
export function setVolumeOscMode() {
  S.volumeOscSettings.mode = document.getElementById('vol-osc-mode').value;
  S.toolDisplayDefaults.volume_osc.mode = S.volumeOscSettings.mode;
  updateVolumeOscHint();
  redraw();
  saveToolDisplayDefaults();
}

// buy_fraction = (close-low)/(high-low), 0.5 on a flat bar (high==low —
// CLV is undefined there, 0.5 means "no lean either way"). buy+sell always
// sum back to the bar's own volume.
function buySellVolumes(c) {
  const range = c.high - c.low;
  const buyFraction = range > 0 ? (c.close - c.low) / range : 0.5;
  return { buy: c.volume * buyFraction, sell: c.volume * (1 - buyFraction) };
}

// ── registry entry: subpanel bar trace(s) ────────────────────────────────
function buildVolumeSubpanelTraces() {
  const bars = displayCandles();
  if (!bars.length) return [];
  if (S.volumeOscSettings.mode === 'buysell') {
    const up = S.colorProfile.volume_up, down = S.colorProfile.volume_down;
    const x = bars.map(c => c.begin);
    const buy = [], sell = [];
    for (const c of bars) {
      const { buy: b, sell: s } = buySellVolumes(c);
      buy.push(b); sell.push(-s); // sell drawn BELOW zero, same x as buy — baseLayout's barmode:'overlay' keeps both full-width
    }
    return [
      { type: 'bar', name: 'Покупки (CLV)', x, y: buy, yaxis: 'y2', marker: { color: up }, hovertemplate: '%{y:,.0f}<extra></extra>' },
      { type: 'bar', name: 'Продажи (CLV)', x, y: sell, yaxis: 'y2', marker: { color: down }, hovertemplate: '%{y:,.0f}<extra></extra>' },
    ];
  }
  const up = S.colorProfile.volume_up, down = S.colorProfile.volume_down; // settings.js
  return [{
    type: 'bar', name: 'Объём',
    x: bars.map(c => c.begin), y: bars.map(c => c.volume),
    yaxis: 'y2',
    marker: { color: bars.map(c => (c.close >= c.open ? up : down)) },
    hovertemplate: '%{y:,.0f}<extra></extra>',
  }];
}

// yaxis2 policy (project feedback 2026-08-20, §3.3 of docs/plans/
// frontend_improvements_plan.md): 'plain' STARTS with floor at 0 (volume is
// never negative); 'buysell' shows two SIGNED bars (buy up, sell down), so
// 0-centered symmetric instead, padded to whichever side is currently
// larger — same split variance_osc's slope/var modes use, for the same
// reason. Both zoomable (fixedrange:false).
function volumeYAxisPolicy() {
  const bars = displayCandles();
  if (!bars.length) return { key: 'volume', fixedrange: false, range: null };
  if (S.volumeOscSettings.mode === 'buysell') {
    let maxV = 0;
    for (const c of bars) {
      const { buy, sell } = buySellVolumes(c);
      if (buy > maxV) maxV = buy;
      if (sell > maxV) maxV = sell;
    }
    const pad = maxV * 1.05 || 1;
    return { key: 'volume:buysell', fixedrange: false, range: [-pad, pad] };
  }
  let maxV = 0;
  for (const c of bars) if (c.volume > maxV) maxV = c.volume;
  return { key: 'volume:plain', fixedrange: false, range: [0, maxV > 0 ? maxV * 1.05 : 1] };
}

registerTool({
  type: 'volume',
  icon: 'volume',
  label: 'Объём',
  panelId: 'tool-panel-volume',
  buildSubpanelTraces: buildVolumeSubpanelTraces,
  subpanelYAxisPolicy: volumeYAxisPolicy,
  // no buildMainTraces/buildOriginShape/onOriginClick/resetOrigin — this
  // tool never draws on the main chart and has no origin concept.
});
