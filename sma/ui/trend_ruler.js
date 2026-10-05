import { S } from './state.js';
import { api, setStatus } from './api.js';
import { renderChart, paperY } from './chart.js';
import { registerTool, isToolObjectVisible } from './tools.js';
import { getToolShowOnChart, saveToolShowOnChart } from './local_prefs.js';
import { hexToRgbTriplet } from './color_utils.js';
import { saveToolDisplayDefaults } from './settings.js';

// ── trend_ruler analyzer (Основной график tab) ──────────────────────────
// Main-chart half of what used to be a single combined "trend_variance"
// analyzer — split in two (project feedback 2026-08-18, round 5) because
// they're conceptually independent: this tool is the rolling OLS trend +
// ±k·std residual bands, drawn ONLY on the main chart, entirely client-side
// (see sma/core/analysis/trend_variance.py for the Python this JS mirrors —
// module name predates the split, still the same math). The OTHER half —
// full-history slope/var, which still needs the backend — is now its own
// oscillator-only tool, see variance_oscillator.js. That module reads this
// one's CURRENT window via S.trendRulerSettings.window for its "Подтянуть
// окно с линейки" button, but the two windows are otherwise independent —
// no live sync, a one-time pull.
//
// Design points carried over from before the split (project feedback
// 2026-08-18, earlier rounds):
// - Display-only tweaks react INSTANTLY; window/bands/origin/extension are
//   computed by the backend over the full history (debounced request, see
//   refreshTrendRulerPreview).
// - "Показывать на графике" is an explicit, persistent toggle (default
//   OFF), independent of tab/which tool is active — same model as forecast
//   pins, so several analyzers can show their own chart objects at once.
// - Its own origin (S.trendRulerOriginTs), default null = "live" (always
//   last bar). A candle click sets it, but ONLY when this tool is the
//   active one in the toolbar (see tools.js:selectTool + app.js
//   click routing).

function redraw() {
  renderChart({ preserveRange: true });
}

// ── client-side port of sma/core/analysis/trend_variance.py ────────────
// Live preview is computed on the BACKEND over the full stored history
// (POST /series/trend-ruler → sma/core/analysis/trend_ruler.py), not on the
// loaded candle window — the chart's candle limit is display-only.
// Sequenced: only the newest response is applied; settings edits are
// debounced so a slider drag sends one request, not one per tick.
let _trSeq = 0;
let _trTimer = null;

async function refreshTrendRulerPreview() {
  if (!S.instrumentId) return;
  const seq = ++_trSeq;
  const settings = S.trendRulerSettings;
  try {
    const preview = await api('POST', '/series/trend-ruler', {
      ticker: S.ticker, data_source: S.dataSource, interval: S.interval,
      window: settings.window, bands: settings.bands,
      origin_ts: S.trendRulerOriginTs ?? null,
      need_extension: settings.extendBands || settings.extendBorders,
      n_future: settings.nFuture,
    });
    if (seq !== _trSeq) return;
    if (preview) {
      S.trendRulerData = preview;
      document.getElementById('tr-meta').textContent =
        `origin=${preview.origin_date.slice(0, 10)} · окно=${preview.window} · полос=${preview.bands.length}`;
    } else {
      // Not enough history before the origin for this window. Previous data
      // stays drawn, same as before the move to the backend.
      setStatus(`Недостаточно истории до этой точки для окна ${settings.window}`, 'err');
    }
    redraw();
  } catch (e) {
    if (seq === _trSeq) setStatus(e.message, 'err');
  }
}

// ── bands list UI (произвольное число полос — project feedback 2026-08-18:
// a bare comma-separated text field wasn't a discoverable enough affordance
// for "you can add more than one band") ─────────────────────────────────

function renderBandsList() {
  const container = document.getElementById('tr-bands-list');
  container.innerHTML = '';
  S.trendRulerSettings.bands.forEach((k, i) => {
    const row = document.createElement('div');
    row.className = 'field-row';
    row.innerHTML = `
      <input type="number" step="0.1" min="0.1" value="${k}" style="flex:1" data-band-idx="${i}">
      <button type="button" data-band-remove="${i}" title="Убрать полосу" style="width:28px">×</button>
    `;
    container.appendChild(row);
  });
  container.querySelectorAll('input[data-band-idx]').forEach(inp => {
    inp.addEventListener('input', () => {
      const idx = +inp.dataset.bandIdx;
      S.trendRulerSettings.bands[idx] = parseFloat(inp.value) || 0;
      onTrendRulerSettingsChange();
    });
  });
  container.querySelectorAll('button[data-band-remove]').forEach(btn => {
    btn.addEventListener('click', () => {
      S.trendRulerSettings.bands.splice(+btn.dataset.bandRemove, 1);
      renderBandsList();
      onTrendRulerSettingsChange();
    });
  });
}

export function addTrendRulerBand() {
  const bands = S.trendRulerSettings.bands;
  const next = bands.length ? Math.round((Math.max(...bands) + 1) * 10) / 10 : 1;
  bands.push(next);
  renderBandsList();
  onTrendRulerSettingsChange();
}

// ── settings read/apply/persist ─────────────────────────────────────────

function readTrendRulerSettings() {
  return {
    window:          +document.getElementById('tr-window').value,
    bands:           [...S.trendRulerSettings.bands], // list UI is the source of truth, see renderBandsList
    showBands:       document.getElementById('tr-show-bands').checked,
    showBandBorders: document.getElementById('tr-show-band-borders').checked,
    bandOpacity:     +document.getElementById('tr-band-opacity').value / 100,
    extendBands:     document.getElementById('tr-extend-bands').checked,
    extendBorders:   document.getElementById('tr-extend-borders').checked,
    nFuture:         +document.getElementById('tr-n-future').value,
    // No longer a form field — toggled independently by the eye button in
    // the panel header (toggleTrendRulerShowChart), see §2.6 of
    // docs/plans/frontend_improvements_plan.md. Sourced from current state
    // so it survives every OTHER settings change reassigning
    // S.trendRulerSettings wholesale below.
    showOnChart:     S.trendRulerSettings.showOnChart,
  };
}

function applyTrendRulerSettings(p) {
  if (!p) return;
  document.getElementById('tr-window').value = p.window;
  document.getElementById('tr-window-range').value = p.window;
  document.getElementById('tr-show-bands').checked = p.showBands !== false;
  document.getElementById('tr-show-band-borders').checked = p.showBandBorders !== false;
  const opacityPct = Math.round((p.bandOpacity ?? 0.18) * 100);
  document.getElementById('tr-band-opacity').value = opacityPct;
  document.getElementById('tr-band-opacity-label').textContent = `${opacityPct}%`;
  document.getElementById('tr-extend-bands').checked = !!p.extendBands;
  document.getElementById('tr-extend-borders').checked = p.extendBorders !== false;
  document.getElementById('tr-n-future').value = p.nFuture;
  S.trendRulerSettings = { ...p, bands: [...p.bands] };
  renderBandsList();
  updateShowChartButton();
}

// Eye-icon toggle in the panel header (replaces the old "Показывать на
// графике" checkbox at the bottom of the panel, project feedback
// 2026-08-20, §2.6 of docs/plans/frontend_improvements_plan.md) — now means
// "закреплено": stays visible even while this tool ISN'T the active one
// (see buildTrendRulerMainTraces/buildTrendRulerOriginShape below, which OR
// it with S.activeMainTool === 'trend_ruler' so the live preview is always
// visible while actually configuring it, pinned or not).
export function toggleTrendRulerShowChart() {
  S.trendRulerSettings.showOnChart = !S.trendRulerSettings.showOnChart;
  updateShowChartButton();
  redraw();
  saveTrendRulerSettings();
  saveToolShowOnChart('trend_ruler', S.trendRulerSettings.showOnChart); // global, survives a reload/ticker switch — project feedback 2026-08-20, see local_prefs.js
}

function updateShowChartButton() {
  document.getElementById('tr-show-chart-btn')
    ?.classList.toggle('active', !!S.trendRulerSettings.showOnChart);
}

// Last-used WINDOW/BANDS for this (instrument, interval) — per-ticker,
// unlike the rest of the panel below (project request 2026-10-04: "для
// тренд линейки... кроме настроек окна и самих полос - они нужны
// per-tiker"). Merged with S.toolDisplayDefaults.trend_ruler's GLOBAL slice
// (already loaded once at startup, see settings.js:loadAppSettings) — that
// merge is what applyTrendRulerSettings actually writes to the form/state,
// so the rest of this module never needs to know which field came from
// which source.
export async function loadTrendRulerDefaults() {
  if (!S.instrumentId) return;
  let perTicker = { window: 200, bands: [2.0] };
  try {
    const res = await api(
      'GET',
      `/series/analysis-settings?instrument_id=${S.instrumentId}&interval=${S.interval}&analyzer_type=trend_ruler`
    );
    if (res.params?.window) perTicker = { window: res.params.window, bands: res.params.bands };
  } catch (_) { /* non-fatal — keeps current form values */ }
  applyTrendRulerSettings({
    ...perTicker, ...S.toolDisplayDefaults.trend_ruler,
    showOnChart: S.trendRulerSettings.showOnChart,
  });
}

let _trendRulerSaveTimer = null;

// Per-ticker half only — window + bands, see module docstring above.
function saveTrendRulerSettings() {
  if (!S.instrumentId) return;
  clearTimeout(_trendRulerSaveTimer);
  _trendRulerSaveTimer = setTimeout(() => {
    api('POST', '/series/analysis-settings', {
      instrument_id: S.instrumentId, interval: S.interval,
      analyzer_type: 'trend_ruler',
      params: { window: S.trendRulerSettings.window, bands: S.trendRulerSettings.bands },
    }).catch(() => {});
  }, 500);
}

// Global half — everything else in the panel (bands fill/border display,
// opacity, extension toggles, n_future). settings.js:saveToolDisplayDefaults
// owns the actual debounce/POST, shared across every tool that uses this
// mechanism, so this just updates the in-memory slice and asks it to save.
function saveGlobalTrendRulerDisplay() {
  const s = S.trendRulerSettings;
  S.toolDisplayDefaults.trend_ruler = {
    showBands: s.showBands, showBandBorders: s.showBandBorders, bandOpacity: s.bandOpacity,
    extendBands: s.extendBands, extendBorders: s.extendBorders, nFuture: s.nFuture,
  };
  saveToolDisplayDefaults();
}

// ── reactive live preview — this analyzer's entire compute/draw cycle
// is instant, no "Рассчитать" button at all (unlike variance_oscillator.js,
// which still needs one backend round trip) ──────────────────────────────

function updateOriginLabel() {
  const el = document.getElementById('tr-origin-label');
  if (!el) return;
  el.textContent = S.trendRulerOriginTs
    ? `клик: ${String(S.trendRulerOriginTs).slice(0, 10)}`
    : 'последний бар (живой)';
}

export function onTrendRulerSettingsChange() {
  const settings = readTrendRulerSettings();
  S.trendRulerSettings = settings;
  updateOriginLabel();
  redraw(); // display-only changes apply at once on the data we already have
  clearTimeout(_trTimer);
  _trTimer = setTimeout(refreshTrendRulerPreview, 250); // window/bands/origin/extension need the backend
  saveTrendRulerSettings();       // per-ticker: window + bands
  saveGlobalTrendRulerDisplay();  // global: bands display/opacity/extension/n_future
}

export function onTrendRulerWindowInput(value) {
  document.getElementById('tr-window').value = value;
  document.getElementById('tr-window-range').value = value;
  onTrendRulerSettingsChange();
}

// Own origin — set by clicking a candle while THIS analyzer's tool is
// active (see app.js click routing / tools.js:selectTool).
// Independent of S.originTs (Прогноз tab) and of any other analyzer's own
// origin.
export function setTrendRulerOrigin(ts) {
  S.trendRulerOriginTs = ts;
  onTrendRulerSettingsChange();
}

export function resetTrendRulerOrigin() {
  S.trendRulerOriginTs = null;
  onTrendRulerSettingsChange();
}

// Called once after candles (re)load (app.js:loadCandles) — populates the
// live preview immediately regardless of showOnChart (cheap, no request).
// showOnChart is then OVERRIDDEN by the global localStorage flag (project
// feedback 2026-08-20: pinned/unpinned should be a standing preference
// across tickers/reloads, not per-ticker DB state — see local_prefs.js) —
// runs AFTER loadTrendRulerDefaults() already applied whatever this
// ticker's own analysis_settings row happened to have, so the global
// preference always wins for THIS one field.
export function initTrendRulerForTicker() {
  renderBandsList(); // reflects state.js defaults even when nothing was persisted yet
  S.trendRulerSettings.showOnChart = getToolShowOnChart('trend_ruler', S.trendRulerSettings.showOnChart);
  onTrendRulerSettingsChange();
  updateShowChartButton();
}

// Visibility policy (project feedback 2026-08-20, §2.6 of
// docs/plans/frontend_improvements_plan.md): drawn when EITHER pinned
// (showOnChart, toggled by the eye button) OR this tool is the one
// currently active in the toolbar — so configuring it always shows a live
// preview even before you've decided to pin it, and pinning keeps it up
// once you switch to something else.
function isTrendRulerVisible() {
  return isToolObjectVisible('trend_ruler', S.trendRulerSettings.showOnChart);
}

// ── registry entry: main-chart traces + origin shape (moved out of
// chart.js so chart.js never needs to import this module — see
// tools.js) ───────────────────────────────────────────────────────

// Main-chart traces for the rolling OLS trend + ±k·std residual bands.
// Gated on this analyzer's OWN "показывать на графике" setting —
// independent of tab or which tool is active in the toolbar, same as a
// pinned forecast.
//
// Four independent display toggles (project feedback 2026-08-18, round 4):
// showBands (shaded fill) / showBandBorders (a visible line at each band's
// hi/lo edge — previously there was none, only the fill's own color
// implied the boundary, hard to tell apart with several overlapping bands)
// × extendBands / extendBorders (continue either past the window into the
// dashed future range). Any combination is valid, e.g. "borders only, but
// extrapolate the fill" — so the two loops below (in-window, extension)
// each check both flags independently rather than deriving one from the
// other. bandOpacity is ONE user-set alpha applied uniformly to every
// band's fill — same "same opacity, widest drawn first" stacking trick
// chart.js:buildBandZoneShapes already uses for band_lambda's zones, so
// Plotly's own compositing makes the center read denser without a manual
// per-band gradient.
function buildTrendRulerMainTraces() {
  const d = S.trendRulerData;
  if (!d || !isTrendRulerVisible()) return [];

  const traces = [];
  const trend = d.trend;
  const { showBands, showBandBorders, bandOpacity, extendBands, extendBorders } = S.trendRulerSettings;
  const lineColor = S.colorProfile.trend_ruler_line; // "Линия тренд-линейки" role — settings.js
  const BAND_COLOR = hexToRgbTriplet(lineColor); // bands share the trend line's own color

  if (showBands) {
    d.bands.forEach(b => {
      traces.push({
        type: 'scatter', mode: 'lines', x: trend.times, y: b.hi,
        line: { width: 0 }, showlegend: false, hoverinfo: 'skip',
      });
      traces.push({
        type: 'scatter', mode: 'lines', x: trend.times, y: b.lo,
        line: { width: 0 }, fill: 'tonexty', fillcolor: `rgba(${BAND_COLOR},${bandOpacity})`,
        name: `±${b.k}·std`, hoverinfo: 'skip',
      });
    });
  }

  if (showBandBorders) {
    d.bands.forEach(b => {
      traces.push({
        type: 'scatter', mode: 'lines', x: trend.times, y: b.hi, showlegend: false,
        line: { color: `rgb(${BAND_COLOR})`, width: 1, dash: 'dot' }, hoverinfo: 'skip',
      });
      traces.push({
        type: 'scatter', mode: 'lines', x: trend.times, y: b.lo, showlegend: false,
        line: { color: `rgb(${BAND_COLOR})`, width: 1, dash: 'dot' }, hoverinfo: 'skip',
      });
    });
  }

  traces.push({
    type: 'scatter', mode: 'lines', name: 'тренд окна',
    x: trend.times, y: trend.price,
    line: { color: lineColor, width: 2.5 },
  });

  // Dashed continuation past the window — linear extrapolation of the
  // window's own current slope, constant width. Visual reference
  // (regression-channel style), NOT a validated price forecast — see
  // trend_variance.py module docstring. The center trend line itself is
  // never continued, only the bands (fill and/or border, per the two
  // extend* flags — d.extension itself is computed whenever EITHER is on,
  // see refreshTrendRulerPreview).
  if (d.extension && (extendBands || extendBorders)) {
    const ext = d.extension;
    const extTimes = ext.times; // computed by the backend (sma/core/analysis/trend_ruler.py)
    if (extendBands) {
      ext.bands.forEach(b => {
        traces.push({
          type: 'scatter', mode: 'lines', x: extTimes, y: b.hi, showlegend: false,
          line: { width: 0 }, hoverinfo: 'skip',
        });
        traces.push({
          type: 'scatter', mode: 'lines', x: extTimes, y: b.lo, showlegend: false,
          line: { width: 0 }, fill: 'tonexty', fillcolor: `rgba(${BAND_COLOR},${bandOpacity})`, hoverinfo: 'skip',
        });
      });
    }
    if (extendBorders) {
      ext.bands.forEach(b => {
        traces.push({
          type: 'scatter', mode: 'lines', x: extTimes, y: b.hi, showlegend: false,
          line: { color: `rgb(${BAND_COLOR})`, width: 1, dash: 'dot' }, hoverinfo: 'skip',
        });
        traces.push({
          type: 'scatter', mode: 'lines', x: extTimes, y: b.lo, showlegend: false,
          line: { color: `rgb(${BAND_COLOR})`, width: 1, dash: 'dot' }, hoverinfo: 'skip',
        });
      });
    }
  }

  return traces;
}

// Vertical crosshair at this analyzer's OWN clicked origin (only when
// showOnChart AND an explicit origin was picked — "live" mode has nothing
// to point at, it's just always the last bar). Own configurable role
// (trend_ruler_origin) distinct from the Прогноз tab's origin line
// (next_origin_marker) so the two are never mistaken.
function buildTrendRulerOriginShape() {
  if (!isTrendRulerVisible() || !S.trendRulerOriginTs) return [];
  const [y0, y1] = paperY();
  return [{
    type: 'line', x0: S.trendRulerOriginTs, x1: S.trendRulerOriginTs, y0, y1, yref: 'paper',
    line: { color: S.colorProfile.trend_ruler_origin, width: 1, dash: 'dash' },
  }];
}

registerTool({
  type: 'trend_ruler',
  icon: 'trend',
  label: 'Линейка тренда',
  panelId: 'tool-panel-trend_ruler',
  buildMainTraces: buildTrendRulerMainTraces,
  buildOriginShape: buildTrendRulerOriginShape,
  onOriginClick: setTrendRulerOrigin,
  resetOrigin: resetTrendRulerOrigin,
});
