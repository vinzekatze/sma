import { S } from './state.js';
import { api, setStatus } from './api.js';
import { renderChart, futureDateAt, paperY } from './chart.js';
import { registerTool, isToolObjectVisible } from './tools.js';
import { getToolShowOnChart, saveToolShowOnChart } from './local_prefs.js';
import { hexToRgbTriplet, hexToRgba } from './color_utils.js';

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
// - Reacts INSTANTLY to every settings tweak — ported single_window_trend's
//   OLS math to JS so it runs on S.candles (already loaded, see
//   app.js:loadCandles) without a round trip.
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
// Same math as rolling_trend_variance/single_window_trend, just not
// vectorized (window/m_accel-bounded loops are plenty fast in JS).

// OLS fit of logPrices[origin-window+1 .. origin] — mirrors
// trend_variance.py:single_window_trend.
function localWindowTrend(logPrices, origin, window) {
  const j0 = origin - window + 1;
  if (j0 < 0) return null;
  const n = window;
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (let i = 0; i < n; i++) {
    const y = logPrices[j0 + i];
    sumX += i; sumY += y; sumXY += i * y; sumXX += i * i;
  }
  const denom = n * sumXX - sumX * sumX;
  const s = (n * sumXY - sumX * sumY) / denom;
  const b = (sumY - s * sumX) / n;
  const fitted = new Array(n);
  let ssr = 0;
  for (let i = 0; i < n; i++) {
    fitted[i] = b + s * i;
    const resid = logPrices[j0 + i] - fitted[i];
    ssr += resid * resid;
  }
  return { j0, fitted, std: Math.sqrt(ssr / (n - 2)), slope: s };
}

// slope[t] for t in [window-1, y.length-1] — mirrors
// trend_variance.py:rolling_trend_variance (var not needed here — that's
// variance_oscillator.js's job now — the accel fan only ever reads slope).
// O(n), same cumulative-sum trick.
function rollingSlope(y, window) {
  const n = y.length;
  const slope = new Array(n).fill(null);
  if (window < 3 || n < window) return slope;
  const csY = new Float64Array(n + 1), csJY = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    csY[i + 1] = csY[i] + y[i];
    csJY[i + 1] = csJY[i] + i * y[i];
  }
  const w = window;
  const Sx = w * (w - 1) / 2;
  const Sxx = (w - 1) * w * (2 * w - 1) / 6;
  const denom = w * Sxx - Sx * Sx;
  for (let t = w - 1; t < n; t++) {
    const j0 = t - w + 1;
    const Sy = csY[t + 1] - csY[j0];
    const Sxy = (csJY[t + 1] - csJY[j0]) - j0 * Sy;
    slope[t] = (w * Sxy - Sx * Sy) / denom;
  }
  return slope;
}

// Mirrors trend_variance.py:_slope_window_for_accel — slope at each of the
// last m_accel points ending at origin, on the minimal sub-array (not the
// full history).
function accelFan(logPrices, origin, window, mAccel, nAccel, localSlope, fittedLog) {
  const subStart = Math.max(0, origin - window - mAccel + 2);
  const subLog = logPrices.slice(subStart, origin + 1);
  const slopeSub = rollingSlope(subLog, window);
  const originLocal = slopeSub.length - 1;
  const fit = localWindowTrend(slopeSub, originLocal, mAccel);
  if (!fit) return null;
  const d0 = (fit.fitted[fit.fitted.length - 1] - fit.fitted[0]) / (mAccel - 1);
  const slopeHyp = localSlope + nAccel * d0;
  const price = new Array(window);
  for (let i = 0; i < window; i++) price[i] = Math.exp(fittedLog[0] + slopeHyp * i);
  return { price, direction: d0 >= 0 ? 'up' : 'down', d0 };
}

function extensionBands(fittedLog, stdLog, localSlope, bandList, nFuture) {
  if (!bandList.length) return null;
  const trendPrice = [];
  const bandsOut = bandList.map(k => ({ k, hi: [], lo: [] }));
  for (let h = 0; h <= nFuture; h++) {
    const logv = fittedLog[fittedLog.length - 1] + h * localSlope;
    trendPrice.push(Math.exp(logv));
    bandList.forEach((k, idx) => {
      bandsOut[idx].hi.push(Math.exp(logv + k * stdLog));
      bandsOut[idx].lo.push(Math.exp(logv - k * stdLog));
    });
  }
  return { n_future: nFuture, trend_price: trendPrice, bands: bandsOut };
}

// This analyzer's own origin resolves independently of any other analyzer's
// (S.originTs is the Прогноз tab's, untouched here) — null means "live",
// i.e. the last available bar. A click gives an exact date match almost
// always; the at-or-before fallback only matters for stale ISO strings.
function resolveOriginIndex(candles, originTs) {
  if (!originTs) return candles.length - 1;
  const targetDate = String(originTs).slice(0, 10);
  const exact = candles.findIndex(c => c.begin.slice(0, 10) === targetDate);
  if (exact !== -1) return exact;
  for (let i = candles.length - 1; i >= 0; i--) {
    if (candles[i].begin.slice(0, 10) <= targetDate) return i;
  }
  return candles.length - 1;
}

// Returns null if there isn't enough history before the origin for the
// current settings. Silently, on purpose — the form is mid-edit, not worth
// a toast, except when the null was caused by an explicit click, handled
// by the caller (setStatus in onTrendRulerSettingsChange).
function computeLivePreview(candles, settings, originTs) {
  const n = candles.length;
  if (!n || settings.window < 3) return null;
  const origin = resolveOriginIndex(candles, originTs);
  let originMin = settings.window - 1;
  if (settings.showAccelFan) originMin = Math.max(originMin, settings.window + settings.mAccel - 2);
  if (origin < originMin) return null;

  const logPrices = candles.map(c => Math.log(c.close));
  const fit = localWindowTrend(logPrices, origin, settings.window);
  if (!fit) return null;

  const segTimes = candles.slice(fit.j0, origin + 1).map(c => c.begin);
  const fittedPrice = fit.fitted.map(Math.exp);
  const bandList = [...new Set(settings.bands.filter(k => k > 0))].sort((a, b) => b - a);
  const bandsOut = bandList.map(k => ({
    k,
    hi: fit.fitted.map(v => Math.exp(v + k * fit.std)),
    lo: fit.fitted.map(v => Math.exp(v - k * fit.std)),
  }));

  // Extension is computed whenever EITHER extrapolation flag is on — the
  // drawing side (buildTrendRulerMainTraces) then independently decides
  // whether to render the fill and/or the border portion of it, so
  // "extrapolate bands but not borders" (or vice versa) is just a display
  // choice on the SAME computed data, not two separate computations.
  const needExtension = settings.extendBands || settings.extendBorders;

  return {
    origin_date: candles[origin].begin,
    window: settings.window,
    trend: { times: segTimes, price: fittedPrice },
    bands: bandsOut,
    extension: needExtension
      ? extensionBands(fit.fitted, fit.std, fit.slope, bandList, settings.nFuture)
      : null,
    accel_fan: settings.showAccelFan
      ? accelFan(logPrices, origin, settings.window, settings.mAccel, settings.nAccel, fit.slope, fit.fitted)
      : null,
  };
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
    showAccelFan:    document.getElementById('tr-show-accel-fan').checked,
    mAccel:          +document.getElementById('tr-m-accel').value,
    nAccel:          +document.getElementById('tr-n-accel').value,
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
  document.getElementById('tr-show-accel-fan').checked = !!p.showAccelFan;
  document.getElementById('tr-m-accel').value = p.mAccel;
  document.getElementById('tr-n-accel').value = p.nAccel;
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

// Last-used settings for this (instrument, interval) — same mechanism as
// analysis.js:loadSpectrogramDefaults.
export async function loadTrendRulerDefaults() {
  if (!S.instrumentId) return;
  try {
    const res = await api(
      'GET',
      `/series/analysis-settings?instrument_id=${S.instrumentId}&interval=${S.interval}&analyzer_type=trend_ruler`
    );
    if (res.params) applyTrendRulerSettings(res.params);
  } catch (_) { /* non-fatal — keeps current form values */ }
}

let _trendRulerSaveTimer = null;

function saveTrendRulerSettings() {
  if (!S.instrumentId) return;
  clearTimeout(_trendRulerSaveTimer);
  _trendRulerSaveTimer = setTimeout(() => {
    api('POST', '/series/analysis-settings', {
      instrument_id: S.instrumentId, interval: S.interval,
      analyzer_type: 'trend_ruler', params: S.trendRulerSettings,
    }).catch(() => {});
  }, 500);
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
  const preview = computeLivePreview(S.candles, settings, S.trendRulerOriginTs);
  if (preview) {
    S.trendRulerData = preview;
    document.getElementById('tr-meta').textContent =
      `origin=${preview.origin_date.slice(0, 10)} · окно=${preview.window} · полос=${preview.bands.length}`;
  } else if (S.candles.length) {
    setStatus(`Недостаточно истории до этой точки для окна ${settings.window}`, 'err');
  }
  updateOriginLabel();
  redraw();
  saveTrendRulerSettings();
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
  // see computeLivePreview).
  if (d.extension && (extendBands || extendBorders)) {
    const ext = d.extension;
    const extTimes = Array.from({ length: ext.n_future + 1 }, (_, h) => futureDateAt(d.origin_date, h, S.interval, S.candles));
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

  // "Fan" — where the window's trend line would land if the measured
  // (m_accel-averaged) acceleration acted for n_accel more steps, drawn
  // strictly WITHIN the window (not a forecast either — see accel_fan
  // computation in trend_variance.py).
  if (d.accel_fan) {
    const fillColor = hexToRgba(
      d.accel_fan.direction === 'up' ? S.colorProfile.trend_ruler_accel_up : S.colorProfile.trend_ruler_accel_down,
      0.25,
    );
    traces.push({
      type: 'scatter', mode: 'lines', x: trend.times, y: trend.price,
      line: { width: 0 }, showlegend: false, hoverinfo: 'skip',
    });
    traces.push({
      type: 'scatter', mode: 'lines', name: 'тренд + n·ускорение',
      x: trend.times, y: d.accel_fan.price,
      line: { color: S.colorProfile.trend_ruler_accel_line, width: 1.5, dash: 'dot' },
      fill: 'tonexty', fillcolor: fillColor,
    });
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
