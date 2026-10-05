import { S } from './state.js';
import { api, setBusy, setIdle, setStatus } from './api.js';
import { renderChart } from './chart.js';
import { registerTool } from './tools.js';
import { hexToRgba } from './color_utils.js';
import { saveToolDisplayDefaults } from './settings.js';

// ── variance_oscillator analyzer (Осциллятор tab) ────────────────────────
// Oscillator half of what used to be a single combined "trend_variance"
// analyzer — split in two (project feedback 2026-08-18, round 5), see
// trend_ruler.js's module docstring for the full rationale. This tool is
// ONLY the full-history rolling slope/var pass (sma/core/analysis/
// trend_variance.py:rolling_trend_variance) — the one part of the old
// combined analyzer that genuinely needs the backend (O(n) over the whole
// series, not a small window).
//
// Has its OWN `window` setting, independent of trend_ruler's — "Подтянуть
// окно с линейки" copies trend_ruler's CURRENT window value once (reading
// S.trendRulerSettings.window directly; no import of trend_ruler.js needed
// for that, S is already shared global state), rather than keeping the two
// permanently linked. You may deliberately want the oscillator's
// whole-history view at a different window than the ruler's local trend.
//
// oscMode (направление/дисперсия-внутри-окна/дисперсия-направления) is a
// GLOBAL standing preference (project request 2026-10-04), unlike
// window/slopeVarWindow which stay per-ticker — see loadVarianceOscDefaults/
// saveGlobalVarianceOscMode below and settings.js:saveToolDisplayDefaults.

function redraw() {
  renderChart({ preserveRange: true });
}

// ── settings read/apply/persist ─────────────────────────────────────────

function readVarianceOscSettings() {
  return {
    window:         +document.getElementById('vo-window').value,
    oscMode:        document.getElementById('vo-osc-mode').value,
    slopeVarWindow: +document.getElementById('vo-slopevar-window').value,
  };
}

function applyVarianceOscSettings(p) {
  if (!p) return;
  document.getElementById('vo-window').value = p.window;
  document.getElementById('vo-window-range').value = p.window;
  document.getElementById('vo-osc-mode').value = p.oscMode;
  // slopeVarWindow didn't exist before this field was added (2026-10-04) —
  // old saved settings rows won't have it, fall back to the prototype's own
  // default (app21-slope-variance-forecast.py: N=20).
  const slopeVarWindow = p.slopeVarWindow ?? 20;
  document.getElementById('vo-slopevar-window').value = slopeVarWindow;
  document.getElementById('vo-slopevar-window-range').value = slopeVarWindow;
  S.varianceOscSettings = { window: p.window, oscMode: p.oscMode, slopeVarWindow };
  syncSlopeVarWindowVisibility();
}

// The second window control only makes sense in 'slope_var' mode — hidden
// otherwise so the panel doesn't carry a confusing, inert slider for the
// other two modes.
function syncSlopeVarWindowVisibility() {
  const wrap = document.getElementById('vo-slopevar-window-wrap');
  if (wrap) wrap.style.display = S.varianceOscSettings.oscMode === 'slope_var' ? '' : 'none';
}

// Last-used WINDOW/SLOPEVARWINDOW for this (instrument, interval) — oscMode
// is GLOBAL (see applyGlobalVarianceOscMode), applied AFTER this so it wins
// regardless of whatever an older per-ticker row happens to still carry.
export async function loadVarianceOscDefaults() {
  if (!S.instrumentId) return;
  let perTicker = { window: 200, slopeVarWindow: 20 };
  try {
    const res = await api(
      'GET',
      `/series/analysis-settings?instrument_id=${S.instrumentId}&interval=${S.interval}&analyzer_type=variance_osc`
    );
    if (res.params) perTicker = { window: res.params.window, slopeVarWindow: res.params.slopeVarWindow };
  } catch (_) { /* non-fatal — keeps current form values */ }
  applyVarianceOscSettings({ ...perTicker, oscMode: S.toolDisplayDefaults.variance_osc.oscMode });
}

let _varianceOscSaveTimer = null;

// Per-ticker half only — window + slopeVarWindow, see loadVarianceOscDefaults.
function saveVarianceOscSettings() {
  if (!S.instrumentId) return;
  clearTimeout(_varianceOscSaveTimer);
  _varianceOscSaveTimer = setTimeout(() => {
    api('POST', '/series/analysis-settings', {
      instrument_id: S.instrumentId, interval: S.interval,
      analyzer_type: 'variance_osc',
      params: { window: S.varianceOscSettings.window, slopeVarWindow: S.varianceOscSettings.slopeVarWindow },
    }).catch(() => {});
  }, 500);
}

// Global half — oscMode. settings.js:saveToolDisplayDefaults owns the
// actual debounce/POST, shared across every tool using this mechanism.
function saveGlobalVarianceOscMode() {
  S.toolDisplayDefaults.variance_osc.oscMode = S.varianceOscSettings.oscMode;
  saveToolDisplayDefaults();
}

// ── oscillator (backend, full-history rolling pass) ─────────────────────
// bands/show_extension are irrelevant here — this endpoint call only ever
// wants the oscillator field — sent as off/empty to keep the request/
// compute minimal (same reasoning trend_ruler.js's predecessor had before
// the split).
export async function calculateVarianceOscillator() {
  if (!S.ticker) { setStatus('Сначала загрузите свечи', 'err'); return; }
  const settings = readVarianceOscSettings();
  S.varianceOscSettings = settings;
  setBusy('Расчёт осциллятора…');
  try {
    const res = await api('POST', '/series/trend-variance', {
      ticker: S.ticker, data_source: S.dataSource, interval: S.interval,
      window: settings.window, bands: [],
      show_extension: false, show_oscillator: true,
    });
    S.varianceOscData = res.oscillator;
    redraw();
    saveVarianceOscSettings();
    setIdle('Осциллятор рассчитан');
  } catch (e) {
    setIdle(e.message, false);
  }
}

// Window is the only setting the oscillator depends on (rolling_trend_
// variance takes window, nothing else) — debounced so dragging the slider
// doesn't fire a request per tick. Only refetches while this tool is
// actually the active/visible oscillator (project feedback 2026-08-18's
// original reasoning: no point recomputing something nobody's looking at —
// same idea, now keyed off S.activeOscillatorTool instead of a separate
// "показать" checkbox, see tools.js §3.1).
let _oscRefreshTimer = null;

export function onVarianceOscWindowInput(value) {
  document.getElementById('vo-window').value = value;
  document.getElementById('vo-window-range').value = value;
  if (S.activeOscillatorTool !== 'variance_osc') return;
  clearTimeout(_oscRefreshTimer);
  _oscRefreshTimer = setTimeout(calculateVarianceOscillator, 500);
}

// Pure display switch (направление/дисперсия-внутри-окна/дисперсия-
// направления) — all three ultimately read off the same slope[]/var[]
// fetch (rolling_trend_variance computes both together; slope_var is a
// client-side rolling-variance pass over slope[], see
// rollingVarianceOfSlope below), so switching mode never triggers a new
// request.
export function setVarianceOscMode() {
  if (!S.varianceOscData) return;
  S.varianceOscSettings.oscMode = document.getElementById('vo-osc-mode').value;
  syncSlopeVarWindowVisibility();
  redraw();
  saveGlobalVarianceOscMode();
}

// Unlike the main window (onVarianceOscWindowInput), changing this one
// never needs a network round-trip — the slope[] array it operates on is
// already cached client-side from the last /series/trend-variance fetch;
// rolling variance OF that array is a plain client computation
// (rollingVarianceOfSlope below).
let _slopeVarRefreshTimer = null;

export function onVarianceOscSlopeVarWindowInput(value) {
  document.getElementById('vo-slopevar-window').value = value;
  document.getElementById('vo-slopevar-window-range').value = value;
  S.varianceOscSettings.slopeVarWindow = +value;
  if (S.activeOscillatorTool !== 'variance_osc' || !S.varianceOscData) return;
  clearTimeout(_slopeVarRefreshTimer);
  _slopeVarRefreshTimer = setTimeout(() => { redraw(); saveVarianceOscSettings(); }, 300);
}

// Copies trend_ruler's CURRENT window into this tool's own window field —
// a one-time pull, not a live link (see module docstring). Refetches if
// the oscillator is currently shown, same as any other window change.
export function pullWindowFromRuler() {
  const w = S.trendRulerSettings.window;
  document.getElementById('vo-window').value = w;
  document.getElementById('vo-window-range').value = w;
  if (S.activeOscillatorTool === 'variance_osc') calculateVarianceOscillator();
}

// Called when this tool becomes the active oscillator (tools.js:selectTool/
// activateTool's onSelected hook, §3.1 of docs/plans/frontend_
// improvements_plan.md) — fetches only if there's no data yet (e.g. first
// selection this session/ticker); re-selecting an already-computed
// oscillator is free, tools.js already redraws after selection.
function onVarianceOscSelected() {
  if (!S.varianceOscData) calculateVarianceOscillator();
}

// Called once after candles (re)load (app.js:loadCandles) — no longer
// restores "was it showing" (that's now a single cross-tool concept,
// S.activeOscillatorTool, reset to 'none' on ticker switch same as
// S.activeMainTool resets to 'free' — see app.js:loadCandles); kept as a
// documented no-op rather than removing the call site, consistent with
// every other analyzer's init hook.
export function initVarianceOscForTicker() {}

// Rolling sample variance (ddof=1) of the slope series itself, over `win`
// trailing points — ported from app21-slope-variance-forecast.py:
// compute_slope_var's `pd.Series(slope).rolling(n_win,
// min_periods=n_win).var(ddof=1)`. slope[] already carries `window-1`
// leading nulls from rolling_trend_variance (see trend_variance.py); this
// pass only ever reads a FULLY non-null trailing window before producing a
// value, so the combined series is null until both windows are satisfied —
// still strictly causal, same guarantee the prototype relied on.
function rollingVarianceOfSlope(slope, win) {
  const n = slope.length;
  const out = new Array(n).fill(null);
  if (win < 2) return out;
  for (let t = win - 1; t < n; t++) {
    let sum = 0, sumSq = 0, ok = true;
    for (let i = t - win + 1; i <= t; i++) {
      const v = slope[i];
      if (v == null) { ok = false; break; }
      sum += v; sumSq += v * v;
    }
    if (!ok) continue;
    const mean = sum / win;
    out[t] = (sumSq - win * mean * mean) / (win - 1);
  }
  return out;
}

// Memoized on (slope array identity, window) — buildVarianceOscSubpanelTraces
// AND varianceOscYAxisPolicy both need this series on the same render pass;
// avoids computing it twice.
let _slopeVarCache = { slope: null, win: null, data: null };

function getSlopeVarSeries(osc, win) {
  if (_slopeVarCache.slope === osc.slope && _slopeVarCache.win === win) return _slopeVarCache.data;
  const data = rollingVarianceOfSlope(osc.slope, win);
  _slopeVarCache = { slope: osc.slope, win, data };
  return data;
}

// ── registry entry: subpanel traces (moved out of chart.js so chart.js
// never needs to import this module — see tools.js) ─────────────────
function buildVarianceOscSubpanelTraces() {
  const osc = S.varianceOscData;
  if (!osc) return [];
  if (S.varianceOscSettings.oscMode === 'var') {
    const color = S.colorProfile.variance_var; // "Дисперсия внутри окна тренда" role — settings.js
    return [{
      type: 'scatter', mode: 'lines', name: 'дисперсия внутри окна тренда',
      x: osc.times, y: osc.var, yaxis: 'y2',
      line: { color, width: 1 },
      fill: 'tozeroy', fillcolor: hexToRgba(color, 0.2),
    }];
  }
  if (S.varianceOscSettings.oscMode === 'slope_var') {
    const color = S.colorProfile.variance_slopevar; // "Дисперсия направления тренда" role — settings.js
    const sv = getSlopeVarSeries(osc, S.varianceOscSettings.slopeVarWindow);
    return [{
      type: 'scatter', mode: 'lines', name: 'дисперсия направления тренда',
      x: osc.times, y: sv, yaxis: 'y2',
      line: { color, width: 1 },
      fill: 'tozeroy', fillcolor: hexToRgba(color, 0.2),
    }];
  }
  const up = S.colorProfile.variance_slope_up, down = S.colorProfile.variance_slope_down;
  return [{
    type: 'bar', name: 'направление тренда',
    x: osc.times, y: osc.slope, yaxis: 'y2',
    marker: { color: osc.slope.map(v => (v == null ? 'rgba(0,0,0,0)' : (v >= 0 ? up : down))) },
  }];
}

function maxAbs(arr) {
  let m = 0;
  for (const v of arr) if (v != null && Math.abs(v) > m) m = Math.abs(v);
  return m;
}

function maxOf(arr) {
  let m = 0;
  for (const v of arr) if (v != null && v > m) m = v;
  return m;
}

// yaxis2 policy (project feedback 2026-08-20, §3.3 of docs/plans/
// frontend_improvements_plan.md): the two display modes need opposite
// INITIAL centering, so `key` includes oscMode — switching between them
// forces a fresh range instead of inheriting whichever mode's stale range
// happened to be current (see chart.js:applyY2RangePolicy for why that
// distinction matters). Both are zoomable (fixedrange:false) — an earlier
// pass fixed var's floor at 0 with NO zoom at all, but follow-up feedback
// the same day asked for zoom there too ("дисперсия и объем — нет
// возможности зумировать"): the computed range below is only the STARTING
// view, not an enforced clamp.
// - slope (направление тренда): 0 exactly centered — symmetric [-max|slope|, +max|slope|].
// - var (дисперсия внутри окна тренда) / slope_var (дисперсия направления
//   тренда): both start with floor at 0 (variance is never negative), auto
//   ceiling — but the user can zoom/pan away from that afterward.
function varianceOscYAxisPolicy() {
  const osc = S.varianceOscData;
  const mode = S.varianceOscSettings.oscMode;
  if (mode === 'var') {
    if (!osc) return { key: 'variance_osc:var', fixedrange: false, range: null };
    const maxV = maxOf(osc.var);
    return { key: 'variance_osc:var', fixedrange: false, range: [0, maxV > 0 ? maxV * 1.05 : 1] };
  }
  if (mode === 'slope_var') {
    if (!osc) return { key: 'variance_osc:slope_var', fixedrange: false, range: null };
    const sv = getSlopeVarSeries(osc, S.varianceOscSettings.slopeVarWindow);
    const maxV = maxOf(sv);
    return { key: 'variance_osc:slope_var', fixedrange: false, range: [0, maxV > 0 ? maxV * 1.05 : 1] };
  }
  if (!osc) return { key: 'variance_osc:slope', fixedrange: false, range: null };
  const pad = maxAbs(osc.slope) * 1.05 || 1;
  return { key: 'variance_osc:slope', fixedrange: false, range: [-pad, pad] };
}

registerTool({
  type: 'variance_osc',
  icon: 'variance',
  label: 'Осциллятор тренда',
  panelId: 'tool-panel-variance_osc',
  onSelected: onVarianceOscSelected,
  buildSubpanelTraces: buildVarianceOscSubpanelTraces,
  subpanelYAxisPolicy: varianceOscYAxisPolicy,
  // no buildMainTraces/buildOriginShape/onOriginClick/resetOrigin — this
  // tool never draws on the main chart and has no origin concept.
});
