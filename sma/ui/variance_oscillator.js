import { S } from './state.js';
import { api, setBusy, setIdle, setStatus } from './api.js';
import { renderChart } from './chart.js';
import { registerTool } from './tools.js';
import { hexToRgba } from './color_utils.js';

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

function redraw() {
  renderChart({ preserveRange: true });
}

// ── settings read/apply/persist ─────────────────────────────────────────

function readVarianceOscSettings() {
  return {
    window:  +document.getElementById('vo-window').value,
    oscMode: document.getElementById('vo-osc-mode').value,
  };
}

function applyVarianceOscSettings(p) {
  if (!p) return;
  document.getElementById('vo-window').value = p.window;
  document.getElementById('vo-window-range').value = p.window;
  document.getElementById('vo-osc-mode').value = p.oscMode;
  S.varianceOscSettings = { window: p.window, oscMode: p.oscMode };
}

// Last-used settings for this (instrument, interval) — same mechanism as
// analysis.js:loadSpectrogramDefaults.
export async function loadVarianceOscDefaults() {
  if (!S.instrumentId) return;
  try {
    const res = await api(
      'GET',
      `/series/analysis-settings?instrument_id=${S.instrumentId}&interval=${S.interval}&analyzer_type=variance_osc`
    );
    if (res.params) applyVarianceOscSettings(res.params);
  } catch (_) { /* non-fatal — keeps current form values */ }
}

let _varianceOscSaveTimer = null;

function saveVarianceOscSettings() {
  if (!S.instrumentId) return;
  clearTimeout(_varianceOscSaveTimer);
  _varianceOscSaveTimer = setTimeout(() => {
    api('POST', '/series/analysis-settings', {
      instrument_id: S.instrumentId, interval: S.interval,
      analyzer_type: 'variance_osc', params: S.varianceOscSettings,
    }).catch(() => {});
  }, 500);
}

// ── oscillator (backend, full-history rolling pass) ─────────────────────
// bands/show_extension/show_accel_fan are irrelevant here — this endpoint
// call only ever wants the oscillator field — sent as off/empty to keep
// the request/compute minimal (same reasoning trend_ruler.js's predecessor
// had before the split).
export async function calculateVarianceOscillator() {
  if (!S.ticker) { setStatus('Сначала загрузите свечи', 'err'); return; }
  const settings = readVarianceOscSettings();
  S.varianceOscSettings = settings;
  setBusy('Расчёт осциллятора…');
  try {
    const res = await api('POST', '/series/trend-variance', {
      ticker: S.ticker, data_source: S.dataSource, interval: S.interval,
      window: settings.window, bands: [],
      show_extension: false, show_oscillator: true, show_accel_fan: false,
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

// Pure display switch (slope vs var) — both already came back from the
// last fetch (rolling_trend_variance computes both together), so this
// never triggers a new request.
export function setVarianceOscMode() {
  if (!S.varianceOscData) return;
  S.varianceOscSettings.oscMode = document.getElementById('vo-osc-mode').value;
  redraw();
  saveVarianceOscSettings();
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

// ── registry entry: subpanel traces (moved out of chart.js so chart.js
// never needs to import this module — see tools.js) ─────────────────
function buildVarianceOscSubpanelTraces() {
  const osc = S.varianceOscData;
  if (!osc) return [];
  if (S.varianceOscSettings.oscMode === 'var') {
    const color = S.colorProfile.variance_var; // "Дисперсия: величина" role — settings.js
    return [{
      type: 'scatter', mode: 'lines', name: 'resid_var (дисперсия)',
      x: osc.times, y: osc.var, yaxis: 'y2',
      line: { color, width: 1 },
      fill: 'tozeroy', fillcolor: hexToRgba(color, 0.2),
    }];
  }
  const up = S.colorProfile.variance_slope_up, down = S.colorProfile.variance_slope_down;
  return [{
    type: 'bar', name: 'slope (тренд)',
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
// - slope (тренд): 0 exactly centered — symmetric [-max|slope|, +max|slope|].
// - var (дисперсия): starts with floor at 0 (variance is never negative),
//   auto ceiling — but the user can zoom/pan away from that afterward.
function varianceOscYAxisPolicy() {
  const osc = S.varianceOscData;
  if (S.varianceOscSettings.oscMode === 'var') {
    if (!osc) return { key: 'variance_osc:var', fixedrange: false, range: null };
    const maxV = maxOf(osc.var);
    return { key: 'variance_osc:var', fixedrange: false, range: [0, maxV > 0 ? maxV * 1.05 : 1] };
  }
  if (!osc) return { key: 'variance_osc:slope', fixedrange: false, range: null };
  const pad = maxAbs(osc.slope) * 1.05 || 1;
  return { key: 'variance_osc:slope', fixedrange: false, range: [-pad, pad] };
}

registerTool({
  type: 'variance_osc',
  icon: 'variance',
  label: 'Осциллятор дисперсии',
  panelId: 'tool-panel-variance_osc',
  onSelected: onVarianceOscSelected,
  buildSubpanelTraces: buildVarianceOscSubpanelTraces,
  subpanelYAxisPolicy: varianceOscYAxisPolicy,
  // no buildMainTraces/buildOriginShape/onOriginClick/resetOrigin — this
  // tool never draws on the main chart and has no origin concept.
});
