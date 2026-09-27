import { S } from './state.js';
import { api, setBusy, setIdle } from './api.js';
import { renderChart } from './chart.js';
import { registerTool, activateTool } from './tools.js';
import { hexToRgba } from './color_utils.js';

// ── Анализ tab: spectrogram analyzer ────────────────────────────────────
// See sma/core/analysis/spectrogram.py. Settings live here; recomputes
// reactively on every parameter change (depth/nperseg/overlap/fmin/fmax),
// debounced — no manual "Рассчитать" button any more (project feedback
// 2026-08-20: "можно тоже расчитывать автоматом... вроде она не даёт
// значимой нагрузки" — an earlier round had deliberately kept a manual step
// here citing cost, revised once the user confirmed it isn't actually
// heavy). onSelected (tools.js) triggers an initial calc the first time
// this tool is picked and no data exists yet, same pattern
// variance_oscillator.js already uses. Purely visual tweaks (contrast
// percentile, log-Y) still recompute client-side from the already-fetched
// data — no new request, same pattern as band_lambda's zone levels
// (sma/ui/chart.js:weightedQuantile). Registered into the shared analyzer
// registry at the bottom of this file — chart.js finds this analyzer's
// trace/shape builders generically through that registry, not by importing
// this module.

function redraw() {
  renderChart({ preserveRange: true });
}

function readSpectrogramSettings() {
  return {
    depthBars:   +document.getElementById('spectrogram-depth').value,
    nperseg:     +document.getElementById('spectrogram-nperseg').value,
    overlapPct:  +document.getElementById('spectrogram-overlap').value,
    fmin:        +document.getElementById('spectrogram-fmin').value,
    fmax:        +document.getElementById('spectrogram-fmax').value,
    logY:        document.getElementById('spectrogram-logy').checked,
    contrastPct: +document.getElementById('spectrogram-contrast').value,
  };
}

function applySpectrogramSettings(p) {
  if (!p) return;
  document.getElementById('spectrogram-depth').value = p.depthBars;
  document.getElementById('spectrogram-nperseg').value = p.nperseg;
  document.getElementById('spectrogram-overlap').value = p.overlapPct;
  document.getElementById('spectrogram-overlap-label').textContent = `${p.overlapPct}%`;
  document.getElementById('spectrogram-fmin').value = p.fmin;
  document.getElementById('spectrogram-fmax').value = p.fmax;
  document.getElementById('spectrogram-contrast').value = p.contrastPct;
  document.getElementById('spectrogram-contrast-label').textContent = `${p.contrastPct}%`;
  document.getElementById('spectrogram-logy').checked = !!p.logY;
  S.spectrogramSettings = p;
}

// Last-used settings for this (instrument, interval) — see db.py
// analysis_settings table docstring. Called on ticker/interval switch
// (app.js:loadCandles); falls back to the hardcoded form defaults (index.html
// value= attributes / state.js) when nothing is saved yet.
export async function loadSpectrogramDefaults() {
  if (!S.instrumentId) return;
  try {
    const res = await api(
      'GET',
      `/series/analysis-settings?instrument_id=${S.instrumentId}&interval=${S.interval}&analyzer_type=spectrogram`
    );
    if (res.params) applySpectrogramSettings(res.params);
  } catch (_) { /* non-fatal — keeps current form values */ }
}

// Client-saved (the backend never sees contrastPct/logY — pure display
// params computed client-side, see module docstring) — fire-and-forget,
// mirrors band_lambda's applyDisplaySettings save pattern. Debounced: the
// contrast slider fires this on every `oninput` tick while dragging.
let _spectrogramSaveTimer = null;

function saveSpectrogramSettings() {
  if (!S.instrumentId) return;
  clearTimeout(_spectrogramSaveTimer);
  _spectrogramSaveTimer = setTimeout(() => {
    api('POST', '/series/analysis-settings', {
      instrument_id: S.instrumentId, interval: S.interval,
      analyzer_type: 'spectrogram', params: S.spectrogramSettings,
    }).catch(() => {});
  }, 500);
}

export async function calculateSpectrogram() {
  if (!S.ticker) return; // no toast here any more — this can now fire from a reactive settings change, not just an explicit user click
  const settings = readSpectrogramSettings();
  S.spectrogramSettings = settings;

  setBusy('Расчёт спектрограммы…');
  try {
    const res = await api('POST', '/series/spectrogram', {
      ticker: S.ticker, data_source: S.dataSource, interval: S.interval,
      depth_bars: settings.depthBars, nperseg: settings.nperseg,
      overlap_pct: settings.overlapPct, fmin: settings.fmin, fmax: settings.fmax,
    });
    S.spectrogramData = res;
    activateTool('spectrogram'); // freshly calculated result should be visible without a separate toolbar click

    const m = res.meta;
    document.getElementById('spectrogram-meta').textContent =
      `N=${m.n_bars.toLocaleString('ru-RU')} баров · окно=${m.nperseg} · перекрытие=${m.noverlap} · ` +
      `Δf=${m.freq_resolution.toFixed(4)} цикл/бар · шаг=${m.time_step_bars} баров`;

    redraw();
    saveSpectrogramSettings();
    setIdle('Спектрограмма рассчитана');
  } catch (e) {
    setIdle(e.message, false);
  }
}

// Debounced wrapper for the "expensive" (backend STFT) parameters —
// depth/nperseg/overlap/fmin/fmax — bound to their own oninput/onchange in
// index.html. contrast/logY stay on the separate, undebounced
// applySpectrogramContrast (pure client-side, no request to coalesce).
let _spectrogramCalcTimer = null;

export function onSpectrogramSettingsChange() {
  clearTimeout(_spectrogramCalcTimer);
  _spectrogramCalcTimer = setTimeout(calculateSpectrogram, 500);
}

// Called when this tool becomes the active oscillator (tools.js:selectTool/
// activateTool's onSelected hook, same pattern as variance_oscillator.js) —
// computes only if there's no data yet; re-selecting an already-computed
// spectrogram is free.
function onSpectrogramSelected() {
  if (!S.spectrogramData) calculateSpectrogram();
}

// Percentile clip for the heatmap color range + log-Y toggle — both purely
// visual, recomputed from S.spectrogramData without touching the backend.
export function applySpectrogramContrast() {
  if (!S.spectrogramData) return;
  S.spectrogramSettings.contrastPct = +document.getElementById('spectrogram-contrast').value;
  S.spectrogramSettings.logY = document.getElementById('spectrogram-logy').checked;
  redraw();
  saveSpectrogramSettings();
}

// ── registry entries: subpanel traces/shapes (moved out of chart.js so
// chart.js never needs to import this module — see tools.js) ───────

// Percentile clip for the spectrogram color range — "contrastPct=5" means
// the color scale spans [5th, 95th] percentile of the visible dB values,
// same idea as the prototype's fixed 5/95 clip, just user-adjustable (see
// project feedback 2026-08-08). Pure client-side, recomputed from the
// already-fetched matrix — no backend round trip for a cosmetic slider.
function percentile(sorted, p) {
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

// Flattening + sorting the full sxx_db matrix isn't free, and this trace
// builder runs on EVERY renderChart while the spectrogram is the active
// subpanel — including renders triggered by something else entirely (e.g.
// another analyzer's own settings, now that "show on chart"/"show
// oscillator" are independent per analyzer). Cached by (matrix identity,
// contrastPct) so it only recomputes when one of those two actually
// changed, not on every unrelated re-render (project feedback 2026-08-18).
let _contrastRangeCache = { sxxDb: null, contrastPct: null, range: null };

function computeContrastRange(sxxDb, contrastPct) {
  const cache = _contrastRangeCache;
  if (cache.sxxDb === sxxDb && cache.contrastPct === contrastPct) return cache.range;
  const flat = [].concat(...sxxDb).filter(Number.isFinite).sort((a, b) => a - b);
  const p = Math.min(Math.max(contrastPct, 0), 49);
  const range = flat.length ? [percentile(flat, p), percentile(flat, 100 - p)] : [0, 1];
  _contrastRangeCache = { sxxDb, contrastPct, range };
  return range;
}

function buildSpectrogramSubpanelTraces() {
  if (!S.spectrogramData) return [];
  const d = S.spectrogramData;
  const [zmin, zmax] = computeContrastRange(d.sxx_db, S.spectrogramSettings.contrastPct);
  return [{
    type: 'heatmap', name: 'Спектрограмма Δratio',
    x: d.times, y: d.freqs, z: d.sxx_db,
    yaxis: 'y2',
    colorscale: S.colorProfile.spectrogram_colorscale, zmin, zmax, // "Спектрограмма: цветовая схема" role — settings.js (a Plotly colorscale NAME, not a hex)
    colorbar: { thickness: 10, len: 0.18, y: 0.09, title: { text: 'дБ', font: { size: 9 } } },
    hoverongaps: false,
    hovertemplate: 'f=%{y:.4f} цикл/бар<br>%{x}<br>%{z:.1f} дБ<extra></extra>',
    showlegend: false,
  }];
}

// Horizontal dotted reference lines at the filter-bank C0..C5 band edges
// (see sma/core/analysis/spectrogram.py:filter_bank_cutoffs) — only within
// the currently visible frequency range.
function buildSpectrogramCutoffShapes() {
  if (!S.spectrogramData) return [];
  const freqs = S.spectrogramData.freqs;
  if (!freqs.length) return [];
  const fLo = freqs[0], fHi = freqs[freqs.length - 1];
  return S.spectrogramData.cutoffs
    .filter(c => c.freq >= fLo && c.freq <= fHi)
    .map(c => ({
      type: 'line', xref: 'paper', yref: 'y2',
      x0: 0, x1: 1, y0: c.freq, y1: c.freq,
      line: { color: hexToRgba(S.colorProfile.spectrogram_cutoff_line, 0.75), width: 1, dash: 'dot' },
    }));
}

// yaxis2 policy (project feedback 2026-08-20, §3.3 of docs/plans/
// frontend_improvements_plan.md): fully fixed — "незачем" to zoom/pan a
// frequency axis whose range is already exactly what fmin/fmax on the
// request asked for. Range mirrors the backend response exactly (not left
// to Plotly's own autorange, which could pad it slightly) — recomputed
// fresh every render since fixedrange:true means there's no interactive
// state to preserve, so a fresh spectrogram calc (different fmin/fmax) just
// takes effect immediately.
function spectrogramYAxisPolicy() {
  const freqs = S.spectrogramData?.freqs;
  if (!freqs?.length) return { key: 'spectrogram', fixedrange: true, range: null };
  let lo = freqs[0], hi = freqs[freqs.length - 1];
  if (S.spectrogramSettings.logY) { lo = Math.log10(Math.max(lo, 1e-9)); hi = Math.log10(hi); }
  return { key: 'spectrogram', fixedrange: true, range: [lo, hi] };
}

registerTool({
  type: 'spectrogram',
  icon: 'spectrogram',
  label: 'Спектрограмма Δratio',
  panelId: 'tool-panel-spectrogram',
  onSelected: onSpectrogramSelected,
  buildSubpanelTraces: buildSpectrogramSubpanelTraces,
  buildSubpanelShapes: buildSpectrogramCutoffShapes,
  subpanelYAxisPolicy: spectrogramYAxisPolicy,
  // no buildMainTraces/buildOriginShape/onOriginClick/resetOrigin —
  // spectrogram never draws on the main chart and has no origin concept.
});
