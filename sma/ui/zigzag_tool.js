import { S } from './state.js';
import { api, setStatus } from './api.js';
import { renderChart } from './chart.js';
import { registerTool, isToolObjectVisible } from './tools.js';
import { getToolShowOnChart, saveToolShowOnChart } from './local_prefs.js';

// ── zigzag_tool analyzer (Основной график tab) ───────────────────────────
// A freestanding set of zigzags, independent of the Прогноз tab's own
// calibration-bound zigzag (S.zigzagPivots, sma/ui/forecast.js:loadZigzag —
// that one is tied to the currently selected T on the Прогноз tab; this one
// lets you compare several arbitrary T%/min_bars zigzags side by side while
// working the main chart). Reuses the SAME backend endpoint —
// GET /forecasts/zigzag?instrument_id&interval&t_query&min_bars —
// unchanged: despite the "/forecasts/" path, sma/api/routes/forecasts.py's
// build_zigzag call there is entirely generic (t_query/min_bars only, no
// forecast/calibration state), confirmed during phase-3 planning. No
// backend work needed for this tool at all.
//
// Each configured series needs an actual network round trip (unlike
// trend_ruler's math, this isn't ported to JS — build_zigzag is a simple
// O(n) scan, but re-implementing it client-side wasn't worth it for this
// pass; see project memory if that changes) — so edits are debounced, and a
// per-series request token discards stale responses if you edit T/min_bars
// again before the previous fetch lands.

let _nextSeriesId = 1;
const _fetchTokens = new Map(); // seriesId -> latest request token

function redraw() {
  renderChart({ preserveRange: true });
}

// Palette is the "Палитра Zig-Zag" role of the app-wide color profile
// (S.colorProfile.zigzag_tool_palette, docs/plans/frontend_improvements_plan.md
// §1.1a — settings.js/"Настройки приложения"), read live (not cached) so a
// profile save recolors existing series immediately.
function colorFor(index) {
  const palette = S.colorProfile.zigzag_tool_palette;
  if (!palette?.length) return '#d29922';
  return palette[index % palette.length];
}

// ── series list UI ───────────────────────────────────────────────────────

// Descriptive (full-history, not causally restricted) count/duration/
// amplitude per direction for one series — "живой" quick-look next to
// whatever T you're currently poking at, independent of the Прогноз tab's
// calibration-bound T-selector (which only ever shows already-calibrated
// T's — not useful for free exploration). Backend: sma/core/forecast/
// pivot_time_band.py:zigzag_direction_stats, already returned by GET
// /forecasts/zigzag alongside pivots — this tool just wasn't reading that
// field yet (2026-08-26 conversation).
function statsHtml(seriesId) {
  const series = S.zigzagToolSettings.series.find(x => x.id === seriesId);
  if (!series?.showStats) return ''; // hidden by default — project feedback 2026-08-26: "иначе неудобно"
  const s = S.zigzagToolStats[seriesId];
  if (!s || (!s.up?.n && !s.down?.n)) return '<span class="muted-val">—</span>';
  const row = (dir, label) => {
    const d = s[dir];
    if (!d || !d.n) return `<div class="field-row"><label>${label}</label><span class="muted-val">нет данных</span></div>`;
    return `
      <div class="field-row"><label>${label}</label><span class="muted-val">n=${d.n}</span></div>
      <div class="field-row"><label style="padding-left:8px">длительность, бар</label>
        <span class="muted-val">${d.duration_bars.median.toFixed(0)} медиана (±${d.duration_bars.std.toFixed(0)})</span></div>
      <div class="field-row"><label style="padding-left:8px">амплитуда, %</label>
        <span class="muted-val">${d.amplitude_pct.median.toFixed(1)} медиана (±${d.amplitude_pct.std.toFixed(1)})</span></div>
    `;
  };
  return `
    <div class="field-row"><label>Всего пивотов</label><span class="muted-val">${s.n_total}</span></div>
    ${row('up', '▲ Up-плечи')}
    ${row('down', '▼ Down-плечи')}
  `;
}

function renderZigzagList() {
  const container = document.getElementById('zz-list');
  if (!container) return;
  container.innerHTML = S.zigzagToolSettings.series.map((s, i) => `
    <div class="field-row" data-series-id="${s.id}" style="gap:4px">
      <span class="icon" style="width:10px;height:10px;border-radius:50%;background:${colorFor(i)};flex-shrink:0"></span>
      <button type="button" class="hist-eye-btn${s.showStats ? ' active' : ''}" title="Показать статистику (count/длительность/амплитуда)" data-zz-stats-toggle style="font-size:20px;line-height:1">${s.showStats ? '▾' : '▸'}</button>
      <input type="number" step="0.1" min="0.1" value="${s.tQuery}" style="width:56px" data-zz-field="tQuery" title="T, %">
      <span class="muted-val">%</span>
      <input type="number" step="1" min="0" value="${s.minBars}" style="width:52px" data-zz-field="minBars" title="min_bars">
      <button type="button" class="icon-btn danger" title="Убрать" data-zz-remove style="width:24px">
        <svg class="icon"><use href="#icon-close"/></svg>
      </button>
    </div>
    <div class="zz-stats" id="zz-stats-${s.id}" style="font-size:11px;margin:2px 0 6px 14px">${statsHtml(s.id)}</div>
  `).join('');

  container.querySelectorAll('[data-series-id]').forEach(row => {
    const id = +row.dataset.seriesId;
    row.querySelectorAll('input[data-zz-field]').forEach(inp => {
      inp.addEventListener('input', () => {
        const s = S.zigzagToolSettings.series.find(x => x.id === id);
        if (!s) return;
        s[inp.dataset.zzField] = +inp.value;
        scheduleFetch(id);
        saveZigzagToolSettings();
      });
    });
    row.querySelector('[data-zz-stats-toggle]').addEventListener('click', (e) => {
      const s = S.zigzagToolSettings.series.find(x => x.id === id);
      if (!s) return;
      s.showStats = !s.showStats;
      e.currentTarget.classList.toggle('active', s.showStats);
      e.currentTarget.textContent = s.showStats ? '▾' : '▸';
      const statsEl = document.getElementById(`zz-stats-${id}`);
      if (statsEl) statsEl.innerHTML = statsHtml(id);
      saveZigzagToolSettings();
    });
    row.querySelector('[data-zz-remove]').addEventListener('click', () => {
      S.zigzagToolSettings.series = S.zigzagToolSettings.series.filter(x => x.id !== id);
      delete S.zigzagToolData[id];
      delete S.zigzagToolStats[id];
      _fetchTokens.delete(id);
      renderZigzagList();
      redraw();
      saveZigzagToolSettings();
    });
  });
}

export function addZigzagSeries() {
  const s = { id: _nextSeriesId++, tQuery: 4, minBars: 5, showStats: false };
  S.zigzagToolSettings.series.push(s);
  renderZigzagList();
  fetchSeries(s.id);
  saveZigzagToolSettings();
}

// ── fetch (debounced per-series, stale-response-safe) ────────────────────

const _debounceTimers = new Map();

function scheduleFetch(seriesId) {
  clearTimeout(_debounceTimers.get(seriesId));
  _debounceTimers.set(seriesId, setTimeout(() => fetchSeries(seriesId), 500)); // same debounce every other tool uses (variance_oscillator.js/trend_ruler.js/risk_corridor.js/forecast.js) — no reason for this one to differ
}

async function fetchSeries(seriesId) {
  const s = S.zigzagToolSettings.series.find(x => x.id === seriesId);
  if (!s || !S.instrumentId) return;
  const token = (_fetchTokens.get(seriesId) ?? 0) + 1;
  _fetchTokens.set(seriesId, token);
  try {
    const res = await api(
      'GET',
      `/forecasts/zigzag?instrument_id=${S.instrumentId}&interval=${S.interval}` +
      `&t_query=${s.tQuery / 100}&min_bars=${s.minBars}`
    );
    if (_fetchTokens.get(seriesId) !== token) return; // superseded — discard
    S.zigzagToolData[seriesId] = res.pivots || [];
    S.zigzagToolStats[seriesId] = res.stats ?? null;
    // Targeted update (not renderZigzagList()) — that would rebuild the T%/
    // min_bars <input>s too and steal focus/cursor position while the user
    // is still typing (this fires 500ms after every keystroke).
    const statsEl = document.getElementById(`zz-stats-${seriesId}`);
    if (statsEl) statsEl.innerHTML = statsHtml(seriesId);
    redraw();
  } catch (e) {
    if (_fetchTokens.get(seriesId) !== token) return;
    setStatus(e.message, 'err');
  }
}

// ── settings persist ──────────────────────────────────────────────────────

function applyZigzagToolSettings(p) {
  if (!p) return;
  S.zigzagToolSettings = { showOnChart: !!p.showOnChart, series: (p.series || []).map(s => ({ ...s })) };
  const maxId = S.zigzagToolSettings.series.reduce((m, s) => Math.max(m, s.id), 0);
  _nextSeriesId = maxId + 1;
  renderZigzagList();
  updateShowChartButton();
}

export async function loadZigzagToolDefaults() {
  if (!S.instrumentId) return;
  try {
    const res = await api(
      'GET',
      `/series/analysis-settings?instrument_id=${S.instrumentId}&interval=${S.interval}&analyzer_type=zigzag_tool`
    );
    if (res.params) applyZigzagToolSettings(res.params);
  } catch (_) { /* non-fatal */ }
}

let _saveTimer = null;

function saveZigzagToolSettings() {
  if (!S.instrumentId) return;
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    api('POST', '/series/analysis-settings', {
      instrument_id: S.instrumentId, interval: S.interval,
      analyzer_type: 'zigzag_tool', params: S.zigzagToolSettings,
    }).catch(() => {});
  }, 500);
}

// Eye-icon toggle in the panel header (replaces the old "Показывать на
// графике" checkbox at the bottom of the panel, project feedback
// 2026-08-20, §2.6 of docs/plans/frontend_improvements_plan.md) — now means
// "закреплено": stays visible even while this tool ISN'T the active one
// (see isZigzagToolVisible/buildZigzagToolMainTraces below).
export function onZigzagToolShowChartChange() {
  S.zigzagToolSettings.showOnChart = !S.zigzagToolSettings.showOnChart;
  updateShowChartButton();
  redraw();
  saveZigzagToolSettings();
  saveToolShowOnChart('zigzag_tool', S.zigzagToolSettings.showOnChart); // global, survives a reload/ticker switch — project feedback 2026-08-20, see local_prefs.js
}

function updateShowChartButton() {
  document.getElementById('zz-show-chart-btn')
    ?.classList.toggle('active', !!S.zigzagToolSettings.showOnChart);
}

// Visibility policy (project feedback 2026-08-20, §2.6 of
// docs/plans/frontend_improvements_plan.md): pinned (showOnChart) OR this
// tool is the one currently active in the toolbar.
function isZigzagToolVisible() {
  return isToolObjectVisible('zigzag_tool', S.zigzagToolSettings.showOnChart);
}

// Called once after candles (re)load — re-fetches every configured series
// (pivots themselves aren't persisted, only the T%/min_bars settings that
// produce them — cheap to just re-derive). showOnChart is then overridden
// by the global localStorage flag (project feedback 2026-08-20 — see
// local_prefs.js and trend_ruler.js:initTrendRulerForTicker for the
// identical reasoning), overriding whatever this ticker's own
// analysis_settings row had.
export function initZigzagToolForTicker() {
  S.zigzagToolData = {};
  S.zigzagToolStats = {};
  renderZigzagList();
  S.zigzagToolSettings.series.forEach(s => fetchSeries(s.id));
  S.zigzagToolSettings.showOnChart = getToolShowOnChart('zigzag_tool', S.zigzagToolSettings.showOnChart);
  updateShowChartButton();
}

// ── registry entry: main-chart trace per series ─────────────────────────
function buildZigzagToolMainTraces() {
  if (!isZigzagToolVisible()) return [];
  return S.zigzagToolSettings.series.map((s, i) => {
    const pivots = S.zigzagToolData[s.id];
    if (!pivots?.length) return null;
    return {
      type: 'scatter', name: `Zig-Zag T=${s.tQuery}%`,
      x: pivots.map(p => p.extreme_date), y: pivots.map(p => p.price),
      mode: 'lines+markers',
      line: { color: colorFor(i), width: 1 },
      marker: { size: 4, color: colorFor(i) },
      hovertemplate: '%{y:.4f}<extra></extra>',
    };
  }).filter(Boolean);
}

registerTool({
  type: 'zigzag_tool',
  icon: 'zigzag',
  label: 'Zig-Zag',
  panelId: 'tool-panel-zigzag_tool',
  buildMainTraces: buildZigzagToolMainTraces,
  // no buildOriginShape/onOriginClick/resetOrigin — no origin concept, each
  // series is a whole-history zigzag, not window-relative.
});
