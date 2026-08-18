import { S } from './state.js';
import { api, setStatus } from './api.js';
import { renderChart } from './chart.js';
import { registerTool } from './tools.js';

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

const PALETTE = ['#d29922', '#f0883e', '#a5d6ff', '#7ee787', '#ffa198', '#d2a8ff'];
let _nextSeriesId = 1;
const _fetchTokens = new Map(); // seriesId -> latest request token

function redraw() {
  renderChart({ preserveRange: true });
}

function colorFor(index) {
  return PALETTE[index % PALETTE.length];
}

// ── series list UI ───────────────────────────────────────────────────────

function renderZigzagList() {
  const container = document.getElementById('zz-list');
  if (!container) return;
  container.innerHTML = S.zigzagToolSettings.series.map((s, i) => `
    <div class="field-row" data-series-id="${s.id}" style="gap:4px">
      <span class="icon" style="width:10px;height:10px;border-radius:50%;background:${colorFor(i)};flex-shrink:0"></span>
      <input type="number" step="0.1" min="0.1" value="${s.tQuery}" style="width:56px" data-zz-field="tQuery" title="T, %">
      <span class="muted-val">%</span>
      <input type="number" step="1" min="0" value="${s.minBars}" style="width:52px" data-zz-field="minBars" title="min_bars">
      <button type="button" class="icon-btn danger" title="Убрать" data-zz-remove style="width:24px">
        <svg class="icon"><use href="#icon-close"/></svg>
      </button>
    </div>
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
    row.querySelector('[data-zz-remove]').addEventListener('click', () => {
      S.zigzagToolSettings.series = S.zigzagToolSettings.series.filter(x => x.id !== id);
      delete S.zigzagToolData[id];
      _fetchTokens.delete(id);
      renderZigzagList();
      redraw();
      saveZigzagToolSettings();
    });
  });
}

export function addZigzagSeries() {
  const s = { id: _nextSeriesId++, tQuery: 4, minBars: 5 };
  S.zigzagToolSettings.series.push(s);
  renderZigzagList();
  fetchSeries(s.id);
  saveZigzagToolSettings();
}

// ── fetch (debounced per-series, stale-response-safe) ────────────────────

const _debounceTimers = new Map();

function scheduleFetch(seriesId) {
  clearTimeout(_debounceTimers.get(seriesId));
  _debounceTimers.set(seriesId, setTimeout(() => fetchSeries(seriesId), 350));
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
    redraw();
  } catch (e) {
    if (_fetchTokens.get(seriesId) !== token) return;
    setStatus(e.message, 'err');
  }
}

// ── settings persist ──────────────────────────────────────────────────────

function readZigzagToolSettings() {
  return {
    showOnChart: document.getElementById('zz-show-chart').checked,
    series: S.zigzagToolSettings.series.map(s => ({ ...s })), // list UI is the source of truth
  };
}

function applyZigzagToolSettings(p) {
  if (!p) return;
  document.getElementById('zz-show-chart').checked = !!p.showOnChart;
  S.zigzagToolSettings = { showOnChart: !!p.showOnChart, series: (p.series || []).map(s => ({ ...s })) };
  const maxId = S.zigzagToolSettings.series.reduce((m, s) => Math.max(m, s.id), 0);
  _nextSeriesId = maxId + 1;
  renderZigzagList();
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
  S.zigzagToolSettings.showOnChart = document.getElementById('zz-show-chart')?.checked ?? S.zigzagToolSettings.showOnChart;
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    api('POST', '/series/analysis-settings', {
      instrument_id: S.instrumentId, interval: S.interval,
      analyzer_type: 'zigzag_tool', params: S.zigzagToolSettings,
    }).catch(() => {});
  }, 500);
}

export function onZigzagToolShowChartChange() {
  S.zigzagToolSettings.showOnChart = document.getElementById('zz-show-chart').checked;
  redraw();
  saveZigzagToolSettings();
}

// Called once after candles (re)load — re-fetches every configured series
// (pivots themselves aren't persisted, only the T%/min_bars settings that
// produce them — cheap to just re-derive).
export function initZigzagToolForTicker() {
  S.zigzagToolData = {};
  renderZigzagList();
  S.zigzagToolSettings.series.forEach(s => fetchSeries(s.id));
}

// ── registry entry: main-chart trace per series ─────────────────────────
function buildZigzagToolMainTraces() {
  if (!S.zigzagToolSettings.showOnChart) return [];
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
