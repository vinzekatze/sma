import { S } from './state.js';
import { api } from './api.js';
import { renderChart } from './chart.js';
import { registerTool } from './tools.js';

// ── moving_averages analyzer (Основной график tab) ──────────────────────
// N configurable moving averages, own type (SMA/EMA/WMA) + period each —
// purely client-side (S.candles already loaded), no origin concept (each
// line is a whole-history rolling series, not window-relative like
// trend_ruler). Recomputed on every settings change — even a few hundred-
// period MA over a few thousand candles is a trivial amount of JS work,
// no debounce needed (unlike zigzag_tool's series, which need a real
// network round trip per edit).

const PALETTE = ['#f0883e', '#a5d6ff', '#d2a8ff', '#7ee787', '#ffa198', '#79c0ff'];
let _nextSeriesId = 1;

function redraw() {
  renderChart({ preserveRange: true });
}

function colorFor(index) {
  return PALETTE[index % PALETTE.length];
}

// ── math (causal — null before `period` closes are available) ──────────

function sma(closes, period) {
  const out = new Array(closes.length).fill(null);
  let sum = 0;
  for (let i = 0; i < closes.length; i++) {
    sum += closes[i];
    if (i >= period) sum -= closes[i - period];
    if (i >= period - 1) out[i] = sum / period;
  }
  return out;
}

function wma(closes, period) {
  const out = new Array(closes.length).fill(null);
  const denom = period * (period + 1) / 2;
  for (let i = period - 1; i < closes.length; i++) {
    let acc = 0;
    for (let j = 0; j < period; j++) acc += closes[i - period + 1 + j] * (j + 1);
    out[i] = acc / denom;
  }
  return out;
}

function ema(closes, period) {
  const out = new Array(closes.length).fill(null);
  const alpha = 2 / (period + 1);
  let prev = null;
  for (let i = 0; i < closes.length; i++) {
    if (i < period - 1) continue;
    if (i === period - 1) {
      let sum = 0;
      for (let j = 0; j <= i; j++) sum += closes[j];
      prev = sum / period; // seed = SMA of the first `period` points
    } else {
      prev = closes[i] * alpha + prev * (1 - alpha);
    }
    out[i] = prev;
  }
  return out;
}

function computeSeries(closes, type, period) {
  if (type === 'ema') return ema(closes, period);
  if (type === 'wma') return wma(closes, period);
  return sma(closes, period);
}

// ── series list UI ───────────────────────────────────────────────────────

function renderMaList() {
  const container = document.getElementById('ma-list');
  if (!container) return;
  container.innerHTML = S.maSettings.series.map((s, i) => `
    <div class="field-row" data-series-id="${s.id}" style="gap:4px">
      <span class="icon" style="width:10px;height:10px;border-radius:50%;background:${colorFor(i)};flex-shrink:0"></span>
      <select style="flex:0 0 64px" data-ma-field="type">
        <option value="sma" ${s.type === 'sma' ? 'selected' : ''}>SMA</option>
        <option value="ema" ${s.type === 'ema' ? 'selected' : ''}>EMA</option>
        <option value="wma" ${s.type === 'wma' ? 'selected' : ''}>WMA</option>
      </select>
      <input type="number" min="2" max="2000" step="1" value="${s.period}" style="flex:1" data-ma-field="period" title="Период (баров)">
      <button type="button" class="icon-btn danger" title="Убрать" data-ma-remove style="width:24px">
        <svg class="icon"><use href="#icon-close"/></svg>
      </button>
    </div>
  `).join('');

  container.querySelectorAll('[data-series-id]').forEach(row => {
    const id = +row.dataset.seriesId;
    row.querySelectorAll('[data-ma-field]').forEach(inp => {
      inp.addEventListener('input', () => {
        const s = S.maSettings.series.find(x => x.id === id);
        if (!s) return;
        s[inp.dataset.maField] = inp.dataset.maField === 'period' ? +inp.value : inp.value;
        onMaSettingsChange();
      });
    });
    row.querySelector('[data-ma-remove]').addEventListener('click', () => {
      S.maSettings.series = S.maSettings.series.filter(x => x.id !== id);
      renderMaList();
      onMaSettingsChange();
    });
  });
}

export function addMovingAverage() {
  S.maSettings.series.push({ id: _nextSeriesId++, type: 'sma', period: 20 });
  renderMaList();
  onMaSettingsChange();
}

// ── settings persist + recompute ─────────────────────────────────────────

export function onMaSettingsChange() {
  S.maSettings.showOnChart = document.getElementById('ma-show-chart').checked;
  recompute();
  redraw();
  saveMaSettings();
}

function recompute() {
  if (!S.candles.length) { S.maData = {}; return; }
  const closes = S.candles.map(c => c.close);
  const times = S.candles.map(c => c.begin);
  const out = {};
  for (const s of S.maSettings.series) {
    if (s.period >= 2 && s.period <= closes.length) {
      out[s.id] = { times, values: computeSeries(closes, s.type, s.period) };
    }
  }
  S.maData = out;
}

function applyMaSettings(p) {
  if (!p) return;
  document.getElementById('ma-show-chart').checked = !!p.showOnChart;
  S.maSettings = { showOnChart: !!p.showOnChart, series: (p.series || []).map(s => ({ ...s })) };
  const maxId = S.maSettings.series.reduce((m, s) => Math.max(m, s.id), 0);
  _nextSeriesId = maxId + 1;
  renderMaList();
}

export async function loadMaDefaults() {
  if (!S.instrumentId) return;
  try {
    const res = await api(
      'GET',
      `/series/analysis-settings?instrument_id=${S.instrumentId}&interval=${S.interval}&analyzer_type=moving_averages`
    );
    if (res.params) applyMaSettings(res.params);
  } catch (_) { /* non-fatal */ }
}

let _saveTimer = null;

function saveMaSettings() {
  if (!S.instrumentId) return;
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    api('POST', '/series/analysis-settings', {
      instrument_id: S.instrumentId, interval: S.interval,
      analyzer_type: 'moving_averages', params: S.maSettings,
    }).catch(() => {});
  }, 500);
}

// Called once after candles (re)load — recomputes immediately (cheap, no
// request) so lines are ready the instant showOnChart is on.
export function initMaForTicker() {
  renderMaList();
  recompute();
}

// ── registry entry: one line trace per configured MA ─────────────────────
function buildMaMainTraces() {
  if (!S.maSettings.showOnChart) return [];
  return S.maSettings.series.map((s, i) => {
    const d = S.maData[s.id];
    if (!d) return null;
    return {
      type: 'scatter', mode: 'lines', name: `${s.type.toUpperCase()}(${s.period})`,
      x: d.times, y: d.values,
      line: { color: colorFor(i), width: 1.5 },
      connectgaps: false,
    };
  }).filter(Boolean);
}

registerTool({
  type: 'moving_averages',
  icon: 'ma',
  label: 'Скользящие средние',
  panelId: 'tool-panel-moving_averages',
  buildMainTraces: buildMaMainTraces,
  // no buildOriginShape/onOriginClick/resetOrigin — no origin concept.
});
