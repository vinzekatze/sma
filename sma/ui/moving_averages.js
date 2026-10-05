import { S } from './state.js';
import { api, setStatus } from './api.js';
import { renderChart, displayIndexRange } from './chart.js';
import { registerTool, isToolObjectVisible } from './tools.js';
import { getToolShowOnChart, saveToolShowOnChart } from './local_prefs.js';

// ── moving_averages analyzer (Основной график tab) ──────────────────────
// N configurable moving averages, own type (SMA/EMA/WMA) + period each —
// computed by the backend over the full history, no origin concept (each
// line is a whole-history rolling series, not window-relative like
// trend_ruler). Recomputed on every settings change — even a few hundred-
// period MA over a few thousand candles is a trivial amount of JS work,
// no debounce needed (unlike zigzag_tool's series, which need a real
// network round trip per edit).

let _nextSeriesId = 1;

function redraw() {
  renderChart({ preserveRange: true });
}

// Palette is the "Палитра MA" role of the app-wide color profile
// (S.colorProfile.ma_palette, docs/plans/frontend_improvements_plan.md
// §1.1a — settings.js/"Настройки приложения"), read live (not cached) so a
// profile save recolors existing series immediately. Falls back to a
// literal default only if the profile is somehow empty (shouldn't happen —
// state.js/db.py both seed a non-empty default).
function colorFor(index) {
  const palette = S.colorProfile.ma_palette;
  if (!palette?.length) return '#f0883e';
  return palette[index % palette.length];
}

// ── math (causal — null before `period` closes are available) ──────────

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

export async function onMaSettingsChange() {
  await recompute();
  redraw();
  saveMaSettings();
}

// Eye-icon toggle in the panel header (replaces the old "Показывать на
// графике" checkbox at the bottom of the panel, project feedback
// 2026-08-20, §2.6 of docs/plans/frontend_improvements_plan.md) — now means
// "закреплено": stays visible even while this tool ISN'T the active one
// (see isMaVisible/buildMaMainTraces below).
export function toggleMaShowChart() {
  S.maSettings.showOnChart = !S.maSettings.showOnChart;
  updateShowChartButton();
  redraw();
  saveMaSettings();
  saveToolShowOnChart('moving_averages', S.maSettings.showOnChart); // global, survives a reload/ticker switch — project feedback 2026-08-20, see local_prefs.js
}

function updateShowChartButton() {
  document.getElementById('ma-show-chart-btn')
    ?.classList.toggle('active', !!S.maSettings.showOnChart);
}

// Visibility policy (project feedback 2026-08-20, §2.6 of
// docs/plans/frontend_improvements_plan.md): pinned (showOnChart) OR this
// tool is the one currently active in the toolbar.
function isMaVisible() {
  return isToolObjectVisible('moving_averages', S.maSettings.showOnChart);
}

// Values come from the BACKEND over the full stored history
// (POST /series/moving-averages → sma/core/analysis/moving_averages.py),
// not from the loaded candle window. Sequenced: only the newest response is
// applied. Switching ticker/interval drops the previous lines at once.
let _maSeq = 0;
let _maKey = null;

async function recompute() {
  const seq = ++_maSeq;
  const key = `${S.dataSource}|${S.ticker}|${S.interval}`;
  if (key !== _maKey) { S.maData = {}; _maKey = key; }
  if (!S.instrumentId) { S.maData = {}; return; }
  const series = S.maSettings.series
    .filter(s => s.period >= 2)
    .map(s => ({ id: s.id, type: s.type, period: s.period }));
  try {
    const res = await api('POST', '/series/moving-averages', {
      ticker: S.ticker, data_source: S.dataSource, interval: S.interval, series,
    });
    if (seq === _maSeq) S.maData = res;
  } catch (e) {
    if (seq === _maSeq) setStatus(e.message, 'err');
  }
}

function applyMaSettings(p) {
  if (!p) return;
  S.maSettings = { showOnChart: !!p.showOnChart, series: (p.series || []).map(s => ({ ...s })) };
  const maxId = S.maSettings.series.reduce((m, s) => Math.max(m, s.id), 0);
  _nextSeriesId = maxId + 1;
  renderMaList();
  updateShowChartButton();
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
// request) so lines are ready the instant showOnChart is on. showOnChart is
// then overridden by the global localStorage flag (project feedback
// 2026-08-20 — see local_prefs.js and trend_ruler.js:initTrendRulerForTicker
// for the identical reasoning), overriding whatever this ticker's own
// analysis_settings row had.
export function initMaForTicker() {
  renderMaList();
  recompute().then(() => redraw()); // values arrive from the backend (full history)
  S.maSettings.showOnChart = getToolShowOnChart('moving_averages', S.maSettings.showOnChart);
  updateShowChartButton();
}

// ── registry entry: one line trace per configured MA ─────────────────────
function buildMaMainTraces() {
  if (!isMaVisible()) return [];
  return S.maSettings.series.map((s, i) => {
    const d = S.maData[s.id];
    if (!d) return null;
    // values cover the full history; only the displayed slice is drawn
    const [i0, i1] = displayIndexRange(d.times);
    return {
      type: 'scatter', mode: 'lines', name: `${s.type.toUpperCase()}(${s.period})`,
      x: d.times.slice(i0, i1), y: d.values.slice(i0, i1),
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
