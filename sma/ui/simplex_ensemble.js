// simplex_ensemble — «Прогноз»-инструмент модели simplex_ensemble (ансамбль
// Simplex projection по нескольким origin), свой toolbar tool наравне с
// band_lambda (forecast.js) — не заменяет его, а дополняет как
// вспомогательный инструмент оценки ситуации (см. sma/core/forecast/
// simplex_ensemble.py докстринг и /home/kali/.claude/plans/
// unified-kindling-lagoon.md).
//
// История/выбор/пины — общие механизмом (не общим DOM) для всех моделей, см.
// forecast_history.js (этот файл зависит от него, не от forecast.js —
// никакого дублирования и никакого цикла импортов); результаты и история
// отображаются в СВОИХ #results-simplex_ensemble/#history-simplex_ensemble
// (index.html). Прогресс выполнения задачи отображается ТОЛЬКО в менеджере
// задач левой панели (#task-status/#task-list, sma/ui/tasks.js) — здесь
// только refreshTasks() для мгновенного обновления той панели по
// WS-событиям.
//
// Origin — общий маркер графика (S.originTs, setOrigin в chart.js, тот же
// клик, что и у band_lambda). Здесь резолвится НАПРЯМУЮ в бар (без пивотов
// зигзага, которых у этой модели нет) — см. resolveOriginCandle.

import { S } from './state.js';
import { api, setStatus, connectTaskWS } from './api.js';
import { renderChart, setOrigin } from './chart.js';
import { refreshTasks } from './tasks.js';
import { registerModelHandlers, loadAndRenderForecast } from './forecast_history.js';
import { registerTool } from './tools.js';

// Wires this model into the unified history/selection system — when a
// simplex_ensemble forecast becomes the selected one, refill its params form
// (same UX simplex_ensemble always had via applySimplexParams).
registerModelHandlers('simplex_ensemble', {
  onSelected(f) {
    applySimplexParams(f.params);
  },
});

// Registered as a "forecaster"-group tool — see tools.js module docstring /
// forecast.js's own registerTool call for why forecast models share
// S.activeMainTool with analyzers instead of a separate exclusivity slot.
// No buildMainTraces — chart presence flows through the pre-existing pin/
// select overlay system, this is purely toolbar placement + click-target
// arbitration. Unlike band_lambda, no zigzag-pivot lookup on click — just
// the shared origin marker (see resolveOriginCandle below for how origin
// resolves to a candle at submit time).
registerTool({
  type: 'simplex_ensemble',
  surface: 'main',
  group: 'forecaster',
  icon: 'ensemble',
  label: 'simplex_ensemble — ансамбль по origin',
  panelId: 'tool-panel-simplex_ensemble',
  onOriginClick: setOrigin,
});

// ── params: read/write form <-> S.simplexParams ─────────────────────────────

function readSimplexParams() {
  const useAllBars = document.getElementById('simplex-use-all-bars').checked;
  return {
    window: +document.getElementById('simplex-window').value,
    horizon: +document.getElementById('simplex-horizon').value,
    xy_x: +document.getElementById('simplex-xy-x').value,
    xy_y: +document.getElementById('simplex-xy-y').value,
    xi_add: +document.getElementById('simplex-xi-add').value,
    blend_alpha: +document.getElementById('simplex-blend-alpha').value,
    n_levels: +document.getElementById('simplex-n-levels').value,
    p_cascade_max: +document.getElementById('simplex-p-cascade-max').value,
    bars: useAllBars ? 0 : +document.getElementById('simplex-bars').value,
    pca_p_range: [
      +document.getElementById('simplex-pca-p-min').value,
      +document.getElementById('simplex-pca-p-max').value,
    ],
    pca_thr_range: [
      +document.getElementById('simplex-pca-thr1').value,
      +document.getElementById('simplex-pca-thr2').value,
    ],
    use_lp_corr: document.getElementById('simplex-use-lp-corr').checked,
  };
}

function applySimplexParams(p) {
  if (!p) return;
  document.getElementById('simplex-window').value = p.window;
  document.getElementById('simplex-horizon').value = p.horizon;
  document.getElementById('simplex-xy-x').value = p.xy_x;
  document.getElementById('simplex-xy-y').value = p.xy_y;
  document.getElementById('simplex-xi-add').value = p.xi_add;
  document.getElementById('simplex-blend-alpha').value = p.blend_alpha;
  const alphaLabel = document.getElementById('simplex-blend-alpha-label');
  if (alphaLabel) alphaLabel.textContent = (+p.blend_alpha).toFixed(2);
  document.getElementById('simplex-n-levels').value = p.n_levels;
  document.getElementById('simplex-p-cascade-max').value = p.p_cascade_max;
  const useAll = !p.bars || p.bars === 0;
  document.getElementById('simplex-use-all-bars').checked = useAll;
  const barsInput = document.getElementById('simplex-bars');
  barsInput.value = useAll ? 3000 : p.bars;
  barsInput.disabled = useAll;
  document.getElementById('simplex-pca-p-min').value = p.pca_p_range[0];
  document.getElementById('simplex-pca-p-max').value = p.pca_p_range[1];
  document.getElementById('simplex-pca-thr1').value = p.pca_thr_range[0];
  document.getElementById('simplex-pca-thr2').value = p.pca_thr_range[1];
  document.getElementById('simplex-use-lp-corr').checked = !!p.use_lp_corr;
  S.simplexParams = p;
}

// Last-used params for this (instrument, interval) — see db.py
// forecast_defaults table docstring; only ever written server-side after a
// forecast that actually completed (task_manager.py:_run_forecast_simplex).
// Falls back to the hardcoded S.simplexParams defaults (state.js) when
// nothing is saved yet.
export async function loadSimplexDefaults() {
  if (!S.instrumentId) return;
  try {
    const res = await api(
      'GET',
      `/forecast-settings/defaults?instrument_id=${S.instrumentId}&interval=${S.interval}&model_type=simplex_ensemble`
    );
    applySimplexParams(res.params || S.simplexParams);
  } catch (_) {
    applySimplexParams(S.simplexParams);
  }
}

// ── origin resolution ─────────────────────────────────────────────────────
// No zigzag/pivot here (unlike band_lambda's findOriginPivot) — origin IS
// the clicked bar itself. S.originTs is a 10-char date (chart.js click
// handler always truncates to date, regardless of interval — same
// limitation band_lambda's own pivot matching has for intraday intervals);
// candle.begin may carry a full timestamp, hence the slice comparison. When
// a date has several intraday bars, the LAST one is "the last known bar"
// for that day — matches the model's own semantics (origin = last known
// bar, h=1 predicts the bar AFTER it).
function resolveOriginCandle() {
  if (!S.originTs) return null;
  const matches = S.candles.filter(c => c.begin.slice(0, 10) === S.originTs);
  return matches.length ? matches[matches.length - 1] : null;
}

// ── submit / progress ────────────────────────────────────────────────────

export async function submitSimplexForecast() {
  if (!S.instrumentId) { setStatus('Сначала загрузите свечи', 'err'); return; }

  const originCandle = resolveOriginCandle();
  const btn = document.getElementById('simplex-forecast-btn');
  btn.disabled = true;
  setStatus('Создание задачи…', 'busy');
  try {
    const body = {
      instrument_id: S.instrumentId, interval: S.interval,
      model_type: 'simplex_ensemble',
      params: readSimplexParams(),
    };
    if (originCandle) body.origin_candle_id = originCandle.id;
    const task = await api('POST', '/forecasts', body);
    btn.disabled = false;
    setStatus(`Задача #${task.task_id} поставлена в очередь`, 'ok');
    refreshTasks(); // immediate task-manager refresh — progress lives there now (#task-status/#task-list)
    connectTaskWS(task.task_id, msg => {
      if (msg.status === 'running') {
        setStatus(
          msg.total > 0
            ? `Задача #${task.task_id} — origin ${msg.done}/${msg.total}…`
            : `Задача #${task.task_id} выполняется…`,
          'busy'
        );
      }
      if (msg.status === 'done') {
        loadAndRenderForecast(msg.forecast_id);
      }
      if (msg.status === 'error') {
        setStatus(`Ошибка: ${msg.error}`, 'err');
      }
      if (msg.status === 'cancelled') {
        setStatus('Задача отменена', 'err');
      }
      refreshTasks();
    });
  } catch (e) {
    btn.disabled = false;
    setStatus(e.message, 'err');
  }
}

// ── display: show-mean toggle, band-width slider (reactive, no re-fetch) ───
// MODEL_DISPLAY.simplex_ensemble.buildOverlay (chart.js) reads
// S.simplexShowMean/S.simplexBandPct live at render time — these just update
// that state and ask for a redraw, no result-lookup needed here.

export function refreshSimplexDisplay() {
  S.simplexShowMean = document.getElementById('simplex-show-mean').checked;
  renderChart({ preserveRange: true });
}

export function onSimplexBandPctChange(val) {
  S.simplexBandPct = +val;
  const label = document.getElementById('simplex-band-pct-label');
  if (label) label.textContent = `${val}%`;
  renderChart({ preserveRange: true });
}
