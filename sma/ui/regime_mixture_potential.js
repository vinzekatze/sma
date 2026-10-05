// regime_mixture_potential — «Прогноз»-инструмент модели
// regime_mixture_potential (regime-conditioned pairlag lagged-ensemble +
// GaussianMixture-сценарии потенциала, с перемоткой по origin), свой toolbar
// tool наравне с band_lambda/simplex_ensemble. Портировано из
// prototype/forcaster/ui/app33-regime-mixture-rewind.py, см. docs/plans/
// app33_regime_mixture_potential_migration_plan.md.
//
// История/выбор/пины — общие механизмом (forecast_history.js), результаты и
// история — в СВОИХ #results-regime_mixture_potential/#history-regime_
// mixture_potential (index.html). Построение Plotly-трейсов (heatmap/
// границы) живёт в chart.js (MODEL_DISPLAY.regime_mixture_potential),
// НЕ здесь — chart.js никогда не импортирует из модельных файлов, только
// наоборот (см. план §4.2 для истории этого решения).
//
// Origin — общий маркер графика (S.originTs, setOrigin в chart.js), как у
// simplex_ensemble: резолвится напрямую в бар (нет пивотов зигзага).
// Премотка — S.potentialRewindIdx, чисто клиентская (выбирает снапшот из
// уже прогретого result.snapshots), не требует нового запроса.

import { S } from './state.js';
import { api, setStatus, connectTaskWS } from './api.js';
import { renderChart, setOrigin, resolveOriginCandle } from './chart.js';
import { refreshTasks } from './tasks.js';
import { registerModelHandlers, loadAndRenderForecast } from './forecast_history.js';
import { registerTool } from './tools.js';
import { saveToolDisplayDefaults } from './settings.js';

registerModelHandlers('regime_mixture_potential', {
  onSelected(f) {
    applyPotentialParams(f.params);
    S.potentialRewindIdx = 0; // новый выбранный прогноз — всегда начинаем с самого свежего origin
    _syncRewindSliderFromResult(f.result);
  },
});

registerTool({
  type: 'regime_mixture_potential',
  surface: 'main',
  group: 'forecaster',
  icon: 'potential',
  label: 'Потенциал — режим + сценарии',
  panelId: 'tool-panel-regime_mixture_potential',
  onOriginClick: setOrigin,
});

// ── params: read/write form <-> S.potentialParams ───────────────────────────

function readPotentialParams() {
  const useAllBars = document.getElementById('potential-use-all-bars').checked;
  return {
    theta: +document.getElementById('potential-theta').value,
    warmup: +document.getElementById('potential-warmup').value,
    theiler_window: +document.getElementById('potential-theiler').value,
    horizon: +document.getElementById('potential-horizon').value,
    bars: useAllBars ? 0 : +document.getElementById('potential-bars').value,
    n_lookback: +document.getElementById('potential-n-lookback').value,
    lookback_step: +document.getElementById('potential-lookback-step').value,
    n_sim: +document.getElementById('potential-n-sim').value,
    seed: +document.getElementById('potential-seed').value,
    mix_n_resample: +document.getElementById('potential-mix-n-resample').value,
    bin_height_pct: +document.getElementById('potential-bin-height').value,
    // Coverage lives in "Отображение" now (a display-only slider, see
    // onPotentialCoverageInput) rather than in the request form — the
    // border is recomputed live from raw_histogram on every render (chart.js:
    // potentialCoverageBoundsFromHistogram), so the value submitted here only
    // matters for the snapshot's OWN stored `bounds` field, unused for
    // display any more.
    coverage_pct: S.potentialCoveragePct,
    n_forecasts: +document.getElementById('potential-n-forecasts').value,
    rewind_step: +document.getElementById('potential-rewind-step').value,
  };
}

function applyPotentialParams(p) {
  if (!p) return;
  document.getElementById('potential-theta').value = p.theta;
  document.getElementById('potential-warmup').value = p.warmup;
  document.getElementById('potential-theiler').value = p.theiler_window;
  document.getElementById('potential-horizon').value = p.horizon;
  const useAll = !p.bars || p.bars === 0;
  document.getElementById('potential-use-all-bars').checked = useAll;
  const barsInput = document.getElementById('potential-bars');
  barsInput.value = useAll ? 5000 : p.bars;
  barsInput.disabled = useAll;
  document.getElementById('potential-n-lookback').value = p.n_lookback;
  document.getElementById('potential-lookback-step').value = p.lookback_step;
  document.getElementById('potential-n-sim').value = p.n_sim;
  document.getElementById('potential-seed').value = p.seed;
  document.getElementById('potential-mix-n-resample').value = p.mix_n_resample;
  document.getElementById('potential-bin-height').value = p.bin_height_pct;
  _setPotentialCoverage(p.coverage_pct);
  document.getElementById('potential-n-forecasts').value = p.n_forecasts;
  document.getElementById('potential-rewind-step').value = p.rewind_step;
  S.potentialParams = {
    theta: p.theta, warmup: p.warmup, theilerWindow: p.theiler_window, horizon: p.horizon,
    bars: p.bars, nLookback: p.n_lookback, lookbackStep: p.lookback_step, nSim: p.n_sim,
    seed: p.seed, mixNResample: p.mix_n_resample, binHeightPct: p.bin_height_pct,
    coveragePct: p.coverage_pct, nForecasts: p.n_forecasts, rewindStep: p.rewind_step,
  };
}

// Last-used params for this (instrument, interval) — see db.py
// forecast_defaults table docstring; falls back to hardcoded S.potentialParams
// (state.js) when nothing is saved yet.
export async function loadPotentialDefaults() {
  if (!S.instrumentId) return;
  try {
    const res = await api(
      'GET',
      `/forecast-settings/defaults?instrument_id=${S.instrumentId}&interval=${S.interval}&model_type=regime_mixture_potential`
    );
    applyPotentialParams(res.params || _paramsFromState());
  } catch (_) {
    applyPotentialParams(_paramsFromState());
  }
  applyGlobalPotentialDisplay(); // "Границы покрытия" checkbox + slider — GLOBAL, not per-ticker (project request 2026-10-04), see below
}

// GLOBAL standing preference (settings.js:saveToolDisplayDefaults /
// sma/core/db.py:DEFAULT_TOOL_DISPLAY), unlike coverage_pct INSIDE
// S.potentialParams above — that one is a model SUBMISSION parameter
// (per-ticker, forecast_defaults), a different thing with a similar name;
// S.potentialShowBounds/potentialCoveragePct are purely the client-side
// "Отображение" slider (see onPotentialCoverageInput's own docstring).
// Re-applied every ticker switch (idempotent), same pattern trend_ruler's
// showOnChart already uses.
function applyGlobalPotentialDisplay() {
  const g = S.toolDisplayDefaults.regime_mixture_potential;
  S.potentialShowBounds = g.showBounds;
  document.getElementById('potential-show-bounds').checked = g.showBounds;
  _setPotentialCoverage(g.coveragePct);
}

function _paramsFromState() {
  const p = S.potentialParams;
  return {
    theta: p.theta, warmup: p.warmup, theiler_window: p.theilerWindow, horizon: p.horizon,
    bars: p.bars, n_lookback: p.nLookback, lookback_step: p.lookbackStep, n_sim: p.nSim,
    seed: p.seed, mix_n_resample: p.mixNResample, bin_height_pct: p.binHeightPct,
    coverage_pct: p.coveragePct, n_forecasts: p.nForecasts, rewind_step: p.rewindStep,
  };
}

// ── submit / progress ────────────────────────────────────────────────────

export async function submitPotentialForecast() {
  if (!S.instrumentId) { setStatus('Сначала загрузите свечи', 'err'); return; }

  const originCandle = resolveOriginCandle();
  const btn = document.getElementById('potential-forecast-btn');
  btn.disabled = true;
  setStatus('Создание задачи…', 'busy');
  try {
    const body = {
      instrument_id: S.instrumentId, interval: S.interval,
      model_type: 'regime_mixture_potential',
      params: readPotentialParams(),
    };
    if (originCandle) body.origin_candle_id = originCandle.id;
    const task = await api('POST', '/forecasts', body);
    btn.disabled = false;
    setStatus(`Задача #${task.task_id} поставлена в очередь`, 'ok');
    refreshTasks();
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
        S.potentialRewindIdx = 0; // свежий прогноз — начинаем с самого свежего origin
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

// ── «Отображение»: heatmap-тип / XY-цвет / покрытие / перемотка / границы
// (реактивные, без рефетча) ────────────────────────────────────────────────
// MODEL_DISPLAY.regime_mixture_potential.buildOverlay (chart.js) читает
// S.potential* живьём при каждом рендере — эти функции просто обновляют
// состояние и просят перерисовку, без похода на сервер (все данные для
// ЛЮБОГО положения ползунка/режима уже прогреты в result.snapshots).

export function onPotentialHeatmapModeChange(mode) {
  S.potentialHeatmapMode = mode;
  renderChart({ preserveRange: true });
}

export function onPotentialColorXYChange(x, y) {
  S.potentialColorShift = x;
  S.potentialColorSteepness = y;
  renderChart({ preserveRange: true });
}

export function onPotentialShowBoundsChange() {
  S.potentialShowBounds = document.getElementById('potential-show-bounds').checked;
  S.toolDisplayDefaults.regime_mixture_potential.showBounds = S.potentialShowBounds;
  renderChart({ preserveRange: true });
  saveToolDisplayDefaults();
}

// Coverage slider — unlike risk_corridor's coverage (which re-fetches exact
// quantiles from the server), this one is purely reactive: every snapshot
// already carries raw_histogram (an empirical density per h, unconditionally
// computed server-side — see regime_mixture_potential.py:_build_one_snapshot),
// which is enough to derive the lo/hi bound for ANY coverage % on the client
// (chart.js:potentialCoverageBoundsFromHistogram) — no debounce, no request,
// same "instant" feel as the heatmap-mode/rewind/XY-pad controls.
export function onPotentialCoverageInput(pct) {
  _setPotentialCoverage(+pct);
  S.toolDisplayDefaults.regime_mixture_potential.coveragePct = +pct;
  renderChart({ preserveRange: true });
  saveToolDisplayDefaults();
}

function _setPotentialCoverage(pct) {
  S.potentialCoveragePct = pct;
  const slider = document.getElementById('potential-coverage-slider');
  const label = document.getElementById('potential-coverage-label');
  if (slider) slider.value = pct;
  if (label) label.textContent = `${pct}%`;
}

export function onPotentialRewindChange(idx) {
  S.potentialRewindIdx = +idx;
  const label = document.getElementById('potential-rewind-label');
  const f = S.renderedForecasts.get(S.selectedForecastId);
  if (label && f?.result?.snapshots) {
    const snap = f.result.snapshots[+idx];
    if (snap) label.textContent = snap.origin_date.slice(0, 10);
  }
  renderChart({ preserveRange: true });
}

// Слайдер перемотки — max/value/подпись зависят от ТЕКУЩЕГО выбранного
// прогноза (число снапшотов), пересинхронизируется при каждом выборе
// прогноза из истории/значка (см. registerModelHandlers.onSelected выше).
function _syncRewindSliderFromResult(result) {
  const slider = document.getElementById('potential-rewind-slider');
  const label = document.getElementById('potential-rewind-label');
  if (!slider) return;
  const n = result?.snapshots?.length || 1;
  slider.max = n - 1;
  slider.value = 0;
  slider.disabled = n <= 1;
  if (label) label.textContent = result?.snapshots?.[0]?.origin_date?.slice(0, 10) ?? 'origin';
}

// ── XY-пад (крутизна × сдвиг цвета heatmap) — квадратный 2D-контроллер, как
// на синтезаторах/секвенсорах: X — горизонталь (крутизна, [Y_MIN,Y_MAX]), Y —
// вертикаль, ИНВЕРТИРОВАНА (верх пада = максимум сдвига) — драг мышью/тачем,
// без библиотек (см. style.css:.xy-pad). Крутизна на X / сдвиг на Y — не
// наоборот, как в первой версии — оказалось интуитивнее (проект. фидбэк):
// горизонтальная протяжка пальцем/мышью естественнее читается как «жёстче/
// мягче», а не как сдвиг диапазона. onPotentialColorXYChange(shift,
// steepness) ниже сохраняет свою историческую сигнатуру (shift первым) —
// меняется только то, какая физическая ось пада что двигает. ─────────────

const POTENTIAL_XY_Y_MIN = 0.5, POTENTIAL_XY_Y_MAX = 20.0;

function _updatePotentialXYPadUI(shift, steepness) {
  const thumb = document.getElementById('potential-xy-thumb');
  if (thumb) {
    thumb.style.left = `${(steepness - POTENTIAL_XY_Y_MIN) / (POTENTIAL_XY_Y_MAX - POTENTIAL_XY_Y_MIN) * 100}%`;
    thumb.style.top = `${(1 - shift) * 100}%`;
  }
  const xLabel = document.getElementById('potential-xy-x-label');
  const yLabel = document.getElementById('potential-xy-y-label');
  if (xLabel) xLabel.textContent = `X: ${steepness.toFixed(1)}`;
  if (yLabel) yLabel.textContent = `Y: ${shift.toFixed(2)}`;
}

function _handlePotentialPadPointer(pad, evt) {
  const rect = pad.getBoundingClientRect();
  const xFrac = Math.min(Math.max((evt.clientX - rect.left) / rect.width, 0), 1);
  const steepness = POTENTIAL_XY_Y_MIN + xFrac * (POTENTIAL_XY_Y_MAX - POTENTIAL_XY_Y_MIN);
  const yFrac = Math.min(Math.max((evt.clientY - rect.top) / rect.height, 0), 1);
  const shift = 1 - yFrac;
  _updatePotentialXYPadUI(shift, steepness);
  onPotentialColorXYChange(shift, steepness);
}

function initPotentialXYPad() {
  const pad = document.getElementById('potential-xy-pad');
  if (!pad) return;
  _updatePotentialXYPadUI(S.potentialColorShift, S.potentialColorSteepness);
  let dragging = false;
  pad.addEventListener('pointerdown', e => { dragging = true; pad.setPointerCapture(e.pointerId); _handlePotentialPadPointer(pad, e); });
  pad.addEventListener('pointermove', e => { if (dragging) _handlePotentialPadPointer(pad, e); });
  pad.addEventListener('pointerup', () => { dragging = false; });
  pad.addEventListener('pointercancel', () => { dragging = false; });
}

initPotentialXYPad();
