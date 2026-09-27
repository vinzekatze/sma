// risk_corridor — «Риск-корридор» инструмент: риск-огибающая High/Low
// (running max/min) + полоса Close на h=1..H шагов вперёд,
// sma/core/forecast/risk_corridor.py. Портировано из prototype/forcaster/
// ui/app27-risk-corridor.py, см. docs/plans/app27_risk_corridor_migration_
// plan.md. Четвёртый, независимый forecaster-инструмент — ПРОЩЕ range_
// forecast: нет калибровки вообще (параметры глобальны, валидированы
// честным temporal walk-forward и устойчиво обобщаются на новые тикеры без
// подгонки — эксп.06 фазы 19), поэтому нет settings-таблицы/списка/баннера
// "нужна калибровка" — только coveragePct (единственный видимый контрол вне
// аккордеона «Параметры») + живой прогноз при каждом изменении origin/
// настроек — POST /risk-corridor/live, без сохранения в БД (план §2).
//
// Визуализация — 4 обычных Scatter-линии (полоса Close верх/низ — тонкий
// белый пунктир, риск-огибающая High — зелёная, Low — красная, толще и
// точечная — 1-в-1 цвета/стили прототипа), собранные через ОБЫЧНЫЙ
// buildMainTraces() — тот же generic-диспетчер (chart.js:
// buildAnalyzerMainTraces), которым уже пользуется range_forecast.js —
// никакой доработки chart.js не требуется.

import { S } from './state.js';
import { api, setStatus } from './api.js';
import { renderChart, setOrigin, futureDateAt, resolveOriginCandle } from './chart.js';
import { registerTool, isToolObjectVisible } from './tools.js';
import { getToolShowOnChart, saveToolShowOnChart } from './local_prefs.js';

registerTool({
  type: 'risk_corridor',
  surface: 'main',
  group: 'forecaster',
  icon: 'risk',
  label: 'Риск-корридор (High/Low)',
  panelId: 'tool-panel-risk_corridor',
  onOriginClick: onRiskCorridorOriginClick,
  onSelected: recalcRiskCorridorLive, // первый показ инструмента — сразу живой прогноз, без ожидания клика/смены настройки
  buildMainTraces: buildRiskCorridorMainTraces,
});

// ── "закреплено на графике" (тот же паттерн, что range_forecast.js/
// moving_averages.js/zigzag_tool.js) ──────────────────────────────────────

export function toggleRiskCorridorShowChart() {
  S.riskCorridorShowOnChart = !S.riskCorridorShowOnChart;
  updateRiskCorridorShowChartButton();
  renderChart({ preserveRange: true });
  saveToolShowOnChart('risk_corridor', S.riskCorridorShowOnChart);
}

function updateRiskCorridorShowChartButton() {
  document.getElementById('risk-corridor-show-chart-btn')
    ?.classList.toggle('active', !!S.riskCorridorShowOnChart);
}

function isRiskCorridorVisible() {
  return isToolObjectVisible('risk_corridor', S.riskCorridorShowOnChart);
}

function onRiskCorridorOriginClick(ts) {
  setOrigin(ts);
  recalcRiskCorridorLive();
}

// ── settings form <-> S.riskCorridorSettings ────────────────────────────

function readRiskCorridorSettings() {
  const useAllCandles = document.getElementById('risk-corridor-use-all-candles').checked;
  return {
    h: +document.getElementById('risk-corridor-h').value,
    pFit: +document.getElementById('risk-corridor-p-fit').value,
    blendAlpha: +document.getElementById('risk-corridor-blend-alpha').value,
    theta: +document.getElementById('risk-corridor-theta').value,
    nSim: +document.getElementById('risk-corridor-n-sim').value,
    seed: +document.getElementById('risk-corridor-seed').value,
    coveragePct: +document.getElementById('risk-corridor-coverage-slider').value,
    // 0 = вся история, без обрезки — borrow range_forecast.js/simplex_ensemble.js's useAllCandles/maxCandles pair
    maxCandles: useAllCandles ? 0 : +document.getElementById('risk-corridor-max-candles').value,
  };
}

function applyRiskCorridorSettings(s) {
  document.getElementById('risk-corridor-h').value = s.h;
  document.getElementById('risk-corridor-p-fit').value = s.pFit;
  document.getElementById('risk-corridor-blend-alpha').value = s.blendAlpha;
  document.getElementById('risk-corridor-theta').value = s.theta;
  document.getElementById('risk-corridor-n-sim').value = s.nSim;
  document.getElementById('risk-corridor-seed').value = s.seed;
  document.getElementById('risk-corridor-coverage-slider').value = s.coveragePct;
  document.getElementById('risk-corridor-coverage-label').textContent = `${s.coveragePct}%`;
  const useAll = !s.maxCandles || s.maxCandles <= 0;
  document.getElementById('risk-corridor-use-all-candles').checked = useAll;
  const maxCandlesInput = document.getElementById('risk-corridor-max-candles');
  maxCandlesInput.value = useAll ? 5000 : s.maxCandles;
  maxCandlesInput.disabled = useAll;
}

// Called on every settings-control onchange (see index.html).
export function onRiskCorridorSettingsChange() {
  S.riskCorridorSettings = readRiskCorridorSettings();
  recalcRiskCorridorLive();
}

// Coverage slider — continuous drag, debounced like analysis.js's spectrogram
// settings (500мс) so dragging doesn't fire a server request per pixel.
let _coverageDebounceTimer = null;

export function onRiskCorridorCoverageInput() {
  const pct = +document.getElementById('risk-corridor-coverage-slider').value;
  document.getElementById('risk-corridor-coverage-label').textContent = `${pct}%`;
  S.riskCorridorSettings.coveragePct = pct;
  clearTimeout(_coverageDebounceTimer);
  _coverageDebounceTimer = setTimeout(recalcRiskCorridorLive, 500);
}

// ── live recompute ───────────────────────────────────────────────────────
// Request token guards against an in-flight response landing AFTER a newer
// one already did (rapid setting changes can race POST /risk-corridor/live
// calls out of order) — same pattern zigzag_tool.js/forecast.js already use
// for the same class of bug.
let _liveRequestToken = 0;

export async function recalcRiskCorridorLive() {
  if (!S.instrumentId) return;
  const settings = S.riskCorridorSettings;
  const originCandle = resolveOriginCandle();
  const token = ++_liveRequestToken;
  try {
    const res = await api('POST', '/risk-corridor/live', {
      instrument_id: S.instrumentId, interval: S.interval,
      origin_ts: originCandle ? originCandle.begin : null,
      h: settings.h, p_fit: settings.pFit, blend_alpha: settings.blendAlpha,
      theta: settings.theta, n_sim: settings.nSim, coverage_pct: settings.coveragePct,
      seed: settings.seed, max_candles: settings.maxCandles,
    });
    if (token !== _liveRequestToken) return; // superseded — discard
    S.riskCorridorResult = res;
  } catch (e) {
    if (token !== _liveRequestToken) return;
    S.riskCorridorResult = null;
    setStatus(e.message, 'err');
  }
  renderResultSummary();
  renderChart({ preserveRange: true });
}

function renderResultSummary() {
  const el = document.getElementById('risk-corridor-result-summary');
  if (!el) return;
  const res = S.riskCorridorResult;
  if (!res) { el.innerHTML = '<span class="muted-val">—</span>'; return; }
  const s = S.riskCorridorSettings;
  el.innerHTML = `<small class="hint">Соседей в пуле: ${res.n_neighbors} · p_fit=${s.pFit} · ` +
    `blend_alpha=${s.blendAlpha.toFixed(2)} · theta=${s.theta.toFixed(1)} · nu=1.0 · N_SIM=${s.nSim} · ` +
    `origin: ${res.origin_date}</small>`;
}

// ── main chart traces ────────────────────────────────────────────────────

function buildRiskCorridorMainTraces() {
  if (!isRiskCorridorVisible()) return [];
  const res = S.riskCorridorResult;
  if (!res || !res.steps) return [];
  const steps = Object.keys(res.steps).map(Number).sort((a, b) => a - b);
  if (!steps.length) return [];

  const originDate = res.origin_date;
  const xOrigin = new Date(originDate).toISOString();
  const xClose = [xOrigin], xRisk = [xOrigin];
  const yCloseLo = [res.close_at_origin], yCloseHi = [res.close_at_origin];
  const yRiskHi = [res.close_at_origin], yRiskLo = [res.close_at_origin];

  for (const h of steps) {
    const b = res.steps[String(h)];
    const x = futureDateAt(originDate, h, S.interval, S.candles);
    xClose.push(x); xRisk.push(x);
    yCloseLo.push(b.close_low); yCloseHi.push(b.close_high);
    yRiskHi.push(b.risk_high); yRiskLo.push(b.risk_low);
  }

  const cClose = S.colorProfile.risk_corridor_close;
  const cHigh = S.colorProfile.risk_corridor_high;
  const cLow = S.colorProfile.risk_corridor_low;
  const pct = res.coverage_pct;

  return [
    {
      type: 'scatter', mode: 'lines', x: xClose, y: yCloseHi,
      name: `${pct}% Close верх`, line: { color: cClose, width: 1.3, dash: 'dash' },
    },
    {
      type: 'scatter', mode: 'lines', x: xClose, y: yCloseLo,
      name: `${pct}% Close низ`, line: { color: cClose, width: 1.3, dash: 'dash' },
    },
    {
      type: 'scatter', mode: 'lines', x: xRisk, y: yRiskHi,
      name: `${pct}% риск-огибающая High`, line: { color: cHigh, width: 1.8, dash: 'dot' },
    },
    {
      type: 'scatter', mode: 'lines', x: xRisk, y: yRiskLo,
      name: `${pct}% риск-огибающая Low`, line: { color: cLow, width: 1.8, dash: 'dot' },
    },
  ];
}

// ── ticker/interval switch init (тот же паттерн, что initRangeForecastForTicker) ──
export async function initRiskCorridorForTicker() {
  applyRiskCorridorSettings(S.riskCorridorSettings); // hardcoded defaults — нет персистентности параметров (см. план §5.2 п.6, опционально позже)
  S.riskCorridorShowOnChart = getToolShowOnChart('risk_corridor', S.riskCorridorShowOnChart);
  updateRiskCorridorShowChartButton();
  await recalcRiskCorridorLive();
}
