// range_forecast — «Прогноз» инструмент прогноза ДИАПАЗОНА (min(low)..
// max(high)) на h=1..H шагов вперёд, sma/core/forecast/range_forecast.py.
// Третий, независимый forecaster-инструмент (см. docs/plans/
// app16_range_forecast_migration_plan.md) — БЕЗ персистентности прогнозов
// (нет forecasts-записи, план §3): калибровка (θ+5λ+read-квантиль по
// уровням) пишется в БД (range_forecast_settings), сам прогноз считается
// ЖИВЬЁМ при каждом изменении origin/настроек — POST /range-forecast/live,
// миллисекунды (один поиск соседей + дешёвые квантили).
//
// В отличие от band_lambda/simplex_ensemble (явная кнопка «Прогноз» ->
// forecast_id с историей, sma/ui/forecast_history.js), этот инструмент
// ближе по духу к trend_ruler.js (живой реактивный предпросмотр) — только
// компute здесь на сервере, не клиенте, поэтому каждый пересчёт — async
// fetch, а не синхронный вызов. Никакой истории/пинов — нечего пинить, нет
// forecast_id.
//
// Визуализация — НЕ прямоугольники (band_lambda:buildBandZoneShapes) и НЕ
// line+fill:'tonexty' полосы (simplex/trend_ruler style): для каждого шага
// h — вертикальная линия (тонкий Scatter, mode:'lines', 2 точки) от min до
// max по всем включённым уровням + горизонтальные засечки на границах
// каждого уровня (тоже Scatter, opacity по уровню — узкий уровень плотнее).
// Собрано через ОБЫЧНЫЙ buildMainTraces() — тот же generic-диспетчер
// (chart.js:buildAnalyzerMainTraces), которым уже пользуется trend_ruler.js
// для своих fill:'tonexty' полос; отдельного Plotly-shapes хука для
// main-chart траекторий в chart.js нет и не нужен — Scatter-трасса
// работает для тонких линий не хуже layout.shape и не требует трогать
// chart.js вообще.

import { S, INTERVAL_SECONDS } from './state.js';
import { api, setStatus, connectTaskWS } from './api.js';
import { renderChart, setOrigin, futureDateAt, resolveOriginCandle } from './chart.js';
import { registerTool, isToolObjectVisible } from './tools.js';
import { getToolShowOnChart, saveToolShowOnChart } from './local_prefs.js';

registerTool({
  type: 'range_forecast',
  surface: 'main',
  group: 'forecaster',
  icon: 'range',
  label: 'Прогноз диапазона (H шагов)',
  panelId: 'tool-panel-range_forecast',
  onOriginClick: onRangeForecastOriginClick,
  onSelected: recalcRangeForecastLive, // первый показ инструмента — сразу живой прогноз, без ожидания клика/смены настройки
  buildMainTraces: buildRangeForecastMainTraces,
});

// ── "закреплено на графике" (см. moving_averages.js/zigzag_tool.js — тот же
// паттерн: eye-иконка в заголовке панели, видно даже когда инструмент не
// активен, глобально по типу инструмента, переживает смену тикера/reload) ──

export function toggleRangeForecastShowChart() {
  S.rangeForecastShowOnChart = !S.rangeForecastShowOnChart;
  updateRangeForecastShowChartButton();
  renderChart({ preserveRange: true });
  saveToolShowOnChart('range_forecast', S.rangeForecastShowOnChart);
}

function updateRangeForecastShowChartButton() {
  document.getElementById('range-forecast-show-chart-btn')
    ?.classList.toggle('active', !!S.rangeForecastShowOnChart);
}

function isRangeForecastVisible() {
  return isToolObjectVisible('range_forecast', S.rangeForecastShowOnChart);
}

function onRangeForecastOriginClick(ts) {
  setOrigin(ts);
  recalcRangeForecastLive();
}

// ── settings form <-> S.rangeForecastSettings ───────────────────────────

function readRangeForecastSettings() {
  const levels = [];
  if (document.getElementById('range-forecast-level-50')?.checked) levels.push(50);
  if (document.getElementById('range-forecast-level-75')?.checked) levels.push(75);
  if (document.getElementById('range-forecast-level-90')?.checked) levels.push(90);
  const useAllCandles = document.getElementById('range-forecast-use-all-candles').checked;
  return {
    hSteps: +document.getElementById('range-forecast-h').value,
    p: +document.getElementById('range-forecast-p').value,
    theiler: +document.getElementById('range-forecast-theiler').value,
    levels: levels.length ? levels : [90],
    mode: document.getElementById('range-forecast-mode').value,
    // 0 = вся история, без обрезки — borrow simplex_ensemble.js:readSimplexSettings's
    // useAllBars/bars pair ("все точки библиотеки"/"точек в библиотеке");
    // параметр запроса, НЕ молчаливая константа на бэкенде (см. range_forecast.py).
    maxCandles: useAllCandles ? 0 : +document.getElementById('range-forecast-max-candles').value,
  };
}

function applyRangeForecastSettings(s) {
  document.getElementById('range-forecast-h').value = s.hSteps;
  document.getElementById('range-forecast-p').value = s.p;
  document.getElementById('range-forecast-theiler').value = s.theiler;
  document.getElementById('range-forecast-level-50').checked = s.levels.includes(50);
  document.getElementById('range-forecast-level-75').checked = s.levels.includes(75);
  document.getElementById('range-forecast-level-90').checked = s.levels.includes(90);
  document.getElementById('range-forecast-mode').value = s.mode;
  const useAll = !s.maxCandles || s.maxCandles <= 0;
  document.getElementById('range-forecast-use-all-candles').checked = useAll;
  const maxCandlesInput = document.getElementById('range-forecast-max-candles');
  maxCandlesInput.value = useAll ? 5000 : s.maxCandles;
  maxCandlesInput.disabled = useAll;
}

// Called on every settings-control onchange (see index.html) — re-reads the
// whole form (simplest correct thing: cheap, and several fields can be
// interdependent — e.g. mode + levels both affect isCalibrationFresh below).
export function onRangeForecastSettingsChange() {
  S.rangeForecastSettings = readRangeForecastSettings();
  recalcRangeForecastLive();
}

// ── calibration freshness ("нужна калибровка" banner) ────────────────────
// Mirrors prototype/forcaster/ui/app16-range-forecast.py's `_cal_fresh`
// check — a calibration only applies to the EXACT (h_steps,p,theiler) it
// was run for; changing any of the three means the stored θ/λ/read-quantile
// answered a different question and must not be applied silently (plan §8
// п.1 — decided WITH the user: never substitute quietly).

function isCalibrationFresh() {
  const { hSteps, p, theiler } = S.rangeForecastSettings;
  return S.rangeForecastCalibrationList.some(
    row => row.h_steps === hSteps && row.p === p && row.theiler === theiler
  );
}

function updateCalibrationBanner() {
  const banner = document.getElementById('range-forecast-need-calibration');
  if (!banner) return;
  const { mode } = S.rangeForecastSettings;
  banner.style.display = (mode === 'calibrated' && !isCalibrationFresh()) ? '' : 'none';
}

export async function refreshRangeForecastCalibrationList() {
  if (!S.instrumentId) return;
  try {
    S.rangeForecastCalibrationList = await api(
      'GET', `/range-forecast/settings?instrument_id=${S.instrumentId}&interval=${S.interval}`
    );
  } catch (_) {
    S.rangeForecastCalibrationList = [];
  }
  updateCalibrationBanner();
  renderCalibrationResult();
}

// ── live recompute ───────────────────────────────────────────────────────
// Own, independent decision (not spelled out in the plan beyond "не
// подставлять молча"): when mode='calibrated' and the current
// (hSteps,p,theiler) has no fresh calibration, show the banner AND clear
// any previously drawn zones rather than drawing a stale/disabled-looking
// forecast — simplest, safest reading of "don't substitute silently".

// Request token guards against an in-flight response landing AFTER a newer
// one already did (rapid setting changes can race POST /range-forecast/live
// calls out of order) — same pattern zigzag_tool.js/forecast.js already use
// for the same class of bug, just a single module-level counter since only
// one "live" recompute is ever in flight for this tool (not one per series).
let _liveRequestToken = 0;

export async function recalcRangeForecastLive() {
  if (!S.instrumentId) return;
  const settings = S.rangeForecastSettings;
  updateCalibrationBanner();
  if (settings.mode === 'calibrated' && !isCalibrationFresh()) {
    S.rangeForecastZones = null;
    renderChart({ preserveRange: true });
    return;
  }
  const originCandle = resolveOriginCandle();
  const token = ++_liveRequestToken;
  try {
    const res = await api('POST', '/range-forecast/live', {
      instrument_id: S.instrumentId, interval: S.interval,
      origin_ts: originCandle ? originCandle.begin : null,
      h_steps: settings.hSteps, p: settings.p, theiler: settings.theiler,
      levels: settings.levels, mode: settings.mode, max_candles: settings.maxCandles,
    });
    if (token !== _liveRequestToken) return; // superseded — discard
    S.rangeForecastZones = res;
  } catch (e) {
    if (token !== _liveRequestToken) return;
    S.rangeForecastZones = null;
    setStatus(e.message, 'err');
  }
  renderChart({ preserveRange: true });
}

// ── calibration task ─────────────────────────────────────────────────────

export async function calibrateRangeForecast() {
  if (!S.instrumentId) { setStatus('Сначала загрузите свечи', 'err'); return; }
  const btn = document.getElementById('range-forecast-calibrate-btn');
  btn.disabled = true;
  setStatus('Калибровка диапазона: постановка в очередь…', 'busy');
  const { hSteps, p, theiler, levels, maxCandles } = S.rangeForecastSettings;
  try {
    const task = await api('POST', '/range-forecast/calibrate', {
      instrument_id: S.instrumentId, interval: S.interval,
      h_steps: hSteps, p, theiler, levels, max_candles: maxCandles,
    });
    setStatus(`Задача #${task.task_id} поставлена в очередь`, 'busy');
    connectTaskWS(task.task_id, async msg => {
      if (msg.status === 'running') {
        setStatus(
          msg.total > 0 ? `Калибровка диапазона — ${msg.done}/${msg.total}…` : 'Калибровка диапазона выполняется…',
          'busy'
        );
      }
      if (msg.status === 'done') {
        btn.disabled = false;
        setStatus('Калибровка диапазона завершена', 'ok');
        await refreshRangeForecastCalibrationList();
        await recalcRangeForecastLive();
      }
      if (msg.status === 'error') {
        btn.disabled = false;
        setStatus(`Ошибка калибровки: ${msg.error}`, 'err');
      }
      if (msg.status === 'cancelled') {
        btn.disabled = false;
        setStatus('Калибровка отменена', 'err');
      }
    });
  } catch (e) {
    btn.disabled = false;
    setStatus(e.message, 'err');
  }
}

// ── calibration result display (θ+5λ, сходимость, per-level read-квантиль) ──

function renderCalibrationResult() {
  const el = document.getElementById('range-forecast-calibration-result');
  if (!el) return;
  const { hSteps, p, theiler } = S.rangeForecastSettings;
  const row = S.rangeForecastCalibrationList.find(r => r.h_steps === hSteps && r.p === p && r.theiler === theiler);
  if (!row) { el.innerHTML = '<small class="hint">Для этой комбинации H/p/theiler ещё нет калибровки.</small>'; return; }

  const { params, q_read_by_level, calibration_meta: m } = row;
  const paramsLine = Object.entries(params).map(([k, v]) => `${k}=${(+v).toFixed(2)}`).join(', ');
  const qrRows = Object.entries(q_read_by_level)
    .sort((a, b) => +b[0] - +a[0])
    .map(([lv, qr]) => `<tr><td>${lv}%</td><td>${qr.low.toFixed(3)}</td><td>${qr.high.toFixed(3)}</td></tr>`)
    .join('');
  el.innerHTML = `
    <small class="hint">${paramsLine}</small>
    <div class="field-row"><label>Сходимость</label><span class="muted-val">${m.converged ? 'да' : 'нет (потолок проходов)'}, ${m.n_passes} проход(ов)</span></div>
    <div class="field-row"><label>Ratio vs uncond</label><span class="muted-val">${m.final_baseline_ratio?.toFixed(4)} → ${m.final_ratio?.toFixed(4)}</span></div>
    <table class="bl-coverage-table">
      <thead><tr><th>Уровень</th><th>q_low read</th><th>q_high read</th></tr></thead>
      <tbody>${qrRows}</tbody>
    </table>
  `;
}

// ── main chart traces (per-step vertical line + per-level horizontal ticks) ──

const LEVEL_OPACITY = { 50: 0.95, 75: 0.65, 90: 0.35 }; // узкий уровень (менее вероятный выход за границу) рисуется плотнее — см. прототип

function buildRangeForecastMainTraces() {
  if (!isRangeForecastVisible()) return [];
  const res = S.rangeForecastZones;
  if (!res || !res.zones_by_step) return [];
  const { levels } = S.rangeForecastSettings;
  const color = S.colorProfile.range_forecast_line;
  const originDate = res.origin_date || '';
  const stepSec = INTERVAL_SECONDS[S.interval] ?? 86400;
  const tickHalfMs = stepSec * 1000 * 0.35; // доля межбарного интервала — см. прототип's tick_half

  const traces = [];
  const steps = Object.keys(res.zones_by_step).map(Number).sort((a, b) => a - b);
  for (const h of steps) {
    const levelBounds = res.zones_by_step[String(h)];
    const availableLevels = levels.filter(lv => levelBounds[String(lv)]);
    if (!availableLevels.length) continue;

    // Снап на РЕАЛЬНЫЙ бар, если он уже есть (прогноз в прошлом с пробелами
    // истории — origin не последний бар) — иначе равномерный шаг БЕЗ
    // угадывания выходных (см. futureDateAt, project finding 2026-09-06:
    // BSPB иногда торгуется по выходным, жёсткий Sat/Sun-скип был неверен).
    const xCenterMs = new Date(futureDateAt(originDate, h, S.interval, S.candles)).getTime();
    const xCenter = new Date(xCenterMs).toISOString();
    const xLo = new Date(xCenterMs - tickHalfMs).toISOString();
    const xHi = new Date(xCenterMs + tickHalfMs).toISOString();

    let spineLo = Infinity, spineHi = -Infinity;
    for (const lv of availableLevels) {
      const [lo, hi] = levelBounds[String(lv)];
      if (lo < spineLo) spineLo = lo;
      if (hi > spineHi) spineHi = hi;
    }
    traces.push({
      type: 'scatter', mode: 'lines', x: [xCenter, xCenter], y: [spineLo, spineHi],
      line: { color, width: 1.5 }, opacity: 0.5, showlegend: false, hoverinfo: 'skip',
    });
    for (const lv of availableLevels) {
      const [lo, hi] = levelBounds[String(lv)];
      const op = LEVEL_OPACITY[lv] ?? 0.5;
      for (const y of [lo, hi]) {
        traces.push({
          type: 'scatter', mode: 'lines', x: [xLo, xHi], y: [y, y],
          line: { color, width: 2.5 }, opacity: op, showlegend: false,
          hoverinfo: 'text', text: `шаг ${h}, ${lv}%: ${y.toFixed(4)}`,
        });
      }
    }
  }
  return traces;
}

// ── ticker/interval switch init (called from app.js, same pattern as
// initTrendRulerForTicker/loadSimplexDefaults) ───────────────────────────
export async function initRangeForecastForTicker() {
  applyRangeForecastSettings(S.rangeForecastSettings); // hardcoded defaults — no per-ticker persistence for these (unlike band_lambda/simplex params), H/p/theiler are a "what to ask", not a "last used" preference worth round-tripping through the DB
  S.rangeForecastShowOnChart = getToolShowOnChart('range_forecast', S.rangeForecastShowOnChart);
  updateRangeForecastShowChartButton();
  await refreshRangeForecastCalibrationList();
  await recalcRangeForecastLive();
}
