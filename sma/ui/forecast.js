import { S } from './state.js';
import { api, setStatus, connectTaskWS } from './api.js';
import { renderChart, computeGeometryFromWidth, LEVEL_OPTIONS, setOrigin, DEFAULT_STEP_BARS } from './chart.js';
import { refreshTasks } from './tasks.js';
import { registerModelHandlers, loadAndRenderForecast } from './forecast_history.js';
import { registerTool } from './tools.js';
import { iconHtml } from './icons.js';
import { saveZigzagPinnedTs } from './local_prefs.js';

const MODEL_TYPE = 'band_lambda';

// Wires this model into the unified history/selection system (see
// forecast_history.js docstring) — when a band_lambda forecast becomes the
// selected one (marker/history-row click), reflect its T/m/theta back into
// the spinner/inputs and keep the zigzag overlay in sync with it.
registerModelHandlers('band_lambda', {
  async onSelected(f) {
    applyBandLambdaParams(f.params);
    syncBandWidthSliders(f);
    await loadZigzag();
    updateForecastButtonLabel();
  },
});

// Registered as a "forecaster"-group tool — see tools.js module docstring
// for why forecast models share the SAME toolbar/exclusivity mechanism as
// analyzers (S.activeMainTool) instead of a separate one: only one click
// target can be armed at a time regardless of kind. No buildMainTraces —
// this tool's actual chart presence flows through the pre-existing pin/
// select overlay system (chart.js:buildVisibleOverlays + MODEL_DISPLAY),
// registered here purely for toolbar placement + click-target arbitration.
registerTool({
  type: 'band_lambda',
  surface: 'main',
  group: 'forecaster',
  icon: 'band',
  label: 'Band Lambda — полоса неопределённости',
  panelId: 'tool-panel-band_lambda',
  onOriginClick: setOrigin,
  onMainOriginChanged: updateForecastButtonLabel,
});

// ── band_lambda params (T/m/theta — free live parameters, see state.js) ────
// No more per-T saved rows (λ-calibration removed 2026-09-12, see memory
// project_phase7_calibration_removed_final) — T is a plain spinner, m/theta
// plain inputs; all three persist only via forecast_defaults (last-used,
// same mechanism simplex_ensemble already uses), not a settings table.

function applyBandLambdaParams(p) {
  S.selectedTs = p.t_query;
  S.bandLambdaModelParams = { m: p.m, theta: p.theta };
  const tInput = document.getElementById('bl-t-input');
  if (tInput) tInput.value = Math.round(p.t_query * 100);
  const mInput = document.getElementById('bl-calib-m');
  if (mInput) mInput.value = p.m;
  const thetaInput = document.getElementById('bl-calib-theta');
  if (thetaInput) thetaInput.value = p.theta;
  updateZigzagPinButton();
}

// Last-used T/m/theta for this (instrument, interval) — falls back to
// whatever's currently in S.bandLambdaModelParams/S.selectedTs (itself
// initialized from the hardcoded state.js defaults) when nothing is saved
// yet, same pattern simplex_ensemble.js:loadSimplexDefaults uses.
export async function loadBandLambdaDefaults() {
  if (!S.instrumentId) return;
  let params = { t_query: S.selectedTs, ...S.bandLambdaModelParams };
  try {
    const res = await api(
      'GET', `/forecast-settings/defaults?instrument_id=${S.instrumentId}&interval=${S.interval}&model_type=${MODEL_TYPE}`
    );
    if (res.params) params = res.params;
  } catch (_) { /* fall back to current/hardcoded defaults */ }
  applyBandLambdaParams(params);
  await loadBandLambdaPool();
  await loadZigzag();
}

// ── pool summary (sidebar "Пул" section — full editing lives in the modal,
// band_pool_modal.js) ───────────────────────────────────────────────────────

export async function loadBandLambdaPool() {
  if (!S.instrumentId) { S.bandLambdaPool = null; renderPoolSummary(); return; }
  try {
    const res = await api(
      'GET', `/forecast-settings/pool?instrument_id=${S.instrumentId}&interval=${S.interval}&model_type=${MODEL_TYPE}`
    );
    S.bandLambdaPool = res.pool;
  } catch (_) {
    S.bandLambdaPool = null;
  }
  renderPoolSummary();
}

export function renderPoolSummary() {
  const box = document.getElementById('bl-pool-summary');
  if (!box) return;
  box.innerHTML = S.bandLambdaPool
    ? `${S.bandLambdaPool.resolved_tickers.length} тикеров <span class="ticker-badge">${S.bandLambdaPool.pool_key}</span>`
    : '<span class="muted-val">не настроен</span>';
}

// T input onchange (index.html) — reloads the zigzag overlay for the new T,
// which updates the forecast button label itself; no persistence here (only
// a completed forecast writes forecast_defaults, see task_manager.py:
// _run_forecast_band_lambda).
export function onBandTChange() {
  const input = document.getElementById('bl-t-input');
  const pct = +input.value;
  if (!pct || pct <= 0) return;
  S.selectedTs = pct / 100;
  updateZigzagPinButton();
  loadZigzag();
}

// m/theta inputs onchange (index.html) — no zigzag/chart effect (only T
// does), just keeps S.bandLambdaModelParams in sync with what's actually
// typed so submitForecast reads the current values, not stale ones.
export function onBandModelParamsChange() {
  S.bandLambdaModelParams = {
    m: +document.getElementById('bl-calib-m').value,
    theta: +document.getElementById('bl-calib-theta').value,
  };
}

function updateZigzagPinButton() {
  document.getElementById('bl-zigzag-pin-btn')
    ?.classList.toggle('active', S.zigzagPinnedTs.has(S.selectedTs));
}

// Pins/unpins the CURRENTLY selected T (S.selectedTs) — reuses whatever's
// already in S.zigzagPivots (already fetched for this T by loadZigzag) as
// the pinned cache instead of a second network round trip; the data is
// identical either way (same endpoint/params), so copying it is just as
// correct and free. See buildForecastZigzagTraces (chart.js) for how a
// pinned-but-not-selected T's data gets drawn from this cache once the
// selector moves to a different T.
export function toggleZigzagPin() {
  const t = S.selectedTs;
  if (t == null) return;
  if (S.zigzagPinnedTs.has(t)) {
    S.zigzagPinnedTs.delete(t);
    delete S.zigzagPinnedData[t];
  } else {
    S.zigzagPinnedTs.add(t);
    S.zigzagPinnedData[t] = S.zigzagPivots;
  }
  updateZigzagPinButton();
  renderZigzagPinnedList();
  saveZigzagPinnedTs(S.zigzagPinnedTs);
  redraw();
}

function unpinZigzagT(t) {
  S.zigzagPinnedTs.delete(t);
  delete S.zigzagPinnedData[t];
  updateZigzagPinButton();
  renderZigzagPinnedList();
  saveZigzagPinnedTs(S.zigzagPinnedTs);
  redraw();
}

function renderZigzagPinnedList() {
  const box = document.getElementById('bl-zigzag-pinned-list');
  if (!box) return;
  if (!S.zigzagPinnedTs.size) {
    box.innerHTML = '';
    return;
  }
  box.innerHTML = [...S.zigzagPinnedTs].sort((a, b) => a - b).map(t => `
    <span class="zigzag-pin-chip">
      T=${(t * 100).toFixed(0)}%
      <button class="icon-btn danger" title="Снять закрепление" data-unpin-t="${t}" style="padding:0">${iconHtml('close')}</button>
    </span>
  `).join('');
  box.querySelectorAll('[data-unpin-t]').forEach(btn => {
    btn.onclick = () => unpinZigzagT(+btn.dataset.unpinT);
  });
}

// Called on ticker switch (app.js:loadCandles) — the pinned T's pivot cache
// is tied to the previous instrument's price/date scale, meaningless (and
// actively misleading if drawn) once the chart switches to a different one.
export function resetZigzagPins() {
  S.zigzagPinnedTs.clear();
  S.zigzagPinnedData = {};
  updateZigzagPinButton();
  renderZigzagPinnedList();
  saveZigzagPinnedTs(S.zigzagPinnedTs);
}

// ── zigzag overlay ────────────────────────────────────────────────────────

// Guards against a stale response overwriting a newer one — without this,
// switching T quickly (e.g. 10% -> 20%) could race: if the 10% response
// arrives AFTER the 20% one, S.zigzagPivots silently reverts to the wrong
// T's pivots while the selector still reads "20%" (reported 2026-08-07:
// "выбран 20%, он прогнозирует в 10% по клику").
let _zigzagRequestToken = 0;

// Fetches unconditionally whenever a T is selected — no manual "показать
// зигзаг" toggle any more (project feedback 2026-08-18: "Своя кнопка
// «показать зиг-заг на графике» не нужна"; §2.4 of docs/plans/
// frontend_improvements_plan.md). Chart VISIBILITY is a separate concern
// handled entirely by chart.js:buildForecastZigzagTraces (active tool ∪
// pinned T's) — this function's only job is keeping S.zigzagPivots (and, if
// this T happens to be pinned, its S.zigzagPinnedData entry) up to date.
export async function loadZigzag() {
  const token = ++_zigzagRequestToken;
  if (!S.instrumentId || S.selectedTs == null) {
    if (token !== _zigzagRequestToken) return;
    S.zigzagPivots = [];
    redraw();
    return;
  }
  let pivots = [];
  try {
    // min_bars omitted — server default (bl.DEFAULT_MIN_BARS) applies; it's
    // no longer a saved per-T value (λ-calibration removed 2026-09-12).
    const res = await api(
      'GET',
      `/forecasts/zigzag?instrument_id=${S.instrumentId}&interval=${S.interval}&t_query=${S.selectedTs}`
    );
    pivots = res.pivots || [];
  } catch (_) {
    pivots = [];
  }
  if (token !== _zigzagRequestToken) return; // superseded by a newer request — discard
  S.zigzagPivots = pivots;
  if (S.zigzagPinnedTs.has(S.selectedTs)) S.zigzagPinnedData[S.selectedTs] = pivots; // keep a pinned T's cache fresh, not just its first snapshot
  updateForecastButtonLabel();
  redraw();
}

function redraw() {
  renderChart({ preserveRange: true });
}

// ── origin resolution — clicking the chart no longer "selects" a pivot ─────
// The click just places the ordinary origin marker (setOrigin, shared with
// the Analysis tab). Pressing "Прогноз" then picks the last zigzag event at
// or before that marker — exactly on it if the marker sits on the event,
// otherwise the closest one before it (project feedback 2026-08-07: "если
// маркер стоит прям на событии - то от него. Если нет, то последнее до
// маркера"). No marker placed yet -> null -> submitForecast falls back to
// "latest" (the live forecast).
//
// ⚠️ Matched against extreme_date, NOT confirm_date — extreme_date is what
// the zigzag actually DRAWS on the chart (see chart.js:buildZigzagTrace),
// so it's the only date the user can see and click near. Matching against
// confirm_date instead was a real bug (2026-08-08): confirmation can lag
// the visible pivot by weeks (see band_lambda.py:pool_values_and_weights
// docstring), so a marker placed "a few bars after the event I can see"
// often landed BEFORE that pivot's actual confirm_date and silently fell
// back to the PREVIOUS pivot instead. The forecast itself still anchors on
// confirm_date exactly as before (submitForecast below) — only the search
// that decides WHICH pivot the user meant changed; causality is unaffected
// either way since the origin_candle_id sent to the backend is always the
// confirmation bar.
function findOriginPivot() {
  if (!S.originTs || !S.zigzagPivots.length) return null;
  const markerDate = S.originTs.slice(0, 10);
  const candidates = S.zigzagPivots.filter(p => p.extreme_date.slice(0, 10) <= markerDate);
  return candidates.length ? candidates[candidates.length - 1] : null;
}

export function updateForecastButtonLabel() {
  const btn = document.getElementById('bl-forecast-btn');
  if (!btn) return;
  const pivot = findOriginPivot();
  if (!pivot) {
    btn.textContent = '▶ Прогноз (последнее событие)';
    return;
  }
  const ext = pivot.extreme_date.slice(0, 10);
  const conf = pivot.confirm_date.slice(0, 10);
  // Surface the confirmation lag right on the button — the exact
  // information whose absence caused the bug above.
  btn.textContent = ext === conf
    ? `▶ Прогноз (событие ${ext})`
    : `▶ Прогноз (событие ${ext}, подтв. ${conf})`;
}

// ── submit / progress ────────────────────────────────────────────────────

// `until` is inclusive and `limit` takes the last row ≤ begin, so one row
// back is the bar itself when it exists.
async function _findCandleByBegin(begin) {
  const rows = await api('GET',
    `/candles?ticker=${S.ticker}&data_source=${S.dataSource}&interval=${S.interval}&until=${begin}&limit=1`);
  return rows.find(r => r.begin === begin) ?? null;
}

export async function submitForecast() {
  if (!S.instrumentId) { setStatus('Сначала загрузите свечи', 'err'); return; }
  if (S.selectedTs == null) { setStatus('Укажите T (порог зигзага)', 'err'); return; }

  let originCandleId = null;
  const pivot = findOriginPivot();
  if (pivot) {
    // S.candles holds only the loaded window (sma/ui/candle_window.js) — an
    // older pivot may sit outside it, so ask the backend for that exact bar.
    const candle = S.candles.find(c => c.begin === pivot.confirm_date) ?? await _findCandleByBegin(pivot.confirm_date);
    if (!candle) { setStatus('Не найден бар для события — обновите зигзаг', 'err'); return; }
    originCandleId = candle.id;
  }

  const btn = document.getElementById('bl-forecast-btn');
  btn.disabled = true;
  setStatus('Создание задачи…', 'busy');
  try {
    const body = {
      instrument_id: S.instrumentId, interval: S.interval,
      model_type: MODEL_TYPE, t_query: S.selectedTs,
      m: S.bandLambdaModelParams.m, theta: S.bandLambdaModelParams.theta,
    };
    if (originCandleId != null) body.origin_candle_id = originCandleId;
    const task = await api('POST', '/forecasts', body);
    btn.disabled = false;
    setStatus(`Задача #${task.task_id} поставлена в очередь`, 'ok');
    refreshTasks(); // immediate task-manager refresh — progress lives there now (#task-status/#task-list)
    connectTaskWS(task.task_id, msg => {
      if (msg.status === 'running' && msg.total > 0) {
        setStatus(`Задача #${task.task_id} — ${Math.round(msg.done * 100 / msg.total)}%…`, 'busy');
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

// ── display settings: levels (checkboxes), граница шага 2, opacity, width,
// presets ────────────────────────────────────────────────────────────────
// Levels/opacity/trade-level/toggles persist together as one display_preset
// (see chart.js:buildBandZoneShapes for how they're consumed). Width has no
// preset equivalent — it edits the current forecast's zone_geometry, same
// persistence path a drag would have used.

export function renderLevelCheckboxes() {
  const box = document.getElementById('bl-display-levels-box');
  if (!box) return;
  box.innerHTML = LEVEL_OPTIONS.map(lvl => `
    <label><input type="checkbox" class='bl-display-level' value="${lvl}"
      onchange="applyDisplaySettings()"
      ${S.displayPreset.levels.includes(lvl) ? 'checked' : ''}> ${lvl}%</label>
  `).join('');
}

function readDisplayLevels() {
  return [...document.querySelectorAll('.bl-display-level:checked')].map(el => +el.value);
}

function setDisplayLevelCheckboxes(levels) {
  document.querySelectorAll('.bl-display-level').forEach(el => {
    el.checked = levels.includes(+el.value);
  });
}

// Single named preset ("__default__") — no picker, no per-name save prompt.
// Applies immediately (levels/opacity checkbox+slider call this directly on
// change/input, see index.html) and persists it as the one default,
// auto-loaded next time — a separate "Применить" button used to gate both of
// these on an explicit click, but that read as redundant once every other
// display setting in the app applies live (project feedback 2026-08-19,
// §2.5 of docs/plans/frontend_improvements_plan.md). The render itself stays
// synchronous/instant; only the persistence POST is debounced (same
// fire-and-forget 500ms pattern analysis_settings already uses, see
// docs/plans/trend_variance_analyzer_migration_plan.md §3.1) so dragging the
// opacity slider doesn't fire a request per animation frame.
const DISPLAY_PRESET_NAME = '__default__';
let _displayPresetSaveTimer = null;

export function applyDisplaySettings() {
  const levels = readDisplayLevels();
  const opacity = +document.getElementById('bl-display-opacity').value;
  if (!levels.length) { setStatus('Включите хотя бы один уровень', 'err'); return; }
  const tradeLevelPct = +document.getElementById('bl-trade-level-pct').value;
  const showZones = document.getElementById('bl-show-zones').checked;
  const showTradeLevel = document.getElementById('bl-show-trade-level').checked;
  const trimZone1 = document.getElementById('bl-trim-zone1').checked;
  S.displayPreset = { levels, opacity, tradeLevelPct, showZones, showTradeLevel, trimZone1 };
  renderChart({ preserveRange: true });

  clearTimeout(_displayPresetSaveTimer);
  _displayPresetSaveTimer = setTimeout(async () => {
    try {
      await api('POST', '/display-presets', {
        model_type: MODEL_TYPE, name: DISPLAY_PRESET_NAME,
        levels, opacity, is_default: true,
        trade_level_pct: tradeLevelPct, show_zones: showZones,
        show_trade_level: showTradeLevel, trim_zone1: trimZone1,
      });
    } catch (e) {
      setStatus(e.message, 'err');
    }
  }, 500);
}

let _geometrySaveTimer = null;

// Only meaningful while the currently SELECTED forecast (S.selectedForecastId)
// is a band_lambda one — see S.renderedForecasts (state.js) for the shared
// cache this reads/writes zone_geometry on. Two independent sliders (project
// feedback 2026-08-26: up/down leg durations are asymmetric enough that one
// shared "Ширина шага" no longer makes sense) — each reads BOTH current
// slider values so either one can be dragged without resetting the other.
export function applyBandWidth() {
  const width1Bars = +document.getElementById('bl-band-width1-slider').value;
  const width2Bars = +document.getElementById('bl-band-width2-slider').value;
  setBandWidthLabels(width1Bars, width2Bars);

  const id = S.selectedForecastId;
  const entry = id != null ? S.renderedForecasts.get(id) : null;
  if (!entry || entry.model_type !== 'band_lambda') return;

  const geometry = computeGeometryFromWidth(entry.result.origin_extreme_date, S.interval, width1Bars, width2Bars);
  entry.zone_geometry = geometry;
  renderChart({ preserveRange: true });
  clearTimeout(_geometrySaveTimer);
  _geometrySaveTimer = setTimeout(() => {
    api('POST', `/forecasts/${id}/geometry`, { zone_geometry: geometry }).catch(() => {});
  }, 500);
}

function setBandWidthLabels(width1Bars, width2Bars) {
  const l1 = document.getElementById('bl-band-width1-label');
  const l2 = document.getElementById('bl-band-width2-label');
  if (l1) l1.textContent = `${width1Bars} бар.`;
  if (l2) l2.textContent = `${width2Bars} бар.`;
}

// Restores the two width sliders to whatever the just-selected forecast
// actually carries — the persisted zone_geometry's own width1_bars/
// width2_bars if the user (or a past auto-default) already set one,
// otherwise the ticker-specific auto-default from
// result.default_step_widths_bars, otherwise the fixed fallback. Called
// from band_lambda's onSelected handler below; without this the sliders
// kept showing whatever the PREVIOUSLY selected forecast left them at.
function syncBandWidthSliders(f) {
  const geo = f.zone_geometry;
  const defaults = f.result.default_step_widths_bars;
  const width1Bars = geo?.width1_bars ?? defaults?.step1 ?? DEFAULT_STEP_BARS;
  const width2Bars = geo?.width2_bars ?? defaults?.step2 ?? DEFAULT_STEP_BARS;
  const s1 = document.getElementById('bl-band-width1-slider');
  const s2 = document.getElementById('bl-band-width2-slider');
  if (s1) s1.value = width1Bars;
  if (s2) s2.value = width2Bars;
  setBandWidthLabels(width1Bars, width2Bars);
}

export async function loadDisplayPresets() {
  renderLevelCheckboxes();
  try {
    const presets = await api('GET', `/display-presets?model_type=${MODEL_TYPE}`);
    const def = presets.find(p => p.name === DISPLAY_PRESET_NAME) ?? presets.find(p => p.is_default);
    if (def) {
      S.displayPreset = {
        levels: def.levels, opacity: def.opacity,
        tradeLevelPct: def.trade_level_pct, showZones: def.show_zones,
        showTradeLevel: def.show_trade_level, trimZone1: def.trim_zone1,
      };
      setDisplayLevelCheckboxes(def.levels);
      document.getElementById('bl-display-opacity').value = def.opacity;
      document.getElementById('bl-trade-level-pct').value = def.trade_level_pct;
      document.getElementById('bl-show-zones').checked = def.show_zones;
      document.getElementById('bl-show-trade-level').checked = def.show_trade_level;
      document.getElementById('bl-trim-zone1').checked = def.trim_zone1;
    }
  } catch (_) { /* non-fatal — falls back to the hardcoded UI defaults */ }
}
