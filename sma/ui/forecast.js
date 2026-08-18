import { S } from './state.js';
import { api, setStatus, connectTaskWS } from './api.js';
import { renderChart, computeGeometryFromWidth, LEVEL_OPTIONS, setOrigin } from './chart.js';
import { refreshTasks } from './tasks.js';
import { registerModelHandlers, loadAndRenderForecast } from './forecast_history.js';
import { registerTool } from './tools.js';
import { iconHtml } from './icons.js';

const MODEL_TYPE = 'band_lambda';

// Wires this model into the unified history/selection system (see
// forecast_history.js docstring) — when a band_lambda forecast becomes the
// selected one (marker/history-row click), keep the T-selector and zigzag
// overlay in sync with it, same as the old loadAndRenderBandForecast did.
registerModelHandlers('band_lambda', {
  async onSelected(f) {
    S.selectedTs = f.params.t_query;
    S.activeSettingsId = f.params.forecast_settings_id;
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
  label: 'band_lambda — полоса неопределённости',
  panelId: 'tool-panel-band_lambda',
  onOriginClick(ts) {
    setOrigin(ts);
    updateForecastButtonLabel();
  },
});

// ── forecast_settings / T-selector ──────────────────────────────────────────

export async function loadForecastSettings() {
  if (!S.instrumentId) return;
  try {
    S.forecastSettingsList = await api(
      'GET', `/forecast-settings?instrument_id=${S.instrumentId}&interval=${S.interval}&model_type=${MODEL_TYPE}`
    );
  } catch (_) {
    S.forecastSettingsList = [];
  }
  const byT = new Map();
  for (const s of S.forecastSettingsList) {
    if (!byT.has(s.t_query)) byT.set(s.t_query, []);
    byT.get(s.t_query).push(s);
  }
  S.calibratedTs = [...byT.keys()].sort((a, b) => a - b);
  if (S.selectedTs == null || !S.calibratedTs.includes(S.selectedTs)) {
    S.selectedTs = S.calibratedTs.at(-1) ?? null;
  }
  renderTSelector(byT);
  renderSettingsList();
  updateForecastButtonLabel();
  await loadZigzag();
}

function renderTSelector(byT) {
  const box = document.getElementById('bl-t-selector');
  const variantsBox = document.getElementById('bl-t-pool-variants');
  const btn = document.getElementById('bl-forecast-btn');

  if (!S.calibratedTs.length) {
    box.innerHTML = '<span class="muted-val">Нет калибровок — откройте «Калибровка» ниже</span>';
    variantsBox.innerHTML = '';
    btn.disabled = true;
    S.activeSettingsId = null;
    return;
  }

  box.innerHTML = '';
  const row = document.createElement('div');
  row.className = 'field-row';
  const select = document.createElement('select');
  select.style.flex = '1';
  S.calibratedTs.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t;
    opt.textContent = `${(t * 100).toFixed(0)}%`;
    if (t === S.selectedTs) opt.selected = true;
    select.appendChild(opt);
  });
  select.onchange = () => {
    S.selectedTs = +select.value;
    onSelectedTChange(byT); // reloads zigzag for the new T, which updates the button label itself
  };
  row.appendChild(select);
  box.appendChild(row);

  onSelectedTChange(byT);
  btn.disabled = false;
}

function onSelectedTChange(byT) {
  const variants = (byT.get(S.selectedTs) || []).sort((a, b) => b.is_active - a.is_active);
  const active = variants.find(v => v.is_active) || variants[0];
  S.activeSettingsId = active?.id ?? null;

  const variantsBox = document.getElementById('bl-t-pool-variants');
  if (variants.length <= 1) {
    variantsBox.innerHTML = '';
  } else {
    variantsBox.innerHTML = `<small class="hint">${variants.length} варианта пула для этого T:</small>`;
    variants.forEach(v => {
      const row = document.createElement('div');
      row.className = 'field-row';
      const label = v.pool_key + (v.calibration_meta?.pinball_rel != null
        ? ` (pinball ${v.calibration_meta.pinball_rel.toFixed(3)})` : '');
      row.innerHTML = `
        <label style="${v.is_active ? 'color:var(--text);font-weight:600' : ''}">${label}</label>
        ${v.is_active ? '<span class="muted-val">активен</span>' : `<button class="icon-btn" title="Сделать активным">${iconHtml('check')}</button>`}
      `;
      if (!v.is_active) {
        row.querySelector('button').onclick = () => activateSettings(v.id);
      }
      variantsBox.appendChild(row);
    });
  }

  loadZigzag();
}

export async function activateSettings(id) {
  try {
    await api('POST', `/forecast-settings/${id}/activate`);
    await loadForecastSettings();
  } catch (e) {
    setStatus(e.message, 'err');
  }
}

function renderSettingsList() {
  const box = document.getElementById('bl-settings-list');
  if (!S.forecastSettingsList.length) {
    box.innerHTML = '<span class="muted-val">—</span>';
    return;
  }
  box.innerHTML = '';
  [...S.forecastSettingsList].sort((a, b) => a.t_query - b.t_query).forEach(s => {
    const row = document.createElement('div');
    row.className = 'ticker-row';
    const rel = s.calibration_meta?.pinball_rel;
    row.innerHTML = `
      <div class="ticker-main">
        <div class="ticker-name">
          T=${(s.t_query * 100).toFixed(0)}% <span class="ticker-badge">${s.pool_key}</span>
          ${s.is_active ? '<span class="ticker-badge">активен</span>' : ''}
        </div>
        <div class="coverage-line">
          ${rel != null ? `pinball_rel=${rel.toFixed(3)}` : 'без метрики'} · min_bars=${s.min_bars} · m=${s.m} · θ=${s.theta}
        </div>
      </div>
      <button class="icon-btn danger" title="Удалить">${iconHtml('close')}</button>
    `;
    row.querySelector('button').onclick = () => removeSettings(s.id);
    box.appendChild(row);
  });
}

async function removeSettings(id) {
  if (!confirm('Удалить эту калибровку?')) return;
  try {
    await api('DELETE', `/forecast-settings/${id}`);
    await loadForecastSettings();
  } catch (e) {
    setStatus(e.message, 'err');
  }
}

// ── zigzag overlay ────────────────────────────────────────────────────────

// Guards against a stale response overwriting a newer one — without this,
// switching T quickly (e.g. 10% -> 20%) could race: if the 10% response
// arrives AFTER the 20% one, S.zigzagPivots silently reverts to the wrong
// T's pivots while the selector still reads "20%" (reported 2026-08-07:
// "выбран 20%, он прогнозирует в 10% по клику").
let _zigzagRequestToken = 0;

export async function loadZigzag() {
  const token = ++_zigzagRequestToken;
  const enabled = document.getElementById('bl-show-zigzag')?.checked;
  if (!S.instrumentId || S.selectedTs == null || !enabled) {
    if (token !== _zigzagRequestToken) return;
    S.zigzagPivots = [];
    redraw();
    return;
  }
  const settings = S.forecastSettingsList.find(s => s.id === S.activeSettingsId);
  const minBars = settings?.min_bars ?? 5;
  let pivots = [];
  try {
    const res = await api(
      'GET',
      `/forecasts/zigzag?instrument_id=${S.instrumentId}&interval=${S.interval}` +
      `&t_query=${S.selectedTs}&min_bars=${minBars}`
    );
    pivots = res.pivots || [];
  } catch (_) {
    pivots = [];
  }
  if (token !== _zigzagRequestToken) return; // superseded by a newer request — discard
  S.zigzagPivots = pivots;
  updateForecastButtonLabel();
  redraw();
}

export function toggleZigzagDisplay() {
  loadZigzag();
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

export async function submitForecast() {
  if (!S.instrumentId) { setStatus('Сначала загрузите свечи', 'err'); return; }
  if (!S.activeSettingsId) { setStatus('Сначала откалибруйте этот T', 'err'); return; }

  let originCandleId = null;
  const pivot = findOriginPivot();
  if (pivot) {
    const candle = S.candles.find(c => c.begin === pivot.confirm_date);
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
      forecast_settings_id: S.activeSettingsId,
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

// ── calibration targets (dynamic rows) ──────────────────────────────────────

let _targetRowId = 0;

export function addCalibTargetRow(tQuery = 0.20, minBars = 5) {
  const list = document.getElementById('bl-calib-targets-list');
  const id = `calib-target-${_targetRowId++}`;
  const row = document.createElement('div');
  row.className = 'field-row';
  row.id = id;
  row.innerHTML = `
    <label>T (%)</label>
    <input type="number" class="ct-t" value="${(tQuery * 100).toFixed(0)}" min="1" max="60" step="1" style="width:55px">
    <label>min_bars</label>
    <input type="number" class="ct-minbars" value="${minBars}" min="0" max="100" style="width:50px">
    <button class="icon-btn danger" title="Убрать">${iconHtml('close')}</button>
  `;
  row.querySelector('button').onclick = () => row.remove();
  list.appendChild(row);
}

function readCalibTargets() {
  return [...document.querySelectorAll('#bl-calib-targets-list .field-row')].map(row => ({
    t_query: +row.querySelector('.ct-t').value / 100,
    min_bars: +row.querySelector('.ct-minbars').value,
  }));
}

function readPoolSpec() {
  const categories = [...document.querySelectorAll('.bl-pool-cat:checked')].map(el => el.value);
  const n = +document.getElementById('bl-pool-n').value;
  return { categories, n };
}

export async function runCalibration() {
  if (!S.instrumentId) { setStatus('Сначала загрузите свечи', 'err'); return; }
  const pool = readPoolSpec();
  if (!pool.categories.length) { setStatus('Выберите хотя бы одну категорию пула', 'err'); return; }
  const targets = readCalibTargets();
  if (!targets.length) { setStatus('Добавьте хотя бы один T', 'err'); return; }

  const btn = document.getElementById('bl-calibrate-btn');
  btn.disabled = true;
  setStatus('Резолвинг пула…', 'busy');
  try {
    const task = await api('POST', '/forecast-settings/calibrate', {
      instrument_id: S.instrumentId, interval: S.interval, model_type: MODEL_TYPE,
      targets, m: +document.getElementById('bl-calib-m').value,
      theta: +document.getElementById('bl-calib-theta').value,
      pool,
    });
    setStatus(`Калибровка #${task.task_id} поставлена в очередь`, 'ok');
    refreshTasks();
    connectTaskWS(task.task_id, msg => {
      if (msg.status === 'running' && msg.total > 0) {
        setStatus(`Калибровка — ${msg.done}/${msg.total} T…`, 'busy');
      }
      if (msg.status === 'done') {
        setStatus('Калибровка завершена', 'ok');
        btn.disabled = false;
        loadForecastSettings();
      }
      if (msg.status === 'error') {
        setStatus(`Ошибка калибровки: ${msg.error}`, 'err');
        btn.disabled = false;
      }
      if (msg.status === 'cancelled') {
        setStatus('Калибровка отменена', 'err');
        btn.disabled = false;
      }
      refreshTasks();
    });
  } catch (e) {
    btn.disabled = false;
    setStatus(e.message, 'err');
  }
}

export async function runPretest() {
  if (!S.instrumentId) { setStatus('Сначала загрузите свечи', 'err'); return; }
  const pool = readPoolSpec();
  if (!pool.categories.length) { setStatus('Выберите хотя бы одну категорию пула', 'err'); return; }
  const targets = readCalibTargets();
  const tQuery = targets[0]?.t_query ?? 0.20;
  const minBars = targets[0]?.min_bars ?? 5;

  const resultBox = document.getElementById('bl-pretest-result');
  resultBox.innerHTML = '<span class="muted-val">Уровень 1 (быстрый, без скачивания)…</span>';
  try {
    const task = await api('POST', '/forecast-settings/pretest', {
      instrument_id: S.instrumentId, interval: S.interval,
      pool, t_query: tQuery, min_bars: minBars, level: 1,
    });
    refreshTasks();
    connectTaskWS(task.task_id, msg => {
      if (msg.status === 'done') renderPretestSummary(msg.summary);
      if (msg.status === 'error') resultBox.innerHTML = `<span class="muted-val">${msg.error}</span>`;
      refreshTasks();
    });
  } catch (e) {
    resultBox.innerHTML = `<span class="muted-val">${e.message}</span>`;
  }
}

function renderPretestSummary(summary) {
  const box = document.getElementById('bl-pretest-result');
  if (summary.level === 1) {
    box.innerHTML = `
      <div class="result-row"><span class="result-key">Кандидатов</span><span class="result-val">${summary.n_candidates}</span></div>
      <div class="result-row"><span class="result-key">С историей</span><span class="result-val">${summary.n_resolved}</span></div>
      <div class="result-row"><span class="result-key">Лет истории (сумм./сред.)</span><span class="result-val">${summary.total_years} / ${summary.avg_years}</span></div>
      <button style="width:100%;margin-top:6px" onclick="runPretestLevel2()">Точнее (уровень 2, со скачиванием)</button>
    `;
  } else {
    box.innerHTML = `
      <div class="result-row"><span class="result-key">Тикеров</span><span class="result-val">${summary.n_tickers}</span></div>
      <div class="result-row"><span class="result-key">Событий (сумм./сред.)</span><span class="result-val">${summary.total_events} / ${summary.avg_events}</span></div>
    `;
  }
}

export async function runPretestLevel2() {
  const pool = readPoolSpec();
  const targets = readCalibTargets();
  const tQuery = targets[0]?.t_query ?? 0.20;
  const minBars = targets[0]?.min_bars ?? 5;
  const box = document.getElementById('bl-pretest-result');
  box.innerHTML = '<span class="muted-val">Уровень 2 (загрузка + подсчёт событий)…</span>';
  try {
    const task = await api('POST', '/forecast-settings/pretest', {
      instrument_id: S.instrumentId, interval: S.interval,
      pool, t_query: tQuery, min_bars: minBars, level: 2,
    });
    refreshTasks();
    connectTaskWS(task.task_id, msg => {
      if (msg.status === 'running' && msg.total > 0) {
        box.innerHTML = `<span class="muted-val">Загрузка данных пула — ${msg.done}/${msg.total}…</span>`;
      }
      if (msg.status === 'done') renderPretestSummary(msg.summary);
      if (msg.status === 'error') box.innerHTML = `<span class="muted-val">${msg.error}</span>`;
      refreshTasks();
    });
  } catch (e) {
    box.innerHTML = `<span class="muted-val">${e.message}</span>`;
  }
}

// ── display settings: levels (checkboxes), opacity, width, presets ─────────
// Levels/opacity match prototype/forcaster/ui/app9.py exactly (LEVEL_OPTIONS
// checkboxes, fixed per-level opacity — see chart.js:buildBandZoneShapes).
// Width has no prototype equivalent — replaces an unreliable drag-to-resize
// attempt (project feedback 2026-08-07); it is NOT part of a display_preset
// (those only ever held levels/opacity) — it edits the current forecast's
// zone_geometry, same persistence path a drag would have used.

export function renderLevelCheckboxes() {
  const box = document.getElementById('bl-display-levels-box');
  if (!box) return;
  box.innerHTML = LEVEL_OPTIONS.map(lvl => `
    <label style="margin-right:10px"><input type="checkbox" class='bl-display-level' value="${lvl}"
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
// "Применить" both updates the current display AND persists these settings
// as the one default, auto-loaded next time (project feedback 2026-08-08:
// "пускай будет один дефолтный пресет по сути, настройки которого
// сохраняются кнопкой применить"). Ширина шага (applyBandWidth) is
// intentionally NOT part of this — it stays per-forecast (zone_geometry).
const DISPLAY_PRESET_NAME = '__default__';

export async function applyDisplaySettings() {
  const levels = readDisplayLevels();
  const opacity = +document.getElementById('bl-display-opacity').value;
  if (!levels.length) { setStatus('Включите хотя бы один уровень', 'err'); return; }
  S.displayPreset = { levels, opacity };
  renderChart({ preserveRange: true });
  try {
    await api('POST', '/display-presets', {
      model_type: MODEL_TYPE, name: DISPLAY_PRESET_NAME,
      levels, opacity, is_default: true,
    });
    setStatus('Настройки отображения сохранены', 'ok');
  } catch (e) {
    setStatus(e.message, 'err');
  }
}

let _geometrySaveTimer = null;

// Only meaningful while the currently SELECTED forecast (S.selectedForecastId)
// is a band_lambda one — see S.renderedForecasts (state.js) for the shared
// cache this reads/writes zone_geometry on.
export function applyBandWidth() {
  const widthBars = +document.getElementById('bl-band-width-slider').value;
  S.bandWidthBars = widthBars;
  const label = document.getElementById('bl-band-width-label');
  if (label) label.textContent = `${widthBars} бар.`;

  const id = S.selectedForecastId;
  const entry = id != null ? S.renderedForecasts.get(id) : null;
  if (!entry || entry.model_type !== 'band_lambda') return;

  const geometry = computeGeometryFromWidth(entry.result.origin_extreme_date, S.interval, widthBars);
  entry.zone_geometry = geometry;
  renderChart({ preserveRange: true });
  clearTimeout(_geometrySaveTimer);
  _geometrySaveTimer = setTimeout(() => {
    api('POST', `/forecasts/${id}/geometry`, { zone_geometry: geometry }).catch(() => {});
  }, 500);
}

export async function loadDisplayPresets() {
  renderLevelCheckboxes();
  try {
    const presets = await api('GET', `/display-presets?model_type=${MODEL_TYPE}`);
    const def = presets.find(p => p.name === DISPLAY_PRESET_NAME) ?? presets.find(p => p.is_default);
    if (def) {
      S.displayPreset = { levels: def.levels, opacity: def.opacity };
      setDisplayLevelCheckboxes(def.levels);
      document.getElementById('bl-display-opacity').value = def.opacity;
    }
  } catch (_) { /* non-fatal — falls back to the hardcoded UI defaults */ }
}
