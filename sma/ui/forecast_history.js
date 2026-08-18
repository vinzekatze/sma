// forecast_history.js — forecast history/results, one list+result panel PER
// MODEL now (project feedback 2026-08-18, round 6: "Отображают только
// индивидуальную историю прогнозов") — was a single list mixing all models
// with just a color badge to tell them apart; each forecast model is its
// own toolbar tool now (see tools.js), so its own history/results live
// inside its own panel (#history-<model_type>/#results-<model_type>).
// select (temporary focus — drives the chart overlay + the owning model's
// results div), pin ("eye" — session-only simultaneous display, see
// S.pinnedForecastIds in state.js), delete. The generic counterpart of
// forecast.js's/simplex_ensemble.js's model-specific submit flows — see
// chart.js's MODEL_DISPLAY for the rendering side of the same "one
// registry, per-model plugin" idea.
//
// Deliberately does NOT import forecast.js/simplex_ensemble.js — both of
// those import refreshHistory() FROM here after a submit completes, so the
// reverse import would cycle. Model-specific "a forecast of my type got
// selected, sync my own UI state" side effects (band_lambda's T-selector/
// zigzag, simplex_ensemble's params form) are wired from app.js via
// registerModelHandlers, mirroring how chart.js's initChartEvents keeps
// chart.js itself decoupled from forecast.js. Importing tools.js is safe
// though (no cycle — tools.js imports nothing from here), needed to make
// the owning model's tool panel visible when a forecast gets selected.

import { S, DAY_RU } from './state.js';
import { api, setIdle } from './api.js';
import { renderChart } from './chart.js';
import { activateTool } from './tools.js';
import { iconHtml } from './icons.js';

// Every model this history/results system knows about — kept as a small
// local list (not derived from tools.js's TOOLS, which also holds
// non-forecast tools) since this file only ever needs the two model_type
// strings themselves, to fetch/render one history+results pair per model.
const MODEL_TYPES = ['band_lambda', 'simplex_ensemble'];

const _modelHandlers = {}; // modelType -> {onSelected(f)}

// Registered once at startup (app.js) — e.g. band_lambda wants selecting one
// of its forecasts to also update S.selectedTs/activeSettingsId and reload
// the zigzag overlay; simplex_ensemble wants its params form refilled.
export function registerModelHandlers(modelType, handlers) {
  _modelHandlers[modelType] = handlers;
}

// Fetches each model's history SEPARATELY (own model_type filter, own
// limit) rather than one shared unbounded-mix fetch — otherwise one
// model's recent activity could crowd the other's out of a shared limit
// (project feedback 2026-08-18: "Отображают только индивидуальную
// историю"). Merged back into ONE S.historyForecasts array afterwards —
// chart.js's buildForecastMarkerTraces already groups markers by
// model_type into separate traces regardless of how the array was
// assembled, and app.js's click routing looks up a marker's model_type
// from this same array — no reason to keep two separate state arrays.
export async function refreshHistory() {
  if (!S.instrumentId) return;
  const perModel = await Promise.all(MODEL_TYPES.map(async modelType => {
    try {
      const list = await api(
        'GET',
        `/forecasts?instrument_id=${S.instrumentId}&interval=${S.interval}&model_type=${modelType}&limit=20`
      );
      return list.filter(f => f.origin_ts).map(f => ({
        id: f.id, model_type: f.model_type,
        origin_ts: f.origin_ts, origin_extreme_ts: f.origin_extreme_ts,
        origin_price: f.origin_price,
        origin_direction: f.origin_direction, is_stale: !!f.is_stale, created_at: f.created_at,
      }));
    } catch (_) {
      return [];
    }
  }));
  S.historyForecasts = perModel.flat();
  for (const modelType of MODEL_TYPES) renderHistoryList(modelType);
}

function renderHistoryList(modelType) {
  const box = document.getElementById(`history-${modelType}`);
  if (!box) return;
  const list = S.historyForecasts.filter(f => f.model_type === modelType);
  if (!list.length) {
    box.innerHTML = '<span class="muted-val">Пока нет прогнозов этой модели</span>';
    return;
  }
  box.innerHTML = list.map(f => {
    // origin_extreme_ts (matches the chart marker's date) — falls back to
    // origin_ts for old rows saved before this field existed.
    const dateForDisplay = f.origin_extreme_ts || f.origin_ts;
    const d = new Date(dateForDisplay.slice(0, 10) + 'T12:00:00');
    const originDay = DAY_RU[d.getDay()] + ' ' + dateForDisplay.slice(0, 10);
    const active = f.id === S.selectedForecastId ? ' style="background:rgba(88,166,255,0.08)"' : '';
    const dir = f.origin_direction > 0 ? '▲' : '▼';
    const pinned = S.pinnedForecastIds.has(f.id);
    return `
      <div class="hist-item" onclick="loadAndRenderForecast(${f.id})"${active}>
        <div class="hist-item-header">
          <span class="hist-date">${originDay}</span>
          ${f.is_stale ? '<span class="hist-stale">⚠ устарел</span>' : ''}
          <button class="hist-eye-btn${pinned ? ' active' : ''}" onclick="toggleForecastPin(${f.id}, event)"
            title="${pinned ? 'Снять с отображения' : 'Закрепить на графике (показывать вместе с другими)'}">${iconHtml('eye')}</button>
          <button class="hist-del-btn" onclick="deleteForecast(${f.id}, event)" title="Удалить">${iconHtml('close')}</button>
        </div>
        <div><span class="hist-price">${dir} ${f.origin_price.toFixed(4)}</span></div>
        <div style="color:var(--muted);font-size:10px">${(f.created_at || '').slice(0, 16).replace('T', ' ')}</div>
      </div>
    `;
  }).join('');
}

// Generic entry point — dispatched to by window.loadAndRenderForecast
// (registered in app.js), called from .hist-item onclick (switchTool
// defaults true — you clicked directly inside that model's OWN panel, so
// activating it is a harmless no-op at worst) AND from chart.js's
// marker-click callback via app.js (switchTool: false — project feedback
// round 8: "выбор объектов не должно просто происходить переключение на
// режим прогноза при клике на прогноз... это отдельные режимы"; a marker
// click must never steal the active tool away from whatever the user
// currently has selected, 'object_select' included). Caches into
// S.renderedForecasts so re-selecting (or pinning) an already-loaded
// forecast doesn't refetch.
export async function loadAndRenderForecast(forecastId, { switchTool = true } = {}) {
  try {
    const cached = S.renderedForecasts.get(forecastId);
    let f;
    if (cached) {
      f = { model_type: cached.model_type, result: cached.result, params: cached.params, zone_geometry: cached.zone_geometry };
    } else {
      f = await api('GET', `/forecasts/${forecastId}`);
      S.renderedForecasts.set(forecastId, {
        model_type: f.model_type, result: f.result, params: f.params, zone_geometry: f.zone_geometry,
      });
    }

    S.selectedForecastId = forecastId;
    S.originTs = f.result.origin_date.slice(0, 10);

    if (switchTool) activateTool(f.model_type); // show the owning model's panel (toolbar tool, not just a display div)
    await _modelHandlers[f.model_type]?.onSelected?.(f);

    await refreshHistory();
    renderChart({ preserveRange: true });
    showResultsFor(f);
    setIdle(`Прогноз готов  •  ${f.model_type}  •  ${f.result.origin_extreme_date.slice(0, 10)}`);
  } catch (e) {
    setIdle(e.message, false);
  }
}

// Renders into the CURRENTLY ACTIVE tool's own results area — normally
// that's the forecast's own model_type (switchTool just activated it, or
// it was already active), but while 'object_select' is the active tool
// (round 8 fix above — a marker click never switches tools), results go
// into object_select's OWN results area instead (#results-object_select,
// see index.html) so inspecting a forecast never requires leaving
// object_select mode.
function showResultsFor(f) {
  const target = S.activeMainTool === 'object_select' ? 'object_select' : f.model_type;
  const content = document.getElementById(`results-${target}`);
  const section = document.getElementById(`results-section-${target}`);
  if (!content || !section) return;
  content.innerHTML = f.model_type === 'simplex_ensemble' ? _simplexResultsHtml(f) : _bandResultsHtml(f);
  section.style.display = '';
  if (target === 'object_select') {
    document.getElementById('object-select-pin-btn')
      ?.classList.toggle('active', S.pinnedForecastIds.has(S.selectedForecastId));
  }
}

function _bandResultsHtml(f) {
  const r = f.result;
  const dirLabel = r.origin_direction > 0 ? '▲ HIGH' : '▼ LOW';
  const dirClass = r.origin_direction > 0 ? 'up' : 'down';

  const stepRow = (h, label) => {
    const s = r.steps[String(h)] ?? r.steps[h];
    if (!s?.ok) return `<div class="result-row"><span class="result-key">${label}</span><span class="result-val">пул ${s?.pool_size ?? 0} — недостаточно</span></div>`;
    const b = s.band_price;
    const keys = Object.keys(b).map(Number).sort((a, c) => a - c);
    const lo = b[keys[0]], hi = b[keys.at(-1)], mid = b[keys[Math.floor(keys.length / 2)]];
    const isDown = h === 1 ? r.origin_direction > 0 : r.origin_direction < 0;
    return `<div class="result-row">
      <span class="result-key">${label} (пул ${s.pool_size})</span>
      <span class="result-val ${isDown ? 'down' : 'up'}">${lo.toFixed(4)} … ${mid.toFixed(4)} … ${hi.toFixed(4)}</span>
    </div>`;
  };

  return `
    <div class="result-row">
      <span class="result-key">Событие (origin)</span>
      <span class="result-val ${dirClass}">${r.origin_price.toFixed(4)}  ${dirLabel}</span>
    </div>
    <div class="result-row">
      <span class="result-key">Дата события</span>
      <span class="result-val">${r.origin_extreme_date.slice(0, 10)}</span>
    </div>
    <div class="result-row">
      <span class="result-key">Подтверждено</span>
      <span class="result-val muted-val">${r.origin_date.slice(0, 10)}</span>
    </div>
    ${stepRow(1, 'Шаг 1 (уход)')}
    ${stepRow(2, 'Шаг 2 (уход+возврат)')}
  `;
}

function _simplexResultsHtml(f) {
  const r = f.result;
  const dirLabel = r.origin_direction > 0 ? '▲ вверх' : '▼ вниз';
  const dirClass = r.origin_direction > 0 ? 'up' : 'down';
  const relH = r.per_origin[0]?.rel?.at(-1);
  return `
    <div class="result-row">
      <span class="result-key">Origin</span>
      <span class="result-val ${dirClass}">${r.origin_price.toFixed(4)}  ${dirLabel}</span>
    </div>
    <div class="result-row">
      <span class="result-key">Дата origin</span>
      <span class="result-val">${r.origin_date.slice(0, 10)}</span>
    </div>
    <div class="result-row">
      <span class="result-key">Ансамбль</span>
      <span class="result-val">${r.per_origin.length} origin  ·  d∈[${r.d_min},${r.d_max}]</span>
    </div>
    ${relH != null ? `
    <div class="result-row">
      <span class="result-key">rel(H), главный origin</span>
      <span class="result-val">${(relH * 100).toFixed(2)}%</span>
    </div>` : ''}
    ${r.skipped?.length ? `
    <div class="result-row">
      <span class="result-key">Пропущено origin</span>
      <span class="result-val muted-val">${r.skipped.length}</span>
    </div>` : ''}
  `;
}

// "Eye" toggle — session-only pin (NOT persisted to DB, by design — see
// plan). event is optional — history-row eye buttons pass it (needed to
// stopPropagation so the row's own onclick doesn't ALSO fire); the
// standalone object_select panel's pin button (not nested inside a
// clickable row) calls toggleSelectedForecastPin below instead, which has
// nothing to stop propagating to.
export function toggleForecastPin(id, event) {
  event?.stopPropagation();
  if (S.pinnedForecastIds.has(id)) S.pinnedForecastIds.delete(id);
  else S.pinnedForecastIds.add(id);
  refreshHistory();
  renderChart({ preserveRange: true });
  if (S.activeMainTool === 'object_select' && S.selectedForecastId === id) {
    document.getElementById('object-select-pin-btn')?.classList.toggle('active', S.pinnedForecastIds.has(id));
  }
}

// Pins/unpins whatever forecast is currently selected — the object_select
// panel's own "Закрепить" control (round 8: "выбор объекта позволяет...
// зафиксировать отображение на графике" without leaving object_select
// mode, so it needs its own pin button rather than relying on a history
// row's eye icon inside a DIFFERENT model's panel).
export function toggleSelectedForecastPin() {
  if (S.selectedForecastId != null) toggleForecastPin(S.selectedForecastId);
}

export async function deleteForecast(id, event) {
  event.stopPropagation();
  try {
    const modelType = S.historyForecasts.find(f => f.id === id)?.model_type
      || S.renderedForecasts.get(id)?.model_type;
    await api('DELETE', `/forecasts/${id}`);
    S.renderedForecasts.delete(id);
    S.pinnedForecastIds.delete(id);
    if (S.selectedForecastId === id) {
      S.selectedForecastId = null;
      if (modelType) {
        const section = document.getElementById(`results-section-${modelType}`);
        if (section) section.style.display = 'none';
      }
    }
    await refreshHistory();
    renderChart({ preserveRange: true });
  } catch (e) {
    setIdle(e.message, false);
  }
}
