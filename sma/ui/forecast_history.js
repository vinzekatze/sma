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
import { activateTool, syncDisplayPeek } from './tools.js';
import { iconHtml } from './icons.js';
import { savePinnedForecastIds } from './local_prefs.js';

// Every model this history/results system knows about — kept as a small
// local list (not derived from tools.js's TOOLS, which also holds
// non-forecast tools) since this file only ever needs the two model_type
// strings themselves, to fetch/render one history+results pair per model.
const MODEL_TYPES = ['band_lambda', 'simplex_ensemble', 'regime_mixture_potential'];

// Every results panel this module can show into — the three models above
// plus object_select's own peek panel (showResultsFor routes there while
// that tool is active, see resultsTargetFor below).
const ALL_RESULT_PANELS = [...MODEL_TYPES, 'object_select'];

// Called from app.js on every ticker switch, right alongside the
// S.historyForecasts/S.selectedForecastId reset — refreshHistory()'s
// renderHistoryList only ever touches the history LIST (#history-<type>),
// never the detail panel (#results-<type>) above it, so without this the
// previously viewed ticker's forecast detail stayed visible (display:'')
// with its stale content until a forecast was explicitly selected on the
// new ticker — reported as "activating the forecast tool on a ticker with
// no forecasts yet still shows the old ticker's result" (2026-10-04).
export function hideAllForecastResults() {
  for (const modelType of ALL_RESULT_PANELS) {
    const section = document.getElementById(`results-section-${modelType}`);
    if (section) section.style.display = 'none';
  }
}

const _modelHandlers = {}; // modelType -> {onSelected(f)}

// Registered once at startup (app.js) — e.g. band_lambda wants selecting one
// of its forecasts to also update S.selectedTs/model params and reload the
// zigzag overlay; simplex_ensemble wants its params form refilled.
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
        origin_direction: f.origin_direction, t_query: f.t_query, created_at: f.created_at,
      }));
    } catch (_) {
      return [];
    }
  }));
  S.historyForecasts = perModel.flat();
  for (const modelType of MODEL_TYPES) renderHistoryList(modelType);
}

// The list rows refreshHistory() fetches are deliberately lightweight (no
// result/zone_geometry — see the GET /forecasts query above) — the actual
// band/zone overlay (buildVisibleOverlays in chart.js) needs the FULL
// GET /forecasts/{id} payload, which normally only gets fetched+cached into
// S.renderedForecasts when a forecast is explicitly opened
// (loadAndRenderForecast below). A pinned forecast's id survives a page
// reload / ticker switch (S.pinnedForecastIds, localStorage) but
// S.renderedForecasts does NOT — it's cleared on ticker switch (app.js) and
// starts empty on a fresh page load — so a pin that was never re-opened this
// session had nothing for buildVisibleOverlays to draw: the marker showed
// (pure function of S.historyForecasts + S.pinnedForecastIds) but the band
// itself silently didn't (project feedback 2026-08-25: "закрепленные...
// прогнозы... не отображаются после перезагрузки страницы / загрузки
// тикера"). Called from app.js's loadCandles right after refreshHistory(),
// so every currently-pinned id that's present in the freshly loaded history
// gets its full data fetched once and cached, same cache
// loadAndRenderForecast reads from — re-selecting/re-pinning it afterward
// costs nothing extra.
export async function hydratePinnedForecasts() {
  const ids = [...S.pinnedForecastIds].filter(id =>
    !S.renderedForecasts.has(id) && S.historyForecasts.some(f => f.id === id)
  );
  await Promise.all(ids.map(async id => {
    try {
      const f = await api('GET', `/forecasts/${id}`);
      S.renderedForecasts.set(id, {
        model_type: f.model_type, result: f.result, params: f.params, zone_geometry: f.zone_geometry,
      });
    } catch (_) {
      // stale/deleted pinned forecast — leave unrendered, harmless (same as
      // any other id in S.pinnedForecastIds absent from S.historyForecasts)
    }
  }));
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
    const tLabel = f.t_query != null ? ` · T=${(f.t_query * 100).toFixed(0)}%` : '';
    return `
      <div class="hist-item" onclick="loadAndRenderForecast(${f.id})"${active}>
        <div class="hist-item-header">
          <span class="hist-date">${originDay}</span>
          <button class="hist-eye-btn${pinned ? ' active' : ''}" onclick="toggleForecastPin(${f.id}, event)"
            title="${pinned ? 'Снять с отображения' : 'Закрепить на графике (показывать вместе с другими)'}">${iconHtml('eye')}</button>
          <button class="hist-del-btn" onclick="deleteForecast(${f.id}, event)" title="Удалить">${iconHtml('close')}</button>
        </div>
        <div><span class="hist-price">${dir} ${f.origin_price.toFixed(4)}${tLabel}</span></div>
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
    syncDisplayPeek(); // activateTool above already triggers this via renderToolbar, but the switchTool:false path (a marker click while already in object_select) doesn't call activateTool at all
    await _modelHandlers[f.model_type]?.onSelected?.(f);

    await refreshHistory();
    renderChart({ preserveRange: true });
    showResultsFor(f);
    setIdle(`Прогноз готов  •  ${f.model_type}  •  ${f.result.origin_extreme_date.slice(0, 10)}`);
  } catch (e) {
    setIdle(e.message, false);
  }
}

// Which results panel a forecast of the given model_type currently renders
// into — 'object_select' while that cursor tool is active (round 8: a
// marker click must never switch tools, so browsing must stay inside
// object_select's own panel), otherwise the model's own panel. Shared by
// showResultsFor (which panel to fill) and toggleForecastPin (which panel's
// eye button, id `results-pin-btn-${target}`, needs its 'active' class
// synced — every model's results header now has one, project feedback
// 2026-08-25: "не хватает иконки глазика рядом с 'Результат', как в режиме
// 'выбор объектов'" — previously only object_select had it).
function resultsTargetFor(modelType) {
  return S.activeMainTool === 'object_select' ? 'object_select' : modelType;
}

// Renders into the CURRENTLY ACTIVE tool's own results area — normally
// that's the forecast's own model_type (switchTool just activated it, or
// it was already active), but while 'object_select' is the active tool
// (round 8 fix above — a marker click never switches tools), results go
// into object_select's OWN results area instead (#results-object_select,
// see index.html) so inspecting a forecast never requires leaving
// object_select mode.
function showResultsFor(f) {
  const target = resultsTargetFor(f.model_type);
  const content = document.getElementById(`results-${target}`);
  const section = document.getElementById(`results-section-${target}`);
  if (!content || !section) return;
  content.innerHTML = f.model_type === 'simplex_ensemble' ? _simplexResultsHtml(f)
    : f.model_type === 'regime_mixture_potential' ? _potentialResultsHtml(f)
    : _bandResultsHtml(f);
  section.style.display = '';
  document.getElementById(`results-pin-btn-${target}`)
    ?.classList.toggle('active', S.pinnedForecastIds.has(S.selectedForecastId));
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
      <span class="result-key">T (порог зигзага)</span>
      <span class="result-val">${(f.params.t_query * 100).toFixed(0)}%</span>
    </div>
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

const REGIME_LABELS = { 0: 'низкая', 1: 'средняя', 2: 'высокая', '-1': 'н/д' };

function _potentialResultsHtml(f) {
  const r = f.result;
  const dirLabel = r.origin_direction > 0 ? '▲ вверх' : '▼ вниз';
  const dirClass = r.origin_direction > 0 ? 'up' : 'down';
  const main = r.snapshots[0]; // самый свежий (rewind_idx=0) — см. regime_mixture_potential.py
  const topH = main.components_by_h.at(-1);
  const dominant = topH.reduce((a, b) => (b[2] > a[2] ? b : a));
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
      <span class="result-key">Премотка</span>
      <span class="result-val">${r.n_forecasts} прогноз(ов)  ·  шаг ${r.rewind_step}</span>
    </div>
    <div class="result-row">
      <span class="result-key">Режим волатильности</span>
      <span class="result-val">${REGIME_LABELS[main.cur_regime] ?? '?'}${main.regime_used ? '' : ' (fallback на полный пул)'}</span>
    </div>
    <div class="result-row">
      <span class="result-key">Доминантный сценарий (h=${r.horizon})</span>
      <span class="result-val">${dominant[2] * 100 | 0}%  @ ${dominant[0].toFixed(4)}</span>
    </div>
    ${r.skipped?.length ? `
    <div class="result-row">
      <span class="result-key">Пропущено origin премотки</span>
      <span class="result-val muted-val">${r.skipped.length}</span>
    </div>` : ''}
  `;
}

// "Eye" toggle — session-only pin (NOT persisted to the DB, by design — see
// plan). Persisted to localStorage instead (§1.1b of docs/plans/
// frontend_improvements_plan.md) — see local_prefs.js's docstring for the
// caveat that this currently only has a visible effect before the next
// app.js:loadCandles call, which unconditionally clears pins per ticker.
// event is optional — history-row eye buttons pass it (needed to
// stopPropagation so the row's own onclick doesn't ALSO fire); every
// results-panel pin button (not nested inside a clickable row — object_select
// AND, as of project feedback 2026-08-25, every model's own "Результат"
// header too) calls toggleSelectedForecastPin below instead, which has
// nothing to stop propagating to.
export function toggleForecastPin(id, event) {
  event?.stopPropagation();
  if (S.pinnedForecastIds.has(id)) S.pinnedForecastIds.delete(id);
  else S.pinnedForecastIds.add(id);
  refreshHistory();
  renderChart({ preserveRange: true });
  if (S.selectedForecastId === id) {
    const modelType = S.historyForecasts.find(f => f.id === id)?.model_type
      || S.renderedForecasts.get(id)?.model_type;
    if (modelType) {
      document.getElementById(`results-pin-btn-${resultsTargetFor(modelType)}`)
        ?.classList.toggle('active', S.pinnedForecastIds.has(id));
    }
  }
  savePinnedForecastIds(S.pinnedForecastIds); // §1.1b
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
    savePinnedForecastIds(S.pinnedForecastIds); // §1.1b — deleted forecast can't stay a stale pin in localStorage
    if (S.selectedForecastId === id) {
      S.selectedForecastId = null;
      syncDisplayPeek();
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
