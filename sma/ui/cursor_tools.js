import { S } from './state.js';
import { api } from './api.js';
import { renderChart, priceToPixelY, registerOverlayRefresher, setMainLogScale } from './chart.js';
import { registerTool } from './tools.js';
import { saveMainLogScale, saveObjectsHidden } from './local_prefs.js';
import { hexToRgba } from './color_utils.js';

// ── Основной график — "cursor" tools: free / price_level / object_select ──
// Three of the tools competing for the SAME S.activeMainTool exclusivity
// slot as analyzers/forecast models (see tools.js module docstring) —
// merged in project feedback 2026-08-18, round 7 ("Одновременно может быть
// выбран только один инструмент... разные режимы курсора в сочетании с
// прогнозами начинают вести себя непредсказуемо"): S.cursorTool used to be
// a SEPARATE exclusivity axis from S.activeMainTool, so e.g. 'object_select'
// cursor mode + an armed forecast tool could both be "active" at once with
// ambiguous click semantics. Now there's only ONE active tool, period —
// clicking a cursor-group icon deselects whatever analyzer/forecaster was
// active, and vice versa. group:'cursor' only affects which toolbar
// icon-CLUSTER the button renders into (tools.js:renderToolbarButtons) —
// round 7 follow-up restored the three separate toolbar SECTIONS
// (Курсор/Анализ/Прогноз, each own header+icon row — project feedback
// round 8: "группировка... нужна, а то путаница") while keeping the single
// settings-panel block below all three ("с одним блоком настроек всё
// верно") — grouping is purely visual placement, never a second
// exclusivity axis.
//
// 'free' (round 8: "инструмент свободной мыши куда-то потерялся, пусть
// будет") is a REAL selectable entry, not a null/absent state — S starts
// with it active by default (see state.js), matching the pre-round-7
// behavior where the free-mouse icon was highlighted on load. It registers
// no onOriginClick, so selecting it is functionally "nothing armed": a
// candle click does nothing beyond whatever the (harmlessly absent) active
// tool's onOriginClick would have done.
//
// Price levels: S.pendingPriceLevel (the not-yet-fixed one currently
// following clicks) is pure client-side, never persisted. S.priceLevels
// (fixed levels) is per-ticker DB state as of 2026-08-20 (project feedback:
// "значения зафиксированных ценовых уровней тоже желательно сохранять —
// возможно в БД") — same analysis_settings mechanism trend_ruler/
// moving_averages/zigzag_tool already use (analyzer_type='price_level'),
// loaded/saved via loadPriceLevelDefaults/savePriceLevelsToDb below. Reset
// in app.js's loadCandles (same as every other per-ticker analyzer state) —
// see that reset's own comment for why it calls clearPriceLevels WITHOUT
// persisting (persist:false) rather than this module's normal path.

registerTool({
  type: 'free',
  surface: 'main',
  group: 'cursor',
  icon: 'cursor',
  label: 'Свободная мышь',
  panelId: 'tool-panel-free',
  // No onOriginClick — see module docstring.
});

registerTool({
  type: 'price_level',
  surface: 'main',
  group: 'cursor',
  icon: 'price-level',
  label: 'Ценовой уровень',
  panelId: 'tool-panel-price_level',
  // No onOriginClick — this tool needs the actual clicked PRICE (a Y-axis
  // pixel converted to a data value), not a candle timestamp, so it's
  // handled as a dedicated early-return case directly in chart.js's
  // plotly_click dispatcher (see _priceAtClick) rather than through the
  // generic onOriginClick(ts) signature every origin-based tool uses.
  onDeselected: () => {
    // Not-yet-fixed level is scratch state that follows the cursor while
    // this tool is active — must not survive switching to a different tool
    // (project feedback 2026-08-25: "не зафиксированный уровень не
    // сбрасывается при переключении на другой инструмент"). Fixed levels
    // (S.priceLevels) are untouched — those are the persisted, committed
    // ones.
    if (S.pendingPriceLevel) {
      S.pendingPriceLevel = null;
      const fixBtn = document.getElementById('price-level-fix-btn');
      if (fixBtn) fixBtn.disabled = true;
      redraw();
    }
  },
});

registerTool({
  type: 'object_select',
  surface: 'main',
  group: 'cursor',
  icon: 'target',
  label: 'Выбор объектов',
  panelId: 'tool-panel-object_select',
  // No onOriginClick either — this mode only changes marker-click ownership
  // (app.js: bypasses the "own model type only" restriction on which
  // forecast markers are clickable), a plain candle click does nothing
  // special while it's active. Selecting a forecast while this tool is
  // active does NOT switch S.activeMainTool to that forecast's own model
  // (project feedback round 8: "выбор объекта... отдельный режим" — a
  // marker click must never itself change which tool is armed) — its
  // result renders into THIS tool's own #results-object_select area
  // instead (forecast_history.js:showResultsFor), with its own pin
  // ("Закрепить") control, so inspecting/pinning a forecast never requires
  // leaving object_select mode.
});

let _nextLevelId = 1;

function redraw() {
  renderChart({ preserveRange: true });
}

// ── log scale (main chart only — independent of spectrogram's own logY,
// which lives on the oscillator's frequency axis) ───────────────────────
export function toggleMainLogScale() {
  setMainLogScale(!S.mainLogScale); // chart.js owns the relayout + queued-render race safety, see there
  document.getElementById('log-scale-btn')?.classList.toggle('active', S.mainLogScale);
  saveMainLogScale(S.mainLogScale); // §1.1b of docs/plans/frontend_improvements_plan.md — survives a page reload
}

// ── declutter toggle ─────────────────────────────────────────────────────
export function toggleObjectsHidden() {
  S.objectsHidden = !S.objectsHidden;
  document.getElementById('show-objects-btn')?.classList.toggle('active', !S.objectsHidden);
  redraw();
  saveObjectsHidden(S.objectsHidden); // §1.1b
}

// ── price levels ──────────────────────────────────────────────────────────

// Continuous cursor-following preview (project feedback 2026-08-25 —
// replaces Plotly's own bar-snapped crosshair, see chart.js:
// _bindPriceLevelEventsIfNeeded) — fires on every native 'mousemove' while
// the price_level tool is active. Calls renderPriceLevelOverlay() DIRECTLY
// (a plain DOM read+write, see its own docstring below) rather than going
// through Plotly at all — an earlier version called chart.js:applyShapes()
// here (a Plotly.relayout with just shapes/annotations), which turned out
// to fight Plotly's own drag-tracking state whenever this fired WHILE the
// user was panning/zooming the chart (project feedback 2026-08-25: "уровни
// 'плывут'... шкала тоже отвязывается"). The identity check skips redundant
// work when the price hasn't materially changed (e.g. repeated nulls from
// mouseleave, or the pixel rounds to the same value while barely moving).
export function setPriceLevelHoverPrice(price) {
  if (S.priceLevelHoverPrice === price) return;
  S.priceLevelHoverPrice = price;
  renderPriceLevelOverlay();
}

export function setPendingPriceLevel(price) {
  S.pendingPriceLevel = { price };
  const fixBtn = document.getElementById('price-level-fix-btn');
  if (fixBtn) fixBtn.disabled = false;
  redraw();
}

export function fixPendingPriceLevel() {
  if (!S.pendingPriceLevel) return;
  S.priceLevels.push({ id: _nextLevelId++, price: S.pendingPriceLevel.price });
  S.pendingPriceLevel = null;
  const fixBtn = document.getElementById('price-level-fix-btn');
  if (fixBtn) fixBtn.disabled = true;
  renderPriceLevelList();
  redraw();
  savePriceLevelsToDb();
}

export function removePriceLevel(id) {
  S.priceLevels = S.priceLevels.filter(l => l.id !== id);
  renderPriceLevelList();
  redraw();
  savePriceLevelsToDb();
}

// persist:false is used ONLY by app.js's loadCandles (ticker switch) — it
// needs to wipe S.priceLevels in memory for the OUTGOING ticker's data
// without writing an empty list to the DB, since loadPriceLevelDefaults()
// (called right after, in the same reset sequence) is about to load the
// NEW ticker's own persisted levels. Persisting the empty state here too
// would race the debounced save against that load and could wipe the new
// ticker's real levels 500ms later. The "Очистить" button in the panel
// (index.html) calls this with no args — persist:true, a genuine user
// action that SHOULD write the empty state.
export function clearPriceLevels({ persist = true } = {}) {
  S.priceLevels = [];
  S.pendingPriceLevel = null;
  const fixBtn = document.getElementById('price-level-fix-btn');
  if (fixBtn) fixBtn.disabled = true;
  renderPriceLevelList();
  redraw();
  if (persist) savePriceLevelsToDb();
}

function formatPrice(p) {
  return p.toFixed(p >= 100 ? 2 : 4);
}

// Last-used price levels for this (instrument, interval) — same mechanism
// as trend_ruler.js:loadTrendRulerDefaults (analysis_settings, generic
// analyzer_type string, no schema change needed). Called from app.js's
// loadCandles, right after the ticker-switch clearPriceLevels({persist:false}).
export async function loadPriceLevelDefaults() {
  if (!S.instrumentId) return;
  try {
    const res = await api(
      'GET',
      `/series/analysis-settings?instrument_id=${S.instrumentId}&interval=${S.interval}&analyzer_type=price_level`
    );
    const levels = res.params?.levels ?? [];
    S.priceLevels = levels.map(l => ({ ...l }));
    const maxId = S.priceLevels.reduce((m, l) => Math.max(m, l.id), 0);
    _nextLevelId = maxId + 1;
  } catch (_) {
    S.priceLevels = [];
  }
  renderPriceLevelList();
  redraw();
}

let _priceLevelSaveTimer = null;

function savePriceLevelsToDb() {
  if (!S.instrumentId) return;
  clearTimeout(_priceLevelSaveTimer);
  _priceLevelSaveTimer = setTimeout(() => {
    api('POST', '/series/analysis-settings', {
      instrument_id: S.instrumentId, interval: S.interval,
      analyzer_type: 'price_level', params: { levels: S.priceLevels },
    }).catch(() => {});
  }, 500);
}

function renderPriceLevelList() {
  const box = document.getElementById('price-level-list');
  if (!box) return;
  if (!S.priceLevels.length) {
    box.innerHTML = '<span class="muted-val">Нет зафиксированных уровней</span>';
    return;
  }
  box.innerHTML = S.priceLevels
    .slice()
    .sort((a, b) => b.price - a.price)
    .map(l => `
      <div class="field-row">
        <span class="num">${formatPrice(l.price)}</span>
        <button class="icon-btn danger" title="Удалить" onclick="removePriceLevel(${l.id})">
          <svg class="icon"><use href="#icon-close"/></svg>
        </button>
      </div>
    `)
    .join('');
}

// ── main-chart rendering: plain HTML/CSS overlay, NOT Plotly shapes/
// annotations (registered into chart.js, see registerOverlayRefresher
// there) ─────────────────────────────────────────────────────────────────
// Moved off Plotly entirely 2026-08-25, after two separate problems with
// the Plotly-based approach: (1) the price LABEL (annotation) silently
// failed to appear when the page loaded already in log scale — root cause
// never confirmed despite checking Plotly's actual source (see chart.js:
// priceToPixelY's docstring for that investigation); (2) the mousemove-
// driven hover preview called chart.js:applyShapes() (a Plotly.relayout)
// on every tick, which fought Plotly's own drag-tracking state whenever it
// fired WHILE the user was panning/zooming ("уровни 'плывут'"). Rather than
// keep chasing Plotly-specific quirks, price_level's ENTIRE visual — lines
// AND labels, fixed/pending/hover alike — is now a plain absolutely-
// positioned overlay (#price-level-overlay, sma/ui/index.html, a sibling of
// #chart inside the already-`position:relative` #chart-wrap) positioned via
// chart.js:priceToPixelY (real price -> pixel, using the SAME axis math
// already verified against Plotly's source for click/hover) — no
// Plotly.relayout call anywhere in this path any more, so nothing here can
// conflict with Plotly's own interaction state again, and no log-axis
// annotation quirk applies since Plotly never sees this text at all.
//
// Color is the "Ценовой уровень" role of the app-wide color profile
// (S.colorProfile.price_level, docs/plans/frontend_improvements_plan.md
// §1.1a), read live on every build (not cached) so a profile save takes
// effect immediately. Same color for every level and for the not-yet-fixed
// preview — the profile is one fixed role, not a per-level choice. Hover
// preview only renders while price_level is actually the active tool —
// S.priceLevelHoverPrice itself is cleared on mouseleave/tool switch (see
// chart.js:_bindPriceLevelEventsIfNeeded), but double-checking here too
// means a stale value can never leak onto the chart. Thinner/more
// translucent than the fixed/pending lines (via hexToRgba) so it reads as
// "just a preview."
function isPriceLevelHoverVisible() {
  return S.activeMainTool === 'price_level' && S.priceLevelHoverPrice != null;
}

function renderPriceLevelOverlay(gd) {
  const container = document.getElementById('price-level-overlay');
  if (!container) return;
  // Same declutter toggle every other main-chart overlay respects
  // (tools.js:isToolObjectVisible) — this one lives outside that mechanism
  // entirely (see module comment above), so it needs its own check: when
  // the toggle is soloing the active tool, price levels only survive while
  // price_level itself is active — same rule, applied by hand.
  // registerOverlayRefresher (chart.js) calls this on EVERY paint and every
  // plotly_relayouting tick during ANY drag/zoom app-wide, regardless of
  // whether price_level was ever used this session — with nothing to draw
  // (the common case: no levels set), the old code still ran the full
  // `items.map/join` + innerHTML write every single frame. Cheap per call,
  // but unconditional forced DOM writes on every interaction frame add up,
  // and are one of the few concrete (rather than architectural) costs this
  // codebase controls — flagged as a contributor while looking into project
  // feedback 2026-08-25 ("в Firefox наблюдаются тормоза на графике"; see
  // docs/plans/frontend_improvements_plan.md for the full write-up, incl.
  // why the DOMINANT cost is very likely Plotly's own SVG candlestick
  // rendering, not this). Skip touching the DOM at all when there's nothing
  // to show and the container is already empty.
  const nothingToShow = (S.objectsHidden && S.activeMainTool !== 'price_level')
    || (!S.priceLevels.length && !S.pendingPriceLevel && !isPriceLevelHoverVisible());
  if (nothingToShow) {
    if (container.childElementCount) container.innerHTML = '';
    return;
  }
  gd = gd || document.getElementById('chart');
  if (!gd) return;

  const color = S.colorProfile.price_level;
  const items = S.priceLevels.map(l => ({ price: l.price, text: formatPrice(l.price), color, dashed: false }));
  if (S.pendingPriceLevel) {
    items.push({
      price: S.pendingPriceLevel.price,
      text: `${formatPrice(S.pendingPriceLevel.price)} (не зафиксирован)`,
      color, dashed: true,
    });
  }
  if (isPriceLevelHoverVisible()) {
    items.push({
      price: S.priceLevelHoverPrice, text: formatPrice(S.priceLevelHoverPrice),
      color: hexToRgba(color, 0.7), lineColor: hexToRgba(color, 0.5), dashed: true,
    });
  }

  container.innerHTML = items.map(it => {
    const py = priceToPixelY(gd, it.price);
    if (py == null) return '';
    return `
      <div class="price-level-line${it.dashed ? ' dashed' : ''}" style="top:${py}px;border-top-color:${it.lineColor ?? it.color}">
        <span class="price-level-label" style="color:${it.color}">${it.text}</span>
      </div>
    `;
  }).join('');
}

registerOverlayRefresher(renderPriceLevelOverlay);
