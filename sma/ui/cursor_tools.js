import { S } from './state.js';
import { renderChart, registerShapeProvider, registerAnnotationProvider, setMainLogScale } from './chart.js';
import { registerTool } from './tools.js';

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
// Price levels are pure client-side state (S.priceLevels/S.pendingPriceLevel)
// — no backend, no persistence across a ticker switch (reset in app.js's
// loadCandles, same as S.originTs).

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
}

// ── declutter toggle ─────────────────────────────────────────────────────
export function toggleObjectsHidden() {
  S.objectsHidden = !S.objectsHidden;
  document.getElementById('show-objects-btn')?.classList.toggle('active', !S.objectsHidden);
  redraw();
}

// ── price levels ──────────────────────────────────────────────────────────
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
}

export function removePriceLevel(id) {
  S.priceLevels = S.priceLevels.filter(l => l.id !== id);
  renderPriceLevelList();
  redraw();
}

export function clearPriceLevels() {
  S.priceLevels = [];
  S.pendingPriceLevel = null;
  const fixBtn = document.getElementById('price-level-fix-btn');
  if (fixBtn) fixBtn.disabled = true;
  renderPriceLevelList();
  redraw();
}

function formatPrice(p) {
  return p.toFixed(p >= 100 ? 2 : 4);
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

// ── main-chart shapes/annotations (registered into chart.js, see
// registerShapeProvider/registerAnnotationProvider there) ───────────────
function buildPriceLevelShapes() {
  const shapes = S.priceLevels.map(l => ({
    type: 'line', xref: 'paper', yref: 'y',
    x0: 0, x1: 1, y0: l.price, y1: l.price,
    line: { color: '#d29922', width: 1, dash: 'dot' },
  }));
  if (S.pendingPriceLevel) {
    shapes.push({
      type: 'line', xref: 'paper', yref: 'y',
      x0: 0, x1: 1, y0: S.pendingPriceLevel.price, y1: S.pendingPriceLevel.price,
      line: { color: '#d29922', width: 1, dash: 'dash' },
    });
  }
  return shapes;
}

function buildPriceLevelAnnotations() {
  const ann = S.priceLevels.map(l => ({
    x: 1, xref: 'paper', xanchor: 'right',
    y: l.price, yref: 'y', yanchor: 'bottom',
    text: formatPrice(l.price), showarrow: false,
    font: { size: 10, color: '#d29922' },
    bgcolor: 'rgba(13,17,23,0.6)',
  }));
  if (S.pendingPriceLevel) {
    ann.push({
      x: 1, xref: 'paper', xanchor: 'right',
      y: S.pendingPriceLevel.price, yref: 'y', yanchor: 'top',
      text: `${formatPrice(S.pendingPriceLevel.price)} (не зафиксирован)`, showarrow: false,
      font: { size: 10, color: '#d29922' },
      bgcolor: 'rgba(13,17,23,0.6)',
    });
  }
  return ann;
}

registerShapeProvider(buildPriceLevelShapes);
registerAnnotationProvider(buildPriceLevelAnnotations);
