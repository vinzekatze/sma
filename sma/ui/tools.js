import { S } from './state.js';
import { saveActiveMainTool, saveActiveOscillatorTool } from './local_prefs.js';

// ── Основной график / Осциллятор — unified tool registry ────────────────
// Started as an "analyzer registry" (see git history / project memory if
// curious) and grew to also hold forecast MODELS (band_lambda,
// simplex_ensemble — project feedback 2026-08-18, round 6: "Каждая модель
// как отдельный инструмент"), then CURSOR tools (price_level, object_select
// — project feedback 2026-08-18, round 7: "Одновременно может быть выбран
// только один инструмент... разные режимы курсора в сочетании с прогнозами
// начинают вести себя непредсказуемо") once it became clear all three kinds
// compete for the exact same thing — a single click's meaning on the main
// chart — so they need to share ONE exclusivity slot (S.activeMainTool),
// not several independent ones. Renamed analyzers.js -> tools.js /
// ANALYZERS -> TOOLS accordingly rather than leave a misnomer once forecast
// models and cursor tools moved in. The three kinds stay visually grouped
// in THREE SEPARATE toolbar sections (Курсор/Анализ/Прогноз, each own
// header+icon row — project feedback round 8: "группировка... нужна, а то
// путаница", reverting round 7's single merged toolbar row) because
// grouping is still useful — "мы можем точно настроить поведение кликов
// под каждый" (round 7) — grouping is purely cosmetic placement, never a
// second exclusivity axis; the single settings-panel BLOCK below all three
// stayed merged (round 8 confirmed that part: "с одним блоком настроек всё
// верно"). S.activeMainTool is NEVER null — 'free' (cursor_tools.js) is a
// real, always-available entry that plays that role explicitly (round 8:
// "инструмент свободной мыши... пусть будет" — a null/absent state had no
// toolbar icon to show as active, which read as broken/inconsistent once a
// tool is always meant to be "selected").
//
// Each tool module (analysis.js for spectrogram, trend_ruler.js,
// variance_oscillator.js, moving_averages.js, zigzag_tool.js,
// volume_oscillator.js, forecast.js for band_lambda, simplex_ensemble.js,
// and whatever comes next) self-registers ONE entry here at module load
// time — this file itself imports NOTHING from them, so chart.js/app.js can
// import this registry without ever importing a tool module directly
// (which would risk a cycle: tool modules already import renderChart from
// chart.js). Adding tool #N+1 means: write its module, call
// registerTool({...}) once, add one settings-panel <div> to index.html — no
// edits here, no pairwise coupling with existing tools.
//
// UI model (project feedback 2026-08-18, several rounds): a graphics-editor
// style TOOLBAR of icon buttons, not an accordion/dropdown — clicking a
// tool selects it (S.activeMainTool or S.activeOscillatorTool, by surface),
// which shows ITS settings panel and routes chart clicks to it (see
// app.js). Two different visibility policies per surface (revised
// 2026-08-20, §2.6/§3.1 of docs/plans/frontend_improvements_plan.md):
// - MAIN surface: selecting a tool is INDEPENDENT of whether its output is
//   actually drawn — visibility (showOnChart/pin, an eye-icon toggle in the
//   tool's own panel) is each tool's own explicit, persistent flag, ORed
//   with "is this the active tool" (see e.g. trend_ruler.js:
//   isTrendRulerVisible) — so several tools' objects can be pinned visible
//   at once regardless of which one's toolbar-selection is currently
//   active.
// - OSCILLATOR surface: selecting a tool directly IS showing it — no
//   separate pin/checkbox, S.subpanel always mirrors S.activeOscillatorTool
//   (see selectTool/activateTool below). No pin concept here on purpose —
//   oscillators share one physical slot and don't layer well, unlike main-
//   chart overlays.
//
// Entry shape (every field but `type`/`icon`/`label`/`panelId` is optional
// — a tool only implements what applies to it, e.g. spectrogram has no
// buildMainTraces/onOriginClick/resetOrigin since it never draws on the
// main chart and has no origin concept; a forecast model has no
// buildMainTraces/buildSubpanelTraces AT ALL — its chart presence flows
// through the separate, pre-existing pin/select overlay system
// (chart.js:buildVisibleOverlays + MODEL_DISPLAY), this registry only
// arbitrates toolbar/click-target selection for it):
//   type:                 string, matches S.subpanel and either
//                            S.activeMainTool (main-surface) or
//                            S.activeOscillatorTool (oscillator-surface) —
//                            see surfaceOf() below for how that's decided.
//                            For a forecast model, MUST equal its
//                            model_type (band_lambda/simplex_ensemble) —
//                            marker-click scoping (app.js) compares
//                            S.activeMainTool against a marker's
//                            f.model_type directly, no extra field needed.
//   surface:               optional explicit 'main'|'oscillator' override —
//                            forecast models and cursor tools set this
//                            (neither has buildMainTraces to derive it
//                            from); every other tool leaves it unset and
//                            lets surfaceOf() derive it.
//   group:                 optional, PURELY which toolbar icon-cluster a
//                            main-surface button renders into — 'cursor',
//                            'analyzer' (default), or 'forecaster'. Does
//                            not affect exclusivity (still ONE
//                            S.activeMainTool across all three clusters —
//                            see module docstring above).
//   icon:                  sprite symbol name, i.e. the suffix of an
//                            "icon-<name>" <symbol id> in index.html's
//                            #icon-sprite (see sma/ui/icons.js)
//   label:                 toolbar button tooltip / title text
//   panelId:               DOM id of this tool's settings-panel <div>
//   onSelected():          called right after this tool becomes the active
//                            one on its surface (toolbar click via
//                            selectTool, or activateTool) — oscillator tools
//                            use it to fetch their data if this is the
//                            first time being shown (see
//                            variance_oscillator.js); most tools don't need
//                            it at all (e.g. spectrogram needs an explicit
//                            "Рассчитать" click regardless of selection).
//   onDeselected():        called on the PREVIOUSLY active tool of a surface
//                            right before it stops being active (selectTool/
//                            activateTool switching to a different type, or
//                            selectTool's own toggle-off) — for transient,
//                            not-yet-committed per-tool state that should
//                            never survive a tool switch (e.g. price_level's
//                            S.pendingPriceLevel, project feedback
//                            2026-08-25: "не зафиксированный уровень не
//                            сбрасывается при переключении на другой
//                            инструмент"). Most tools don't need it — only
//                            for state that is otherwise persistent/visible
//                            beyond the tool's own active session.
//   buildMainTraces():     -> Plotly traces for the MAIN chart (own
//                             "показывать на графике" flag checked INSIDE
//                             the tool's own function, same pattern
//                             buildZigzagTrace already used)
//   buildSubpanelTraces(): -> Plotly traces for the shared yaxis2 slot,
//                             called only when S.subpanel === this type
//   buildSubpanelShapes(): -> Plotly shapes for the shared yaxis2 slot
//                             (reference lines etc.; none currently use it)
//   subpanelYAxisPolicy(): -> { key, fixedrange, range } describing how the
//                             shared yaxis2 should center/zoom while THIS
//                             tool owns it (chart.js:activeY2Policy) — `key`
//                             names the current MODE (e.g.
//                             'variance_osc:slope' vs '...:var', since one
//                             tool can have several distinct policies); only
//                             called when S.subpanel === this type, same as
//                             buildSubpanelTraces. Oscillator-surface tools
//                             only (§3.3 of docs/plans/frontend_
//                             improvements_plan.md).
//   buildOriginShape():    -> Plotly shapes for this tool's own origin
//                             crosshair on the main chart
//   onOriginClick(ts):     called when a candle is clicked while THIS
//                             tool is active (see app.js) — a forecast
//                             model's entry uses this to call the shared
//                             setOrigin(ts) (chart.js) instead of a
//                             per-tool origin the way analyzers do.
//   resetOrigin():         called by the toolbar's shared "reset origin"
//                             action button when this tool is active —
//                             absent if the tool has no "live" origin
//   onMainOriginChanged(): called when the MAIN origin (S.originTs) is set
//                             or cleared while this tool is active — the
//                             place to recompute live results / labels
//                             that depend on it (chart.js:setOrigin,
//                             clearMainOrigin)
//                             concept (the button hides itself, see
//                             renderToolbar; forecast models don't set this
//                             — there's no "live" default to reset to).
export const TOOLS = [];

export function registerTool(entry) {
  TOOLS.push(entry);
}

export function getTool(type) {
  return TOOLS.find(a => a.type === type);
}

// The subpanel (yaxis2, bottom of the chart) is a SINGLE physical slot —
// only one tool's oscillator can occupy it. Used to be claimed by each
// tool's own "показать осциллятор" checkbox (claimSubpanel/releaseSubpanel,
// which unchecked every OTHER tool's checkbox as a side effect — fragile:
// three independent per-tool checkboxes each trying to stay in sync with
// one shared slot). Project feedback 2026-08-20, §3.1 of
// docs/plans/frontend_improvements_plan.md: replaced with the same "select
// tool = show it" model the main surface already uses — S.subpanel is now
// just kept identical to S.activeOscillatorTool below, no separate
// checkbox/claim step at all. A dedicated 'none' tool (registered further
// down, icon "eye-off") is the oscillator surface's explicit "nothing
// shown" entry — mirrors 'free' on the main surface (see module docstring)
// rather than leaving a null/absent state with no toolbar representation.
registerTool({
  type: 'none',
  surface: 'oscillator',
  icon: 'eye-off',
  label: 'Выключить осциллятор',
  panelId: 'tool-panel-none',
});

// ── toolbar ──────────────────────────────────────────────────────────────
// A tool with buildMainTraces (or an explicit surface:'main' override —
// forecast models have no buildMainTraces to derive it from, see module
// docstring) draws on the main chart, so its button lives on the Основной
// график tab and its "active" state is tracked in S.activeMainTool (which
// also drives origin-click routing, see app.js); one WITHOUT either only
// ever draws in the shared subpanel, so its button lives on the Осциллятор
// tab and its "active" state is tracked in S.activeOscillatorTool (panel
// visibility only — no origin concept applies there).
function surfaceOf(a) {
  return a.surface ?? (a.buildMainTraces ? 'main' : 'oscillator');
}

function activeToolFor(surface) {
  return surface === 'main' ? S.activeMainTool : S.activeOscillatorTool;
}

// Builds the actual <button class="tool-btn"> elements from the registry
// into #cursor-tool-buttons / #main-tool-buttons / #forecast-tool-buttons /
// #oscillator-tool-buttons (each a `display:contents` wrapper — see
// style.css — so they lay out as direct children of their .tool-toolbar).
// Called once from app.js's DOMContentLoaded handler, after every tool
// module has already self-registered (static imports resolve before that
// event fires) — a new tool's button appears with zero HTML edits.
export function renderToolbarButtons() {
  const buttonHtml = a => `
    <button class="tool-btn" id="tool-btn-${a.type}" title="${a.label}"
      onclick="selectTool('${a.type}')">
      <svg class="icon"><use href="#icon-${a.icon}"/></svg>
    </button>
  `;
  const cursorContainer = document.getElementById('cursor-tool-buttons');
  if (cursorContainer) {
    cursorContainer.innerHTML = TOOLS
      .filter(a => surfaceOf(a) === 'main' && a.group === 'cursor')
      .map(buttonHtml).join('');
  }
  const mainContainer = document.getElementById('main-tool-buttons');
  if (mainContainer) {
    mainContainer.innerHTML = TOOLS
      .filter(a => surfaceOf(a) === 'main' && (a.group ?? 'analyzer') === 'analyzer')
      .map(buttonHtml).join('');
  }
  const forecastContainer = document.getElementById('forecast-tool-buttons');
  if (forecastContainer) {
    forecastContainer.innerHTML = TOOLS
      .filter(a => surfaceOf(a) === 'main' && a.group === 'forecaster')
      .map(buttonHtml).join('');
  }
  const oscContainer = document.getElementById('oscillator-tool-buttons');
  if (oscContainer) {
    oscContainer.innerHTML = TOOLS.filter(a => surfaceOf(a) === 'oscillator').map(buttonHtml).join('');
  }
}

// Selecting a tool toggles it. On the OSCILLATOR surface, clicking the
// already-active one falls back to 'none' (the explicit "off" tool
// registered above) — same "never null, always a real toolbar entry"
// principle 'free' already established on the main surface (project
// feedback round 8), extended to this surface 2026-08-20 once 'none' became
// a real entry too. On the MAIN surface, clicking the already-active one
// falls back to 'free'. Also keeps S.subpanel mirroring S.activeOscillatorTool
// (the single physical yaxis2 slot, see chart.js) — replaces the old
// claimSubpanel/releaseSubpanel checkbox dance, see registerTool({type:
// 'none', ...}) above. Finally calls the newly active tool's own
// onSelected() hook, if it has one — e.g. variance_osc uses it to fetch its
// data if this is the first time it's being shown (see
// variance_oscillator.js). Purely UI-focus bookkeeping otherwise: which
// panel shows, which tool's origin a click sets (main surface only), and
// whether the shared "reset origin" toolbar button is relevant — never
// touches any tool's own display/data settings beyond that one hook.
export function selectTool(type) {
  const a = getTool(type);
  if (!a) return;
  const surface = surfaceOf(a);
  const prevType = activeToolFor(surface);
  if (surface === 'main') {
    S.activeMainTool = S.activeMainTool === type ? 'free' : type;
    saveActiveMainTool(S.activeMainTool); // §1.1b of docs/plans/frontend_improvements_plan.md
  } else {
    S.activeOscillatorTool = S.activeOscillatorTool === type ? 'none' : type;
    S.subpanel = S.activeOscillatorTool;
    saveActiveOscillatorTool(S.activeOscillatorTool); // §1.1b — now persisted too, project feedback 2026-08-25 (was deliberately NOT, see local_prefs.js docstring for the reversal)
  }
  const newType = activeToolFor(surface);
  if (prevType !== newType) getTool(prevType)?.onDeselected?.();
  renderToolbar();
  getTool(newType)?.onSelected?.();
}

// Non-toggling variant — sets a tool as active outright regardless of
// current state, used by forecast_history.js when loading/selecting a
// forecast (its results should be visible without a separate manual click
// on the toolbar icon, unlike the toggle behavior a direct icon click gets).
export function activateTool(type) {
  const a = getTool(type);
  if (!a) return;
  const surface = surfaceOf(a);
  const prevType = activeToolFor(surface);
  if (surface === 'main') {
    S.activeMainTool = type;
    saveActiveMainTool(S.activeMainTool); // §1.1b
  } else {
    S.activeOscillatorTool = type;
    S.subpanel = type;
    saveActiveOscillatorTool(S.activeOscillatorTool); // §1.1b
  }
  if (prevType !== type) getTool(prevType)?.onDeselected?.();
  renderToolbar();
  a.onSelected?.();
}

// "Выбор объектов" peek (project feedback round 2 — the first version of
// this peeked EVERY registered tool's displaySectionId simultaneously, keyed
// on S.activeMainTool alone: "какая-то чехарда с zig-zag, линейками, всеми
// другими инструментами... завязано на то, открыт ли сендвич с отображением
// в своём инструменте"). What's actually wanted: select ONE forecast result
// (band_lambda/simplex_ensemble/regime_mixture_potential — the models with a
// selectable per-result history, forecast_history.js:MODEL_TYPES) and THAT
// model's own "Отображение" controls appear — nothing else's, and forced
// open (no accordion click needed). Selecting a different model's result
// swaps it. Implemented by REPARENTING that one model's `<model>-display-
// section` element into #object-select-display-slot (index.html) rather
// than CSS-hiding siblings inside its own hidden panel — the element keeps
// its ids/onclick handlers, so every existing control inside it (XY-pad,
// sliders, checkboxes) keeps working untouched wherever it currently lives
// in the DOM. Reads S.selectedForecastId/S.renderedForecasts directly
// (plain state, both already imported) rather than importing forecast_
// history.js's own resolution helpers, to avoid a cycle (that file already
// imports activateTool from here — see its own module docstring).
const _PEEKABLE_MODEL_TYPES = ['band_lambda', 'simplex_ensemble', 'regime_mixture_potential'];
const _peekHome = new Map(); // modelType -> {parent, next, wasOpen} — captured once, on first peek

function _peekedModelType() {
  if (S.activeMainTool !== 'object_select') return null;
  const f = S.renderedForecasts.get(S.selectedForecastId);
  return f && _PEEKABLE_MODEL_TYPES.includes(f.model_type) ? f.model_type : null;
}

// Exported so forecast_history.js can resync immediately after a marker
// click while ALREADY in object_select mode — that path (loadAndRenderForecast
// with switchTool:false, project feedback round 8: a marker click must never
// switch tools) never calls activateTool/selectTool, so renderToolbar (which
// also calls this) wouldn't otherwise run.
export function syncDisplayPeek() {
  const slot = document.getElementById('object-select-display-slot');
  if (!slot) return;
  const wanted = _peekedModelType();
  for (const child of [...slot.children]) {
    const mt = _PEEKABLE_MODEL_TYPES.find(t => child.id === `${t}-display-section`);
    if (mt && mt !== wanted) {
      const home = _peekHome.get(mt);
      if (home) {
        if (home.next) home.parent.insertBefore(child, home.next); else home.parent.appendChild(child);
        child.classList.toggle('open', home.wasOpen);
      }
    }
  }
  if (!wanted) return;
  const el = document.getElementById(`${wanted}-display-section`);
  if (!el) return;
  if (el.parentElement !== slot) {
    if (!_peekHome.has(wanted)) {
      _peekHome.set(wanted, { parent: el.parentElement, next: el.nextElementSibling, wasOpen: el.classList.contains('open') });
    }
    slot.appendChild(el);
  }
  el.classList.add('open');
}

export function renderToolbar() {
  for (const a of TOOLS) {
    const surface = surfaceOf(a);
    const isActive = a.type === activeToolFor(surface);
    document.getElementById(`tool-btn-${a.type}`)?.classList.toggle('active', isActive);
    const panel = document.getElementById(a.panelId);
    if (panel) panel.style.display = isActive ? '' : 'none';
  }
  syncDisplayPeek();

  // No "nothing selected" hint needed on either surface any more —
  // S.activeMainTool/S.activeOscillatorTool are never null ('free'/'none'
  // play that role explicitly, see selectTool), so their own
  // tool-panel-free/tool-panel-none always cover that spot via the loop
  // above instead of a separate hint element.
  // (the Курсор toolbar's reset button is always enabled — see app.js:resetActiveToolOrigin)
}


// ── "показать/скрыть активные объекты" declutter toggle (S.objectsHidden,
// cursor_tools.js:toggleObjectsHidden) — shared visibility rule every
// main-surface tool with its own persisted showOnChart flag applies (trend_
// ruler/moving_averages/zigzag_tool/range_forecast/risk_corridor's own
// isXVisible() used to each hand-roll this identical one-liner, see project
// audit before the 2026-09 GitHub-readiness pass). Normally ("declutter"
// off) an object shows if it's pinned/shown OR its own tool is active — the
// long-standing "active ∪ pinned" policy. With the toggle ON (S.objectsHidden
// true — the button is meant to SOLO the current work, not blank the chart),
// every OTHER tool's pin is ignored: only the actually active tool's own
// objects remain, regardless of what's pinned (project feedback: the old
// "hide literally everything" behavior for this state was replaced with
// "show only what I'm working on right now").
export function isToolObjectVisible(type, ownShowFlag) {
  return S.objectsHidden ? S.activeMainTool === type : (ownShowFlag || S.activeMainTool === type);
}
