import { S } from './state.js';

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
// app.js). Selecting a tool is INDEPENDENT of whether its output is
// actually drawn on the chart — visibility (showOnChart/showOscillator/pin)
// is each tool's own explicit, persistent toggle, so several tools' objects
// can be visible in any combination regardless of which one's
// panel/toolbar-selection is currently active.
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
//   showCheckboxId:        DOM id of this tool's "показать осциллятор" checkbox
//   buildMainTraces():     -> Plotly traces for the MAIN chart (own
//                             "показывать на графике" flag checked INSIDE
//                             the tool's own function, same pattern
//                             buildZigzagTrace already used)
//   buildSubpanelTraces(): -> Plotly traces for the shared yaxis2 slot,
//                             called only when S.subpanel === this type
//   buildSubpanelShapes(): -> Plotly shapes for the shared yaxis2 slot
//                             (e.g. spectrogram's filter-bank cutoff lines)
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
// only one tool's oscillator can occupy it. Claiming it for `type`
// unchecks every OTHER registered tool's own "show oscillator" checkbox,
// so two checkboxes never end up both showing "checked" while only one is
// actually visible (project feedback 2026-08-18). Scales to N tools with
// zero added coupling per new one — each just needs showCheckboxId set at
// registration.
export function claimSubpanel(type) {
  for (const a of TOOLS) {
    if (a.type !== type && a.showCheckboxId) {
      const el = document.getElementById(a.showCheckboxId);
      if (el) el.checked = false;
    }
  }
  S.subpanel = type;
}

export function releaseSubpanel(type) {
  if (S.subpanel === type) S.subpanel = 'none';
}

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
// already-active one deselects it, leaving nothing active (no equivalent of
// 'free' there — that surface has no origin/click-routing concept at all,
// see tools.js module docstring). On the MAIN surface, clicking the
// already-active one instead falls back to 'free' (cursor_tools.js) rather
// than null — S.activeMainTool is never null (project feedback round 8:
// a null/absent state had no toolbar icon to show as active, which read as
// broken once 'free' became a real, always-selectable entry). Clicking
// 'free' while it's already active is therefore a harmless no-op (falls
// back to itself). Purely UI-focus bookkeeping: which panel shows, which
// tool's origin a click sets (main surface only), and whether the shared
// "reset origin" toolbar button is relevant — never touches any tool's own
// display/data settings.
export function selectTool(type) {
  const a = getTool(type);
  if (!a) return;
  if (surfaceOf(a) === 'main') {
    S.activeMainTool = S.activeMainTool === type ? 'free' : type;
  } else {
    S.activeOscillatorTool = S.activeOscillatorTool === type ? null : type;
  }
  renderToolbar();
}

// Non-toggling variant — sets a tool as active outright regardless of
// current state, used by forecast_history.js when loading/selecting a
// forecast (its results should be visible without a separate manual click
// on the toolbar icon, unlike the toggle behavior a direct icon click gets).
export function activateTool(type) {
  const a = getTool(type);
  if (!a) return;
  if (surfaceOf(a) === 'main') S.activeMainTool = type;
  else S.activeOscillatorTool = type;
  renderToolbar();
}

export function renderToolbar() {
  for (const a of TOOLS) {
    const surface = surfaceOf(a);
    const isActive = a.type === activeToolFor(surface);
    document.getElementById(`tool-btn-${a.type}`)?.classList.toggle('active', isActive);
    const panel = document.getElementById(a.panelId);
    if (panel) panel.style.display = isActive ? '' : 'none';
  }

  // No "nothing selected" hint on the main surface any more — S.activeMainTool
  // is never null ('free' plays that role explicitly, see selectTool), so
  // its own tool-panel-free always covers that spot instead. Oscillator
  // keeps its own hint (that surface genuinely can have nothing selected —
  // no 'free' equivalent there, see module docstring).
  const activeMain = getTool(S.activeMainTool);
  const hintOsc = document.getElementById('no-tool-selected-hint-osc');
  if (hintOsc) hintOsc.style.display = S.activeOscillatorTool ? 'none' : '';

  const resetBtn = document.getElementById('tool-reset-origin-btn');
  if (resetBtn) resetBtn.style.display = activeMain?.resetOrigin ? '' : 'none';
}

// Bound to the Основной график toolbar's single shared "reset origin"
// button — acts on whichever MAIN-surface tool is currently active
// (analyzer OR forecaster — though no forecaster registers resetOrigin, see
// module docstring), so N tools with an origin concept don't each need
// their own duplicated reset button in their own panel.
export function resetActiveToolOrigin() {
  getTool(S.activeMainTool)?.resetOrigin?.();
}
