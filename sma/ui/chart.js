import { S, MONTH_RU, DAY_RU, INTERVAL_SECONDS } from './state.js';
import { TOOLS } from './tools.js';
import { hexToRgbTriplet, hexToRgba } from './color_utils.js';

// ── constants ─────────────────────────────────────────────────────────────────
const CHART_BG     = '#0d1117';
const CHART_GRID   = '#1c2129';
const CHART_X_BARS = 200;   // bars shown on fresh load / reset
const CHART_X_PAD  = 5;     // empty bars of space to the right of last candle
const CHART_Y_PAD  = 0.06;  // 6% price padding above and below visible range

// Last CHART_X_BARS bars + right padding of CHART_X_PAD bar-lengths
function defaultXRange(c) {
  if (!c.length) return null;
  const start = c.length > CHART_X_BARS ? c.at(-CHART_X_BARS).begin : c[0].begin;
  const sec   = INTERVAL_SECONDS[S.interval] ?? 86400;
  const endMs = new Date(c.at(-1).begin.slice(0, 10)).getTime() + CHART_X_PAD * sec * 1000;
  return [start, new Date(endMs).toISOString().slice(0, 10)];
}

// Y range covering candles visible within xRange (or last CHART_X_BARS as fallback)
// — ALWAYS in linear price space, regardless of the axis's current type.
function computeYForXRange(xRange) {
  const c = S.candles;
  let vis;
  if (xRange) {
    const x0 = String(xRange[0]).slice(0, 10);
    const x1 = String(xRange[1]).slice(0, 10);
    vis = c.filter(r => { const d = r.begin.slice(0, 10); return d >= x0 && d <= x1; });
  }
  if (!vis?.length) vis = c.slice(-CHART_X_BARS);
  if (!vis.length)  return null;
  const yMin = Math.min(...vis.map(r => r.low));
  const yMax = Math.max(...vis.map(r => r.high));
  const pad  = (yMax - yMin) * CHART_Y_PAD;
  return [yMin - pad, yMax + pad];
}

// Converts a LINEAR price range into whatever space the y-axis is currently
// rendering in — a no-op when S.mainLogScale is off, log10 of each bound
// when it's on. Plotly.js requires 'yaxis.range' in log10 units whenever
// 'yaxis.type' is 'log'; computeYForXRange() (and any other linear-space
// range) must always be passed through this before being applied as a
// range, or the axis lands on a nonsense zoom (project feedback
// 2026-08-18: "баг с масштабированием при переключении из логарифмического
// режима... сброс осей... работает некорректно в логарифическом
// отображении" — root cause traced to exactly this: the "Сброс осей"/
// "Автомасштаб Y" modebar buttons, and the fresh-candle-load default range,
// all applied computeYForXRange()'s LINEAR output directly regardless of
// axis type). safeLo guards against a non-positive lower bound (log10 of
// zero/negative is undefined) — falls back to a tiny fraction of the upper
// bound rather than clipping the range to nothing.
function toAxisYRange(linearRange) {
  if (!linearRange || !S.mainLogScale) return linearRange;
  const [lo, hi] = linearRange;
  const safeLo = lo > 0 ? lo : hi * 1e-6;
  return [Math.log10(safeLo), Math.log10(hi)];
}

const PLOTLY_CONFIG = {
  responsive:   true,
  displaylogo:  false,
  scrollZoom:   true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'resetScale2d'],
  modeBarButtonsToAdd: [
    {
      name: 'resetAxes', title: 'Сброс осей',
      icon: Plotly.Icons.home,
      click(gd) {
        const xr = defaultXRange(S.candles);
        if (!xr) return;
        Plotly.relayout(gd, { 'xaxis.range': xr, 'yaxis.range': toAxisYRange(computeYForXRange(xr)) });
      },
    },
    {
      name: 'autoscaleY', title: 'Автомасштаб Y',
      icon: Plotly.Icons.autoscale,
      click(gd) {
        const yr = toAxisYRange(computeYForXRange(gd._fullLayout?.xaxis?.range ?? null));
        if (yr) Plotly.relayout(gd, { 'yaxis.range': yr });
      },
    },
  ],
};

function buildXAxis(rangeOverride = null) {
  const cfg = {
    type: 'date',
    gridcolor: CHART_GRID,
    linecolor: '#30363d',
    rangeslider: { visible: false },
    showspikes: true,
    spikecolor: '#8b949e',
    spikedash: 'dot',
    spikethickness: 1,
  };
  if (rangeOverride) cfg.range = rangeOverride;
  return cfg;
}

const BASE_YAXIS = {
  gridcolor: CHART_GRID,
  linecolor: '#30363d',
  side: 'right',
  showspikes: true,
  spikecolor: '#8b949e',
  spikedash: 'dot',
  spikethickness: 1,
};

const baseLayout = {
  dragmode:      'pan',
  paper_bgcolor: CHART_BG,
  plot_bgcolor:  CHART_BG,
  font:          { color: '#c9d1d9', size: 11 },
  margin:        { t: 16, r: 60, b: 36, l: 10 },
  // Plotly defaults multi-trace bars to 'group' (same x slot split into
  // narrower side-by-side bars) — 'overlay' instead draws each bar trace
  // full-width at its own x, which is what volume_oscillator.js's buysell
  // mode wants (a positive buy bar and a negative sell bar stacked on the
  // SAME x, not squeezed side-by-side). Every other panel here only ever
  // has one bar trace at a time, so this is a no-op for them.
  barmode:       'overlay',
  showlegend:    true,
  legend: {
    bgcolor: 'rgba(0,0,0,0.4)', bordercolor: '#30363d', borderwidth: 1,
    x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', font: { size: 11 },
  },
  hovermode: 'x unified',
};


function buildDivider() {
  if (S.subpanel === 'none') return [];
  return [{
    type: 'line', xref: 'paper', yref: 'paper',
    x0: 0, x1: 1, y0: 0.20, y1: 0.20,
    line: { color: '#3d444d', width: 1 },
    layer: 'above',
  }];
}

// Shapes for the active subpanel analyzer — generic dispatch through the
// registry, see tools.js (none currently implement this hook; kept as a
// registry-level capability, same as buildSubpanelTraces, for whichever
// future analyzer needs reference lines/shapes on the shared subpanel).
function buildSubpanelShapes() {
  if (S.subpanel === 'none') return [];
  return TOOLS.find(a => a.type === S.subpanel)?.buildSubpanelShapes?.() ?? [];
}

// Every registered analyzer's own main-chart origin crosshair (only the
// ones with an origin concept AND currently shown draw anything — each
// analyzer's buildOriginShape checks its own state, see e.g.
// trend_ruler.js:buildTrendRulerOriginShape).
function buildAnalyzerOriginShapes() {
  const shapes = [];
  for (const a of TOOLS) {
    if (a.buildOriginShape) shapes.push(...a.buildOriginShape());
  }
  return shapes;
}

// ── HTML/CSS overlay refreshers (bypass Plotly's shapes/annotations
// entirely) ──────────────────────────────────────────────────────────────
// One-directional registration hook, same cycle-avoidance reasoning as the
// TOOLS registry (cursor_tools.js imports FROM this file and registers its
// renderer INTO it, so chart.js never needs to import cursor_tools.js
// back) — for callers that render their own plain absolutely-positioned DOM
// elements, positioned via priceToPixelY() below, instead of going through
// Plotly's shapes/annotations layout at all. Currently just the price-level
// cursor tool's line+label (cursor_tools.js:renderPriceLevelOverlay) — it
// used to be a registered Plotly shape+annotation provider (see git history
// if curious) until two separate problems, both project feedback
// 2026-08-25: the price LABEL silently failed to appear on a page load
// already in log scale (root cause never confirmed despite checking
// Plotly's actual source — see priceToPixelY's docstring), and mousemove-
// driven Plotly.relayout calls fought Plotly's own drag-tracking state
// during an interactive pan ("уровни 'плывут'... шкала отвязывается").
// refreshOverlays() runs after every paint AND on plotly_relayouting/
// plotly_relayout (see _bindClickIfNeeded) so overlay elements track
// pan/zoom smoothly WITHOUT this file ever calling Plotly.relayout from a
// mousemove handler again — a plain DOM read+write can't conflict with
// Plotly's own interaction state the way a competing relayout call could.
const _overlayRefreshers = [];

export function registerOverlayRefresher(fn) {
  _overlayRefreshers.push(fn);
}

function refreshOverlays() {
  const gd = document.getElementById('chart');
  if (!gd) return;
  for (const fn of _overlayRefreshers) fn(gd);
}

// Same one-directional registration idea as registerOverlayRefresher just
// above (chart.js never imports the registering module) — fired once a
// pan/zoom gesture SETTLES (plotly_relayout, not the continuous
// plotly_relayouting), currently only consumed by candle_window.js to check
// whether the user just panned close enough to the left edge of the loaded
// window to fetch an older chunk.
const _relayoutHooks = [];

export function registerRelayoutHook(fn) {
  _relayoutHooks.push(fn);
}

// Full shapes array: divider + active analyzer's subpanel shapes + every
// analyzer's origin crosshair + user shapes (S.shapes). The "declutter"
// toggle (S.objectsHidden, see cursor_tools.js:toggleObjectsHidden) only
// affects the MAIN chart's own objects — the subpanel (divider/
// buildSubpanelShapes) is a separate area, left alone. Each analyzer's own
// buildOriginShape (e.g. trend_ruler's) already gates itself through its own
// isXVisible(), which itself defers to tools.js:isToolObjectVisible — no
// extra gate needed here, so this always calls buildAnalyzerOriginShapes().
function allShapes() {
  return [...buildDivider(), ...buildSubpanelShapes(), ...buildAnalyzerOriginShapes(), ...S.shapes];
}

// Builds a full layout with dynamic yaxis domains based on active subpanel.
// Pass y1Range to lock the main price axis (prevents autorange on Plotly.react).
// y2 params come from applyY2RangePolicy() below — see its docstring for why
// this needs its own preserve-across-renders logic distinct from y1Range.
function buildLayout(xRange, y1Range = null, y2Range = null, y2Fixed = false) {
  const hasSub = S.subpanel !== 'none';
  const logY = S.subpanel === 'spectrogram' && S.spectrogramSettings.logY;
  // Plotly's own bar-snapped crosshair (showspikes) is replaced by a plain
  // cursor-following horizontal line while price_level is active — project
  // feedback 2026-08-25: "перекрестие... для ценового уровня оно только
  // путает" — see chart.js:_bindPriceLevelEventsIfNeeded for the line
  // itself. Both axes' spikes are suppressed (not just Y) — the vertical
  // one is equally irrelevant for a horizontal-only tool.
  const priceLevelActive = S.activeTab === 'main' && S.activeMainTool === 'price_level';

  return {
    ...baseLayout,
    shapes: allShapes(),
    annotations: S.bandAnnotations ?? [],
    xaxis: { ...buildXAxis(xRange), ...(priceLevelActive ? { showspikes: false } : {}) },
    yaxis: {
      ...BASE_YAXIS,
      type: S.mainLogScale ? 'log' : 'linear', // main chart's own log toggle — independent of spectrogram's logY (that one's the oscillator's frequency axis), see sma/ui/cursor_tools.js:toggleMainLogScale
      domain: hasSub ? [0.22, 1.0] : [0.0, 1.0],
      ...(priceLevelActive ? { showspikes: false } : {}),
      ...(y1Range ? { range: y1Range } : {}),
    },
    yaxis2: {
      domain: [0.0, 0.18],
      gridcolor: CHART_GRID,
      linecolor: '#30363d',
      tickcolor: '#30363d',
      tickfont:  { color: '#8b949e', size: 10 },
      side: 'right',
      fixedrange: y2Fixed,
      visible: hasSub,
      showticklabels: hasSub,
      ...(y2Range ? { range: y2Range } : {}),
      ...(logY ? { type: 'log' } : {}),
    },
  };
}

// Per-tool y2 (subpanel) centering policy (project feedback 2026-08-20, §3.3
// of docs/plans/frontend_improvements_plan.md — "спектрограмма: полностью
// фиксировано; тренд: 0 строго посередине, зум доступен; дисперсия: 0 строго
// внизу, зум не нужен"). Generic dispatch through the registry, same pattern
// as buildSubpanelTraces — a tool's own subpanelYAxisPolicy() returns
// { key, fixedrange, range }: `key` names the current MODE (e.g.
// 'variance_osc:slope' vs 'variance_osc:var' — switching between two modes
// of the SAME tool still needs a fresh range, not the old mode's stale one)
// and `range` is [lo, hi] or null when there's no data yet to base one on.
function activeY2Policy() {
  if (S.subpanel === 'none') return null;
  return TOOLS.find(a => a.type === S.subpanel)?.subpanelYAxisPolicy?.() ?? null;
}

// A fixedrange tool (spectrogram only, as of 2026-08-20 — variance
// dispersion mode and volume used to be fixedrange too, but project
// feedback that day asked for zoom there as well: "дисперсия и объем — нет
// возможности зумировать") has no interactive state to preserve — so
// recomputing its range fresh every render is not just safe, it's correct
// (picks up new data immediately, e.g. a fresh spectrogram calc with
// different fmin/fmax). Every OTHER tool/mode DOES have interactive state
// worth preserving, the same problem y1Range already solves for the main
// price axis via getCurrentRanges() — but reusing that exact technique
// naively breaks the moment the SUBPANEL's owning tool (or mode) changes:
// yaxis2._fullLayout still holds the PREVIOUS tool's range value until this
// render overwrites it, so "preserve current" would silently inherit e.g.
// spectrogram's frequency range as variance_osc's initial view. _lastY2
// tracks whose/which-mode's range is currently live (and whether it was
// ever actually applied — see its own docstring for a related bug this
// caught) so a change forces a fresh compute exactly once, then preserves
// user zoom across every unrelated re-render after that (settings
// elsewhere, forecast selection, ...) same as y1Range does.
// Tracks BOTH the key AND whether the last render actually had a real
// range to apply — not just the key alone (bug found 2026-08-20, project
// feedback: "осцилятор дисперсии (тренд) — нет фиксирования 0
// посередине"). A tool picked for the first time renders ONCE before its
// data has arrived (S.varianceOscData is still null right after
// selectTool's synchronous renderChart, the actual fetch resolves later,
// see variance_oscillator.js:onVarianceOscSelected) — that first render
// already stamps `key` into this tracker even though `policy.range` was
// null (nothing to show yet). Comparing only `key` on the SECOND render
// (data now loaded, a real symmetric range finally computed) saw the same
// key as "unchanged" and preserved the first render's meaningless
// autorange instead of applying the freshly computed centered range — so
// the axis never actually got recentered, only LOOKED like it might on the
// very next unrelated interaction. Tracking hadRange too forces a fresh
// apply on that specific "went from no data to real data" transition,
// while still preserving user zoom on every later re-render once real data
// is already showing.
let _lastY2 = { key: null, hadRange: false };

function applyY2RangePolicy(preserveRange, currentY2Range) {
  const policy = activeY2Policy();
  const key = policy?.key ?? null;
  const hasRange = !!policy?.range;
  const changed = key !== _lastY2.key || !_lastY2.hadRange;
  _lastY2 = { key, hadRange: hasRange };
  if (!policy || !hasRange) return { y2Range: null, y2Fixed: !!policy?.fixedrange };
  const y2Range = (!policy.fixedrange && preserveRange && !changed && currentY2Range)
    ? currentY2Range
    : policy.range;
  return { y2Range, y2Fixed: !!policy.fixedrange };
}

// Traces for the shared subpanel slot (yaxis: 'y2') — generic dispatch
// through the registry (see tools.js); the actual heatmap/bar/scatter
// building lives in each analyzer's own module (analysis.js, variance_oscillator.js, ...)
// now, not here — chart.js only knows "ask whoever owns S.subpanel".
function buildSubpanelTraces() {
  if (S.subpanel === 'none') return [];
  return TOOLS.find(a => a.type === S.subpanel)?.buildSubpanelTraces?.() ?? [];
}

// Every registered analyzer's own main-chart traces (trend lines, bands,
// markers, ...) — each analyzer's buildMainTraces checks its own
// "показывать на графике" setting internally and returns [] when off, same
// pattern buildZigzagTrace already used for S.zigzagPivots. Any number can
// be visible at once (independent per-analyzer toggle, not mutually
// exclusive like the subpanel) — see tools.js module docstring.
function buildAnalyzerMainTraces() {
  const traces = [];
  for (const a of TOOLS) {
    if (a.buildMainTraces) traces.push(...a.buildMainTraces());
  }
  return traces;
}

// ── helpers ───────────────────────────────────────────────────────────────────

// Reads the actual rendered axis ranges from _fullLayout (includes user zoom/pan).
// Falls back to the user-provided layout, then null for each axis.
export function getCurrentRanges() {
  const el = document.getElementById('chart');
  const fl = el?._fullLayout;
  const ly = el?.layout;
  return {
    x:  fl?.xaxis?.range  ?? ly?.xaxis?.range  ?? null,
    y1: fl?.yaxis?.range  ?? ly?.yaxis?.range  ?? null,
    y2: fl?.yaxis2?.range ?? ly?.yaxis2?.range ?? null,
  };
}

export function dayLabel(beginStr) {
  const d = new Date(beginStr.slice(0, 10) + 'T12:00:00');
  return `${d.getDate()} ${MONTH_RU[d.getMonth()]} ${d.getFullYear()}, ${DAY_RU[d.getDay()]}`;
}

// Position of a forecast step `stepsAhead` bars past `baseDateStr` — used by
// simplex_ensemble/range_forecast/trend_ruler for steps beyond a fixed h=1/
// h=2 geometry (band_lambda doesn't need this — computeGeometryFromWidth
// stays inside real bar-count geometry).
//
// Used to guess trading-day positions via a hardcoded Sat/Sun skip (pandas
// BDay-style) — REMOVED 2026-09-06 (project finding, ticker BSPB: some
// tickers DO trade weekends, so "weekend = never a trading day" was
// confidently wrong, not just imprecise, for those). New rule instead:
//   1. If `candles` (S.candles) is given and a REAL bar already exists at
//      the target position (this origin sits before the end of history —
//      a historical replay/ensemble-member origin, not the live one),
//      return that bar's EXACT timestamp — correct by construction for
//      ANY trading calendar quirk, no guessing needed at all.
//   2. Otherwise (genuinely beyond known history — nothing to snap to),
//      extrapolate with a STRICT fixed step — exactly 1 bar of `interval`
//      (INTERVAL_SECONDS), same for every step, no adaptive/median guess.
//      A per-ticker median step was tried and explicitly rejected by the
//      user (2026-09-06): a variable step just adds a second kind of
//      unpredictability on top of "no calendar assumption" — harder to
//      reason about than a plain, always-the-same interval — and makes no
//      stronger claim about which real dates are trading days either way.
// Resolves S.originTs to the actual candle row backing it. S.originTs is a
// 10-char date (the click handler always truncates to date, regardless of
// interval — same limitation band_lambda's own pivot matching has for
// intraday intervals); candle.begin may carry a full timestamp, hence the
// slice comparison. When a date has several intraday bars, the LAST one is
// "the last known bar" for that day — matches every live-forecast tool's
// own semantics (origin = last known bar, h=1 predicts the bar after it).
// Every tool needing a request origin used to redefine this identical
// function locally (range_forecast.js/risk_corridor.js/simplex_ensemble.js/
// regime_mixture_potential.js); one shared copy here instead.
export function resolveOriginCandle() {
  if (!S.originTs) return null;
  const matches = S.candles.filter(c => c.begin.slice(0, 10) === S.originTs);
  return matches.length ? matches[matches.length - 1] : null;
}

export function futureDateAt(baseDateStr, stepsAhead, interval, candles) {
  if (candles && candles.length) {
    let baseIdx = candles.findIndex(c => c.begin === baseDateStr);
    if (baseIdx < 0) {
      const d10 = baseDateStr.slice(0, 10);
      for (let i = candles.length - 1; i >= 0; i--) {
        if (candles[i].begin.slice(0, 10) === d10) { baseIdx = i; break; }
      }
    }
    if (baseIdx >= 0 && candles[baseIdx + stepsAhead]) {
      return candles[baseIdx + stepsAhead].begin;
    }
  }
  const base = new Date(baseDateStr.slice(0, 10) + 'T00:00:00Z');
  const stepMs = (INTERVAL_SECONDS[interval] ?? 86400) * 1000;
  return new Date(base.getTime() + stepsAhead * stepMs).toISOString();
}

// ── shape factories ───────────────────────────────────────────────────────────

// Returns [y0, y1] clipped to the main chart domain (above subpanel if
// active) — exported so analyzer modules can build their own origin
// crosshairs at the same vertical extent (e.g. trend_ruler.js:
// buildTrendRulerOriginShape) without duplicating the 0.22/1.0 domain
// split constants.
export function paperY() {
  return S.subpanel !== 'none' ? [0.22, 1.0] : [0.0, 1.0];
}

export function originLineShape(ts) {
  const [y0, y1] = paperY();
  return {
    type: 'line', x0: ts, x1: ts, y0, y1, yref: 'paper',
    line: { color: S.colorProfile.next_origin_marker, width: 1, dash: 'dash' }, // "Точка отсчёта нового прогноза" role — settings.js
  };
}

export function applyShapes() {
  Plotly.relayout('chart', {
    shapes: allShapes(),
    annotations: S.bandAnnotations ?? [],
  });
}

// Switches the main y-axis between linear/log — owns BOTH the actual
// Plotly.relayout AND the S.mainLogScale flag, so cursor_tools.js's toggle
// button just calls this instead of reaching into chart.js internals.
// Cancels any renderChart() already QUEUED via schedulePlotlyReact before
// doing the switch: that queued call captured its layout (including
// 'yaxis.type'/'yaxis.range') from BEFORE this toggle, at the OLD
// S.mainLogScale value — letting it fire afterward (same or next animation
// frame) would silently overwrite this relayout with a stale, wrong-space
// range, which is what "баг с масштабированием при переключении из
// логарифмического режима" (project feedback 2026-08-18) traced back to on
// top of the toAxisYRange bug above. 'yaxis.autorange:true' (rather than
// hand-converting the previous range to the new space) is the
// Plotly-documented safe way to switch a log/linear axis type.
// Keeps the VISIBLE price window across the switch: the current y range is
// converted into the new scale instead of autorange. Autorange re-fits to
// every trace — the potential heatmap's grid reaches toward zero, so going
// back to linear used to squash the price chart (drift on toggle).
export function setMainLogScale(enabled) {
  const fl = document.getElementById('chart')?._fullLayout;
  const curRange = fl?.yaxis?.range;
  const wasLog = fl?.yaxis?.type === 'log';
  S.mainLogScale = enabled;
  if (_rafHandle) {
    cancelAnimationFrame(_rafHandle);
    _rafHandle = null;
    _pendingPlotArgs = null;
  }
  // Plotly reports the range in the axis' current units; go through linear
  // prices so the conversion is the same one renderChart uses (toAxisYRange).
  let yRange = null;
  if (curRange) {
    const linear = wasLog ? curRange.map(v => 10 ** v) : curRange;
    yRange = toAxisYRange(linear);
  }
  Plotly.relayout('chart', {
    'yaxis.type': enabled ? 'log' : 'linear',
    ...(yRange ? { 'yaxis.range': yRange } : { 'yaxis.autorange': true }),
  });
}

// Places the plain origin marker without touching whatever forecast overlays
// (pinned/selected, or solo-active-tool-only while "показать/скрыть активные
// объекты" is decluttering — see buildVisibleOverlays) are currently
// showing — recomputes their shapes fresh so the new origin line doesn't
// clobber them.
// Clears the MAIN origin (S.originTs, the "next forecast starts here" line).
// Used by the Курсор reset button for tools that have no origin of their own.
// Every origin change goes through notifyMainOriginChanged so the ACTIVE
// tool can recompute whatever depends on it (live results, button labels).
function notifyMainOriginChanged() {
  const active = TOOLS.find(a => a.type === S.activeMainTool);
  active?.onMainOriginChanged?.();
}

export function clearMainOrigin() {
  const had = !!S.originTs;
  S.originTs = null;
  if (had) renderChart({ preserveRange: true });
  notifyMainOriginChanged(); // also when it was already empty — the tool may still need its live recalc
}

export function setOrigin(ts) {
  S.originTs = ts;
  const { shapes } = buildVisibleOverlays();
  S.shapes = [...shapes, originLineShape(ts)];
  applyShapes();
  notifyMainOriginChanged();
}

// ── zigzag overlay (band_lambda) ────────────────────────────────────────────
// Drawn at extreme_date (where the reversal actually printed) — confirm_date
// (the causal point) is used only for the forecast origin marker, never for
// the zigzag line itself. See sma/core/forecast/band_lambda.py:build_zigzag.
// `bold`/`name` support §2.4 of docs/plans/frontend_improvements_plan.md — a
// T that's pinned (S.zigzagPinnedTs) draws thicker so it stands out from an
// unpinned, merely-currently-selected T's zigzag when several are visible
// together (see buildForecastZigzagTraces below).
export function buildZigzagTrace(pivots, { bold = false, name = 'Зигзаг' } = {}) {
  if (!pivots?.length) return null;
  const color = S.colorProfile.forecast_zigzag; // "Прогноз-зигзаг (band_lambda)" role — see settings.js
  return {
    type: 'scatter', name,
    x: pivots.map(p => p.extreme_date),
    y: pivots.map(p => p.price),
    mode: 'lines+markers',
    line: { color, width: bold ? 2.5 : 1 },
    marker: { size: bold ? 6 : 4, color },
    hovertemplate: '%{y:.4f}<extra></extra>',
  };
}

// Composes every band_lambda forecast-zigzag trace that should currently be
// on the chart: the actively selected T (only while band_lambda is the
// active tool — same "active ∪ pinned" policy §2.6 already applies
// elsewhere) plus every T explicitly pinned via toggleZigzagPin
// (forecast.js), regardless of active tool. A T that's both (selected AND
// pinned) draws once, bold — never duplicated. Pinned-only T's use their own
// cached pivots (S.zigzagPinnedData, populated when pinned — see
// forecast.js:toggleZigzagPin) since S.zigzagPivots only ever holds the
// CURRENTLY selected T's data.
function buildForecastZigzagTraces() {
  const traces = [];
  const shown = new Set();
  const activeSelected = S.activeMainTool === 'band_lambda' && S.selectedTs != null;
  if (activeSelected) {
    const bold = S.zigzagPinnedTs.has(S.selectedTs);
    const t = buildZigzagTrace(S.zigzagPivots, { bold, name: `Зигзаг T=${(S.selectedTs * 100).toFixed(0)}%` });
    if (t) { traces.push(t); shown.add(S.selectedTs); }
  }
  // "showing every pinned T regardless of active tool" is itself part of the
  // "active ∪ pinned" policy — same "показать/скрыть активные объекты" solo
  // rule as buildVisibleOverlays applies: while S.objectsHidden is soloing
  // the active tool, a pin from band_lambda not currently being worked on
  // shouldn't leak through here either.
  if (!S.objectsHidden) {
    for (const tQuery of S.zigzagPinnedTs) {
      if (shown.has(tQuery)) continue;
      const pivots = S.zigzagPinnedData[tQuery];
      const t = buildZigzagTrace(pivots, { bold: true, name: `Зигзаг T=${(tQuery * 100).toFixed(0)}%` });
      if (t) traces.push(t);
    }
  }
  return traces;
}

// ── band zones ───────────────────────────────────────────────────────────
// Faithful port of prototype/forcaster/ui/app9.py's build_zones (2026-07-08
// design, re-confirmed by the user 2026-08-07 as the reference to match):
//   - color is decided by ZIGZAG ALTERNATION, not a computed median. A
//     confirmed pivot's NEXT leg (step1) always runs opposite the pivot's
//     own direction (that's what makes it a confirmed reversal) — origin
//     HIGH -> step1 down (red) -> step2 back up (green); origin LOW -> step1
//     up (green) -> step2 back down (red). This is a structural fact of the
//     zigzag, not something to infer per-forecast.
//   - each enabled level (e.g. 50/75/90%) is ONE central-interval rect at
//     FIXED opacity, widest drawn first / narrowest last so Plotly's own
//     alpha compositing makes the center read denser — no manual gradient
//     formula, no per-ring borders (matches app9 exactly: line width=0).
// "Зона вероятности: рост/падение" roles (settings.js) — read live (not
// cached as a module-level const any more, project feedback 2026-08-25:
// "вообще все используемые инструментами цвета настраивать") so a profile
// save recolors existing zones immediately. Returns the bare "r,g,b"
// triplet buildBandZoneShapes below wraps in its own rgba(...,opacity).
function colorDown() { return hexToRgbTriplet(S.colorProfile.band_zone_down); }
function colorUp() { return hexToRgbTriplet(S.colorProfile.band_zone_up); }
export const LEVEL_OPTIONS = [50, 75, 90];
export const DEFAULT_LEVELS = [50, 75, 90];
export const DEFAULT_ZONE_OPACITY = 0.22;
export const DEFAULT_TRADE_LEVEL_PCT = 70;

export const DEFAULT_STEP_BARS = 5;

// Horizontal span for the two step zones, driven by the two "Ширина шага"
// sliders (sma/ui/forecast.js:applyBandWidth) — replaces an earlier
// drag-to-resize attempt that turned out unreliable (1px shape borders are
// very hard to grab with a mouse; see project feedback 2026-08-07). Kept
// as a named export so forecast.js can compute the same geometry the
// sliders persist via POST /forecasts/{id}/geometry. width1Bars/width2Bars
// are independent (project feedback 2026-08-26: up/down leg durations are
// asymmetric — see result.default_step_widths_bars, sma/core/forecast/
// pivot_time_band.py — so one shared width no longer makes sense). The
// widths are embedded in the returned geometry itself so a forecast
// re-selected later can restore its slider positions without reverse-
// engineering them from x0/x1 dates.
export function computeGeometryFromWidth(
  originTs, interval, width1Bars = DEFAULT_STEP_BARS, width2Bars = DEFAULT_STEP_BARS,
) {
  const sec = INTERVAL_SECONDS[interval] ?? 86400;
  const base = new Date(originTs.slice(0, 10)).getTime();
  const at = n => new Date(base + n * sec * 1000).toISOString().slice(0, 10);
  return {
    step1: { x0: at(1), x1: at(width1Bars) },
    step2: { x0: at(width1Bars), x1: at(width1Bars + width2Bars) },
    width1_bars: width1Bars,
    width2_bars: width2Bars,
  };
}

// Weighted quantile — JS port of sma/core/forecast/band_lambda.py:
// weighted_quantile (same midpoint-corrected weighted-CDF interpolation),
// so the frontend can recompute ARBITRARY quantile levels reactively from
// the pool_values/pool_weights a forecast carries, without a new backend
// call — mirrors app9.py's "зоны пересчитываются реактивно... без
// повторной загрузки пула". Exported: sma/ui/simplex_ensemble.js reuses it
// (equal weights = ordinary quantile) to recompute its band-width slider
// without a new /forecasts call either.
export function weightedQuantile(values, weights, quantiles) {
  const n = values.length;
  const order = [...Array(n).keys()].sort((a, b) => values[a] - values[b]);
  const v = order.map(i => values[i]);
  const w = order.map(i => weights[i]);
  const totalW = w.reduce((a, b) => a + b, 0);
  if (totalW < 1e-14) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = sorted.length % 2 === 1
      ? sorted[(sorted.length - 1) / 2]
      : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
    return quantiles.map(() => mid);
  }
  let cum = 0;
  const cumW = w.map(wi => { cum += wi; return (cum - 0.5 * wi) / totalW; });
  return quantiles.map(q => npInterp(q, cumW, v));
}

function npInterp(x, xp, fp) {
  if (x <= xp[0]) return fp[0];
  if (x >= xp[xp.length - 1]) return fp[fp.length - 1];
  for (let i = 0; i < xp.length - 1; i++) {
    if (x >= xp[i] && x <= xp[i + 1]) {
      const denom = xp[i + 1] - xp[i];
      const t = denom === 0 ? 0 : (x - xp[i]) / denom;
      return fp[i] + t * (fp[i + 1] - fp[i]);
    }
  }
  return fp[fp.length - 1];
}

// Central [lo, hi] price interval covering `levelPct`% of the pool's
// probability mass around the median — e.g. levelPct=50 -> [25th, 75th]
// percentile, matching app9.py's build_zones exactly.
function levelBounds(step, originLogPrice, levelPct) {
  const frac = levelPct / 100;
  const [lrLo, lrHi] = weightedQuantile(step.pool_values, step.pool_weights, [(1 - frac) / 2, (1 + frac) / 2]);
  return { lo: Math.exp(originLogPrice + lrLo), hi: Math.exp(originLogPrice + lrHi) };
}

// "Граница шага 2" (UI label, project rename 2026-10-04 — formerly "уровень
// доверия", a less accurate name for what this actually is) — a single
// one-sided price threshold drawn on step1, but its VALUE is read from
// step2's pool (the "уход+возврат" distribution), not step1's own — the
// undershoot/overshoot tail of the round-trip return (recovery fell short
// of the origin for a decline setup, or overshot it for a rise setup).
// Uses the SAME symmetric-interval quantile as levelBounds (q=(1-frac)/2 /
// (1+frac)/2) — not an independent one-sided convention — so setting this
// to e.g. 90 lines up EXACTLY with the 90% zone's own edge when that zone
// is also shown, instead of silently meaning a different threshold under
// the same "%" label. Internal identifiers (tradeLevelPct/showTradeLevel/
// tradePrice, display_presets.trade_level_pct column) keep their original
// names — only the user-facing label changed, not the schema/API.
function tradeLevelPrice(step2, originLogPrice, tradeLevelPct, isDownStep1) {
  if (!step2?.ok || !step2.pool_values?.length) return null;
  const frac = tradeLevelPct / 100;
  const q = isDownStep1 ? (1 - frac) / 2 : (1 + frac) / 2;
  const [lr] = weightedQuantile(step2.pool_values, step2.pool_weights, [q]);
  return Math.exp(originLogPrice + lr);
}

export function buildBandZoneShapes(forecastResult, geometry, displayPreset) {
  const shapes = [];
  if (!forecastResult) return shapes;

  // Anchored to origin_extreme_date, NOT origin_date (confirmation bar) —
  // origin_price is the pivot's EXTREME price, a different bar entirely
  // from confirmation; placing zones at the confirm date would visually
  // detach them from the price they're actually centered on. origin_date
  // remains the causal anchor everywhere else (task truncation,
  // origin_candle_id resolution) — only the VISUAL placement changes here.
  const originTs = forecastResult.origin_extreme_date;
  const originLogPrice = forecastResult.origin_log_price;
  const defaultW = forecastResult.default_step_widths_bars;
  const geo = geometry || computeGeometryFromWidth(originTs, S.interval, defaultW?.step1, defaultW?.step2);
  const dp = displayPreset || {};
  // Widest first, narrowest last — later shapes draw on top, so same-alpha
  // overlapping rects naturally stack denser toward the center.
  const activeLevels = [...(dp.levels?.length ? dp.levels : DEFAULT_LEVELS)].sort((a, b) => b - a);
  const baseOpacity = dp.opacity ?? DEFAULT_ZONE_OPACITY;
  const showZones = dp.showZones ?? true;
  const showTradeLevel = dp.showTradeLevel ?? true;
  const trimZone1 = dp.trimZone1 ?? false;
  const tradeLevelPct = dp.tradeLevelPct ?? DEFAULT_TRADE_LEVEL_PCT;
  const direction = forecastResult.origin_direction;

  const isDownStep1 = direction > 0;
  const tradeColor = isDownStep1 ? colorDown() : colorUp();
  const tradePrice = (showTradeLevel || trimZone1)
    ? tradeLevelPrice(forecastResult.steps?.[2], originLogPrice, tradeLevelPct, isDownStep1)
    : null;

  if (showZones) {
    for (const h of [1, 2]) {
      const step = forecastResult.steps?.[h];
      const g = geo[`step${h}`];
      if (!step?.ok || !g || !step.pool_values?.length) continue;
      const isDown = h === 1 ? direction > 0 : direction < 0;
      const color = isDown ? colorDown() : colorUp();

      activeLevels.forEach(levelPct => {
        let { lo, hi } = levelBounds(step, originLogPrice, levelPct);
        // "Обрезать зону 1 по границе шага 2" — keep only the side of the
        // band beyond the step-2-boundary line, in the direction of the
        // extreme (below it for a decline/buy setup, above it for a
        // rise/short setup) — the sub-region actually implied by that line,
        // not the full unconditional quantile band.
        if (h === 1 && trimZone1 && tradePrice != null) {
          if (isDown) {
            if (lo >= tradePrice) return;
            hi = Math.min(hi, tradePrice);
          } else {
            if (hi <= tradePrice) return;
            lo = Math.max(lo, tradePrice);
          }
        }
        shapes.push({
          type: 'rect', xref: 'x', yref: 'y',
          x0: g.x0, x1: g.x1, y0: lo, y1: hi,
          fillcolor: `rgba(${color},${baseOpacity})`,
          line: { width: 0 },
          layer: 'above',
        });
      });
    }
  }

  if (showTradeLevel && tradePrice != null && geo.step1) {
    shapes.push({
      type: 'line', xref: 'x', yref: 'y',
      x0: geo.step1.x0, x1: geo.step1.x1, y0: tradePrice, y1: tradePrice,
      // Wide dash (not solid) — project request 2026-10-04: risk_calculator.js's
      // entry/stop/tp overlay lines are now solid, so this one needs to read
      // as visually distinct at a glance rather than conflict with them.
      line: { color: `rgb(${tradeColor})`, width: 2, dash: 'longdash' },
      layer: 'above',
    });
  }

  // Divider between step1 and step2 so two adjacent zones (often the same
  // color pair-wise across different forecasts) never read as one field.
  const g1 = geo.step1, g2 = geo.step2;
  if (showZones && g1 && g2 && forecastResult.steps?.[1]?.ok && forecastResult.steps?.[2]?.ok) {
    const [y0, y1] = paperY();
    shapes.push({
      type: 'line', xref: 'x', yref: 'paper',
      x0: g1.x1, x1: g1.x1, y0, y1,
      line: { color: '#6e7681', width: 1, dash: 'dot' },
      layer: 'above',
    });
  }
  return shapes;
}

// A horizontal dotted line at the exact origin price, paired with the
// existing vertical originLineShape — together they form a crosshair
// pinpointing the event's (date, price), addressing "прогноз должен
// ставиться в точку события зигзага" (a vertical line alone only says
// WHEN, not WHERE).
export function originPriceLineShape(price) {
  return {
    type: 'line', xref: 'paper', yref: 'y',
    x0: 0, x1: 1, y0: price, y1: price,
    // Same role as the selected forecast's marker highlight (§1.1a) — this
    // crosshair only ever gets drawn for S.selectedForecastId (see
    // buildVisibleOverlays), so it's visually "the same thing" as that
    // marker's color, not an independent role of its own.
    line: { color: S.colorProfile.forecast_marker_selected, width: 1, dash: 'dot' },
  };
}

// ── model registry: single extension point for a new forecast model ────────
// Every builder referenced here (buildBandZoneShapes/
// buildSimplexOriginTraces/buildSimplexMeanBandTraces) already lives in this
// file — kept as a plain local map (not a separate module) specifically to
// avoid a chart.js <-> forecast_models.js import cycle: the builders are
// chart.js internals, and the composer that dispatches through this map
// (buildVisibleOverlays, renderChart) also lives here. Adding a THIRD model
// means: write its own buildX traces/shapes function above, add one entry
// below with a distinct marker symbol — nothing else in this file changes.
// marker.colorRole names a key in S.colorProfile (docs/plans/
// frontend_improvements_plan.md §1.1a — settings.js/"Настройки приложения")
// instead of a literal color — pinnedMarkerColor() below resolves it live on
// every trace build, so a profile save recolors existing pinned markers
// immediately. Adding a THIRD model needs its own
// "forecast_marker_pinned_<model_type>" role added to
// state.js/db.py:DEFAULT_COLOR_PROFILE + settings.js's form alongside the
// two existing ones.
const MODEL_DISPLAY = {
  band_lambda: {
    label: 'Band Lambda',
    marker: { symbol: 'triangle-up', colorRole: 'forecast_marker_pinned_band_lambda' },
    buildOverlay(result, geometry) {
      return {
        shapes: buildBandZoneShapes(result, geometry, S.displayPreset),
      };
    },
  },
  simplex_ensemble: {
    label: 'Simplex-ансамбль',
    marker: { symbol: 'circle', colorRole: 'forecast_marker_pinned_simplex_ensemble' },
    buildOverlay(result) {
      const traces = [...buildSimplexOriginTraces(result)];
      if (S.simplexShowMean) traces.push(...buildSimplexMeanBandTraces(result, S.simplexBandPct));
      return { traces };
    },
  },
  regime_mixture_potential: {
    label: 'Потенциал',
    marker: { symbol: 'diamond', colorRole: 'forecast_marker_pinned_regime_mixture_potential' },
    buildOverlay(result) {
      return { traces: buildPotentialOverlayTraces(result) };
    },
  },
};

function pinnedMarkerColor(style) {
  return S.colorProfile[style.colorRole] ?? '#8b949e';
}

// Where a forecast marker sits relative to its origin BAR, and which way its
// symbol points — model-specific behavior (project feedback 2026-08-20/21:
// "маркеры прогнозов сейчас прямо на барах — не всегда заметны... для
// зигзага" — clarified afterward: "ниже минимума, выше максимума в
// зависимости от типа прогноза (падение, рост)" — i.e. band_lambda MIRRORS
// the pivot itself, not a fixed side): origin_direction>0 means the origin
// IS a HIGH pivot (the next leg goes down from here, see buildBandZoneShapes
// above) — marker sits ABOVE the bar's high, triangle tip pointing DOWN at
// it ("триугольник тоже можно поворачивать углом в сторону бара");
// origin_direction<0 is a LOW pivot — BELOW the low, tip pointing UP.
// simplex_ensemble has no pivot-direction concept of its own (its origin is
// just "the last known bar", not a zigzag reversal) — always BELOW the low,
// plain circle (no orientation to speak of). A model absent from this map
// (future addition) falls back to the old at-origin-price placement in
// buildForecastMarkerTraces below.
const MARKER_PLACEMENT = {
  band_lambda(candle, f) {
    const above = f.origin_direction > 0;
    return { basePrice: above ? candle.high : candle.low, sign: above ? 1 : -1, symbol: above ? 'triangle-down' : 'triangle-up' };
  },
  simplex_ensemble(candle) {
    return { basePrice: candle.low, sign: -1, symbol: 'circle' };
  },
  regime_mixture_potential(candle) {
    return { basePrice: candle.low, sign: -1, symbol: 'diamond' };
  },
};

// candle.begin -> candle lookup, rebuilt only when S.candles itself changes
// (ticker/interval switch) — buildForecastMarkerTraces needs a bar's
// high/low for every visible marker, and a linear S.candles.find() per
// marker would be wasteful across possibly dozens of forecasts on every
// cache-invalidating render.
let _candleByDateCache = { candles: null, map: null };

function candleByDateMap() {
  if (_candleByDateCache.candles === S.candles) return _candleByDateCache.map;
  const map = new Map();
  for (const c of S.candles) map.set(c.begin.slice(0, 10), c);
  _candleByDateCache = { candles: S.candles, map };
  return map;
}

// ── traces ────────────────────────────────────────────────────────────────────
const MARKER_STACK_STEP  = 0.006; // 0.6% of price per EXTRA forecast stacked at the same date/side
const MARKER_OFFSET_PCT  = 0.012; // 1.2% of price — base gap between the bar's high/low and the first marker there, so it reads as "pointing at" the bar rather than sitting on top of it

// One scatter trace PER MODEL (distinct symbol/color from MODEL_DISPLAY),
// one marker per past forecast of that model — placed at (extreme_date,
// origin_price), the event itself. Forecasts sharing the same event (re-run
// at the same pivot, or a live forecast re-run before new data arrived) land
// on the EXACT same point — stacked downward per extra forecast at that date
// so each stays distinguishable/clickable (old LA-model marker-stack
// behaviour, "важно чтобы для разных T в одной точке прогнозы также
// отображались", 2026-08-07). Opacity/size encode pin/selection state (see
// S.pinnedForecastIds/S.selectedForecastId) so a glance at the chart shows
// which markers are "on".
// Pure function of (S.historyForecasts, S.pinnedForecastIds,
// S.selectedForecastId, S.activeMainTool) — cached so a renderChart
// triggered by something entirely unrelated (an analyzer's own settings)
// doesn't re-sort/re-group the full forecast history every time.
// S.historyForecasts is reassigned wholesale on every fetch
// (forecast_history.js:refreshHistory), so identity works for it;
// S.pinnedForecastIds is a Set mutated IN PLACE (.add/.delete/.clear), so
// its identity never changes — comparing a sorted snapshot string instead
// (cheap: only as many entries as are actually pinned, not the full history).
let _markerTracesCache = { forecasts: null, pinnedKey: null, selected: undefined, activeTool: undefined, traces: null, pointsByDate: null };

// Visibility policy (project feedback 2026-08-19, §2.6/§2.7 of
// docs/plans/frontend_improvements_plan.md): a forecast's marker is only
// drawn when its model is "in view" — either the 'выбор объектов' cursor
// tool is active (browse everything) or THIS marker's own model is the
// active tool (band_lambda/simplex_ensemble panel open) — UNLESS the
// forecast is individually pinned or the one currently selected, which
// always show regardless of what tool is active (same "active ∪ pinned"
// rule the rest of §2.6 applies to other tools).
function forecastMarkersInView(modelType) {
  return S.activeMainTool === 'object_select' || S.activeMainTool === modelType;
}

function buildForecastMarkerTraces() {
  if (!S.historyForecasts.length) return [];

  const pinnedKey = S.pinnedForecastIds.size ? [...S.pinnedForecastIds].sort().join(',') : '';
  const cache = _markerTracesCache;
  if (
    cache.forecasts === S.historyForecasts &&
    cache.pinnedKey === pinnedKey &&
    cache.selected === S.selectedForecastId &&
    cache.activeTool === S.activeMainTool
  ) {
    _markerPointsByDate = cache.pointsByDate;
    return cache.traces;
  }

  const byModel = new Map();
  for (const f of S.historyForecasts) {
    if (!byModel.has(f.model_type)) byModel.set(f.model_type, []);
    byModel.get(f.model_type).push(f);
  }

  const candles = candleByDateMap();
  const traces = [];
  // Shared across EVERY model type (not re-declared per type) — the
  // stacking index is a per-DATE counter, not per-(date,type). Declaring it
  // inside the modelType loop used to reset it to 0 for each type, so two
  // DIFFERENT-type forecasts sharing an origin both got idx=0 → identical
  // offset from their own basePrice → markers could land on top of each
  // other (bug report: "иконки на графике находят одна на другую"). One
  // shared counter guarantees any two markers on the same date — same type
  // or not — get distinct offsets.
  const dateCount = {};
  // Every marker point, keyed by date, independent of which Plotly trace
  // (curveNumber) ends up carrying it — _bindClickIfNeeded reads this
  // instead of Plotly's own plotly_click data.points, which (hovermode:'x
  // unified') reports at most ONE point per trace: since every forecast of
  // one model type shares a SINGLE trace, two same-type forecasts stacked
  // at one origin could never both appear as click candidates, so a click
  // could never cycle between them (bug report: "прогнозы одного типа...
  // не чередуются по клику"). This map has no such limit.
  const pointsByDate = new Map();
  for (const [modelType, list] of byModel) {
    const style = MODEL_DISPLAY[modelType]?.marker ?? { symbol: 'triangle-up', colorRole: null };
    const place = MARKER_PLACEMENT[modelType];
    const pinnedColor = pinnedMarkerColor(style);
    const inView = forecastMarkersInView(modelType);
    const xs = [], ys = [], ids = [], texts = [], opacities = [], sizes = [], symbols = [];
    // Oldest first so the newest forecast at a shared date ends up on top
    // (last in the array = drawn last = on top in Plotly scatter traces).
    const ordered = [...list].sort((a, b) => a.id - b.id);
    for (const f of ordered) {
      const pinned = S.pinnedForecastIds.has(f.id);
      const selected = f.id === S.selectedForecastId;
      if (!pinned && !selected && !inView) continue;
      const d = f.origin_extreme_ts;
      const idx = dateCount[d] ?? 0;
      dateCount[d] = idx + 1;
      // Offset above/below the origin BAR (not sitting ON it, project
      // feedback 2026-08-20: "не всегда заметны") — see MARKER_PLACEMENT's
      // docstring for the per-model above/below-and-which-way-the-triangle-
      // points rule. Falls back to the old at-origin-price placement if the
      // bar can't be found (shouldn't normally happen — origin_extreme_ts
      // IS one of S.candles' own dates) or the model has no placement rule.
      const candle = candles.get(d.slice(0, 10));
      const placement = place && candle ? place(candle, f) : null;
      const y = placement
        ? placement.basePrice * (1 + placement.sign * (MARKER_OFFSET_PCT + idx * MARKER_STACK_STEP))
        : f.origin_price * (1 - idx * MARKER_STACK_STEP);
      xs.push(d);
      ys.push(y);
      symbols.push(placement?.symbol ?? style.symbol);
      ids.push(f.id);
      texts.push(`#${f.id} — ${f.origin_direction > 0 ? '▲' : '▼'} ${f.origin_price.toFixed(4)}`);
      opacities.push(selected || pinned ? 0.95 : 0.55);
      sizes.push(selected ? 18 : (pinned ? 14 : 12));
      const dateKey = d.slice(0, 10);
      if (!pointsByDate.has(dateKey)) pointsByDate.set(dateKey, []);
      pointsByDate.get(dateKey).push({ id: f.id, y, modelType });
    }
    if (!xs.length) continue;

    traces.push({
      type: 'scatter',
      name: MODEL_DISPLAY[modelType]?.label ?? modelType,
      x: xs, y: ys,
      customdata: ids,
      text: texts,
      mode: 'markers',
      marker: {
        symbol: symbols, size: sizes,
        color: ids.map(id => id === S.selectedForecastId ? S.colorProfile.forecast_marker_selected : pinnedColor),
        opacity: opacities, line: { width: 0 },
      },
      hovertemplate: '<b>%{x|%d.%m.%Y}</b><br>%{text}<extra></extra>',
      showlegend: true,
    });
  }
  _markerTracesCache = {
    forecasts: S.historyForecasts, pinnedKey, selected: S.selectedForecastId, activeTool: S.activeMainTool,
    traces, pointsByDate,
  };
  _markerPointsByDate = pointsByDate;
  return traces;
}

// Shapes/annotations/traces for every currently-visible forecast (pinned +
// the temporarily selected one, if any) — the core of the "show several
// forecasts, incl. across models, at once" mechanism. Only the ACTIVELY
// SELECTED forecast gets the full-width origin crosshair (vertical +
// horizontal dotted lines) — pinned-but-not-selected forecasts used to get
// one each too (chart.js pre-2026-08-19), which multiplied into a field of
// crossing dashed lines with several pins active and was reported as
// clutter (project feedback 2026-08-19: "путаются с другими инструментами").
// Pinned forecasts stay identifiable via their marker alone (brighter
// opacity, see buildForecastMarkerTraces) — no line.
//
// "показать/скрыть активные объекты" (S.objectsHidden, tools.js:
// isToolObjectVisible) is handled RIGHT HERE, not by the caller — every
// caller (renderChart, setOrigin) gets the correct set this way with no
// risk of one of them forgetting to gate it externally (that mismatch used
// to let a click that calls setOrigin() re-draw a pinned/selected forecast
// even with the toggle off, since setOrigin() rebuilt shapes straight from
// this function without checking S.objectsHidden at all — bug report
// 2026-09-27: "band lambda... выпрыгивает сквозь него на график"). While
// the toggle is soloing the active tool, PINS stop mattering entirely —
// whatever's selected wins over them; only with nothing selected does it
// fall back to every forecast already fetched for the active model
// (S.renderedForecasts), regardless of pin state.
function visibleForecastIds() {
  let visibleIds;
  if (S.objectsHidden) {
    visibleIds = new Set();
    // Prefer whatever's actually SELECTED right now — picking a specific
    // past forecast from a model's own history list while that model is
    // the active tool is a normal, expected action (not just object_select
    // browsing), and solo mode should track it: "показывает выбранный
    // прогноз", not every forecast of that type ever cached this session.
    // Only object_select itself has no "own type" of its own to fall back
    // to, so a selection there is the ONLY thing solo mode can show.
    const selected = S.renderedForecasts.get(S.selectedForecastId);
    if (selected && (S.activeMainTool === 'object_select' || selected.model_type === S.activeMainTool)) {
      visibleIds.add(S.selectedForecastId);
    } else if (S.activeMainTool !== 'object_select') {
      for (const f of S.renderedForecasts.values()) {
        if (f.model_type === S.activeMainTool) visibleIds.add(f.id);
      }
    }
  } else {
    visibleIds = new Set(S.pinnedForecastIds);
    if (S.selectedForecastId != null) visibleIds.add(S.selectedForecastId);
  }
  return visibleIds;
}

function buildVisibleOverlays() {
  const visibleIds = visibleForecastIds();
  const shapes = [], annotations = [], traces = [];
  for (const id of visibleIds) {
    const f = S.renderedForecasts.get(id);
    if (!f) continue;
    const display = MODEL_DISPLAY[f.model_type];
    if (!display) continue;
    const out = display.buildOverlay(f.result, f.zone_geometry) || {};
    if (out.shapes) shapes.push(...out.shapes);
    if (out.annotations) annotations.push(...out.annotations);
    if (out.traces) traces.push(...out.traces);
    if (id === S.selectedForecastId) {
      shapes.push(originLineShape(f.result.origin_extreme_date), originPriceLineShape(f.result.origin_price));
    }
  }
  return { shapes, annotations, traces };
}

// dayTrace's per-bar weekday label never changes for a given candle array
// (it's a pure function of the date), but buildBaseTraces used to recompute
// it — including a `new Intl.DateTimeFormat().format()` call PER BAR — on
// EVERY renderChart call, even ones triggered by something unrelated (e.g.
// dragging an analyzer's window slider). Cached keyed by array identity:
// invalidates exactly when S.candles gets reassigned (ticker/interval
// switch, app.js:loadCandles), never on a settings-only re-render.
let _dayLabelsCache = { candles: null, labels: null };

function dayLabelsFor(candles) {
  if (_dayLabelsCache.candles === candles) return _dayLabelsCache.labels;
  const labels = candles.map(r => {
    const d = new Date(r.begin.slice(0, 10) + 'T12:00:00');
    const name = new Intl.DateTimeFormat('ru-RU', { weekday: 'long' }).format(d);
    return name.charAt(0).toUpperCase() + name.slice(1);
  });
  _dayLabelsCache = { candles, labels };
  return labels;
}

// ── display window (ограничение «свечей на графике») ──────────────────
// Limits DRAWING only. S.candles keeps every loaded bar — forecasts, MA,
// risk calculator and the origin lookup all read it. Traces are built from
// the visible window plus DISPLAY_MARGIN_SPANS of view-span on each side, so
// a small pan doesn't reveal an empty edge before the next settle re-slices
// (candle_window.js calls displayNeedsRefresh on settle).
const DISPLAY_MARGIN_SPANS = 1.0;
let _displayBounds = null; // { lo, hi } in ms — the window the current traces were built from

function _msOf(s) { return new Date(s).getTime(); }

function _computeDisplayBounds(xRange) {
  if (!xRange) return null;
  const a = _msOf(xRange[0]), b = _msOf(xRange[1]);
  if (!Number.isFinite(a) || !Number.isFinite(b) || b <= a) return null;
  const span = b - a;
  return { lo: a - span * DISPLAY_MARGIN_SPANS, hi: b + span * DISPLAY_MARGIN_SPANS };
}

// [i0, i1) of an ascending `times` array (ISO begin strings) inside the
// display window. No window yet → everything.
export function displayIndexRange(times) {
  if (!_displayBounds) return [0, times.length];
  const { lo, hi } = _displayBounds;
  const i0 = times.findIndex(t => _msOf(t) >= lo);
  if (i0 < 0) return [times.length, times.length];
  let i1 = times.length;
  while (i1 > i0 && _msOf(times[i1 - 1]) > hi) i1--;
  return [i0, i1];
}

// Visible-window slice of S.candles (drawing only — never assign the result
// back to S.candles).
export function displayCandles() {
  const [i0, i1] = displayIndexRange(S.candles.map(c => c.begin));
  return S.candles.slice(i0, i1);
}

// True when the view has moved close to an edge of what was drawn, or has
// zoomed in far enough that the drawn window is much wider than needed.
// Checked on relayout settle (candle_window.js), not during a drag.
export function displayNeedsRefresh() {
  if (!_displayBounds) return false;
  const { x } = getCurrentRanges();
  if (!x) return false;
  const a = _msOf(x[0]), b = _msOf(x[1]);
  const span = b - a;
  if (!(span > 0)) return false;
  const { lo, hi } = _displayBounds;
  return a < lo + span * 0.25 || b > hi - span * 0.25 || (hi - lo) > span * 4;
}

function buildBaseTraces() {
  const [i0, i1] = displayIndexRange(S.candles.map(r => r.begin));
  const c = S.candles.slice(i0, i1);

  const candlestick = {
    type: 'candlestick', name: `${S.ticker} ${S.interval}`,
    x: c.map(r => r.begin), open: c.map(r => r.open),
    high: c.map(r => r.high), low: c.map(r => r.low), close: c.map(r => r.close),
    increasing: { line: { color: S.colorProfile.candle_up }, fillcolor: S.colorProfile.candle_up },
    decreasing: { line: { color: S.colorProfile.candle_down }, fillcolor: S.colorProfile.candle_down },
    whiskerwidth: 0, showlegend: true, // legend's own first entry names which ticker+interval is loaded (project feedback 2026-10-04: interval alone wasn't shown anywhere) — otherwise nothing on the chart says so
  };

  const dayTrace = {
    type: 'scatter', x: c.map(r => r.begin), y: c.map(r => r.close),
    mode: 'none', name: '',
    text: dayLabelsFor(S.candles).slice(i0, i1), // labels cached per full S.candles, sliced for drawing
    hovertemplate: '%{text}<extra></extra>',
    showlegend: false,
  };

  const traces = [dayTrace, candlestick];

  return traces;
}

// The actual `Plotly.react` paint is the expensive part of a render — full
// trace/layout diff over however many points are on screen. Some callers
// (an analyzer's `oninput` on a range slider, e.g. trend_variance's window)
// fire renderChart() many times within a single animation frame while the
// user drags; without coalescing, EVERY tick did a full Plotly.react (this
// was the main cause of drag lag, not candle count — project feedback
// 2026-08-18). `schedulePlotlyReact` keeps only the LATEST queued
// traces/layout and paints once per frame via requestAnimationFrame — the
// first paint still happens next frame (~16ms, imperceptible), so nothing
// about the "instant preview" feel changes; only redundant repaints within
// the same frame are dropped. `_bindClickIfNeeded` (needs the `#chart` div
// to already have Plotly's `.on` attached, which happens synchronously
// inside `Plotly.react`/`newPlot`) moves here, after the actual paint call,
// instead of unconditionally after every renderChart() call.
let _rafHandle = null;
let _pendingPlotArgs = null;

// The 2026-08-25 "re-apply annotations once after first paint" workaround
// that used to live here is gone — it was a defensive guess at fixing
// price levels' price LABEL not appearing on a fresh log-scale load,
// without a confirmed root cause. Superseded by removing Plotly annotations
// from the price-level feature entirely (see cursor_tools.js:
// renderPriceLevelOverlay — a plain HTML/CSS overlay instead), which makes
// the workaround moot rather than actually fixing the original mechanism.
// Zoom/pan gesture freeze. A repaint built from a range captured before the
// latest wheel/drag step, and painted a frame later, snaps the view back for
// a moment ("телепортируется"). So while a gesture is running we don't paint
// at all — the latest args stay queued, and once the view has been quiet for
// GESTURE_QUIET_MS we paint them with the CURRENT x/y ranges.
const GESTURE_QUIET_MS = 150;
let _lastRelayoutAt = 0;
let _gestureTimer = null;

function _gestureActive() {
  return performance.now() - _lastRelayoutAt < GESTURE_QUIET_MS;
}

function _paintPending() {
  const [t, l, c] = _pendingPlotArgs;
  _pendingPlotArgs = null;
  Plotly.react('chart', t, l, c);
  _bindClickIfNeeded();
  refreshOverlays();
}

function _schedulePaintAfterGesture() {
  if (_gestureTimer) return;
  _gestureTimer = setTimeout(() => {
    _gestureTimer = null;
    if (!_pendingPlotArgs) return;
    if (_gestureActive()) { _schedulePaintAfterGesture(); return; }
    // Take the view as it is NOW, not as it was when this render was built.
    const fl = document.getElementById('chart')?._fullLayout;
    const l = _pendingPlotArgs[1];
    if (fl && l.xaxis?.range && fl.xaxis?.range) l.xaxis = { ...l.xaxis, range: fl.xaxis.range };
    if (fl && l.yaxis?.range && fl.yaxis?.range) l.yaxis = { ...l.yaxis, range: fl.yaxis.range };
    _paintPending();
  }, GESTURE_QUIET_MS);
}

function schedulePlotlyReact(traces, layout, config) {
  _pendingPlotArgs = [traces, layout, config];
  if (_rafHandle) return;
  _rafHandle = requestAnimationFrame(() => {
    _rafHandle = null;
    if (_gestureActive()) { _schedulePaintAfterGesture(); return; }
    _paintPending();
  });
}

// ── render ────────────────────────────────────────────────────────────────────
// THE single render entry point — every state change that affects what's on
// the chart (candle load, zigzag toggle, spectrogram toggle, pin/select a
// forecast, submit completes, drag the band-width slider) just calls this
// again with preserveRange:true. Replaces the earlier separate renderChart/
// renderBandForecast/renderSimplexEnsemble/refreshBandDisplay entry points —
// merged so there is exactly one place that assembles the trace list, always
// consistent with S.pinnedForecastIds/S.selectedForecastId (see
// buildVisibleOverlays above). Trace/layout assembly stays synchronous here
// (cheap, and some of it has synchronous side effects like S.shapes) — only
// the Plotly paint itself is deferred/coalesced, see schedulePlotlyReact.
export function renderChart({ preserveRange = false } = {}) {
  const c = S.candles;
  if (!c.length) return;

  // S.objectsHidden ("показать/скрыть активные объекты", cursor_tools.js) —
  // solos the currently active tool's own drawn objects (pinned/selected
  // forecast zones, analyzer overlays, zigzag), ignoring every OTHER tool's
  // pin — each builder below (buildVisibleOverlays, buildForecastZigzagTraces,
  // every tool's own buildMainTraces via tools.js:isToolObjectVisible) checks
  // S.objectsHidden itself, so this function just calls them unconditionally
  // and trusts them to decide what's actually visible. Forecast MARKERS stay
  // visible either way — they're a lightweight index, not clutter, and the
  // "выбор объектов" cursor tool still needs them clickable.
  const { shapes, annotations, traces: overlayTraces } = buildVisibleOverlays();
  // S.originTs ("next forecast starts here", set by setOrigin on a plain
  // chart click) is independent of which past forecasts are pinned/selected
  // — always show it too, alongside any forecast crosshairs, so a full
  // renderChart (tab switch, candle reload, pin toggle, ...) never drops it.
  if (S.originTs) shapes.push(originLineShape(S.originTs));
  S.shapes = shapes;
  S.bandAnnotations = annotations;

  // Display window first — every trace builder below reads it.
  const ranges0 = getCurrentRanges();
  const xRange0 = preserveRange ? ranges0.x : defaultXRange(c);
  _displayBounds = _computeDisplayBounds(xRange0);

  const baseTraces = buildBaseTraces();
  // Forecast zigzag: the actively selected T while band_lambda is the
  // active tool, plus every explicitly pinned T regardless of active tool
  // (§2.4/§2.6 of docs/plans/frontend_improvements_plan.md — same "active ∪
  // pinned" policy the rest of the app uses) — see buildForecastZigzagTraces.
  const zzTraces = buildForecastZigzagTraces();
  const allTraces = [...baseTraces, ...zzTraces, ...buildAnalyzerMainTraces()];
  const markers = buildForecastMarkerTraces();
  _markerTraceIndices = markers.map((_, i) => allTraces.length + i);
  allTraces.push(...markers);
  allTraces.push(...buildSubpanelTraces());
  allTraces.push(...overlayTraces);

  // preserveRange=false only on fresh candle load → reset to default window.
  // All other callers keep whatever the user has set (no auto-scale).
  const ranges  = getCurrentRanges();
  const xRange  = preserveRange ? ranges.x  : defaultXRange(c);
  const y1Range = preserveRange ? ranges.y1 : toAxisYRange(computeYForXRange(xRange));
  const { y2Range, y2Fixed } = applyY2RangePolicy(preserveRange, ranges.y2);

  schedulePlotlyReact(allTraces, buildLayout(xRange, y1Range, y2Range, y2Fixed), PLOTLY_CONFIG);
}

// ── simplex_ensemble overlay builders ────────────────────────────────────────
// Different shape from band_lambda's two discrete h=1/h=2 zones: H
// continuous per-origin trajectories (DATA traces, not shapes — there's no
// fixed-width geometry to persist/drag here) + an optional reactive
// mean/percentile-band pair, recomputed client-side from the "raw" per-
// origin rel arrays every result carries (see sma/core/forecast/
// simplex_ensemble.py:forecast_ensemble — per_origin[*].rel).

function simplexTrajectoryX(originDateStr, horizon, interval) {
  const xs = [originDateStr];
  for (let h = 1; h <= horizon; h++) xs.push(futureDateAt(originDateStr, h, interval, S.candles));
  return xs;
}

// One line per origin in the ensemble, ALL the same color/opacity — no
// recency fade (unlike the Streamlit prototype) per explicit project
// decision (session 2026-08-15): fading was an artifact of the prototype's
// exploratory UI, not something requested for the production tool.
function buildSimplexOriginTraces(result) {
  const horizon = result.horizon;
  return result.per_origin.map((r, idx) => {
    const xs = simplexTrajectoryX(r.origin_date, horizon, S.interval);
    const ys = [r.origin_price, ...r.rel.map(rel => r.origin_price * (1 + rel))];
    return {
      type: 'scatter', mode: 'lines',
      name: idx === 0 ? 'origin (главный)' : `origin −${idx}`,
      x: xs, y: ys,
      line: { width: idx === 0 ? 1.5 : 1, color: hexToRgba(S.colorProfile.simplex_origin_lines, 0.55) },
      showlegend: idx === 0,
      hoverinfo: 'skip',
    };
  });
}

// Mean + [50-P/2, 50+P/2] percentile band across the ensemble, anchored at
// the MAIN origin (index 0) — mirrors how band_lambda's combined band
// anchors at origin_extreme_date. bandPct=50 -> p25-p75, bandPct=90 -> p5-p95.
function buildSimplexMeanBandTraces(result, bandPct) {
  const per = result.per_origin;
  if (!per.length) return [];
  const horizon = result.horizon;
  const relMatrix = per.map(r => r.rel);
  const ones = per.map(() => 1);

  const meanRel = [], loRel = [], hiRel = [];
  const qLo = (100 - bandPct) / 200;
  const qHi = 1 - qLo;
  for (let h = 0; h < horizon; h++) {
    const col = relMatrix.map(row => row[h]);
    meanRel.push(col.reduce((a, b) => a + b, 0) / col.length);
    const [lo, hi] = weightedQuantile(col, ones, [qLo, qHi]);
    loRel.push(lo); hiRel.push(hi);
  }

  const originPrice = result.origin_price;
  const xs = simplexTrajectoryX(result.origin_date, horizon, S.interval);
  const toY = relArr => [originPrice, ...relArr.map(r => originPrice * (1 + r))];
  const pLo = Math.round(50 - bandPct / 2), pHi = Math.round(50 + bandPct / 2);

  return [
    {
      type: 'scatter', mode: 'lines', x: xs, y: toY(hiRel),
      line: { width: 0 }, showlegend: false, hoverinfo: 'skip',
    },
    {
      type: 'scatter', mode: 'lines', x: xs, y: toY(loRel),
      line: { width: 0 }, fill: 'tonexty', fillcolor: hexToRgba(S.colorProfile.simplex_mean_band, 0.15),
      name: `p${pLo}–p${pHi}`, hoverinfo: 'skip',
    },
    {
      type: 'scatter', mode: 'lines+markers', x: xs, y: toY(meanRel),
      name: `среднее по ${per.length} origin`,
      line: { color: S.colorProfile.simplex_mean_band, width: 2.5 }, marker: { size: 3 },
    },
  ];
}

// ── regime_mixture_potential — heatmap/цвет/границы (JS-порт density_
// colorscale/scenario_density_heatmap_from_components из sma/core/forecast/
// regime_mixture_potential.py — см. app33-regime-mixture-rewind.py, где эта
// механика была выработана в прототипе) ──────────────────────────────────

// Тусклый синевато-серый → тил → жёлтый → красный — те же 4 стопа, что и в
// прототипе (DEFAULT_COLOR_BAND), чтобы соседние уровни плотности
// разводились по ОТТЕНКУ, не только по яркости.
const POTENTIAL_COLOR_BAND = [[60, 80, 120], [40, 165, 175], [255, 210, 60], [235, 30, 30]];

function potentialLerpBand(colors, frac) {
  frac = Math.min(Math.max(frac, 0), 1);
  const nSeg = colors.length - 1;
  const pos = frac * nSeg;
  const i = Math.min(Math.floor(pos), nSeg - 1);
  const local = pos - i;
  const c0 = colors[i], c1 = colors[i + 1];
  return [0, 1, 2].map(k => c0[k] + (c1[k] - c0[k]) * local);
}

// Нормированная tanh-кривая t∈[0,1] → [0,1] с двумя независимыми осями:
// shift — ГДЕ сидит точка перехода, steepness — КАК резко она проходит.
function potentialShiftSteepnessRamp(shift, steepness) {
  const raw = t => Math.tanh(steepness * (t - shift));
  const raw0 = raw(0), raw1 = raw(1);
  const denom = Math.abs(raw1 - raw0) > 1e-12 ? (raw1 - raw0) : 1.0;
  return t => Math.min(Math.max((raw(t) - raw0) / denom, 0), 1);
}

// alpha = alpha_fn(mix), НЕ alpha_fn(t) — гаснет только у самого холодного
// конца цветового градиента (mix≈0), остальной градиент (тил/жёлтый/
// оранжевый/красный) всегда полностью непрозрачен независимо от
// colorSteepness (см. density_colorscale в regime_mixture_potential.py для
// разбора, почему альфа не может быть отдельной кривой по t).
function potentialDensityColorscale(colorShift, colorSteepness, coldCutoff = 0.12, cutoffSteepness = 14.0, nStops = 48) {
  const colorFn = potentialShiftSteepnessRamp(colorShift, colorSteepness);
  const alphaFn = potentialShiftSteepnessRamp(coldCutoff, cutoffSteepness);
  const stops = [];
  for (let i = 0; i <= nStops; i++) {
    const t = i / nStops;
    const mix = colorFn(t);
    const a = alphaFn(mix);
    const [r, g, b] = potentialLerpBand(POTENTIAL_COLOR_BAND, mix);
    stops.push([t, `rgba(${r.toFixed(0)},${g.toFixed(0)},${b.toFixed(0)},${a.toFixed(4)})`]);
  }
  return stops;
}

// Режим «сценарии» — сервер хранит только components_by_h (компактно:
// mean/std/weight на компоненту), не готовую сетку, поэтому диапазон Y
// строится здесь из самих компонент (±4σ от каждого центра) — единственный
// способ определить разумные границы без сырых путей (которые НЕ хранятся
// в result_json, слишком тяжело). Формула бугра — БЕЗ нормировки на
// σ·√(2π) (как в scenario_density_heatmap_from_components) — пик РОВНО на
// высоте веса, прозрачность в heatmap напрямую = вероятность сценария.
// Диапазон Y считается по ВСЕМ снапшотам перемотки сразу (не по текущему
// одному) — иначе сетка heatmap-«сценариев» пересчитывается на каждый шаг
// перемотки (±4σ у каждого снапшота свои mean/std), и заметно "дышит"/
// дёргается при скролле ползунка (пользовательский репорт 2026-09-28,
// заметнее в Firefox, но причина в данных — разный рендер лишь по-разному
// показывает один и тот же прыжок сетки). Кэш по identity result — не
// пересчитывать на каждый рендер (объект result не меняется, пока не выбран
// другой прогноз).
let _potentialRangeCache = { result: null, range: null };

function potentialGlobalRange(result) {
  if (_potentialRangeCache.result === result) return _potentialRangeCache.range;
  let lo = Infinity, hi = -Infinity;
  for (const snap of result.snapshots) {
    for (const comps of snap.components_by_h) {
      for (const [m, s] of comps) {
        lo = Math.min(lo, m - 4 * s);
        hi = Math.max(hi, m + 4 * s);
      }
    }
  }
  // mean-4σ on a wide-uncertainty snapshot (long horizon, thin pool) can go
  // non-positive — invisible in linear scale (just off-screen below the
  // candle-only y-range), but a heatmap row at y<=0 breaks Plotly's log10
  // transform and blanks the WHOLE trace from that row down (project
  // report 2026-10-04: regime_mixture_potential heatmap missing its bottom
  // half specifically in log scale, e.g. LKOH). Same floor convention as
  // toAxisYRange's safeLo above.
  if (Number.isFinite(lo) && Number.isFinite(hi) && hi > 0) {
    lo = Math.max(lo, hi * 1e-6);
  }
  const range = Number.isFinite(lo) && Number.isFinite(hi) && hi > lo ? { lo, hi } : null;
  _potentialRangeCache = { result, range };
  return range;
}

// binHeight — та же величина (в единицах цены), что сервер использует для
// «сырой плотности» (bin_height_pct × origin_price, см. corridor_density_
// heatmap в regime_mixture_potential.py) — единая высота бина для ОБОИХ
// режимов heatmap, чтобы «сценарии» и «сырая плотность» были визуально
// сопоставимы (по прямому запросу пользователя 2026-09-28).
function potentialScenarioGrid(componentsByH, range, binHeight) {
  if (!range) return null;
  const { lo, hi } = range;
  // Cap защищает от пары тысяч бинов при очень мелком bin_height_pct на
  // широком (по всем снапшотам перемотки) диапазоне — не ожидается в
  // обычном использовании, просто страховка от подвисания рендера.
  const nBins = Math.min(2000, Math.max(1, Math.ceil((hi - lo) / binHeight)));
  const centers = Array.from({ length: nBins }, (_, i) => lo + binHeight * (i + 0.5));
  const hMax = componentsByH.length;
  const z = Array.from({ length: nBins }, () => new Array(hMax).fill(0));
  for (let j = 0; j < hMax; j++) {
    for (const [m, s, w] of componentsByH[j]) {
      const sEff = Math.max(s, 1e-6);
      for (let i = 0; i < nBins; i++) {
        const d = (centers[i] - m) / sEff;
        z[i][j] += w * Math.exp(-0.5 * d * d);
      }
    }
  }
  return { centers, z };
}

function potentialFutureX(snap, horizon) {
  return Array.from({ length: horizon }, (_, h) => futureDateAt(snap.origin_date, h + 1, S.interval, S.candles));
}

// Coverage bound, computed on the CLIENT from the snapshot's own
// raw_histogram (an empirical density per h, always present regardless of
// heatmap mode — see regime_mixture_potential.py:_build_one_snapshot) —
// replaces the server's `snap.bounds` (frozen at whatever coverage_pct the
// forecast was originally submitted with) so the "Покрытие" slider in
// «Отображение» can react to ANY % instantly, the same way heatmap-mode/
// rewind/XY-pad already do, without a resubmit. For each h, walks the
// bin_centers (ascending, per corridor_density_heatmap's construction) and
// accumulates the normalized density until crossing the lower/upper tail
// fraction — coarser than the server's exact quantile-of-raw-paths (limited
// to bin resolution, bin_height_pct), but that resolution is already fine
// enough to be visually indistinguishable at normal chart zoom.
function potentialCoverageBoundsFromHistogram(rawHistogram, coveragePct) {
  const { bin_centers: centers, z } = rawHistogram;
  if (!centers?.length) return null;
  const half = (100 - coveragePct) / 200;
  const hMax = z[0].length;
  const lo = new Array(hMax), hi = new Array(hMax);
  for (let j = 0; j < hMax; j++) {
    const total = centers.reduce((s, _, i) => s + z[i][j], 0);
    if (total <= 0) { lo[j] = centers[0]; hi[j] = centers[centers.length - 1]; continue; }
    const loTarget = half * total, hiTarget = (1 - half) * total;
    let cum = 0;
    for (let i = 0; i < centers.length; i++) {
      cum += z[i][j];
      if (lo[j] === undefined && cum >= loTarget) lo[j] = centers[i];
      if (hi[j] === undefined && cum >= hiTarget) { hi[j] = centers[i]; break; }
    }
    if (lo[j] === undefined) lo[j] = centers[0];
    if (hi[j] === undefined) hi[j] = centers[centers.length - 1];
  }
  return { lo, hi };
}

// buildOverlay для MODEL_DISPLAY.regime_mixture_potential — S.potential*
// (heatmap-режим/цвет XY/покрытие/перемотка/границы) читаются ЖИВЬЁМ на
// каждый рендер, без пересчёта на сервере: result.snapshots[idx] уже несёт
// всё нужное для ЛЮБОГО положения «Отображения» (см. модуль core — перемотка
// прогревается целиком на «▶ Прогноз»).
function buildPotentialOverlayTraces(result) {
  const idx = Math.min(S.potentialRewindIdx, result.snapshots.length - 1);
  const snap = result.snapshots[idx];
  const futureX = potentialFutureX(snap, result.horizon);
  const traces = [];

  let grid = null;
  if (S.potentialHeatmapMode === 'raw' && snap.raw_histogram) {
    grid = { centers: snap.raw_histogram.bin_centers, z: snap.raw_histogram.z };
  } else if (S.potentialHeatmapMode !== 'raw') {
    const binHeight = result.origin_price * result.bin_height_pct / 100;
    grid = potentialScenarioGrid(snap.components_by_h, potentialGlobalRange(result), binHeight);
  }
  if (grid) {
    // zmax=1.0 ФИКСИРОВАН для «сценарии» (не max(z)) — иначе один ранний
    // доминантный сценарий на h=1 задаёт масштаб цвета для всех h, и более
    // мелкие сценарии визуально тонут (см. regime_mixture_potential.py).
    const zMax = S.potentialHeatmapMode === 'raw' ? Math.max(1e-9, ...grid.z.map(row => Math.max(...row))) : 1.0;
    traces.push({
      type: 'heatmap', x: futureX, y: grid.centers, z: grid.z,
      colorscale: potentialDensityColorscale(S.potentialColorShift, S.potentialColorSteepness),
      zmin: 0, zmax: zMax, zsmooth: false, showscale: false, hoverinfo: 'skip', name: 'плотность',
      // zsmooth:false — без доп. интерполяции между ячейками; без явного
      // значения поведение зависело от дефолта Plotly + от того, как сам
      // браузер масштабирует получившуюся картинку под размер графика —
      // Chrome/Firefox могут делать это по-разному (жалоба пользователя
      // 2026-09-28: "в хроме цвета тусклее"), явное false убирает эту
      // degrees of freedom хотя бы на стороне Plotly.
    });
  }

  if (S.potentialShowBounds && snap.raw_histogram) {
    const bounds = potentialCoverageBoundsFromHistogram(snap.raw_histogram, S.potentialCoveragePct);
    if (bounds) {
      const corridorX = [snap.origin_date, ...futureX];
      traces.push({
        type: 'scatter', mode: 'lines', x: corridorX, y: [snap.origin_price, ...bounds.hi],
        name: `${S.potentialCoveragePct}% верх`, line: { color: '#ffffff', width: 1.3, dash: 'dash' }, hoverinfo: 'skip',
      });
      traces.push({
        type: 'scatter', mode: 'lines', x: corridorX, y: [snap.origin_price, ...bounds.lo],
        name: `${S.potentialCoveragePct}% низ`, line: { color: '#ffffff', width: 1.3, dash: 'dash' }, hoverinfo: 'skip',
      });
    }
  }

  return traces;
}

// ── click events ──────────────────────────────────────────────────────────────
// Clicking a forecast marker sets it as the temporary "selected" focus (see
// S.selectedForecastId — sma/ui/forecast_history.js:selectForecast handles
// fetching/caching/rendering); clicking empty chart area clears that focus
// if it isn't pinned. Either way, the plain origin line is ALSO placed
// (_onCandleClick) — the point the "Прогноз" button reads for the NEXT
// submit (see forecast.js:findOriginPivot / simplex_ensemble.js:
// resolveOriginCandle). Pins/visibility are independent of this click
// handling — a pinned forecast keeps showing regardless of what gets clicked.
//
// The "price level" cursor tool (S.activeMainTool === 'price_level', see
// sma/ui/cursor_tools.js) is handled ENTIRELY through native DOM mouse
// events (_bindPriceLevelEventsIfNeeded below), not Plotly's own click/hover
// system — see that function's docstring for why.
let _onForecastClick     = null;
let _onCandleClick       = null;
let _onPriceLevelClick   = null;
let _onPriceLevelHover   = null;
let _chartClickBound     = false;
let _markerTraceIndices  = []; // updated by renderChart — one index per model's marker trace
let _markerPointsByDate  = new Map(); // date (YYYY-MM-DD) -> [{id, y, modelType}], set by buildForecastMarkerTraces

// Register callbacks; the forecast/candle click binding is deferred until
// after first Plotly.react (needs Plotly's own `.on()` API, see
// _bindClickIfNeeded), but price_level's native listeners can bind right
// away — they only need the (already-present, static) #chart div itself.
export function initChartEvents(onForecastClick, onCandleClick, onPriceLevelClick, onPriceLevelHover) {
  _onForecastClick   = onForecastClick;
  _onCandleClick     = onCandleClick;
  _onPriceLevelClick = onPriceLevelClick;
  _onPriceLevelHover = onPriceLevelHover;
  _bindPriceLevelEventsIfNeeded();
}

let _priceLevelEventsBound = false;

// Native DOM listeners, not Plotly's plotly_click/plotly_hover — deliberate,
// after two rounds of the Plotly-event-based approach failing (project
// feedback 2026-08-20, then 2026-08-25): with hovermode:'x unified' (see
// baseLayout), Plotly's own hover/click machinery only ever fires when the
// cursor is near SOME trace's data along x — clicking or hovering genuinely
// empty chart area (the padding right of the last candle, or above/below
// every bar) never reaches a plotly_click/plotly_hover handler AT ALL, no
// matter how that handler's internals are reordered. A native 'mousemove'/
// 'click' listener on the chart div fires unconditionally for any pointer
// position within its bounding box, regardless of what Plotly thinks is
// nearby — sidesteps the whole class of bug rather than chasing it inside
// Plotly's hit-testing again. Also replaces Plotly's own bar-snapped
// crosshair (showspikes) for this tool specifically (see buildXAxis/
// BASE_YAXIS below) — project feedback 2026-08-25: "перекрестие... для
// ценового уровня оно только путает, лучше горизонтальная линия, следящая
// за курсором" — _onPriceLevelHover drives that replacement line
// (cursor_tools.js:setPriceLevelHoverPrice), continuously, not snapped to
// any bar.
function _bindPriceLevelEventsIfNeeded() {
  if (_priceLevelEventsBound) return;
  const gd = document.getElementById('chart');
  if (!gd) return;
  _priceLevelEventsBound = true;

  const isPriceLevelActive = () => S.activeTab === 'main' && S.activeMainTool === 'price_level';

  gd.addEventListener('mousemove', ev => {
    // ev.buttons===0 (no mouse button held) AND !_interacting (no Plotly
    // drag-pan/drag-zoom currently in progress, tracked via plotly_relayouting/
    // plotly_relayout in _bindClickIfNeeded below) — BOTH guard against the
    // same bug (project feedback 2026-08-25: "при изменении масштаба и
    // перемещения графика в режиме ценового уровня уровни 'плывут'... X
    // шкала тоже отвязывается"): this handler used to fire unconditionally
    // on every mousemove, INCLUDING while the user was actively dragging to
    // pan/zoom the chart (dragmode:'pan', baseLayout) — each tick called
    // applyShapes() -> Plotly.relayout mid-gesture, fighting Plotly's own
    // internal drag-tracking state and visibly desyncing the axes from
    // where the cursor/drag actually was. ev.buttons alone catches a
    // click-drag pan; _interacting also catches drag-to-zoom-box and
    // similar interactive relayouts scroll/wheel-zoom doesn't sustain a
    // "buttons held" state for.
    if (!isPriceLevelActive() || ev.buttons !== 0 || _interacting) return;
    // Pass null through too (not just skip) — moving from the main chart
    // down into the oscillator subpanel stays within #chart's bounds (no
    // mouseleave fires), so without this the LAST real hover price would
    // stick around indefinitely while hovering over unrelated territory.
    _onPriceLevelHover?.(_priceAtY(gd, ev.clientY));
  });
  gd.addEventListener('mouseleave', () => {
    if (isPriceLevelActive()) _onPriceLevelHover?.(null);
  });
  gd.addEventListener('click', ev => {
    if (!isPriceLevelActive() || _interacting) return;
    const price = _priceAtY(gd, ev.clientY);
    if (price != null) _onPriceLevelClick?.(price);
  });
}

// Set true for the duration of an interactive Plotly drag-pan/drag-zoom
// (plotly_relayouting fires repeatedly WHILE dragging, plotly_relayout once
// at the end) — bound in _bindClickIfNeeded below since, unlike the plain
// DOM listeners above, this needs Plotly's own `.on()` event system, only
// available after the first Plotly.react/newPlot call.
let _interacting = false;

// Pixel Y -> real price. Two bugs, found and fixed in sequence (project
// feedback 2026-08-20, then 2026-08-25 confirming the first fix didn't
// actually work): the ORIGINAL code fed a pixel computed relative to the
// full plot area's top margin (`clientY - rect.top - _size.t`) into
// `yaxis.p2d()`. Checked against Plotly.js's actual source (package_data/
// plotly.min.js — no live browser available this session, see feedback_
// chrome_extension_unavailable in project memory, so this is the only way
// to verify Plotly internals rather than guess a second time): `p2d`/`p2l`
// (`ax.l2p = function(v){ return round(_b + _m*v, 2) }`, its inverse `p2l`)
// operate PURELY in the axis's OWN pixel span [0, _length] — `_offset`
// (`= _size.t + (1-domain[1])*_size.h` for a y-axis, confirmed in the same
// source) is a SEPARATE value the CALLER must subtract first; `p2d` itself
// never adds or removes it. So the pixel fed in was off by (_offset -
// _size.t) — zero only when the axis's own domain top is exactly 1 (no
// oscillator subpanel active) — otherwise silently wrong. Once confirmed
// `p2d`/`p2l` DO correctly round-trip through log10 (source: `t.p2d =
// function(T){ return l2d(p2l(T)) }` for a log-type axis, where l2d is
// literally `Math.pow(base, v)`), the earlier "reimplement the log10 un-
// transform by hand" fix (previous version of this function) was solving
// the wrong half of the problem — reverted in favor of just fixing the
// pixel input and trusting the (now-confirmed-correct) built-in method.
// A SMALL pixel error here is why this looked like a "log scale" bug
// specifically: a modest, easy-to-miss offset error is barely noticeable
// on a linear axis (a few % off), but on a log axis the SAME error lands
// in log10-space and comes back out through Math.pow() exponentiated into
// an enormous, obviously-wrong price — "запредельно вверх" is exactly what
// a log10 value a few units too high looks like once un-logged.
function _priceAtY(gd, clientY) {
  const fl = gd._fullLayout;
  const yaxis = fl?.yaxis;
  if (!yaxis?.p2d || yaxis._offset == null) return null;
  const rect = gd.getBoundingClientRect();
  const py = clientY - rect.top - yaxis._offset;
  // Outside the main axis's OWN plot rectangle (e.g. clientY over the
  // oscillator subpanel below it) — p2d would still happily extrapolate a
  // number, but it wouldn't mean anything (that pixel belongs to yaxis2's
  // domain, not yaxis's). Same bound priceToPixelY (its inverse) now
  // enforces — see that function's docstring for the render-side half of
  // this bug (project feedback 2026-08-25).
  if (py < 0 || py > yaxis._length) return null;
  return yaxis.p2d(py);
}

// Public wrapper around _priceAtY — lets a tool OUTSIDE this module (e.g.
// risk_calculator.js's "установить на графике" click-to-pick-price button)
// bind its own native click listener and read the same correctly log-scale-
// aware price the price_level tool's native listeners already use, without
// duplicating _priceAtY's pixel math (see its own docstring above for why
// that math is easy to get subtly wrong).
export function priceAtClientY(gd, clientY) {
  return _priceAtY(gd, clientY);
}

function _priceAtClick(gd, data) {
  return data.event ? _priceAtY(gd, data.event.clientY) : null;
}

// Inverse of _priceAtY — real price -> pixel Y relative to the #chart div's
// OWN top edge (same coordinate space _priceAtY's `py` already used, since
// both go through the same `- rect.top` / `- _offset` convention), so a
// caller can position an absolutely-placed HTML overlay element (e.g.
// #price-level-overlay, a SIBLING of #chart filling the same #chart-wrap
// box at the same origin) with `top: <result>px` directly. `d2p` is
// confirmed (checked against actual Plotly.js source, not assumed —
// package_data/plotly.min.js, see _priceAtY's own docstring for the full
// story) to correctly round-trip through log10 for a log-type axis, same
// as `p2d`.
//
// Returns null for a price outside the axis's OWN visible range (`d2p`
// itself doesn't clamp — it happily extrapolates past [0, yaxis._length]) —
// a Plotly-native shape would simply be clipped to its subplot's rectangle
// and vanish, but our HTML overlay (#price-level-overlay) has no per-axis
// clip region of its own, only overflow:hidden on the whole #chart-wrap
// (main chart + oscillator subpanel together). Without this bound check, a
// price level outside the currently visible Y-range (routine after a pan/
// zoom that changes the visible range) extrapolated to a pixel that can
// land inside the OSCILLATOR's pixel band below and render its full-width
// line right over the subpanel (project feedback 2026-08-25: "при
// перетаскивании графика ценовые уровни остаются поверх осциллятора").
export function priceToPixelY(gd, price) {
  const yaxis = gd._fullLayout?.yaxis;
  if (!yaxis?.d2p || yaxis._offset == null || price == null) return null;
  const py = yaxis.d2p(price) + yaxis._offset;
  if (py < yaxis._offset || py > yaxis._offset + yaxis._length) return null;
  return py;
}

function _bindClickIfNeeded() {
  if (_chartClickBound || !_onForecastClick) return;
  const gd = document.getElementById('chart');
  gd.on('plotly_click', data => {
    // price_level is no longer handled via Plotly's own click system at
    // all — see _bindPriceLevelEventsIfNeeded below (native DOM listeners).
    // Reordering this branch ahead of the `data.points` guard (2026-08-20)
    // didn't actually fix "клик по пустому пространству" (project feedback
    // 2026-08-25): with hovermode:'x unified', Plotly's plotly_click never
    // FIRES AT ALL when the click x falls outside every trace's data range
    // (the padding right of the last candle, or above/below every bar) —
    // no in-handler reordering can fix a handler that's never invoked.
    if (S.activeTab === 'main' && S.activeMainTool === 'price_level') return;

    if (!data.points.length) return;

    // dayTrace is always curveNumber 0 (moved before candlestick for hover order).
    const dayPt = data.points.find(p => p.curveNumber === 0) ?? data.points[0];
    if (!dayPt) return;
    const clickDate = String(dayPt.x).slice(0, 10);
    if (!/^\d{4}-\d{2}-\d{2}/.test(clickDate)) return;

    // Every marker on the EXACT clicked bar (clickDate, from dayPt above) —
    // no "nearest by date" fallback: with hovermode:'x unified', Plotly's
    // click event reports every trace's nearest-by-x point regardless of
    // actual distance from the cursor, so a fallback here would grab a
    // forecast from anywhere in the visible history, including on truly
    // empty chart area that should clear the selection instead. Landing on
    // the right BAR is enough precision — dayTrace already guarantees
    // clickDate is the nearest bar to the cursor, so this reads as "click
    // that bar", not "click the marker's exact pixel" — there is no
    // forgiving/near-miss click matching beyond that.
    //
    // Candidates come from _markerPointsByDate (built alongside the traces
    // in buildForecastMarkerTraces), NOT from data.points — Plotly reports
    // at most one point per TRACE, and every forecast of a given model type
    // shares one trace, so two same-type forecasts stacked at this date
    // could never both appear in data.points (bug: a click could never
    // cycle between them). _markerPointsByDate has no such limit — it lists
    // every marker on this date regardless of which trace carries it, so
    // disambiguating several candidates (same type or different) by the
    // click's actual price via _priceAtClick always sees the full set.
    const markerCandidates = _markerPointsByDate.get(clickDate) ?? [];
    let markerPt = null;
    if (markerCandidates.length === 1) {
      markerPt = markerCandidates[0];
    } else if (markerCandidates.length > 1) {
      const clickPrice = _priceAtClick(gd, data);
      markerPt = clickPrice == null
        ? markerCandidates[0]
        : markerCandidates.reduce((best, p) => {
            const d = Math.abs(p.y - clickPrice);
            return !best || d < best.d ? { p, d } : best;
          }, null).p;
    }
    _onForecastClick(markerPt ? markerPt.id : null, clickDate);

    _onCandleClick(clickDate);
  });
  // See _interacting's own docstring (above _bindPriceLevelEventsIfNeeded) —
  // suspends price_level's hover/click handling for the duration of an
  // interactive drag-pan/drag-zoom, since Plotly.relayout calls fired mid-
  // gesture (from applyShapes()) were fighting Plotly's own drag state.
  // Also keeps the HTML overlay (price-level line/labels) tracking pan/zoom
  // in real time — refreshOverlays() is a pure DOM read+write (no
  // Plotly.relayout call), so calling it here doesn't reintroduce that
  // same conflict.
  gd.on('plotly_relayouting', () => { _interacting = true; _lastRelayoutAt = performance.now(); refreshOverlays(); });
  gd.on('plotly_relayout', () => {
    _interacting = false;
    _lastRelayoutAt = performance.now();
    refreshOverlays();
    for (const fn of _relayoutHooks) fn();
  });
  _chartClickBound = true;
}

