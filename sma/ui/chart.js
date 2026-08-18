import { S, MONTH_RU, DAY_RU, INTERVAL_SECONDS } from './state.js';
import { TOOLS } from './tools.js';

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

// Shapes for the active subpanel analyzer (e.g. spectrogram's filter-bank
// cutoff lines) — generic dispatch through the registry, see tools.js.
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

// ── extensible shape/annotation providers ───────────────────────────────
// Generic hook for things that aren't a registered analyzer (see
// tools.js) but still need to draw on the main chart every render —
// currently just the price-level cursor tool (sma/ui/cursor_tools.js).
// One-directional: cursor_tools.js imports renderChart/paperY FROM this
// file and registers its builder INTO it, so chart.js never needs to
// import cursor_tools.js back (same cycle-avoidance reasoning as the
// TOOLS registry).
const _shapeProviders = [];
const _annotationProviders = [];

export function registerShapeProvider(fn) {
  _shapeProviders.push(fn);
}

export function registerAnnotationProvider(fn) {
  _annotationProviders.push(fn);
}

function extraShapes() {
  return _shapeProviders.flatMap(fn => fn());
}

function extraAnnotations() {
  return _annotationProviders.flatMap(fn => fn());
}

// Full shapes array: divider + active analyzer's subpanel shapes + every
// analyzer's origin crosshair + registered extras (price levels, ...) +
// user shapes (S.shapes). The "declutter" toggle (S.objectsHidden, see
// cursor_tools.js:toggleObjectsHidden) only affects the MAIN chart's own
// objects — the subpanel (divider/buildSubpanelShapes) is a separate area,
// left alone.
function allShapes() {
  const extras = S.objectsHidden ? [] : [...buildAnalyzerOriginShapes(), ...extraShapes()];
  return [...buildDivider(), ...buildSubpanelShapes(), ...extras, ...S.shapes];
}

// Builds a full layout with dynamic yaxis domains based on active subpanel.
// Pass y1Range to lock the main price axis (prevents autorange on Plotly.react).
function buildLayout(xRange, y1Range = null) {
  const hasSub = S.subpanel !== 'none';
  const logY = S.subpanel === 'spectrogram' && S.spectrogramSettings.logY;

  return {
    ...baseLayout,
    shapes: allShapes(),
    annotations: [...(S.bandAnnotations ?? []), ...(S.objectsHidden ? [] : extraAnnotations())],
    xaxis: buildXAxis(xRange),
    yaxis: {
      ...BASE_YAXIS,
      type: S.mainLogScale ? 'log' : 'linear', // main chart's own log toggle — independent of spectrogram's logY (that one's the oscillator's frequency axis), see sma/ui/cursor_tools.js:toggleMainLogScale
      domain: hasSub ? [0.22, 1.0] : [0.0, 1.0],
      ...(y1Range ? { range: y1Range } : {}),
    },
    yaxis2: {
      domain: [0.0, 0.18],
      gridcolor: CHART_GRID,
      linecolor: '#30363d',
      tickcolor: '#30363d',
      tickfont:  { color: '#8b949e', size: 10 },
      side: 'right',
      fixedrange: false,
      visible: hasSub,
      showticklabels: hasSub,
      ...(logY ? { type: 'log' } : {}),
    },
  };
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
function getCurrentRanges() {
  const el = document.getElementById('chart');
  const fl = el?._fullLayout;
  const ly = el?.layout;
  return {
    x:  fl?.xaxis?.range  ?? ly?.xaxis?.range  ?? null,
    y1: fl?.yaxis?.range  ?? ly?.yaxis?.range  ?? null,
  };
}

export function dayLabel(beginStr) {
  const d = new Date(beginStr.slice(0, 10) + 'T12:00:00');
  return `${d.getDate()} ${MONTH_RU[d.getMonth()]} ${d.getFullYear()}, ${DAY_RU[d.getDay()]}`;
}

// Date `stepsAhead` forecast steps past `baseDateStr` — needed only for
// simplex_ensemble, whose per-origin trajectories run H steps into the
// future with no known real bar to anchor each step on (unlike band_lambda's
// h=1/h=2, which stay inside computeGeometryFromWidth's bar-count geometry).
// Port of prototype/forcaster/ui/app7-simplex-ensemble.py's future_times:
// business-day step for daily bars (pandas BDay — skips Sat/Sun), plain
// interval-second step otherwise. Purely a VISUALIZATION convenience — the
// model has no opinion on which real calendar date a future step lands on.
export function futureDateAt(baseDateStr, stepsAhead, interval) {
  const base = new Date(baseDateStr.slice(0, 10) + 'T00:00:00Z');
  if (interval === '1d') {
    let d = new Date(base);
    let remaining = stepsAhead;
    while (remaining > 0) {
      d = new Date(d.getTime() + 86400 * 1000);
      const day = d.getUTCDay(); // 0=Sun, 6=Sat
      if (day !== 0 && day !== 6) remaining--;
    }
    return d.toISOString().slice(0, 10);
  }
  const sec = INTERVAL_SECONDS[interval] ?? 86400;
  return new Date(base.getTime() + stepsAhead * sec * 1000).toISOString().slice(0, 10);
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
    line: { color: '#58a6ff', width: 1, dash: 'dash' },
  };
}

export function applyShapes() {
  Plotly.relayout('chart', { shapes: allShapes(), annotations: S.bandAnnotations ?? [] });
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
export function setMainLogScale(enabled) {
  S.mainLogScale = enabled;
  if (_rafHandle) {
    cancelAnimationFrame(_rafHandle);
    _rafHandle = null;
    _pendingPlotArgs = null;
  }
  Plotly.relayout('chart', {
    'yaxis.type': enabled ? 'log' : 'linear',
    'yaxis.autorange': true,
  });
}

// Places the plain origin marker without touching whatever forecast overlays
// (pinned/selected) are currently showing — recomputes their shapes fresh so
// the new origin line doesn't clobber them (see buildVisibleOverlays below).
export function setOrigin(ts) {
  S.originTs = ts;
  const { shapes } = buildVisibleOverlays();
  S.shapes = [...shapes, originLineShape(ts)];
  applyShapes();
}

// ── zigzag overlay (band_lambda) ────────────────────────────────────────────
// Drawn at extreme_date (where the reversal actually printed) — confirm_date
// (the causal point) is used only for the forecast origin marker, never for
// the zigzag line itself. See sma/core/forecast/band_lambda.py:build_zigzag.
export function buildZigzagTrace(pivots) {
  if (!pivots?.length) return null;
  return {
    type: 'scatter', name: 'Зигзаг',
    x: pivots.map(p => p.extreme_date),
    y: pivots.map(p => p.price),
    mode: 'lines+markers',
    line: { color: '#d29922', width: 1 },
    marker: { size: 4, color: '#d29922' },
    hovertemplate: '%{y:.4f}<extra></extra>',
  };
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
export const COLOR_DOWN = '248,81,73';   // matches candlestick decreasing color #f85149
export const COLOR_UP    = '63,185,80';  // matches candlestick increasing color #3fb950
export const LEVEL_OPTIONS = [25, 50, 60, 75, 90, 95];
export const DEFAULT_LEVELS = [50, 75, 90];
export const DEFAULT_ZONE_OPACITY = 0.22;
const STEP_LABEL = { 1: 'Шаг 1 (уход)', 2: 'Шаг 2 (уход+возврат)' };

export const DEFAULT_STEP_BARS = 5;

// Horizontal span for the two step zones, driven by the "Ширина шага"
// slider (sma/ui/forecast.js:applyBandWidth) — replaces an earlier
// drag-to-resize attempt that turned out unreliable (1px shape borders are
// very hard to grab with a mouse; see project feedback 2026-08-07). Kept
// as a named export so forecast.js can compute the same geometry the
// slider persists via POST /forecasts/{id}/geometry.
export function computeGeometryFromWidth(originTs, interval, widthBars = DEFAULT_STEP_BARS) {
  const sec = INTERVAL_SECONDS[interval] ?? 86400;
  const base = new Date(originTs.slice(0, 10)).getTime();
  const at = n => new Date(base + n * sec * 1000).toISOString().slice(0, 10);
  return {
    step1: { x0: at(1), x1: at(widthBars) },
    step2: { x0: at(widthBars), x1: at(widthBars * 2) },
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

export function buildBandZoneShapes(forecastResult, geometry, levels, opacity) {
  const shapes = [];
  if (!forecastResult) return shapes;

  // Anchored to origin_extreme_date, NOT origin_date (confirmation bar) —
  // origin_price is the pivot's EXTREME price, a different bar entirely
  // from confirmation; placing zones at the confirm date would visually
  // detach them from the price they're actually centered on (project
  // feedback 2026-08-07: "22 апреля... а сам прогноз строится в точке 26
  // мая"). origin_date remains the causal anchor everywhere else (task
  // truncation, origin_candle_id resolution) — only the VISUAL placement
  // changes here.
  const originTs = forecastResult.origin_extreme_date;
  const originLogPrice = forecastResult.origin_log_price;
  const geo = geometry || computeGeometryFromWidth(originTs, S.interval);
  // Widest first, narrowest last — later shapes draw on top, so same-alpha
  // overlapping rects naturally stack denser toward the center.
  const activeLevels = [...(levels?.length ? levels : DEFAULT_LEVELS)].sort((a, b) => b - a);
  const baseOpacity = opacity ?? DEFAULT_ZONE_OPACITY;
  const direction = forecastResult.origin_direction;

  for (const h of [1, 2]) {
    const step = forecastResult.steps?.[h];
    const g = geo[`step${h}`];
    if (!step?.ok || !g || !step.pool_values?.length) continue;
    const isDown = h === 1 ? direction > 0 : direction < 0;
    const color = isDown ? COLOR_DOWN : COLOR_UP;

    activeLevels.forEach(levelPct => {
      const { lo, hi } = levelBounds(step, originLogPrice, levelPct);
      shapes.push({
        type: 'rect', xref: 'x', yref: 'y',
        x0: g.x0, x1: g.x1, y0: lo, y1: hi,
        fillcolor: `rgba(${color},${baseOpacity})`,
        line: { width: 0 },
        layer: 'above',
      });
    });
  }

  // Divider between step1 and step2 so two adjacent zones (often the same
  // color pair-wise across different forecasts) never read as one field.
  const g1 = geo.step1, g2 = geo.step2;
  if (g1 && g2 && forecastResult.steps?.[1]?.ok && forecastResult.steps?.[2]?.ok) {
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
    line: { color: '#ffd600', width: 1, dash: 'dot' },
  };
}

// Small text labels ("Шаг 1 (уход)" / "Шаг 2 (уход+возврат)") above each
// zone — shapes carry no hover/legend of their own, so without this the
// two colored fields have no way to say what they mean at a glance.
export function buildBandAnnotations(forecastResult, geometry, levels) {
  if (!forecastResult) return [];
  const geo = geometry || computeGeometryFromWidth(forecastResult.origin_extreme_date, S.interval);
  const widest = Math.max(...(levels?.length ? levels : DEFAULT_LEVELS));
  const out = [];
  for (const h of [1, 2]) {
    const step = forecastResult.steps?.[h];
    const g = geo[`step${h}`];
    if (!step?.ok || !g || !step.pool_values?.length) continue;
    const { hi, lo } = levelBounds(step, forecastResult.origin_log_price, widest);
    const isDown = h === 1 ? forecastResult.origin_direction > 0 : forecastResult.origin_direction < 0;
    out.push({
      x: g.x0, y: isDown ? lo : hi, xref: 'x', yref: 'y',
      text: STEP_LABEL[h], showarrow: false,
      font: { size: 10, color: '#c9d1d9' },
      bgcolor: 'rgba(13,17,23,0.6)',
      xanchor: 'left', yanchor: isDown ? 'top' : 'bottom',
    });
  }
  return out;
}

// ── model registry: single extension point for a new forecast model ────────
// Every builder referenced here (buildBandZoneShapes/buildBandAnnotations/
// buildSimplexOriginTraces/buildSimplexMeanBandTraces) already lives in this
// file — kept as a plain local map (not a separate module) specifically to
// avoid a chart.js <-> forecast_models.js import cycle: the builders are
// chart.js internals, and the composer that dispatches through this map
// (buildVisibleOverlays, renderChart) also lives here. Adding a THIRD model
// means: write its own buildX traces/shapes function above, add one entry
// below with a distinct marker symbol — nothing else in this file changes.
const MODEL_DISPLAY = {
  band_lambda: {
    label: 'band_lambda',
    marker: { symbol: 'triangle-up', color: '#58a6ff' },
    buildOverlay(result, geometry) {
      return {
        shapes: buildBandZoneShapes(result, geometry, S.displayPreset.levels, S.displayPreset.opacity),
        annotations: buildBandAnnotations(result, geometry, S.displayPreset.levels),
      };
    },
  },
  simplex_ensemble: {
    label: 'simplex_ensemble',
    marker: { symbol: 'circle', color: '#f0883e' },
    buildOverlay(result) {
      const traces = [...buildSimplexOriginTraces(result)];
      if (S.simplexShowMean) traces.push(...buildSimplexMeanBandTraces(result, S.simplexBandPct));
      return { traces };
    },
  },
};

// ── traces ────────────────────────────────────────────────────────────────────
const MARKER_STACK_STEP = 0.006; // 0.6% of price per stacking level, matches the old LA-model marker stack

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
// Pure function of (S.historyForecasts, S.pinnedForecastIds, S.selectedForecastId)
// — cached so a renderChart triggered by something entirely unrelated (an
// analyzer's own settings) doesn't re-sort/re-group the full forecast
// history every time. S.historyForecasts is reassigned wholesale on every
// fetch (forecast_history.js:refreshHistory), so identity works for it;
// S.pinnedForecastIds is a Set mutated IN PLACE (.add/.delete/.clear), so
// its identity never changes — comparing a sorted snapshot string instead
// (cheap: only as many entries as are actually pinned, not the full history).
let _markerTracesCache = { forecasts: null, pinnedKey: null, selected: undefined, traces: null };

function buildForecastMarkerTraces() {
  if (!S.historyForecasts.length) return [];

  const pinnedKey = S.pinnedForecastIds.size ? [...S.pinnedForecastIds].sort().join(',') : '';
  const cache = _markerTracesCache;
  if (
    cache.forecasts === S.historyForecasts &&
    cache.pinnedKey === pinnedKey &&
    cache.selected === S.selectedForecastId
  ) {
    return cache.traces;
  }

  const byModel = new Map();
  for (const f of S.historyForecasts) {
    if (!byModel.has(f.model_type)) byModel.set(f.model_type, []);
    byModel.get(f.model_type).push(f);
  }

  const traces = [];
  for (const [modelType, list] of byModel) {
    const style = MODEL_DISPLAY[modelType]?.marker ?? { symbol: 'triangle-up', color: '#8b949e' };
    const dateCount = {};
    const xs = [], ys = [], ids = [], texts = [], opacities = [], sizes = [];
    // Oldest first so the newest forecast at a shared date ends up on top
    // (last in the array = drawn last = on top in Plotly scatter traces).
    const ordered = [...list].sort((a, b) => a.id - b.id);
    for (const f of ordered) {
      const d = f.origin_extreme_ts;
      const idx = dateCount[d] ?? 0;
      dateCount[d] = idx + 1;
      const pinned = S.pinnedForecastIds.has(f.id);
      const selected = f.id === S.selectedForecastId;
      xs.push(d);
      ys.push(f.origin_price * (1 - idx * MARKER_STACK_STEP));
      ids.push(f.id);
      texts.push(`#${f.id}${f.is_stale ? ' ⚠ устарел' : ''} — ${f.origin_direction > 0 ? '▲' : '▼'} ${f.origin_price.toFixed(4)}`);
      opacities.push(selected || pinned ? 0.95 : 0.55);
      sizes.push(selected ? 14 : 11);
    }

    traces.push({
      type: 'scatter',
      name: MODEL_DISPLAY[modelType]?.label ?? modelType,
      x: xs, y: ys,
      customdata: ids,
      text: texts,
      mode: 'markers',
      marker: {
        symbol: style.symbol, size: sizes,
        color: ids.map(id => id === S.selectedForecastId ? '#ffd600' : style.color),
        opacity: opacities, line: { width: 0 },
      },
      hovertemplate: '<b>%{x|%d.%m.%Y}</b><br>%{text}<extra></extra>',
      showlegend: true,
    });
  }
  _markerTracesCache = { forecasts: S.historyForecasts, pinnedKey, selected: S.selectedForecastId, traces };
  return traces;
}

// Shapes/annotations/traces for every currently-visible forecast (pinned +
// the temporarily selected one, if any) — the core of the "show several
// forecasts, incl. across models, at once" mechanism. Each visible forecast
// also gets its own origin crosshair (vertical + horizontal dotted lines).
function buildVisibleOverlays() {
  const visibleIds = new Set(S.pinnedForecastIds);
  if (S.selectedForecastId != null) visibleIds.add(S.selectedForecastId);

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
    shapes.push(originLineShape(f.result.origin_extreme_date), originPriceLineShape(f.result.origin_price));
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

function buildBaseTraces() {
  const c = S.candles;

  const candlestick = {
    type: 'candlestick', name: S.ticker,
    x: c.map(r => r.begin), open: c.map(r => r.open),
    high: c.map(r => r.high), low: c.map(r => r.low), close: c.map(r => r.close),
    increasing: { line: { color: '#3fb950' }, fillcolor: '#3fb950' },
    decreasing: { line: { color: '#f85149' }, fillcolor: '#f85149' },
    whiskerwidth: 0, showlegend: false,
  };

  const dayTrace = {
    type: 'scatter', x: c.map(r => r.begin), y: c.map(r => r.close),
    mode: 'none', name: '',
    text: dayLabelsFor(c),
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

function schedulePlotlyReact(traces, layout, config) {
  _pendingPlotArgs = [traces, layout, config];
  if (_rafHandle) return;
  _rafHandle = requestAnimationFrame(() => {
    _rafHandle = null;
    const [t, l, c] = _pendingPlotArgs;
    Plotly.react('chart', t, l, c);
    _bindClickIfNeeded();
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
  // a declutter toggle for the MAIN chart's own drawn objects (pinned/
  // selected forecast zones, analyzer overlays, zigzag): hides what's
  // DRAWN without touching any of their own show/pin settings. Forecast
  // MARKERS stay visible either way — they're a lightweight index, not
  // clutter, and the "выбор объектов" cursor tool still needs them clickable.
  const hideObjects = S.objectsHidden;

  const { shapes, annotations, traces: overlayTraces } = hideObjects
    ? { shapes: [], annotations: [], traces: [] }
    : buildVisibleOverlays();
  // S.originTs ("next forecast starts here", set by setOrigin on a plain
  // chart click) is independent of which past forecasts are pinned/selected
  // — always show it too, alongside any forecast crosshairs, so a full
  // renderChart (tab switch, candle reload, pin toggle, ...) never drops it.
  if (S.originTs) shapes.push(originLineShape(S.originTs));
  S.shapes = shapes;
  S.bandAnnotations = annotations;

  const baseTraces = buildBaseTraces();
  const zz = hideObjects ? null : buildZigzagTrace(S.zigzagPivots);
  const allTraces = [...baseTraces];
  if (zz) allTraces.push(zz);
  if (!hideObjects) allTraces.push(...buildAnalyzerMainTraces());
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

  schedulePlotlyReact(allTraces, buildLayout(xRange, y1Range), PLOTLY_CONFIG);
}

// ── simplex_ensemble overlay builders ────────────────────────────────────────
// Different shape from band_lambda's two discrete h=1/h=2 zones: H
// continuous per-origin trajectories (DATA traces, not shapes — there's no
// fixed-width geometry to persist/drag here) + an optional reactive
// mean/percentile-band pair, recomputed client-side from the "raw" per-
// origin rel arrays every result carries (see sma/core/forecast/
// simplex_ensemble.py:forecast_ensemble — per_origin[*].rel).

function simplexTrajectoryX(originDateStr, horizon, interval) {
  const xs = [originDateStr.slice(0, 10)];
  for (let h = 1; h <= horizon; h++) xs.push(futureDateAt(originDateStr, h, interval));
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
      line: { width: idx === 0 ? 1.5 : 1, color: 'rgba(100,180,255,0.55)' },
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
      line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(255,214,0,0.15)',
      name: `p${pLo}–p${pHi}`, hoverinfo: 'skip',
    },
    {
      type: 'scatter', mode: 'lines+markers', x: xs, y: toY(meanRel),
      name: `среднее по ${per.length} origin`,
      line: { color: '#ffd600', width: 2.5 }, marker: { size: 3 },
    },
  ];
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
// The "price level" cursor tool (S.activeMainTool === 'price_level', see sma/ui/cursor_tools.js)
// needs the click's actual Y-axis DATA value at the raw pixel position, not
// a trace's snapped data point — Plotly only reports snapped values in
// data.points. Converting the pixel back to a price uses the y-axis
// object's own `.p2d()` (pixel-to-data), an internal-but-long-stable Plotly
// method attached to `_fullLayout.yaxis` after a plot exists — same class
// of internals this file already reaches into for getCurrentRanges().
let _onForecastClick    = null;
let _onCandleClick      = null;
let _onPriceLevelClick  = null;
let _chartClickBound    = false;
let _markerTraceIndices = []; // updated by renderChart — one index per model's marker trace

// Register callbacks; actual binding deferred until after first Plotly.react.
export function initChartEvents(onForecastClick, onCandleClick, onPriceLevelClick) {
  _onForecastClick   = onForecastClick;
  _onCandleClick     = onCandleClick;
  _onPriceLevelClick = onPriceLevelClick;
}

function _priceAtClick(gd, data) {
  const yaxis = gd._fullLayout?.yaxis;
  if (!yaxis?.p2d || !data.event) return null;
  const rect = gd.getBoundingClientRect();
  const py = data.event.clientY - rect.top - gd._fullLayout._size.t;
  return yaxis.p2d(py);
}

function _bindClickIfNeeded() {
  if (_chartClickBound || !_onForecastClick) return;
  const gd = document.getElementById('chart');
  gd.on('plotly_click', data => {
    if (!data.points.length) return;

    // Price-level tool consumes the click entirely — no marker inspection,
    // no origin placement, just "where on the price axis did this land"
    // (see cursor_tools.js:setPendingPriceLevel). Gated on the Основной
    // график tab specifically — the tool's own toolbar/panel only ever
    // shows there, but S.activeMainTool itself isn't reset just by
    // switching tabs, so a stale 'price_level' selection must not hijack
    // clicks on the Осциллятор tab too.
    if (S.activeTab === 'main' && S.activeMainTool === 'price_level') {
      const price = _priceAtClick(gd, data);
      if (price != null) _onPriceLevelClick?.(price);
      return;
    }

    // dayTrace is always curveNumber 0 (moved before candlestick for hover order).
    const dayPt = data.points.find(p => p.curveNumber === 0) ?? data.points[0];
    if (!dayPt) return;
    const clickDate = String(dayPt.x).slice(0, 10);
    if (!/^\d{4}-\d{2}-\d{2}/.test(clickDate)) return;

    // Any of the per-model marker traces (_markerTraceIndices, set by
    // renderChart) — customdata on that point is the forecast id directly,
    // no need to re-derive it from the date (robust even when several
    // models/forecasts share a date).
    const markerPt = data.points.find(p => _markerTraceIndices.includes(p.curveNumber));
    _onForecastClick(markerPt ? markerPt.customdata : null, clickDate);

    _onCandleClick(clickDate);
  });
  _chartClickBound = true;
}
