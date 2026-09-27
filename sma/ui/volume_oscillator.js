import { S } from './state.js';
import { registerTool } from './tools.js';

// ── volume analyzer (Осциллятор tab) ──────────────────────────────────────
// Standard red/green volume bars, entirely client-side — `volume` is
// already present on every candle from GET /candles (sma/core/db.py's
// candles table has had the column all along; the frontend just never read
// it before this tool). Bar color follows the SAME rule the candlesticks
// themselves use (close >= open -> green, else red), not a separate
// up/down-vs-previous-close convention, so the two panels read consistently
// at a glance.
//
// No settings at all (nothing to configure — a fixed derived view of
// S.candles) and so no analysis_settings persistence either: selecting this
// tool on the toolbar directly shows it (S.subpanel mirrors
// S.activeOscillatorTool, see tools.js §3.1 of docs/plans/frontend_
// improvements_plan.md) — before that change this module still needed a
// "показать осциллятор" checkbox + its own persisted showOscillator flag
// just to remember whether it was on; now that concept lives once, on
// S.activeOscillatorTool, not per-tool.

// ── registry entry: subpanel bar trace ───────────────────────────────────
function buildVolumeSubpanelTraces() {
  if (!S.candles.length) return [];
  const up = S.colorProfile.volume_up, down = S.colorProfile.volume_down; // settings.js
  return [{
    type: 'bar', name: 'Объём',
    x: S.candles.map(c => c.begin), y: S.candles.map(c => c.volume),
    yaxis: 'y2',
    marker: { color: S.candles.map(c => (c.close >= c.open ? up : down)) },
    hovertemplate: '%{y:,.0f}<extra></extra>',
  }];
}

// yaxis2 policy (project feedback 2026-08-20, §3.3 of docs/plans/
// frontend_improvements_plan.md): STARTS with floor at 0 (volume is never
// negative), auto ceiling — zoomable (fixedrange:false, revised the same
// day: "дисперсия и объем — нет возможности зумировать", an earlier pass
// had locked this and variance_osc's var mode entirely).
function volumeYAxisPolicy() {
  if (!S.candles.length) return { key: 'volume', fixedrange: false, range: null };
  let maxV = 0;
  for (const c of S.candles) if (c.volume > maxV) maxV = c.volume;
  return { key: 'volume', fixedrange: false, range: [0, maxV > 0 ? maxV * 1.05 : 1] };
}

registerTool({
  type: 'volume',
  icon: 'volume',
  label: 'Объём',
  panelId: 'tool-panel-volume',
  buildSubpanelTraces: buildVolumeSubpanelTraces,
  subpanelYAxisPolicy: volumeYAxisPolicy,
  // no buildMainTraces/buildOriginShape/onOriginClick/resetOrigin — this
  // tool never draws on the main chart and has no origin concept.
});
