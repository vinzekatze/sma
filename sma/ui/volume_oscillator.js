import { S } from './state.js';
import { api } from './api.js';
import { renderChart } from './chart.js';
import { registerTool, claimSubpanel, releaseSubpanel } from './tools.js';

// ── volume analyzer (Осциллятор tab) ──────────────────────────────────────
// Standard red/green volume bars, entirely client-side — `volume` is
// already present on every candle from GET /candles (sma/core/db.py's
// candles table has had the column all along; the frontend just never read
// it before this tool). Bar color follows the SAME rule the candlesticks
// themselves use (close >= open -> green, else red), not a separate
// up/down-vs-previous-close convention, so the two panels read consistently
// at a glance.

function redraw() {
  renderChart({ preserveRange: true });
}

// ── settings persist ──────────────────────────────────────────────────────

function readVolumeSettings() {
  return { showOscillator: document.getElementById('vol-show-oscillator').checked };
}

function applyVolumeSettings(p) {
  if (!p) return;
  document.getElementById('vol-show-oscillator').checked = !!p.showOscillator;
  S.volumeSettings = { ...p };
}

export async function loadVolumeDefaults() {
  if (!S.instrumentId) return;
  try {
    const res = await api(
      'GET',
      `/series/analysis-settings?instrument_id=${S.instrumentId}&interval=${S.interval}&analyzer_type=volume`
    );
    if (res.params) applyVolumeSettings(res.params);
  } catch (_) { /* non-fatal */ }
}

let _saveTimer = null;

function saveVolumeSettings() {
  if (!S.instrumentId) return;
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    api('POST', '/series/analysis-settings', {
      instrument_id: S.instrumentId, interval: S.interval,
      analyzer_type: 'volume', params: S.volumeSettings,
    }).catch(() => {});
  }, 500);
}

// "Показать осциллятор" is the ONLY control — no fetch needed at all (data
// is already in S.candles), so unlike variance_oscillator's equivalent
// toggle this never has an async branch.
export function toggleVolumeDisplay() {
  const settings = readVolumeSettings();
  S.volumeSettings = settings;
  if (settings.showOscillator) claimSubpanel('volume'); else releaseSubpanel('volume');
  redraw();
  saveVolumeSettings();
}

// Called once after candles (re)load — nothing to compute ahead of time
// (built directly from S.candles on each render), just restores the
// checkbox/subpanel state if it was on last time this ticker was open.
export function initVolumeForTicker() {
  if (S.volumeSettings.showOscillator) claimSubpanel('volume');
}

// ── registry entry: subpanel bar trace ───────────────────────────────────
function buildVolumeSubpanelTraces() {
  if (!S.candles.length) return [];
  return [{
    type: 'bar', name: 'Объём',
    x: S.candles.map(c => c.begin), y: S.candles.map(c => c.volume),
    yaxis: 'y2',
    marker: { color: S.candles.map(c => (c.close >= c.open ? '#3fb950' : '#f85149')) },
    hovertemplate: '%{y:,.0f}<extra></extra>',
  }];
}

registerTool({
  type: 'volume',
  icon: 'volume',
  label: 'Объём',
  panelId: 'tool-panel-volume',
  showCheckboxId: 'vol-show-oscillator',
  buildSubpanelTraces: buildVolumeSubpanelTraces,
  // no buildMainTraces/buildOriginShape/onOriginClick/resetOrigin — this
  // tool never draws on the main chart and has no origin concept.
});
