// local_prefs.js — client-only UI/session state persisted to localStorage
// (docs/plans/frontend_improvements_plan.md §1.1b). Distinct from BOTH other
// persistence mechanisms in this app:
//   - analysis_settings (DB, per-ticker analyzer params — trend_ruler
//     window, spectrogram fmin/fmax, price levels' actual values as of
//     2026-08-20, see cursor_tools.js:loadPriceLevelDefaults) — this module
//     never touches that.
//   - app_settings / color profile (DB, global but edited explicitly via a
//     "Настройки приложения" form, see settings.js) — also untouched here.
// This module is for session-ish UI state that isn't worth a network round
// trip and shouldn't live in the DB at all: which cursor/tool was active,
// pinned forecast ids, the declutter/log-scale toggles, and (project
// feedback 2026-08-20: "лучше чтобы для всех инструментов сохранялось —
// допустимо делать это на фронте") each main-chart analyzer's own
// "закреплено на графике" flag (toolShowOnChart, GLOBAL — not per-ticker,
// since it's a standing preference like "I always want to see MA," not
// per-instrument data). Restores the chart's look instantly on page load,
// before any API call resolves.
//
// A caveat worth knowing before relying on this for anything ticker-scoped:
// activeMainTool is still UNCONDITIONALLY reset to 'free' by
// app.js:loadCandles on every ticker load (cursor/analyzer/forecaster
// origins genuinely don't carry meaning across a switch) — so restoring it
// from localStorage only has a visible effect in the window before the
// user picks a ticker; the instant they do, loadCandles's reset wins.
// pinnedForecastIds and activeOscillatorTool are DIFFERENT (revised
// 2026-08-25, project feedback: "отображение зафиксированных прогнозов и
// осциляторов не сохраняется") — loadCandles no longer clears/resets
// either: a pinned forecast id that doesn't exist in the new ticker's
// S.historyForecasts simply renders nothing (chart.js:
// buildForecastMarkerTraces only draws pins present in the current
// history), so leaving stale ids in place is harmless and lets them
// "reappear" if the user switches back; the still-active oscillator gets
// its onSelected hook re-triggered after the switch (see loadCandles) so it
// refetches fresh data for the new instrument instead of showing stale
// data or nothing. mainLogScale/objectsHidden/toolShowOnChart were never
// touched by loadCandles in the first place.

const STORAGE_KEY = 'sma_local_prefs';
const VERSION = 1;
const SAVE_DEBOUNCE_MS = 400;

// pinnedForecastIds has no natural cap from the UI (a user COULD pin dozens
// of forecasts over a long session) — bounded here so localStorage can't
// grow unboundedly; oldest pins are dropped first, newest kept, matching
// "most likely still relevant" instead of an arbitrary alphabetical/numeric
// cutoff. 200 is generous headroom over any realistic manual pin count.
const MAX_PINNED_IDS = 200;

// In-memory mirror of the persisted blob — read once (see _load), then kept
// in sync by every setPref call, so getPref never re-parses localStorage.
let _cache = null;

function _load() {
  if (_cache) return _cache;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      // Version check: a future schema change bumps VERSION — old,
      // incompatible blobs are discarded wholesale (silently falls back to
      // defaults) rather than risking a partially-wrong shape leaking into
      // S. Simpler and safer than per-field migration for state this
      // disposable (worst case: user re-picks their toolbar tool once).
      if (parsed && typeof parsed === 'object' && parsed.v === VERSION) {
        _cache = parsed;
        return _cache;
      }
    }
  } catch (_) {
    // localStorage unavailable (private mode, disabled, quota weirdness) or
    // corrupt JSON — degrade to defaults, never throw into a caller that
    // isn't expecting a persistence layer to fail.
  }
  _cache = { v: VERSION };
  return _cache;
}

let _saveTimer = null;

function _scheduleSave() {
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(_cache));
    } catch (_) {
      // Quota exceeded or unavailable — nothing to degrade TO here (the
      // in-memory _cache already has the value, only the persisted copy is
      // stale/missing), so this session just won't survive a reload.
    }
  }, SAVE_DEBOUNCE_MS);
}

export function getPref(key, defaultValue) {
  const blob = _load();
  return key in blob ? blob[key] : defaultValue;
}

export function setPref(key, value) {
  const blob = _load();
  blob[key] = value;
  _scheduleSave();
}

// ── whitelist-specific glue ──────────────────────────────────────────────
// Kept in this module (not scattered as ad-hoc getPref/setPref calls at
// every call site) so the full set of persisted fields is visible in one
// place — the mutation call sites (cursor_tools.js/tools.js/
// forecast_history.js) each call one of the small wrapper functions below,
// named after what they're doing, not "the localStorage layer".

export function hydrateLocalPrefs(S) {
  S.mainLogScale = getPref('mainLogScale', S.mainLogScale);
  S.objectsHidden = getPref('objectsHidden', S.objectsHidden);
  S.activeMainTool = getPref('activeMainTool', S.activeMainTool);
  S.activeOscillatorTool = getPref('activeOscillatorTool', S.activeOscillatorTool);
  const pinnedIds = getPref('pinnedForecastIds', []);
  S.pinnedForecastIds = new Set(pinnedIds);
  const pinnedTs = getPref('zigzagPinnedTs', []);
  S.zigzagPinnedTs = new Set(pinnedTs);
}

export function saveMainLogScale(value) {
  setPref('mainLogScale', value);
}

export function saveObjectsHidden(value) {
  setPref('objectsHidden', value);
}

export function saveActiveMainTool(value) {
  setPref('activeMainTool', value);
}

// Project feedback 2026-08-25: "отображение зафиксированных прогнозов и
// осциляторов не сохраняется" — the oscillator surface previously reset to
// 'none' on every ticker switch (§3.1 of docs/plans/frontend_improvements_
// plan.md, matching S.activeMainTool's reset to 'free') on the assumption
// that, like most main-surface tools' origins, it "carries no meaning"
// across a switch — revised: the user DOES want the same oscillator to stay
// selected/visible across ticker switches and reloads, refetching its data
// for the new instrument instead of dropping back to nothing (see
// app.js:loadCandles, which now re-triggers the still-active oscillator's
// onSelected hook after switching).
export function saveActiveOscillatorTool(value) {
  setPref('activeOscillatorTool', value);
}

export function savePinnedForecastIds(pinnedIdsSet) {
  const ids = [...pinnedIdsSet];
  setPref('pinnedForecastIds', ids.length > MAX_PINNED_IDS ? ids.slice(-MAX_PINNED_IDS) : ids);
}

export function saveZigzagPinnedTs(pinnedTsSet) {
  setPref('zigzagPinnedTs', [...pinnedTsSet]);
}

// ── per-tool "закреплено на графике" (§2.6/project feedback 2026-08-20) ──
// GLOBAL (not per-ticker) — a standing "I always want to see this" choice,
// same across every instrument, unlike each tool's own series/params
// (which stay per-ticker via analysis_settings, untouched by this). Keyed
// by tool type string ('trend_ruler'/'moving_averages'/'zigzag_tool', ...)
// in ONE map so adding a fourth pinnable main-chart tool needs no new
// top-level pref key, just another get/save call at its own toggle/init
// site.
export function getToolShowOnChart(toolType, defaultValue) {
  const map = getPref('toolShowOnChart', {});
  return toolType in map ? map[toolType] : defaultValue;
}

export function saveToolShowOnChart(toolType, value) {
  const map = { ...getPref('toolShowOnChart', {}) };
  map[toolType] = value;
  setPref('toolShowOnChart', map);
}
