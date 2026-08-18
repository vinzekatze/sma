export const S = {
  candles:              [],
  ticker:               '',
  dataSource:           'moex',
  interval:             '1d',
  originTs:             null,
  instrumentId:         null,
  shapes:               [],
  activeTab:            'main', // 'main' (Основной график) | 'oscillator' — Прогноз folded into 'main' as a third toolbar row, see sma/ui/tools.js
  historyForecasts:     [],   // unified across ALL model_types now, see sma/ui/forecast_history.js
  subpanel:             'none', // oscillator panel: 'none' | <tool type> — see sma/ui/tools.js registry
  activeMainTool:       'free',   // which main-chart-surface tool (cursor tool OR analyzer OR forecast model — all THREE share ONE exclusivity slot, see tools.js module docstring) is selected — its settings panel shows AND it receives chart clicks (onOriginClick, or the price-level special case in chart.js), see sma/ui/tools.js:selectTool. Defaults to 'free' (project feedback round 8 — the free-mouse tool is a real selectable entry, not a null/absent state) — a plain click does nothing special while it's active, same as before cursor tools joined this slot (round 7).
  activeOscillatorTool: null,   // same idea for oscillator-surface tools (no buildMainTraces) — no origin/click routing, purely which panel shows on the Осциллятор tab, kept as its own separate slot (different tab/physical subpanel, no click-routing overlap with activeMainTool)

  // Основной график tab — price levels (S.activeMainTool === 'price_level',
  // see sma/ui/cursor_tools.js) — pure client-side state, no backend.
  priceLevels:        [],     // fixed horizontal levels: [{id, price}], session-only
  pendingPriceLevel:  null,   // the not-yet-fixed level currently following clicks, or null
  mainLogScale:       false,  // main chart y-axis — independent of spectrogram's own logY (that one's the oscillator's frequency axis)
  objectsHidden:      false,  // master declutter toggle — temporarily hides pinned forecasts/analyzer overlays/zigzag/price levels without touching their own settings

  // Прогноз section — unified multi-forecast display (any number of forecasts,
  // any mix of models, shown at once). See sma/ui/forecast_models.js
  // (MODEL_REGISTRY, per-model marker style + overlay builder) and
  // sma/ui/chart.js:renderForecastOverlays (the composer). Replaces the
  // earlier one-forecast-at-a-time fields (forecastId/activeForecastOutput/
  // bandGeometry/bandShapes/bandAnnotations/simplexResult).
  renderedForecasts:  new Map(), // forecastId -> {model_type, result, params, zone_geometry} — cache of GET /forecasts/{id}, avoids refetching on every visibility toggle
  pinnedForecastIds:  new Set(), // "eye" toggle in history — session-only, NOT persisted to DB (by design, see plan)
  selectedForecastId: null,      // temporary focus (marker/history-row click) — drives the owning model's #results-<model_type> AND, if not pinned, a temporary chart overlay

  // Анализ tab — spectrogram analyzer (first of what's meant to be a small
  // family; see sma/core/analysis/)
  spectrogramData:     null, // raw POST /series/spectrogram response (times/freqs/sxx_db/cutoffs)
  spectrogramSettings: {
    depthBars: 500, nperseg: 64, overlapPct: 75, fmin: 0, fmax: 0.5,
    logY: false, contrastPct: 5, // contrastPct=5 -> color range = [5th, 95th] percentile, matches the prototype default
  },

  // Основной график tab — trend_ruler analyzer (rolling OLS trend + ±k·std
  // residual bands, see sma/core/analysis/trend_variance.py — the backend
  // module name predates the split and stays as-is, it's still the same
  // math). Split from a single combined "trend_variance" analyzer into TWO
  // tools (project feedback 2026-08-18, round 5): this one is main-chart
  // only (trend line/bands/accel fan/origin); the FULL-history slope/var
  // rolling pass moved to its own oscillator-only tool, see
  // variance_oscillator.js. showOnChart gates the main-chart overlay —
  // independent of which tab is active (same "explicit toggle, persists
  // until turned off" model as forecast pins), default OFF so loading a
  // ticker never auto-shows it (project feedback 2026-08-18: "он постоянно
  // отображается").
  //
  // Bands display is FOUR independent toggles (project feedback 2026-08-18,
  // round 4 — explicit combinations like "borders only, but extrapolate the
  // fill" must be possible, not just a single showExtension flag):
  //   showBands       — the shaded ±k·std fill, within the actual window
  //   showBandBorders — a visible border LINE at each band's hi/lo edge,
  //                     within the actual window (previously invisible —
  //                     only the fill's own color implied where a band's
  //                     edge was, hard to tell apart with several bands)
  //   extendBands     — continue the shaded fill past the window (dashed
  //                     time range in the future), default OFF
  //   extendBorders   — continue the border LINE past the window, default
  //                     ON (this is what the old single showExtension did)
  // nFuture is shared by both extend* flags (same step count either way).
  // bandOpacity is ONE user-set alpha applied uniformly to every band's
  // fill — replaces an earlier per-band computed gradient that, on
  // inspection, had the widest/narrowest alpha backwards from its own
  // comment; matches the same "same opacity, widest drawn first" stacking
  // trick already used for band_lambda's zones (chart.js:buildBandZoneShapes).
  trendRulerData:     null, // client-side computed {trend,bands,extension,accel_fan} — see trend_ruler.js:computeLivePreview
  trendRulerOriginTs: null, // this analyzer's OWN origin — null = "live" (always last bar); set by clicking a candle while its toolbar tool is active
  trendRulerSettings: {
    window: 200, bands: [2.0],
    showBands: true, showBandBorders: true, bandOpacity: 0.18,
    extendBands: false, extendBorders: true, nFuture: 50,
    showOnChart: false, showAccelFan: false, mAccel: 50, nAccel: 50,
  },

  // Осциллятор tab — variance_oscillator analyzer: the OTHER half of the
  // pre-split trend_variance (full-history rolling slope/var, backend-only
  // — see sma/core/analysis/trend_variance.py). Has its OWN window setting,
  // independent of trend_ruler's — "Подтянуть окно с линейки" (see
  // variance_oscillator.js:pullWindowFromRuler) copies it once rather than
  // keeping the two permanently linked, since you may deliberately want a
  // different window for the oscillator's whole-history view than for the
  // ruler's local trend window.
  varianceOscData:     null, // raw {times, slope, var} from POST /series/trend-variance's oscillator field
  varianceOscSettings: {
    window: 200, oscMode: 'slope', showOscillator: false,
  },

  // Основной график tab — moving_averages analyzer (sma/ui/moving_averages.js).
  // N independently configured lines (own type SMA/EMA/WMA + period each),
  // entirely client-side — no origin concept (whole-history rolling series,
  // not window-relative). maData is keyed by series id: {times, values}.
  maData:     {},
  maSettings: { showOnChart: false, series: [] },

  // Основной график tab — zigzag_tool analyzer (sma/ui/zigzag_tool.js). A
  // freestanding set of zigzags (own T%/min_bars each), INDEPENDENT of the
  // Прогноз section's calibration-bound zigzag (S.zigzagPivots below) — reuses
  // the same generic GET /forecasts/zigzag endpoint per series. zigzagToolData
  // is keyed by series id: pivots[] (fetched, not persisted — only the
  // T%/min_bars settings that produce them are saved).
  zigzagToolData:     {},
  zigzagToolSettings: { showOnChart: false, series: [] },

  // Осциллятор tab — volume analyzer (sma/ui/volume_oscillator.js). No own
  // data field — built directly from S.candles.volume on each render, see
  // buildVolumeSubpanelTraces.
  volumeSettings: { showOscillator: false },

  // band_lambda (Прогноз section)
  forecastSettingsList:  [],   // GET /forecast-settings — all calibrated (T, pool_key) rows
  calibratedTs:          [],   // distinct t_query values with at least one row, sorted
  selectedTs:            null, // currently selected t_query in the quick T-selector
  activeSettingsId:      null, // is_active forecast_settings row id for selectedTs
  zigzagPivots:          [],   // GET /forecasts/zigzag — full pivot list; rendered as a trace, not a shape
  bandWidthBars:         5,    // "Ширина шага" slider live value — applied to S.selectedForecastId's zone_geometry (if it's band_lambda), see chart.js:computeGeometryFromWidth
  displayPreset:         { levels: [50, 75, 90], opacity: 0.22 }, // matches prototype/forcaster/ui/app9.py defaults

  // simplex_ensemble — ансамбль Simplex projection по origin,
  // sma/ui/simplex_ensemble.js. Params mirror sma/core/forecast/
  // simplex_ensemble.py's DEFAULT_* constants; prefilled from GET
  // /forecast-settings/defaults on ticker/interval switch (last-used, see
  // db.py forecast_defaults table docstring) or left at these hardcoded
  // fallbacks when nothing is saved yet.
  simplexParams: {
    window: 10, horizon: 20,
    xy_x: 3, xy_y: 0, xi_add: 1, blend_alpha: 0.5,
    n_levels: 6, p_cascade_max: 5000, bars: 0,
    pca_p_range: [3, 150], pca_thr_range: [0.80, 0.85],
    use_lp_corr: true,
  },
  simplexBandPct:  50,    // "Ширина полосы P" slider — band = [50-P/2, 50+P/2] percentiles, recomputed client-side
  simplexShowMean: true,  // "Показывать среднее + полосу" checkbox
};

export const INTERVAL_SECONDS = {
  '1m': 60, '10m': 600, '1h': 3600,
  '1d': 86400, '1w': 604800, '1mo': 2592000,
};

export const MONTH_RU = ['янв','фев','мар','апр','май','июн','июл','авг','сен','окт','ноя','дек'];
export const DAY_RU   = ['Вс','Пн','Вт','Ср','Чт','Пт','Сб'];
