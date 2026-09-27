export const S = {
  candles:              [],
  // Bars kept loaded on the main chart at once (sma/ui/candle_window.js) —
  // global app_settings, like moex_pool_workers/calibration_workers; the
  // hardcoded 1000 here matches sma/core/db.py:DEFAULT_CHART_WINDOW_BARS as
  // a synchronous default before settings.js:loadAppSettings' GET /settings
  // resolves and overwrites it, same reasoning as colorProfile below.
  chartWindowBars:      1000,
  candlesFullyLoadedBack: false, // true once a pan-back fetch returns fewer than requested — nothing older left for this ticker/interval
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
  activeOscillatorTool: 'none',  // same idea for oscillator-surface tools (no buildMainTraces) — no origin/click routing, purely which panel shows on the Осциллятор tab, kept as its own separate slot (different tab/physical subpanel, no click-routing overlap with activeMainTool). Never null — 'none' is a real registered tool (icon "eye-off", tools.js), same "always a real toolbar entry" principle 'free' uses on the main surface (project feedback 2026-08-20, §3.2 of docs/plans/frontend_improvements_plan.md). Always mirrors S.subpanel — see tools.js:selectTool/activateTool.

  // App-wide color profile (docs/plans/frontend_improvements_plan.md §1.1a,
  // expanded 2026-08-25 to every tool-meaningful color — see sma/ui/
  // settings.js:COLOR_ROLES for the single source of truth on which key
  // means what/where it's grouped) — fixed named roles, edited in the left
  // panel's own "Цветовой профиль" section (sma/ui/settings.js), backed by
  // app_settings in the DB (NOT per-ticker analysis_settings, NOT
  // localStorage — global to the app, like moex_pool_workers). Hardcoded
  // here as a synchronous default matching sma/core/db.py:DEFAULT_COLOR_PROFILE
  // exactly (keys AND values), so every consuming module never reads
  // `undefined` on the very first render, before settings.js:loadAppSettings'
  // GET /settings has resolved — that call overwrites this wholesale once
  // it lands.
  colorProfile: {
    price_level: '#ff0000',
    next_origin_marker: '#58a6ff',
    trend_ruler_line: '#1f77b4',
    trend_ruler_origin: '#bc8cff',
    trend_ruler_accel_up: '#2ca02c',
    trend_ruler_accel_down: '#d62728',
    trend_ruler_accel_line: '#7f7f7f',
    ma_palette: ['#f0883e', '#a5d6ff', '#d2a8ff', '#7ee787', '#ffa198', '#79c0ff'],
    zigzag_tool_palette: ['#d29922', '#f0883e', '#a5d6ff', '#7ee787', '#ffa198', '#d2a8ff'],
    forecast_zigzag: '#d29922',
    band_zone_up: '#3fb950',
    band_zone_down: '#f85149',
    candle_up: '#3fb950',
    candle_down: '#f85149',
    forecast_marker_selected: '#ffd600',
    forecast_marker_pinned_band_lambda: '#58a6ff',
    forecast_marker_pinned_simplex_ensemble: '#f0883e',
    forecast_marker_pinned_regime_mixture_potential: '#d2a8ff',
    simplex_origin_lines: '#64b4ff',
    simplex_mean_band: '#ffd600',
    range_forecast_line: '#ffd600',
    risk_corridor_close: '#ffffff',
    risk_corridor_high: '#26a69a',
    risk_corridor_low: '#ef5350',
    spectrogram_colorscale: 'Viridis',
    spectrogram_cutoff_line: '#ff5050',
    variance_slope_up: '#2ca02c',
    variance_slope_down: '#d62728',
    variance_var: '#9467bd',
    volume_up: '#3fb950',
    volume_down: '#f85149',
  },

  // Основной график tab — price levels (S.activeMainTool === 'price_level',
  // see sma/ui/cursor_tools.js). Fixed levels are per-ticker DB state as of
  // 2026-08-20 (analysis_settings, analyzer_type='price_level' — see
  // cursor_tools.js:loadPriceLevelDefaults/savePriceLevelsToDb), loaded on
  // every ticker switch; pendingPriceLevel (the not-yet-fixed one currently
  // following clicks) stays pure client-side, never persisted anywhere.
  priceLevels:        [],     // fixed horizontal levels: [{id, price}]
  pendingPriceLevel:  null,   // the not-yet-fixed level currently following clicks, or null
  priceLevelHoverPrice: null, // continuous cursor-following preview (project feedback 2026-08-25), never persisted — see cursor_tools.js:setPriceLevelHoverPrice
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
  varianceOscSettings: { window: 200, oscMode: 'slope' },

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
  zigzagToolStats:    {}, // keyed by series id: GET /forecasts/zigzag response's "stats" field (descriptive count/duration/amplitude per direction, full history) — see pivot_time_band.py:zigzag_direction_stats
  zigzagToolSettings: { showOnChart: false, series: [] },

  // Осциллятор tab — volume analyzer (sma/ui/volume_oscillator.js). No
  // settings at all — built directly from S.candles.volume on each render,
  // see buildVolumeSubpanelTraces; visibility is purely toolbar selection
  // (§3.1 of docs/plans/frontend_improvements_plan.md), nothing to persist.

  // band_lambda (Прогноз section) — T/m/theta are free live parameters on
  // every forecast request (no saved-per-T settings row any more — λ-
  // calibration removed 2026-09-12, see memory
  // project_phase7_calibration_removed_final); prefilled from GET
  // /forecast-settings/defaults on ticker/interval switch (last-used, same
  // forecast_defaults mechanism simplexParams below already uses) or left
  // at these hardcoded fallbacks when nothing is saved yet.
  selectedTs:            0.20, // T entered in the spinner (fraction, e.g. 0.20 = 20%)
  bandLambdaModelParams: { m: 6, theta: 0 },
  bandLambdaPool:        null, // GET /forecast-settings/pool — {categories,n,resolved_instrument_ids,resolved_tickers,pool_key} or null if unsaved
  zigzagPivots:          [],   // GET /forecasts/zigzag — full pivot list for the CURRENTLY SELECTED T only; rendered as a trace, not a shape
  // "Закрепить на графике" per T (§2.4 of docs/plans/frontend_improvements_
  // plan.md) — a pinned T's zigzag stays visible even while band_lambda
  // isn't the active tool or a different T is selected, same "active ∪
  // pinned" policy the rest of the app uses (§2.6). zigzagPinnedTs is
  // session-only, persisted to localStorage (sma/ui/local_prefs.js,
  // saveZigzagPinnedTs/hydrateLocalPrefs) — NOT the DB, same as
  // pinnedForecastIds. zigzagPinnedData is a plain fetch cache keyed by
  // t_query (populated once when a T is pinned, see forecast.js:
  // toggleZigzagPin — reuses whatever's already in S.zigzagPivots if that IS
  // the T being pinned, otherwise fetches fresh), never persisted itself
  // (cheap to re-derive, same reasoning zigzag_tool.js's own per-series data
  // already uses).
  zigzagPinnedTs:        new Set(),
  zigzagPinnedData:      {},
  // levels/opacity — zone rects (50/75/90% central intervals). tradeLevelPct
  // — one-sided "уровень доверия" line drawn on step1, valued from step2's
  // pool (see chart.js:tradeLevelPrice). showZones/showTradeLevel/trimZone1
  // — independent visibility toggles (§6 of the band_lambda settings panel).
  displayPreset: {
    levels: [50, 75, 90], opacity: 0.22,
    tradeLevelPct: 70, showZones: true, showTradeLevel: true, trimZone1: false,
  },

  // simplex_ensemble — ансамбль Simplex projection по origin,
  // sma/ui/simplex_ensemble.js. Params mirror sma/core/forecast/
  // simplex_ensemble.py's DEFAULT_* constants; prefilled from GET
  // /forecast-settings/defaults on ticker/interval switch (last-used, see
  // db.py forecast_defaults table docstring) or left at these hardcoded
  // fallbacks when nothing is saved yet.
  simplexParams: {
    window: 20, horizon: 20, // window default per project feedback 2026-08-25 (was 10)
    xy_x: 3, xy_y: 0, xi_add: 1, blend_alpha: 0.5,
    n_levels: 6, p_cascade_max: 5000, bars: 0,
    pca_p_range: [3, 150], pca_thr_range: [0.80, 0.85],
    use_lp_corr: true,
  },
  simplexBandPct:  50,    // "Ширина полосы P" slider — band = [50-P/2, 50+P/2] percentiles, recomputed client-side
  simplexShowMean: true,  // "Показывать среднее + полосу" checkbox

  // range_forecast — прогноз ДИАПАЗОНА (min(low)..max(high)) на h=1..H шагов
  // вперёд, sma/ui/range_forecast.js / sma/core/forecast/range_forecast.py.
  // Третий, независимый forecaster-инструмент — БЕЗ персистентности
  // прогнозов (нет forecasts-записи, см. docs/plans/app16_range_forecast_
  // migration_plan.md §3): только калибровка (θ+5λ+read-квантиль по
  // уровням) персистится (range_forecast_settings), сам прогноз считается
  // ЖИВЬЁМ при каждом изменении origin/настроек (POST /range-forecast/live).
  // mode:'calibrated' без свежей калибровки для (hSteps,p,theiler) — баннер
  // "нужна калибровка", прогноз не рисуется (см. план §8 п.1 — не
  // подставлять молча дефолтные θ=0/λ=0).
  rangeForecastSettings: { hSteps: 5, p: 8, theiler: 5, levels: [50, 75, 90], mode: 'calibrated', maxCandles: 0 }, // maxCandles<=0 = вся история (borrow simplex's bars=0 "все точки библиотеки") — вся история точнее, тикер справляется
  rangeForecastShowOnChart: false,      // "закреплено" — видно даже когда этот инструмент не активен (см. moving_averages.js/local_prefs.js:getToolShowOnChart)
  rangeForecastZones: null,             // последний POST /range-forecast/live ответ целиком (zones_by_step, calibrated, n_neighbors, origin_date, close_at_origin) или null
  rangeForecastCalibrationList: [],     // GET /range-forecast/settings — все откалиброванные (h_steps,p,theiler) для текущего (ticker,interval)

  // risk_corridor — риск-корридор High/Low (running max/min) + полоса Close
  // на h=1..H шагов вперёд, sma/ui/risk_corridor.js / sma/core/forecast/
  // risk_corridor.py. Портировано из prototype/forcaster/ui/app27-risk-
  // corridor.py, см. docs/plans/app27_risk_corridor_migration_plan.md.
  // ПРОЩЕ range_forecast — нет калибровки вообще (параметры глобальны,
  // валидированы temporal walk-forward, обобщаются на новые тикеры без
  // подгонки — эксп.06 фазы 19), поэтому нет settings-таблицы/списка/
  // баннера "нужна калибровка" — только coveragePct (единственный видимый
  // контрол вне аккордеона «Параметры», по прямому требованию пользователя)
  // + сам прогноз, живьём при каждом изменении origin/настроек (POST
  // /risk-corridor/live).
  riskCorridorSettings: {
    h: 20, pFit: 20, blendAlpha: 0.75, theta: 20.0, nSim: 15000,
    coveragePct: 95, seed: 42, maxCandles: 0, // 0 = вся история — точнее, тикер справляется
  },
  riskCorridorShowOnChart: false, // "закреплено" — видно даже когда этот инструмент не активен (см. moving_averages.js/local_prefs.js:getToolShowOnChart)
  riskCorridorResult: null,       // последний POST /risk-corridor/live ответ целиком (steps, n_neighbors, origin_date, close_at_origin, coverage_pct) или null

  // regime_mixture_potential — regime-conditioned pairlag lagged-ensemble +
  // GaussianMixture-сценарии потенциала, с премоткой по origin.
  // sma/ui/regime_mixture_potential.js / sma/core/forecast/
  // regime_mixture_potential.py. Портировано из prototype/forcaster/ui/
  // app33-regime-mixture-rewind.py, см. docs/plans/
  // app33_regime_mixture_potential_migration_plan.md. Асинхронный persisted
  // паттерн, как simplex_ensemble (не live, как range_forecast/risk_corridor)
  // — тяжёлый расчёт (N независимых origin, N_LOOKBACK-пул на каждый +
  // GMM-фит), сохраняется в forecasts.
  //
  // potentialParams — серверные (требуют новый «▶ Прогноз» при изменении,
  // прогреваются ЦЕЛИКОМ на каждый снапшот премотки — см. модуль core).
  // nForecasts — ОБЩЕЕ число прогнозов премотки (включая origin), мин=1.
  potentialParams: {
    theta: 5.0, warmup: 250, theilerWindow: 20, horizon: 30, bars: 0, // 0 = вся история — точнее, тикер справляется
    nLookback: 8, lookbackStep: 1, nSim: 10000, seed: 42, mixNResample: 1500,
    binHeightPct: 0.2, coveragePct: 95, nForecasts: 20, rewindStep: 1, // nForecasts=20 (было 10) — быстрый метод, справляется
  },
  // «Отображение» — чисто клиентские преобразования уже сохранённых данных
  // (components_by_h/raw_histogram для КАЖДОГО снапшота перемотки уже
  // прогреты на сервере), без похода на сервер при изменении:
  potentialHeatmapMode: 'scenarios',   // 'scenarios' | 'raw'
  potentialColorShift: 0.0,            // XY-пад — X, [0,1]
  potentialColorSteepness: 1.0,        // XY-пад — Y, [0.5,20]
  potentialRewindIdx: 0,               // индекс в result.snapshots (0 = origin, самый свежий)
  potentialShowBounds: false,          // живой, как simplexShowMean
  potentialCoveragePct: 95,             // живой — граница считается на клиенте из raw_histogram (chart.js:potentialCoverageBoundsFromHistogram), не из снапшота
};

export const INTERVAL_SECONDS = {
  '1m': 60, '10m': 600, '1h': 3600,
  '1d': 86400, '1w': 604800, '1mo': 2592000,
};

export const MONTH_RU = ['янв','фев','мар','апр','май','июн','июл','авг','сен','окт','ноя','дек'];
export const DAY_RU   = ['Вс','Пн','Вт','Ср','Чт','Пт','Сб'];
