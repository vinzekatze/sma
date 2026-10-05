import { S } from './state.js';
import { api, setBusy, setIdle, setStatus } from './api.js';
import { renderChart, initChartEvents, clearMainOrigin } from './chart.js';
import { getTool, selectTool, renderToolbarButtons, renderToolbar } from './tools.js';
import { hydrateLocalPrefs } from './local_prefs.js';
import { dialogConfirmClick, dialogCancelClick } from './dialog.js';
import { initSpinInputs } from './spin_input.js';
import { initTooltips } from './tooltip.js';
import { initRangeProgress } from './range_progress.js';
import { initCandleWindowing } from './candle_window.js';
import {
  toggleMainLogScale, toggleObjectsHidden,
  setPendingPriceLevel, fixPendingPriceLevel, removePriceLevel, clearPriceLevels,
  loadPriceLevelDefaults, setPriceLevelHoverPrice,
} from './cursor_tools.js';
import {
  applySpectrogramContrast, loadSpectrogramDefaults, onSpectrogramSettingsChange,
} from './analysis.js';
import {
  onTrendRulerSettingsChange, onTrendRulerWindowInput, addTrendRulerBand,
  loadTrendRulerDefaults, initTrendRulerForTicker, toggleTrendRulerShowChart,
} from './trend_ruler.js';
import {
  onVarianceOscWindowInput, setVarianceOscMode, pullWindowFromRuler,
  loadVarianceOscDefaults, initVarianceOscForTicker, onVarianceOscSlopeVarWindowInput,
} from './variance_oscillator.js';
import {
  addMovingAverage, onMaSettingsChange, loadMaDefaults, initMaForTicker, toggleMaShowChart,
} from './moving_averages.js';
import {
  addZigzagSeries, onZigzagToolShowChartChange, loadZigzagToolDefaults, initZigzagToolForTicker,
} from './zigzag_tool.js';
import { setVolumeOscMode, loadVolumeOscDefaults } from './volume_oscillator.js';
import {
  loadBandLambdaDefaults, loadDisplayPresets,
  submitForecast, onBandTChange, onBandModelParamsChange, toggleZigzagPin,
  applyDisplaySettings, applyBandWidth,
  resetZigzagPins,
} from './forecast.js';
import {
  openBandPoolModal, closeBandPoolModal,
} from './band_pool_modal.js';
import {
  loadSimplexDefaults, submitSimplexForecast, refreshSimplexDisplay, onSimplexBandPctChange,
} from './simplex_ensemble.js';
import {
  loadPotentialDefaults, submitPotentialForecast, onPotentialHeatmapModeChange,
  onPotentialColorXYChange, onPotentialShowBoundsChange, onPotentialRewindChange,
  onPotentialCoverageInput,
} from './regime_mixture_potential.js';
import {
  initRangeForecastForTicker, onRangeForecastSettingsChange, calibrateRangeForecast,
  toggleRangeForecastShowChart,
} from './range_forecast.js';
import {
  initRiskCorridorForTicker, onRiskCorridorSettingsChange, onRiskCorridorCoverageInput,
  toggleRiskCorridorShowChart,
} from './risk_corridor.js';
import {
  loadRiskCalcDefaults, loadRiskCalcTradingParams, onRiskCalcSettingsChange,
  onRiskCalcDirectionChange, onRiskCalcAnchorChange, setRiskCalcEntryToCurrent,
  toggleRiskCalcPickOnChart, toggleRiskCalcShowChart,
} from './risk_calculator.js';
import {
  refreshHistory, loadAndRenderForecast, deleteForecast, toggleForecastPin, toggleSelectedForecastPin,
  hydratePinnedForecasts, hideAllForecastResults,
} from './forecast_history.js';
import {
  refreshAll, toggleAccordion, onQuickFilterInput, onGlobalIntervalChange,
  refreshCurrentTicker, searchTickers,
} from './tickers.js';
import { initTaskManager, resumeAllTasks } from './tasks.js';
import { loadAppSettings, saveAppSettings, saveColorProfile } from './settings.js';

// ── expose globals for HTML onclick / onchange handlers ───────────────────────
window.loadCandles           = loadCandles;
window.switchTab             = switchTab;
window.renderChart           = renderChart;
window.applySpectrogramContrast    = applySpectrogramContrast;
window.onSpectrogramSettingsChange = onSpectrogramSettingsChange;
window.onTrendRulerSettingsChange      = onTrendRulerSettingsChange;
window.onTrendRulerWindowInput         = onTrendRulerWindowInput;
window.addTrendRulerBand               = addTrendRulerBand;
window.toggleTrendRulerShowChart       = toggleTrendRulerShowChart;
window.onVarianceOscWindowInput        = onVarianceOscWindowInput;
window.setVarianceOscMode              = setVarianceOscMode;
window.pullWindowFromRuler             = pullWindowFromRuler;
window.onVarianceOscSlopeVarWindowInput = onVarianceOscSlopeVarWindowInput;
window.setVolumeOscMode                = setVolumeOscMode;
window.addMovingAverage       = addMovingAverage;
window.onMaSettingsChange     = onMaSettingsChange;
window.toggleMaShowChart      = toggleMaShowChart;
window.addZigzagSeries        = addZigzagSeries;
window.onZigzagToolShowChartChange = onZigzagToolShowChartChange;
window.onRangeForecastSettingsChange = onRangeForecastSettingsChange;
window.calibrateRangeForecast        = calibrateRangeForecast;
window.toggleRangeForecastShowChart  = toggleRangeForecastShowChart;
window.onRiskCorridorSettingsChange  = onRiskCorridorSettingsChange;
window.onRiskCorridorCoverageInput   = onRiskCorridorCoverageInput;
window.toggleRiskCorridorShowChart   = toggleRiskCorridorShowChart;
window.onRiskCalcSettingsChange      = onRiskCalcSettingsChange;
window.onRiskCalcDirectionChange     = onRiskCalcDirectionChange;
window.onRiskCalcAnchorChange        = onRiskCalcAnchorChange;
window.setRiskCalcEntryToCurrent     = setRiskCalcEntryToCurrent;
window.toggleRiskCalcPickOnChart     = toggleRiskCalcPickOnChart;
window.toggleRiskCalcShowChart       = toggleRiskCalcShowChart;
// Wrapped (not the bare tools.js export) so a tool switch also redraws the
// chart — forecast marker visibility now depends on S.activeMainTool
// (chart.js:forecastMarkersInView, project feedback 2026-08-19), so without
// this the toolbar icon's active state changed but the chart itself stayed
// stale until something else happened to call renderChart (a chart click, or
// selecting a forecast from history) to trigger it incidentally. Can't put
// this inside tools.js itself — chart.js already imports TOOLS FROM tools.js,
// so tools.js importing renderChart back would be a cycle; app.js is the
// composition root that already has both, same as switchTab below.
// Also clears the temporary forecast selection — picking a different tool
// used to leave whatever was selected still showing its full origin
// crosshair, which read as "hanging" once it no longer had anything to do
// with the newly active tool (project feedback 2026-08-19: "чтобы оно не
// оставалось висеть"). Unconditional — even if the forecast is PINNED
// (project feedback 2026-08-19, follow-up: "пускай снимается, даже если
// запинен, а то остается горизонтальная линия, которая не нужна" — pin only
// promises the MARKER stays visible, see chart.js:buildForecastMarkerTraces;
// it was never meant to force the origin crosshair to persist across tool
// switches too, only buildVisibleOverlays drawing it for
// S.selectedForecastId regardless of tool made it look that way). Same
// unconditional clear applies to the empty-chart-click deselect below.
window.selectTool             = (type) => {
  S.selectedForecastId = null;
  selectTool(type);
  renderChart({ preserveRange: true });
};
// Курсор reset button: the active tool's own origin if it has one (trend
// ruler), otherwise the main origin — so every selected tool has something
// to reset here.
window.resetActiveToolOrigin = () => {
  const tool = getTool(S.activeMainTool);
  if (tool?.resetOrigin) tool.resetOrigin();
  else clearMainOrigin();
};
window.toggleMainLogScale     = toggleMainLogScale;
window.toggleObjectsHidden    = toggleObjectsHidden;
window.fixPendingPriceLevel   = fixPendingPriceLevel;
window.removePriceLevel       = removePriceLevel;
window.clearPriceLevels       = clearPriceLevels;
window.saveAppSettings        = saveAppSettings;
window.saveColorProfile       = saveColorProfile;
window.submitForecast        = submitForecast;
window.loadAndRenderForecast = loadAndRenderForecast;
window.deleteForecast        = deleteForecast;
window.toggleForecastPin     = toggleForecastPin;
window.toggleSelectedForecastPin = toggleSelectedForecastPin;
window.applyDisplaySettings  = applyDisplaySettings;
window.applyBandWidth        = applyBandWidth;
window.onBandTChange         = onBandTChange;
window.onBandModelParamsChange = onBandModelParamsChange;
window.toggleZigzagPin       = toggleZigzagPin;
window.openBandPoolModal     = openBandPoolModal;
window.closeBandPoolModal    = closeBandPoolModal;
window.submitSimplexForecast   = submitSimplexForecast;
window.refreshSimplexDisplay   = refreshSimplexDisplay;
window.onSimplexBandPctChange  = onSimplexBandPctChange;
window.submitPotentialForecast     = submitPotentialForecast;
window.onPotentialHeatmapModeChange = onPotentialHeatmapModeChange;
window.onPotentialColorXYChange    = onPotentialColorXYChange;
window.onPotentialShowBoundsChange = onPotentialShowBoundsChange;
window.onPotentialRewindChange     = onPotentialRewindChange;
window.onPotentialCoverageInput    = onPotentialCoverageInput;
window.searchTickers         = searchTickers;
window.toggleAccordion       = toggleAccordion;
window.refreshCurrentTicker  = refreshCurrentTicker;
window.resumeAllTasks        = resumeAllTasks;
window.dialogConfirmClick    = dialogConfirmClick;
window.dialogCancelClick     = dialogCancelClick;

// ── tabs ───────────────────────────────────────────────────────────────────────
// Two tabs: 'main' (Основной график — now also hosts the Прогноз toolbar
// row, see tools.js/index.html) and 'oscillator'. Matched by data-tab
// attribute, not array position (was fragile back when there were three).
function switchTab(name) {
  S.activeTab = name;
  // Scoped to #panel (right sidebar) so this never touches unrelated
  // .tab-btn/.tab-pane elements elsewhere in the page.
  document.querySelectorAll('#panel .tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === name);
  });
  document.querySelectorAll('#panel .tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === `tab-${name}`);
  });
  // renderChart always redraws whatever's currently pinned/selected (plus
  // the plain origin line, if any) — no per-tab branching needed any more.
  renderChart({ preserveRange: true });
}

// ── candles ────────────────────────────────────────────────────────────────────
// Тикер выбирается только в левой панели (быстрый переключатель / менеджер);
// интервал по умолчанию — текущее значение верхнего #interval-select.
async function loadCandles(ticker, dataSource, interval) {
  ticker     = ticker || S.ticker;
  dataSource = dataSource || S.dataSource || 'moex';
  interval   = interval || document.getElementById('interval-select').value;
  if (!ticker) { setStatus('Выберите тикер в менеджере слева', 'err'); return; }

  document.getElementById('interval-select').value = interval;

  setBusy(`Загрузка ${ticker} [${interval}]…`);
  document.getElementById('chart-loading-overlay')?.classList.remove('hidden');
  try {
    // limit — last N bars only (sma/ui/candle_window.js keeps loading older
    // chunks as the user pans back; project feedback: tickers with a long
    // intraday history load ~250k bars at once otherwise, which Firefox
    // renders far worse than Chrome).
    const rows = await api('GET', `/candles?ticker=${ticker}&data_source=${dataSource}&interval=${interval}&limit=${S.chartWindowBars}`);
    if (!rows.length) {
      S.ticker = ticker; S.dataSource = dataSource; S.interval = interval;
      S.instrumentId = null;
      setIdle(`Нет данных для ${ticker} [${interval}]. Скачайте в менеджере тикеров.`, false);
      return;
    }
    S.candles              = rows;
    S.candlesFullyLoadedBack = rows.length < S.chartWindowBars;
    S.ticker               = ticker;
    S.dataSource           = dataSource;
    S.interval              = interval;
    S.instrumentId          = rows[0].instrument_id;
    S.originTs              = null;
    S.spectrogramData       = null;
    S.trendRulerData        = null;
    S.trendRulerOriginTs    = null;
    S.varianceOscData       = null;
    S.maData                = {};
    S.zigzagToolData        = {};
    S.shapes                = [];
    S.historyForecasts      = [];
    S.renderedForecasts.clear();
    // pinnedForecastIds NOT cleared any more (project feedback 2026-08-25:
    // "отображение зафиксированных прогнозов... не сохраняется") — a pinned
    // id that doesn't exist in the NEW ticker's S.historyForecasts (set by
    // refreshHistory below) simply draws nothing
    // (chart.js:buildForecastMarkerTraces only renders pins present in the
    // current history), so leaving stale ids in place is harmless and lets
    // them "reappear" if the user switches back to the original ticker.
    S.selectedForecastId    = null;
    hideAllForecastResults(); // previous ticker's forecast detail must not linger visible — see that function's docstring
    S.zigzagPivots          = [];
    resetZigzagPins();       // pinned T's price/date cache IS per-instrument (unlike pinnedForecastIds above) — see forecast.js
    S.selectedTs            = null; // loadBandLambdaDefaults() below sets it back (saved or fallback) before anything reads it
    S.bandLambdaPool        = null;
    S.bandAnnotations       = [];
    S.activeMainTool        = 'free'; // back to the default cursor tool — none of the others (analyzer/forecaster origins, price levels) carry meaningfully across a ticker switch
    // Raw assignment above (not selectTool/activateTool) never fires
    // risk_calc's own onDeselected — same reason clearPriceLevels right
    // below has to run explicitly for price_level's pending state instead
    // of relying on that hook.
    S.riskCalcPickTarget    = null;
    document.getElementById('chart')?.classList.remove('picking-price');
    // S.activeOscillatorTool/S.subpanel are NOT reset to 'none' any more
    // (same 2026-08-25 feedback, "...и осциляторов [не сохраняется]") — the
    // still-selected oscillator (if any) gets its onSelected hook
    // re-triggered below, once the new ticker's data is otherwise ready, so
    // it refetches for the new instrument instead of either showing stale
    // data or dropping back to nothing. S.subpanel just mirrors whatever
    // S.activeOscillatorTool already is, same invariant tools.js maintains.
    S.subpanel = S.activeOscillatorTool;
    renderToolbar();           // reflects the reset above in the toolbar/panels immediately
    // persist:false — this is the "switching away" reset, not a user
    // "Очистить" action; the new ticker's own persisted levels are about to
    // be loaded right below (loadPriceLevelDefaults) and must not race
    // against a debounced empty-list save for it (see cursor_tools.js:
    // clearPriceLevels's own docstring for the full reasoning).
    clearPriceLevels({ persist: false }); // also refreshes #price-level-list's DOM

    renderChart();                 // establishes correct x+y range for 300-bar window
    await refreshHistory();        // per-model forecast history for the new instrument (both models fetched, see forecast_history.js)
    // refreshHistory() only just populated S.historyForecasts (empty at the
    // renderChart() call above) — pinned MARKERS (S.pinnedForecastIds, which
    // survives a ticker switch/page reload since 2026-08-25) depend on it
    // via chart.js:buildVisibleOverlays. The pinned forecast's actual
    // band/zone overlay needs a SEPARATE fetch (S.renderedForecasts, cleared
    // on every ticker switch/page load) — see hydratePinnedForecasts's own
    // docstring. Both awaited before this render so pinned forecasts show
    // fully (marker + band) on the very first paint after a reload/ticker
    // switch, not just after some unrelated interaction happens to trigger
    // a redraw (project feedback 2026-08-25: "закрепленные... прогнозы...
    // не отображаются после перезагрузки страницы / загрузки тикера").
    await hydratePinnedForecasts();
    renderChart({ preserveRange: true });
    await loadBandLambdaDefaults(); // T/m/theta spinner + pool summary + zigzag overlay (band_lambda)
    await loadSimplexDefaults();   // last-used simplex_ensemble params — both forecast tools are always live now, no dropdown gating which one loads
    await loadPotentialDefaults(); // last-used regime_mixture_potential params
    await loadSpectrogramDefaults();  // last-used spectrogram params for this ticker (Осциллятор tab)
    await loadTrendRulerDefaults();   // last-used trend_ruler params for this ticker (Основной график tab)
    await loadVarianceOscDefaults();  // last-used variance_osc params for this ticker (Осциллятор tab)
    await loadVolumeOscDefaults();    // last-used volume_osc mode for this ticker (Осциллятор tab)
    await loadMaDefaults();           // last-used moving_averages params (Основной график tab)
    await loadZigzagToolDefaults();   // last-used zigzag_tool params (Основной график tab)
    await loadPriceLevelDefaults();   // last-used fixed price levels for this ticker (project feedback 2026-08-20)
    initTrendRulerForTicker();        // live trend+bands preview
    initVarianceOscForTicker();       // no-op — variance_osc's OWN persisted state was per-tool, superseded by S.activeOscillatorTool itself now surviving the switch (see below)
    initMaForTicker();                // live MA lines
    initZigzagToolForTicker();        // re-fetch every configured zigzag series
    await initRangeForecastForTicker(); // live range forecast + refresh calibrated (H,p,theiler) list for the new instrument
    await initRiskCorridorForTicker();  // live risk-corridor — no calibration to refresh, params are global (см. план app27)
    await loadRiskCalcTradingParams();  // lot+price step first — loadRiskCalcDefaults' first render below already needs them
    await loadRiskCalcDefaults();       // last-planned trade for this ticker (entry/stop/tp/direction/anchor), analysis_settings
    // The still-active oscillator (if any survived the switch — see the
    // S.activeOscillatorTool comment above) needs its data refetched for
    // the NEW instrument: its own *Data field was nulled earlier in this
    // function, so without this it would just show an empty subpanel until
    // the user manually reselects it. Re-running onSelected is exactly what
    // clicking its toolbar icon would do (tools.js:selectTool/activateTool).
    if (S.activeOscillatorTool !== 'none') getTool(S.activeOscillatorTool)?.onSelected?.();
    setIdle(`${rows.length} свечей  •  ${rows.at(-1).begin.slice(0, 10)}`);
  } catch (e) {
    setIdle(e.message, false);
  } finally {
    document.getElementById('chart-loading-overlay')?.classList.add('hidden');
  }
}

// ── init ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // §1.1b of docs/plans/frontend_improvements_plan.md — restores
  // mainLogScale/objectsHidden/activeMainTool/activeOscillatorTool/
  // pinnedForecastIds from localStorage BEFORE the toolbar/chart render for
  // the first time, so a reload shows the same look instantly rather than
  // flashing defaults then jumping. See local_prefs.js's own docstring for
  // the one remaining caveat: activeMainTool still resets to 'free' the
  // moment loadCandles fires (below) — pinnedForecastIds/
  // activeOscillatorTool do NOT reset any more (2026-08-25 revision).
  hydrateLocalPrefs(S);
  initSpinInputs(); // custom cross-browser number-input spinners (project feedback 2026-08-25) — wraps every input[type=number] present now AND added later (MutationObserver), see spin_input.js
  initTooltips(); // floating .info-badge tooltip that isn't clipped by #panel's overflow:hidden — see tooltip.js
  initRangeProgress(); // WebKit-only "filled track" for input[type=range], matching Firefox's native ::-moz-range-progress — see range_progress.js
  initCandleWindowing(); // fetches an older chunk of candles when panning near the loaded window's left edge — see candle_window.js
  S.subpanel = S.activeOscillatorTool; // mirrors the hydrated value — tools.js normally keeps these in lockstep, but hydration writes S directly, bypassing that
  // toggleMainLogScale/toggleObjectsHidden (cursor_tools.js) are the only
  // places that normally sync these two buttons' 'active' class — harmless
  // there since they're always called via an actual click, but hydration
  // sets S directly without going through either function, so the buttons
  // need the same sync done explicitly here once, on load.
  document.getElementById('log-scale-btn')?.classList.toggle('active', S.mainLogScale);
  document.getElementById('show-objects-btn')?.classList.toggle('active', !S.objectsHidden);
  renderToolbarButtons(); // every tool module has self-registered by now (static imports)
  renderToolbar();        // reflects the hydrated activeMainTool/activeOscillatorTool immediately
  await refreshAll();
  initTaskManager();
  loadDisplayPresets();
  loadAppSettings();

  initChartEvents(
    // forecastId is the marker's customdata (chart.js resolves this via
    // _markerTraceIndices/point.customdata — exact, no date-matching/
    // cycling needed even when several forecasts stack on the same date).
    // Two ways a marker click means something (project feedback 2026-08-18,
    // round 6, updated round 7 once cursor tools joined the same
    // S.activeMainTool slot as analyzers/forecasters):
    // - the active tool IS that marker's own forecast model — only
    //   OWN-type markers are clickable while a specific model is armed
    //   (project feedback: "позволяют выбирать на графике только прогнозы
    //   своего типа"), looked up from S.historyForecasts rather than a
    //   second fetch;
    // - OR the 'object_select' tool is active — a general inspect mode that
    //   bypasses the "own type only" restriction.
    // Anywhere else, markers stay visible for context but are inert
    // (clicking one while configuring an unrelated analyzer, or with the
    // price-level tool active, was confusing). null means the click missed
    // every marker: clear the temporary selection — unconditionally, even if
    // pinned (project decision 2026-08-15 originally gated this on "не
    // запинен"; revised 2026-08-19 — pin only promises the MARKER stays
    // visible, see chart.js:buildForecastMarkerTraces/buildVisibleOverlays,
    // it was never meant to keep the full origin crosshair stuck on screen
    // once the user has clicked away from it: "остается горизонтальная
    // линия, которая не нужна"; same unconditional clear as
    // window.selectTool above, for the same reason). switchTool:false always
    // (round 8: "выбор объекта... отдельный режим" — a marker click must
    // never itself change S.activeMainTool; when ownType is true it's
    // already the right tool anyway, and object_select must stay
    // object_select).
    (forecastId, _clickDate) => {
      if (forecastId != null) {
        const f = S.historyForecasts.find(x => x.id === forecastId);
        const ownType = f && S.activeMainTool === f.model_type;
        const objectSelectActive = S.activeTab === 'main' && S.activeMainTool === 'object_select';
        if (!ownType && !objectSelectActive) return;
        loadAndRenderForecast(forecastId, { switchTool: false });
      } else if (S.selectedForecastId != null) {
        S.selectedForecastId = null;
        renderToolbar(); // moves any peeked "Отображение" section (tools.js:syncDisplayPeek) back home now that nothing's selected
        renderChart({ preserveRange: true });
      }
    },
    // Routed to whichever tool is currently active on the Основной график
    // (S.activeMainTool, see tools.js:selectTool) — cursor tools, analyzers
    // and forecast models all share this ONE exclusivity slot (project
    // feedback 2026-08-18, round 7: "Одновременно может быть выбран только
    // один инструмент... разные режимы курсора в сочетании с прогнозами
    // начинают вести себя непредсказуемо" — merging removed the second,
    // independent S.cursorTool axis that used to cause that). Each tool's
    // own onOriginClick decides what a click means: an analyzer sets its
    // own private origin; a forecast model calls the shared setOrigin(ts);
    // 'object_select' and 'price_level' register none, so this is
    // naturally a no-op while either is active (price_level never actually
    // reaches here anyway — chart.js's click dispatcher special-cases it
    // and returns before calling this callback at all, since it needs the
    // clicked PRICE, not a candle timestamp). No tool active -> no-op too.
    ts => {
      if (S.activeTab === 'main' && S.activeMainTool) {
        getTool(S.activeMainTool)?.onOriginClick?.(ts);
      }
    },
    // Price-level tool — see cursor_tools.js:setPendingPriceLevel. Handled
    // via native DOM listeners now (chart.js:_bindPriceLevelEventsIfNeeded),
    // not Plotly's click system — already gated on S.activeMainTool ===
    // 'price_level' there.
    price => setPendingPriceLevel(price),
    // Continuous hover preview (project feedback 2026-08-25: "горизонтальная
    // линия, следящая за курсором" replacing Plotly's bar-snapped
    // crosshair) — price is null on mouseleave.
    price => setPriceLevelHoverPrice(price)
  );

  document.getElementById('ticker-search-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') searchTickers();
  });
  document.getElementById('ticker-quick-filter').addEventListener('input', onQuickFilterInput);
  document.getElementById('interval-select').addEventListener('change', e => {
    onGlobalIntervalChange();
    if (S.ticker) loadCandles(S.ticker, S.dataSource, e.target.value);
  });
});
