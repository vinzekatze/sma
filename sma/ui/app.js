import { S } from './state.js';
import { api, setBusy, setIdle, setStatus } from './api.js';
import { renderChart, initChartEvents } from './chart.js';
import { getTool, selectTool, resetActiveToolOrigin, renderToolbarButtons, renderToolbar } from './tools.js';
import {
  toggleMainLogScale, toggleObjectsHidden,
  setPendingPriceLevel, fixPendingPriceLevel, removePriceLevel, clearPriceLevels,
} from './cursor_tools.js';
import {
  calculateSpectrogram, applySpectrogramContrast, toggleSpectrogramDisplay, loadSpectrogramDefaults,
} from './analysis.js';
import {
  onTrendRulerSettingsChange, onTrendRulerWindowInput, addTrendRulerBand,
  loadTrendRulerDefaults, initTrendRulerForTicker,
} from './trend_ruler.js';
import {
  onVarianceOscWindowInput, setVarianceOscMode, toggleVarianceOscillator, pullWindowFromRuler,
  loadVarianceOscDefaults, initVarianceOscForTicker,
} from './variance_oscillator.js';
import {
  addMovingAverage, onMaSettingsChange, loadMaDefaults, initMaForTicker,
} from './moving_averages.js';
import {
  addZigzagSeries, onZigzagToolShowChartChange, loadZigzagToolDefaults, initZigzagToolForTicker,
} from './zigzag_tool.js';
import {
  toggleVolumeDisplay, loadVolumeDefaults, initVolumeForTicker,
} from './volume_oscillator.js';
import {
  loadForecastSettings, loadDisplayPresets,
  submitForecast,
  toggleZigzagDisplay, applyDisplaySettings, applyBandWidth,
  addCalibTargetRow, runCalibration, runPretest, runPretestLevel2,
} from './forecast.js';
import {
  loadSimplexDefaults, submitSimplexForecast, refreshSimplexDisplay, onSimplexBandPctChange,
} from './simplex_ensemble.js';
import {
  refreshHistory, loadAndRenderForecast, deleteForecast, toggleForecastPin, toggleSelectedForecastPin,
} from './forecast_history.js';
import {
  refreshAll, toggleAccordion, onQuickFilterInput, onGlobalIntervalChange,
  refreshCurrentTicker, searchTickers,
} from './tickers.js';
import { initTaskManager, resumeAllTasks } from './tasks.js';
import { loadAppSettings, saveAppSettings } from './settings.js';

// ── expose globals for HTML onclick / onchange handlers ───────────────────────
window.loadCandles           = loadCandles;
window.switchTab             = switchTab;
window.renderChart           = renderChart;
window.calculateSpectrogram        = calculateSpectrogram;
window.applySpectrogramContrast    = applySpectrogramContrast;
window.toggleSpectrogramDisplay    = toggleSpectrogramDisplay;
window.onTrendRulerSettingsChange      = onTrendRulerSettingsChange;
window.onTrendRulerWindowInput         = onTrendRulerWindowInput;
window.addTrendRulerBand               = addTrendRulerBand;
window.onVarianceOscWindowInput        = onVarianceOscWindowInput;
window.setVarianceOscMode              = setVarianceOscMode;
window.toggleVarianceOscillator        = toggleVarianceOscillator;
window.pullWindowFromRuler             = pullWindowFromRuler;
window.addMovingAverage       = addMovingAverage;
window.onMaSettingsChange     = onMaSettingsChange;
window.addZigzagSeries        = addZigzagSeries;
window.onZigzagToolShowChartChange = onZigzagToolShowChartChange;
window.toggleVolumeDisplay    = toggleVolumeDisplay;
window.selectTool             = selectTool;
window.resetActiveToolOrigin           = resetActiveToolOrigin;
window.toggleMainLogScale     = toggleMainLogScale;
window.toggleObjectsHidden    = toggleObjectsHidden;
window.fixPendingPriceLevel   = fixPendingPriceLevel;
window.removePriceLevel       = removePriceLevel;
window.clearPriceLevels       = clearPriceLevels;
window.saveAppSettings        = saveAppSettings;
window.submitForecast        = submitForecast;
window.loadAndRenderForecast = loadAndRenderForecast;
window.deleteForecast        = deleteForecast;
window.toggleForecastPin     = toggleForecastPin;
window.toggleSelectedForecastPin = toggleSelectedForecastPin;
window.toggleZigzagDisplay   = toggleZigzagDisplay;
window.applyDisplaySettings  = applyDisplaySettings;
window.applyBandWidth        = applyBandWidth;
window.addCalibTargetRow     = addCalibTargetRow;
window.runCalibration        = runCalibration;
window.runPretest            = runPretest;
window.runPretestLevel2      = runPretestLevel2;
window.submitSimplexForecast   = submitSimplexForecast;
window.refreshSimplexDisplay   = refreshSimplexDisplay;
window.onSimplexBandPctChange  = onSimplexBandPctChange;
window.searchTickers         = searchTickers;
window.toggleAccordion       = toggleAccordion;
window.refreshCurrentTicker  = refreshCurrentTicker;
window.resumeAllTasks        = resumeAllTasks;

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
  try {
    const rows = await api('GET', `/candles?ticker=${ticker}&data_source=${dataSource}&interval=${interval}`);
    if (!rows.length) {
      S.ticker = ticker; S.dataSource = dataSource; S.interval = interval;
      S.instrumentId = null;
      setIdle(`Нет данных для ${ticker} [${interval}]. Скачайте в менеджере тикеров.`, false);
      return;
    }
    S.candles              = rows;
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
    S.subpanel               = 'none';
    S.shapes                = [];
    S.historyForecasts      = [];
    S.renderedForecasts.clear();
    S.pinnedForecastIds.clear();
    S.selectedForecastId    = null;
    S.zigzagPivots          = [];
    S.forecastSettingsList  = [];
    S.calibratedTs          = [];
    S.selectedTs            = null;
    S.activeSettingsId      = null;
    S.bandAnnotations       = [];
    S.activeMainTool        = 'free'; // back to the default cursor tool — none of the others (analyzer/forecaster origins, price levels) carry meaningfully across a ticker switch
    renderToolbar();           // reflects the reset above in the toolbar/panels immediately
    clearPriceLevels();       // prices are ticker-specific, meaningless carried over — also refreshes #price-level-list's DOM

    renderChart();                 // establishes correct x+y range for 300-bar window
    await refreshHistory();        // per-model forecast history for the new instrument (both models fetched, see forecast_history.js)
    await loadForecastSettings();  // T-selector + zigzag overlay (band_lambda)
    await loadSimplexDefaults();   // last-used simplex_ensemble params — both forecast tools are always live now, no dropdown gating which one loads
    await loadSpectrogramDefaults();  // last-used spectrogram params for this ticker (Осциллятор tab)
    await loadTrendRulerDefaults();   // last-used trend_ruler params for this ticker (Основной график tab)
    await loadVarianceOscDefaults();  // last-used variance_osc params for this ticker (Осциллятор tab)
    await loadMaDefaults();           // last-used moving_averages params (Основной график tab)
    await loadZigzagToolDefaults();   // last-used zigzag_tool params (Основной график tab)
    await loadVolumeDefaults();       // last-used volume params (Осциллятор tab)
    initTrendRulerForTicker();        // live trend+bands preview
    initVarianceOscForTicker();       // restore oscillator if it was on
    initMaForTicker();                // live MA lines
    initZigzagToolForTicker();        // re-fetch every configured zigzag series
    initVolumeForTicker();            // restore oscillator if it was on
    setIdle(`${rows.length} свечей  •  ${rows.at(-1).begin.slice(0, 10)}`);
  } catch (e) {
    setIdle(e.message, false);
  }
}

// ── init ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  renderToolbarButtons(); // every tool module has self-registered by now (static imports)
  await refreshAll();
  initTaskManager();
  loadDisplayPresets();
  loadAppSettings();
  addCalibTargetRow(); // one empty T row so «Калибровка» isn't blank on first open

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
    // every marker: clear the temporary selection if it isn't pinned
    // (project decision 2026-08-15 — "снять временный превью, если он не
    // запинен"). switchTool:false always (round 8: "выбор объекта...
    // отдельный режим" — a marker click must never itself change
    // S.activeMainTool; when ownType is true it's already the right tool
    // anyway, and object_select must stay object_select).
    (forecastId, _clickDate) => {
      if (forecastId != null) {
        const f = S.historyForecasts.find(x => x.id === forecastId);
        const ownType = f && S.activeMainTool === f.model_type;
        const objectSelectActive = S.activeTab === 'main' && S.activeMainTool === 'object_select';
        if (!ownType && !objectSelectActive) return;
        loadAndRenderForecast(forecastId, { switchTool: false });
      } else if (S.selectedForecastId != null && !S.pinnedForecastIds.has(S.selectedForecastId)) {
        S.selectedForecastId = null;
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
    // Price-level tool — see cursor_tools.js:setPendingPriceLevel. chart.js
    // only calls this when S.activeMainTool === 'price_level', already
    // short-circuiting before the two callbacks above.
    price => setPendingPriceLevel(price)
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
