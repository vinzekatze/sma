import { S } from './state.js';
import { api, setIdle, setStatus } from './api.js';
import { renderChart } from './chart.js';

// ── Настройки приложения / Цветовой профиль (левая панель) ─────────────────
// Two separate top-level accordion sections in the left panel, backed by the
// SAME app_settings singleton row (sma/core/db.py) but edited independently
// (project feedback 2026-08-25: "цветовой профиль... лучше отдельным
// окном/страницей" — clarified to "подвкладкой в настройках, либо отдельно
// от настроек" — landed on a separate top-level accordion, same pattern as
// "Менеджер тикеров"/"Менеджер задач", not nested inside "Настройки
// приложения" any more):
//   - "Настройки приложения": MOEX request concurrency (pool liquidity
//     ranking, see sma/core/forecast/pool_selection.py) + calibration
//     worker count (ProcessPoolExecutor, shared by simplex_ensemble and
//     range_forecast_calibration — see sma/api/task_manager.py).
//   - "Цветовой профиль": every tool-meaningful color in the app (see
//     COLOR_ROLES below — single source of truth for label/grouping/type;
//     keys MUST match sma/core/db.py:DEFAULT_COLOR_PROFILE exactly).
// Both forms POST the FULL /settings body (moex_pool_workers +
// calibration_workers + color_profile) regardless of which one the user
// actually edited — reading the OTHER section's current value straight from
// its own DOM inputs (still present even if that accordion is currently
// collapsed — toggleAccordion only toggles a CSS class, see tickers.js) —
// so saving colors never resets the performance numbers and vice versa;
// the backend's own color_profile=None "leave alone" convenience
// (sma/api/routes/settings.py) exists for OTHER callers, not needed here
// since both forms always have the full picture available client-side.
// Explicit "Сохранить" buttons, not reactive/debounced like §2.5's display
// settings — deliberate app-wide config changes, not live visual tweaks.

// Keep in sync with sma/core/db.py:DEFAULT_COLOR_PROFILE — used only as a
// fallback until the first GET /settings resolves (so the form never shows
// blank swatches), the server is the actual source of truth.
const DEFAULT_COLOR_PROFILE = {
  price_level: '#ff0000',
  next_origin_marker: '#58a6ff',
  trend_ruler_line: '#1f77b4',
  trend_ruler_origin: '#bc8cff',
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
  risk_calc_entry: '#58a6ff',
  risk_calc_stop: '#f85149',
  risk_calc_profit: '#3fb950',
  spectrogram_colorscale: 'Viridis',
  variance_slope_up: '#2ca02c',
  variance_slope_down: '#d62728',
  variance_var: '#9467bd',
  variance_slopevar: '#e8a33d',
  volume_up: '#3fb950',
  volume_down: '#f85149',
};

// Single source of truth for the color-profile FORM: one row per role,
// grouped into panel-sections (`group`) then, project feedback 2026-08-25
// ("не хватает разделения для инструментов, использующих несколько
// цветов"), sub-grouped by owning tool (`tool` — matches that tool's own
// toolbar label exactly, e.g. trend_ruler.js's registerTool `label:
// 'Линейка тренда'`, so the same name means the same thing in both places;
// 'Общее' for roles no single tool owns, e.g. the shared origin crosshair
// used by every analyzer/forecaster with an origin concept). `type` picks
// the input widget:
//   'color'   — one <input type=color>
//   'palette' — a row of <input type=color>, one per array entry (MA/
//               zigzag_tool's cycled series palettes)
//   'select'  — a <select> of Plotly colorscale NAMES (spectrogram's heatmap
//               isn't a single hue, so a color picker doesn't apply)
const COLOR_ROLES = [
  { group: 'Основной график', tool: 'Общее', key: 'next_origin_marker', label: 'Точка отсчёта нового прогноза', type: 'color' },
  { group: 'Основной график', tool: 'Общее', key: 'forecast_marker_selected', label: 'Маркер выбранного прогноза', type: 'color' },
  { group: 'Основной график', tool: 'Ценовой уровень', key: 'price_level', label: 'Ценовой уровень', type: 'color' },
  { group: 'Основной график', tool: 'Линейка тренда', key: 'trend_ruler_line', label: 'Линия тренд-линейки', type: 'color' },
  { group: 'Основной график', tool: 'Линейка тренда', key: 'trend_ruler_origin', label: 'Точка отсчёта тренд-линейки', type: 'color' },
  { group: 'Основной график', tool: 'Скользящие средние', key: 'ma_palette', label: 'Палитра MA', type: 'palette' },
  { group: 'Основной график', tool: 'Zig-Zag', key: 'zigzag_tool_palette', label: 'Палитра Zig-Zag', type: 'palette' },
  { group: 'Основной график', tool: 'band_lambda', key: 'forecast_zigzag', label: 'Зигзаг прогноза', type: 'color' },
  { group: 'Основной график', tool: 'band_lambda', key: 'band_zone_up', label: 'Зона вероятности: рост', type: 'color' },
  { group: 'Основной график', tool: 'band_lambda', key: 'band_zone_down', label: 'Зона вероятности: падение', type: 'color' },
  { group: 'Основной график', tool: 'band_lambda', key: 'forecast_marker_pinned_band_lambda', label: 'Маркер закреплённого прогноза', type: 'color' },
  { group: 'Основной график', tool: 'simplex_ensemble', key: 'forecast_marker_pinned_simplex_ensemble', label: 'Маркер закреплённого прогноза', type: 'color' },
  { group: 'Основной график', tool: 'simplex_ensemble', key: 'simplex_origin_lines', label: 'Линии происхождения', type: 'color' },
  { group: 'Основной график', tool: 'simplex_ensemble', key: 'simplex_mean_band', label: 'Средняя полоса', type: 'color' },
  { group: 'Основной график', tool: 'Потенциал', key: 'forecast_marker_pinned_regime_mixture_potential', label: 'Маркер закреплённого прогноза', type: 'color' },
  { group: 'Основной график', tool: 'Прогноз диапазона', key: 'range_forecast_line', label: 'Линии шагов', type: 'color' },
  { group: 'Основной график', tool: 'Риск-корридор', key: 'risk_corridor_close', label: 'Полоса Close', type: 'color' },
  { group: 'Основной график', tool: 'Риск-корридор', key: 'risk_corridor_high', label: 'Риск-огибающая High', type: 'color' },
  { group: 'Основной график', tool: 'Риск-корридор', key: 'risk_corridor_low', label: 'Риск-огибающая Low', type: 'color' },
  { group: 'Основной график', tool: 'Калькулятор риска', key: 'risk_calc_entry', label: 'Линия входа', type: 'color' },
  { group: 'Основной график', tool: 'Калькулятор риска', key: 'risk_calc_stop', label: 'Линия стопа', type: 'color' },
  { group: 'Основной график', tool: 'Калькулятор риска', key: 'risk_calc_profit', label: 'Линия тейк-профита', type: 'color' },
  { group: 'Основной график', tool: 'Свечи', key: 'candle_up', label: 'Свеча: рост', type: 'color' },
  { group: 'Основной график', tool: 'Свечи', key: 'candle_down', label: 'Свеча: падение', type: 'color' },
  {
    group: 'Осциллятор', tool: 'Спектрограмма Δratio', key: 'spectrogram_colorscale', label: 'Цветовая схема', type: 'select',
    options: ['Viridis', 'Plasma', 'Turbo', 'Cividis', 'Inferno', 'Magma', 'Blues', 'Greens', 'YlOrRd', 'Greys'],
  },
  { group: 'Осциллятор', tool: 'Осциллятор тренда', key: 'variance_slope_up', label: 'Направление тренда — рост', type: 'color' },
  { group: 'Осциллятор', tool: 'Осциллятор тренда', key: 'variance_slope_down', label: 'Направление тренда — падение', type: 'color' },
  { group: 'Осциллятор', tool: 'Осциллятор тренда', key: 'variance_var', label: 'Дисперсия внутри окна тренда', type: 'color' },
  { group: 'Осциллятор', tool: 'Осциллятор тренда', key: 'variance_slopevar', label: 'Дисперсия направления тренда', type: 'color' },
  { group: 'Осциллятор', tool: 'Объём', key: 'volume_up', label: 'Рост', type: 'color' },
  { group: 'Осциллятор', tool: 'Объём', key: 'volume_down', label: 'Падение', type: 'color' },
];

function roleRowHtml(role, value) {
  if (role.type === 'palette') {
    const swatches = (value || []).map((c, i) => `
      <input type="color" value="${c}" data-cp-palette="${role.key}" data-cp-idx="${i}" title="Цвет №${i + 1}">
    `).join('');
    return `
      <div class="field-row"><label>${role.label}</label></div>
      <div class="cp-palette-row">${swatches}</div>
    `;
  }
  if (role.type === 'select') {
    const opts = role.options.map(o => `<option value="${o}" ${o === value ? 'selected' : ''}>${o}</option>`).join('');
    return `
      <div class="field-row">
        <label>${role.label}</label>
        <select data-cp-select="${role.key}" style="flex:1;width:auto">${opts}</select>
      </div>
    `;
  }
  return `
    <div class="field-row">
      <label>${role.label}</label>
      <input type="color" value="${value}" data-cp-color="${role.key}">
    </div>
  `;
}

function renderColorProfileForm(profile) {
  const container = document.getElementById('color-profile-form');
  if (!container) return;
  // group ("Основной график"/"Осциллятор", own panel-section) -> tool (the
  // owning tool's own toolbar label, e.g. "Линейка тренда" — a lighter
  // sub-heading, not a nested section) -> roles. Project feedback
  // 2026-08-25: "не хватает разделения для инструментов, использующих
  // несколько цветов" — 19 flat rows under one "Основной график" heading
  // (5 of them just trend_ruler's) made it hard to tell which color
  // belonged to which tool at a glance.
  const groups = new Map();
  for (const role of COLOR_ROLES) {
    if (!groups.has(role.group)) groups.set(role.group, new Map());
    const tools = groups.get(role.group);
    if (!tools.has(role.tool)) tools.set(role.tool, []);
    tools.get(role.tool).push(role);
  }
  container.innerHTML = [...groups.entries()].map(([group, tools]) => `
    <section class="panel-section">
      <h4>${group}</h4>
      ${[...tools.entries()].map(([tool, roles]) => `
        <div class="cp-tool-group">
          <div class="cp-tool-label">${tool}</div>
          ${roles.map(r => roleRowHtml(r, profile[r.key])).join('')}
        </div>
      `).join('')}
    </section>
  `).join('');
}

function readColorProfileForm() {
  const profile = {};
  for (const role of COLOR_ROLES) {
    if (role.type === 'palette') {
      profile[role.key] = [...document.querySelectorAll(`[data-cp-palette="${role.key}"]`)].map(el => el.value);
    } else if (role.type === 'select') {
      profile[role.key] = document.querySelector(`[data-cp-select="${role.key}"]`)?.value ?? DEFAULT_COLOR_PROFILE[role.key];
    } else {
      profile[role.key] = document.querySelector(`[data-cp-color="${role.key}"]`)?.value ?? DEFAULT_COLOR_PROFILE[role.key];
    }
  }
  return profile;
}

// Merges GET /settings' tool_display (if any) over S.toolDisplayDefaults'
// hardcoded fallback, ONE LEVEL DEEP per tool — same forward-compat
// reasoning as DEFAULT_COLOR_PROFILE's flat merge, just nested because each
// tool owns its own sub-object (see sma/core/db.py:_parse_tool_display,
// the server-side twin of this).
function mergeToolDisplay(stored) {
  const merged = {};
  for (const [tool, defaults] of Object.entries(S.toolDisplayDefaults)) {
    merged[tool] = { ...defaults, ...(stored?.[tool] || {}) };
  }
  return merged;
}

export async function loadAppSettings() {
  try {
    const s = await api('GET', '/settings');
    document.getElementById('settings-moex-workers').value = s.moex_pool_workers;
    document.getElementById('settings-calibration-workers').value = s.calibration_workers;
    document.getElementById('settings-chart-window-bars').value = s.chart_window_bars;
    S.chartWindowBars = s.chart_window_bars;
    S.toolDisplayDefaults = mergeToolDisplay(s.tool_display);

    const profile = { ...DEFAULT_COLOR_PROFILE, ...(s.color_profile || {}) };
    S.colorProfile = profile;
    renderColorProfileForm(profile);
  } catch (e) {
    // Non-fatal for the color profile — fall back to defaults so the rest of
    // the app (chart colors) still works even if /settings is unreachable.
    S.colorProfile = { ...DEFAULT_COLOR_PROFILE };
    renderColorProfileForm(S.colorProfile);
    setStatus(e.message, 'err');
  }
}

export async function saveAppSettings() {
  const moexWorkers = +document.getElementById('settings-moex-workers').value;
  const calibWorkers = +document.getElementById('settings-calibration-workers').value;
  const chartWindowBars = +document.getElementById('settings-chart-window-bars').value;
  const btn = document.getElementById('settings-save-btn');
  btn.disabled = true;
  try {
    await api('POST', '/settings', {
      moex_pool_workers: moexWorkers,
      calibration_workers: calibWorkers,
      chart_window_bars: chartWindowBars,
      color_profile: S.colorProfile, // current in-memory profile — the color form lives in its OWN accordion, untouched by this save
      tool_display: S.toolDisplayDefaults, // ditto — owned by each tool's own panel, not this form
    });
    S.chartWindowBars = chartWindowBars; // takes effect from the next ticker/interval load or pan-back fetch, not retroactively on already-loaded candles
    setIdle('Настройки сохранены');
  } catch (e) {
    setIdle(e.message, false);
  } finally {
    btn.disabled = false;
  }
}

export async function saveColorProfile() {
  const colorProfile = readColorProfileForm();
  const btn = document.getElementById('color-profile-save-btn');
  btn.disabled = true;
  try {
    const saved = await api('POST', '/settings', {
      // Read straight from the Настройки-приложения form's own DOM inputs
      // (present regardless of that accordion's open/collapsed state) —
      // NOT omitted, since AppSettingsIn's moex_pool_workers/
      // calibration_workers/chart_window_bars have no "leave alone"
      // convenience the way color_profile does; omitting them would
      // silently reset all three to their Pydantic defaults.
      moex_pool_workers: +document.getElementById('settings-moex-workers').value,
      calibration_workers: +document.getElementById('settings-calibration-workers').value,
      chart_window_bars: +document.getElementById('settings-chart-window-bars').value,
      color_profile: colorProfile,
      tool_display: S.toolDisplayDefaults,
    });
    S.colorProfile = { ...DEFAULT_COLOR_PROFILE, ...(saved.color_profile || colorProfile) };
    renderChart({ preserveRange: true }); // colors changed — redraw everything that reads S.colorProfile
    setIdle('Цветовой профиль сохранён');
  } catch (e) {
    setIdle(e.message, false);
  } finally {
    btn.disabled = false;
  }
}

// Called by trend_ruler.js/simplex_ensemble.js/regime_mixture_potential.js
// whenever one of THEIR global display toggles changes (project request
// 2026-10-04) — same "resend the full /settings body" requirement as
// saveColorProfile above (moex_pool_workers/calibration_workers/
// chart_window_bars have no leave-alone convenience), but silent/no-status-
// message and debounced HERE (shared single timer) so a tool doesn't need
// its own — e.g. dragging trend_ruler's opacity slider fires this on every
// tick without a request-per-tick.
let _toolDisplaySaveTimer = null;

export function saveToolDisplayDefaults() {
  clearTimeout(_toolDisplaySaveTimer);
  _toolDisplaySaveTimer = setTimeout(() => {
    api('POST', '/settings', {
      moex_pool_workers: +document.getElementById('settings-moex-workers').value,
      calibration_workers: +document.getElementById('settings-calibration-workers').value,
      chart_window_bars: +document.getElementById('settings-chart-window-bars').value,
      color_profile: S.colorProfile,
      tool_display: S.toolDisplayDefaults,
    }).catch(() => {});
  }, 500);
}
