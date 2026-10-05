// risk_calculator.js — «Калькулятор риска»: размер позиции от % депозита +
// соотношение риск:прибыль (R:R), с корректным учётом стоимости заёма для
// шорта (брокер берёт плату за заём акций, ~13%/год). Аналитический
// инструмент (project request 2026-10-04) — ничего не прогнозирует, только
// измеряет/визуализирует параметры СДЕЛКИ, которую пользователь сам
// планирует.
//
// Вся математика — ЧИСТО клиентская (как volume_oscillator.js's netDelta
// или variance_oscillator.js's rollingVarianceOfSlope): простая арифметика
// без истории цен, никакого backend-эндпоинта для расчёта не нужно.
// Backend отвечает только за:
//   - лот + шаг цены (GET /instruments/{id}/trading-params — тянется с
//     MOEX ISS, кэшируется на instruments.lot_size/price_step после
//     первого запроса);
//   - персистентность последней настроенной сделки (GET/POST
//     /series/analysis-settings, analyzer_type='risk_calc' — тот же
//     механизм, что variance_osc/volume_osc/trend_ruler/price_level).
//
// Для шорта НЕТ "наивного" варианта расчёта (проектное решение
// 2026-10-04: наивные числа только путают, показываем сразу правильные) —
// stopDistance корректируется стоимостью заёма ДО вычисления размера
// позиции, так что итоговый риск (цена + заём) держится у цели. Срок
// удержания по умолчанию 30 дней.
//
// anchor='stop'|'profit' — R:R работает как "замочек" между стопом и
// тейком в ОБЕ стороны: сделку можно собирать и от стопа (стоп — вход,
// тейк derived), и от тейка (наоборот). Для short R:R-замочек САМ учитывает
// стоимость заёма (project request 2026-10-04, п.5) — resolveStopAndTp
// решает tp/stop так, чтобы ФАКТИЧЕСКИЙ (после вычета/с учётом заёма) R:R
// точно совпадал с целевым rr, а не только ценовое расстояние (голая
// rr·stopDistance систематически давала ФАКТИЧЕСКИЙ R:R хуже целевого —
// заём откусывает и от risk, и от reward одновременно).
//
// Взаимодействие с графиком — тот же HTML/CSS-оверлей поверх Plotly, что и
// у инструмента «Ценовой уровень» (cursor_tools.js): полноширинные линии +
// подписи цены сбоку, живой курсор-preview при "Установить на графике",
// НЕ Plotly-трейсы/shapes (см. renderRiskCalcOverlay). Три поля (вход,
// стоп, тейк) можно задать кликом по графику — выбор стопа/тейка на
// графике автоматически переключает якорь R:R на выбранное поле.
//
// Линии — ТОЛЬКО линии, без заливки зон (не смешивать с зонами
// band_lambda), сплошные (не точечные, как у ценового уровня, который
// рисует пунктир для зафиксированных уровней) — цвет (вход/стоп/тейк)
// визуально отличает их от произвольного ценового уровня на том же
// графике, подпись — на том же месте (справа), что у ценового уровня.
//
// Депозит/риск%/ставка заёма — GLOBAL (project request 2026-10-04: это
// правило риск-менеджмента, не свойство тикера) — persist через
// settings.js:saveToolDisplayDefaults, см. S.toolDisplayDefaults.risk_calc.
// Сама сделка (направление/вход/стоп/тейк/якорь/срок удержания) остаётся
// per-ticker (analysis_settings), как и раньше.

import { S } from './state.js';
import { api, setStatus } from './api.js';
import { renderChart, priceToPixelY, priceAtClientY, registerOverlayRefresher } from './chart.js';
import { registerTool, isToolObjectVisible } from './tools.js';
import { getToolShowOnChart, saveToolShowOnChart } from './local_prefs.js';
import { hexToRgba } from './color_utils.js';
import { saveToolDisplayDefaults } from './settings.js';

// Настоящий дефолт (не S.riskCalcSettings — тот мог унести значения
// ПРЕДЫДУЩЕГО тикера, см. loadRiskCalcDefaults) — для "чистого" тикера без
// сохранённой сделки.
const DEFAULT_SETTINGS = {
  direction: 'short', anchor: 'stop',
  deposit: 300000, riskPct: 2,
  entry: null, stop: null, tp: null, rr: 3,
  borrowPct: 13, holdDays: 30,
};

const PICK_TARGET_LABEL = { entry: 'входа', stop: 'стопа', tp: 'тейк-профита' };

function redraw() {
  renderChart({ preserveRange: true });
}

// ── чистая математика ────────────────────────────────────────────────────

// Разворачивает цену (стоп ИЛИ тейк) на правильную сторону от входа для
// заданного направления, СОХРАНЯЯ дистанцию — при смене направления (стоп
// автоматически переворачивается) и симметрично для тейка, если якорь —
// тейк.
function flipAcrossEntry(entry, price, direction, role) {
  const dist = Math.abs(entry - price);
  const wantsAbove = role === 'stop' ? direction === 'short' : direction === 'long';
  return wantsAbove ? entry + dist : entry - dist;
}

// anchor='stop' — стоп задан пользователем, тейк ВСЕГДА derived.
// anchor='profit' — симметрично наоборот. Для LONG (f=0) — голая
// rr·stopDistance, как раньше. Для SHORT — rewardDistance включает
// поправку на стоимость заёма (borrowCost/size = entry·f, НЕ зависит от
// самого stopDistance — см. вывод формулы в памяти сессии), так что
// ФАКТИЧЕСКИЙ net R:R (computeTrade.rr) точно равен целевому s.rr, а не
// только голое ценовое расстояние:
//   rewardDistance = rr·stopDistance + (rr+1)·entry·f          (anchor=stop)
//   stopDistance = (rewardDistance - entry·f)/rr - entry·f     (anchor=profit, обратная формула)
function resolveStopAndTp(s) {
  const dir = s.direction === 'long' ? 1 : -1;
  const f = s.direction === 'short' ? (s.borrowPct / 100 / 365) * s.holdDays : 0;
  if (s.anchor === 'stop') {
    const stopDistance = Math.abs(s.entry - s.stop);
    const rewardDistance = s.rr * stopDistance + (s.rr + 1) * s.entry * f;
    return { stop: s.stop, tp: s.entry + dir * rewardDistance, stopDistance, rewardDistance };
  }
  const rewardDistance = Math.abs(s.entry - s.tp);
  const stopDistance = (rewardDistance - s.entry * f) / s.rr - s.entry * f;
  return { stop: s.entry - dir * stopDistance, tp: s.tp, stopDistance, rewardDistance };
}

// raw/lot → целое число лотов, минимум ОДИН лот (реальная позиция не может
// быть меньше) — если риска не хватает даже на 1 лот, фактический риск
// будет выше цели, это отражается в riskPctActual результата.
function sizeFromRisk(riskAmount, effDistance, lot) {
  if (effDistance <= 0) return 0;
  const raw = riskAmount / effDistance;
  const lots = Math.floor(raw / lot);
  return lots > 0 ? lots * lot : lot;
}

// Единый результат для long/short — borrowCost=0 для long.
function computeTrade(s, lot) {
  const { stop, tp, stopDistance, rewardDistance } = resolveStopAndTp(s);
  const riskAmount = s.deposit * s.riskPct / 100;

  if (s.direction === 'long') {
    const size = sizeFromRisk(riskAmount, stopDistance, lot);
    const notional = size * s.entry;
    const risk = size * stopDistance;
    const reward = size * rewardDistance;
    return {
      stop, tp, size, lots: size / lot, notional, borrowCost: 0, risk, reward,
      rr: risk > 0 ? reward / risk : 0,
      riskPctActual: s.deposit > 0 ? risk / s.deposit : 0,
    };
  }

  const dailyRate = s.borrowPct / 100 / 365;
  const effDistance = stopDistance + s.entry * dailyRate * s.holdDays;
  const size = sizeFromRisk(riskAmount, effDistance, lot);
  const notional = size * s.entry;
  const borrowCost = notional * dailyRate * s.holdDays;
  const risk = size * stopDistance + borrowCost;
  const grossReward = size * rewardDistance;
  const reward = grossReward - borrowCost;
  return {
    stop, tp, size, lots: size / lot, notional, borrowCost, risk, reward,
    rr: risk > 0 ? reward / risk : 0,
    riskPctActual: s.deposit > 0 ? risk / s.deposit : 0,
  };
}

function roundToStep(price, step) {
  if (!step || step <= 0 || price == null) return price;
  return Math.round(price / step) * step;
}

function validate(s) {
  if (!s.deposit || s.deposit <= 0) throw new Error('Укажите депозит больше нуля.');
  if (!s.riskPct || s.riskPct <= 0) throw new Error('Риск на сделку должен быть больше нуля.');
  if (!s.entry || s.entry <= 0) throw new Error('Укажите цену входа больше нуля.');
  if (!s.rr || s.rr <= 0) throw new Error('Risk:Reward должно быть больше нуля.');
  if (s.anchor === 'stop') {
    if (!s.stop || s.stop <= 0) throw new Error('Укажите цену стопа больше нуля.');
    if (s.direction === 'long' && s.stop >= s.entry) throw new Error('Для Long стоп должен быть ниже цены входа.');
    if (s.direction === 'short' && s.stop <= s.entry) throw new Error('Для Short стоп должен быть выше цены входа.');
  } else {
    if (!s.tp || s.tp <= 0) throw new Error('Укажите цену тейка больше нуля.');
    if (s.direction === 'long' && s.tp <= s.entry) throw new Error('Для Long тейк должен быть выше цены входа.');
    if (s.direction === 'short' && s.tp >= s.entry) throw new Error('Для Short тейк должен быть ниже цены входа.');
  }
  if (s.direction === 'short' && (s.borrowPct < 0 || s.holdDays < 0)) {
    throw new Error('Ставка заёма и срок удержания не могут быть отрицательными.');
  }
}

// ── форма <-> S.riskCalcSettings ─────────────────────────────────────────

function readFormIntoSettings() {
  const s = S.riskCalcSettings;
  s.deposit = +document.getElementById('rc-deposit').value;
  s.riskPct = +document.getElementById('rc-risk-pct').value;
  s.entry = +document.getElementById('rc-entry').value || null;
  s.anchor = document.getElementById('rc-anchor').value;
  // Какое из двух (stop/tp) реально используется решает resolveStopAndTp по
  // s.anchor — читаем оба без вреда, неактивное просто игнорируется этим
  // вызовом (но оба поля всё равно синхронизируются обратно в DOM после
  // пересчёта, см. recomputeAndRender).
  s.stop = +document.getElementById('rc-stop').value || null;
  s.tp = +document.getElementById('rc-tp').value || null;
  s.rr = +document.getElementById('rc-rr').value;
  s.borrowPct = +document.getElementById('rc-borrow-pct').value;
  s.holdDays = +document.getElementById('rc-hold-days').value;
}

function applyRiskCalcSettings(s) {
  document.getElementById('rc-deposit').value = s.deposit;
  document.getElementById('rc-risk-pct').value = s.riskPct;
  document.getElementById('rc-direction').value = s.direction;
  document.getElementById('rc-anchor').value = s.anchor;
  document.getElementById('rc-entry').value = s.entry ?? '';
  document.getElementById('rc-stop').value = s.stop ?? '';
  document.getElementById('rc-tp').value = s.tp ?? '';
  document.getElementById('rc-rr').value = s.rr;
  document.getElementById('rc-borrow-pct').value = s.borrowPct;
  document.getElementById('rc-hold-days').value = s.holdDays;
  document.getElementById('rc-stop').disabled = s.anchor !== 'stop';
  document.getElementById('rc-tp').disabled = s.anchor !== 'profit';
  document.getElementById('rc-short-section').style.display = s.direction === 'short' ? '' : 'none';
}

// Свежий тикер без сохранённой сделки — вход от последней цены, стоп на
// плоских 2% (пользователь тут же подправит), тейк derived через обычный
// пересчёт (anchor уже 'stop' по DEFAULT_SETTINGS).
async function initDefaultTradeFromLastClose(s) {
  const last = await fetchLastClose();
  if (last == null) return;
  s.entry = last;
  const dist = last * 0.02;
  s.stop = s.direction === 'long' ? last - dist : last + dist;
}

// ── персистентность — сделка per-ticker (analysis_settings), депозит/
// риск%/ставка заёма GLOBAL (см. модульный докстринг) ────────────────────

export async function loadRiskCalcDefaults() {
  if (!S.instrumentId) return;
  let s = { ...DEFAULT_SETTINGS };
  try {
    const res = await api(
      'GET', `/series/analysis-settings?instrument_id=${S.instrumentId}&interval=${S.interval}&analyzer_type=risk_calc`
    );
    if (res.params) s = { ...s, ...res.params };
  } catch (_) { /* fall back to defaults */ }
  // Global slice wins regardless of whatever an older per-ticker row still
  // carries for these three fields.
  Object.assign(s, S.toolDisplayDefaults.risk_calc);
  if (s.entry == null) await initDefaultTradeFromLastClose(s);
  S.riskCalcSettings = s;
  applyRiskCalcSettings(s);
  S.riskCalcShowOnChart = getToolShowOnChart('risk_calc', S.riskCalcShowOnChart);
  updateShowChartButton();
  recomputeAndRender();
}

let _saveTimer = null;

// Per-ticker half only — the trade itself (direction/anchor/entry/stop/tp/
// rr/holdDays), NOT deposit/riskPct/borrowPct (see saveGlobalRiskCalcSettings).
function saveRiskCalcSettings() {
  if (!S.instrumentId) return;
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    const s = S.riskCalcSettings;
    api('POST', '/series/analysis-settings', {
      instrument_id: S.instrumentId, interval: S.interval,
      analyzer_type: 'risk_calc',
      params: {
        direction: s.direction, anchor: s.anchor, entry: s.entry, stop: s.stop, tp: s.tp,
        rr: s.rr, holdDays: s.holdDays,
      },
    }).catch(() => {});
  }, 500);
}

// Global half — deposit/riskPct/borrowPct. settings.js:saveToolDisplayDefaults
// owns the actual debounce/POST, shared across every tool using this mechanism.
function saveGlobalRiskCalcSettings() {
  const s = S.riskCalcSettings;
  S.toolDisplayDefaults.risk_calc = { deposit: s.deposit, riskPct: s.riskPct, borrowPct: s.borrowPct };
  saveToolDisplayDefaults();
}

// ── "закреплено на графике" (тот же паттерн, что risk_corridor.js/
// moving_averages.js/zigzag_tool.js) ─────────────────────────────────────

export function toggleRiskCalcShowChart() {
  S.riskCalcShowOnChart = !S.riskCalcShowOnChart;
  updateShowChartButton();
  redraw();
  saveToolShowOnChart('risk_calc', S.riskCalcShowOnChart);
}

function updateShowChartButton() {
  document.getElementById('rc-show-chart-btn')?.classList.toggle('active', !!S.riskCalcShowOnChart);
}

// ── лот + шаг цены (MOEX, кэшируется на instruments.lot_size/price_step) ──

export async function loadRiskCalcTradingParams() {
  if (!S.instrumentId) { S.riskCalcLotSize = 1; S.riskCalcPriceStep = 0.01; return; }
  try {
    const res = await api('GET', `/instruments/${S.instrumentId}/trading-params`);
    S.riskCalcLotSize = res.lot_size || 1;
    S.riskCalcPriceStep = res.price_step || 0.01;
  } catch (_) {
    S.riskCalcLotSize = 1;
    S.riskCalcPriceStep = 0.01;
  }
  const el = document.getElementById('rc-lot-size');
  if (el) el.textContent = `${S.riskCalcLotSize} шт. / ${S.riskCalcPriceStep}`;
  for (const id of ['rc-entry', 'rc-stop', 'rc-tp']) {
    const input = document.getElementById(id);
    if (input) input.step = S.riskCalcPriceStep;
  }
}

// ── пересчёт + рендер (форма -> результат -> график) ─────────────────────

function showError(msg) {
  const box = document.getElementById('rc-error');
  if (!box) return;
  box.textContent = msg;
  box.style.display = '';
  const results = document.getElementById('rc-results');
  if (results) results.innerHTML = '';
}

function clearError() {
  const box = document.getElementById('rc-error');
  if (box) box.style.display = 'none';
}

function recomputeAndRender() {
  const s = S.riskCalcSettings;
  const step = S.riskCalcPriceStep || 0.01;
  if (s.entry != null) s.entry = roundToStep(s.entry, step);
  if (s.anchor === 'stop' && s.stop != null) s.stop = roundToStep(s.stop, step);
  if (s.anchor === 'profit' && s.tp != null) s.tp = roundToStep(s.tp, step);

  try {
    validate(s);
  } catch (e) {
    showError(e.message);
    redraw();
    return;
  }

  let resolved;
  try {
    resolved = resolveStopAndTp(s);
    if (resolved.stopDistance <= 0 || resolved.rewardDistance <= 0) {
      throw new Error(
        'При таком сроке удержания/ставке заёма и целевом R:R сделка не складывается — ' +
        'увеличьте тейк, уменьшите R:R или срок удержания.'
      );
    }
  } catch (e) {
    showError(e.message);
    redraw();
    return;
  }
  clearError();

  s.stop = roundToStep(resolved.stop, step);
  s.tp = roundToStep(resolved.tp, step);
  document.getElementById('rc-entry').value = s.entry;
  document.getElementById('rc-stop').value = s.stop;
  document.getElementById('rc-tp').value = s.tp;

  const result = computeTrade(s, S.riskCalcLotSize);
  renderResults(s, result);
  redraw();
  saveRiskCalcSettings();       // per-ticker: the trade itself
  saveGlobalRiskCalcSettings(); // global: deposit/riskPct/borrowPct
}

export function onRiskCalcSettingsChange() {
  readFormIntoSettings();
  recomputeAndRender();
}

export function onRiskCalcDirectionChange() {
  const s = S.riskCalcSettings;
  const newDirection = document.getElementById('rc-direction').value;
  // Стоп (или тейк, если якорь — тейк) автоматически переворачивается на
  // другую сторону от входа с той же дистанцией — вторая цена derived и
  // пересчитается сама.
  if (s.entry != null) {
    if (s.anchor === 'stop' && s.stop != null) s.stop = flipAcrossEntry(s.entry, s.stop, newDirection, 'stop');
    else if (s.anchor === 'profit' && s.tp != null) s.tp = flipAcrossEntry(s.entry, s.tp, newDirection, 'profit');
  }
  s.direction = newDirection;
  document.getElementById('rc-short-section').style.display = newDirection === 'short' ? '' : 'none';
  if (s.stop != null) document.getElementById('rc-stop').value = s.stop;
  if (s.tp != null) document.getElementById('rc-tp').value = s.tp;
  recomputeAndRender();
}

export function onRiskCalcAnchorChange() {
  const s = S.riskCalcSettings;
  s.anchor = document.getElementById('rc-anchor').value;
  document.getElementById('rc-stop').disabled = s.anchor !== 'stop';
  document.getElementById('rc-tp').disabled = s.anchor !== 'profit';
  recomputeAndRender();
}

// Latest close comes from the backend (the newest stored bar), not from the
// loaded chart window.
async function fetchLastClose() {
  if (!S.instrumentId) return null;
  const rows = await api('GET', `/candles?ticker=${S.ticker}&data_source=${S.dataSource}&interval=${S.interval}&limit=1`);
  return rows.length ? rows[0].close : null;
}

export async function setRiskCalcEntryToCurrent() {
  const last = await fetchLastClose();
  if (last == null) { setStatus('Сначала загрузите свечи', 'err'); return; }
  S.riskCalcSettings.entry = last;
  document.getElementById('rc-entry').value = last;
  recomputeAndRender();
}

// ── "установить на графике" — клик по цене, как у инструмента «Ценовой
// уровень» (cursor_tools.js): живой курсор-preview (mousemove) + клик
// фиксирует цену. Полностью самодостаточный обработчик — не трогает
// price_level's собственную логику, priceAtClientY — тонкая публичная
// обёртка над той же проверенной пиксель-математикой (см. chart.js). Любое
// из трёх полей (entry/stop/tp) можно выбрать — pick по stop/tp заодно
// переключает якорь R:R на выбранное поле (иначе выбранная на графике цена
// тут же перезаписалась бы обратным пересчётом от другого якоря). ──────────

let _pickBound = false;

function isRiskCalcToolActive() {
  return S.activeTab === 'main' && S.activeMainTool === 'risk_calc';
}

function bindPickingOnce() {
  if (_pickBound) return;
  _pickBound = true;
  const gd = document.getElementById('chart');
  if (!gd) return;

  gd.addEventListener('mousemove', ev => {
    if (!S.riskCalcPickTarget || ev.buttons !== 0) return;
    S.riskCalcPickHoverPrice = priceAtClientY(gd, ev.clientY);
    renderRiskCalcOverlay(gd);
  });
  gd.addEventListener('mouseleave', () => {
    if (S.riskCalcPickTarget && S.riskCalcPickHoverPrice != null) {
      S.riskCalcPickHoverPrice = null;
      renderRiskCalcOverlay(gd);
    }
  });
  gd.addEventListener('click', ev => {
    const target = S.riskCalcPickTarget;
    if (!target) return;
    const price = priceAtClientY(gd, ev.clientY);
    if (price == null) return;
    const s = S.riskCalcSettings;
    s[target] = price;
    if (target === 'stop') s.anchor = 'stop';
    if (target === 'tp') s.anchor = 'profit';
    document.getElementById('rc-anchor').value = s.anchor;
    document.getElementById('rc-stop').disabled = s.anchor !== 'stop';
    document.getElementById('rc-tp').disabled = s.anchor !== 'profit';
    document.getElementById(`rc-${target}`).value = price;
    S.riskCalcPickTarget = null;
    S.riskCalcPickHoverPrice = null;
    gd.classList.remove('picking-price');
    updatePickButtonsState();
    setStatus('Цена установлена', 'ok');
    recomputeAndRender();
  });
}

function updatePickButtonsState() {
  for (const t of ['entry', 'stop', 'tp']) {
    document.getElementById(`rc-pick-${t}-btn`)?.classList.toggle('active', S.riskCalcPickTarget === t);
  }
}

export function toggleRiskCalcPickOnChart(target) {
  bindPickingOnce();
  const gd = document.getElementById('chart');
  if (S.riskCalcPickTarget === target) {
    S.riskCalcPickTarget = null;
    S.riskCalcPickHoverPrice = null;
    gd?.classList.remove('picking-price');
  } else {
    S.riskCalcPickTarget = target;
    S.riskCalcPickHoverPrice = null;
    gd?.classList.add('picking-price');
    setStatus(`Кликните на графике, чтобы задать цену ${PICK_TARGET_LABEL[target]}`, 'busy');
  }
  updatePickButtonsState();
  renderRiskCalcOverlay(gd);
}

// ── результат (ЛОТЫ первичны, штуки — вторичны) ──────────────────────────

function fmtMoney(x) { return `${Math.round(x).toLocaleString('ru-RU')} ₽`; }
function fmtNum(x) { return Math.round(x).toLocaleString('ru-RU'); }
function fmtPrice(x) { return x.toLocaleString('ru-RU', { minimumFractionDigits: 2, maximumFractionDigits: 4 }); }
function fmtPct(x) { return `${(x * 100).toLocaleString('ru-RU', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`; }

function resultRow(key, val, subText, cssClass) {
  const cls = cssClass ? ` ${cssClass}` : '';
  const sub = subText ? ` <span class="muted-val">(${subText})</span>` : '';
  return `<div class="result-row"><span class="result-key">${key}</span><span class="result-val${cls}">${val}${sub}</span></div>`;
}

function renderResults(s, r) {
  const box = document.getElementById('rc-results');
  if (!box) return;
  const lots = Math.round(r.lots);
  let html = '';
  html += resultRow('Вход', `${fmtPrice(s.entry)} ₽`);
  html += resultRow('Стоп', `${fmtPrice(r.stop)} ₽`, null, 'down');
  html += resultRow('Тейк-профит', `${fmtPrice(r.tp)} ₽`, null, 'up');
  html += resultRow('Размер позиции', `${fmtNum(lots)} лот.`, `${fmtNum(r.size)} шт.`);
  html += resultRow('Объём позиции', fmtMoney(r.notional));
  if (s.direction === 'short') {
    html += resultRow('Стоимость заёма', fmtMoney(r.borrowCost), `${s.holdDays} дн. · ${s.borrowPct}%/год`);
    html += resultRow('Итоговый риск', fmtMoney(r.risk), `${fmtPct(r.riskPctActual)} депозита`, 'down');
    html += resultRow('Чистая прибыль', fmtMoney(r.reward), null, 'up');
  } else {
    html += resultRow('Риск на сделку', fmtMoney(r.risk), `${fmtPct(r.riskPctActual)} депозита`, 'down');
    html += resultRow('Прибыль на тейке', fmtMoney(r.reward), null, 'up');
  }
  html += resultRow('R : R по факту', `1 : ${r.rr.toFixed(2)}`);
  box.innerHTML = html;
}

// ── график — HTML/CSS-оверлей (НЕ Plotly traces/shapes), см. модульный
// докстринг и #price-level-overlay (cursor_tools.js) для образца ─────────

function renderRiskCalcOverlay(gd) {
  const container = document.getElementById('risk-calc-overlay');
  if (!container) return;

  const s = S.riskCalcSettings;
  const linesVisible = isToolObjectVisible('risk_calc', S.riskCalcShowOnChart);
  const hoverVisible = isRiskCalcToolActive() && S.riskCalcPickTarget && S.riskCalcPickHoverPrice != null;
  if (!linesVisible && !hoverVisible) {
    if (container.childElementCount) container.innerHTML = '';
    return;
  }
  gd = gd || document.getElementById('chart');
  if (!gd) return;

  const items = [];
  if (linesVisible) {
    if (s.entry != null) items.push({ price: s.entry, label: 'Вход', color: S.colorProfile.risk_calc_entry });
    if (s.stop != null) items.push({ price: s.stop, label: 'Стоп', color: S.colorProfile.risk_calc_stop });
    if (s.tp != null) items.push({ price: s.tp, label: 'Тейк', color: S.colorProfile.risk_calc_profit });
  }
  if (hoverVisible) {
    const baseColor = S.riskCalcPickTarget === 'entry' ? S.colorProfile.risk_calc_entry
      : S.riskCalcPickTarget === 'stop' ? S.colorProfile.risk_calc_stop : S.colorProfile.risk_calc_profit;
    items.push({
      price: S.riskCalcPickHoverPrice, label: null,
      color: hexToRgba(baseColor, 0.8), lineColor: hexToRgba(baseColor, 0.55), dashed: true,
    });
  }

  container.innerHTML = items.map(it => {
    const py = priceToPixelY(gd, it.price);
    if (py == null) return '';
    const text = it.label ? `${it.label} ${fmtPrice(it.price)}` : fmtPrice(it.price);
    return `
      <div class="risk-calc-line${it.dashed ? ' dashed' : ''}" style="top:${py}px;border-top-color:${it.lineColor ?? it.color}">
        <span class="risk-calc-label" style="color:${it.color}">${text}</span>
      </div>
    `;
  }).join('');
}

registerOverlayRefresher(renderRiskCalcOverlay);

// ── init hooks ────────────────────────────────────────────────────────────

async function onRiskCalcSelected() {
  const s = S.riskCalcSettings;
  if (s.entry == null) {
    await initDefaultTradeFromLastClose(s);
    applyRiskCalcSettings(s);
  }
  recomputeAndRender();
}

function onRiskCalcDeselected() {
  // Незавершённый "пик" цены со графика — транзитное состояние, не должно
  // переживать переключение на другой инструмент (тот же принцип, что
  // price_level's S.pendingPriceLevel, см. cursor_tools.js).
  if (S.riskCalcPickTarget) {
    S.riskCalcPickTarget = null;
    S.riskCalcPickHoverPrice = null;
    document.getElementById('chart')?.classList.remove('picking-price');
    updatePickButtonsState();
  }
  renderRiskCalcOverlay(); // hide immediately if not pinned — don't wait for the next incidental redraw
}

registerTool({
  type: 'risk_calc',
  surface: 'main', // no buildMainTraces (overlay-based, like price_level) — must be explicit, see tools.js:surfaceOf
  icon: 'scale',
  label: 'Калькулятор риска',
  panelId: 'tool-panel-risk_calc',
  onSelected: onRiskCalcSelected,
  onDeselected: onRiskCalcDeselected,
  // no onOriginClick/resetOrigin — рисуется всегда от последнего бара, нет
  // исторической origin-точки.
});
