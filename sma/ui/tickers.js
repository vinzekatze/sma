import { S } from './state.js';
import { api, setStatus, connectTaskWS } from './api.js';
import { refreshTasks } from './tasks.js';
import { iconHtml } from './icons.js';

// Ключ — asset_type (точнее engine: currency_metal и currency_selt делят один
// engine="currency", но это разные по смыслу инструменты).
const ASSET_TYPE_LABELS = {
  stock: 'Акция', fund: 'Фонд', bond: 'Облигация', index: 'Индекс',
  currency: 'Валюта', metal: 'Металл', futures: 'Фьючерс', option: 'Опцион',
};

let _instrumentsCache = [];   // последний список /instruments
let _coverageCache = {};      // instrument_id -> coverage[]

const MANAGER_PAGE_SIZE = 20;
let _managerPage = 0;         // «Мои тикеры» — постраничный вывод (список может разрастись на сотни при калибровке моделей)

function esc(s) {
  const div = document.createElement('div');
  div.textContent = s ?? '';
  return div.innerHTML;
}

function fmtDate(ts) {
  return ts ? ts.slice(0, 10) : '';
}

function currentInterval() {
  return document.getElementById('interval-select').value;
}

function hasData(coverage, interval) {
  return coverage.some(c => c.interval === interval);
}

// ── общая загрузка + рендер обеих панелей ──────────────────────────────────
export async function refreshAll() {
  const list = await api('GET', '/instruments');
  const coverages = await Promise.all(
    list.map(inst => api('GET', `/instruments/${inst.id}/coverage`).catch(() => []))
  );
  _instrumentsCache = list;
  _coverageCache = {};
  list.forEach((inst, i) => { _coverageCache[inst.id] = coverages[i]; });
  renderQuickSwitcher();
  renderTickerManagerList();
  return list;
}

// ── аккордеон (generic: работает для любого количества .accordion-item) ────
export function toggleAccordion(header) {
  const body = header.nextElementSibling;
  const isOpen = body.classList.toggle('open');
  header.classList.toggle('open', isOpen);
}

// ── быстрый переключатель (избранное + фильтр по всем добавленным) ─────────
// Одно поле фильтра работает на обе панели: сужает и «Мои тикеры» тоже —
// иначе среди сотен добавленных при калибровке моделей тикеров конкретный
// не найти постраничным пролистыванием.
export function onQuickFilterInput() {
  _managerPage = 0;
  renderQuickSwitcher();
  renderTickerManagerList();
}

// Интервал в тулбаре общий для всей навигации — пересчитать «нет данных» и
// дефолт select'а в менеджере при его смене, без повторного похода на сервер.
export function onGlobalIntervalChange() {
  renderQuickSwitcher();
  renderTickerManagerList();
}

function renderQuickSwitcher() {
  const box = document.getElementById('quick-ticker-list');
  const filterVal = document.getElementById('ticker-quick-filter').value.trim().toLowerCase();

  let items;
  if (!filterVal) {
    items = _instrumentsCache.filter(i => i.is_favorite);
    if (!items.length) {
      box.innerHTML = '<span class="muted-val">Нет избранных — отметьте иконкой избранного в менеджере ниже</span>';
      return;
    }
  } else {
    items = _instrumentsCache.filter(i =>
      i.ticker.toLowerCase().includes(filterVal) ||
      (i.full_name || '').toLowerCase().includes(filterVal)
    );
    if (!items.length) {
      box.innerHTML = '<span class="muted-val">Ничего не найдено</span>';
      return;
    }
  }

  items = [...items].sort((a, b) => a.ticker.localeCompare(b.ticker));
  box.innerHTML = '';
  const interval = currentInterval();
  items.forEach(inst => box.appendChild(renderQuickRow(inst, interval)));
}

function renderQuickRow(inst, interval) {
  const coverage = _coverageCache[inst.id] || [];
  const noData = !hasData(coverage, interval);

  const row = document.createElement('div');
  row.className = 'quick-row' + (noData ? ' no-data' : '');
  row.innerHTML = `
    <span class="fav-star-static">${inst.is_favorite ? iconHtml('star', 'icon-fill') : ''}</span>
    <span class="quick-ticker">${esc(inst.ticker)}</span>
    <span class="ticker-badge">${esc(ASSET_TYPE_LABELS[inst.asset_type] || inst.asset_type)}</span>
    ${noData ? `<span class="no-data-badge" title="Нет данных на интервале ${esc(interval)}">нет данных</span>` : ''}
  `;
  row.onclick = () => window.loadCandles(inst.ticker, inst.data_source, interval);
  return row;
}

// ── «Мои тикеры» (внутри аккордеона) ────────────────────────────────────────
function renderTickerManagerList() {
  const box = document.getElementById('my-tickers-list');
  if (!box) return;
  if (!_instrumentsCache.length) {
    box.innerHTML = '<span class="muted-val">Тикеров ещё нет — добавьте ниже</span>';
    return;
  }

  const filterVal = document.getElementById('ticker-quick-filter').value.trim().toLowerCase();
  let items = _instrumentsCache;
  if (filterVal) {
    items = items.filter(i =>
      i.ticker.toLowerCase().includes(filterVal) ||
      (i.full_name || '').toLowerCase().includes(filterVal)
    );
  }
  if (!items.length) {
    box.innerHTML = '<span class="muted-val">Ничего не найдено</span>';
    return;
  }

  const sorted = [...items].sort((a, b) => {
    if (a.is_favorite !== b.is_favorite) return a.is_favorite ? -1 : 1;
    return a.ticker.localeCompare(b.ticker);
  });

  const pageCount = Math.ceil(sorted.length / MANAGER_PAGE_SIZE);
  _managerPage = Math.min(Math.max(_managerPage, 0), pageCount - 1);
  const pageItems = sorted.slice(
    _managerPage * MANAGER_PAGE_SIZE,
    (_managerPage + 1) * MANAGER_PAGE_SIZE
  );

  const interval = currentInterval();
  box.innerHTML = '';
  pageItems.forEach(inst => box.appendChild(renderTickerRow(inst, _coverageCache[inst.id] || [], interval)));
  box.appendChild(renderManagerPager(pageCount, sorted.length));
}

function renderManagerPager(pageCount, total) {
  const bar = document.createElement('div');
  bar.className = 'pager';
  if (pageCount <= 1) {
    bar.innerHTML = `<span class="muted-val">${total} тикеров</span>`;
    return bar;
  }
  bar.innerHTML = `
    <button class="icon-btn">‹</button>
    <span class="muted-val">${_managerPage + 1} / ${pageCount} · ${total} тикеров</span>
    <button class="icon-btn">›</button>
  `;
  const [prevBtn, nextBtn] = bar.querySelectorAll('button');
  prevBtn.disabled = _managerPage === 0;
  nextBtn.disabled = _managerPage >= pageCount - 1;
  prevBtn.onclick = () => { _managerPage--; renderTickerManagerList(); };
  nextBtn.onclick = () => { _managerPage++; renderTickerManagerList(); };
  return bar;
}

function coverageLine(coverage) {
  if (!coverage.length) return '<div class="coverage-line">нет загруженных данных</div>';
  const badges = coverage
    .map(c => `<span class="coverage-badge">${esc(c.interval)}: ${fmtDate(c.first)}–${fmtDate(c.last)} (${c.n})</span>`)
    .join('');
  return `<div class="coverage-line">${badges}</div>`;
}

function renderTickerRow(inst, coverage, interval) {
  const noData = !hasData(coverage, interval);

  const row = document.createElement('div');
  row.className = 'ticker-row' + (noData ? ' no-data' : '');
  row.innerHTML = `
    <button class="fav-star ${inst.is_favorite ? 'active' : ''}" title="Избранное">${iconHtml('star', inst.is_favorite ? 'icon-fill' : '')}</button>
    <div class="ticker-main">
      <div class="ticker-name">
        ${esc(inst.ticker)} <span class="ticker-badge">${esc(ASSET_TYPE_LABELS[inst.asset_type] || inst.asset_type)}</span>
        ${noData ? `<span class="no-data-badge" title="Нет данных на интервале ${esc(interval)}">нет ${esc(interval)}</span>` : ''}
      </div>
      ${coverageLine(coverage)}
    </div>
    <button class="icon-btn" title="Скачать / дозагрузить текущий интервал (${esc(interval)})">${iconHtml('refresh')}</button>
    <button class="icon-btn danger" title="Удалить тикер и все связанные данные">${iconHtml('close')}</button>
  `;
  const [starBtn, refreshBtn, delBtn] = row.querySelectorAll('button');
  starBtn.onclick = () => toggleFavorite(inst.id, !inst.is_favorite);
  refreshBtn.onclick = () => fetchTickerInterval(inst.ticker, inst.data_source, interval, refreshBtn);
  delBtn.onclick = () => removeTicker(inst.id, inst.ticker);
  row.querySelector('.ticker-main').onclick = () => window.loadCandles(inst.ticker, inst.data_source, interval);
  return row;
}

async function toggleFavorite(id, value) {
  try {
    await api('PATCH', `/instruments/${id}`, { is_favorite: value });
    await refreshAll();
  } catch (e) {
    setStatus(e.message, 'err');
  }
}

// Дозагрузка теперь идёт через очередь задач (POST /candles/fetch отвечает
// task_id, не результатом сразу) — не блокирует остальной интерфейс, пока
// ждёт своей очереди. Прогресс/результат — по WS, тем же паттерном, что и
// у прогноза (sma/ui/forecast.js:connectWS), через общий connectTaskWS.
async function fetchTickerInterval(ticker, dataSource, interval, btn) {
  btn.disabled = true;
  try {
    const res = await api('POST', '/candles/fetch', { ticker, data_source: dataSource, interval });
    setStatus(`Задача #${res.task_id} поставлена в очередь`, 'ok');
    refreshTasks(); // immediate task-manager refresh (#task-status/#task-list) — progress lives there now

    connectTaskWS(res.task_id, msg => {
      if (msg.status === 'running' && msg.total > 0) {
        const pct = Math.round(msg.done * 100 / msg.total);
        setStatus(`${ticker} [${interval}] — ${pct}%…`, 'busy');
        refreshTasks();
        return;
      }
      if (msg.status === 'done') {
        setStatus(`${ticker} [${interval}]: ${msg.mode === 'incremental' ? 'догружено' : 'загружено'} ${msg.candles_saved} свечей`, 'ok');
        if (S.ticker === ticker && S.dataSource === dataSource && S.interval === interval) {
          window.loadCandles(ticker, dataSource, interval);
        }
        refreshAll();
      } else if (msg.status === 'error') {
        setStatus(msg.error, 'err');
      } else if (msg.status === 'cancelled') {
        setStatus(`${ticker} [${interval}]: задача отменена`, 'err');
      }
      if (msg.status !== 'running') btn.disabled = false;
      refreshTasks();
    });
  } catch (e) {
    setStatus(e.message, 'err');
    btn.disabled = false;
  }
}

// Кнопка «Обновить» в тулбаре — дозагружает (или скачивает с нуля) данные
// текущего открытого тикера на текущем глобальном интервале.
export async function refreshCurrentTicker(btn) {
  if (!S.ticker) { setStatus('Сначала выберите тикер в менеджере слева', 'err'); return; }
  await fetchTickerInterval(S.ticker, S.dataSource, currentInterval(), btn);
}

async function removeTicker(id, ticker) {
  if (!confirm(`Удалить тикер ${ticker} и все связанные данные (свечи, прогнозы, задачи)?`)) return;
  try {
    await api('DELETE', `/instruments/${id}`);
    if (S.ticker === ticker) {
      S.candles = [];
      S.ticker = '';
      S.originTs = null;
      S.renderedForecasts.clear();
      S.pinnedForecastIds.clear();
      S.selectedForecastId = null;
      S.shapes = [];
      S.spectrogramData = null;
      S.subpanel = 'none';
      if (window.Plotly) window.Plotly.purge('chart');
    }
    await refreshAll();
  } catch (e) {
    setStatus(e.message, 'err');
  }
}

// ── «Добавить тикер» (поиск MOEX по тексту и/или категории) ─────────────────
function matchesQuery(hit, q) {
  const needle = q.toLowerCase();
  return hit.secid.toLowerCase().includes(needle)
    || (hit.shortname || '').toLowerCase().includes(needle)
    || (hit.name || '').toLowerCase().includes(needle);
}

export async function searchTickers() {
  const q = document.getElementById('ticker-search-input').value.trim();
  const category = document.getElementById('ticker-search-category').value;
  const box = document.getElementById('ticker-search-results');

  // TQBR — цельный листинг борда одним запросом (полнее и надёжнее, чем
  // постраничный поиск по группе); текст, если введён, фильтруется на клиенте.
  if (category === 'tqbr') {
    box.innerHTML = '<span class="muted-val">Загрузка списка TQBR…</span>';
    try {
      const hits = await api('GET', '/instruments/board');
      renderSecurityResults(q ? hits.filter(h => matchesQuery(h, q)) : hits);
    } catch (e) {
      box.innerHTML = `<span class="muted-val">${esc(e.message)}</span>`;
    }
    return;
  }

  if (!category && !q) {
    box.innerHTML = '<span class="muted-val">Введите запрос или выберите категорию</span>';
    return;
  }

  box.innerHTML = '<span class="muted-val">Поиск…</span>';
  try {
    const params = new URLSearchParams();
    if (q) params.set('q', q);
    if (category) params.set('category', category);
    const hits = await api('GET', `/instruments/search?${params}`);
    renderSecurityResults(hits);
  } catch (e) {
    box.innerHTML = `<span class="muted-val">${esc(e.message)}</span>`;
  }
}

function renderSecurityResults(hits) {
  const box = document.getElementById('ticker-search-results');
  if (!hits.length) {
    box.innerHTML = '<span class="muted-val">Ничего не найдено</span>';
    return;
  }
  box.innerHTML = '';
  hits.forEach(hit => box.appendChild(renderSecurityRow(hit)));
}

function renderSecurityRow(hit) {
  const row = document.createElement('div');
  row.className = 'sec-row';
  row.innerHTML = `
    <div class="sec-main">
      <div class="sec-secid">${esc(hit.secid)} <span class="ticker-badge">${esc(ASSET_TYPE_LABELS[hit.asset_type] || hit.asset_type)}</span></div>
      <div class="sec-name">${esc(hit.shortname || hit.name || '')}</div>
    </div>
    <button ${hit.already_added ? 'disabled' : ''}>${hit.already_added ? 'Добавлен' : '+ Добавить'}</button>
  `;
  const btn = row.querySelector('button');
  if (!hit.already_added) {
    btn.onclick = () => addTicker(hit, btn);
  }
  return row;
}

async function addTicker(hit, btn) {
  btn.disabled = true;
  btn.textContent = '…';
  try {
    await api('POST', '/instruments', {
      ticker: hit.secid,
      data_source: 'moex',
      asset_type: hit.asset_type,
      full_name: hit.name,
      engine: hit.engine,
      market: hit.market,
      board: hit.board,
    });
    btn.textContent = 'Добавлен';
    await refreshAll();
  } catch (e) {
    btn.disabled = false;
    btn.textContent = '+ Добавить';
    setStatus(e.message, 'err');
  }
}
