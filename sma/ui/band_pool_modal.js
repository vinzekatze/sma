import { S } from './state.js';
import { api, setStatus, connectTaskWS } from './api.js';
import { refreshTasks } from './tasks.js';
import { iconHtml } from './icons.js';
import { loadBandLambdaPool } from './forecast.js';

// ── band_lambda "Пул" modal ─────────────────────────────────────────────────
// Replaces the old multi-T "Настройки" modal (band_settings_modal.js,
// removed 2026-09-12 along with λ-calibration and per-T saved rows — see
// memory project_phase7_calibration_removed_final). ONE pool per
// (instrument, interval) now, no T/min_bars/m/theta here (those are free
// live parameters on the forecast request itself, edited in the sidebar —
// see forecast.js) and no coverage check (removed along with calibration,
// nothing left to compare against a baseline).
//
// _pool holds the modal's current (possibly unsaved) edit state, loaded
// fresh from the backend every time the modal opens (openBandPoolModal),
// then possibly overlaid by reconnectPendingResolve() with an in-flight or
// just-finished "Обновить по категориям" result — the pool_resolve task
// itself is the only thing that persists across opens/closes, not any
// local JS state.

const MODEL_TYPE = 'band_lambda';
const POOL_CATEGORIES = [
  { value: 'stock', label: 'Акции' },
  { value: 'bond', label: 'Облигации' },
  { value: 'fund', label: 'Фонды (ПИФ)' },
  { value: 'currency', label: 'Валюта' },
  { value: 'metal', label: 'Металлы' },
  { value: 'futures', label: 'Фьючерсы' },
];

let _pool = null;          // {categories: [...], n, tickers: [{id,ticker}]} — editable copy
let _searchResults = [];

export async function openBandPoolModal() {
  if (!S.instrumentId) { setStatus('Сначала загрузите свечи', 'err'); return; }
  document.getElementById('bl-pool-overlay').classList.remove('hidden');

  let cfg = null;
  try {
    const res = await api(
      'GET', `/forecast-settings/pool?instrument_id=${S.instrumentId}&interval=${S.interval}&model_type=${MODEL_TYPE}`
    );
    cfg = res.pool;
  } catch (_) { /* falls back to the empty draft below */ }

  _pool = cfg
    ? { categories: cfg.categories || [], n: cfg.n || 25, tickers: (cfg.resolved_tickers || []).map(t => ({ ...t })) }
    : { categories: ['stock'], n: 25, tickers: [] };
  _searchResults = [];
  renderPoolModal();

  await reconnectPendingResolve();
}

// Picks up a pool_resolve task for THIS (instrument, interval) that was
// still in flight, or finished, since the modal was last open — without
// this, closing the modal mid-resolve (or just missing the one-shot WS
// 'done' message) meant starting the whole resolve over from scratch
// (reported 2026-09-12: "резолв пула... не подвязан к окну... заново
// приходится ждать и заново проводить резолв"). Runs after renderPoolModal
// so it can react on top of whatever the saved pool already showed.
async function reconnectPendingResolve() {
  let tasks;
  try {
    tasks = await api('GET', '/tasks?kind=pool_resolve&limit=20');
  } catch (_) { return; }
  const task = tasks.find(t => t.instrument_id === S.instrumentId && t.interval === S.interval);
  if (!task) return;

  const { categories, n } = task.params.pool;

  if (task.status === 'pending' || task.status === 'running') {
    setStatus('Резолвинг пула ещё выполняется…', 'busy');
    document.getElementById('blp-resolve-btn').disabled = true;
    subscribeResolveTask(task.id, categories, n);
    return;
  }

  // A finished PREVIEW (save=false) whose result nobody was listening for
  // (modal closed, WS message missed) — its candidate list still lives in
  // pool_resolution_cache, so it's recoverable without hitting MOEX again.
  // A finished SAVE needs no recovery: GET /forecast-settings/pool above
  // already reflects the true persisted state either way.
  if (task.status === 'done' && task.params.save === false) {
    let cache;
    try {
      cache = await api('GET', `/forecast-settings/resolve-pool-cache?categories=${categories.join(',')}&n=${n}`);
    } catch (_) { return; }
    if (!cache.resolved_tickers) return;
    _pool = { categories, n, tickers: cache.resolved_tickers };
    setStatus(`Восстановлен результат последнего резолва — ${cache.resolved_tickers.length} тикеров`, 'ok');
    document.querySelectorAll('.blp-pool-cat').forEach(el => { el.checked = categories.includes(el.value); });
    document.getElementById('blp-pool-n').value = n;
    renderPoolChips();
  }
}

function subscribeResolveTask(taskId, categories, n) {
  const btn = document.getElementById('blp-resolve-btn');
  connectTaskWS(taskId, msg => {
    if (msg.status === 'done') {
      _pool = { categories, n, tickers: msg.pool.resolved_tickers };
      setStatus(`Пул обновлён — ${msg.pool.resolved_tickers.length} тикеров`, 'ok');
      renderPoolChips();
      btn.disabled = false;
    }
    if (msg.status === 'error') { setStatus(`Ошибка: ${msg.error}`, 'err'); btn.disabled = false; }
    if (msg.status === 'cancelled') { setStatus('Отменено', 'err'); btn.disabled = false; }
    refreshTasks();
  });
}

export function closeBandPoolModal() {
  document.getElementById('bl-pool-overlay').classList.add('hidden');
}

function renderPoolModal() {
  const box = document.getElementById('bl-pool-detail');
  box.innerHTML = `
    <section class="panel-section">
      <h4>
        Пул
        <span class="info-badge" data-tip="Список ниже — то, что реально используется. Категории+N — рецепт, по которому его можно пересобрать заново кнопкой «Обновить по категориям»; ручные правки (добавить/убрать тикер) в рецепт не попадают.">?</span>
      </h4>
      ${POOL_CATEGORIES.map(c => `
        <div class="field-row checkbox-row"><label>
          <input type="checkbox" class="blp-pool-cat" value="${c.value}" ${_pool.categories.includes(c.value) ? 'checked' : ''}> ${c.label}
        </label></div>
      `).join('')}
      <div class="field-row"><label>N на категорию</label><input id="blp-pool-n" type="number" value="${_pool.n}" min="1" max="200"></div>
      <button type="button" id="blp-resolve-btn" style="width:100%;margin:4px 0">Обновить по категориям</button>

      <div class="field-row"><label>Тикеров в пуле</label><span class="muted-val">${_pool.tickers.length}</span></div>
      <div id="blp-pool-chips">${poolChipsHtml()}</div>
      <div class="field-row" style="margin-top:6px">
        <input id="blp-ticker-search" type="text" placeholder="Тикер вручную…" style="flex:1">
        <button type="button" id="blp-search-btn">Найти</button>
      </div>
      <div id="blp-search-results"></div>
    </section>

    <section class="panel-section">
      <div class="field-row">
        <button type="button" id="blp-save-btn" style="flex:1">Сохранить</button>
      </div>
    </section>
  `;

  wireDetailEvents();
}

function poolChipsHtml() {
  if (!_pool.tickers.length) return '<span class="muted-val">пусто</span>';
  return _pool.tickers.map(t => `
    <span class="bl-pool-ticker-chip">${t.ticker}
      <button type="button" class="icon-btn danger" data-remove-ticker="${t.id}" style="width:16px;height:16px;padding:0">${iconHtml('close')}</button>
    </span>
  `).join('');
}

function renderPoolChips() {
  const box = document.getElementById('blp-pool-chips');
  if (box) box.innerHTML = poolChipsHtml();
  wirePoolChipRemovers();
}

function wirePoolChipRemovers() {
  document.querySelectorAll('[data-remove-ticker]').forEach(btn => {
    btn.onclick = () => {
      const id = +btn.dataset.removeTicker;
      _pool.tickers = _pool.tickers.filter(t => t.id !== id);
      renderPoolChips();
    };
  });
}

function readPoolCategoriesFromDom() {
  return {
    categories: [...document.querySelectorAll('.blp-pool-cat:checked')].map(el => el.value),
    n: +document.getElementById('blp-pool-n').value,
  };
}

function wireDetailEvents() {
  wirePoolChipRemovers();

  document.getElementById('blp-resolve-btn').onclick = async () => {
    const { categories, n } = readPoolCategoriesFromDom();
    if (!categories.length) { setStatus('Выберите хотя бы одну категорию пула', 'err'); return; }
    const btn = document.getElementById('blp-resolve-btn');
    btn.disabled = true;
    setStatus('Резолвинг пула…', 'busy');
    try {
      // Async (202 + task_id) since 2026-09-12 — resolving can hit MOEX for
      // 100+ candidates and occasionally runs long; see task_manager.py:
      // _run_pool_resolve for why this is no longer a synchronous request.
      // Closing/reopening the modal while this runs picks the SAME task
      // back up — see reconnectPendingResolve above.
      const task = await api('POST', '/forecast-settings/resolve-pool', {
        instrument_id: S.instrumentId, interval: S.interval, pool: { categories, n },
      });
      refreshTasks();
      subscribeResolveTask(task.task_id, categories, n);
    } catch (e) {
      setStatus(e.message, 'err');
      btn.disabled = false;
    }
  };

  document.getElementById('blp-search-btn').onclick = () => searchTickerForPool();
  document.getElementById('blp-ticker-search').addEventListener('keydown', e => {
    if (e.key === 'Enter') searchTickerForPool();
  });

  document.getElementById('blp-save-btn').onclick = () => saveDetail();
}

async function searchTickerForPool() {
  const q = document.getElementById('blp-ticker-search').value.trim();
  if (!q) return;
  const box = document.getElementById('blp-search-results');
  box.innerHTML = '<span class="muted-val">Поиск…</span>';
  try {
    const hits = await api('GET', `/instruments/search?q=${encodeURIComponent(q)}`);
    _searchResults = hits.slice(0, 10);
    box.innerHTML = _searchResults.length
      ? _searchResults.map((h, i) => `
          <div class="sec-row">
            <div class="sec-main"><span class="sec-secid">${h.secid}</span> <span class="sec-name">${h.shortname || h.name || ''}</span></div>
            <button type="button" data-add-hit="${i}" ${_pool.tickers.some(t => t.ticker === h.secid) ? 'disabled' : ''}>+ Добавить</button>
          </div>
        `).join('')
      : '<span class="muted-val">Ничего не найдено</span>';
    box.querySelectorAll('[data-add-hit]').forEach(btn => {
      btn.onclick = () => addSearchHitToPool(_searchResults[+btn.dataset.addHit], btn);
    });
  } catch (e) {
    box.innerHTML = `<span class="muted-val">${e.message}</span>`;
  }
}

async function addSearchHitToPool(hit, btn) {
  btn.disabled = true;
  btn.textContent = '…';
  try {
    const instr = await api('POST', '/instruments', {
      ticker: hit.secid, data_source: 'moex', asset_type: hit.asset_type,
      full_name: hit.name || hit.shortname, engine: hit.engine, market: hit.market, board: hit.board,
    });
    if (!_pool.tickers.some(t => t.id === instr.id)) {
      _pool.tickers.push({ id: instr.id, ticker: instr.ticker });
      renderPoolChips();
    }
    btn.textContent = 'Добавлено';
  } catch (e) {
    setStatus(e.message, 'err');
    btn.disabled = false;
    btn.textContent = '+ Добавить';
  }
}

// ── save action ───────────────────────────────────────────────────────────

async function saveDetail() {
  if (!S.instrumentId) { setStatus('Сначала загрузите свечи', 'err'); return; }
  if (!_pool.tickers.length) { setStatus('Добавьте хотя бы один тикер в пул', 'err'); return; }

  const { categories, n } = readPoolCategoriesFromDom();
  _pool.categories = categories;
  _pool.n = n;
  const pool = { categories, n, manual_instrument_ids: _pool.tickers.map(t => t.id) };
  const body = { instrument_id: S.instrumentId, interval: S.interval, model_type: MODEL_TYPE, pool };

  const btn = document.getElementById('blp-save-btn');
  btn.disabled = true;
  try {
    setStatus('Сохранение…', 'busy');
    // Async (202 + task_id) since 2026-09-12 — see task_manager.py:
    // _run_pool_resolve. In practice this save always carries
    // manual_instrument_ids (already resolved above), so the task finishes
    // near-instantly — the task/WS round trip is just for a uniform
    // contract with every other MOEX-adjacent operation, not because this
    // particular save is expected to be slow.
    const task = await api('POST', '/forecast-settings/pool', body);
    refreshTasks();
    connectTaskWS(task.task_id, async msg => {
      if (msg.status === 'done') {
        setStatus('Пул сохранён', 'ok');
        await loadBandLambdaPool(); // refreshes the sidebar "Пул" summary
        closeBandPoolModal();
      }
      if (msg.status === 'error') { setStatus(`Ошибка: ${msg.error}`, 'err'); btn.disabled = false; }
      if (msg.status === 'cancelled') { setStatus('Отменено', 'err'); btn.disabled = false; }
      refreshTasks();
    });
  } catch (e) {
    setStatus(e.message, 'err');
    btn.disabled = false;
  }
}
