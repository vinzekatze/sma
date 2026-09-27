import { api, setStatus } from './api.js';
import { toggleAccordion } from './tickers.js';
import { iconHtml } from './icons.js';

const STATUS_LABELS = {
  pending:     'в очереди',
  running:     'выполняется',
  done:        'готово',
  error:       'ошибка',
  cancelled:   'отменена',
  interrupted: 'прервана',
};

const RESUMABLE = ['cancelled', 'interrupted', 'error'];
const POLL_MS = 3000;

let _tasksCache = [];

function esc(s) {
  const div = document.createElement('div');
  div.textContent = s ?? '';
  return div.innerHTML;
}

function miniProgress(task) {
  if (!task.progress_total) return '';
  const pct = Math.round(task.progress_done * 100 / task.progress_total);
  return `<div class="mini-progress"><div class="mini-progress-fill" style="width:${pct}%"></div></div>`;
}

// ── всегда видимая секция «Текущая задача» ──────────────────────────────────
function renderTaskStatus(tasks) {
  const box = document.getElementById('task-status-body');
  if (!box) return;

  const running = tasks.find(t => t.status === 'running');
  if (!running) {
    box.innerHTML = '<span class="muted-val">Нет активных задач</span>';
    return;
  }
  const queueCount = tasks.filter(t => t.status === 'pending').length;
  box.innerHTML = `
    <div class="task-status-current">
      <div class="task-status-label">${esc(running.label || running.kind)}</div>
      ${miniProgress(running)}
      ${queueCount ? `<div class="task-status-queue">+${queueCount} в очереди</div>` : ''}
    </div>
  `;
  box.querySelector('.task-status-current').onclick = () => {
    const header = document.querySelector('#task-manager-body')?.previousElementSibling;
    const body = document.getElementById('task-manager-body');
    if (header && body && !body.classList.contains('open')) toggleAccordion(header);
  };
}

// ── полный список (внутри аккордеона) ───────────────────────────────────────
function renderTaskList(tasks) {
  const box = document.getElementById('task-list');
  if (!box) return;
  if (!tasks.length) {
    box.innerHTML = '<span class="muted-val">Задач ещё нет</span>';
    return;
  }
  box.innerHTML = '';
  tasks.forEach(task => box.appendChild(renderTaskRow(task)));
}

function renderTaskRow(task) {
  const row = document.createElement('div');
  row.className = 'task-row';
  // No progress bar here (project feedback 2026-08-25: "у симплекса
  // продолжают отображаться прогресс бары после завершения... в целом можно
  // полностью убрать из развёрнутого списка") — a finished task's row would
  // otherwise keep showing its LAST progress_done/progress_total (still
  // truthy after completion, see miniProgress), reading as "still running".
  // Progress now lives ONLY in the always-visible «Текущая задача» section
  // (renderTaskStatus below), which only ever shows the actually-running task.
  row.innerHTML = `
    <div class="task-main">
      <div class="task-label" title="${esc(task.label || task.kind)}">${esc(task.label || task.kind)}</div>
    </div>
    <span class="status-badge ${esc(task.status)}">${esc(STATUS_LABELS[task.status] || task.status)}</span>
  `;

  if (task.status === 'pending' || task.status === 'running') {
    const btn = document.createElement('button');
    btn.className = 'icon-btn danger';
    btn.title = 'Отменить';
    btn.innerHTML = iconHtml('close');
    btn.onclick = () => cancelTask(task.id);
    row.appendChild(btn);
  } else {
    // Every terminal status (done/cancelled/interrupted/error) gets a
    // delete button — cancelled/interrupted/error rows never auto-prune
    // (unlike 'done', see prune_old_done_tasks) and previously had no way
    // to be removed at all (reported 2026-09-12).
    if (RESUMABLE.includes(task.status)) {
      const resumeBtn = document.createElement('button');
      resumeBtn.className = 'icon-btn';
      resumeBtn.title = 'Возобновить';
      resumeBtn.innerHTML = iconHtml('refresh');
      resumeBtn.onclick = () => resumeTask(task.id);
      row.appendChild(resumeBtn);
    }
    const delBtn = document.createElement('button');
    delBtn.className = 'icon-btn danger';
    delBtn.title = 'Удалить';
    delBtn.innerHTML = iconHtml('close');
    delBtn.onclick = () => deleteTaskRow(task.id);
    row.appendChild(delBtn);
  }

  return row;
}

// ── действия ─────────────────────────────────────────────────────────────────
export async function cancelTask(id) {
  try {
    await api('POST', `/tasks/${id}/cancel`);
    await refreshTasks();
  } catch (e) {
    setStatus(e.message, 'err');
  }
}

export async function resumeTask(id) {
  try {
    await api('POST', `/tasks/${id}/resume`);
    await refreshTasks();
  } catch (e) {
    setStatus(e.message, 'err');
  }
}

export async function deleteTaskRow(id) {
  try {
    await api('DELETE', `/tasks/${id}`);
    await refreshTasks();
  } catch (e) {
    setStatus(e.message, 'err');
  }
}

export async function resumeAllTasks() {
  try {
    await api('POST', '/tasks/resume-all');
    await refreshTasks();
  } catch (e) {
    setStatus(e.message, 'err');
  }
}

// ── polling (работает постоянно, не только пока аккордеон открыт) ──────────
// Exported: forecast.js/simplex_ensemble.js/tickers.js call this right after
// each WS progress/terminal message so the task manager panel (the ONE place
// progress now shows, see #task-status/#task-list in index.html) updates
// immediately instead of waiting for the next POLL_MS tick.
export async function refreshTasks() {
  try {
    _tasksCache = await api('GET', '/tasks?limit=100');
  } catch (_) {
    return; // transient — следующий тик поллинга повторит
  }
  renderTaskStatus(_tasksCache);
  renderTaskList(_tasksCache);

  const btn = document.getElementById('tasks-resume-all-btn');
  if (btn) btn.disabled = !_tasksCache.some(t => RESUMABLE.includes(t.status));
}

export function initTaskManager() {
  refreshTasks();
  setInterval(refreshTasks, POLL_MS);
}
