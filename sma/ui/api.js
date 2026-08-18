export async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  if (!r.ok) {
    const detail = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(detail.detail || r.statusText);
  }
  if (r.status === 204) return null;
  return r.json();
}

export function setStatus(msg, type = '') {
  const el = document.getElementById('statusbar');
  el.textContent = msg;
  el.className = type;
}

export function setBusy(msg) { setStatus(msg, 'busy'); disableControls(true); }

export function setIdle(msg = '', ok = true) {
  setStatus(msg, ok ? 'ok' : 'err');
  disableControls(false);
}

export function disableControls(on) {
  ['spectrogram-calculate-btn', 'refresh-current-btn'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = on;
  });
}

// Общий WS-подписчик на прогресс задачи (/tasks/{id}/ws) — messages вида
// {status:'running',done,total} | {status:'done',...} | {status:'error',error}
// | {status:'cancelled'}. Закрывается сам на терминальном статусе.
export function connectTaskWS(taskId, onMessage) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/tasks/${taskId}/ws`);
  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    onMessage(msg);
    if (['done', 'error', 'cancelled'].includes(msg.status)) ws.close();
  };
  ws.onerror = () => onMessage({ status: 'error', error: 'WebSocket disconnected' });
  return ws;
}
