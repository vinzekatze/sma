// dialog.js — custom confirm dialog replacing native window.confirm().
// Native confirm()/alert() render with the OS/browser's own chrome (title
// bar, button style, font) — visibly inconsistent with the app's dark theme
// and, per project feedback 2026-08-25 ("Спинеры и скролбары... стандартные
// диалоговые окна"), the specific kind of cross-browser inconsistency
// already flagged once for scrollbars/range-sliders. Modeled on
// workspace/apps/trudozatrati/app's showConfirm/showDialog (same project,
// already solved this — user pointed at it directly), trimmed to just the
// confirm case since that's the only native dialog sma currently calls
// (forecast.js:deleteCalibration, tickers.js:deleteTicker — both plain
// yes/no, no alert()/prompt() anywhere in the codebase to replace).
//
// Promise-based rather than a callback, so a call site reads the same as
// the native confirm() it replaces: `if (!(await showConfirm(...))) return;`

let _resolve = null;

function _close(result) {
  document.getElementById('dlg-overlay')?.classList.add('hidden');
  _resolve?.(result);
  _resolve = null;
}

export function showConfirm(message, { title = 'Подтверждение', confirmLabel = 'Удалить', danger = true } = {}) {
  const overlay = document.getElementById('dlg-overlay');
  if (!overlay) return Promise.resolve(window.confirm(message)); // markup missing — degrade rather than silently do nothing

  document.getElementById('dlg-title').textContent = title;
  document.getElementById('dlg-body').textContent = message;
  const confirmBtn = document.getElementById('dlg-confirm-btn');
  confirmBtn.textContent = confirmLabel;
  confirmBtn.classList.toggle('btn-danger', danger);

  overlay.classList.remove('hidden');
  confirmBtn.focus();

  return new Promise(resolve => { _resolve = resolve; });
}

export function dialogConfirmClick() { _close(true); }
export function dialogCancelClick() { _close(false); }
