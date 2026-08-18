import { api, setIdle, setStatus } from './api.js';

// ── Настройки приложения (левая панель) ─────────────────────────────────────
// Global tunables backed by app_settings (singleton row, sma/core/db.py) —
// currently MOEX request concurrency (pool liquidity ranking, see
// sma/core/forecast/pool_selection.py) and calibration worker count
// (ProcessPoolExecutor, see sma/api/task_manager.py:_run_calibration).

export async function loadAppSettings() {
  try {
    const s = await api('GET', '/settings');
    document.getElementById('settings-moex-workers').value = s.moex_pool_workers;
    document.getElementById('settings-calibration-workers').value = s.calibration_workers;
  } catch (e) {
    setStatus(e.message, 'err');
  }
}

export async function saveAppSettings() {
  const moexWorkers = +document.getElementById('settings-moex-workers').value;
  const calibWorkers = +document.getElementById('settings-calibration-workers').value;
  const btn = document.getElementById('settings-save-btn');
  btn.disabled = true;
  try {
    await api('POST', '/settings', {
      moex_pool_workers: moexWorkers,
      calibration_workers: calibWorkers,
    });
    setIdle('Настройки сохранены');
  } catch (e) {
    setIdle(e.message, false);
  } finally {
    btn.disabled = false;
  }
}
