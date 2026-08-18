"""
107_decouple_m_k.py — урезанный скрин: разделить роли параметров LP-фильтра.

Идея пользователя: текущая конвенция `m=3d=p_fit` смешивает две разные роли
— m/k определяют КАЧЕСТВО очистки att внутри LP-фильтра (Grassberger-Hegger
проекция), а p_fit определяет, сколько лагов очищенного att видит LWR-
регрессия downstream. Эмпирически m=3d было выбрано пользователем для
наглядности ручной калибровки, не как теоретически обоснованное значение —
m≥2d+1 (Такенс) — это лишь достаточная теоретическая граница для чистой
generic-системы, не гарантия оптимальности для шумных данных.

Протокол этого скрина:
  - d=11, p_fit=33 (=3·11, по текущей конвенции), n_iter=2 — ГРУБЫЕ
    параметры, зафиксированы на установленном пике эксп.101 (rMAE=0.327
    без гейта). p_fit/xi_lwr/levels/p_max — НЕ зависят от m_filter.
  - m_filter ∈ {2d+1, round(2.5d), 3d} = {23, 28, 33} — ТОНКИЙ параметр,
    окно вложения ВНУТРИ LP-фильтра (lp_proj_signal), решает только качество
    очистки att, p_fit-регрессия не знает, какое m_filter использовалось.
  - k_filter ∈ {5d, 10d, 15d} = {55, 110, 165} — ТОНКИЙ параметр, число
    соседей для локального PCA внутри LP-фильтра.
  - 9 комбинаций (m_filter × k_filter), m_filter=33,k_filter=110 — baseline
    (точное совпадение с эксп.101 при d=11,n_iter=2).

Метрика: rMAE (h=1, att) + lb_before_q (финальный уровень) + доля origins
с d≥LB — то же, что в эксп.101, но теперь как функция (m_filter,k_filter)
при фиксированных d/p_fit, а не как функция d.

Выход: results/screen_decouple_mk{_test}.csv
  ticker,n_iter,d,p_fit,m_filter,k_filter,origin,pred,true,abs_error,
  lb_before_q,dgeLB

Запуск:
  source /home/kali/.venvs/sma/bin/activate
  TEST_MODE=1 python research/phase6_attractor/101_lb_diag_accuracy/107_decouple_m_k.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("RESULTS_DIR", str(HERE / "results")))

_spec = importlib.util.spec_from_file_location("exp101", HERE / "101_lb_diag_accuracy.py")
exp101 = importlib.util.module_from_spec(_spec)
sys.modules["exp101"] = exp101
_spec.loader.exec_module(exp101)


def _lp_proj_signal_safe(ratio: np.ndarray, m: int, d_proj: int, k: int, n_iter: int) -> np.ndarray:
    """Та же защита от LinAlgError, что в 106 — локальный monkey-patch,
    101_lb_diag_accuracy.py не меняется."""
    from scipy.spatial import KDTree
    s = ratio.copy().astype(np.float64)
    N = len(s)
    k_eff = min(k, N - m)
    d_eff = min(d_proj, m - 1)
    for _ in range(n_iter):
        n_pts = N - m + 1
        if n_pts < k_eff + 1:
            break
        rows = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
        X = s[rows]
        tree = KDTree(X)
        _, inds = tree.query(X, k=k_eff + 1)
        X_proj = np.empty_like(X)
        for i in range(n_pts):
            nn = inds[i, 1:]
            X_nn = X[nn]
            centroid = X_nn.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(X_nn - centroid, full_matrices=False)
            except np.linalg.LinAlgError:
                try:
                    import scipy.linalg
                    _, _, Vt = scipy.linalg.svd(X_nn - centroid, full_matrices=False,
                                                 lapack_driver="gesvd")
                except np.linalg.LinAlgError:
                    X_proj[i] = X[i]
                    continue
            V_d = Vt[:d_eff].T
            xc = X[i] - centroid
            X_proj[i] = centroid + V_d @ (V_d.T @ xc)
        result = np.zeros(N); count = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i + m] += X_proj[i]; count[i:i + m] += 1
        s = result / np.maximum(count, 1)
    return np.diff(s)


exp101.lp_proj_signal = _lp_proj_signal_safe

TICKERS_FULL = exp101.TICKERS_FULL
TICKERS_TEST = ["SBER", "CHMF"]

D_FIXED = 11
N_ITER_FIXED = 2
P_FIT_FIXED = 3 * D_FIXED  # 33, грубый параметр — НЕ зависит от m_filter

M_GRID_FULL = [2 * D_FIXED + 1, round(2.5 * D_FIXED), 3 * D_FIXED]       # 23, 28, 33
K_GRID_FULL = [5 * D_FIXED, 10 * D_FIXED, 15 * D_FIXED]                  # 55, 110, 165
M_GRID_TEST = [23, 33]
K_GRID_TEST = [55, 110]

N_ORIGINS_SCREEN = 15
N_ORIGINS_TEST = 5
STEP_WF = 5

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
WORKERS = int(os.environ.get("WORKERS", str(os.cpu_count() or 4)))

EXPERIMENT_ID = "107_decouple_m_k"
IMAGE_VERSION = "v1"


def run_one_origin(ticker: str, close: np.ndarray, origin: int, m_filter: int, k_filter: int) -> dict:
    d = D_FIXED; n_iter = N_ITER_FIXED; p_fit = P_FIT_FIXED
    p_max = p_fit * (2 ** (exp101.N_LEVELS - 1))
    xi_lwr = 3 * (p_fit + 1) + exp101.XI_EXTRA
    k_lb_d = max(3, min(exp101.K_LB_MAX, xi_lwr - 2))

    row = dict(ticker=ticker, n_iter=n_iter, d=d, p_fit=p_fit, m_filter=m_filter,
               k_filter=k_filter, origin=origin, pred=np.nan, true=np.nan,
               abs_error=np.nan, lb_before_q=np.nan, dgeLB=np.nan, ok=0, skip_reason="")

    att = exp101.build_att(close[:origin + 1], m_filter, d, k_filter, n_iter)
    att_ext = exp101.build_att(close[:origin + 2], m_filter, d, k_filter, n_iter)
    if len(att_ext) == 0:
        row["skip_reason"] = "att_ext empty"; return row
    true_val = float(att_ext[-1])

    n = len(att)
    levels = exp101.levels_aligned(p_fit, p_max)
    p_top = levels[0]
    if n - p_top - 1 < 3:
        row["skip_reason"] = "insufficient history"; row["true"] = true_val; return row

    t_arr = np.arange(p_top, n - 1)
    X_full = np.column_stack([att[t_arr - (p_top - 1 - j)] for j in range(p_top)])
    y_base = att[t_arr + 1]
    if len(X_full) < xi_lwr:
        row["skip_reason"] = "pool smaller than xi_lwr"; row["true"] = true_val; return row

    vec_full0 = att[-p_top:].copy()
    cands = np.arange(len(X_full))
    for k_lvl, p_lvl in enumerate(levels):
        is_last = (k_lvl == len(levels) - 1)
        xi_clip = min(xi_lwr, len(cands))
        if len(cands) > xi_clip:
            dists = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full0[-p_lvl:], axis=1)
            cands = cands[np.argpartition(dists, xi_clip - 1)[:xi_clip]]
        if is_last:
            x_q = vec_full0[-p_lvl:]
            X_neigh = X_full[cands, -p_lvl:]
            lb_q, _ = exp101.dim_diag_metrics(np.vstack([x_q[None, :], X_neigh]), k_lb=k_lb_d)
            row["lb_before_q"] = lb_q
            row["dgeLB"] = int(d >= lb_q) if np.isfinite(lb_q) else np.nan
        else:
            p_next = levels[k_lvl + 1]
            radius = p_lvl - p_next
            offsets = np.arange(radius + 1)
            expanded = cands[:, None] - offsets[None, :]
            expanded = np.clip(expanded, 0, len(X_full) - 1)
            cands = np.unique(expanded)

    if len(cands) < p_fit + 2:
        row["skip_reason"] = "final pool too small for LWR"; row["true"] = true_val; return row

    X_nn = X_full[cands, -p_fit:]
    y_nn = y_base[cands]
    vec_f = vec_full0[-p_fit:]
    h_bw = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
    pred_val = exp101.lwr_approx(X_nn, y_nn, vec_f, h_bw)

    row["pred"] = pred_val; row["true"] = true_val
    row["abs_error"] = abs(pred_val - true_val)
    row["ok"] = 1
    return row


_CLOSES: dict[str, np.ndarray] = {}


def _init_worker(closes: dict[str, np.ndarray]) -> None:
    global _CLOSES
    _CLOSES = closes


def _worker(args: tuple) -> dict:
    ticker, origin, m_filter, k_filter = args
    return run_one_origin(ticker, _CLOSES[ticker], origin, m_filter, k_filter)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = TICKERS_TEST if TEST_MODE else TICKERS_FULL
    m_grid = M_GRID_TEST if TEST_MODE else M_GRID_FULL
    k_grid = K_GRID_TEST if TEST_MODE else K_GRID_FULL
    n_origins = N_ORIGINS_TEST if TEST_MODE else N_ORIGINS_SCREEN
    suffix = "_test" if TEST_MODE else ""

    print(f"=== {EXPERIMENT_ID}  image={IMAGE_VERSION}  TEST_MODE={TEST_MODE} ===")
    print(f"tickers={tickers}  d={D_FIXED} p_fit={P_FIT_FIXED} n_iter={N_ITER_FIXED}")
    print(f"m_grid={m_grid}  k_grid={k_grid}  n_origins={n_origins}  workers={WORKERS}")

    closes = {t: exp101.load_close(t) for t in tickers}
    tasks = []
    for t in tickers:
        origins = exp101.origins_for(closes[t], n_origins, STEP_WF)
        for m_filter in m_grid:
            for k_filter in k_grid:
                for o in origins:
                    tasks.append((t, o, m_filter, k_filter))
    print(f"всего задач: {len(tasks)}")

    out_path = OUT_DIR / f"screen_decouple_mk{suffix}.csv"
    cols = ["ticker", "n_iter", "d", "p_fit", "m_filter", "k_filter", "origin",
             "pred", "true", "abs_error", "lb_before_q", "dgeLB", "ok", "skip_reason"]

    t0 = time.time()
    n_done = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker,
                                  initargs=(closes,)) as ex:
            futures = [ex.submit(_worker, task) for task in tasks]
            for fut in as_completed(futures):
                writer.writerow(fut.result())
                f.flush()
                n_done += 1
                if n_done % 50 == 0 or n_done == len(tasks):
                    dt = time.time() - t0
                    rate = n_done / dt
                    eta = (len(tasks) - n_done) / rate if rate > 0 else float("nan")
                    print(f"  {n_done}/{len(tasks)}  {dt:.1f}s  rate={rate:.2f}/s  eta={eta:.0f}s")

    print(f"Готово: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
