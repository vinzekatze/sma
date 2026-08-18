"""
104_lb_greedy_select.py — пилот: отбор финального пула соседей каскада через
жадное сужение по изоляции (приближённая минимизация lb_before_n), вместо
чистого геометрического расстояния (baseline эксп.101).

Гипотеза H4 (согласована с пользователем в чате, не записана в README101,
т.к. это отдельный пилот, а не часть основного эксперимента):
  Отбор финального пула, который явно учитывает локальную самооценку
  размерности соседей (lb_before_n), даёт более низкий rMAE, чем чистое
  geometric-L2 отбор (эксп.101), в зоне лучшей точности (n_iter=2/3, d=7..13).

Метод (согласован):
  - Целевая величина — lb_before_n (среднее LB по соседям; lb_before_q
    практически не зависит от curации пула за пределами top-K_LB_MAX=8
    ближайших к запросу точек — отбором не двигается, поэтому не годится
    как цель).
  - Жадное сужение = дешёвая аппроксимация (без пересчёта формулы LB на
    каждом шаге): на каждом шаге удаляется точка с наибольшим расстоянием
    до её k_lb-го соседа внутри ТЕКУЩЕГО пула (самая «изолированная»).
  - Источник широкого пула — НЕ искусственное расширение поиска, а уже
    существующий промежуточный candidate-set каскада, который естественно
    шире xi_lwr (натурный коэффициент измерен эмпирически ~1.9-2.6×, см.
    обсуждение в чате) — пул после уровня N-2 (предпоследнего), перед тем
    как baseline-код 101 сузил бы его до xi_lwr на финальном уровне.
  - Baseline (для парного сравнения на том же origin) — top-xi_lwr этого же
    пула по чистому расстоянию (= ровно метод эксп.101).
  - И baseline, и новый метод применяются к ОДНОМУ И ТОМУ ЖЕ origin/att/
    промежуточному пулу — честное парное сравнение, не два отдельных прогона.

Пилотная сетка (согласована): n_iter∈{2,3}, d∈{7..13} (14 комбинаций),
8 тикеров (TICKERS_FULL из эксп.101), 40 origins/тикер, шаг 5 — тот же
walk-forward протокол, что в эксп.101 (origins_for).

Переиспользует чистые функции из 101_lb_diag_accuracy.py (build_att,
levels_aligned, lwr_approx, dim_diag_metrics, load_close, origins_for) через
importlib — чтобы не дублировать алгоритм каскада/LP-фильтра отдельной
копией (см. feedback_critical_libraries: прогнозные функции критичны).

Выход: results/pilot_greedy_select{_test}.csv
  ticker,n_iter,d,origin,xi_lwr,wide_pool_size,
  pred_base,pred_new,true,abs_error_base,abs_error_new,
  lb_n_base,lb_n_new,ok,skip_reason

Запуск (локально, без Docker — см. оценку времени в README после первого
тестового прогона):
  source /home/kali/.venvs/sma/bin/activate
  TEST_MODE=1 python research/phase6_attractor/101_lb_diag_accuracy/104_lb_greedy_select.py
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

TICKERS_FULL = exp101.TICKERS_FULL
TICKERS_TEST = ["SBER", "CHMF"]

PILOT_GRID_FULL: dict[int, list[int]] = {2: list(range(7, 14)), 3: list(range(7, 14))}
PILOT_GRID_TEST: dict[int, list[int]] = {2: [8, 10]}

N_ORIGINS_FULL = 40
N_ORIGINS_TEST = 5
STEP_WF = 5

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
WORKERS = int(os.environ.get("WORKERS", str(os.cpu_count() or 4)))

EXPERIMENT_ID = "104_lb_greedy_select"
IMAGE_VERSION = "v1"


def greedy_isolation_narrow(X: np.ndarray, target_n: int, k_lb: int) -> np.ndarray:
    """Итеративно убирает самую изолированную точку (по расстоянию до её
    k_lb-го соседа в текущем наборе) до размера target_n. Возвращает индексы
    (в исходном X) оставшихся точек.

    Полная матрица расстояний считается один раз; на каждом шаге удаления
    только маскируется (без повторного пересчёта diff/einsum по координатам)
    — иначе пересчёт O(n^2*dim) на каждом из ~100 шагов на origin был
    основным узким местом (замерено: ~2.5с/задачу при наивной версии)."""
    idx = np.arange(len(X))
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(D, np.inf)
    while len(idx) > target_n:
        k = max(1, min(k_lb, len(idx) - 2))
        kth_dist = np.partition(D, k - 1, axis=1)[:, k - 1]
        worst = np.argmax(kth_dist)
        keep = np.ones(len(idx), dtype=bool); keep[worst] = False
        D = D[keep][:, keep]
        idx = idx[keep]
    return idx


def run_one_origin(ticker: str, close: np.ndarray, origin: int, n_iter: int, d: int) -> dict:
    m = 3 * d; k_lp = 10 * d; p_fit = m
    p_max = p_fit * (2 ** (exp101.N_LEVELS - 1))
    xi_lwr = 3 * (p_fit + 1) + exp101.XI_EXTRA
    k_lb_d = max(3, min(exp101.K_LB_MAX, xi_lwr - 2))

    row = {"ticker": ticker, "n_iter": n_iter, "d": d, "origin": origin,
           "xi_lwr": xi_lwr, "wide_pool_size": 0,
           "pred_base": float("nan"), "pred_new": float("nan"), "true": float("nan"),
           "abs_error_base": float("nan"), "abs_error_new": float("nan"),
           "lb_n_base": float("nan"), "lb_n_new": float("nan"),
           "ok": 0, "skip_reason": ""}

    att = exp101.build_att(close[:origin + 1], m, d, k_lp, n_iter)
    att_ext = exp101.build_att(close[:origin + 2], m, d, k_lp, n_iter)
    if len(att_ext) == 0:
        row["skip_reason"] = "att_ext empty"; return row
    true_val = float(att_ext[-1])

    n = len(att)
    levels = exp101.levels_aligned(p_fit, p_max)
    p_top = levels[0]
    if n - p_top - 1 < 3:
        row["skip_reason"] = "insufficient history for p_top"; return row

    t_arr = np.arange(p_top, n - 1)
    X_full = np.column_stack([att[t_arr - (p_top - 1 - j)] for j in range(p_top)])
    y_base = att[t_arr + 1]
    if len(X_full) < xi_lwr:
        row["skip_reason"] = "pool smaller than xi_lwr"; return row

    vec_full0 = att[-p_top:].copy()
    cands = np.arange(len(X_full))

    # ── каскад по уровням levels[:-1] (узкий+расширение), как в baseline 101 ──
    for k_lvl, p_lvl in enumerate(levels[:-1]):
        xi_clip = min(xi_lwr, len(cands))
        if len(cands) > xi_clip:
            dists = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full0[-p_lvl:], axis=1)
            cands = cands[np.argpartition(dists, xi_clip - 1)[:xi_clip]]
        p_next = levels[k_lvl + 1]
        radius = p_lvl - p_next
        offsets = np.arange(radius + 1)
        expanded = cands[:, None] - offsets[None, :]
        expanded = np.clip(expanded, 0, len(X_full) - 1)
        cands = np.unique(expanded)

    p_fit_lvl = levels[-1]  # = p_fit
    wide_pool = cands  # пул, входящий в финальный уровень — естественно шире xi_lwr
    row["wide_pool_size"] = len(wide_pool)
    if len(wide_pool) < xi_lwr + 2:
        row["skip_reason"] = "wide pool too small"; return row

    X_wide = X_full[wide_pool, -p_fit_lvl:]
    vec_f = vec_full0[-p_fit_lvl:]

    # ── baseline: top-xi_lwr по расстоянию (= метод эксп.101) ──
    dists_wide = np.linalg.norm(X_wide - vec_f, axis=1)
    xi_clip = min(xi_lwr, len(wide_pool))
    base_local_idx = np.argpartition(dists_wide, xi_clip - 1)[:xi_clip]
    cands_base = wide_pool[base_local_idx]

    # ── новый метод: жадное сужение по изоляции до xi_lwr ──
    new_local_idx = greedy_isolation_narrow(X_wide, target_n=xi_clip, k_lb=k_lb_d)
    cands_new = wide_pool[new_local_idx]

    def _predict_and_lbn(sel_cands: np.ndarray) -> tuple[float, float]:
        X_nn = X_full[sel_cands, -p_fit:]
        y_nn = y_base[sel_cands]
        h_bw = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
        pred = exp101.lwr_approx(X_nn, y_nn, vec_f, h_bw)
        _, lb_n = exp101.dim_diag_metrics(np.vstack([vec_f[None, :], X_nn]), k_lb=k_lb_d)
        return pred, lb_n

    pred_base, lb_n_base = _predict_and_lbn(cands_base)
    pred_new, lb_n_new = _predict_and_lbn(cands_new)

    row["pred_base"] = pred_base; row["pred_new"] = pred_new; row["true"] = true_val
    row["abs_error_base"] = abs(pred_base - true_val)
    row["abs_error_new"] = abs(pred_new - true_val)
    row["lb_n_base"] = lb_n_base; row["lb_n_new"] = lb_n_new
    row["ok"] = 1
    return row


_CLOSES: dict[str, np.ndarray] = {}


def _init_worker(closes: dict[str, np.ndarray]) -> None:
    global _CLOSES
    _CLOSES = closes


def _worker(args: tuple) -> dict:
    ticker, origin, n_iter, d = args
    return run_one_origin(ticker, _CLOSES[ticker], origin, n_iter, d)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = TICKERS_TEST if TEST_MODE else TICKERS_FULL
    grid = PILOT_GRID_TEST if TEST_MODE else PILOT_GRID_FULL
    n_origins = N_ORIGINS_TEST if TEST_MODE else N_ORIGINS_FULL
    suffix = "_test" if TEST_MODE else ""

    print(f"=== {EXPERIMENT_ID}  image={IMAGE_VERSION}  TEST_MODE={TEST_MODE} ===")
    print(f"tickers={tickers}  grid={grid}  n_origins={n_origins}  workers={WORKERS}")

    closes = {t: exp101.load_close(t) for t in tickers}
    tasks = []
    for t in tickers:
        origins = exp101.origins_for(closes[t], n_origins, STEP_WF)
        for n_iter, ds in grid.items():
            for d in ds:
                for o in origins:
                    tasks.append((t, o, n_iter, d))
    print(f"всего задач: {len(tasks)}")

    out_path = OUT_DIR / f"pilot_greedy_select{suffix}.csv"
    cols = ["ticker", "n_iter", "d", "origin", "xi_lwr", "wide_pool_size",
            "pred_base", "pred_new", "true", "abs_error_base", "abs_error_new",
            "lb_n_base", "lb_n_new", "ok", "skip_reason"]

    t0 = time.time()
    n_done = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker,
                                  initargs=(closes,)) as ex:
            futures = [ex.submit(_worker, task) for task in tasks]
            for fut in as_completed(futures):
                row = fut.result()
                writer.writerow(row)
                f.flush()
                n_done += 1
                if n_done % 200 == 0 or n_done == len(tasks):
                    dt = time.time() - t0
                    rate = n_done / dt
                    eta = (len(tasks) - n_done) / rate if rate > 0 else float("nan")
                    print(f"  {n_done}/{len(tasks)}  {dt:.1f}s  rate={rate:.2f}/s  eta={eta:.0f}s")

    print(f"Готово: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
