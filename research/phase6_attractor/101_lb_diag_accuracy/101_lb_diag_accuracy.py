"""
101_lb_diag_accuracy.py — связь диагностики Levina-Bickel (LB) каскада
с точностью прогноза, как функция d (размерность LP-фильтра).

Алгоритм — точное повторение app3 (prototype/forcaster/ui/app3.py) при настройках:
  - 4 уровня стандартного (×2 octave) каскада
  - p = m (LP-фильтр)
  - ξ_lwr добавка = 5
  - global_blend / acc_ang / LP-коррекция траектории — выключены
  - LP-фильтр: «Авто m/k по d», формула m = 3·d, k = 10·d
Все параметры (m, k, p_fit, p_max, ξ_lwr, уровни каскада) — функции одного d.

Сетка:
  n_iter (итераций LP-фильтра) = 1 → d = 1..20
  n_iter = 2 → d = 1..15
  n_iter = 3 → d = 1..10

Точность — rMAE на h=1 в att-пространстве (см. research/phase5_attractor/93_aligned_cascade.py):
  pred = LWR-прогноз att[origin+1] по каскаду, построенному на att, посчитанном из
         close[:origin+1] (без утечки будущего — LP-фильтр пересчитывается заново
         под каждый origin, как в app3).
  true = последний элемент att, посчитанного из close[:origin+2] (одна свеча вперёд,
         ровно тот бар, что прогнозируется — это не утечка, а раскрытие ответа).
  error = pred - true;  rMAE = mean(|error|) / std(att) (агрегируется отдельно).

Диагностика LB снимается на каждом из 4 уровней каскада (до и после локальной
LP-очистки пула соседей этого уровня), для запроса и для среднего по соседям.
Параметры локальной очистки: d_lp=d, k_lp=10·d (= k фильтра), n_iter=1 (фиксировано,
не связано со sweep'ом n_iter самого LP-фильтра).

Запуск (локально, без Docker — для теста на 2 тикерах):
  cd /home/kali/workspace/apps/sma
  source /home/kali/.venvs/sma/bin/activate
  TEST_MODE=1 python research/phase6_attractor/101_lb_diag_accuracy/101_lb_diag_accuracy.py

Полный прогон — через Docker (см. README.md в этом каталоге).
"""

from __future__ import annotations

import os

# Ограничиваем потоки BLAS до импорта numpy: при ProcessPoolExecutor каждый
# воркер — отдельный процесс, многопоточный OpenBLAS внутри каждого даёт
# oversubscription (N_workers × N_threads потоков на N_cores ядер) и резко
# замедляет работу вместо ускорения. Один поток BLAS на процесс — параллелизм
# обеспечивается самим пулом процессов.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

ROOT     = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 \
           else Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(ROOT / "data" / "candles")))
OUT_DIR  = Path(os.environ.get("RESULTS_DIR", str(Path(__file__).resolve().parent / "results")))
INTERVAL = "1d"

TICKERS_FULL = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
TICKERS_TEST = ["SBER", "CHMF"]

GRID_FULL: dict[int, list[int]] = {
    1: list(range(1, 21)),
    2: list(range(1, 16)),
    3: list(range(1, 11)),
}
GRID_TEST: dict[int, list[int]] = {1: [1, 5, 10]}

N_ORIGINS_FULL = 40
N_ORIGINS_TEST = 5
STEP_WF        = 5

N_LEVELS  = 4        # уровней каскада ×2 (octave)
XI_EXTRA  = 5        # добавка к ξ_lwr = 3*(p_fit+1) + XI_EXTRA
LP_CLEAN_N_ITER = 1  # n_iter локальной LP-очистки пула диагностики (фикс.)
K_LB_MAX  = 8         # верхняя граница k для Levina-Bickel (как в app3)

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
WORKERS   = int(os.environ.get("WORKERS", str(os.cpu_count() or 4)))

EXPERIMENT_ID = "101_lb_diag_accuracy"
IMAGE_VERSION = "v1"


# ── алгоритм (копия чистых функций из app3.py, без streamlit) ──────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=float)
    cn = np.arange(1, n + 1, dtype=float)
    ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    den = cn * ct2 - ct ** 2
    b = np.where(den > 0, (cn * cty - ct * cy) / den, 0.0)
    a = (cy - b * ct) / cn
    tr = np.exp(a + b * t); tr[:2] = close[:2]
    return tr


def lp_proj_signal(ratio: np.ndarray, m: int, d_proj: int, k: int, n_iter: int) -> np.ndarray:
    """Local Projective noise reduction (Grassberger-Hegger) на ratio → diff.
    Офлайн (не каузально внутри переданного окна) — поэтому ratio должен быть
    заранее обрезан до нужного origin вызывающим кодом."""
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
            _, _, Vt = np.linalg.svd(X_nn - centroid, full_matrices=False)
            V_d = Vt[:d_eff].T
            xc = X[i] - centroid
            X_proj[i] = centroid + V_d @ (V_d.T @ xc)
        result = np.zeros(N); count = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i + m] += X_proj[i]; count[i:i + m] += 1
        s = result / np.maximum(count, 1)
    return np.diff(s)


def levels_aligned(p_fit: int, p_max: int) -> list[int]:
    """P-aligned каскад ×2: p_fit × {1,2,4,...} ≤ p_max (скр.93, app3)."""
    levs = [p_fit]; p = p_fit
    while p * 2 <= p_max:
        p *= 2; levs.append(p)
    return list(reversed(levs))


def lwr_approx(X_nn: np.ndarray, y_nn: np.ndarray, vec_fit: np.ndarray, h_bw: float) -> float:
    w = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_fit, axis=1) / h_bw) ** 2)
    A = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vec_fit @ c[1:])


def lp_clean_set(X: np.ndarray, d: int, k: int, n_iter: int = 1) -> np.ndarray:
    """Local Projective на малом наборе: проекция каждой точки на d-мерную
    касательную плоскость её k ближайших соседей внутри набора (app3._lp_clean_set)."""
    Xc = X.copy()
    k = min(k, len(Xc) - 1)
    for _ in range(n_iter):
        diff = Xc[:, None, :] - Xc[None, :, :]
        D = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        np.fill_diagonal(D, np.inf)
        Xn = np.empty_like(Xc)
        for i in range(len(Xc)):
            nn = np.argsort(D[i])[:k]
            mu = Xc[nn].mean(0)
            _, _, Vt = np.linalg.svd(Xc[nn] - mu, full_matrices=False)
            v = Xc[i] - mu
            Xn[i] = mu + Vt[:d].T @ (Vt[:d] @ v)
        Xc = Xn
    return Xc


def dim_diag_metrics(X_all: np.ndarray, k_lb: int) -> tuple[float, float]:
    """Levina-Bickel: X_all[0] = запрос, X_all[1:] = соседи (app3._dim_diag_metrics)."""
    n = len(X_all)
    k = max(3, min(k_lb, n - 2))
    diff = X_all[:, None, :] - X_all[None, :, :]
    D = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(D, np.inf)
    D_s = np.sort(D, axis=1)

    def _lb(row: int) -> float:
        rk = D_s[row, k - 1]
        rj = D_s[row, :k - 1]
        if rk <= 1e-14 or np.any(rj <= 1e-14):
            return float("nan")
        return (k - 2) / float(np.sum(np.log(rk / rj)))

    lb_q = _lb(0)
    lb_ns = [_lb(i) for i in range(1, n)]
    return (float(lb_q) if np.isfinite(lb_q) else float("nan"),
            float(np.nanmean(lb_ns)))


# ── один прогон каскада + диагностика по уровням ────────────────────────────

@dataclass
class OriginResult:
    ticker: str
    n_iter: int
    d: int
    m: int
    k_lp: int
    p_fit: int
    p_max: int
    xi_lwr: int
    origin: int
    pred: float = float("nan")
    true: float = float("nan")
    ok: bool = False
    skip_reason: str = ""
    levels: list[dict] = field(default_factory=list)


def build_att(close_slice: np.ndarray, m: int, d: int, k: int, n_iter: int) -> np.ndarray:
    lt = logtrend_causal(close_slice)
    ratio = close_slice / np.maximum(lt, 1e-10)
    return lp_proj_signal(ratio, m, d, k, n_iter)


def run_one_origin(ticker: str, close: np.ndarray, origin: int,
                    n_iter: int, d: int) -> OriginResult:
    m = 3 * d
    k_lp = 10 * d
    p_fit = m
    p_max = p_fit * (2 ** (N_LEVELS - 1))
    xi_lwr = 3 * (p_fit + 1) + XI_EXTRA

    res = OriginResult(ticker=ticker, n_iter=n_iter, d=d, m=m, k_lp=k_lp,
                        p_fit=p_fit, p_max=p_max, xi_lwr=xi_lwr, origin=origin)

    # ── att без утечки (close[:origin+1]) + att с раскрытым 1 баром (ground truth) ──
    att = build_att(close[:origin + 1], m, d, k_lp, n_iter)
    att_ext = build_att(close[:origin + 2], m, d, k_lp, n_iter)
    if len(att_ext) == 0:
        res.skip_reason = "att_ext empty"; return res
    true_val = float(att_ext[-1])

    n = len(att)
    levels = levels_aligned(p_fit, p_max)
    p_top = levels[0]
    m_cas = n - p_top - 1
    if m_cas < 3:
        res.skip_reason = "insufficient history for p_top"; return res

    t_arr = np.arange(p_top, n - 1)
    X_full = np.column_stack([att[t_arr - (p_top - 1 - j)] for j in range(p_top)])
    y_base = att[t_arr + 1]
    if len(X_full) < xi_lwr:
        res.skip_reason = "pool smaller than xi_lwr"; return res

    vec_full0 = att[-p_top:].copy()
    cands = np.arange(len(X_full))

    for k_lvl, p_lvl in enumerate(levels):
        is_last = (k_lvl == len(levels) - 1)
        xi_clip = min(xi_lwr, len(cands))
        if len(cands) > xi_clip:
            dists = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full0[-p_lvl:], axis=1)
            cands = cands[np.argpartition(dists, xi_clip - 1)[:xi_clip]]

        # ── диагностика на этом уровне ──
        X_neigh = X_full[cands, -p_lvl:]
        x_q = vec_full0[-p_lvl:]
        n_neigh = len(cands)
        k_lb_d = max(3, min(K_LB_MAX, n_neigh - 2))
        lb_before_q, lb_before_n = dim_diag_metrics(
            np.vstack([x_q[None, :], X_neigh]), k_lb=k_lb_d)

        k_clean = min(k_lp, len(X_neigh) - 1)
        if k_clean >= 3:
            X_lp = lp_clean_set(X_neigh, d=d, k=k_clean, n_iter=LP_CLEAN_N_ITER)
            D_q = np.linalg.norm(X_lp - x_q, axis=1)
            nn_q = np.argsort(D_q)[:k_clean]
            mu_q = X_lp[nn_q].mean(0)
            _, _, Vt_q = np.linalg.svd(X_lp[nn_q] - mu_q, full_matrices=False)
            v_q = x_q - mu_q
            d_eff = min(d, Vt_q.shape[0])
            x_q_lp = mu_q + Vt_q[:d_eff].T @ (Vt_q[:d_eff] @ v_q)
            lb_after_q, lb_after_n = dim_diag_metrics(
                np.vstack([x_q_lp[None, :], X_lp]), k_lb=k_lb_d)
        else:
            lb_after_q, lb_after_n = float("nan"), float("nan")

        res.levels.append({
            "level": k_lvl, "p_lvl": p_lvl, "n_neigh": n_neigh,
            "lb_before_q": lb_before_q, "lb_before_n": lb_before_n,
            "lb_after_q": lb_after_q, "lb_after_n": lb_after_n,
        })

        if not is_last:
            p_next = levels[k_lvl + 1]
            radius = p_lvl - p_next
            offsets = np.arange(radius + 1)
            expanded = cands[:, None] - offsets[None, :]
            expanded = np.clip(expanded, 0, len(X_full) - 1)
            cands = np.unique(expanded)

    if len(cands) < p_fit + 2:
        res.skip_reason = "final pool too small for LWR"; return res

    X_nn = X_full[cands, -p_fit:]
    y_nn = y_base[cands]
    vec_f = vec_full0[-p_fit:]
    h_bw = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
    pred_val = lwr_approx(X_nn, y_nn, vec_f, h_bw)

    res.pred = pred_val
    res.true = true_val
    res.ok = True
    return res


# ── загрузка данных ──────────────────────────────────────────────────────────

def load_close(ticker: str) -> np.ndarray:
    p = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw = json.loads(p.read_text())
    return np.array([c["close"] for c in raw], dtype=float)


def origins_for(close: np.ndarray, n_origins: int, step: int) -> list[int]:
    n_total = len(close)
    last = n_total - 2  # нужен origin+2 ≤ n_total-1 для ground truth
    start = last - n_origins * step
    return list(range(max(0, start), last, step))


# ── воркер для ProcessPoolExecutor ───────────────────────────────────────────
# close-массивы передаются один раз на процесс через initializer, а не в каждой
# задаче — иначе ~14400 копий по сети IPC.

_CLOSES: dict[str, np.ndarray] = {}


def _init_worker(closes: dict[str, np.ndarray]) -> None:
    global _CLOSES
    _CLOSES = closes


def _worker(args: tuple) -> dict:
    ticker, origin, n_iter, d = args
    t0 = time.time()
    res = run_one_origin(ticker, _CLOSES[ticker], origin, n_iter, d)
    dt = time.time() - t0
    return {"res": res, "dt": dt}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = TICKERS_TEST if TEST_MODE else TICKERS_FULL
    grid = GRID_TEST if TEST_MODE else GRID_FULL
    n_origins = N_ORIGINS_TEST if TEST_MODE else N_ORIGINS_FULL
    suffix = "_test" if TEST_MODE else ""

    print(f"=== {EXPERIMENT_ID}  image={IMAGE_VERSION}  TEST_MODE={TEST_MODE} ===")
    print(f"tickers={tickers}")
    print(f"grid={grid}")
    print(f"n_origins={n_origins}  step={STEP_WF}  workers={WORKERS}")

    diag_path = OUT_DIR / f"diag_levels{suffix}.csv"
    acc_path = OUT_DIR / f"forecast_accuracy{suffix}.csv"
    meta_path = OUT_DIR / f"run_meta{suffix}.json"

    meta = {
        "experiment_id": EXPERIMENT_ID, "image_version": IMAGE_VERSION,
        "test_mode": TEST_MODE, "tickers": tickers, "grid": grid,
        "n_origins": n_origins, "step_wf": STEP_WF, "n_levels": N_LEVELS,
        "xi_extra": XI_EXTRA, "lp_clean_n_iter": LP_CLEAN_N_ITER,
        "k_lb_max": K_LB_MAX, "workers": WORKERS,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    acc_cols = ["ticker", "n_iter", "d", "m", "k_lp", "p_fit", "p_max", "xi_lwr",
                "origin", "pred", "true", "error", "abs_error", "ok", "skip_reason"]
    diag_cols = ["ticker", "n_iter", "d", "origin", "level", "p_lvl", "n_neigh",
                 "lb_before_q", "lb_before_n", "lb_after_q", "lb_after_n"]

    f_acc = open(acc_path, "w")
    f_diag = open(diag_path, "w")
    f_acc.write(",".join(acc_cols) + "\n")
    f_diag.write(",".join(diag_cols) + "\n")

    tasks = []
    closes: dict[str, np.ndarray] = {}
    for ticker in tickers:
        close = load_close(ticker)
        closes[ticker] = close
        origins = origins_for(close, n_origins, STEP_WF)
        for n_iter, ds in grid.items():
            for d in ds:
                for origin in origins:
                    tasks.append((ticker, origin, n_iter, d))

    n_total_tasks = len(tasks)
    print(f"total tasks: {n_total_tasks}")

    t_start = time.time()
    n_done = 0
    n_ok = 0
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker,
                              initargs=(closes,)) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futures):
            out = fut.result()
            res: OriginResult = out["res"]
            n_done += 1

            if res.ok:
                n_ok += 1
                err = res.pred - res.true
                row = [res.ticker, res.n_iter, res.d, res.m, res.k_lp, res.p_fit,
                       res.p_max, res.xi_lwr, res.origin, res.pred, res.true,
                       err, abs(err), 1, ""]
            else:
                row = [res.ticker, res.n_iter, res.d, res.m, res.k_lp, res.p_fit,
                       res.p_max, res.xi_lwr, res.origin, "", "", "", "", 0,
                       res.skip_reason]
            f_acc.write(",".join(str(x) for x in row) + "\n")

            for lvl in res.levels:
                drow = [res.ticker, res.n_iter, res.d, res.origin, lvl["level"],
                        lvl["p_lvl"], lvl["n_neigh"], lvl["lb_before_q"],
                        lvl["lb_before_n"], lvl["lb_after_q"], lvl["lb_after_n"]]
                f_diag.write(",".join(str(x) for x in drow) + "\n")

            # онлайн-flush — частичные результаты переживут краш
            f_acc.flush(); f_diag.flush()

            if n_done % 50 == 0 or n_done == n_total_tasks:
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (n_total_tasks - n_done)
                print(f"  {n_done}/{n_total_tasks}  ok={n_ok}  "
                      f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    f_acc.close(); f_diag.close()
    print(f"\nГотово: {n_done} задач, {n_ok} успешных.")
    print(f"  {acc_path}")
    print(f"  {diag_path}")


if __name__ == "__main__":
    main()
