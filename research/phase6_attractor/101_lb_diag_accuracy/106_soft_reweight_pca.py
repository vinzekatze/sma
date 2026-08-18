"""
106_soft_reweight_pca.py — урезанный пилот для оценки потенциала двух идей,
предложенных после отрицательного результата 104 (жёсткий greedy-отбор по
изоляции не улучшил точность):

Идея 1 (мягкое перевзвешивание, аналог acc_ang из фазы 5):
  Вместо удаления «изолированных» соседей из пула — штрафовать их в самой
  дистанции для Гауссова веса LWR: d_total = d_pos + λ·s, где s — расстояние
  соседа до его k_lb-го соседа ВНУТРИ ОБЫЧНОГО ФИНАЛЬНОГО ПУЛА эксп.101
  (xi_lwr точек, без расширения — решение согласовано с пользователем:
  дешевле, пул не расширяется, нет жадной итерации). λ=0 ⇒ совпадает с
  baseline (встроенная проверка корректности).

Идея 3 (PCA/Махаланобис-метрика вместо Евклида при отборе):
  На естественно широком пуле, входящем в финальный уровень каскада (тот же
  пул, что эксп.101 сужает до xi_lwr по расстоянию — здесь без изменений
  кода каскада, тот же объём ~1.9-2.6×xi_lwr), считаем локальный PCA (как
  lp_clean_set, но теперь для САМОГО ОТБОРА, не только диагностики),
  проецируем на top-d компонент (d = размерность LP-фильтра, согласовано
  с пользователем), отбираем xi_lwr кандидатов по расстоянию В ПРОЕКЦИИ
  (не в исходном пространстве). Вес LWR считается по ОБЫЧНОМУ (сырому)
  расстоянию отобранных точек — меняется только КТО попал в пул, не как
  взвешивается готовый пул (изолирует эффект отбора от эффекта взвешивания).

Урезанный масштаб (оценка потенциала перед длинными расчётами):
  n_iter=2 (фиксировано), d∈{8,10,11,13}, 8 тикеров, N_ORIGINS_SCREEN origins
  (меньше, чем основной грид, чтобы быстро оценить направление эффекта).

Выход: results/screen_soft_pca{_test}.csv (long-format: 1 строка =
  ticker,n_iter,d,origin,method,param,pred,true,abs_error)

Запуск:
  source /home/kali/.venvs/sma/bin/activate
  TEST_MODE=1 python research/phase6_attractor/101_lb_diag_accuracy/106_soft_reweight_pca.py
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
    """Копия exp101.lp_proj_signal с защитой от LinAlgError на вырожденных
    наборах точек (встречено на полной сетке этого скрина — не было видно
    на сетке эксп.101). Патчится только локально в этом скрипте, исходный
    101_lb_diag_accuracy.py не меняется (см. README §8, эксп.101 закрыт и
    задокументирован на старом коде)."""
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
                                                 lapack_driver="gesvd")  # медленнее, устойчивее
                except np.linalg.LinAlgError:
                    X_proj[i] = X[i]  # отказ от проекции для этой точки — безопасный no-op
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

D_GRID_FULL = [8, 10, 11, 13]
D_GRID_TEST = [10]
N_ITER_FIXED = 2

N_ORIGINS_SCREEN = 15
N_ORIGINS_TEST = 5
STEP_WF = 5

LAMBDA_GRID = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
WORKERS = int(os.environ.get("WORKERS", str(os.cpu_count() or 4)))

EXPERIMENT_ID = "106_soft_reweight_pca"
IMAGE_VERSION = "v1"


def lwr_weighted(X_nn: np.ndarray, y_nn: np.ndarray, vec_fit: np.ndarray, dist: np.ndarray) -> float:
    """Как exp101.lwr_approx, но вес считается из переданной dist, а не из
    нормы (X_nn-vec_fit) — позволяет подменить дистанцию (идея 1)."""
    h_bw = max(float(np.max(dist)), 1e-10)
    w = np.exp(-0.5 * (dist / h_bw) ** 2)
    A = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vec_fit @ c[1:])


def isolation_scores(X: np.ndarray, k_lb: int) -> np.ndarray:
    """s_i = расстояние от i до её k_lb-го соседа внутри X (без self)."""
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(D, np.inf)
    k = max(1, min(k_lb, len(X) - 2))
    return np.partition(D, k - 1, axis=1)[:, k - 1]


def run_one_origin(ticker: str, close: np.ndarray, origin: int, n_iter: int, d: int) -> list[dict]:
    m = 3 * d; k_lp = 10 * d; p_fit = m
    p_max = p_fit * (2 ** (exp101.N_LEVELS - 1))
    xi_lwr = 3 * (p_fit + 1) + exp101.XI_EXTRA
    k_lb_d = max(3, min(exp101.K_LB_MAX, xi_lwr - 2))

    base_row = dict(ticker=ticker, n_iter=n_iter, d=d, origin=origin)

    def mkrow(method: str, param, pred, true, ok=1, reason=""):
        r = dict(base_row); r.update(method=method, param=param, pred=pred, true=true,
                                      abs_error=abs(pred - true) if ok else float("nan"),
                                      ok=ok, skip_reason=reason)
        return r

    att = exp101.build_att(close[:origin + 1], m, d, k_lp, n_iter)
    att_ext = exp101.build_att(close[:origin + 2], m, d, k_lp, n_iter)
    if len(att_ext) == 0:
        return [mkrow("base", np.nan, np.nan, np.nan, ok=0, reason="att_ext empty")]
    true_val = float(att_ext[-1])

    n = len(att)
    levels = exp101.levels_aligned(p_fit, p_max)
    p_top = levels[0]
    if n - p_top - 1 < 3:
        return [mkrow("base", np.nan, np.nan, true_val, ok=0, reason="insufficient history")]

    t_arr = np.arange(p_top, n - 1)
    X_full = np.column_stack([att[t_arr - (p_top - 1 - j)] for j in range(p_top)])
    y_base = att[t_arr + 1]
    if len(X_full) < xi_lwr:
        return [mkrow("base", np.nan, np.nan, true_val, ok=0, reason="pool smaller than xi_lwr")]

    vec_full0 = att[-p_top:].copy()
    cands = np.arange(len(X_full))
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

    p_fit_lvl = levels[-1]
    wide_pool = cands
    if len(wide_pool) < xi_lwr + 2:
        return [mkrow("base", np.nan, np.nan, true_val, ok=0, reason="wide pool too small")]

    vec_f = vec_full0[-p_fit_lvl:]
    X_wide = X_full[wide_pool, -p_fit_lvl:]
    raw_dist_wide = np.linalg.norm(X_wide - vec_f, axis=1)
    xi_clip = min(xi_lwr, len(wide_pool))

    rows: list[dict] = []

    # ── baseline: top-xi_lwr по расстоянию ──
    base_idx = np.argpartition(raw_dist_wide, xi_clip - 1)[:xi_clip]
    cands_base = wide_pool[base_idx]
    X_nn_base = X_full[cands_base, -p_fit:]
    y_nn_base = y_base[cands_base]
    dist_base = raw_dist_wide[base_idx]
    pred_base = lwr_weighted(X_nn_base, y_nn_base, vec_f, dist_base)
    rows.append(mkrow("base", np.nan, pred_base, true_val))

    # ── идея 1: мягкое перевзвешивание на ТОМ ЖЕ финальном пуле ──
    s = isolation_scores(X_nn_base, k_lb_d)
    for lam in LAMBDA_GRID:
        d_total = dist_base + lam * s
        pred = lwr_weighted(X_nn_base, y_nn_base, vec_f, d_total)
        rows.append(mkrow("soft", lam, pred, true_val))

    # ── идея 3: отбор по PCA-расстоянию на широком пуле, top-d компонент ──
    mu = X_wide.mean(axis=0)
    k_pca = min(d, X_wide.shape[1] - 1, len(X_wide) - 1)
    _, _, Vt = np.linalg.svd(X_wide - mu, full_matrices=False)
    V_d = Vt[:k_pca].T
    X_proj = (X_wide - mu) @ V_d
    q_proj = (vec_f - mu) @ V_d
    dist_pca = np.linalg.norm(X_proj - q_proj, axis=1)
    pca_idx = np.argpartition(dist_pca, xi_clip - 1)[:xi_clip]
    cands_pca = wide_pool[pca_idx]
    X_nn_pca = X_full[cands_pca, -p_fit:]
    y_nn_pca = y_base[cands_pca]
    dist_pca_sel = raw_dist_wide[pca_idx]  # вес — по сырому расстоянию выбранных точек
    pred_pca = lwr_weighted(X_nn_pca, y_nn_pca, vec_f, dist_pca_sel)
    rows.append(mkrow("pca", np.nan, pred_pca, true_val))

    return rows


_CLOSES: dict[str, np.ndarray] = {}


def _init_worker(closes: dict[str, np.ndarray]) -> None:
    global _CLOSES
    _CLOSES = closes


def _worker(args: tuple) -> list[dict]:
    ticker, origin, n_iter, d = args
    return run_one_origin(ticker, _CLOSES[ticker], origin, n_iter, d)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = TICKERS_TEST if TEST_MODE else TICKERS_FULL
    d_grid = D_GRID_TEST if TEST_MODE else D_GRID_FULL
    n_origins = N_ORIGINS_TEST if TEST_MODE else N_ORIGINS_SCREEN
    suffix = "_test" if TEST_MODE else ""

    print(f"=== {EXPERIMENT_ID}  image={IMAGE_VERSION}  TEST_MODE={TEST_MODE} ===")
    print(f"tickers={tickers}  n_iter={N_ITER_FIXED}  d_grid={d_grid}  n_origins={n_origins}  workers={WORKERS}")

    closes = {t: exp101.load_close(t) for t in tickers}
    tasks = []
    for t in tickers:
        origins = exp101.origins_for(closes[t], n_origins, STEP_WF)
        for d in d_grid:
            for o in origins:
                tasks.append((t, o, N_ITER_FIXED, d))
    print(f"всего origin-задач: {len(tasks)} (каждая даёт {2 + len(LAMBDA_GRID)} строк: base+soft×{len(LAMBDA_GRID)}+pca)")

    out_path = OUT_DIR / f"screen_soft_pca{suffix}.csv"
    cols = ["ticker", "n_iter", "d", "origin", "method", "param", "pred", "true", "abs_error", "ok", "skip_reason"]

    t0 = time.time()
    n_done = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker,
                                  initargs=(closes,)) as ex:
            futures = [ex.submit(_worker, task) for task in tasks]
            for fut in as_completed(futures):
                for row in fut.result():
                    writer.writerow(row)
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
