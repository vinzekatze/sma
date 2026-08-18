"""
111_alpha_scan.py — скрининг мультипликатора alpha в m = max(3, round(alpha · d)).

Проверяем rMAE(alpha) между двумя известными точками:
  alpha≈2 → m≈2d  (как в эксп.109, при больших d)
  alpha=3 → m=3d  (эксп.101, текущий стандарт)

Сетка:
  alpha  ∈ {2.00, 2.25, 2.50, 2.75, 3.00}
  d      ∈ {3, 7, 11, 15}
  n_iter ∈ {2, 3}
  → 40 комбинаций (некоторые (alpha,d) дают одинаковый m — запускаются всё равно,
    дублирование выявляется через колонку m в выходном CSV)

k = 10·d, p_fit = m, N_LEVELS = 4 (×2 octave), xi = 3·(p_fit+1)+5.
acc_ang / global_blend / LP-коррекция — выключены.
LB-диагностика — выключена (скрининг, только rMAE).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

# ── конфигурация ─────────────────────────────────────────────────────────────

EXPERIMENT_ID = "111_alpha_scan"
IMAGE_VERSION = "v1"

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("DATA_DIR", str(ROOT / "data" / "candles")))
OUT_DIR  = Path(os.environ.get("RESULTS_DIR",
                str(Path(__file__).resolve().parent / "results")))
INTERVAL = "1d"

# Скрининговый прогон — 4 тикера, 10 origins.
# Полный прогон (FULL_MODE=1) — 8 тикеров, 40 origins.
TICKERS_SCAN = ["SBER", "MRKP", "CHMF", "NVTK"]
TICKERS_FULL = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
TICKERS_TEST = ["SBER", "CHMF"]

ALPHA_GRID = [2.00, 2.25, 2.50, 2.75, 3.00]
D_GRID     = [3, 7, 11, 15]
NITER_GRID = [2, 3]

N_ORIGINS_SCAN = 10
N_ORIGINS_FULL = 40
N_ORIGINS_TEST = 3
STEP_WF        = 5

N_LEVELS = 4   # уровней каскада ×2 (octave)
XI_EXTRA = 5   # xi_lwr = 3*(p_fit+1) + XI_EXTRA

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
FULL_MODE = os.environ.get("FULL_MODE", "0") == "1"
WORKERS   = int(os.environ.get("WORKERS", str(max(1, (os.cpu_count() or 4) // 2))))


# ── алгоритм (LP фильтр, каскад, LWR) ───────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=float)
    cn = np.arange(1, n + 1, dtype=float)
    ct = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    den = cn * ct2 - ct ** 2
    b = np.where(den > 0, (cn * cty - ct * cy) / den, 0.0)
    a = (cy - b * ct) / cn
    tr = np.exp(a + b * t)
    tr[:2] = close[:2]
    return tr


def lp_proj_signal(ratio: np.ndarray, m: int, d_proj: int,
                   k: int, n_iter: int) -> np.ndarray:
    """Local Projective шумоподавление (Grassberger-Hegger) → diff."""
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
        result = np.zeros(N)
        count  = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i + m] += X_proj[i]
            count[i:i + m]  += 1
        s = result / np.maximum(count, 1)
    return np.diff(s)


def levels_aligned(p_fit: int, p_max: int) -> list[int]:
    """p-aligned каскад ×2: p_fit × {1,2,4,...} ≤ p_max."""
    levs = [p_fit]; p = p_fit
    while p * 2 <= p_max:
        p *= 2; levs.append(p)
    return list(reversed(levs))


def lwr_predict(X_nn: np.ndarray, y_nn: np.ndarray,
                vec_f: np.ndarray, h_bw: float) -> float:
    w = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_f, axis=1) / h_bw) ** 2)
    A = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vec_f @ c[1:])


# ── один прогон ──────────────────────────────────────────────────────────────

@dataclass
class OriginResult:
    ticker: str
    alpha:  float
    d:      int
    n_iter: int
    m:      int
    k_lp:   int
    p_fit:  int
    p_max:  int
    xi_lwr: int
    origin: int
    pred:   float = float("nan")
    true:   float = float("nan")
    ok:     bool  = False
    skip_reason: str = ""


def build_att(close_slice: np.ndarray, m: int, d: int,
              k: int, n_iter: int) -> np.ndarray:
    lt    = logtrend_causal(close_slice)
    ratio = close_slice / np.maximum(lt, 1e-10)
    return lp_proj_signal(ratio, m, d, k, n_iter)


def run_one_origin(ticker: str, close: np.ndarray,
                   origin: int, alpha: float,
                   d: int, n_iter: int) -> OriginResult:
    m      = max(3, round(alpha * d))
    k_lp   = 10 * d
    p_fit  = m
    p_max  = p_fit * (2 ** (N_LEVELS - 1))
    xi_lwr = 3 * (p_fit + 1) + XI_EXTRA

    res = OriginResult(ticker=ticker, alpha=alpha, d=d, n_iter=n_iter,
                       m=m, k_lp=k_lp, p_fit=p_fit, p_max=p_max,
                       xi_lwr=xi_lwr, origin=origin)

    # att без утечки + ground truth (раскрытие одного бара вперёд)
    att     = build_att(close[:origin + 1], m, d, k_lp, n_iter)
    att_ext = build_att(close[:origin + 2], m, d, k_lp, n_iter)
    if len(att_ext) == 0:
        res.skip_reason = "att_ext empty"; return res
    true_val = float(att_ext[-1])

    n      = len(att)
    levels = levels_aligned(p_fit, p_max)
    p_top  = levels[0]
    if n - p_top - 1 < 3:
        res.skip_reason = "insufficient history for p_top"; return res

    t_arr  = np.arange(p_top, n - 1)
    X_full = np.column_stack([att[t_arr - (p_top - 1 - j)] for j in range(p_top)])
    y_base = att[t_arr + 1]
    if len(X_full) < xi_lwr:
        res.skip_reason = "pool smaller than xi_lwr"; return res

    vec_full0 = att[-p_top:].copy()
    cands     = np.arange(len(X_full))

    for k_lvl, p_lvl in enumerate(levels):
        is_last  = (k_lvl == len(levels) - 1)
        xi_clip  = min(xi_lwr, len(cands))
        if len(cands) > xi_clip:
            dists = np.linalg.norm(
                X_full[cands, -p_lvl:] - vec_full0[-p_lvl:], axis=1)
            cands = cands[np.argpartition(dists, xi_clip - 1)[:xi_clip]]
        if not is_last:
            p_next   = levels[k_lvl + 1]
            radius   = p_lvl - p_next
            offsets  = np.arange(radius + 1)
            expanded = cands[:, None] - offsets[None, :]
            expanded = np.clip(expanded, 0, len(X_full) - 1)
            cands    = np.unique(expanded)

    if len(cands) < p_fit + 2:
        res.skip_reason = "final pool too small"; return res

    X_nn  = X_full[cands, -p_fit:]
    y_nn  = y_base[cands]
    vec_f = vec_full0[-p_fit:]
    h_bw  = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
    pred  = lwr_predict(X_nn, y_nn, vec_f, h_bw)

    res.pred = pred
    res.true = true_val
    res.ok   = True
    return res


# ── данные ───────────────────────────────────────────────────────────────────

def load_close(ticker: str) -> np.ndarray:
    p = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw = json.loads(p.read_text())
    candles = raw.get("candles", raw) if isinstance(raw, dict) else raw
    return np.array([c["close"] for c in candles], dtype=float)


def origins_for(close: np.ndarray, n_origins: int, step: int) -> list[int]:
    last  = len(close) - 2   # origin+2 ≤ len-1 для ground truth
    start = last - n_origins * step
    return list(range(max(0, start), last, step))


# ── воркер ───────────────────────────────────────────────────────────────────

_CLOSES: dict[str, np.ndarray] = {}


def _init_worker(closes: dict[str, np.ndarray]) -> None:
    global _CLOSES
    _CLOSES = closes


def _worker(args: tuple) -> dict:
    ticker, origin, alpha, d, n_iter = args
    t0  = time.time()
    res = run_one_origin(ticker, _CLOSES[ticker], origin, alpha, d, n_iter)
    return {"res": res, "dt": time.time() - t0}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if TEST_MODE:
        tickers  = TICKERS_TEST
        n_ori    = N_ORIGINS_TEST
        alpha_g  = [2.00, 3.00]
        d_g      = [3, 11]
        ni_g     = [2]
        suffix   = "_test"
    elif FULL_MODE:
        tickers  = TICKERS_FULL
        n_ori    = N_ORIGINS_FULL
        alpha_g  = ALPHA_GRID
        d_g      = D_GRID
        ni_g     = NITER_GRID
        suffix   = ""
    else:
        tickers  = TICKERS_SCAN
        n_ori    = N_ORIGINS_SCAN
        alpha_g  = ALPHA_GRID
        d_g      = D_GRID
        ni_g     = NITER_GRID
        suffix   = ""

    print(f"=== {EXPERIMENT_ID}  ver={IMAGE_VERSION}  "
          f"TEST={TEST_MODE}  FULL={FULL_MODE} ===")
    print(f"tickers={tickers}  n_ori={n_ori}  workers={WORKERS}")
    print(f"alpha={alpha_g}  d={d_g}  n_iter={ni_g}")

    acc_path  = OUT_DIR / f"forecast_accuracy{suffix}.csv"
    meta_path = OUT_DIR / f"run_meta{suffix}.json"

    meta = {
        "experiment_id": EXPERIMENT_ID, "image_version": IMAGE_VERSION,
        "test_mode": TEST_MODE, "full_mode": FULL_MODE,
        "tickers": tickers, "alpha_grid": alpha_g,
        "d_grid": d_g, "n_iter_grid": ni_g,
        "n_origins": n_ori, "step_wf": STEP_WF,
        "n_levels": N_LEVELS, "xi_extra": XI_EXTRA,
        "k_formula": "10*d", "m_formula": "max(3,round(alpha*d))",
        "workers": WORKERS,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # собираем задачи
    closes: dict[str, np.ndarray] = {}
    tasks: list[tuple] = []
    for ticker in tickers:
        close = load_close(ticker)
        closes[ticker] = close
        origins = origins_for(close, n_ori, STEP_WF)
        for n_iter in ni_g:
            for d in d_g:
                for alpha in alpha_g:
                    for origin in origins:
                        tasks.append((ticker, origin, alpha, d, n_iter))

    n_total = len(tasks)
    print(f"total tasks: {n_total}")

    cols = ["ticker", "alpha", "d", "n_iter", "m", "k_lp", "p_fit",
            "p_max", "xi_lwr", "origin", "pred", "true",
            "error", "abs_error", "ok", "skip_reason"]

    t_start = time.time()
    n_done = n_ok = 0

    with open(acc_path, "w") as f_acc:
        f_acc.write(",".join(cols) + "\n")

        with ProcessPoolExecutor(max_workers=WORKERS,
                                 initializer=_init_worker,
                                 initargs=(closes,)) as ex:
            futures = [ex.submit(_worker, t) for t in tasks]
            for fut in as_completed(futures):
                out = fut.result()
                res: OriginResult = out["res"]
                n_done += 1

                if res.ok:
                    n_ok += 1
                    err = res.pred - res.true
                    row = [res.ticker, res.alpha, res.d, res.n_iter,
                           res.m, res.k_lp, res.p_fit, res.p_max, res.xi_lwr,
                           res.origin, f"{res.pred:.8f}", f"{res.true:.8f}",
                           f"{err:.8f}", f"{abs(err):.8f}", 1, ""]
                else:
                    row = [res.ticker, res.alpha, res.d, res.n_iter,
                           res.m, res.k_lp, res.p_fit, res.p_max, res.xi_lwr,
                           res.origin, "", "", "", "", 0, res.skip_reason]

                f_acc.write(",".join(str(x) for x in row) + "\n")
                f_acc.flush()

                if n_done % 100 == 0 or n_done == n_total:
                    elapsed = time.time() - t_start
                    eta = elapsed / n_done * (n_total - n_done) if n_done < n_total else 0
                    print(f"  {n_done}/{n_total}  ok={n_ok}  "
                          f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    print(f"\nГотово: {n_done} задач, {n_ok} успешных.")
    print(f"  {acc_path}")


if __name__ == "__main__":
    main()
