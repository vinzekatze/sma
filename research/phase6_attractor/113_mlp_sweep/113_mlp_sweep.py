"""
113_mlp_sweep.py — варьирование окна LP-фильтра m_lp при фиксированных d и p_fit.

Стартовые настройки: d=3, p_fit=7 (=2d+1), k_lp=30 (=10d), n_iter=3.
Варьируется только m_lp ∈ {7, 9, 11, 15, 21, 27}.

Гейт-метки (d≥lb_before_q) берутся из diag_levels.csv эксп.109
и джойнятся по (ticker, origin) для разделения анализа.

Вопрос: для origins, прошедших гейт при m_lp=7, помогает ли увеличение m_lp?
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

EXPERIMENT_ID = "113_mlp_sweep"

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("DATA_DIR",  str(ROOT / "data" / "candles")))
OUT_DIR  = Path(os.environ.get("RESULTS_DIR", str(Path(__file__).resolve().parent / "results")))
EXP109_DIR = Path(os.environ.get("EXP109_DIR",
    str(ROOT / "research" / "phase6_attractor" / "109_thin_pfit" / "results")))
INTERVAL = "1d"

TICKERS  = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]

# Фиксированные стартовые настройки
D_FIXED    = 3
P_FIT      = 7      # = 2*D_FIXED + 1
K_LP       = 30     # = 10*D_FIXED
N_ITER_LP  = 3

# Варьируемый параметр: окно LP-фильтра
M_LP_GRID = [7, 9, 11, 15, 21, 27]

N_ORIGINS = 40
STEP_WF   = 5
N_LEVELS  = 4   # ×2 octave cascade
XI_EXTRA  = 5

WORKERS   = int(os.environ.get("WORKERS", str(max(1, (os.cpu_count() or 4) // 2))))
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"


# ── алгоритм (LP фильтр, каскад, LWR) ───────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=float)
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
    s     = ratio.copy().astype(np.float64)
    N     = len(s)
    k_eff = min(k, N - m)
    d_eff = min(d_proj, m - 1)
    for _ in range(n_iter):
        n_pts = N - m + 1
        if n_pts < k_eff + 1:
            break
        rows = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
        X    = s[rows]
        tree = KDTree(X)
        _, inds = tree.query(X, k=k_eff + 1)
        X_proj = np.empty_like(X)
        for i in range(n_pts):
            nn       = inds[i, 1:]
            X_nn     = X[nn]
            centroid = X_nn.mean(axis=0)
            _, _, Vt = np.linalg.svd(X_nn - centroid, full_matrices=False)
            V_d      = Vt[:d_eff].T
            xc       = X[i] - centroid
            X_proj[i] = centroid + V_d @ (V_d.T @ xc)
        result = np.zeros(N)
        count  = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i + m] += X_proj[i]
            count[i:i + m]  += 1
        s = result / np.maximum(count, 1)
    return np.diff(s)


def levels_aligned(p_fit: int, p_max: int) -> list[int]:
    levs = [p_fit]; p = p_fit
    while p * 2 <= p_max:
        p *= 2; levs.append(p)
    return list(reversed(levs))


def lwr_predict(X_nn: np.ndarray, y_nn: np.ndarray,
                vec_f: np.ndarray, h_bw: float) -> float:
    w  = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_f, axis=1) / h_bw) ** 2)
    A  = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vec_f @ c[1:])


# ── один прогон ──────────────────────────────────────────────────────────────

@dataclass
class OriginResult:
    ticker:    str
    m_lp:      int
    origin:    int
    pred:      float = float("nan")
    true:      float = float("nan")
    ok:        bool  = False
    skip_reason: str = ""


def build_att(close_slice: np.ndarray, m_lp: int) -> np.ndarray:
    lt    = logtrend_causal(close_slice)
    ratio = close_slice / np.maximum(lt, 1e-10)
    return lp_proj_signal(ratio, m_lp, D_FIXED, K_LP, N_ITER_LP)


def run_one_origin(ticker: str, close: np.ndarray,
                   origin: int, m_lp: int) -> OriginResult:
    p_fit  = P_FIT
    p_max  = p_fit * (2 ** (N_LEVELS - 1))
    xi_lwr = 3 * (p_fit + 1) + XI_EXTRA

    res = OriginResult(ticker=ticker, m_lp=m_lp, origin=origin)

    att     = build_att(close[:origin + 1], m_lp)
    att_ext = build_att(close[:origin + 2], m_lp)
    if len(att_ext) == 0:
        res.skip_reason = "att_ext empty"; return res
    true_val = float(att_ext[-1])

    n      = len(att)
    levels = levels_aligned(p_fit, p_max)
    p_top  = levels[0]
    if n - p_top - 1 < 3:
        res.skip_reason = "insufficient history"; return res

    t_arr  = np.arange(p_top, n - 1)
    X_full = np.column_stack([att[t_arr - (p_top - 1 - j)] for j in range(p_top)])
    y_base = att[t_arr + 1]
    if len(X_full) < xi_lwr:
        res.skip_reason = "pool < xi_lwr"; return res

    vec_full0 = att[-p_top:].copy()
    cands     = np.arange(len(X_full))

    for k_lvl, p_lvl in enumerate(levels):
        is_last = (k_lvl == len(levels) - 1)
        xi_clip = min(xi_lwr, len(cands))
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
    p   = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw = json.loads(p.read_text())
    c   = raw.get("candles", raw) if isinstance(raw, dict) else raw
    return np.array([x["close"] for x in c], dtype=float)


def origins_for(close: np.ndarray, n_origins: int, step: int) -> list[int]:
    last  = len(close) - 2
    start = last - n_origins * step
    return list(range(max(0, start), last, step))


def load_gate_labels(d: int = D_FIXED, n_iter: int = N_ITER_LP,
                     final_level: int = N_LEVELS - 1) -> dict[tuple, bool]:
    """
    Возвращает {(ticker, origin): gate} из diag_levels.csv эксп.109.
    gate = True если lb_before_q <= d на финальном уровне каскада.
    """
    path = EXP109_DIR / "diag_levels.csv"
    if not path.exists():
        print(f"[предупреждение] не найден {path} — гейт-метки недоступны")
        return {}

    import csv
    labels: dict[tuple, bool] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["n_iter"]) != n_iter:
                continue
            if int(row["d"]) != d:
                continue
            if int(row["level"]) != final_level:
                continue
            ticker = row["ticker"]
            origin = int(row["origin"])
            lb_q   = row["lb_before_q"]
            if lb_q in ("nan", ""):
                gate = False
            else:
                gate = float(lb_q) <= d
            labels[(ticker, origin)] = gate
    return labels


# ── воркер ───────────────────────────────────────────────────────────────────

_CLOSES: dict[str, np.ndarray] = {}


def _init_worker(closes: dict[str, np.ndarray]) -> None:
    global _CLOSES
    _CLOSES = closes


def _worker(args: tuple) -> dict:
    ticker, origin, m_lp = args
    t0  = time.time()
    res = run_one_origin(ticker, _CLOSES[ticker], origin, m_lp)
    return {"res": res, "dt": time.time() - t0}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers   = TICKERS[:2] if TEST_MODE else TICKERS
    n_ori     = 5           if TEST_MODE else N_ORIGINS
    m_lp_grid = [7, 27]     if TEST_MODE else M_LP_GRID
    suffix    = "_test"     if TEST_MODE else ""

    print(f"=== {EXPERIMENT_ID}  TEST={TEST_MODE} ===")
    print(f"D_FIXED={D_FIXED}  P_FIT={P_FIT}  K_LP={K_LP}  N_ITER_LP={N_ITER_LP}")
    print(f"M_LP_GRID={m_lp_grid}")
    print(f"tickers={tickers}  n_ori={n_ori}  workers={WORKERS}")

    # гейт-метки из эксп.109
    gate_labels = load_gate_labels()
    if gate_labels:
        n_gate = sum(gate_labels.values())
        print(f"Гейт-метки из эксп.109: {len(gate_labels)} origins, "
              f"gate=True: {n_gate} ({100*n_gate/len(gate_labels):.0f}%)")
    else:
        print("Гейт-метки не загружены — анализ без разделения по гейту")

    # задачи
    closes: dict[str, np.ndarray] = {}
    tasks: list[tuple] = []
    for ticker in tickers:
        close = load_close(ticker)
        closes[ticker] = close
        for origin in origins_for(close, n_ori, STEP_WF):
            for m_lp in m_lp_grid:
                tasks.append((ticker, origin, m_lp))

    n_total = len(tasks)
    print(f"Всего задач: {n_total}")

    cols = ["ticker", "m_lp", "origin", "pred", "true",
            "error", "abs_error", "ok", "gate", "skip_reason"]

    acc_path  = OUT_DIR / f"results{suffix}.csv"
    meta_path = OUT_DIR / f"run_meta{suffix}.json"

    meta = {
        "experiment_id": EXPERIMENT_ID,
        "d_fixed": D_FIXED, "p_fit": P_FIT, "k_lp": K_LP,
        "n_iter_lp": N_ITER_LP, "m_lp_grid": m_lp_grid,
        "n_levels": N_LEVELS, "xi_extra": XI_EXTRA,
        "tickers": tickers, "n_origins": n_ori, "step_wf": STEP_WF,
        "gate_source": "exp109/diag_levels.csv",
        "gate_criterion": f"lb_before_q <= {D_FIXED} (final level, d={D_FIXED}, n_iter={N_ITER_LP})",
        "workers": WORKERS,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    t_start = time.time()
    n_done = n_ok = 0

    with open(acc_path, "w") as f_out:
        f_out.write(",".join(cols) + "\n")

        with ProcessPoolExecutor(max_workers=WORKERS,
                                 initializer=_init_worker,
                                 initargs=(closes,)) as ex:
            futures = [ex.submit(_worker, t) for t in tasks]
            for fut in as_completed(futures):
                out = fut.result()
                res: OriginResult = out["res"]
                n_done += 1

                gate = gate_labels.get((res.ticker, res.origin), None)
                gate_str = "" if gate is None else ("1" if gate else "0")

                if res.ok:
                    n_ok += 1
                    err = res.pred - res.true
                    row = [res.ticker, res.m_lp, res.origin,
                           f"{res.pred:.8f}", f"{res.true:.8f}",
                           f"{err:.8f}", f"{abs(err):.8f}",
                           1, gate_str, ""]
                else:
                    row = [res.ticker, res.m_lp, res.origin,
                           "", "", "", "", 0, gate_str, res.skip_reason]

                f_out.write(",".join(str(x) for x in row) + "\n")
                f_out.flush()

                if n_done % 200 == 0 or n_done == n_total:
                    elapsed = time.time() - t_start
                    eta = elapsed / n_done * (n_total - n_done) if n_done < n_total else 0
                    print(f"  {n_done}/{n_total}  ok={n_ok}  "
                          f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    print(f"\nГотово: {n_done} задач, {n_ok} успешных.  {acc_path}")


if __name__ == "__main__":
    main()
