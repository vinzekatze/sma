"""
T01_collect.py — суррогатный тест: LP создаёт структуру или находит существующую?

Prism G: TwoNN d_L на att = LP(signal), полная история тикера.
Prism L: LWR rMAE walk-forward, 20 origins, honest per-origin LP (без утечки).

Сигналы:
  ratio  → att_ratio  = diff(lp_smooth(ratio))
  dratio → att_dratio = lp_smooth(diff(ratio))

Суррогаты: shuffle, ft, aaft, block (block_size=20).

Протокол без утечки: LP пересчитывается на close[:origin+1] под каждый origin.
true_att берётся из оригинального close[:origin+2] (не из суррогата).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import json
import time
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.spatial import KDTree

# ── пути и конфигурация ───────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("DATA_DIR", str(ROOT / "data" / "candles")))
OUT_DIR  = Path(os.environ.get("RESULTS_DIR",
               str(Path(__file__).resolve().parent / "results")))
INTERVAL = "1d"

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
WORKERS   = int(os.environ.get("WORKERS", str(max(1, (os.cpu_count() or 4)))))

TICKERS_FULL = ["SBER", "MRKP", "CHMF", "NVTK"]
TICKERS_TEST = ["SBER", "MRKP"]
SIGNALS      = ["ratio", "dratio"]
SURR_TYPES_FULL = ["shuffle", "ft", "aaft", "block"]
SURR_TYPES_TEST = ["shuffle", "ft"]

BLOCK_SIZE = 20

LP_M, LP_D, LP_K, LP_N = 9, 3, 30, 3
P_FIT, P_MAX = 9, 144
XI_LWR  = 3 * (P_FIT + 1) + 5   # = 35
P_EMB   = LP_M                   # для TwoNN delay-embedding
K_LB_MAX = 8

N_SURR_GEOM_FULL = 99
N_SURR_LWR_FULL  = 19
N_SURR_GEOM_TEST = 5
N_SURR_LWR_TEST  = 5
N_ORIGINS_FULL = 20
N_ORIGINS_TEST = 5
STEP_WF = 5

EXPERIMENT_ID = "T01_surrogate_test"
IMAGE_VERSION = "v1"


# ── загрузка данных ───────────────────────────────────────────────────────────

def load_close(ticker: str) -> np.ndarray:
    p = DATA_DIR / ticker / f"{INTERVAL}.json"
    return np.array([c["close"] for c in json.loads(p.read_text())], dtype=float)


def origins_for(close: np.ndarray, n: int, step: int) -> list[int]:
    last = len(close) - 2
    return list(range(max(0, last - n * step), last, step))


# ── logtrend ─────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=float)
    cn = np.arange(1, n + 1, dtype=float)
    ct = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    den = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(den > 0, (cn * cty - ct * cy) / den, 0.0)
    a = (cy - b * ct) / cn
    tr = np.exp(a + b * t)
    tr[:2] = close[:2]
    return tr


# ── LP-фильтр ─────────────────────────────────────────────────────────────────

def lp_smooth(signal: np.ndarray, m: int, d: int, k: int, n_iter: int) -> np.ndarray:
    """Local Projective noise reduction без финального diff."""
    s     = signal.copy().astype(np.float64)
    N     = len(s)
    k_eff = min(k, N - m)
    d_eff = min(d, m - 1)
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
            centroid = X_nn.mean(0)
            try:
                _, _, Vt = np.linalg.svd(X_nn - centroid, full_matrices=False)
            except np.linalg.LinAlgError:
                _, _, Vt = scipy.linalg.svd(
                    X_nn - centroid, full_matrices=False, lapack_driver="gesvd")
            V_d = Vt[:d_eff].T
            xc  = X[i] - centroid
            X_proj[i] = centroid + V_d @ (V_d.T @ xc)
        result = np.zeros(N)
        count  = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i + m] += X_proj[i]
            count[i:i + m]  += 1
        s = result / np.maximum(count, 1)
    return s


def _ratio(close_slice: np.ndarray) -> np.ndarray:
    return close_slice / np.maximum(logtrend_causal(close_slice), 1e-10)


def build_att(close_slice: np.ndarray, signal_type: str) -> np.ndarray:
    r = _ratio(close_slice)
    if signal_type == "ratio":
        return np.diff(lp_smooth(r, LP_M, LP_D, LP_K, LP_N))
    else:  # dratio
        return lp_smooth(np.diff(r), LP_M, LP_D, LP_K, LP_N)


def build_att_surr(close_slice: np.ndarray, signal_type: str,
                   surr_type: str, rng: np.random.Generator) -> np.ndarray:
    r = _ratio(close_slice)
    raw = r if signal_type == "ratio" else np.diff(r)
    surr = _make_surrogate(raw, surr_type, rng)
    if signal_type == "ratio":
        return np.diff(lp_smooth(surr, LP_M, LP_D, LP_K, LP_N))
    else:
        return lp_smooth(surr, LP_M, LP_D, LP_K, LP_N)


# ── суррогаты ─────────────────────────────────────────────────────────────────

def _make_surrogate(sig: np.ndarray, surr_type: str,
                    rng: np.random.Generator) -> np.ndarray:
    if surr_type == "shuffle":
        return rng.permutation(sig)
    if surr_type == "ft":
        return _ft_surr(sig, rng)
    if surr_type == "aaft":
        return _aaft_surr(sig, rng)
    if surr_type == "block":
        return _block_surr(sig, BLOCK_SIZE, rng)
    raise ValueError(surr_type)


def _ft_surr(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n      = len(sig)
    ft     = np.fft.rfft(sig)
    phases = rng.uniform(0.0, 2 * np.pi, len(ft))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(np.abs(ft) * np.exp(1j * phases), n=n).real


def _aaft_surr(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n           = len(sig)
    sorted_vals = np.sort(sig)
    gauss       = rng.standard_normal(n)
    gauss_s     = np.sort(gauss)
    rank        = np.argsort(np.argsort(sig))
    gauss_match = gauss_s[rank]
    ft          = _ft_surr(gauss_match, rng)
    rank_ft     = np.argsort(np.argsort(ft))
    return sorted_vals[rank_ft]


def _block_surr(sig: np.ndarray, bs: int, rng: np.random.Generator) -> np.ndarray:
    n        = len(sig)
    n_blocks = n // bs
    blocks   = [sig[i * bs:(i + 1) * bs] for i in range(n_blocks)]
    order    = rng.permutation(n_blocks)
    parts    = [blocks[i] for i in order]
    tail     = sig[n_blocks * bs:]
    return np.concatenate(parts + ([tail] if len(tail) > 0 else []))


# ── TwoNN ─────────────────────────────────────────────────────────────────────

def twonn_dim(att: np.ndarray, p_emb: int = P_EMB,
              n_sample: int = 500, rng_seed: int = 0) -> float:
    n_att = len(att)
    if n_att < p_emb + 3:
        return float("nan")
    n_rows = n_att - p_emb + 1
    if n_rows < 3:
        return float("nan")
    rows = np.arange(n_rows)[:, None] + np.arange(p_emb)[None, :]
    X    = att[rows]
    rng  = np.random.default_rng(rng_seed)
    idx  = rng.choice(n_rows, size=min(n_sample, n_rows), replace=False)
    mu_list = []
    for i in idx:
        d = np.linalg.norm(X - X[i], axis=1)
        d[i] = np.inf
        ds = np.sort(d)
        r1, r2 = ds[0], ds[1]
        if r1 > 1e-12 and r2 > r1:
            mu_list.append(np.log(r2 / r1))
    if not mu_list:
        return float("nan")
    mu = float(np.mean(mu_list))
    return float(np.log(2) / mu) if mu > 1e-10 else float("nan")


# ── p-aligned каскад ──────────────────────────────────────────────────────────

def _levels(p_fit: int, p_max: int) -> list[int]:
    ls = [p_fit]
    p  = p_fit
    while p * 2 <= p_max:
        p *= 2
        ls.append(p)
    return list(reversed(ls))


# ── Levina-Bickel ─────────────────────────────────────────────────────────────

def _lb_q(X_all: np.ndarray, k_max: int = K_LB_MAX) -> float:
    """LB MLE для X_all[0]=запрос, X_all[1:]=соседи."""
    n = len(X_all)
    k = max(3, min(k_max, n - 2))
    diff = X_all[:, None, :] - X_all[None, :, :]
    D    = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(D, np.inf)
    D_s  = np.sort(D, axis=1)
    rk   = D_s[0, k - 1]
    rj   = D_s[0, :k - 1]
    if rk <= 1e-14 or np.any(rj <= 1e-14):
        return float("nan")
    return float((k - 2) / np.sum(np.log(rk / rj)))


# ── LWR ──────────────────────────────────────────────────────────────────────

def _lwr(X_nn: np.ndarray, y_nn: np.ndarray, vec_f: np.ndarray) -> float:
    d   = np.linalg.norm(X_nn - vec_f, axis=1)
    h   = max(float(d.max()), 1e-10)
    w   = np.exp(-0.5 * (d / h) ** 2)
    A   = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw  = np.sqrt(np.maximum(w, 1e-30))
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vec_f @ c[1:])


# ── ядро Prism G ──────────────────────────────────────────────────────────────

def _geom(close: np.ndarray, signal: str,
          surr_type: str, surr_idx: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    att = (build_att(close, signal)
           if surr_idx == 0
           else build_att_surr(close, signal, surr_type, rng))
    d_L = twonn_dim(att, rng_seed=seed + 1)
    return {"signal": signal,
            "surrogate_type": "original" if surr_idx == 0 else surr_type,
            "surr_idx": surr_idx,
            "d_L": d_L}


# ── ядро Prism L ──────────────────────────────────────────────────────────────

def _lwr_origin(close: np.ndarray, origin: int, signal: str,
                surr_type: str, surr_idx: int, seed: int) -> dict:
    base = {"signal": signal,
            "surrogate_type": "original" if surr_idx == 0 else surr_type,
            "surr_idx": surr_idx, "origin": origin,
            "abs_error": float("nan"), "ok": False,
            "lb_q_final": float("nan")}

    # ground truth всегда из оригинала
    att_ext = build_att(close[:origin + 2], signal)
    if len(att_ext) == 0:
        return base
    true_val = float(att_ext[-1])

    # библиотека: оригинал или суррогат
    rng = np.random.default_rng(seed)
    att = (build_att(close[:origin + 1], signal)
           if surr_idx == 0
           else build_att_surr(close[:origin + 1], signal, surr_type, rng))

    n      = len(att)
    levels = _levels(P_FIT, P_MAX)
    p_top  = levels[0]
    if n - p_top - 1 < 3:
        return base

    t_arr  = np.arange(p_top, n - 1)
    X_full = np.column_stack([att[t_arr - (p_top - 1 - j)] for j in range(p_top)])
    y_base = att[t_arr + 1]
    if len(X_full) < XI_LWR:
        return base

    vec0  = att[-p_top:].copy()
    cands = np.arange(len(X_full))
    lb_q_final = float("nan")

    for k_lvl, p_lvl in enumerate(levels):
        is_last  = (k_lvl == len(levels) - 1)
        xi_clip  = min(XI_LWR, len(cands))
        if len(cands) > xi_clip:
            dists = np.linalg.norm(
                X_full[cands, -p_lvl:] - vec0[-p_lvl:], axis=1)
            cands = cands[np.argpartition(dists, xi_clip - 1)[:xi_clip]]

        if is_last:
            x_q     = vec0[-p_lvl:]
            X_neigh = X_full[cands, -p_lvl:]
            k_lb    = max(3, min(K_LB_MAX, len(cands) - 2))
            lb_q_final = _lb_q(np.vstack([x_q[None, :], X_neigh]), k_lb)

        if not is_last:
            p_next   = levels[k_lvl + 1]
            offsets  = np.arange(p_lvl - p_next + 1)
            expanded = np.clip(
                cands[:, None] - offsets[None, :], 0, len(X_full) - 1)
            cands = np.unique(expanded)

    if len(cands) < P_FIT + 2:
        return base

    X_nn  = X_full[cands, -P_FIT:]
    y_nn  = y_base[cands]
    vec_f = vec0[-P_FIT:]
    pred  = _lwr(X_nn, y_nn, vec_f)

    base["abs_error"]   = float(abs(pred - true_val))
    base["ok"]          = True
    base["lb_q_final"]  = float(lb_q_final) if np.isfinite(lb_q_final) else float("nan")
    return base


# ── воркеры (ProcessPoolExecutor) ────────────────────────────────────────────

_CLOSES: dict[str, np.ndarray] = {}


def _init(closes: dict) -> None:
    global _CLOSES
    _CLOSES = closes


def _w_geom(args: tuple) -> tuple[str, dict]:
    ticker, signal, surr_type, surr_idx, seed = args
    r = _geom(_CLOSES[ticker], signal, surr_type, surr_idx, seed)
    return ticker, r


def _w_lwr(args: tuple) -> tuple[str, dict]:
    ticker, origin, signal, surr_type, surr_idx, seed = args
    r = _lwr_origin(_CLOSES[ticker], origin, signal, surr_type, surr_idx, seed)
    return ticker, r


# ── генерация задач ────────────────────────────────────────────────────────────

def _make_tasks(tickers, closes, n_surr_g, n_surr_l, n_origins, surr_types):
    geom_tasks, lwr_tasks = [], []
    for tk in tickers:
        for sig in SIGNALS:
            # G: original
            geom_tasks.append((tk, sig, "original", 0, 42))
            # G: surrogates
            for st in surr_types:
                for si in range(1, n_surr_g + 1):
                    seed = abs(hash((tk, sig, st, si))) % (2 ** 31)
                    geom_tasks.append((tk, sig, st, si, seed))
        # L tasks
        origins = origins_for(closes[tk], n_origins, STEP_WF)
        for orig in origins:
            for sig in SIGNALS:
                # L: original
                lwr_tasks.append((tk, orig, sig, "original", 0, 42))
                # L: surrogates
                for st in surr_types:
                    for si in range(1, n_surr_l + 1):
                        seed = abs(hash((tk, orig, sig, st, si))) % (2 ** 31)
                        lwr_tasks.append((tk, orig, sig, st, si, seed))
    return geom_tasks, lwr_tasks


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    tickers    = TICKERS_TEST    if TEST_MODE else TICKERS_FULL
    surr_types = SURR_TYPES_TEST if TEST_MODE else SURR_TYPES_FULL
    n_surr_g   = N_SURR_GEOM_TEST if TEST_MODE else N_SURR_GEOM_FULL
    n_surr_l   = N_SURR_LWR_TEST  if TEST_MODE else N_SURR_LWR_FULL
    n_origins  = N_ORIGINS_TEST   if TEST_MODE else N_ORIGINS_FULL

    print(f"[{EXPERIMENT_ID}] {IMAGE_VERSION}  TEST={TEST_MODE}  WORKERS={WORKERS}",
          flush=True)
    print(f"tickers={tickers}  signals={SIGNALS}  surr_types={surr_types}",
          flush=True)
    print(f"n_surr_geom={n_surr_g}  n_surr_lwr={n_surr_l}  n_origins={n_origins}",
          flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    closes = {tk: load_close(tk) for tk in tickers}
    geom_tasks, lwr_tasks = _make_tasks(
        tickers, closes, n_surr_g, n_surr_l, n_origins, surr_types)
    n_total = len(geom_tasks) + len(lwr_tasks)
    print(f"Tasks: {len(geom_tasks)} geometry + {len(lwr_tasks)} LWR = {n_total}",
          flush=True)

    geom_csv = OUT_DIR / "geometry.csv"
    lwr_csv  = OUT_DIR / "lwr.csv"
    gh = ["ticker", "signal", "surrogate_type", "surr_idx", "d_L"]
    lh = ["ticker", "signal", "surrogate_type", "surr_idx", "origin",
          "abs_error", "ok", "lb_q_final"]

    n_done = 0
    t0 = time.time()

    with open(geom_csv, "w", newline="") as gf, open(lwr_csv, "w", newline="") as lf:
        gw = csv.writer(gf); gw.writerow(gh); gf.flush()
        lw = csv.writer(lf); lw.writerow(lh); lf.flush()

        with ProcessPoolExecutor(
                max_workers=WORKERS,
                initializer=_init,
                initargs=(closes,)) as pool:

            futures: dict = {}
            for t in geom_tasks:
                futures[pool.submit(_w_geom, t)] = ("G", t[0])
            for t in lwr_tasks:
                futures[pool.submit(_w_lwr, t)]  = ("L", t[0])

            for fut in as_completed(futures):
                kind, ticker = futures[fut]
                try:
                    _, res = fut.result()
                except Exception as exc:
                    print(f"ERROR {kind} {ticker}: {exc}", flush=True)
                    n_done += 1
                    continue

                if kind == "G":
                    gw.writerow([ticker, res["signal"], res["surrogate_type"],
                                 res["surr_idx"], res["d_L"]])
                    gf.flush()
                else:
                    lw.writerow([ticker, res["signal"], res["surrogate_type"],
                                 res["surr_idx"], res["origin"],
                                 res["abs_error"], res["ok"], res["lb_q_final"]])
                    lf.flush()

                n_done += 1
                if n_done % 200 == 0 or n_done == n_total:
                    elapsed = time.time() - t0
                    eta = (n_total - n_done) / (n_done / elapsed) if n_done else 0
                    print(f"[{n_done}/{n_total}] {elapsed:.0f}s  ETA {eta:.0f}s",
                          flush=True)

    meta = {
        "experiment_id": EXPERIMENT_ID, "image_version": IMAGE_VERSION,
        "run_date": datetime.datetime.now().isoformat(),
        "test_mode": TEST_MODE, "workers": WORKERS,
        "tickers": tickers, "signals": SIGNALS, "surrogate_types": surr_types,
        "lp": {"m": LP_M, "d": LP_D, "k": LP_K, "n": LP_N},
        "cascade": {"p_fit": P_FIT, "p_max": P_MAX, "xi_lwr": XI_LWR},
        "twonn": {"p_emb": P_EMB, "n_sample": 500},
        "block_size": BLOCK_SIZE,
        "n_surr_geom": n_surr_g, "n_surr_lwr": n_surr_l,
        "n_origins": n_origins, "step_wf": STEP_WF,
        "n_tasks_geom": len(geom_tasks), "n_tasks_lwr": len(lwr_tasks),
        "n_done": n_done,
    }
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. {n_done}/{n_total} tasks in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
