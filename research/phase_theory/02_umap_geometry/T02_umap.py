"""
T02_umap.py — UMAP-визуализация геометрии пространства задержек att.

Для каждого (ticker, variant):
  1. att = diff(LP(ratio))          // LP офлайн, вся история
  2. Delay-matrix X: (N-p_emb) × p_emb
  3. LB-оценка локальной размерности в каждой точке
  4. UMAP 2D, seeds=[0, 42, 123]   // проверка устойчивости
  5. Фигура 2×3:
       Row 0: [time | LB | |att|] — seed=0
       Row 1: [seed=0 | seed=42 | seed=123] — time coloring

Variants:
  A: LP(m=9,  d=3,  k=30, n=3), p_emb=9,  umap_nn=15   (стандарт фазы 5)
  B: LP(m=33, d=11, k=30, n=3), p_emb=33, umap_nn=30   (лучший из серии 101)

Выход: results/umap_points.csv, results/figures/umap_{ticker}_{variant}.png
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.spatial import KDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import umap as umap_lib

# ── пути ─────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("DATA_DIR", str(ROOT / "data" / "candles")))
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "results"
FIG_DIR    = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL = "1d"
TICKERS  = ["SBER", "MRKP", "CHMF", "NVTK"]
SEEDS    = [0, 42, 123]
LB_K_MAX = 8

VARIANTS = [
    dict(name="A_d3",     signal="att",    m=9,  d=3,  k=30, n_lp=3, p_emb=9,  umap_nn=15,
         label="A: att=diff(LP(ratio))  p=9"),
    dict(name="B_d11",    signal="att",    m=33, d=11, k=30, n_lp=3, p_emb=33, umap_nn=30,
         label="B: att=diff(LP(ratio))  p=33"),
    dict(name="C_dr_p9",  signal="dratio", m=9,  d=3,  k=30, n_lp=3, p_emb=9,  umap_nn=15,
         label="C: dratio  p=9"),
    dict(name="D_dr_p33", signal="dratio", m=33, d=11, k=30, n_lp=3, p_emb=33, umap_nn=30,
         label="D: dratio  p=33"),
    dict(name="E_r_p9",   signal="ratio",  m=9,  d=3,  k=30, n_lp=3, p_emb=9,  umap_nn=15,
         label="E: ratio  p=9"),
    dict(name="F_r_p33",  signal="ratio",  m=33, d=11, k=30, n_lp=3, p_emb=33, umap_nn=30,
         label="F: ratio  p=33"),
    dict(name="G_att_p144", signal="att", m=9,  d=3,  k=30, n_lp=3, p_emb=144, umap_nn=30,
         label="G: att=diff(LP(ratio))  p=144"),
    dict(name="H_att_p288", signal="att", m=9,  d=3,  k=30, n_lp=3, p_emb=288, umap_nn=50,
         label="H: att=diff(LP(ratio))  p=288"),
    dict(name="I_d11_p144", signal="att", m=33, d=11, k=30, n_lp=3, p_emb=144, umap_nn=30,
         label="I: att=diff(LP(ratio))  d=11 m=33  p=144"),
    dict(name="J_d11_p288", signal="att", m=33, d=11, k=30, n_lp=3, p_emb=288, umap_nn=50,
         label="J: att=diff(LP(ratio))  d=11 m=33  p=288"),
]

# Запускать только эти варианты (None = все)
RUN_ONLY: list[str] | None = None


# ── данные ───────────────────────────────────────────────────────────────────

def load_close(ticker: str) -> np.ndarray:
    p = DATA_DIR / ticker / f"{INTERVAL}.json"
    return np.array([c["close"] for c in json.loads(p.read_text())], dtype=float)


# ── logtrend (causal OLS) ────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n  = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=float)
    cn = np.arange(1, n + 1, dtype=float)
    ct  = np.cumsum(t);   ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc);  cty = np.cumsum(t * lc)
    den = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(den > 0, (cn * cty - ct * cy) / den, 0.0)
    a  = (cy - b * ct) / cn
    tr = np.exp(a + b * t)
    tr[:2] = close[:2]
    return tr


# ── LP-фильтр ────────────────────────────────────────────────────────────────

def lp_smooth(signal: np.ndarray, m: int, d: int, k: int, n_iter: int) -> np.ndarray:
    """Local Projective noise reduction, возвращает сигнал той же длины."""
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


def build_signal(close: np.ndarray, signal: str,
                 m: int = 9, d: int = 3, k: int = 30, n_lp: int = 3) -> np.ndarray:
    """
    signal='att'    → diff(LP(ratio))   стандарт
    signal='dratio' → diff(ratio)        без LP
    signal='ratio'  → ratio              уровень, без LP и без diff
    """
    ratio = close / np.maximum(logtrend_causal(close), 1e-10)
    if signal == "att":
        return np.diff(lp_smooth(ratio, m, d, k, n_lp))
    elif signal == "dratio":
        return np.diff(ratio)
    elif signal == "ratio":
        return ratio
    else:
        raise ValueError(f"Unknown signal: {signal}")


# ── delay embedding ───────────────────────────────────────────────────────────

def delay_matrix(att: np.ndarray, p: int) -> np.ndarray:
    """Матрица задержек: строка i = [att[i], att[i+1], ..., att[i+p-1]]."""
    n_rows = len(att) - p + 1
    idx    = np.arange(n_rows)[:, None] + np.arange(p)[None, :]
    return att[idx]


# ── Levina-Bickel на всех точках ──────────────────────────────────────────────

def lb_per_point(X: np.ndarray, k_max: int = LB_K_MAX) -> np.ndarray:
    """LB MLE локальной размерности для каждой строки X."""
    tree = KDTree(X)
    dists, _ = tree.query(X, k=k_max + 1)   # [0] = self (dist=0)
    rk = dists[:, k_max]                     # расстояние до k_max-го соседа
    rj = dists[:, 1:k_max]                   # расстояния до 1..k_max-1 соседей
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.log(rk[:, None] / np.maximum(rj, 1e-14))
    log_r[~np.isfinite(log_r)] = 0.0
    denom = log_r.sum(axis=1)
    lb    = np.full(len(X), np.nan)
    valid = (rk > 1e-14) & (denom > 1e-14)
    lb[valid] = (k_max - 2) / denom[valid]
    return lb


# ── UMAP ─────────────────────────────────────────────────────────────────────

def run_umap(X: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=seed,
        low_memory=False,
    )
    return reducer.fit_transform(X)


# ── визуализация ──────────────────────────────────────────────────────────────

def _scatter(ax, xy, c, cmap, title, vmin=None, vmax=None):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap,
                    s=3, alpha=0.6, rasterized=True,
                    vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def plot_result(ticker: str, var: dict, att: np.ndarray, X: np.ndarray,
                lb: np.ndarray, embeds: list[np.ndarray]) -> None:
    """Фигура 2×3:
    Row 0 (seed=0): time | LB | |att|
    Row 1 (stability): seed=0 time | seed=42 time | seed=123 time
    """
    e0 = embeds[0]
    t_idx = np.arange(len(X))
    att_last = att[var["p_emb"] - 1:]  # значение att в последней точке каждого вектора
    att_amp  = np.abs(att_last[:len(X)])

    lp_d = var["d"]
    lb_clip = np.clip(lb, 0, lp_d * 3)  # ограничим сверху для цвета

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{ticker}  —  {var['label']}", fontsize=11)

    # Row 0: три окраски seed=0
    sc0 = _scatter(axes[0, 0], e0, t_idx, "viridis", "Время (seed=0)")
    sc1 = _scatter(axes[0, 1], e0, lb_clip, "plasma",
                   f"LB-оценка d (LP_D={lp_d})", vmin=0, vmax=lp_d * 3)
    sc2 = _scatter(axes[0, 2], e0, att_amp, "inferno", "|att|")

    plt.colorbar(sc0, ax=axes[0, 0], fraction=0.03, pad=0.02)
    plt.colorbar(sc1, ax=axes[0, 1], fraction=0.03, pad=0.02)
    plt.colorbar(sc2, ax=axes[0, 2], fraction=0.03, pad=0.02)

    # Row 1: устойчивость — три seed, окраска time
    for col, (seed, emb) in enumerate(zip(SEEDS, embeds)):
        sc = _scatter(axes[1, col], emb, t_idx, "viridis",
                      f"Время (seed={seed})")
        plt.colorbar(sc, ax=axes[1, col], fraction=0.03, pad=0.02)

    fig.tight_layout()
    out = FIG_DIR / f"umap_{ticker}_{var['name']}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → {out.name}", flush=True)


# ── CSV ───────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], append: bool = False) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    out    = OUT_DIR / "umap_points.csv"
    mode   = "a" if (append and out.exists()) else "w"
    with open(out, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            w.writeheader()
        w.writerows(rows)
    print(f"\nCSV {'дополнен' if mode=='a' else 'сохранён'}: {out}  (+{len(rows)} строк)",
          flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0_total = time.time()
    all_rows: list[dict] = []

    variants_to_run = [v for v in VARIANTS
                       if RUN_ONLY is None or v["name"] in RUN_ONLY]

    for ticker in TICKERS:
        print(f"\n{'='*50}", flush=True)
        print(f"Тикер: {ticker}", flush=True)
        close = load_close(ticker)
        print(f"  close: {len(close)} баров", flush=True)

        for var in variants_to_run:
            t0 = time.time()
            print(f"\n  Вариант {var['name']} ({var['label']})", flush=True)

            # пропустить если фигура уже есть
            fig_path = FIG_DIR / f"umap_{ticker}_{var['name']}.png"
            if fig_path.exists():
                print(f"    Пропускаем — фигура уже есть", flush=True)
                continue

            # 1. сигнал
            sig_type = var["signal"]
            print(f"    Сигнал '{sig_type}'...", end=" ", flush=True)
            att = build_signal(close, sig_type, var["m"], var["d"], var["k"], var["n_lp"])
            print(f"len={len(att)}", flush=True)

            # 2. delay matrix
            X = delay_matrix(att, var["p_emb"])
            print(f"    Delay matrix: {X.shape}", flush=True)

            # 3. LB
            print("    LB...", end=" ", flush=True)
            lb = lb_per_point(X, LB_K_MAX)
            lb_finite = lb[np.isfinite(lb)]
            print(f"  median={np.median(lb_finite):.2f}  "
                  f"q05={np.percentile(lb_finite, 5):.2f}  "
                  f"q95={np.percentile(lb_finite, 95):.2f}", flush=True)

            # 4. UMAP × 3 seeds
            embeds = []
            for seed in SEEDS:
                print(f"    UMAP seed={seed}...", end=" ", flush=True)
                t_u = time.time()
                emb = run_umap(X, var["umap_nn"], seed)
                embeds.append(emb)
                print(f"{time.time()-t_u:.1f}s", flush=True)

            # 5. Plot
            plot_result(ticker, var, att, X, lb, embeds)

            # 6. Collect CSV rows
            sig_last = att[var["p_emb"] - 1:]
            for i in range(len(X)):
                row: dict = {
                    "ticker":  ticker,
                    "variant": var["name"],
                    "signal":  var["signal"],
                    "t":       i,
                    "sig_val": float(sig_last[i]) if i < len(sig_last) else float("nan"),
                    "lb_q":    float(lb[i]),
                }
                for si, seed in enumerate(SEEDS):
                    row[f"umap1_s{seed}"] = float(embeds[si][i, 0])
                    row[f"umap2_s{seed}"] = float(embeds[si][i, 1])
                all_rows.append(row)

            print(f"    Вариант готов за {time.time()-t0:.1f}s", flush=True)

    write_csv(all_rows, append=True)
    print(f"\nВсего: {time.time()-t0_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
