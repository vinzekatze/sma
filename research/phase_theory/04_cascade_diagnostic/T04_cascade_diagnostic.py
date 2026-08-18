"""
T04_cascade_diagnostic.py — полная диагностика каскада на одном origin.

Grid: идентичен exp101 GRID_FULL (n_iter ∈ {1,2,3}, d из соответствующих диапазонов).

Для каждой итерации (n_iter, d) запускается p-aligned каскад ×2 (N_LEVELS=4 уровня)
и собирается полная диагностика по каждому уровню. Данные сохраняются в JSON,
фигуры — в results/{TICKER}_{ORIGIN}/.

Запуск:
  cd /home/kali/workspace/apps/sma
  source /home/kali/.venvs/sma/bin/activate

  # дефолт: SBER origin=4772
  python research/phase_theory/04_cascade_diagnostic/T04_cascade_diagnostic.py

  # другой origin/тикер
  python research/phase_theory/04_cascade_diagnostic/T04_cascade_diagnostic.py \\
      --ticker GAZP --origin 4712
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from sklearn.decomposition import PCA

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[warn] umap-learn не установлен — используется PCA вместо UMAP")

# ─── константы ────────────────────────────────────────────────────────────────

# Дефолты — переопределяются аргументами командной строки в main()
TICKER   = "SBER"
ORIGIN   = 4772
INTERVAL = "1d"
N_LEVELS = 4
XI_EXTRA = 5
LP_CLEAN_N_ITER = 1
K_LB_MAX = 8

GRID: dict[int, list[int]] = {
    1: list(range(1, 21)),
    2: list(range(1, 16)),
    3: list(range(1, 11)),
}

PCA_BG_SAMPLE    = 2000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.05

ROOT       = Path(__file__).resolve().parents[3]
DATA_DIR   = Path(os.environ.get("DATA_DIR", str(ROOT / "data" / "candles")))
EXP_DIR    = Path(__file__).resolve().parent
EXP101_CSV = ROOT / "research/phase6_attractor/101_lb_diag_accuracy/results/forecast_accuracy.csv"


def _result_dirs(ticker: str, origin: int) -> tuple[Path, Path]:
    """Возвращает (fig_dir, data_dir) для данного ticker/origin."""
    base = EXP_DIR / "results" / f"{ticker}_{origin}"
    return base / "figures", base / "data"


def _fig_prefix(ticker: str, origin: int) -> str:
    return f"{ticker}_{origin}"

# ─── алгоритм (идентично exp101) ─────────────────────────────────────────────

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


def lp_proj_signal(ratio: np.ndarray, m: int, d_proj: int, k: int,
                   n_iter: int) -> np.ndarray:
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
        from scipy.spatial import KDTree
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
    levs = [p_fit]; p = p_fit
    while p * 2 <= p_max:
        p *= 2; levs.append(p)
    return list(reversed(levs))


def lwr_approx(X_nn: np.ndarray, y_nn: np.ndarray,
               vec_fit: np.ndarray, h_bw: float) -> float:
    w = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_fit, axis=1) / h_bw) ** 2)
    A = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vec_fit @ c[1:])


def lp_clean_set(X: np.ndarray, d: int, k: int, n_iter: int = 1) -> np.ndarray:
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


def build_att(close_slice: np.ndarray, m: int, d: int, k: int,
              n_iter: int) -> np.ndarray:
    lt = logtrend_causal(close_slice)
    ratio = close_slice / np.maximum(lt, 1e-10)
    return lp_proj_signal(ratio, m, d, k, n_iter)


# ─── ядро диагностики ─────────────────────────────────────────────────────────

def run_cascade_diagnostic(close: np.ndarray, origin: int,
                           n_iter: int, d: int) -> dict:
    """Запуск p-aligned каскада с полным сбором данных по каждому уровню."""
    m     = 3 * d
    k_lp  = 10 * d
    p_fit = m
    p_max = p_fit * (2 ** (N_LEVELS - 1))
    xi_lwr = 3 * (p_fit + 1) + XI_EXTRA

    result = {
        "n_iter": n_iter, "d": d, "m": m, "k_lp": k_lp,
        "p_fit": p_fit, "p_max": p_max, "xi_lwr": xi_lwr,
        "origin": origin, "ticker": TICKER,
        "ok": False, "skip_reason": "",
        "pred_val": None, "true_val": None, "abs_error": None,
        "level_data": [], "survival": None,
    }

    # att без утечки
    att = build_att(close[:origin + 1], m, d, k_lp, n_iter)
    att_ext = build_att(close[:origin + 2], m, d, k_lp, n_iter)
    if len(att_ext) == 0:
        result["skip_reason"] = "att_ext empty"; return result
    true_val = float(att_ext[-1])
    result["true_val"] = true_val
    result["att"] = att.tolist()

    n = len(att)
    levels = levels_aligned(p_fit, p_max)
    result["levels"] = levels
    p_top = levels[0]

    if n - p_top - 1 < 3:
        result["skip_reason"] = "history too short"; return result

    t_arr  = np.arange(p_top, n - 1)
    X_full = np.column_stack([att[t_arr - (p_top - 1 - j)] for j in range(p_top)])
    y_base = att[t_arr + 1]
    times_full = t_arr          # time index в att для каждой строки X_full
    result["n_total"] = len(X_full)
    result["times_full"] = times_full.tolist()

    if len(X_full) < xi_lwr:
        result["skip_reason"] = "pool smaller than xi_lwr"; return result

    vec_full0 = att[-p_top:].copy()
    cands = np.arange(len(X_full))

    level_data_list = []

    for k_lvl, p_lvl in enumerate(levels):
        is_last = (k_lvl == len(levels) - 1)

        # -- пул на входе этого уровня --
        cands_entering = cands.copy()
        X_entering = X_full[cands_entering, -p_lvl:]
        x_q = vec_full0[-p_lvl:]

        # расстояния входящих до запроса
        dists_entering = np.linalg.norm(X_entering - x_q, axis=1)

        # -- clip до xi_lwr --
        xi_clip = min(xi_lwr, len(cands))
        if len(cands) > xi_clip:
            sort_idx = np.argpartition(dists_entering, xi_clip - 1)[:xi_clip]
            cands_after_clip = cands_entering[sort_idx]
            dists_after_clip = dists_entering[sort_idx]
        else:
            cands_after_clip = cands_entering.copy()
            dists_after_clip = dists_entering.copy()

        cands = cands_after_clip
        X_after_clip = X_full[cands_after_clip, -p_lvl:]

        # -- LB-диагностика --
        k_lb_d = max(3, min(K_LB_MAX, len(cands_after_clip) - 2))
        lb_before_q, lb_before_n = dim_diag_metrics(
            np.vstack([x_q[None, :], X_after_clip]), k_lb=k_lb_d)

        k_clean = min(k_lp, len(X_after_clip) - 1)
        if k_clean >= 3:
            X_lp = lp_clean_set(X_after_clip, d=d, k=k_clean, n_iter=LP_CLEAN_N_ITER)
            D_q   = np.linalg.norm(X_lp - x_q, axis=1)
            nn_q  = np.argsort(D_q)[:k_clean]
            mu_q  = X_lp[nn_q].mean(0)
            _, _, Vt_q = np.linalg.svd(X_lp[nn_q] - mu_q, full_matrices=False)
            v_q   = x_q - mu_q
            d_eff = min(d, Vt_q.shape[0])
            x_q_lp = mu_q + Vt_q[:d_eff].T @ (Vt_q[:d_eff] @ v_q)
            lb_after_q, lb_after_n = dim_diag_metrics(
                np.vstack([x_q_lp[None, :], X_lp]), k_lb=k_lb_d)
        else:
            X_lp   = X_after_clip.copy()
            x_q_lp = x_q.copy()
            lb_after_q = lb_after_n = float("nan")

        # -- expand для следующего уровня --
        if not is_last:
            p_next  = levels[k_lvl + 1]
            radius  = p_lvl - p_next
            offsets = np.arange(radius + 1)
            expanded = cands[:, None] - offsets[None, :]
            expanded = np.clip(expanded, 0, len(X_full) - 1)
            cands_after_expand = np.unique(expanded)
            cands = cands_after_expand
        else:
            cands_after_expand = None

        lvl_entry = {
            "level"          : k_lvl,
            "p_lvl"          : p_lvl,
            "is_last"        : is_last,
            # размеры пулов
            "n_entering"     : int(len(cands_entering)),
            "n_after_clip"   : int(len(cands_after_clip)),
            "n_after_expand" : int(len(cands_after_expand)) if cands_after_expand is not None else None,
            # индексы (для анализа выживаемости)
            "cands_entering"    : cands_entering.tolist(),
            "cands_after_clip"  : cands_after_clip.tolist(),
            "cands_after_expand": cands_after_expand.tolist() if cands_after_expand is not None else None,
            # временны́е позиции
            "times_entering"  : times_full[cands_entering].tolist(),
            "times_after_clip": times_full[cands_after_clip].tolist(),
            # расстояния
            "dists_entering"  : dists_entering.tolist(),
            "dists_after_clip": dists_after_clip.tolist(),
            "dist_max_entering" : float(dists_entering.max()) if len(dists_entering) else float("nan"),
            "dist_med_clip"     : float(np.median(dists_after_clip)) if len(dists_after_clip) else float("nan"),
            # LB
            "lb_before_q": lb_before_q, "lb_before_n": lb_before_n,
            "lb_after_q" : lb_after_q,  "lb_after_n" : lb_after_n,
            # векторы (для фигур — не сохраняем в JSON, передаём отдельно)
            "_X_entering" : X_entering,
            "_X_after_clip": X_after_clip,
            "_X_lp"       : X_lp,
            "_x_q"        : x_q,
            "_x_q_lp"     : x_q_lp,
        }
        level_data_list.append(lvl_entry)

    # LWR-прогноз (последний cands = cands_after_clip уровня 4)
    final_cands = level_data_list[-1]["cands_after_clip"]
    if len(final_cands) >= p_fit + 2:
        X_nn  = X_full[final_cands, -p_fit:]
        y_nn  = y_base[final_cands]
        vec_f = vec_full0[-p_fit:]
        h_bw  = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
        pred_val = lwr_approx(X_nn, y_nn, vec_f, h_bw)
        result["pred_val"]  = pred_val
        result["abs_error"] = abs(pred_val - true_val)

    result["ok"] = True
    result["level_data"] = level_data_list
    result["_X_full"] = X_full
    result["_times_full_arr"] = times_full

    # анализ выживаемости
    result["survival"] = compute_survival(level_data_list, X_full, times_full)
    return result


# ─── анализ выживаемости ──────────────────────────────────────────────────────

def compute_survival(level_data_list: list[dict],
                     X_full: np.ndarray,
                     times_full: np.ndarray) -> dict:
    """Для каждого финального кандидата: на каком уровне он впервые появился."""
    final_set = set(level_data_list[-1]["cands_after_clip"])
    # after_clip на каждом уровне
    clip_sets = [set(ld["cands_after_clip"]) for ld in level_data_list]

    first_seen: dict[int, int] = {}   # idx → первый уровень где он в after_clip
    for idx in final_set:
        for lvl, cs in enumerate(clip_sets):
            if idx in cs:
                first_seen[idx] = lvl
                break
        else:
            first_seen[idx] = -1    # вошёл только через expand (не через clip)

    counts = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}
    for lvl in first_seen.values():
        counts[lvl] = counts.get(lvl, 0) + 1

    # какие из level-0 after_clip выжили до конца (ядро)
    core = clip_sets[0] & final_set
    # временны́е индексы финальных соседей и ядра
    final_times = times_full[[i for i in final_set]].tolist()
    core_times  = times_full[[i for i in core]].tolist()

    return {
        "n_final"      : len(final_set),
        "n_core"       : len(core),       # выжили из level-0 after_clip
        "n_from_expand": len(final_set) - len(core),
        "first_seen_counts": counts,      # {level_idx: n_candidates}
        "final_times"  : final_times,
        "core_times"   : core_times,
        "first_seen"   : {str(k): v for k, v in first_seen.items()},
    }


# ─── 2D-проекция (PCA или UMAP) ──────────────────────────────────────────────

def _project2d(X: np.ndarray, seed: int = 0,
               force_pca: bool = False) -> tuple[np.ndarray, str]:
    """Возвращает (coords_2d, method_name)."""
    n, p = X.shape
    if n < 4 or p < 2:
        c = np.zeros((n, 2)); c[:, 0] = np.arange(n)
        return c, "trivial"
    if HAS_UMAP and not force_pca and n >= 10:
        nn = min(UMAP_N_NEIGHBORS, n - 1)
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=nn,
                                min_dist=UMAP_MIN_DIST, random_state=seed)
        coords = reducer.fit_transform(X)
        return coords, "UMAP"
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    return coords, "PCA"


def _pca_project(X_fit: np.ndarray, *X_transform: np.ndarray):
    """Fit PCA на X_fit, transform всех переданных массивов."""
    pca = PCA(n_components=2)
    pca.fit(X_fit)
    return [pca.transform(x) for x in X_transform], pca


# ─── визуализация одной итерации ─────────────────────────────────────────────

CMAP_TIME = "plasma"
CMAP_DIST = "viridis"


def _time_colors(times: np.ndarray, t_min: float, t_max: float):
    norm = Normalize(vmin=t_min, vmax=t_max)
    return cm.get_cmap(CMAP_TIME)(norm(times))


def plot_cascade_iteration(res: dict, fig_dir: Path, prefix: str = "") -> None:
    n_iter = res["n_iter"]; d = res["d"]
    levels   = res["levels"]
    att      = np.array(res["att"])
    origin   = res["origin"]
    all_clip_times = [t for ld in res["level_data"]
                      for t in ld["times_after_clip"]]
    t_min = int(min(all_clip_times)) if all_clip_times else 0
    t_max = int(max(all_clip_times)) if all_clip_times else 1
    if t_max <= t_min:
        t_max = t_min + 1

    n_rows = len(levels) + 1   # уровни + строка-сводка
    n_cols = 5
    fig = plt.figure(figsize=(32, 6 * n_rows))
    fig.suptitle(
        f"Cascade diagnostic  |  SBER origin={origin} (2026-04-07)  "
        f"|  n_iter={n_iter}  d={d}  m={3*d}  p_fit={3*d}  p_max={res['p_max']}  "
        f"xi_lwr={res['xi_lwr']}\n"
        f"levels={levels}   "
        f"pred={res['pred_val']:.6f}  true={res['true_val']:.6f}  "
        f"abs_err={res.get('abs_error', float('nan')):.6f}",
        fontsize=11, y=0.995
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.97, bottom=0.04)

    times_full = np.array(res["_times_full_arr"])

    for row, ld in enumerate(res["level_data"]):
        p_lvl       = ld["p_lvl"]
        X_entering  = ld["_X_entering"]
        X_clip      = ld["_X_after_clip"]
        X_lp        = ld["_X_lp"]
        x_q         = ld["_x_q"]
        x_q_lp      = ld["_x_q_lp"]
        t_ent       = np.array(ld["times_entering"])
        t_clip      = np.array(ld["times_after_clip"])
        d_ent       = np.array(ld["dists_entering"])
        d_clip      = np.array(ld["dists_after_clip"])
        lvl_label   = f"L{row+1}  p={p_lvl}  n_in={ld['n_entering']}→clip={ld['n_after_clip']}"
        if not ld["is_last"]:
            lvl_label += f"→exp={ld['n_after_expand']}"

        # ── col 0: PCA полного входящего пула + after_clip ─────────────────────
        ax = fig.add_subplot(gs[row, 0])
        ax.set_title(f"{lvl_label}\nPCA entering pool", fontsize=8)

        # subsample фона для скорости
        bg_idx = (np.random.choice(len(X_entering), min(PCA_BG_SAMPLE, len(X_entering)),
                                   replace=False)
                  if len(X_entering) > PCA_BG_SAMPLE else np.arange(len(X_entering)))
        X_bg = X_entering[bg_idx]
        # map clip к индексам внутри entering для PCA
        cands_ent_set = {int(c): i for i, c in enumerate(ld["cands_entering"])}
        clip_local = np.array([cands_ent_set[int(c)] for c in ld["cands_after_clip"]
                               if int(c) in cands_ent_set])

        X_fit_pca = np.vstack([X_bg, x_q[None, :]])
        [coords_ent, coords_clip_full, q_coord], pca_obj = _pca_project(
            X_fit_pca,
            X_entering,
            X_clip,
            x_q[None, :]
        )
        bg_coords = coords_ent[bg_idx]
        ax.scatter(bg_coords[:, 0], bg_coords[:, 1], s=4, alpha=0.25,
                   color="gray", zorder=1, label="entering (sample)")
        sc = ax.scatter(coords_clip_full[:, 0], coords_clip_full[:, 1],
                        s=20, c=t_clip, cmap=CMAP_TIME,
                        vmin=t_min, vmax=t_max, zorder=3, label="after_clip")
        ax.scatter(q_coord[0, 0], q_coord[0, 1], s=180, marker="*",
                   color="red", zorder=5, label="query")
        ax.set_xlabel("PC1", fontsize=7); ax.set_ylabel("PC2", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(sc, ax=ax, label="time", pad=0.01).ax.tick_params(labelsize=6)

        # ── col 1: UMAP/PCA after_clip + LP-clean overlay ──────────────────────
        ax = fig.add_subplot(gs[row, 1])
        X_proj_input = np.vstack([X_clip, x_q[None, :], X_lp, x_q_lp[None, :]])
        coords_all, method = _project2d(X_proj_input, seed=0)
        n_cl = len(X_clip)
        c_clip = coords_all[:n_cl]
        c_q    = coords_all[n_cl:n_cl+1]
        c_lp   = coords_all[n_cl+1:n_cl+1+len(X_lp)]
        c_qlp  = coords_all[n_cl+1+len(X_lp):]

        ax.set_title(f"after_clip {method} + LP-clean", fontsize=8)
        sc2 = ax.scatter(c_clip[:, 0], c_clip[:, 1], s=25,
                         c=t_clip, cmap=CMAP_TIME,
                         vmin=t_min, vmax=t_max, zorder=3, label="raw")
        ax.scatter(c_lp[:, 0], c_lp[:, 1], s=25, marker="x",
                   c=t_clip, cmap=CMAP_TIME, vmin=t_min, vmax=t_max,
                   zorder=4, label="LP-clean", alpha=0.7)
        # соединяем пары (raw → LP)
        for i in range(min(len(c_clip), len(c_lp))):
            ax.plot([c_clip[i, 0], c_lp[i, 0]],
                    [c_clip[i, 1], c_lp[i, 1]],
                    color="gray", alpha=0.25, lw=0.5, zorder=2)
        ax.scatter(c_q[0, 0], c_q[0, 1], s=180, marker="*",
                   color="red", zorder=6, label="query")
        ax.scatter(c_qlp[0, 0], c_qlp[0, 1], s=100, marker="P",
                   color="darkred", zorder=6, label="query LP")
        ax.tick_params(labelsize=6)
        plt.colorbar(sc2, ax=ax, label="time", pad=0.01).ax.tick_params(labelsize=6)

        # ── col 2: временно́й гистограмм after_clip ────────────────────────────
        ax = fig.add_subplot(gs[row, 2])
        ax.set_title(f"Temporal dist (after_clip)", fontsize=8)
        n_bins = min(60, max(10, len(t_clip) // 3))
        ax.hist(t_clip, bins=n_bins, color="#4a90d9", edgecolor="none", alpha=0.8)
        ax.axvline(origin, color="red", lw=1.5, ls="--", label=f"origin={origin}")
        if len(times_full):
            ax.set_xlim(times_full.min(), times_full.max())
        ax.set_xlabel("time index (att)", fontsize=7)
        ax.set_ylabel("count", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

        # ── col 3: распределение расстояний ───────────────────────────────────
        ax = fig.add_subplot(gs[row, 3])
        ax.set_title(f"Distances to query", fontsize=8)
        parts = ax.violinplot([d_ent, d_clip],
                              positions=[1, 2], showmedians=True, widths=0.6)
        colors = ["#aaaaaa", "#4a90d9"]
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col); pc.set_alpha(0.7)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["entering", "after_clip"], fontsize=7)
        ax.set_ylabel(f"L2 dist in {p_lvl}D", fontsize=7)
        ax.tick_params(labelsize=6)
        # медианы как текст
        ax.text(1, np.median(d_ent) * 1.02, f"{np.median(d_ent):.3f}",
                ha="center", fontsize=6, color="#555555")
        ax.text(2, np.median(d_clip) * 1.02, f"{np.median(d_clip):.3f}",
                ha="center", fontsize=6, color="#1a5fa0")

        # ── col 4: LB метрики ─────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 4])
        ax.set_title(f"Levina-Bickel  (K_LB_MAX={K_LB_MAX})", fontsize=8)
        labels = ["lb_bef_q", "lb_bef_n", "lb_aft_q", "lb_aft_n"]
        vals   = [ld["lb_before_q"], ld["lb_before_n"],
                  ld["lb_after_q"],  ld["lb_after_n"]]
        colors_lb = ["#e07b54", "#f0b080", "#5b9ccc", "#8ac4e8"]
        bars = ax.bar(labels, [v if np.isfinite(v) else 0 for v in vals],
                      color=colors_lb, edgecolor="none")
        ax.axhline(d, color="green", lw=1.2, ls="--", label=f"d={d}")
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                        f"{val:.2f}", ha="center", fontsize=6)
        ax.set_ylabel("LB estimate", fontsize=7)
        ax.tick_params(axis="x", labelsize=6, rotation=25)
        ax.tick_params(axis="y", labelsize=6)
        ax.legend(fontsize=6)

    # ── строка-сводка ─────────────────────────────────────────────────────────
    row_s = len(levels)

    # col 0: потоки пула по уровням
    ax = fig.add_subplot(gs[row_s, 0])
    ax.set_title("Pool flow per level", fontsize=8)
    x = np.arange(len(levels))
    n_ent  = [ld["n_entering"]    for ld in res["level_data"]]
    n_clip = [ld["n_after_clip"]  for ld in res["level_data"]]
    n_exp  = [ld["n_after_expand"] or 0 for ld in res["level_data"]]
    w = 0.28
    ax.bar(x - w, n_ent,  width=w, label="entering", color="#aaaaaa")
    ax.bar(x,     n_clip, width=w, label="after_clip", color="#4a90d9")
    ax.bar(x + w, n_exp,  width=w, label="after_expand", color="#f0b080")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i+1}\np={levels[i]}" for i in range(len(levels))], fontsize=7)
    ax.set_ylabel("n candidates", fontsize=7)
    ax.set_yscale("log")
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

    # col 1: анализ выживаемости
    ax = fig.add_subplot(gs[row_s, 1])
    ax.set_title("Survival analysis (final pool origin)", fontsize=8)
    surv = res["survival"]
    cnt  = surv["first_seen_counts"]
    keys = sorted(k for k in cnt if cnt[k] > 0)
    lbls = {0: "1st clip\n(core)", 1: "expand→L2\nclip", 2: "expand→L3\nclip",
            3: "expand→L4\nclip", -1: "no clip"}
    bar_labels = [lbls.get(k, str(k)) for k in keys]
    bar_vals   = [cnt[k] for k in keys]
    colors_s   = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#95a5a6"]
    ax.bar(bar_labels, bar_vals, color=colors_s[:len(keys)], edgecolor="none")
    for i, v in enumerate(bar_vals):
        ax.text(i, v + 0.2, str(v), ha="center", fontsize=8)
    ax.set_ylabel("n in final pool", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"Survival: core={surv['n_core']}  expand={surv['n_from_expand']}"
        f"  total={surv['n_final']}", fontsize=8)

    # col 2: траектория LB
    ax = fig.add_subplot(gs[row_s, 2])
    ax.set_title("LB trajectory across levels", fontsize=8)
    level_nums = [ld["level"] + 1 for ld in res["level_data"]]
    for key, label, color in [
        ("lb_before_q", "bef_q", "#e07b54"),
        ("lb_before_n", "bef_n", "#f0b080"),
        ("lb_after_q",  "aft_q", "#5b9ccc"),
        ("lb_after_n",  "aft_n", "#8ac4e8"),
    ]:
        vals = [ld[key] for ld in res["level_data"]]
        finite = [(l, v) for l, v in zip(level_nums, vals) if np.isfinite(v)]
        if finite:
            ls, vs = zip(*finite)
            ax.plot(ls, vs, marker="o", label=label, color=color)
    ax.axhline(d, color="green", lw=1.2, ls="--", label=f"d={d}")
    ax.set_xlabel("cascade level", fontsize=7)
    ax.set_ylabel("LB", fontsize=7)
    ax.set_xticks(level_nums)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

    # col 3: траектория медианного расстояния
    ax = fig.add_subplot(gs[row_s, 3])
    ax.set_title("Median distance to query (clip pool)", fontsize=8)
    med_dists = [ld["dist_med_clip"] for ld in res["level_data"]]
    ax.plot(level_nums, med_dists, marker="s", color="#4a90d9")
    for l, v in zip(level_nums, med_dists):
        ax.text(l, v, f"  {v:.4f}", fontsize=7, va="bottom")
    ax.set_xlabel("cascade level", fontsize=7)
    ax.set_ylabel(f"median L2 (at level p)", fontsize=7)
    ax.set_xticks(level_nums)
    ax.tick_params(labelsize=6)

    # col 4: att сигнал + origin + прогноз vs истина
    ax = fig.add_subplot(gs[row_s, 4])
    ax.set_title("att signal  (origin window)", fontsize=8)
    att = np.array(res["att"])
    win = 200
    start = max(0, origin - win)
    t_range = np.arange(start, len(att))
    ax.plot(t_range, att[start:], color="#4a90d9", lw=0.8, label="att")
    ax.axvline(origin, color="red", lw=1.2, ls="--", label=f"origin")
    if res["pred_val"] is not None:
        ax.scatter([origin + 1], [res["pred_val"]],
                   s=80, marker="^", color="orange", zorder=5, label="pred")
        ax.scatter([origin + 1], [res["true_val"]],
                   s=80, marker="v", color="green", zorder=5, label="true")
    ax.set_xlabel("att index", fontsize=7)
    ax.set_ylabel("att value", fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

    p = f"{prefix}_" if prefix else ""
    fname = fig_dir / f"{p}iter_n{n_iter}_d{d:02d}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── drill-down фигура ────────────────────────────────────────────────────────

def plot_drilldown(res: dict, fig_dir: Path, prefix: str = "") -> None:
    """PCA уровня 1 с цветовым кодированием выживаемости по уровням."""
    n_iter = res["n_iter"]; d = res["d"]
    X_full = res["_X_full"]
    times_full = res["_times_full_arr"]
    levels = res["levels"]
    p_top  = levels[0]

    X_l1 = X_full[:, -p_top:]   # весь пул в представлении уровня 1
    x_q  = res["level_data"][0]["_x_q"]
    X_fit = np.vstack([X_l1, x_q[None, :]])

    # PCA (быстрее UMAP для 4000+ точек)
    n_bg = min(PCA_BG_SAMPLE * 2, len(X_l1))
    bg_idx = np.random.choice(len(X_l1), n_bg, replace=False)
    pca = PCA(n_components=2)
    pca.fit(X_l1[bg_idx])
    coords_all = pca.transform(X_l1)
    q_coord    = pca.transform(x_q[None, :])

    # метки выживаемости: 0 = не в L1 clip, 1 = выжил L1, 2 = L2, 3 = L3, 4 = L4 final
    survival_level = np.zeros(len(X_l1), dtype=int)
    for lvl_i, ld in enumerate(res["level_data"]):
        for idx in ld["cands_after_clip"]:
            if survival_level[idx] == 0:
                survival_level[idx] = lvl_i + 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f"Drill-down PCA  |  SBER origin={res['origin']}  "
        f"n_iter={n_iter}  d={d}  p_top={p_top}\n"
        f"Color = deepest cascade level reached  "
        f"(0=not selected, 1=L1 only, ..., 4=final LWR pool)",
        fontsize=11
    )

    surv_colors = {0: "#dddddd", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c", 4: "#2ecc71"}
    surv_labels = {0: "not selected", 1: "L1 only", 2: "→L2",
                   3: "→L3", 4: "→L4 (final)"}
    surv_sizes  = {0: 2, 1: 10, 2: 15, 3: 20, 4: 30}
    zorder_map  = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

    for ax, sample in zip(axes, [True, False]):
        title_sfx = " (subsample bg)" if sample else " (full)"
        ax.set_title(f"Survival map{title_sfx}", fontsize=9)
        for lv in sorted(surv_colors.keys()):
            mask = survival_level == lv
            if not mask.any():
                continue
            c_sub = coords_all[mask]
            if sample and lv == 0:
                idx = np.random.choice(len(c_sub), min(500, len(c_sub)), replace=False)
                c_sub = c_sub[idx]
            ax.scatter(c_sub[:, 0], c_sub[:, 1],
                       s=surv_sizes[lv], color=surv_colors[lv],
                       alpha=0.6 if lv > 0 else 0.2,
                       label=f"{surv_labels[lv]} ({mask.sum()})",
                       zorder=zorder_map[lv])
        ax.scatter(q_coord[0, 0], q_coord[0, 1], s=250, marker="*",
                   color="blue", zorder=10, label="query")
        ax.set_xlabel("PC1 (p_top space)", fontsize=8)
        ax.set_ylabel("PC2 (p_top space)", fontsize=8)
        ax.legend(fontsize=7, markerscale=1.5)
        ax.tick_params(labelsize=7)

        # временны́е метки на финальных соседях
        final_mask = survival_level == 4
        if final_mask.any():
            t_final = times_full[final_mask]
            norm = Normalize(vmin=t_final.min(), vmax=t_final.max())
            fc = coords_all[final_mask]
            sc = ax.scatter(fc[:, 0], fc[:, 1], s=45,
                            c=t_final, cmap=CMAP_TIME, zorder=6,
                            edgecolors="black", linewidths=0.3)
            plt.colorbar(sc, ax=ax, label="time of final neighbors", pad=0.01
                         ).ax.tick_params(labelsize=6)

    p = f"{prefix}_" if prefix else ""
    fname = fig_dir / f"{p}drilldown_n{n_iter}_d{d:02d}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── сводная фигура по сетке ──────────────────────────────────────────────────

def plot_grid_summary(all_results: list[dict], fig_dir: Path,
                      exp101_acc: dict | None, prefix: str = "") -> None:
    """По каждому n_iter: как метрики меняются с d."""
    from itertools import groupby

    n_iters = sorted({r["n_iter"] for r in all_results})
    for ni in n_iters:
        sub = sorted([r for r in all_results if r["n_iter"] == ni and r["ok"]],
                     key=lambda r: r["d"])
        if not sub:
            continue
        ds = [r["d"] for r in sub]

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle(
            f"Grid summary  |  SBER origin={ORIGIN}  n_iter={ni}\n"
            f"x-axis = d,  каждая точка — одна итерация каскада",
            fontsize=12
        )

        def _get(key: str, lvl: int | None = None):
            vals = []
            for r in sub:
                if lvl is None:
                    vals.append(r.get(key))
                else:
                    ld = r["level_data"][lvl] if lvl < len(r["level_data"]) else None
                    vals.append(ld[key] if ld else None)
            return vals

        # (0,0) abs_error vs d
        ax = axes[0, 0]
        ax.set_title("abs_error (forecast)", fontsize=9)
        errs = _get("abs_error")
        ax.plot(ds, errs, marker="o", color="#e74c3c")
        if exp101_acc:
            ref = [exp101_acc.get((ni, d)) for d in ds]
            ax.plot(ds, ref, marker="s", ls="--", color="#999999", label="exp101 (same origin)")
            ax.legend(fontsize=7)
        ax.set_xlabel("d"); ax.set_ylabel("abs_error"); ax.tick_params(labelsize=7)
        ax.set_yscale("log")

        # (0,1) LB_before_q финального уровня
        ax = axes[0, 1]
        ax.set_title("LB before q — final level", fontsize=9)
        vals = _get("lb_before_q", lvl=-1)
        ax.plot(ds, vals, marker="o", color="#e07b54")
        ax.plot(ds, ds, ls="--", color="green", label="d (target)")
        ax.legend(fontsize=7)
        ax.set_xlabel("d"); ax.set_ylabel("LB_before_q"); ax.tick_params(labelsize=7)

        # (0,2) LB_after_q финального уровня
        ax = axes[0, 2]
        ax.set_title("LB after q — final level", fontsize=9)
        vals = _get("lb_after_q", lvl=-1)
        ax.plot(ds, vals, marker="o", color="#5b9ccc")
        ax.plot(ds, ds, ls="--", color="green", label="d (target)")
        ax.legend(fontsize=7)
        ax.set_xlabel("d"); ax.set_ylabel("LB_after_q"); ax.tick_params(labelsize=7)

        # (1,0) n_core / n_final
        ax = axes[1, 0]
        ax.set_title("Core fraction (survived from L1 clip)", fontsize=9)
        core_frac = [r["survival"]["n_core"] / max(r["survival"]["n_final"], 1)
                     for r in sub]
        ax.plot(ds, core_frac, marker="o", color="#2ecc71")
        ax.set_ylim(0, 1); ax.set_xlabel("d"); ax.set_ylabel("n_core / n_final")
        ax.tick_params(labelsize=7)

        # (1,1) медианное расстояние каждого уровня vs d
        ax = axes[1, 1]
        ax.set_title("Median distance to query per level", fontsize=9)
        colors_lvl = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
        for li in range(4):
            vals = []
            for r in sub:
                ld = r["level_data"][li] if li < len(r["level_data"]) else None
                vals.append(ld["dist_med_clip"] if ld else None)
            ax.plot(ds, vals, marker="o", label=f"L{li+1}",
                    color=colors_lvl[li % len(colors_lvl)])
        ax.set_xlabel("d"); ax.set_ylabel("median L2"); ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        # (1,2) n_entering каждого уровня vs d
        ax = axes[1, 2]
        ax.set_title("Pool sizes (entering each level)", fontsize=9)
        for li in range(4):
            vals = [r["level_data"][li]["n_entering"] if li < len(r["level_data"]) else None
                    for r in sub]
            ax.plot(ds, vals, marker="o", label=f"L{li+1}",
                    color=colors_lvl[li % len(colors_lvl)])
        ax.set_xlabel("d"); ax.set_ylabel("n_entering"); ax.legend(fontsize=7)
        ax.set_yscale("log"); ax.tick_params(labelsize=7)

        # (2,0) LB_before_q по всем уровням vs d (heatmap-like)
        ax = axes[2, 0]
        ax.set_title("LB_before_q: trajectory L1→L4", fontsize=9)
        for li in range(4):
            vals = [r["level_data"][li]["lb_before_q"] if li < len(r["level_data"]) else float("nan")
                    for r in sub]
            ax.plot(ds, vals, marker="o", label=f"L{li+1}",
                    color=colors_lvl[li % len(colors_lvl)])
        ax.plot(ds, ds, ls="--", color="green", label="d")
        ax.set_xlabel("d"); ax.set_ylabel("LB_before_q"); ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        # (2,1) drop LB (L1 bef_q → L4 bef_q) vs d
        ax = axes[2, 1]
        ax.set_title("LB drop L1→L4 (bef_q)", fontsize=9)
        drops = []
        for r in sub:
            l0 = r["level_data"][0]["lb_before_q"]
            lf = r["level_data"][-1]["lb_before_q"]
            drops.append(l0 - lf if np.isfinite(l0) and np.isfinite(lf) else float("nan"))
        ax.plot(ds, drops, marker="o", color="#9b59b6")
        ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.set_xlabel("d"); ax.set_ylabel("LB drop (L1-L4)"); ax.tick_params(labelsize=7)

        # (2,2) d≥LB gate: доля уровней где lb_before_q ≤ d
        ax = axes[2, 2]
        ax.set_title("d≥LB_before_q gate  (fraction of levels)", fontsize=9)
        gate_fracs = []
        for r in sub:
            ld_list = r["level_data"]
            frac = sum(1 for ld in ld_list
                       if np.isfinite(ld["lb_before_q"]) and r["d"] >= ld["lb_before_q"]
                       ) / len(ld_list)
            gate_fracs.append(frac)
        ax.plot(ds, gate_fracs, marker="o", color="#1abc9c")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("d"); ax.set_ylabel("fraction levels d≥LB"); ax.tick_params(labelsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        p = f"{prefix}_" if prefix else ""
        fname = fig_dir / f"{p}summary_n{ni}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)


# ─── загрузка данных ──────────────────────────────────────────────────────────

def load_close(ticker: str) -> np.ndarray:
    p = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw = json.loads(p.read_text())
    return np.array([c["close"] for c in raw], dtype=float)


def load_exp101_acc(n_iter_vals: list[int], d_vals_per_ni: dict,
                    origin: int, ticker: str) -> dict:
    """Загружает abs_error из exp101 для заданного origin/ticker."""
    if not EXP101_CSV.exists():
        return {}
    import csv
    ref: dict[tuple, float] = {}
    with open(EXP101_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["ticker"] == ticker and int(row["origin"]) == origin
                    and row["ok"] == "1"):
                key = (int(row["n_iter"]), int(row["d"]))
                ref[key] = float(row["abs_error"])
    return ref


# ─── JSON-сериализация (без numpy-arrays) ─────────────────────────────────────

def to_json_safe(obj):
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            result[str(k)] = to_json_safe(v)
        return result
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="T04 — полная диагностика каскада на одном origin")
    parser.add_argument("--ticker", default="SBER",
                        help="Тикер (default: SBER)")
    parser.add_argument("--origin", type=int, default=4772,
                        help="Индекс origin в массиве close (default: 4772)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    origin = args.origin
    prefix = _fig_prefix(ticker, origin)
    fig_dir, data_out = _result_dirs(ticker, origin)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_out.mkdir(parents=True, exist_ok=True)

    print(f"=== T04_cascade_diagnostic  |  {ticker} origin={origin} ===")
    print(f"HAS_UMAP={HAS_UMAP}  N_LEVELS={N_LEVELS}  XI_EXTRA={XI_EXTRA}")
    print(f"Output: {fig_dir.parent}")
    print(f"Grid: {sum(len(v) for v in GRID.values())} iterations total")
    print()

    close = load_close(ticker)
    if origin >= len(close) - 1:
        raise ValueError(f"origin={origin} выходит за пределы close (len={len(close)})")
    print(f"Loaded {ticker}: {len(close)} bars, origin={origin} (close={close[origin]:.2f})")
    print()

    exp101_ref = load_exp101_acc(list(GRID.keys()), GRID, origin, ticker)
    print(f"exp101 reference: {len(exp101_ref)} entries loaded")

    all_results: list[dict] = []
    t_global = time.time()
    n_total = sum(len(ds) for ds in GRID.values())
    n_done  = 0

    for n_iter, ds in sorted(GRID.items()):
        for d in ds:
            t0 = time.time()
            res = run_cascade_diagnostic(close, origin, n_iter, d)
            t_run = time.time() - t0

            status = "ok" if res["ok"] else f"SKIP:{res['skip_reason']}"
            err_str = f"abs_err={res['abs_error']:.6f}" if res["abs_error"] is not None else ""
            print(f"  n_iter={n_iter} d={d:2d}  m={3*d:3d}  p_fit={3*d:3d}  "
                  f"p_max={res['p_max']:4d}  {status}  {err_str}  [{t_run:.1f}s]")

            if res["ok"]:
                t1 = time.time()
                plot_cascade_iteration(res, fig_dir, prefix=prefix)
                plot_drilldown(res, fig_dir, prefix=prefix)
                t_fig = time.time() - t1

                json_data = to_json_safe(res)
                (data_out / f"{prefix}_cascade_n{n_iter}_d{d:02d}.json").write_text(
                    json.dumps(json_data, indent=2, ensure_ascii=False)
                )
                all_results.append(res)
                print(f"          figures saved  [{t_fig:.1f}s]")

            n_done += 1
            elapsed = time.time() - t_global
            eta = elapsed / n_done * (n_total - n_done)
            print(f"          progress {n_done}/{n_total}  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    print("\nGenerating grid summary figures...")
    plot_grid_summary(all_results, fig_dir, exp101_ref, prefix=prefix)

    index = [
        {"n_iter": r["n_iter"], "d": r["d"], "m": r["m"],
         "p_fit": r["p_fit"], "p_max": r["p_max"], "xi_lwr": r["xi_lwr"],
         "levels": r["levels"], "ok": r["ok"],
         "abs_error": r["abs_error"],
         "n_total": r.get("n_total"),
         "survival": {k: v for k, v in r["survival"].items()
                      if k not in ("final_times", "core_times", "first_seen")}
                      if r.get("survival") else None}
        for r in all_results
    ]
    (data_out / f"{prefix}_index.json").write_text(
        json.dumps({"ticker": ticker, "origin": origin, "runs": index}, indent=2)
    )

    total = time.time() - t_global
    print(f"\nГотово: {len(all_results)}/{n_total} успешных за {total/60:.1f} мин.")
    print(f"Фигуры: {fig_dir}")
    print(f"Данные: {data_out}")


if __name__ == "__main__":
    main()
