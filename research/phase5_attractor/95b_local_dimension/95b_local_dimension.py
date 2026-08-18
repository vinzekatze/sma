"""
EXPERIMENT_ID : 95b_local_dimension
VERSION       : 1.0
ФАЗА          : 5 — исследование аттрактора

Оценщики локальной размерности аттрактора att.
Проверяет, коррелируют ли d̂_LB, d̂_TwoNN, d̂_PCA с oracle_p из 95a.

Источник oracle_p: ../95_oracle_distribution/results/origins_SBER_full.jsonl

Запуск:
  python 95b_local_dimension.py --mode test   # ~2-5 мин
  python 95b_local_dimension.py --mode full   # ~10 мин
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist

# ── Идентификаторы ────────────────────────────────────────────────────────────

EXPERIMENT_ID = "95b_local_dimension"
VERSION       = "1.0"

# ── Пути ──────────────────────────────────────────────────────────────────────

_HERE    = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR",    str(_HERE.parent.parent.parent / "data" / "candles")))
OUT_DIR  = Path(os.environ.get("RESULTS_DIR", str(_HERE / "results")))
FIG_DIR  = Path(os.environ.get("FIGURES_DIR", str(_HERE / "figures")))

# Откуда берём oracle_p
PATH_95A = Path(os.environ.get(
    "PATH_95A",
    str(_HERE.parent / "95_oracle_distribution" / "results" / "origins_SBER_full.jsonl"),
))

INTERVAL = "1d"
P_MAX    = 300
LP_M, LP_D, LP_K, LP_N = 9, 3, 30, 3

# ── Сетка параметров оценщиков ────────────────────────────────────────────────

P_SEARCH_VALS = [20, 40, 80]   # размерность поиска соседей для оценщика
K_DIM_VALS    = [50, 100, 200] # количество соседей для оценщика

# ── Конфигурация режимов ──────────────────────────────────────────────────────

MODES: dict[str, dict] = {
    "test": dict(
        ticker    = "SBER",
        n_origins = 10,
        stride    = 50,  # каждый 50-й из 500 origins 95a
        desc      = "ТЕСТОВЫЙ ПРОГОН  (10 origins)",
    ),
    "full": dict(
        ticker    = "SBER",
        n_origins = 200,
        stride    = 2,   # каждый 2-й из 500 origins 95a → 250, обрезаем до 200
        desc      = "ПОЛНЫЙ ПРОГОН    (200 origins)",
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# LP-пайплайн (идентичен 95a)
# ══════════════════════════════════════════════════════════════════════════════

def _logtrend(close: np.ndarray) -> np.ndarray:
    n  = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);    ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc);   cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a  = (cy - b * ct) / cn
    tr = np.exp(a + b * t)
    tr[:2] = close[:2]
    return tr


def _lp_proj(ratio: np.ndarray, m: int, d: int, k: int, n_iter: int) -> np.ndarray:
    from scipy.spatial import KDTree
    s = ratio.copy().astype(np.float64)
    N = len(s)
    k_eff = min(k, N - m)
    d_eff = min(d, m - 1)
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
            nn     = inds[i, 1:]
            X_nn   = X[nn]
            cen    = X_nn.mean(axis=0)
            _, _, Vt = np.linalg.svd(X_nn - cen, full_matrices=False)
            V_d    = Vt[:d_eff].T
            xc     = X[i] - cen
            X_proj[i] = cen + V_d @ (V_d.T @ xc)
        result = np.zeros(N)
        count  = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i + m] += X_proj[i]
            count[i:i + m]  += 1
        s = result / np.maximum(count, 1)
    return np.diff(s)


def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend(close)
    ratio = close / np.maximum(lt, 1e-10)
    return _lp_proj(ratio, LP_M, LP_D, LP_K, LP_N)


# ══════════════════════════════════════════════════════════════════════════════
# Контекст origin (матрицы библиотеки — идентично 95a)
# ══════════════════════════════════════════════════════════════════════════════

class _Ctx:
    __slots__ = ("X_full", "vec_full", "n_lib", "ok")

    def __init__(self, att: np.ndarray) -> None:
        self.ok = False
        n = len(att)
        if n - P_MAX - 1 < 3:
            return
        att_wins    = sliding_window_view(att[:-1], P_MAX)
        self.X_full = np.asarray(att_wins[:n - P_MAX - 1])
        self.n_lib  = len(self.X_full)
        self.vec_full = att[n - P_MAX:n].copy()
        self.ok = True


# ══════════════════════════════════════════════════════════════════════════════
# Оценщики локальной размерности
# ══════════════════════════════════════════════════════════════════════════════

def _dim_lb(sorted_dists: np.ndarray) -> float:
    """Levina-Bickel MLE: d̂ = (K-2) / Σlog(r_K / r_i).
    sorted_dists: расстояния до K соседей, по возрастанию, без нулей.
    """
    k = len(sorted_dists)
    if k < 3:
        return np.nan
    log_r = np.log(sorted_dists[-1] / sorted_dists[:-1])
    valid  = log_r[log_r > 1e-12]
    if len(valid) < 2:
        return np.nan
    return float((k - 2) / np.sum(log_r))


def _dim_twonn(X_nn: np.ndarray) -> float:
    """Local TwoNN (Facco 2017) на K соседях.
    Для каждой из K точек находим её 2 ближайших соседей внутри X_nn.
    d̂ = log(2) / E[log(r2/r1)].
    """
    K = len(X_nn)
    if K < 3:
        return np.nan
    D = cdist(X_nn, X_nn, "euclidean")
    np.fill_diagonal(D, np.inf)
    sorted_D = np.sort(D, axis=1)
    r1 = sorted_D[:, 0]
    r2 = sorted_D[:, 1]
    valid = (r1 > 1e-12) & (r2 > r1 * 1.0001)
    if valid.sum() < 2:
        return np.nan
    log_ratios = np.log(r2[valid] / r1[valid])
    return float(np.log(2) / np.mean(log_ratios))


def _dim_pca(X_nn: np.ndarray, threshold: float = 0.90) -> float:
    """Число главных компонент, объясняющих >= threshold дисперсии."""
    if len(X_nn) < 2:
        return np.nan
    X_c = X_nn - X_nn.mean(axis=0)
    if np.allclose(X_c, 0):
        return 1.0
    _, s, _ = np.linalg.svd(X_c, full_matrices=False)
    var = s ** 2
    total = var.sum()
    if total < 1e-20:
        return 1.0
    cumvar = np.cumsum(var) / total
    n_comp = int(np.searchsorted(cumvar, threshold)) + 1
    return float(n_comp)


def estimate_dims(ctx: _Ctx, p_search: int, k_dim: int) -> dict[str, float]:
    """Вычисляет d̂_LB, d̂_TwoNN, d̂_PCA для заданных (p_search, k_dim)."""
    q = ctx.vec_full[-p_search:]
    X = ctx.X_full[:, -p_search:]            # (N_lib, p_search)

    dists    = np.linalg.norm(X - q, axis=1)
    k_use    = min(k_dim, ctx.n_lib - 1)
    if k_use < 3:
        return {"lb": np.nan, "twonn": np.nan, "pca": np.nan, "k_used": k_use}

    idx      = np.argpartition(dists, k_use)[:k_use]
    s_dists  = np.sort(dists[idx])
    X_nn     = X[idx]

    # Исключаем нулевые расстояния (крайне редко, но возможно)
    nonzero  = s_dists > 1e-12
    s_valid  = s_dists[nonzero]

    return {
        "lb":     _dim_lb(s_valid),
        "twonn":  _dim_twonn(X_nn),
        "pca":    _dim_pca(X_nn),
        "k_used": k_use,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка oracle_p из 95a
# ══════════════════════════════════════════════════════════════════════════════

def load_oracle_map(path: Path) -> dict[int, int]:
    """Возвращает {t_orig: oracle_p} из JSONL 95a."""
    oracle_map: dict[int, int] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        true_att = r["true_att"]
        per_p    = r["per_p"]
        preds    = {
            int(p): per_p[p]["pred"]
            for p in per_p
            if per_p[p].get("pred") is not None
        }
        if not preds:
            continue
        oracle_p = min(preds.keys(), key=lambda p: abs(preds[p] - true_att))
        oracle_map[int(r["t_orig"])] = oracle_p
    return oracle_map


def select_origins(oracle_map: dict[int, int], n_origins: int, stride: int) -> list[int]:
    """Подвыборка t_orig из oracle_map по хронологии."""
    all_t = sorted(oracle_map.keys())
    selected = all_t[::stride][:n_origins]
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward
# ══════════════════════════════════════════════════════════════════════════════

def run_ticker(
    ticker: str,
    att_full: np.ndarray,
    t_origins: list[int],
    oracle_map: dict[int, int],
    out_file: Path,
) -> None:
    n_total = len(att_full)

    for idx_o, t_orig in enumerate(t_origins):
        if t_orig + 1 >= n_total:
            continue
        t0 = time.time()

        hist = att_full[: t_orig + 1]
        ctx  = _Ctx(hist)
        if not ctx.ok:
            print(f"    origin {t_orig}: пропущен (короткий ряд)")
            continue

        oracle_p = oracle_map[t_orig]

        # Вычисляем d̂ для всех (p_search, k_dim)
        dims: dict[str, dict[str, dict]] = {}
        for ps in P_SEARCH_VALS:
            dims[str(ps)] = {}
            for kd in K_DIM_VALS:
                result = estimate_dims(ctx, ps, kd)
                dims[str(ps)][str(kd)] = {
                    k: (None if (v is not None and np.isnan(v)) else v)
                    for k, v in result.items()
                }

        record = {
            "exp_id":    EXPERIMENT_ID,
            "version":   VERSION,
            "ticker":    ticker,
            "t_orig":    int(t_orig),
            "oracle_p":  oracle_p,
            "dims":      dims,
        }

        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        elapsed = time.time() - t0
        print(
            f"    origin {idx_o + 1}/{len(t_origins)}  t={t_orig}"
            f"  oracle_p={oracle_p}"
            f"  [{elapsed:.2f}с]",
            flush=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Null test (shuffled att)
# ══════════════════════════════════════════════════════════════════════════════

def run_null_test(att: np.ndarray, t_orig: int, rng: np.random.Generator) -> dict:
    att_shuf = att[: t_orig + 1].copy()
    rng.shuffle(att_shuf)
    ctx = _Ctx(att_shuf)
    if not ctx.ok:
        return {}
    ps, kd = 40, 100
    result = estimate_dims(ctx, ps, kd)
    return {
        "t_orig":   t_orig,
        "p_search": ps,
        "k_dim":    kd,
        **{k: (None if (v is not None and np.isnan(v)) else v) for k, v in result.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Анализ: корреляции и фигуры
# ══════════════════════════════════════════════════════════════════════════════

def analyse(jsonl_path: Path, null_path: Path | None, mode: str) -> dict:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not rows:
        print("Нет данных.")
        return {}

    oracle_ps = [r["oracle_p"] for r in rows]
    oracle_arr = np.array(oracle_ps, dtype=float)

    # Матрица корреляций: (method, p_search, k_dim) → r
    methods = ["lb", "twonn", "pca"]
    corr_matrix: dict[tuple[str, int, int], dict] = {}

    for ps in P_SEARCH_VALS:
        for kd in K_DIM_VALS:
            per_method: dict[str, list[float]] = {m: [] for m in methods}
            valid_mask  = []
            for r in rows:
                blk = r["dims"].get(str(ps), {}).get(str(kd), {})
                vals = {m: blk.get(m) for m in methods}
                if any(v is None for v in vals.values()):
                    valid_mask.append(False)
                else:
                    valid_mask.append(True)
                    for m in methods:
                        per_method[m].append(vals[m])

            vm = np.array(valid_mask)
            oa = oracle_arr[vm]

            for m in methods:
                da = np.array(per_method[m])
                if len(da) < 3 or np.std(da) < 1e-10:
                    r_val, p_val = np.nan, np.nan
                else:
                    r_val = float(np.corrcoef(da, oa)[0, 1])
                    # Приближённый p-value через t-тест
                    n = len(da)
                    t = r_val * np.sqrt((n - 2) / max(1 - r_val**2, 1e-12))
                    from scipy.stats import t as t_dist
                    p_val = float(2 * t_dist.sf(abs(t), df=n - 2))

                corr_matrix[(m, ps, kd)] = {
                    "r":      r_val,
                    "r2":     r_val ** 2 if not np.isnan(r_val) else np.nan,
                    "p_val":  p_val,
                    "n":      int(vm.sum()),
                    "d_mean": float(np.mean(per_method[m])) if per_method[m] else np.nan,
                    "d_std":  float(np.std(per_method[m]))  if per_method[m] else np.nan,
                }

    # Печать таблицы
    print(f"\n{'─'*72}")
    print(f"  {'Метод':<8} {'p_search':>8} {'k_dim':>6}  {'r':>7}  {'r²':>7}  {'p_val':>8}  d_mean")
    print(f"{'─'*72}")
    for (m, ps, kd), v in sorted(corr_matrix.items()):
        r_str  = f"{v['r']:+.4f}" if not np.isnan(v['r']) else "   nan"
        r2_str = f"{v['r2']:.4f}" if not np.isnan(v.get('r2', np.nan)) else "  nan"
        pv_str = f"{v['p_val']:.4f}" if not np.isnan(v.get('p_val', np.nan)) else "   nan"
        dm_str = f"{v['d_mean']:.2f}" if not np.isnan(v['d_mean']) else "  nan"
        print(f"  {m:<8} {ps:>8} {kd:>6}  {r_str}  {r2_str}  {pv_str}  {dm_str}")
    print(f"{'─'*72}")

    # Лучшая комбинация
    best_key = max(
        corr_matrix,
        key=lambda k: corr_matrix[k].get("r2", -np.inf) if not np.isnan(corr_matrix[k].get("r2", np.nan)) else -np.inf,
    )
    best = corr_matrix[best_key]
    print(f"\n  Лучшая: method={best_key[0]}  p_search={best_key[1]}  k_dim={best_key[2]}")
    print(f"  r={best['r']:+.4f}  r²={best['r2']:.4f}  p_val={best['p_val']:.4f}")

    # Null test
    null_results: list[dict] = []
    if null_path and null_path.exists():
        null_results = [json.loads(l) for l in null_path.read_text().splitlines() if l.strip()]
        if null_results:
            null_lb  = np.nanmean([r.get("lb") for r in null_results if r.get("lb") is not None])
            null_twn = np.nanmean([r.get("twonn") for r in null_results if r.get("twonn") is not None])
            null_pca = np.nanmean([r.get("pca") for r in null_results if r.get("pca") is not None])
            print(f"\n  Null test (shuffled att, p_search=40, k_dim=100):")
            print(f"    d̂_LB={null_lb:.2f}  d̂_TwoNN={null_twn:.2f}  d̂_PCA={null_pca:.2f}")
            print(f"    (ожидалось ≈ 40.0 при отсутствии структуры)")

    # Сохранение summary
    summary = {
        "exp_id":       EXPERIMENT_ID,
        "version":      VERSION,
        "mode":         mode,
        "n_origins":    len(rows),
        "oracle_p_mean": float(np.mean(oracle_ps)),
        "corr_matrix":  {
            f"{m}|{ps}|{kd}": v
            for (m, ps, kd), v in corr_matrix.items()
        },
        "best": {
            "method":   best_key[0],
            "p_search": best_key[1],
            "k_dim":    best_key[2],
            **best,
        },
        "null_test": null_results[:5] if null_results else [],
    }
    out_path = OUT_DIR / f"summary_{mode}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Сводка → {out_path}")

    _plot_heatmap(corr_matrix, mode)
    _plot_scatter_best(rows, best_key, oracle_ps, mode)
    _plot_dhat_dist(rows, mode)

    return summary


def _plot_heatmap(corr_matrix: dict, mode: str) -> None:
    methods = ["lb", "twonn", "pca"]
    p_search_vals = sorted(set(k[1] for k in corr_matrix))

    fig, axes = plt.subplots(1, len(K_DIM_VALS), figsize=(16, 4), sharey=True)
    if len(K_DIM_VALS) == 1:
        axes = [axes]

    for ax_idx, kd in enumerate(sorted(K_DIM_VALS)):
        ax = axes[ax_idx]
        data = np.zeros((len(methods), len(p_search_vals)))
        for mi, m in enumerate(methods):
            for pi, ps in enumerate(p_search_vals):
                v = corr_matrix.get((m, ps, kd), {})
                r2 = v.get("r2", np.nan)
                data[mi, pi] = r2 if not np.isnan(r2) else 0.0

        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=0.3, aspect="auto")
        ax.set_xticks(range(len(p_search_vals)))
        ax.set_xticklabels([str(ps) for ps in p_search_vals])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xlabel("p_search")
        ax.set_title(f"K_dim={kd}")
        for mi in range(len(methods)):
            for pi in range(len(p_search_vals)):
                ax.text(pi, mi, f"{data[mi, pi]:.3f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, label="r²")

    fig.suptitle(f"95b  r²(d̂, oracle_p)  (режим={mode})", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / f"95b_corr_heatmap_{mode}.png"
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"  Рис. heatmap → {out.name}")


def _plot_scatter_best(rows: list, best_key: tuple, oracle_ps: list, mode: str) -> None:
    m, ps, kd = best_key
    d_vals = []
    o_vals = []
    for r, op in zip(rows, oracle_ps):
        v = r["dims"].get(str(ps), {}).get(str(kd), {}).get(m)
        if v is not None:
            d_vals.append(v)
            o_vals.append(op)

    if len(d_vals) < 3:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d_vals, o_vals, alpha=0.4, s=20, color="#42a5f5")
    # Линия тренда
    z = np.polyfit(d_vals, o_vals, 1)
    xs = np.linspace(min(d_vals), max(d_vals), 100)
    ax.plot(xs, np.polyval(z, xs), color="#ef5350", lw=1.5)
    r = float(np.corrcoef(d_vals, o_vals)[0, 1])
    ax.set_xlabel(f"d̂_{m}  (p_search={ps}, K_dim={kd})")
    ax.set_ylabel("oracle_p")
    ax.set_title(f"95b  Лучшая комбинация: r={r:+.4f}  r²={r**2:.4f}")
    plt.tight_layout()
    out = FIG_DIR / f"95b_scatter_best_{mode}.png"
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"  Рис. scatter → {out.name}")


def _plot_dhat_dist(rows: list, mode: str) -> None:
    """Распределения d̂ по всем методам при k_dim=100."""
    kd = 100
    methods = ["lb", "twonn", "pca"]
    colors  = ["#42a5f5", "#66bb6a", "#ffa726"]

    fig, axes = plt.subplots(1, len(P_SEARCH_VALS), figsize=(16, 4), sharey=False)
    if len(P_SEARCH_VALS) == 1:
        axes = [axes]

    for ax_idx, ps in enumerate(sorted(P_SEARCH_VALS)):
        ax = axes[ax_idx]
        for m, c in zip(methods, colors):
            vals = [
                r["dims"].get(str(ps), {}).get(str(kd), {}).get(m)
                for r in rows
                if r["dims"].get(str(ps), {}).get(str(kd), {}).get(m) is not None
            ]
            if vals:
                ax.hist(vals, bins=30, alpha=0.6, color=c, label=m)
        ax.axvline(ps, color="white", ls="--", lw=1, label=f"p_search={ps}")
        ax.set_xlabel("d̂")
        ax.set_ylabel("count")
        ax.set_title(f"p_search={ps}, k_dim={kd}")
        ax.legend(fontsize=8)

    fig.suptitle(f"95b  Распределение d̂ (K_dim={kd}, режим={mode})", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / f"95b_dhat_dist_{mode}.png"
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"  Рис. dhat_dist → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_ID} v{VERSION}")
    parser.add_argument("--mode", choices=["test", "full"], default="full")
    args = parser.parse_args()
    cfg  = MODES[args.mode]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  EXPERIMENT : {EXPERIMENT_ID}")
    print(f"  VERSION    : {VERSION}")
    print(f"  РЕЖИМ      : {cfg['desc']}")
    print(f"  DATA_DIR   : {DATA_DIR}")
    print(f"  PATH_95A   : {PATH_95A}")
    print(f"  P_SEARCH   : {P_SEARCH_VALS}")
    print(f"  K_DIM      : {K_DIM_VALS}")
    print("=" * 65)

    # Загрузка oracle_p из 95a
    if not PATH_95A.exists():
        print(f"\nОШИБКА: файл 95a не найден: {PATH_95A}")
        print("Запустите сначала 95a_oracle_distribution.py --mode full --n_origins 500 --step 5")
        raise SystemExit(1)

    print(f"\nЗагрузка oracle_p из 95a...")
    oracle_map = load_oracle_map(PATH_95A)
    print(f"  Загружено {len(oracle_map)} origins из 95a")

    ticker = cfg["ticker"]
    print(f"\n── {ticker} ─────────────────────────────────────────────────")
    att = load_att(ticker)
    print(f"   att: {len(att)} баров  N_lib≈{len(att) - P_MAX}")

    # Выбор origins
    t_origins = select_origins(oracle_map, cfg["n_origins"], cfg["stride"])
    print(f"   Origins: {len(t_origins)}  (stride={cfg['stride']}, из {len(oracle_map)} 95a-origins)")
    print(f"   Диапазон t_orig: {min(t_origins)}–{max(t_origins)}")

    out_file  = OUT_DIR / f"origins_{ticker}_{args.mode}.jsonl"
    null_file = OUT_DIR / f"null_test_{args.mode}.jsonl"

    if out_file.exists():
        out_file.unlink()
    if null_file.exists():
        null_file.unlink()

    t_global = time.time()

    run_ticker(ticker, att, t_origins, oracle_map, out_file)

    # Null test — в полном режиме, на нескольких origins
    if args.mode == "full":
        print(f"\n  Null test (shuffled att)...")
        rng = np.random.default_rng(42)
        null_origins = t_origins[::10][:20]   # 20 origins равномерно
        for t in null_origins:
            null_rec = run_null_test(att, t, rng)
            if null_rec:
                with open(null_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(null_rec, ensure_ascii=False) + "\n")
                    f.flush()
        print(f"  Null test → {null_file.name}")

    print(f"\n  Анализ результатов...")
    analyse(
        out_file,
        null_file if args.mode == "full" else None,
        args.mode,
    )

    total = time.time() - t_global
    print(f"\n{'='*65}")
    print(f"  Готово за {total/60:.1f} мин")
    print(f"  Результаты: {OUT_DIR}")
    print(f"  Фигуры:     {FIG_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
