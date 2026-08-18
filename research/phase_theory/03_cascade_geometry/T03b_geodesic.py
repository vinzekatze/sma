"""
T03b_geodesic.py — геодезическое vs Евклидово расстояние на многообразии att.

Для каждого (ticker, variant):
1. att = diff(LP(ratio))
2. X: (N-p_emb+1) × p_emb — delay embedding
3. k-NN граф (k=K_GRAPH, вес = Евклидово расстояние) → Dijkstra → D_geo (N×N)
4. Метрики:
   - knn_overlap(k=K_OVERLAP): % совпадения k ближайших соседей geo vs euc
   - ratio D_geo / D_euc для k-евклидовых пар (медиана, q05, q95)
   - frac_inf: доля пар с D_geo = ∞ (несвязность графа)
5. UMAP: euc-метрика vs geo-метрика (precomputed)
6. k-чувствительность: knn_overlap для k ∈ {5, 10, 20}

Параметры выровнены с T02 (att-варианты A, B, G).
Вариант H (p=288) исключён: k-NN в 288D слишком медленный.
"""

from __future__ import annotations
import csv
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
import umap as umap_lib

# ── параметры ──────────────────────────────────────────────────────────────────
VARIANTS = [
    dict(name="A_d3",      m=9,  d=3,  p_emb=9,   umap_nn=15, label="A: LP(d=3,m=9)  p=9"),
    dict(name="B_d11",     m=33, d=11, p_emb=33,  umap_nn=30, label="B: LP(d=11,m=33) p=33"),
    dict(name="G_att_p144", m=9,  d=3,  p_emb=144, umap_nn=30, label="G: LP(d=3,m=9)  p=144"),
]
TICKERS   = ["SBER", "MRKP", "CHMF", "NVTK"]
K_GRAPH   = 7    # разреженность графа; соседи за пределами k_graph — многоходовые пути
K_OVERLAP = 30   # > K_GRAPH: только для rank > K_GRAPH отношение D_geo/D_euc нетривиально
K_SENS    = [5, 7, 15]
SEEDS     = [0, 42]
LP_K      = 30
LP_N_ITER = 3

DATA_DIR   = Path(__file__).resolve().parents[3] / "data" / "candles"
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "results"
FIG_DIR    = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

CSV_FIELDS = [
    "ticker", "variant", "k_graph", "n_points",
    "n_components", "frac_inf",
    "knn_overlap_k10",
    "ratio_median", "ratio_q05", "ratio_q95",
    "knn_overlap_k5", "knn_overlap_k20",
]


# ── утилиты ───────────────────────────────────────────────────────────────────

def load_close(ticker: str) -> np.ndarray:
    data = json.loads((DATA_DIR / ticker / "1d.json").read_text())
    return np.array([c["close"] for c in data], dtype=float)


def logtrend_causal(close: np.ndarray) -> np.ndarray:
    N   = len(close)
    lc  = np.log(np.maximum(close, 1e-10))
    t   = np.arange(N, dtype=float)
    st  = np.cumsum(t);  st2 = np.cumsum(t * t)
    sy  = np.cumsum(lc); sty = np.cumsum(t * lc)
    n   = np.arange(1, N + 1, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = n * st2 - st ** 2
        b   = np.where(den > 0, (n * sty - st * sy) / den, 0.0)
        a   = np.where(den > 0, (sy - b * st) / n, lc)
    return np.exp(a + b * t)


def lp_smooth(signal: np.ndarray, m: int, d: int, k: int, n_iter: int = 3) -> np.ndarray:
    x    = signal.copy().astype(float)
    half = m // 2
    k_use = min(k + 1, len(x))
    for _ in range(n_iter):
        N   = len(x)
        pad = np.pad(x, (half, half), mode="edge")
        X   = np.lib.stride_tricks.sliding_window_view(pad, m)[:N].copy()
        nn  = NearestNeighbors(n_neighbors=k_use, algorithm="ball_tree")
        nn.fit(X)
        _, idx = nn.kneighbors(X)
        patches  = X[idx]
        center   = patches.mean(axis=1, keepdims=True)
        centered = patches - center
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        Vd       = Vt[:, :d, :]
        delta    = X - center[:, 0, :]
        coeff    = np.einsum("ni,ndi->nd", delta, Vd)
        proj_d   = np.einsum("nd,ndi->ni",  coeff, Vd)
        x        = (center[:, 0, :] + proj_d)[:, half]
    return x


def build_att(close: np.ndarray, m: int, d: int) -> np.ndarray:
    ratio = close / np.maximum(logtrend_causal(close), 1e-10)
    return np.diff(lp_smooth(ratio, m, d, LP_K, LP_N_ITER))


def build_embedding(att: np.ndarray, p: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(att, p).copy()  # (N-p+1, p)


# ── геодезические расстояния ──────────────────────────────────────────────────

def build_geodesic_matrix(X: np.ndarray, k_graph: int
                          ) -> tuple[np.ndarray, int, float]:
    """
    Строит k-NN граф на X и вычисляет матрицу геодезических расстояний (Dijkstra).
    Возвращает: D_geo (N×N), n_components, frac_inf.
    D_geo[i,j] = inf если i и j в разных компонентах связности.
    """
    N  = len(X)
    k_use = min(k_graph + 1, N)
    nn = NearestNeighbors(n_neighbors=k_use, algorithm="auto")
    nn.fit(X)
    dists, idx = nn.kneighbors(X)      # (N, k+1): столбец 0 = сам себя

    # Построить направленный граф (без ручного добавления обратных рёбер —
    # иначе csr_matrix суммирует дубликаты и удваивает веса симметричных пар)
    rows, cols, data = [], [], []
    for i in range(N):
        for j_idx in range(1, k_use):   # skip self (idx[i,0]=i, dist=0)
            j = int(idx[i, j_idx])
            d = float(dists[i, j_idx])
            rows.append(i)
            cols.append(j)
            data.append(d)

    graph_dir = csr_matrix((data, (rows, cols)), shape=(N, N))
    # Симметризация через maximum: где одно направление отсутствует (0),
    # берём другое; где оба присутствуют — берём максимум (≈ одинаково).
    graph = graph_dir.maximum(graph_dir.T)
    graph.eliminate_zeros()

    n_comp, _ = connected_components(graph, directed=False)
    D_geo = shortest_path(graph, method="D", directed=False)

    n_inf   = np.isinf(D_geo).sum()
    frac_inf = float(n_inf) / (N * N)

    return D_geo, int(n_comp), frac_inf


# ── метрики ───────────────────────────────────────────────────────────────────

def knn_overlap(X: np.ndarray, D_geo: np.ndarray, k: int) -> float:
    """
    Для каждой точки: сравнить k ближайших по Евклиду vs k ближайших по геодезической.
    Возвращает среднюю долю совпадения (0..1).
    Игнорирует точки, у которых D_geo=inf для всех соседей.
    """
    N  = len(X)
    tree = KDTree(X)
    _, euc_idx = tree.query(X, k=k + 1)   # +1 т.к. первый = сам себя
    euc_idx    = euc_idx[:, 1:]            # (N, k)

    overlaps = []
    for i in range(N):
        geo_row = D_geo[i].copy()
        geo_row[i] = np.inf                # исключить себя
        if np.all(np.isinf(geo_row)):
            continue
        geo_idx = np.argpartition(geo_row, k)[:k]
        overlap = len(set(euc_idx[i]) & set(geo_idx)) / k
        overlaps.append(overlap)
    return float(np.mean(overlaps)) if overlaps else np.nan


def ratio_stats(X: np.ndarray, D_geo: np.ndarray, k_graph: int, k_far: int
                ) -> dict:
    """
    Ratio = D_geo / D_euc для k_far ближайших Евклидовых соседей.
    Разделяем на:
      'direct'  — rank ≤ k_graph (прямые рёбра в графе; ratio тривиально ≈1)
      'indirect' — rank > k_graph (многоходовые пути; ratio > 1 означает изгиб многообразия)
    Inf-пары исключаются.
    """
    N    = len(X)
    tree = KDTree(X)
    euc_dists, euc_idx = tree.query(X, k=k_far + 1)  # +1 для себя

    direct_r: list[float]   = []
    indirect_r: list[float] = []

    for i in range(N):
        for rank in range(1, k_far + 1):
            j   = euc_idx[i, rank]
            d_e = euc_dists[i, rank]
            d_g = D_geo[i, j]
            if d_e > 1e-14 and np.isfinite(d_g):
                r = d_g / d_e
                if rank <= k_graph:
                    direct_r.append(r)
                else:
                    indirect_r.append(r)

    def _s(arr: list[float]) -> dict:
        if not arr:
            return {"median": np.nan, "q05": np.nan, "q95": np.nan, "n": 0}
        a = np.array(arr)
        return {"median": float(np.median(a)),
                "q05":    float(np.percentile(a, 5)),
                "q95":    float(np.percentile(a, 95)),
                "n":      len(a)}

    return {"direct": _s(direct_r), "indirect": _s(indirect_r)}


# ── визуализация ──────────────────────────────────────────────────────────────

def plot_comparison(X: np.ndarray, D_geo: np.ndarray, t_idx: np.ndarray,
                    var: dict, ticker: str) -> None:
    """
    Два UMAP-рисунка рядом: Евклидово vs Геодезическое.
    Цвет — временно́й индекс (как в T02).
    """
    p    = var["p_emb"]
    nn   = var["umap_nn"]
    name = var["name"]

    # Геодезический UMAP: заменить inf на max_finite*2
    D_geo_umap = D_geo.copy()
    finite_max = D_geo_umap[np.isfinite(D_geo_umap)].max() if np.any(np.isfinite(D_geo_umap)) else 1.0
    D_geo_umap[np.isinf(D_geo_umap)] = finite_max * 2.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ticker} — {var['label']}", fontsize=11)

    for ax, (label, use_geo) in zip(axes, [("Euclidean", False), ("Geodesic", True)]):
        ax.set_title(label, fontsize=10)
        for seed in SEEDS:
            if use_geo:
                reducer = umap_lib.UMAP(n_components=2, metric="precomputed",
                                        n_neighbors=nn, min_dist=0.1,
                                        random_state=seed)
                emb = reducer.fit_transform(D_geo_umap)
            else:
                reducer = umap_lib.UMAP(n_components=2, n_neighbors=nn,
                                        min_dist=0.1, random_state=seed)
                emb = reducer.fit_transform(X)

            if seed == SEEDS[0]:
                sc = ax.scatter(emb[:, 0], emb[:, 1],
                                c=t_idx, cmap="plasma", s=2, alpha=0.6)
                plt.colorbar(sc, ax=ax, label="time")
            else:
                ax.scatter(emb[:, 1] * 0 + emb[:, 0].min() - 5,
                           emb[:, 1], c=t_idx, cmap="plasma", s=0.5,
                           alpha=0.0)   # скрытый scatter только для seed=42 (контроль)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out = FIG_DIR / f"geo_{ticker}_{name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"    → {out.name}", flush=True)


def plot_ratio_hist(indirect_by_variant: dict[str, list[float]],
                    ticker: str) -> None:
    """Гистограмма ratio D_geo/D_euc для НЕПРЯМЫХ соседей (rank > k_graph)."""
    names = list(indirect_by_variant.keys())
    n     = len(names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        vals = indirect_by_variant[name]
        if not vals:
            ax.set_title(f"{name}\n(нет данных)")
            continue
        v = np.clip(vals, 1.0, np.percentile(vals, 99))  # обрезаем хвост для отображения
        ax.hist(v, bins=50, color="steelblue", edgecolor="white", lw=0.3)
        ax.axvline(1.0, color="red", lw=1, linestyle="--", label="ratio=1")
        med = float(np.median(vals))
        ax.axvline(med, color="orange", lw=1.2, linestyle="--", label=f"med={med:.2f}")
        ax.set_title(f"{name}\nmedian={med:.2f}  n={len(vals)}", fontsize=9)
        ax.set_xlabel("D_geo / D_euc  (rank > k_graph)")
        ax.legend(fontsize=7)
    plt.suptitle(f"{ticker} — ratio для непрямых соседей (многоходовые пути)", fontsize=10)
    plt.tight_layout()
    out = FIG_DIR / f"ratio_hist_{ticker}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"    → {out.name}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    CSV_PATH = OUT_DIR / "t03b_metrics.csv"
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    for ticker in TICKERS:
        t0 = time.time()
        print(f"\n══ {ticker} ══", flush=True)
        close = load_close(ticker)
        rows_ticker: list[dict] = []
        ratio_hist_data: dict[str, list[float]] = {}

        for var in VARIANTS:
            name = var["name"]
            p    = var["p_emb"]
            print(f"  {name}  (p={p})...", flush=True)

            t1  = time.time()
            att = build_att(close, var["m"], var["d"])
            X   = build_embedding(att, p)
            N   = len(X)
            t_idx = np.arange(N)
            print(f"    X: {X.shape}  LP={time.time()-t1:.1f}s", flush=True)

            # Геодезическая матрица (основная, k=K_GRAPH)
            t2 = time.time()
            D_geo, n_comp, frac_inf = build_geodesic_matrix(X, K_GRAPH)
            print(f"    geodesic k={K_GRAPH}: {time.time()-t2:.1f}s  "
                  f"components={n_comp}  frac_inf={frac_inf:.4f}", flush=True)

            # k-NN overlap (K_OVERLAP > K_GRAPH → нетривиальный результат)
            ov_main = knn_overlap(X, D_geo, K_OVERLAP)

            # ratio stats: прямые vs непрямые соседи
            rs = ratio_stats(X, D_geo, K_GRAPH, K_OVERLAP)
            indir = rs["indirect"]
            dirct = rs["direct"]

            # k-чувствительность overlap (разные k_graph → разная разреженность)
            ov_sens: dict[int, float] = {}
            for k_s in K_SENS:
                if k_s == K_GRAPH:
                    D_s = D_geo
                else:
                    D_s, _, _ = build_geodesic_matrix(X, k_s)
                ov_sens[k_s] = knn_overlap(X, D_s, K_OVERLAP)
                print(f"    k_sens={k_s}: overlap={ov_sens[k_s]:.3f}", flush=True)

            # Indirect ratios для гистограммы
            tree_ = KDTree(X)
            ed_, ei_ = tree_.query(X, k=K_OVERLAP + 1)
            ind_r: list[float] = []
            for i in range(N):
                for rank in range(K_GRAPH + 1, K_OVERLAP + 1):
                    j = ei_[i, rank]; de = ed_[i, rank]; dg = D_geo[i, j]
                    if de > 1e-14 and np.isfinite(dg):
                        ind_r.append(dg / de)
            ratio_hist_data[name] = ind_r

            rows_ticker.append({
                "ticker":          ticker,
                "variant":         name,
                "k_graph":         K_GRAPH,
                "n_points":        N,
                "n_components":    n_comp,
                "frac_inf":        round(frac_inf, 5),
                "knn_overlap_k10": round(ov_main, 3),
                "ratio_median":    round(indir["median"], 3) if np.isfinite(indir["median"]) else np.nan,
                "ratio_q05":       round(indir["q05"],    3) if np.isfinite(indir["q05"])    else np.nan,
                "ratio_q95":       round(indir["q95"],    3) if np.isfinite(indir["q95"])    else np.nan,
                "knn_overlap_k5":  round(ov_sens.get(5,  np.nan), 3),
                "knn_overlap_k20": round(ov_sens.get(20, np.nan), 3),
            })

            # UMAP-рисунок
            plot_comparison(X, D_geo, t_idx, var, ticker)
            print(f"    {name} готово за {time.time()-t1:.1f}s", flush=True)

        # Гистограмма ratio
        plot_ratio_hist(ratio_hist_data, ticker)

        # Запись CSV
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerows(rows_ticker)

        print(f"  {ticker} всего: {time.time()-t0:.1f}s", flush=True)

    print(f"\nСохранено: {CSV_PATH}", flush=True)
    # Итоговая таблица
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    agg = df.groupby("variant")[["knn_overlap_k10","ratio_median","ratio_q95","frac_inf"]].mean()
    print("\n── Среднее по 4 тикерам ──")
    print(agg.to_string())


if __name__ == "__main__":
    main()
