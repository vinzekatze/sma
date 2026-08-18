"""
Измерение размерности аттрактора зигзага.

Данные: z-ряды (ratio-уровни пивотов) SBER 1d + 1h + 10m, T=2%.
Все три серии рассматриваются как выборки из одного аттрактора.

Методы:
  TwoNN  — Two Nearest Neighbors (Facco et al. 2017)
           Применяется к объединённому облаку вложенных векторов при каждом p.
           Даёт: как локальная оценка d меняется при увеличении p.

  FNN    — False Nearest Neighbors (Kennel et al. 1992)
           Применяется к каждой серии отдельно (нужна временна́я структура),
           результаты усредняются. Даёт: при каком p FNN% → 0.

  D₂     — Корреляционная размерность Грассбергера-Прокаччиа (1983)
           C(r) = доля пар с расстоянием < r, наклон log(C) / log(r) = D₂.
           Вычисляется для нескольких p, чтобы видеть насыщение.

Вложение: стандартное задержанное, X_i = [z_i, z_{i-1}, ..., z_{i-p+1}]
  (не разности, а уровни — стандарт Такенса).
Нормировка: каждая серия центрируется и масштабируется на std перед пулингом.
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree

_BASE    = os.path.dirname(__file__)
DATA_1D  = os.path.join(_BASE, "../../data/candles/SBER/1d.json")
DATA_1H  = os.path.join(_BASE, "../../prototype/data/candles/SBER/1h.json")
DATA_10M = os.path.join(_BASE, "../../prototype/data/candles/SBER/10m.json")
OUT_DIR  = _BASE

THRESHOLD   = 0.02
P_MAX       = 12        # максимальное число лагов для TwoNN/FNN
P_D2        = [1,2,3,5,8]  # для каких p строить D₂-кривые
FNN_R_TOL   = 10.0      # порог FNN (Kennel 1992, типично 10–15)
FNN_A_TOL   = 2.0       # вспомогательный порог по абс. дистанции

# ── данные ────────────────────────────────────────────────────────────────────

def load_z(path):
    with open(path) as f: data = json.load(f)
    df = pd.DataFrame(data)
    df["close"] = pd.to_numeric(df["close"])
    close = df["close"].values.astype(np.float64)
    trend = logtrend_causal(close)
    ratio = close / trend
    pivots = np.array(find_pivots(ratio, THRESHOLD))
    z = ratio[pivots]
    return z

def logtrend_causal(close):
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64); cn = np.arange(1, n+1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2)
    cy = np.cumsum(lc); cty = np.cumsum(t*lc)
    denom = cn*ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn*cty - ct*cy)/denom, 0.0)
    a = (cy - b*ct)/cn; trend = np.exp(a + b*t); trend[:2] = close[:2]
    return trend

def find_pivots(ratio, thr):
    n = len(ratio); pivots = [0]; direction = 0
    ext_val, ext_idx = ratio[0], 0
    for i in range(1, n):
        v = ratio[i]
        if direction == 0:
            if abs(v - ext_val) >= thr*ext_val:
                direction = 1 if v > ext_val else -1; ext_val, ext_idx = v, i
        elif direction == 1:
            if v > ext_val: ext_val, ext_idx = v, i
            elif (ext_val - v) >= thr*ext_val:
                pivots.append(ext_idx); direction = -1; ext_val, ext_idx = v, i
        else:
            if v < ext_val: ext_val, ext_idx = v, i
            elif (v - ext_val) >= thr*ext_val:
                pivots.append(ext_idx); direction = 1; ext_val, ext_idx = v, i
    return pivots

def normalize(z):
    """Центрировать и масштабировать на std."""
    mu = z.mean(); s = z.std()
    return (z - mu) / (s if s > 1e-10 else 1.0)

def embed(z, p):
    """
    Стандартное задержанное вложение:
    X[i] = [z[i], z[i-1], ..., z[i-p+1]], i >= p-1
    Возвращает матрицу (N-p+1, p).
    """
    N = len(z)
    rows = []
    for i in range(p - 1, N):
        rows.append(z[i - p + 1 : i + 1][::-1])   # [z_i, z_{i-1}, ...]
    return np.array(rows)

# ── TwoNN ─────────────────────────────────────────────────────────────────────

def twonn(X):
    """
    Two Nearest Neighbors (Facco et al. 2017).
    Для каждой точки: mu = r2/r1 (расстояние до 2-го соседа / до 1-го).
    MLE оценка: d = 1 / mean(log(mu)).
    """
    tree = cKDTree(X)
    # запрашиваем 3 соседей (1-й — сама точка)
    dists, _ = tree.query(X, k=3)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    # убираем вырожденные точки (r1=0 — дублирующиеся вектора)
    mask = r1 > 1e-12
    if mask.sum() < 10:
        return np.nan
    mu = r2[mask] / r1[mask]
    d  = 1.0 / np.mean(np.log(mu))
    return d

# ── FNN ───────────────────────────────────────────────────────────────────────

def fnn_series(z, p_max, r_tol=FNN_R_TOL, a_tol=FNN_A_TOL):
    """
    FNN (Kennel 1992) для одной нормализованной серии z.
    Возвращает словарь p -> FNN%.
    Проверка: для каждой точки в p-мерном вложении найти ближайшего соседа j,
    затем проверить, является ли он "ложным" в (p+1)-мерном пространстве.
    """
    z_n = normalize(z)
    std_z = 1.0   # после нормировки std=1
    result = {}

    for p in range(1, p_max + 1):
        X_p  = embed(z_n, p)     # (N-p+1, p)
        X_p1 = embed(z_n, p+1)   # (N-p,   p+1)
        # общий диапазон индексов: первые len(X_p1) строк X_p
        n = len(X_p1)
        if n < 5:
            result[p] = np.nan; continue

        X_p_sub = X_p[:n]  # выровнять длины

        tree = cKDTree(X_p_sub)
        dists_p, idx_p = tree.query(X_p_sub, k=2)
        r1 = dists_p[:, 1]     # расстояние до ближайшего соседа в p-мерном
        nn_idx = idx_p[:, 1]   # индекс соседа

        # дополнительная координата в (p+1)-мерном
        z_extra_i  = X_p1[:, -1]          # z_{i-p} для точки i
        z_extra_nn = X_p1[nn_idx, -1]     # z_{j-p} для соседа j

        delta_extra = np.abs(z_extra_i - z_extra_nn)
        dist_p1 = np.sqrt(r1**2 + delta_extra**2)

        # FNN по критерию 1: Kennel R_tol
        crit1 = (r1 > 1e-10) & (delta_extra / np.maximum(r1, 1e-10) > r_tol)
        # FNN по критерию 2: абс. скачок дистанции
        crit2 = (dist_p1 / std_z) > a_tol

        fnn_pct = np.mean(crit1 | crit2) * 100.0
        result[p] = fnn_pct

    return result

# ── Корреляционная размерность D₂ ────────────────────────────────────────────

def correlation_dim(X, n_r=40, subsample=2000):
    """
    Grassberger-Procaccia: C(r) = (пар с dist < r) / (всего пар).
    Возвращает (log_r, log_C, D2_estimate).
    D2 — наклон в линейном участке.
    """
    N = len(X)
    if N > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, subsample, replace=False)
        X = X[idx]
        N = subsample

    # попарные расстояния через broadcast (быстро при N<=2000)
    diff = X[:, None, :] - X[None, :, :]           # (N,N,d)
    dists_sq = (diff**2).sum(-1)                    # (N,N)
    dists = np.sqrt(dists_sq)
    # верхний треугольник (исключить диагональ)
    idx_u = np.triu_indices(N, k=1)
    d_flat = dists[idx_u]

    r_min = np.percentile(d_flat[d_flat > 0], 5)
    r_max = np.percentile(d_flat, 75)
    if r_min >= r_max:
        return None, None, np.nan

    r_arr = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    C_arr = np.array([np.mean(d_flat < r) for r in r_arr])

    # log-log
    mask = C_arr > 0
    if mask.sum() < 4:
        return r_arr, C_arr, np.nan
    lr = np.log10(r_arr[mask])
    lc = np.log10(C_arr[mask])

    # линейный участок: середина кривой
    lo, hi = int(len(lr)*0.2), int(len(lr)*0.8)
    if hi - lo < 3:
        lo, hi = 0, len(lr)
    coef = np.polyfit(lr[lo:hi], lc[lo:hi], 1)
    D2   = coef[0]

    return r_arr, C_arr, D2

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Загрузка данных …")
    z_1d  = load_z(DATA_1D)
    z_1h  = load_z(DATA_1H)
    z_10m = load_z(DATA_10M)
    print(f"  1d:  {len(z_1d)} пивотов")
    print(f"  1h:  {len(z_1h)} пивотов")
    print(f"  10m: {len(z_10m)} пивотов")
    print(f"  всего: {len(z_1d)+len(z_1h)+len(z_10m)}")

    # нормированные серии (mean=0, std=1)
    zn_1d  = normalize(z_1d)
    zn_1h  = normalize(z_1h)
    zn_10m = normalize(z_10m)
    series_list = [("1d", zn_1d), ("1h", zn_1h), ("10m", zn_10m)]

    # ── TwoNN ─────────────────────────────────────────────────────────────────
    print("\nTwoNN sweep p=1..{}  …".format(P_MAX))
    twonn_vals = []
    for p in range(1, P_MAX + 1):
        # объединённое облако
        parts = []
        for _, zn in series_list:
            X = embed(zn, p)
            if len(X) > 0:
                parts.append(X)
        X_all = np.vstack(parts)
        d = twonn(X_all)
        twonn_vals.append(d)
        print(f"  p={p:2d}  N={len(X_all):5d}  TwoNN d={d:.2f}")

    # ── FNN ───────────────────────────────────────────────────────────────────
    print("\nFNN sweep p=1..{} …".format(P_MAX))
    fnn_all = {p: [] for p in range(1, P_MAX + 1)}
    for name, zn in series_list:
        fnn_res = fnn_series(zn, P_MAX)
        print(f"  {name}:", end="")
        for p in range(1, P_MAX + 1):
            v = fnn_res.get(p, np.nan)
            fnn_all[p].append(v)
            print(f"  p={p}:{v:.1f}%", end="")
        print()

    fnn_mean = {p: np.nanmean(fnn_all[p]) for p in range(1, P_MAX + 1)}

    # ── D₂ ────────────────────────────────────────────────────────────────────
    print("\nCorrelation dimension D₂ …")
    d2_results = {}
    for p in P_D2:
        parts = []
        for _, zn in series_list:
            X = embed(zn, p)
            if len(X) > 0:
                parts.append(X)
        X_all = np.vstack(parts)
        r_arr, C_arr, D2 = correlation_dim(X_all)
        d2_results[p] = (r_arr, C_arr, D2)
        print(f"  p={p:2d}  N={len(X_all):5d}  D₂={D2:.2f}")

    # ── сводка ────────────────────────────────────────────────────────────────
    print("\n── Сводка ──────────────────────────────────────────")
    print(f"  {'p':>3}  {'TwoNN d':>9}  {'FNN%':>7}  {'D₂':>6}")
    for p in range(1, P_MAX + 1):
        d_tn = twonn_vals[p-1]
        d_fn = fnn_mean[p]
        d_d2 = d2_results.get(p, (None, None, np.nan))[2]
        d2_str = f"{d_d2:.2f}" if not np.isnan(d_d2) else "  — "
        print(f"  {p:>3}  {d_tn:>9.2f}  {d_fn:>6.1f}%  {d2_str:>6}")

    # ── графики ───────────────────────────────────────────────────────────────
    p_arr = list(range(1, P_MAX + 1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Размерность аттрактора зигзага — SBER 1d+1h+10m  T=2%  "
        f"(N≈{len(z_1d)+len(z_1h)+len(z_10m)})",
        fontsize=11)

    # ── Субплот 1: TwoNN ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(p_arr, twonn_vals, "o-", color="#1f77b4", lw=1.8, ms=6)
    # диагональ: d = p (заполнение пространства без аттрактора)
    ax.plot(p_arr, p_arr, "--", color="#cccccc", lw=1.0, label="d = p (нет структуры)")
    ax.set_title("TwoNN: локальная размерность d(p)", fontsize=9)
    ax.set_xlabel("p (лагов)", fontsize=8)
    ax.set_ylabel("оценка d", fontsize=8)
    ax.set_xticks(p_arr)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    # аннотация насыщения
    for i, (p, d) in enumerate(zip(p_arr, twonn_vals)):
        if i > 0 and abs(twonn_vals[i] - twonn_vals[i-1]) < 0.15:
            ax.annotate(f"{d:.2f}", (p, d),
                        textcoords="offset points", xytext=(0, 6), fontsize=7,
                        color="#1f77b4", ha="center")

    # ── Субплот 2: FNN ────────────────────────────────────────────────────────
    ax = axes[1]
    for (name, _), color in zip(series_list, ["#1f77b4", "#ff7f0e", "#d62728"]):
        fnn_s = fnn_series(normalize(load_z(
            DATA_1D if name=="1d" else DATA_1H if name=="1h" else DATA_10M
        )), P_MAX)
        vals = [fnn_s.get(p, np.nan) for p in p_arr]
        ax.plot(p_arr, vals, "o-", color=color, lw=1.4, ms=5, label=name, alpha=0.7)

    ax.plot(p_arr, [fnn_mean[p] for p in p_arr],
            "k-", lw=2.2, ms=7, marker="D", label="среднее", zorder=5)
    ax.axhline(10, color="#999999", ls="--", lw=1.0, label="порог 10%")
    ax.set_title("FNN: % ложных соседей (Kennel 1992)", fontsize=9)
    ax.set_xlabel("p (лагов)", fontsize=8)
    ax.set_ylabel("FNN %", fontsize=8)
    ax.set_xticks(p_arr)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.set_ylim(-2, max(max(fnn_mean.values()), 100) * 1.05)

    # ── Субплот 3: D₂ кривые ─────────────────────────────────────────────────
    ax = axes[2]
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(P_D2)))
    for i, p in enumerate(P_D2):
        r_arr, C_arr, D2 = d2_results[p]
        if r_arr is None: continue
        mask = C_arr > 0
        ax.plot(np.log10(r_arr[mask]), np.log10(C_arr[mask]),
                color=palette[i], lw=1.6, label=f"p={p}  D₂={D2:.2f}")

    ax.set_title("D₂ (Grassberger-Procaccia): log C(r) vs log r", fontsize=9)
    ax.set_xlabel("log₁₀ r", fontsize=8)
    ax.set_ylabel("log₁₀ C(r)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "zigzag_dim_analysis.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nСохранено: {out}")
    plt.close(fig)
    print("Готово.")


if __name__ == "__main__":
    main()
