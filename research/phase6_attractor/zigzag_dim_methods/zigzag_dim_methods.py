#!/usr/bin/env python3
"""
zigzag_dim_methods.py — Исследование оптимальной размерности вложений
событийного зигзага SBER 10m.

Методы:
  FNN        — False Nearest Neighbors (Kennel 1992), 3 варианта критерия
  TwoNN      — Two Nearest Neighbors (Facco 2017)
  MLE-LB     — MLE Левины-Бикеля (Levina & Bickel 2004), sweep k
  D₂         — Корреляционная размерность (Grassberger-Procaccia 1983)

Нормализация для диагностики: глобальная per-field z-score.
Дополнительно: диагностика дрейфа per-step нормировки.

Запуск:
  source /home/kali/.venvs/sma/bin/activate
  cd /home/kali/workspace/apps/sma
  python research/phase6_attractor/zigzag_dim_methods/zigzag_dim_methods.py
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

# ── пути ──────────────────────────────────────────────────────────────────────

HERE    = os.path.dirname(__file__)
DATA_10M = os.path.join(HERE, "../../../data/candles/SBER/10m.json")
RES_DIR  = os.path.join(HERE, "results")
FIG_DIR  = os.path.join(HERE, "figures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── параметры ─────────────────────────────────────────────────────────────────

T_BIG   = 0.04    # 4%
T_SMALL = 0.004   # 0.4%
P_MAX   = 12

FNN_R_THR_LIST  = [5, 10, 15, 20]
FNN_A_THR       = 2.0          # порог в единицах sigma поля
MLE_K_LIST      = [3, 5, 10, 20, 50]
D2_P_LIST       = [1, 2, 3, 5, 8, 12]
D2_N_SUBSAMPLE  = 2000
D2_N_R          = 40

NORM_DRIFT_WIN  = 50           # окно диагностики дрейфа нормировки
LOCAL_MLE_WIN   = 100          # скользящее окно локального MLE
LOCAL_MLE_K     = 10           # k для локального MLE

RNG = np.random.default_rng(42)


# ════════════════════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА И ПАРСИНГ
# ════════════════════════════════════════════════════════════════════════════

def load_10m():
    with open(DATA_10M) as f:
        raw = json.load(f)
    highs  = np.array([d["high"]  for d in raw], dtype=np.float64)
    lows   = np.array([d["low"]   for d in raw], dtype=np.float64)
    closes = np.array([d["close"] for d in raw], dtype=np.float64)
    return highs, lows, closes


def find_pivots_hl(highs, lows, thr):
    """
    Каузальный зигзаг по high/low.
    Возвращает: (prices, types)
      prices: np.ndarray — цена каждого пивота
      types:  np.ndarray — 1=хай, -1=лоу
    """
    prices, types = [], []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    ext_idx = 0
    ext_type = 0

    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1; ext_val = highs[i]; ext_idx = i; ext_type = 1
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val = lows[i]; ext_idx = i; ext_type = -1
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val = highs[i]; ext_idx = i; ext_type = 1
            elif ext_val - lows[i] >= thr * ext_val:
                prices.append(ext_val); types.append(ext_type)
                direction = -1; ext_val = lows[i]; ext_idx = i; ext_type = -1
        else:
            if lows[i] < ext_val:
                ext_val = lows[i]; ext_idx = i; ext_type = -1
            elif highs[i] - ext_val >= thr * ext_val:
                prices.append(ext_val); types.append(ext_type)
                direction = 1; ext_val = highs[i]; ext_idx = i; ext_type = 1

    return np.array(prices, dtype=np.float64), np.array(types, dtype=np.int8)


# ════════════════════════════════════════════════════════════════════════════
# 2. ВЛОЖЕНИЕ
# ════════════════════════════════════════════════════════════════════════════

def build_hybrid_embedding(prices, p):
    """
    Hybrid embedding: X[i] = [price_i, lr₁, lr₂, ..., lr_{p-1}]
      lr_k = log(price_{i-k+1} / price_{i-k})
    Первые p-1 строк = NaN.
    """
    n  = len(prices)
    lp = np.log(np.maximum(prices, 1e-12))
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def embed_valid(prices, p):
    """Вложение без NaN-строк."""
    X = build_hybrid_embedding(prices, p)
    mask = ~np.any(np.isnan(X), axis=1)
    return X[mask]


# ════════════════════════════════════════════════════════════════════════════
# 3. НОРМАЛИЗАЦИЯ
# ════════════════════════════════════════════════════════════════════════════

def global_norm(X):
    """
    Per-field z-score по всему облаку.
    Возвращает (X_norm, mu, sig).
    """
    mu  = X.mean(axis=0)
    sig = X.std(axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)
    return (X - mu) / sig, mu, sig


def norm_drift_diag(prices, p, window=NORM_DRIFT_WIN):
    """
    Диагностика дрейфа per-step нормировки.
    Для каждого step t = window..N строим вложение по пулу [0..t-1]
    и вычисляем mu/sig поля 0 (price) и поля 1 (lr₁).
    Возвращает DataFrame с колонками: step, mu0, sig0, mu1, sig1.
    """
    X_full = build_hybrid_embedding(prices, p)
    valid  = np.where(~np.any(np.isnan(X_full), axis=1))[0]
    rows   = []
    for t in range(window, len(valid)):
        pool = X_full[valid[:t]]
        mu   = pool.mean(0)
        sig  = pool.std(0)
        rows.append({"step": t, "mu0": mu[0], "sig0": sig[0],
                     "mu1": mu[1] if p > 1 else np.nan,
                     "sig1": sig[1] if p > 1 else np.nan})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 4. FNN
# ════════════════════════════════════════════════════════════════════════════

def fnn_fraction_step(X_p_norm, X_p1_norm, variant, r_thr, a_thr=FNN_A_THR):
    """
    Вычисляет долю FNN для пары нормированных вложений X_p и X_{p+1}.

    variant: "R_only" | "R_A" | "A_only"
    X_p_norm, X_p1_norm — уже нормированные по global_norm.
    std_z = 1 (всё нормировано), поэтому A_thr в единицах sigma поля.
    """
    n = min(len(X_p_norm), len(X_p1_norm))
    if n < 5:
        return np.nan

    Xp  = X_p_norm[:n]
    Xp1 = X_p1_norm[:n]

    tree = cKDTree(Xp)
    dists, idxs = tree.query(Xp, k=2)
    r1     = dists[:, 1]
    nn_idx = idxs[:, 1]

    # расстояние в (p+1)-мерном пространстве (полная L2-норма)
    d_p1 = np.linalg.norm(Xp1 - Xp1[nn_idx], axis=1)

    # критерий R: ratio d_{p+1}/d_p > r_thr
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(r1 > 1e-12, d_p1 / r1, np.inf)
    crit_r = ratio > r_thr

    # критерий A: d_{p+1} / sigma > a_thr
    # после per-field нормировки sigma ≈ 1, но сравниваем с std объединённого облака
    # (норма (p+1)-мерная → масштаб sqrt(p+1))
    # используем нормировку: d_{p+1} / sqrt(p+1) > a_thr
    crit_a = (d_p1 / np.sqrt(Xp1.shape[1])) > a_thr

    if variant == "R_only":
        fnn = crit_r
    elif variant == "A_only":
        fnn = crit_a
    else:   # R_A
        fnn = crit_r | crit_a

    # исключаем точки с r1=0 (дубли)
    valid_pts = r1 > 1e-12
    if valid_pts.sum() < 5:
        return np.nan
    return float(fnn[valid_pts].mean())


def fnn_curve_all(prices, p_max=P_MAX,
                  r_thr_list=FNN_R_THR_LIST, a_thr=FNN_A_THR):
    """
    Полный FNN sweep: variant × r_thr × p.
    Нормировка: для каждого p строим X_p и X_{p+1}, нормируем совместно.
    Возвращает dict: {variant: {r_thr: {p: fraction}}}
    """
    variants = ["R_only", "R_A", "A_only"]
    result = {v: {r: {} for r in r_thr_list} for v in variants}

    # строим X_p для всех p сразу (на базе X_{p_max+1})
    X_big = build_hybrid_embedding(prices, p_max + 1)
    valid  = ~np.any(np.isnan(X_big), axis=1)
    X_big_v = X_big[valid]   # все валидные строки

    for p in range(1, p_max + 1):
        Xp  = X_big_v[:, :p]
        Xp1 = X_big_v[:, :p + 1]

        # per-field нормировка совместно для каждой пары (Xp, Xp1)
        mu_p,  sig_p  = Xp.mean(0),  Xp.std(0)
        mu_p1, sig_p1 = Xp1.mean(0), Xp1.std(0)
        sig_p  = np.where(sig_p  < 1e-12, 1.0, sig_p)
        sig_p1 = np.where(sig_p1 < 1e-12, 1.0, sig_p1)

        Xp_n  = (Xp  - mu_p)  / sig_p
        Xp1_n = (Xp1 - mu_p1) / sig_p1

        for variant in variants:
            for r_thr in r_thr_list:
                frac = fnn_fraction_step(Xp_n, Xp1_n, variant, r_thr, a_thr)
                result[variant][r_thr][p] = frac

    return result


# ════════════════════════════════════════════════════════════════════════════
# 5. TwoNN
# ════════════════════════════════════════════════════════════════════════════

def twonn_dim(X_norm):
    """
    Two Nearest Neighbors (Facco 2017).
    d = 1 / E[log(r₂/r₁)]
    Возвращает (d_estimate, n_valid).
    """
    if len(X_norm) < 10:
        return np.nan, 0

    tree = cKDTree(X_norm)
    dists, _ = tree.query(X_norm, k=3)  # k=3: сама точка + 2 соседа
    r1 = dists[:, 1]
    r2 = dists[:, 2]

    mask = r1 > 1e-12
    if mask.sum() < 5:
        return np.nan, 0

    mu = r2[mask] / r1[mask]
    log_mu = np.log(mu)
    d = 1.0 / log_mu.mean()
    return float(d), int(mask.sum())


def twonn_sweep(configs_matrices, p_max=P_MAX):
    """
    Sweep TwoNN по p=1..p_max для всех конфигураций.
    configs_matrices: {name: prices_array}
    Возвращает DataFrame: config, p, d, n_valid
    """
    rows = []
    for name, prices in configs_matrices.items():
        if len(prices) < 10:
            continue
        X_big = build_hybrid_embedding(prices, p_max)
        valid = ~np.any(np.isnan(X_big), axis=1)
        X_v = X_big[valid]

        for p in range(1, p_max + 1):
            Xp = X_v[:, :p]
            _, mu, sig = global_norm(Xp)
            Xp_n = (Xp - mu) / sig
            d, nv = twonn_dim(Xp_n)
            rows.append({"config": name, "p": p, "d": d, "n_valid": nv})

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 6. MLE ЛЕВИНЫ-БИКЕЛЯ
# ════════════════════════════════════════════════════════════════════════════

def mle_lb_global(X_norm, k):
    """
    MLE Levina-Bickel (2004): глобальная оценка.
      d̂(x_i, k) = (k-1) / Σⱼ₌₁ᵏ⁻¹ log(rₖ/rⱼ)
      d̂ = среднее по всем точкам

    Возвращает (d_mean, d_std, n_valid).
    """
    n = len(X_norm)
    if n < k + 2:
        return np.nan, np.nan, 0

    tree  = cKDTree(X_norm)
    dists, _ = tree.query(X_norm, k=k + 1)   # +1 для самой точки
    r = dists[:, 1:]    # (N, k): расстояния до 1..k соседей

    rk = r[:, k - 1:k]                    # (N, 1) — до k-го соседа
    rj = r[:, :k - 1]                     # (N, k-1) — до 1..(k-1)-х

    log_ratios = np.log(rk / np.maximum(rj, 1e-12))   # (N, k-1)
    sum_log    = log_ratios.sum(axis=1)                 # (N,)

    mask   = (sum_log > 1e-12) & (r[:, 0] > 1e-12)
    if mask.sum() < 3:
        return np.nan, np.nan, 0

    d_local = (k - 1) / sum_log[mask]
    d_mean  = float(d_local.mean())
    d_std   = float(d_local.std())
    return d_mean, d_std, int(mask.sum())


def mle_lb_local(prices_seq, p, k=LOCAL_MLE_K, window=LOCAL_MLE_WIN):
    """
    Локальный MLE-LB: скользящее окно по событийному времени.
    Возвращает список (t_center, d_estimate).
    """
    X_big = build_hybrid_embedding(prices_seq, p)
    valid = np.where(~np.any(np.isnan(X_big), axis=1))[0]
    X_v   = X_big[valid]

    if len(X_v) < window + k + 2:
        return []

    _, mu, sig = global_norm(X_v)  # глобальная нормировка
    X_n = (X_v[:, :p] - mu[:p]) / sig[:p]

    results = []
    for t in range(window, len(X_n)):
        X_win = X_n[t - window:t]
        d, _, nv = mle_lb_global(X_win, k)
        results.append({"t": valid[t], "d": d, "n": nv})
    return results


def mle_sweep(configs_matrices, p_max=P_MAX, k_list=MLE_K_LIST):
    """
    Sweep MLE-LB: config × p × k → d.
    Возвращает DataFrame: config, p, k, d, d_std, n_valid.
    """
    rows = []
    for name, prices in configs_matrices.items():
        if len(prices) < 10:
            continue
        X_big = build_hybrid_embedding(prices, p_max)
        valid = ~np.any(np.isnan(X_big), axis=1)
        X_v   = X_big[valid]

        for p in range(1, p_max + 1):
            Xp = X_v[:, :p]
            _, mu, sig = global_norm(Xp)
            Xp_n = (Xp - mu) / sig

            for k in k_list:
                if len(Xp_n) < k + 2:
                    continue
                d, d_std, nv = mle_lb_global(Xp_n, k)
                rows.append({"config": name, "p": p, "k": k,
                              "d": d, "d_std": d_std, "n_valid": nv})

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 7. D₂ — КОРРЕЛЯЦИОННАЯ РАЗМЕРНОСТЬ
# ════════════════════════════════════════════════════════════════════════════

def correlation_dim(X_norm, n_r=D2_N_R, subsample=D2_N_SUBSAMPLE):
    """
    Grassberger-Procaccia (1983).
    Возвращает (log_r, log_C, D2) или (None, None, nan).
    """
    N = len(X_norm)
    if N > subsample:
        idx = RNG.choice(N, subsample, replace=False)
        X   = X_norm[idx]
    else:
        X = X_norm.copy()

    d_flat = pdist(X).astype(np.float32)

    d_pos = d_flat[d_flat > 0]
    if len(d_pos) < 10:
        return None, None, np.nan

    r_min = float(np.percentile(d_pos, 5))
    r_max = float(np.percentile(d_pos, 80))
    if r_min >= r_max:
        return None, None, np.nan

    r_arr = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    C_arr = np.array([np.mean(d_flat < r) for r in r_arr])

    mask = C_arr > 0
    if mask.sum() < 4:
        return r_arr, C_arr, np.nan

    lr = np.log10(r_arr[mask])
    lc = np.log10(C_arr[mask])

    lo = int(len(lr) * 0.2)
    hi = int(len(lr) * 0.8)
    if hi - lo < 3:
        lo, hi = 0, len(lr)

    coef = np.polyfit(lr[lo:hi], lc[lo:hi], 1)
    return r_arr, C_arr, float(coef[0])


def d2_sweep(configs_matrices, p_list=D2_P_LIST):
    """
    Sweep D₂: config × p → D2.
    """
    rows = []
    for name, prices in configs_matrices.items():
        if len(prices) < 10:
            continue

        p_max_local = max(p_list)
        X_big = build_hybrid_embedding(prices, p_max_local)
        valid = ~np.any(np.isnan(X_big), axis=1)
        X_v   = X_big[valid]

        for p in p_list:
            Xp = X_v[:, :p]
            _, mu, sig = global_norm(Xp)
            Xp_n = (Xp - mu) / sig

            t0 = time.time()
            _, _, D2 = correlation_dim(Xp_n)
            elapsed = time.time() - t0

            rows.append({"config": name, "p": p, "D2": D2, "N": len(Xp_n),
                          "elapsed_s": round(elapsed, 1)})
            print(f"    D₂ {name} p={p:2d}  D2={D2:.3f}  N={len(Xp_n)}  "
                  f"t={elapsed:.1f}s", flush=True)

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 8. ШУМ
# ════════════════════════════════════════════════════════════════════════════

def make_noise_gauss(ref_prices):
    """Gaussian white noise с теми же mean/std что и ref_prices."""
    mu  = ref_prices.mean()
    sig = ref_prices.std()
    return RNG.normal(mu, sig, size=len(ref_prices))


def make_noise_shuffle(ref_prices):
    """Shuffled surrogate: сохраняет маргинальное распределение."""
    shuffled = ref_prices.copy()
    RNG.shuffle(shuffled)
    return shuffled


# ════════════════════════════════════════════════════════════════════════════
# 9. СБОРКА КОНФИГУРАЦИЙ И ДИАГНОСТИКА ПОЛЕЙ
# ════════════════════════════════════════════════════════════════════════════

def field_sigma_report(configs, p_max=P_MAX):
    """
    Выводит sigma каждого поля для каждой конфигурации.
    Показывает, насколько масштабы разнородны.
    """
    rows = []
    for name, prices in configs.items():
        if len(prices) < p_max:
            continue
        X = build_hybrid_embedding(prices, p_max)
        valid = ~np.any(np.isnan(X), axis=1)
        Xv = X[valid]
        mu  = Xv.mean(0)
        sig = Xv.std(0)
        for j in range(p_max):
            rows.append({"config": name, "field": j, "mu": mu[j], "sigma": sig[j]})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 10. ГРАФИКИ
# ════════════════════════════════════════════════════════════════════════════

COLORS = {
    "t4_all":     "#1f77b4",
    "t4t04_all":  "#ff7f0e",
    "t4_H":       "#2ca02c",
    "t4_L":       "#d62728",
    "t4t04_H":    "#9467bd",
    "t4t04_L":    "#8c564b",
    "noise_gauss":"#aaaaaa",
    "noise_shuf": "#cccccc",
}
LS = {
    "t4_all":     "-",
    "t4t04_all":  "-",
    "t4_H":       "--",
    "t4_L":       "--",
    "t4t04_H":    "--",
    "t4t04_L":    "--",
    "noise_gauss":":",
    "noise_shuf": ":",
}


def plot_fnn(fnn_results, p_max=P_MAX):
    """FNN: три subplot — R-only (best R_thr), R+A, A-only."""
    variants    = ["R_only", "R_A", "A_only"]
    v_titles    = ["FNN R-only (ratio d_{p+1}/d_p > R_thr)",
                   "FNN R+A (R_thr=10)",
                   "FNN A-only (d_{p+1}/√(p+1) > 2σ)"]
    p_arr = list(range(1, p_max + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("FNN — три варианта критерия  |  SBER 10m", fontsize=11)

    for ax, variant, title in zip(axes, variants, v_titles):
        for name, fnn_res in fnn_results.items():
            vres = fnn_res[variant]
            # для R_only — sweep по R_thr, берём среднее
            if variant == "R_only":
                # строим одну кривую на лучший R_thr = 10 и одну затенённую бэнд
                best = {}
                for p in p_arr:
                    vals = [vres[r][p] for r in FNN_R_THR_LIST
                            if not np.isnan(vres[r].get(p, np.nan))]
                    best[p] = np.mean(vals) if vals else np.nan
                ys = [best[p] for p in p_arr]
            elif variant == "R_A":
                ys = [vres[10].get(p, np.nan) for p in p_arr]
            else:  # A_only
                ys = [vres[FNN_R_THR_LIST[0]].get(p, np.nan) for p in p_arr]

            ax.plot(p_arr, ys,
                    color=COLORS.get(name, "#333333"),
                    ls=LS.get(name, "-"),
                    marker="o", ms=4, lw=1.4, label=name, alpha=0.85)

        ax.set_title(title, fontsize=8)
        ax.set_xlabel("p (размерность вложения)", fontsize=8)
        ax.set_ylabel("Доля FNN", fontsize=8)
        ax.set_xticks(p_arr)
        ax.legend(fontsize=6.5, ncol=2)
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fnn_variants.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_fnn_rthr_sweep(fnn_results, p_max=P_MAX):
    """Sweep R_thr для R-only: как зависит от порога."""
    configs_main = [n for n in fnn_results if "noise" not in n]
    n_cfg = len(configs_main)
    if n_cfg == 0:
        return

    p_arr = list(range(1, p_max + 1))
    fig, axes = plt.subplots(1, n_cfg, figsize=(4 * n_cfg, 4))
    if n_cfg == 1:
        axes = [axes]
    fig.suptitle("FNN R-only: sweep R_thr  |  SBER 10m", fontsize=10)

    palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(FNN_R_THR_LIST)))

    for ax, name in zip(axes, configs_main):
        vres = fnn_results[name]["R_only"]
        for i, r_thr in enumerate(FNN_R_THR_LIST):
            ys = [vres[r_thr].get(p, np.nan) for p in p_arr]
            ax.plot(p_arr, ys, color=palette[i], marker="o", ms=3, lw=1.2,
                    label=f"R_thr={r_thr}", alpha=0.85)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("p", fontsize=8)
        ax.set_ylabel("FNN fraction", fontsize=8)
        ax.set_xticks(p_arr)
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fnn_rthr_sweep.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_twonn(df_twonn, p_max=P_MAX):
    """TwoNN d(p) для всех конфигураций."""
    p_arr = list(range(1, p_max + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(p_arr, p_arr, "--", color="#cccccc", lw=1.0, label="d = p (шум, нет структуры)")

    for name in df_twonn["config"].unique():
        sub = df_twonn[df_twonn["config"] == name].sort_values("p")
        ax.plot(sub["p"], sub["d"],
                color=COLORS.get(name, "#333333"),
                ls=LS.get(name, "-"),
                marker="o", ms=5, lw=1.6, label=name, alpha=0.85)

    ax.set_title("TwoNN: интринзик-размерность d(p)  |  SBER 10m", fontsize=10)
    ax.set_xlabel("p (размерность вложения)", fontsize=9)
    ax.set_ylabel("d̂ (оценка TwoNN)", fontsize=9)
    ax.set_xticks(p_arr)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "twonn.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_mle(df_mle, k_ref=10, p_max=P_MAX):
    """MLE-LB: d(p) для k=k_ref, и sweep k для одной конфигурации."""
    p_arr = list(range(1, p_max + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MLE Левины-Бикеля  |  SBER 10m", fontsize=10)

    # subplot 1: d(p) при k=k_ref
    ax = axes[0]
    ax.plot(p_arr, p_arr, "--", color="#cccccc", lw=1.0, label="d = p")
    sub_k = df_mle[df_mle["k"] == k_ref]
    for name in sub_k["config"].unique():
        sub = sub_k[sub_k["config"] == name].sort_values("p")
        ax.plot(sub["p"], sub["d"],
                color=COLORS.get(name, "#333333"),
                ls=LS.get(name, "-"),
                marker="o", ms=4, lw=1.4, label=name, alpha=0.85)
    ax.set_title(f"d(p)  k={k_ref}", fontsize=9)
    ax.set_xlabel("p", fontsize=8); ax.set_ylabel("d̂", fontsize=8)
    ax.set_xticks(p_arr); ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=0.25)

    # subplot 2: sweep k для t4_all при p=3
    ax = axes[1]
    p_ref = 3
    sub_cfg = df_mle[df_mle["p"] == p_ref]
    k_vals = sorted(df_mle["k"].unique())
    for name in sub_cfg["config"].unique():
        sub = sub_cfg[sub_cfg["config"] == name].sort_values("k")
        if len(sub) < 2:
            continue
        ax.plot(sub["k"], sub["d"],
                color=COLORS.get(name, "#333333"),
                ls=LS.get(name, "-"),
                marker="s", ms=4, lw=1.4, label=name, alpha=0.85)
    ax.set_title(f"d(k)  p={p_ref}  — чувствительность к k", fontsize=9)
    ax.set_xlabel("k (число соседей)", fontsize=8); ax.set_ylabel("d̂", fontsize=8)
    ax.set_xscale("log"); ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=0.25)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "mle_lb.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_d2(df_d2):
    """D₂(p) для всех конфигураций."""
    fig, ax = plt.subplots(figsize=(10, 5))

    p_vals = sorted(df_d2["p"].unique())
    ax.plot(p_vals, p_vals, "--", color="#cccccc", lw=1.0, label="D₂ = p (шум)")

    for name in df_d2["config"].unique():
        sub = df_d2[df_d2["config"] == name].sort_values("p")
        ax.plot(sub["p"], sub["D2"],
                color=COLORS.get(name, "#333333"),
                ls=LS.get(name, "-"),
                marker="D", ms=5, lw=1.6, label=name, alpha=0.85)

    ax.set_title("D₂ (Grassberger-Procaccia)  |  SBER 10m", fontsize=10)
    ax.set_xlabel("p (размерность вложения)", fontsize=9)
    ax.set_ylabel("D₂", fontsize=9)
    ax.set_xticks(p_vals); ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.25)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "d2.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_norm_drift(df_drift, config_name="t4_all"):
    """Дрейф mu/sigma нормировки от step к step."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"Дрейф per-step нормировки  |  {config_name}", fontsize=10)

    for ax, col_mu, col_sig, label in [
        (axes[0], "mu0", "sig0", "Поле 0: price"),
        (axes[1], "mu1", "sig1", "Поле 1: lr₁"),
    ]:
        ax.plot(df_drift["step"], df_drift[col_mu],
                label="μ", color="#1f77b4", lw=1.2)
        ax2 = ax.twinx()
        ax2.plot(df_drift["step"], df_drift[col_sig],
                 label="σ", color="#ff7f0e", lw=1.2, ls="--")
        ax.set_ylabel(f"μ ({label})", fontsize=8)
        ax2.set_ylabel("σ", fontsize=8)
        ax.set_title(label, fontsize=8)
        ax.grid(alpha=0.25)
        # легенды
        lines  = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=7, loc="upper right")

    axes[1].set_xlabel("step (событие)", fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "norm_drift.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_local_mle(local_res, name, p=3):
    """Локальный MLE в скользящем окне."""
    if not local_res:
        return
    ts = [r["t"] for r in local_res]
    ds = [r["d"] for r in local_res]
    if not ts:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, ds, color="#1f77b4", lw=1.0, alpha=0.7)
    ax.axhline(np.nanmedian(ds), color="#ff7f0e", ls="--", lw=1.5,
               label=f"медиана d={np.nanmedian(ds):.2f}")
    ax.set_title(f"Локальный MLE-LB  p={p}  k={LOCAL_MLE_K}  окно={LOCAL_MLE_WIN}"
                 f"  |  {name}", fontsize=9)
    ax.set_xlabel("t (событие)", fontsize=8)
    ax.set_ylabel("d̂ (локальный)", fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"local_mle_{name}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_summary(df_twonn, df_mle, df_d2, p_max=P_MAX):
    """Сводный график: все методы рядом для t4_all и noise_gauss."""
    p_arr = list(range(1, p_max + 1))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Сводка методов: реальный ряд vs шум  |  SBER 10m T=4%", fontsize=10)

    pairs = [("t4_all", "noise_gauss", "#1f77b4", "#aaaaaa"),
             ("t4t04_all", "noise_shuf", "#ff7f0e", "#cccccc")]

    # TwoNN
    ax = axes[0]
    ax.plot(p_arr, p_arr, "--", color="#cccccc", lw=0.8)
    for real_n, noise_n, c_real, c_noise in pairs:
        for nm, col in [(real_n, c_real), (noise_n, c_noise)]:
            sub = df_twonn[df_twonn["config"] == nm].sort_values("p")
            ax.plot(sub["p"], sub["d"], color=col, marker="o", ms=4, lw=1.4,
                    label=nm, alpha=0.9)
    ax.set_title("TwoNN d(p)", fontsize=9)
    ax.set_xlabel("p"); ax.set_ylabel("d̂"); ax.set_xticks(p_arr)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.25)

    # MLE k=10
    ax = axes[1]
    ax.plot(p_arr, p_arr, "--", color="#cccccc", lw=0.8)
    sub_k = df_mle[df_mle["k"] == 10]
    for real_n, noise_n, c_real, c_noise in pairs:
        for nm, col in [(real_n, c_real), (noise_n, c_noise)]:
            sub = sub_k[sub_k["config"] == nm].sort_values("p")
            ax.plot(sub["p"], sub["d"], color=col, marker="s", ms=4, lw=1.4,
                    label=nm, alpha=0.9)
    ax.set_title("MLE-LB d(p)  k=10", fontsize=9)
    ax.set_xlabel("p"); ax.set_ylabel("d̂"); ax.set_xticks(p_arr)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.25)

    # D₂
    ax = axes[2]
    p_d2 = sorted(df_d2["p"].unique())
    ax.plot(p_d2, p_d2, "--", color="#cccccc", lw=0.8)
    for real_n, noise_n, c_real, c_noise in pairs:
        for nm, col in [(real_n, c_real), (noise_n, c_noise)]:
            sub = df_d2[df_d2["config"] == nm].sort_values("p")
            ax.plot(sub["p"], sub["D2"], color=col, marker="D", ms=5, lw=1.6,
                    label=nm, alpha=0.9)
    ax.set_title("D₂(p)", fontsize=9)
    ax.set_xlabel("p"); ax.set_ylabel("D₂"); ax.set_xticks(p_d2)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.25)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "summary_methods.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ════════════════════════════════════════════════════════════════════════════

def print_section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}", flush=True)


def main():
    t_start = time.time()

    # ── данные ────────────────────────────────────────────────────────────
    print_section("Загрузка данных")
    highs, lows, closes = load_10m()
    print(f"  10m баров: {len(highs)}")

    p4,  tp4  = find_pivots_hl(highs, lows, T_BIG)
    p04, tp04 = find_pivots_hl(highs, lows, T_SMALL)

    mask_H_t4  = tp4  == 1
    mask_L_t4  = tp4  == -1
    mask_H_t04 = tp04 == 1
    mask_L_t04 = tp04 == -1

    print(f"  T=4%  : {len(p4)}  (H={mask_H_t4.sum()}  L={mask_L_t4.sum()})")
    print(f"  T=0.4%: {len(p04)}  (H={mask_H_t04.sum()}  L={mask_L_t04.sum()})")

    # ── конфигурации ──────────────────────────────────────────────────────
    # Для H-only и L-only: строим вложение по подпоследовательности
    # (события в событийном времени, не перемешивая направления)

    noise_g = make_noise_gauss(p4)
    noise_s = make_noise_shuffle(p4)

    configs = {
        "t4_all":     p4,
        "t4t04_all":  np.concatenate([p4, p04]),
        "t4_H":       p4[mask_H_t4],
        "t4_L":       p4[mask_L_t4],
        "t4t04_H":    np.concatenate([p4[mask_H_t4], p04[mask_H_t04]]),
        "t4t04_L":    np.concatenate([p4[mask_L_t4], p04[mask_L_t04]]),
        "noise_gauss": noise_g,
        "noise_shuf":  noise_s,
    }

    for name, arr in configs.items():
        print(f"  {name:20s}: N={len(arr)}")

    # ── диагностика полей ──────────────────────────────────────────────────
    print_section("Диагностика: sigma полей")
    df_fields = field_sigma_report(configs, p_max=P_MAX)
    df_fields.to_csv(os.path.join(RES_DIR, "field_sigma.csv"), index=False)
    # сводка для p=3 (поля 0,1,2)
    print(f"\n  {'config':20s}  {'σ_price':>10}  {'σ_lr1':>10}  {'σ_lr2':>10}")
    for name in configs:
        sub = df_fields[df_fields["config"] == name]
        s0 = sub[sub["field"] == 0]["sigma"].values
        s1 = sub[sub["field"] == 1]["sigma"].values
        s2 = sub[sub["field"] == 2]["sigma"].values
        v0 = f"{s0[0]:.5f}" if len(s0) else "—"
        v1 = f"{s1[0]:.5f}" if len(s1) else "—"
        v2 = f"{s2[0]:.5f}" if len(s2) else "—"
        print(f"  {name:20s}  {v0:>10}  {v1:>10}  {v2:>10}")

    # ── диагностика дрейфа нормировки ─────────────────────────────────────
    print_section("Диагностика: дрейф per-step нормировки (t4_all, p=3)")
    df_drift = norm_drift_diag(configs["t4_all"], p=3, window=NORM_DRIFT_WIN)
    df_drift.to_csv(os.path.join(RES_DIR, "norm_drift.csv"), index=False)
    drift_mu0  = df_drift["mu0"].std()
    drift_sig0 = df_drift["sig0"].std()
    drift_mu1  = df_drift["mu1"].std()
    drift_sig1 = df_drift["sig1"].std()
    print(f"  std(μ_price)={drift_mu0:.4f}  std(σ_price)={drift_sig0:.4f}")
    print(f"  std(μ_lr1)  ={drift_mu1:.5f}  std(σ_lr1)  ={drift_sig1:.5f}")
    print(f"  (малый drift → per-step нормировка стабильна)")

    # ── FNN ───────────────────────────────────────────────────────────────
    print_section("FNN sweep")
    fnn_results_all = {}
    fnn_rows = []

    for name, prices in configs.items():
        print(f"  {name} ...", flush=True)
        fnn_res = fnn_curve_all(prices, p_max=P_MAX)
        fnn_results_all[name] = fnn_res

        for variant, r_dict in fnn_res.items():
            for r_thr, p_dict in r_dict.items():
                for p, frac in p_dict.items():
                    fnn_rows.append({
                        "config": name, "variant": variant,
                        "R_thr": r_thr, "p": p, "frac": frac
                    })

    df_fnn = pd.DataFrame(fnn_rows)
    df_fnn.to_csv(os.path.join(RES_DIR, "fnn_results.csv"), index=False)

    # сводка: минимум FNN для каждой конфигурации (R_A, R_thr=10)
    print(f"\n  Минимум FNN (R+A, R_thr=10):")
    print(f"  {'config':20s}  {'p_min_frac':>10}  {'min_frac':>10}")
    for name in configs:
        sub = df_fnn[(df_fnn["config"] == name) &
                     (df_fnn["variant"] == "R_A") &
                     (df_fnn["R_thr"] == 10)].sort_values("p")
        if len(sub) == 0:
            continue
        idx_min = sub["frac"].idxmin()
        p_min   = sub.loc[idx_min, "p"]
        frac_m  = sub.loc[idx_min, "frac"]
        print(f"  {name:20s}  {p_min:>10}  {frac_m:>10.3f}")

    # ── TwoNN ─────────────────────────────────────────────────────────────
    print_section("TwoNN sweep")
    df_twonn = twonn_sweep(configs, p_max=P_MAX)
    df_twonn.to_csv(os.path.join(RES_DIR, "twonn_results.csv"), index=False)

    print(f"\n  TwoNN d(p=3):")
    print(f"  {'config':20s}  {'d':>8}  {'N':>8}")
    for name in configs:
        sub = df_twonn[(df_twonn["config"] == name) & (df_twonn["p"] == 3)]
        if len(sub) == 0:
            continue
        print(f"  {name:20s}  {sub['d'].values[0]:>8.3f}  "
              f"{sub['n_valid'].values[0]:>8}")

    # ── MLE Левины-Бикеля ──────────────────────────────────────────────────
    print_section("MLE Левины-Бикеля sweep")
    df_mle = mle_sweep(configs, p_max=P_MAX, k_list=MLE_K_LIST)
    df_mle.to_csv(os.path.join(RES_DIR, "mle_results.csv"), index=False)

    print(f"\n  MLE-LB d(p=3, k=10):")
    print(f"  {'config':20s}  {'d':>8}  {'d_std':>8}")
    for name in configs:
        sub = df_mle[(df_mle["config"] == name) &
                     (df_mle["p"] == 3) &
                     (df_mle["k"] == 10)]
        if len(sub) == 0:
            continue
        print(f"  {name:20s}  {sub['d'].values[0]:>8.3f}  "
              f"{sub['d_std'].values[0]:>8.3f}")

    # ── Локальный MLE для t4_all ──────────────────────────────────────────
    print_section("Локальный MLE-LB (t4_all, p=3)")
    local_res = mle_lb_local(configs["t4_all"], p=3,
                             k=LOCAL_MLE_K, window=LOCAL_MLE_WIN)
    df_local = pd.DataFrame(local_res)
    if len(df_local) > 0:
        df_local.to_csv(os.path.join(RES_DIR, "local_mle_t4_all.csv"), index=False)
        print(f"  Шагов: {len(df_local)}  медиана d={df_local['d'].median():.3f}"
              f"  std d={df_local['d'].std():.3f}")

    # ── D₂ ────────────────────────────────────────────────────────────────
    print_section("D₂ (Grassberger-Procaccia)")
    print("  (subsample=2000, медленнее для t4t04_all)")
    df_d2 = d2_sweep(configs, p_list=D2_P_LIST)
    df_d2.to_csv(os.path.join(RES_DIR, "d2_results.csv"), index=False)

    print(f"\n  D₂(p=3):")
    print(f"  {'config':20s}  {'D2':>8}")
    for name in configs:
        sub = df_d2[(df_d2["config"] == name) & (df_d2["p"] == 3)]
        if len(sub) == 0:
            continue
        print(f"  {name:20s}  {sub['D2'].values[0]:>8.3f}")

    # ── сводная таблица ───────────────────────────────────────────────────
    print_section("Сводная таблица (p=3)")
    print(f"  {'config':20s}  {'TwoNN':>7}  {'MLE_k10':>8}  {'D2':>7}  "
          f"{'FNN_min(R+A)':>13}")
    for name in configs:
        tn_sub = df_twonn[(df_twonn["config"] == name) & (df_twonn["p"] == 3)]
        ml_sub = df_mle[(df_mle["config"] == name) &
                        (df_mle["p"] == 3) & (df_mle["k"] == 10)]
        d2_sub = df_d2[(df_d2["config"] == name) & (df_d2["p"] == 3)]
        fn_sub = df_fnn[(df_fnn["config"] == name) &
                        (df_fnn["variant"] == "R_A") &
                        (df_fnn["R_thr"] == 10)].sort_values("p")

        tn_d  = f"{tn_sub['d'].values[0]:.3f}" if len(tn_sub) else "—"
        ml_d  = f"{ml_sub['d'].values[0]:.3f}" if len(ml_sub) else "—"
        d2_d  = f"{d2_sub['D2'].values[0]:.3f}" if len(d2_sub) else "—"
        if len(fn_sub) > 0:
            idx_m = fn_sub["frac"].idxmin()
            fn_d  = f"p={fn_sub.loc[idx_m,'p']}:{fn_sub.loc[idx_m,'frac']:.3f}"
        else:
            fn_d  = "—"

        print(f"  {name:20s}  {tn_d:>7}  {ml_d:>8}  {d2_d:>7}  {fn_d:>13}")

    # ── графики ───────────────────────────────────────────────────────────
    print_section("Графики")
    plot_fnn(fnn_results_all, P_MAX)
    plot_fnn_rthr_sweep(fnn_results_all, P_MAX)
    plot_twonn(df_twonn, P_MAX)
    plot_mle(df_mle, k_ref=10, p_max=P_MAX)
    plot_d2(df_d2)
    plot_norm_drift(df_drift, config_name="t4_all")
    plot_local_mle(local_res, name="t4_all", p=3)
    plot_summary(df_twonn, df_mle, df_d2, P_MAX)

    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  Готово за {elapsed:.0f}с")
    print(f"  results/ → {RES_DIR}")
    print(f"  figures/ → {FIG_DIR}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
