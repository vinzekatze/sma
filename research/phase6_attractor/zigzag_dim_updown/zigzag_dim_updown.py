"""
Размерность аттрактора зигзага: отдельно для up-пивотов и down-пивотов.
SBER 1d+1h+10m, фрактальная гипотеза — все таймфреймы пулируются.

Вариант А — sub-sequence (каждая половина как самостоятельный ряд):
  z_up[k]   = ratio на k-м лое;  lag-embedding по лоям
  z_down[k] = ratio на k-м хае;  lag-embedding по хаям
  → TwoNN(p) и FNN(p) для up / down / full, отдельно на каждом TF

Вариант Б — full embedding, облака разбиты по направлению:
  X[i] = [z[i], z[i-1], ..., z[i-p+1]]  (полный зигзаг)
  up-точки  = i где z[i] < z[i-1]
  down-точки = i где z[i] > z[i-1]
  → TwoNN для каждого облака при p = 2..16

Pooled (фрактальность): перед пулингом ratio стандартизируется per-TF
  → z_std = (ratio_piv - mean) / std
  При p≥2: объединяем delay-векторы из 1d + 1h + 10m

Bootstrap CI (B=200) для всех TwoNN-оценок.
FNN (Kennel 1992): R_thr=15, p=1..14.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = Path(__file__).parent
DATA = Path(__file__).parent.parent.parent.parent / "data" / "candles" / "SBER"
OUT  = BASE / "results"
OUT.mkdir(parents=True, exist_ok=True)

THRESH  = 0.02
P_MAX   = 16       # sweep embedding dim 1..P_MAX
B_BOOT  = 200      # bootstrap iterations
R_THR   = 15.0     # FNN amplitude threshold
A_THR   = 2.0      # FNN relative-to-std threshold
TIMEFRAMES = ["1d", "1h", "10m"]

# ── Базовые утилиты ──────────────────────────────────────────────────────────

def logtrend_causal(close):
    n  = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    N  = np.arange(1, n + 1, dtype=np.float64)
    St, Sp   = np.cumsum(t),      np.cumsum(lc)
    St2, Stp = np.cumsum(t ** 2), np.cumsum(t * lc)
    den = N * St2 - St ** 2
    b   = np.where(den > 1e-12, (N * Stp - St * Sp) / den, 0.0)
    a   = (Sp - b * St) / N
    trd = np.exp(a + b * t); trd[:2] = close[:2]
    return trd


def find_pivots(ratio, thr=THRESH):
    pivots = [0]; direction = 0; ev, ei = ratio[0], 0
    for i in range(1, len(ratio)):
        v = ratio[i]
        if direction == 0:
            if abs(v - ev) >= thr * ev:
                direction = 1 if v > ev else -1; ev, ei = v, i
        elif direction == 1:
            if v > ev: ev, ei = v, i
            elif (ev - v) >= thr * ev:
                pivots.append(ei); direction = -1; ev, ei = v, i
        else:
            if v < ev: ev, ei = v, i
            elif (v - ev) >= thr * ev:
                pivots.append(ei); direction = 1; ev, ei = v, i
    return np.array(pivots, dtype=int)


def load_tf(tf):
    """Возвращает: z_std (стандартизованный ratio на пивотах), маску up/down."""
    with open(DATA / f"{tf}.json") as f:
        data = json.load(f)
    close = np.array([d["close"] for d in data], dtype=np.float64)
    trend = logtrend_causal(close)
    ratio = close / trend
    piv   = find_pivots(ratio)
    z     = ratio[piv]
    z_std = (z - z.mean()) / max(z.std(), 1e-10)
    # direction[i]: тип пивота i (начиная с i=1)
    # up[i]=True: z[i] < z[i-1] → текущий пивот — лой → следующий хай
    up_mask = np.concatenate([[False], z[1:] < z[:-1]])  # длина len(z)
    return z_std, up_mask


# ── TwoNN estimator (Facco et al. 2017) ─────────────────────────────────────

def twonn_dim(X, return_mu=False):
    """
    Intrinsic dimension via TwoNN.
    X: (n, d) array of points.
    d_hat = 1 / mean(log(r2/r1))
    """
    if len(X) < 5:
        return np.nan
    nn  = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    ds, _ = nn.kneighbors(X)
    r1, r2 = ds[:, 1], ds[:, 2]
    ok  = r1 > 1e-12
    if ok.sum() < 3:
        return np.nan
    mu  = r2[ok] / r1[ok]
    d   = 1.0 / np.mean(np.log(mu))
    return (d, mu) if return_mu else d


def twonn_bootstrap(X, B=B_BOOT):
    """Bootstrap CI для TwoNN."""
    d_hat = twonn_dim(X)
    if np.isnan(d_hat):
        return d_hat, np.nan, np.nan
    n   = len(X)
    ds  = []
    for _ in range(B):
        idx = np.random.choice(n, n, replace=True)
        d   = twonn_dim(X[idx])
        if not np.isnan(d):
            ds.append(d)
    if not ds:
        return d_hat, np.nan, np.nan
    return d_hat, np.percentile(ds, 2.5), np.percentile(ds, 97.5)


# ── Delay embedding ──────────────────────────────────────────────────────────

def delay_matrix(z, p):
    """X[i] = [z[i], z[i-1], ..., z[i-p+1]], shape (n-p+1, p)."""
    n = len(z)
    if n < p:
        return np.empty((0, p))
    rows = np.array([z[i : i + p][::-1] for i in range(n - p + 1)])
    return rows  # rows[0] = [z[p-1], z[p-2], ..., z[0]]


def delay_matrix_forward(z, p):
    """
    X[i] = [z[i], z[i-1], ..., z[i-p+1]] for i = p-1 .. n-1.
    i-индекс в ИСХОДНОМ ряду z (для отслеживания up/down).
    Возвращает матрицу X и массив i-индексов.
    """
    n   = len(z)
    idx = np.arange(p - 1, n)
    X   = np.stack([z[i - p + 1 : i + 1][::-1] for i in idx])
    return X, idx


# ── FNN (Kennel 1992) ─────────────────────────────────────────────────────────

def fnn_fractions(z, p_max=14, R_thr=R_THR, A_thr=A_THR):
    """
    FNN fractions для p = 1..p_max на ряде z.
    Возвращает массив длины p_max.
    """
    n     = len(z)
    sigma = z.std()
    fracs = []

    for p in range(1, p_max + 1):
        # Embedding при p и p+1
        X_p, idx_p = delay_matrix_forward(z, p)
        if len(X_p) < 5:
            fracs.append(np.nan); continue

        nn    = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X_p)
        dists, inds = nn.kneighbors(X_p)
        d_p   = dists[:, 1]          # расстояние до ближайшего соседа
        j_loc = inds[:, 1]           # local index соседа в X_p

        # Следующая координата: z[i-p] для каждой точки
        # i = idx_p[k], сосед i = idx_p[j_loc[k]]
        # (p+1)-я координата: z[i-p] = z[idx_p[k]-p]
        ok    = (idx_p >= p) & (idx_p[j_loc] >= p)
        k_ok  = np.where(ok)[0]
        if len(k_ok) == 0:
            fracs.append(np.nan); continue

        i_arr    = idx_p[k_ok]
        j_arr    = idx_p[j_loc[k_ok]]
        extra_i  = z[i_arr - p]
        extra_j  = z[j_arr - p]
        delta    = np.abs(extra_i - extra_j)
        d_ok     = d_p[k_ok]

        with np.errstate(divide="ignore", invalid="ignore"):
            crit1 = delta / np.where(d_ok > 1e-12, d_ok, 1e-12) > R_thr
            crit2 = delta / max(sigma, 1e-10) > A_thr
        fnn_flag = crit1 | crit2
        fracs.append(fnn_flag.mean())

    return np.array(fracs)


# ── Вариант А: sub-sequence analysis ─────────────────────────────────────────

def variant_A(z_full, up_mask, tf_label):
    """
    z_full: стандартизованный ratio на всех пивотах TF.
    up_mask: True там где пивот — лой (направление вверх).
    """
    # Sub-sequences начиная с индекса 1 (первый имеет определённое направление)
    z_up   = z_full[up_mask]    # лои
    z_down = z_full[~up_mask]   # хаи

    results = []
    for direction, z_sub in [("up", z_up), ("down", z_down), ("full", z_full)]:
        for p in range(1, P_MAX + 1):
            X = delay_matrix(z_sub, p)
            if len(X) < 5:
                continue
            d, lo, hi = twonn_bootstrap(X)
            results.append({
                "tf": tf_label, "variant": "A", "direction": direction,
                "p": p, "n": len(X),
                "twonn": d, "ci_lo": lo, "ci_hi": hi,
            })
        print(f"    A {tf_label} {direction}: n_up={len(z_up)} n_down={len(z_down)}", flush=True)

    # FNN
    fnn_rows = []
    for direction, z_sub in [("up", z_up), ("down", z_down), ("full", z_full)]:
        fracs = fnn_fractions(z_sub, p_max=min(14, len(z_sub) // 5))
        for p_idx, frac in enumerate(fracs):
            fnn_rows.append({
                "tf": tf_label, "direction": direction,
                "p": p_idx + 1, "fnn_frac": frac,
            })

    return pd.DataFrame(results), pd.DataFrame(fnn_rows)


# ── Вариант Б: full embedding, split by direction ────────────────────────────

def variant_B_single(z_full, up_mask, tf_label):
    """TwoNN на up/down облаках в полном embedding, один TF."""
    results = []
    for p in range(2, P_MAX + 1):
        X, idx = delay_matrix_forward(z_full, p)
        if len(X) < 5:
            continue
        # up_mask относится к исходным индексам; idx — индексы в z_full
        up_here   = up_mask[idx]
        X_up      = X[up_here]
        X_down    = X[~up_here]
        for direction, Xsub in [("up", X_up), ("down", X_down), ("full", X)]:
            d, lo, hi = twonn_bootstrap(Xsub)
            results.append({
                "tf": tf_label, "variant": "B_single", "direction": direction,
                "p": p, "n": len(Xsub),
                "twonn": d, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(results)


def variant_B_pooled(all_z_std, all_up_masks):
    """
    Пулим delay-векторы из всех TF (фрактальная гипотеза).
    all_z_std: список z_std (стандартизованных per-TF).
    all_up_masks: список up_mask для каждого TF.
    """
    results = []
    for p in range(2, P_MAX + 1):
        Xup_list, Xdn_list, Xfull_list = [], [], []
        for z_std, up_mask in zip(all_z_std, all_up_masks):
            X, idx = delay_matrix_forward(z_std, p)
            if len(X) == 0:
                continue
            up_here = up_mask[idx]
            Xup_list.append(X[up_here])
            Xdn_list.append(X[~up_here])
            Xfull_list.append(X)

        for direction, parts in [("up", Xup_list), ("down", Xdn_list), ("full", Xfull_list)]:
            if not parts:
                continue
            Xpool = np.vstack(parts)
            d, lo, hi = twonn_bootstrap(Xpool)
            results.append({
                "tf": "pooled", "variant": "B_pooled", "direction": direction,
                "p": p, "n": len(Xpool),
                "twonn": d, "ci_lo": lo, "ci_hi": hi,
            })
        if p in [2, 4, 8, 12, 16]:
            n_up   = sum(len(x) for x in Xup_list)
            n_down = sum(len(x) for x in Xdn_list)
            print(f"    B_pooled p={p:2d}: up={n_up}  down={n_down}", flush=True)

    return pd.DataFrame(results)


# ── Графики ──────────────────────────────────────────────────────────────────

DIR_COLORS = {"up": "steelblue", "down": "salmon", "full": "gray"}
DIR_LS     = {"up": "-",         "down": "--",      "full": ":"}


def plot_twonn_curves(df, title, out_path, variant="A"):
    """TwoNN(p) кривые для up/down/full по всем TF (или pooled)."""
    tfs = df["tf"].unique()
    fig, axes = plt.subplots(1, len(tfs), figsize=(6 * len(tfs), 5), squeeze=False)
    fig.suptitle(title, fontsize=11)

    for ax, tf in zip(axes[0], tfs):
        sub = df[df["tf"] == tf]
        for direction in ["full", "up", "down"]:
            s = sub[sub["direction"] == direction].sort_values("p")
            if s.empty:
                continue
            ax.plot(s["p"], s["twonn"],
                    color=DIR_COLORS[direction], ls=DIR_LS[direction],
                    lw=2, marker="o", ms=4, label=direction)
            ax.fill_between(s["p"], s["ci_lo"], s["ci_hi"],
                            color=DIR_COLORS[direction], alpha=0.15)

        ax.set_title(tf, fontsize=10)
        ax.set_xlabel("p (embedding dim)")
        ax.set_ylabel("TwoNN intrinsic d")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax.set_xticks(range(1, P_MAX + 1, 2))

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def plot_fnn_curves(df_fnn, out_path):
    """FNN fractions для up/down/full по TF."""
    tfs = df_fnn["tf"].unique()
    fig, axes = plt.subplots(1, len(tfs), figsize=(6 * len(tfs), 5), squeeze=False)
    fig.suptitle("FNN fraction — Вариант А (sub-sequence)", fontsize=11)

    for ax, tf in zip(axes[0], tfs):
        sub = df_fnn[df_fnn["tf"] == tf]
        for direction in ["full", "up", "down"]:
            s = sub[sub["direction"] == direction].sort_values("p")
            if s.empty:
                continue
            ax.plot(s["p"], s["fnn_frac"],
                    color=DIR_COLORS[direction], ls=DIR_LS[direction],
                    lw=2, marker="o", ms=4, label=direction)
        ax.axhline(0.10, color="black", lw=0.8, ls="--", label="10% threshold")
        ax.set_title(tf)
        ax.set_xlabel("p"); ax.set_ylabel("FNN fraction")
        ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.2)
        ax.set_xticks(range(1, 15, 2))

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def plot_pooled_vs_single(df_pooled, df_single_list, out_path):
    """Pooled TwoNN(p) + overlay одиночных TF."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Вариант Б: TwoNN(p)  pooled (1d+1h+10m) vs отдельные TF", fontsize=11)

    tf_colors = {"1d": "#1f77b4", "1h": "#ff7f0e", "10m": "#2ca02c"}

    for ax, direction in zip(axes, ["full", "up", "down"]):
        # Pooled
        sp = df_pooled[df_pooled["direction"] == direction].sort_values("p")
        if not sp.empty:
            ax.plot(sp["p"], sp["twonn"], color="black", lw=2.5,
                    marker="D", ms=5, label="pooled", zorder=5)
            ax.fill_between(sp["p"], sp["ci_lo"], sp["ci_hi"],
                            color="black", alpha=0.12)

        # Отдельные TF
        for df_s in df_single_list:
            tf_label = df_s["tf"].iloc[0] if not df_s.empty else "?"
            ss = df_s[df_s["direction"] == direction].sort_values("p")
            if ss.empty:
                continue
            ax.plot(ss["p"], ss["twonn"],
                    color=tf_colors.get(tf_label, "gray"), lw=1.2,
                    ls="--", marker=".", ms=4, alpha=0.7,
                    label=tf_label)

        ax.set_title(f"direction = {direction}")
        ax.set_xlabel("p"); ax.set_ylabel("TwoNN d")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
        ax.set_xticks(range(2, P_MAX + 1, 2))

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def plot_mu_distributions(all_z_std, all_up_masks, p=4, out_path=None):
    """
    Распределение mu = r2/r1 для up/down точек (проверка Парето-предположения TwoNN).
    """
    Xup_all, Xdn_all = [], []
    for z_std, up_mask in zip(all_z_std, all_up_masks):
        X, idx = delay_matrix_forward(z_std, p)
        if len(X) == 0:
            continue
        up_here = up_mask[idx]
        Xup_all.append(X[up_here])
        Xdn_all.append(X[~up_here])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Распределение mu = r2/r1  (pooled, p={p})", fontsize=11)

    for ax, (direction, parts) in zip(axes, [("up", Xup_all), ("down", Xdn_all)]):
        if not parts:
            continue
        Xpool = np.vstack(parts)
        _, mu_arr = twonn_dim(Xpool, return_mu=True)
        if isinstance(mu_arr, float):
            continue
        d_hat = 1.0 / np.mean(np.log(mu_arr))

        ax.hist(mu_arr, bins=60, density=True, alpha=0.6,
                color=DIR_COLORS[direction], label=f"empirical  d̂={d_hat:.2f}")
        # Теоретическая Парето(d)
        x = np.linspace(1, mu_arr.max(), 200)
        pareto_pdf = d_hat * x ** (-(d_hat + 1))
        ax.plot(x, pareto_pdf, color="black", lw=1.5, label=f"Pareto(d={d_hat:.2f})")
        ax.set_xlim(1, min(mu_arr.max(), 20))
        ax.set_title(f"{direction}  n={len(Xpool)}")
        ax.set_xlabel("mu = r2/r1"); ax.set_ylabel("density")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

    plt.tight_layout()
    out = out_path or OUT / f"mu_dist_p{p}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ── Сводная таблица ──────────────────────────────────────────────────────────

def print_summary(df_A_all, df_B_pooled, df_fnn_all):
    print(f"\n{'='*65}")
    print("  Вариант А: TwoNN(p=4) sub-sequence  (каждый TF)")
    print(f"{'='*65}")
    print(f"  {'TF':5}  {'direction':10}  {'n':>5}  {'d':>6}  {'95% CI':>14}")
    for _, row in df_A_all[df_A_all["p"] == 4].iterrows():
        ci = f"[{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]"
        print(f"  {row['tf']:5}  {row['direction']:10}  {row['n']:>5}  "
              f"{row['twonn']:>6.3f}  {ci:>14}")

    print(f"\n{'='*65}")
    print("  Вариант Б pooled: TwoNN(p) — фрактальный пул 1d+1h+10m")
    print(f"{'='*65}")
    print(f"  {'p':>3}  {'direction':10}  {'n':>6}  {'d':>6}  {'95% CI':>14}")
    for _, row in df_B_pooled[df_B_pooled["direction"].isin(["up","down"])].iterrows():
        ci = f"[{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]"
        print(f"  {row['p']:>3}  {row['direction']:10}  {row['n']:>6}  "
              f"{row['twonn']:>6.3f}  {ci:>14}")

    print(f"\n{'='*65}")
    print("  FNN: p_min (first p where fnn_frac < 10%)")
    print(f"{'='*65}")
    for tf in df_fnn_all["tf"].unique():
        for direction in ["up", "down", "full"]:
            sub = df_fnn_all[(df_fnn_all["tf"]==tf) &
                             (df_fnn_all["direction"]==direction)].sort_values("p")
            if sub.empty:
                continue
            below = sub[sub["fnn_frac"] < 0.10]
            p_min = int(below["p"].min()) if not below.empty else ">14"
            print(f"  {tf:5}  {direction:10}  p_min={p_min}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print("Загрузка данных SBER ...")
    all_z_std, all_up_masks = [], []
    for tf in TIMEFRAMES:
        z_std, up_mask = load_tf(tf)
        all_z_std.append(z_std)
        all_up_masks.append(up_mask)
        n_up  = up_mask.sum()
        n_dn  = (~up_mask).sum()
        print(f"  {tf:5s}: пивотов={len(z_std):5d}  up={n_up}  down={n_dn}")

    # ── Вариант А ─────────────────────────────────────────────────────────────
    print("\nВариант А (sub-sequence, per TF) ...")
    A_dfs, FNN_dfs = [], []
    for tf, z_std, up_mask in zip(TIMEFRAMES, all_z_std, all_up_masks):
        print(f"  TF={tf}")
        df_a, df_fnn = variant_A(z_std, up_mask, tf)
        A_dfs.append(df_a); FNN_dfs.append(df_fnn)

    df_A_all   = pd.concat(A_dfs,   ignore_index=True)
    df_fnn_all = pd.concat(FNN_dfs, ignore_index=True)
    df_A_all.to_csv(OUT / "variant_A_twonn.csv",   index=False)
    df_fnn_all.to_csv(OUT / "variant_A_fnn.csv",   index=False)

    # ── Вариант Б, одиночные TF ───────────────────────────────────────────────
    print("\nВариант Б (full embedding, per TF) ...")
    B_single_dfs = []
    for tf, z_std, up_mask in zip(TIMEFRAMES, all_z_std, all_up_masks):
        print(f"  TF={tf}")
        df_bs = variant_B_single(z_std, up_mask, tf)
        B_single_dfs.append(df_bs)
    df_B_single = pd.concat(B_single_dfs, ignore_index=True)
    df_B_single.to_csv(OUT / "variant_B_single.csv", index=False)

    # ── Вариант Б, pooled ─────────────────────────────────────────────────────
    print("\nВариант Б pooled (фрактальный пул 1d+1h+10m) ...")
    df_B_pooled = variant_B_pooled(all_z_std, all_up_masks)
    df_B_pooled.to_csv(OUT / "variant_B_pooled.csv", index=False)

    # ── Графики ────────────────────────────────────────────────────────────────
    print("\nГрафики ...")
    plot_twonn_curves(df_A_all,
                      "Вариант А: TwoNN(p) sub-sequence per TF",
                      OUT / "A_twonn_per_tf.png")
    plot_fnn_curves(df_fnn_all, OUT / "A_fnn_per_tf.png")
    plot_pooled_vs_single(df_B_pooled, B_single_dfs,
                          OUT / "B_pooled_vs_single.png")
    for p_check in [4, 8]:
        plot_mu_distributions(all_z_std, all_up_masks, p=p_check,
                              out_path=OUT / f"mu_dist_p{p_check}.png")

    # ── Итоговая таблица ───────────────────────────────────────────────────────
    print_summary(df_A_all, df_B_pooled, df_fnn_all)
    print(f"\nРезультаты: {OUT}")


if __name__ == "__main__":
    main()
