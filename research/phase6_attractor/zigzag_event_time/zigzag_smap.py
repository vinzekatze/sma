"""
S-map sweep: θ × p × aug для зигзага в событийном времени.

S-map (Sugihara 1994):
  w_i = exp(-θ · d_i / d̄),  d̄ = mean(d_i) по всем обучающим точкам
  θ=0  → OLS (все точки равны)
  θ→∞  → near-NN (только ближайший сосед)

Отличие от LWR: S-map использует ВСЕ обучающие точки (не ξ ближайших),
поэтому curse-of-dimensionality влияет иначе — проверяем p=1,2,3.

Параметры:
  p     ∈ {1, 2, 3}
  θ     ∈ {0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16}
  aug   ∈ {no_aug, +1h, +1h+10m}  T=2%, no cap
  признаки: X = [z_i, y_i, ..., y_{i-p+1}]  (нормированные)

Оптимизация: X_full строится один раз на (step, aug);
расстояния — один раз на (step, aug, p); θ-sweep по ним бесплатен.

CAUSALITY GATE:  ce = searchsorted(dates_aux, current_date, side="left")
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_BASE    = os.path.dirname(__file__)
DATA_1D  = os.path.join(_BASE, "../../data/candles/SBER/1d.json")
DATA_1H  = os.path.join(_BASE, "../../prototype/data/candles/SBER/1h.json")
DATA_10M = os.path.join(_BASE, "../../prototype/data/candles/SBER/10m.json")
OUT_DIR  = _BASE

THRESHOLD  = 0.02
TRAIN_WIN  = 200
P_GRID     = [1, 2, 3]
P_MAX      = max(P_GRID)
THETA_GRID = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
HORIZONS   = [1, 2, 3, 4]

# наш лучший результат из p_sweep для справки
LWR_BEST_RMAE_H1 = 0.471   # A_p1 + 1h+10m, Gaussian ξ=15

# ── данные ────────────────────────────────────────────────────────────────────

def load_series(path):
    with open(path) as f: data = json.load(f)
    df = pd.DataFrame(data)
    df["close"] = pd.to_numeric(df["close"])
    close = df["close"].values.astype(np.float64)
    trend = logtrend_causal(close)
    ratio = close / trend
    pivots = np.array(find_pivots(ratio, THRESHOLD))
    z  = ratio[pivots]
    dt = np.append(np.diff(pivots)[0], np.diff(pivots)).astype(float)
    dates = pd.to_datetime(df["begin"]).dt.normalize().values[pivots]
    return z, dt, dates

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

# ── признаки ──────────────────────────────────────────────────────────────────

def build_feature_full(z, i):
    """X = [z_i, y_i, y_{i-1}, ..., y_{i-P_MAX+1}], длина P_MAX+1."""
    if i < P_MAX + 1:
        return None
    feats = [z[i]]
    for k in range(P_MAX):
        feats.append(z[i-k] - z[i-k-1])
    return np.array(feats, dtype=np.float64)

# ── S-map ─────────────────────────────────────────────────────────────────────

def smap_predict(X_n, y, x_n, dists, d_bar, theta):
    """
    S-map предсказание с предвычисленными нормированными матрицей и расстояниями.
    X_n, x_n — уже нормированные. dists — расстояния от x_n до каждой строки X_n.
    """
    N = len(X_n)
    if theta == 0:
        w = np.ones(N)
    else:
        w = np.exp(-theta * dists / (d_bar if d_bar > 1e-10 else 1.0))

    w_sqrt = np.sqrt(w)
    Xw = np.column_stack([np.ones(N), X_n]) * w_sqrt[:, None]
    yw = y * w_sqrt
    coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    return float(coef[0] + coef[1:] @ x_n)

# ── walk-forward ──────────────────────────────────────────────────────────────

def walk_forward_smap(z1d, dt1d, dates1d, aug_configs):
    """
    Возвращает:
      results[aug_name][p][theta][h] — массив предсказаний
      res_m0[h]                       — M0
      actuals[h]                      — реальные значения
      train_sizes[aug_name]           — средний размер выборки
    """
    N          = len(z1d)
    test_start = TRAIN_WIN + P_MAX + 2
    test_end   = N - max(HORIZONS)
    n_steps    = test_end - test_start

    aug_names = [name for name, _ in aug_configs]

    results = {
        name: {
            p: {theta: {h: np.full(n_steps, np.nan) for h in HORIZONS}
                for theta in THETA_GRID}
            for p in P_GRID
        }
        for name in aug_names
    }
    res_m0  = {h: np.full(n_steps, np.nan) for h in HORIZONS}
    actuals = {h: np.full(n_steps, np.nan) for h in HORIZONS}
    train_sizes = {name: [] for name in aug_names}

    for step_idx, i in enumerate(range(test_start, test_end)):
        train_s      = i - TRAIN_WIN
        current_date = dates1d[i]

        for h in HORIZONS:
            if i+h < N:
                actuals[h][step_idx] = z1d[i+h]
                res_m0[h][step_idx]  = z1d[i]

        x_q_full = build_feature_full(z1d, i)
        if x_q_full is None:
            continue

        for aug_name, aux_series in aug_configs:
            # ── строим X_full (P_MAX+1 признаков) ────────────────────────
            X_rows = []
            y_dict = {h: [] for h in HORIZONS}

            for j in range(train_s, i):
                xj = build_feature_full(z1d, j)
                if xj is None: continue
                X_rows.append(xj)
                for h in HORIZONS:
                    y_dict[h].append(z1d[j+h] if j+h < N else np.nan)

            for z_s, dt_s, dates_s in aux_series:
                ce  = int(np.searchsorted(dates_s, current_date, side="left"))
                N_s = len(z_s)
                for k in range(P_MAX+1, ce):
                    xk = build_feature_full(z_s, k)
                    if xk is None: continue
                    X_rows.append(xk)
                    for h in HORIZONS:
                        y_dict[h].append(z_s[k+h] if k+h < N_s else np.nan)

            n_train = len(X_rows)
            train_sizes[aug_name].append(n_train)
            if n_train < 4:
                continue

            X_full = np.array(X_rows)   # (n_train, P_MAX+1)

            # ── для каждого p ─────────────────────────────────────────────
            for p in P_GRID:
                dim  = p + 1
                X_p  = X_full[:, :dim]
                x_q  = x_q_full[:dim]

                # нормировка (единая для всех θ)
                mu    = X_p.mean(0)
                sigma = X_p.std(0)
                sigma = np.where(sigma < 1e-10, 1.0, sigma)
                X_n   = (X_p - mu) / sigma
                x_n   = (x_q - mu) / sigma

                # расстояния (единые для всех θ)
                dists = np.sqrt(((X_n - x_n)**2).sum(1))
                d_bar = dists.mean()

                for h in HORIZONS:
                    y_arr = np.array(y_dict[h])
                    mask  = ~np.isnan(y_arr)
                    if mask.sum() < 4:
                        for theta in THETA_GRID:
                            results[aug_name][p][theta][h][step_idx] = z1d[i]
                        continue

                    X_n_m  = X_n[mask]
                    dists_m = dists[mask]
                    y_m    = y_arr[mask]

                    for theta in THETA_GRID:
                        pred = smap_predict(X_n_m, y_m, x_n, dists_m, d_bar, theta)
                        results[aug_name][p][theta][h][step_idx] = pred

    return results, res_m0, actuals, train_sizes

# ── метрики ───────────────────────────────────────────────────────────────────

def rmae_arr(pred, actual, mean_y):
    mask = ~np.isnan(pred) & ~np.isnan(actual)
    if mask.sum() == 0: return np.nan
    return np.abs(pred[mask] - actual[mask]).mean() / mean_y

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Загрузка данных …")
    z1d,  dt1d,  d1d  = load_series(DATA_1D)
    z1h,  dt1h,  d1h  = load_series(DATA_1H)
    z10m, dt10m, d10m = load_series(DATA_10M)
    mean_y = np.mean(np.abs(np.diff(z1d)))

    print(f"  1d:{len(z1d)}  1h:{len(z1h)}  10m:{len(z10m)}")

    aug_configs = [
        ("no_aug",  []),
        ("+1h",     [(z1h, dt1h, d1h)]),
        ("+1h+10m", [(z1h, dt1h, d1h), (z10m, dt10m, d10m)]),
    ]

    print("Walk-forward S-map sweep …", flush=True)
    results, res_m0, actuals, train_sizes = walk_forward_smap(
        z1d, dt1d, d1d, aug_configs)

    print("\nСредний размер обучающей выборки:")
    for name, _ in aug_configs:
        ts = train_sizes[name]
        print(f"  {name:12s}: mean={np.mean(ts):.0f}  max={np.max(ts):.0f}")

    aug_names = [n for n, _ in aug_configs]
    rmae_m0   = {h: rmae_arr(res_m0[h], actuals[h], mean_y) for h in HORIZONS}

    # ── таблицы rMAE ──────────────────────────────────────────────────────────
    rows = []
    for aug_name in aug_names:
        for p in P_GRID:
            for theta in THETA_GRID:
                row = {"aug": aug_name, "p": p, "theta": theta}
                for h in HORIZONS:
                    row[f"h{h}"] = rmae_arr(
                        results[aug_name][p][theta][h], actuals[h], mean_y)
                rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUT_DIR, "results"), exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "results", "zigzag_smap.csv"), index=False)

    print(f"\n  M0  h=1:{rmae_m0[1]:.3f}  h=2:{rmae_m0[2]:.3f}")
    print(f"  LWR лучший (ref): h=1:{LWR_BEST_RMAE_H1:.3f}")

    # best по каждому (aug, p)
    print("\n── Лучший θ для каждого (aug, p)  h=1 ────────────────")
    print(f"  {'aug':12s}  {'p':>3}  {'best_θ':>7}  {'rMAE':>6}  {'vs M0':>7}  {'vs LWR':>7}")
    for aug_name in aug_names:
        for p in P_GRID:
            sub = df[(df.aug==aug_name) & (df.p==p)].sort_values("h1")
            best = sub.iloc[0]
            vs_m0  = (rmae_m0[1] - best.h1) / rmae_m0[1] * 100
            vs_lwr = (LWR_BEST_RMAE_H1 - best.h1) / LWR_BEST_RMAE_H1 * 100
            print(f"  {aug_name:12s}  {p:>3}  {best.theta:>7.2f}  "
                  f"{best.h1:>6.3f}  {vs_m0:>+6.1f}%  {vs_lwr:>+6.1f}%")

    # полная таблица θ vs rMAE для лучшей aug (+1h+10m)
    print("\n── rMAE h=1 по θ  (aug=+1h+10m) ─────────────────────")
    header = f"  {'θ':>6}  " + "  ".join(f"p={p}" for p in P_GRID)
    print(header)
    for theta in THETA_GRID:
        vals = [df[(df.aug=="+1h+10m")&(df.p==p)&(df.theta==theta)]["h1"].values[0]
                for p in P_GRID]
        best_p_idx = int(np.argmin(vals))
        row_str = f"  {theta:>6.2f}  "
        for idx, v in enumerate(vals):
            marker = "← " if idx == best_p_idx else "  "
            row_str += f"{v:.3f}{marker}   "
        print(row_str)

    # ── графики ───────────────────────────────────────────────────────────────
    colors_p   = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    ls_aug     = {"no_aug": ":", "+1h": "--", "+1h+10m": "-"}
    lw_aug     = {"no_aug": 1.0, "+1h": 1.4, "+1h+10m": 2.0}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "S-map sweep θ — SBER 1d+1h+10m  T=2%  |  walk-forward 200",
        fontsize=11)

    for ax, h in zip(axes, [1, 2]):
        ax.axhline(rmae_m0[h], color="#bbbbbb", ls="--", lw=1.4,
                   label=f"M0  {rmae_m0[h]:.3f}")
        if h == 1:
            ax.axhline(LWR_BEST_RMAE_H1, color="#888888", ls="-.", lw=1.4,
                       label=f"LWR best  {LWR_BEST_RMAE_H1:.3f}")

        for aug_name in aug_names:
            for p in P_GRID:
                vals = [df[(df.aug==aug_name)&(df.p==p)&(df.theta==t)][f"h{h}"].values[0]
                        for t in THETA_GRID]
                label = f"p={p} {aug_name}" if aug_name != "no_aug" else f"p={p} no_aug"
                ax.plot(THETA_GRID, vals,
                        color=colors_p[p], ls=ls_aug[aug_name], lw=lw_aug[aug_name],
                        marker="o", ms=4, label=label, alpha=0.85)

        ax.set_title(f"h = {h} пивот{'а' if h>1 else ''} вперёд", fontsize=10)
        ax.set_xlabel("θ  (S-map локальность)", fontsize=9)
        ax.set_ylabel("rMAE", fontsize=9)
        ax.set_xscale("symlog", linthresh=0.2)
        ax.set_xticks(THETA_GRID)
        ax.set_xticklabels([str(t) for t in THETA_GRID], fontsize=7)
        ax.legend(fontsize=6.5, ncol=2)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "zigzag_smap.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nСохранено: {out}")
    plt.close(fig)
    print("Готово.")


if __name__ == "__main__":
    main()
