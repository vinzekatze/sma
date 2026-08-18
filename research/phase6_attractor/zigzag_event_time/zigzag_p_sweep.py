"""
Sweep оптимальной размерности p (n_lags) с расширенной обучающей выборкой.

Первичный ряд: SBER 1d, T=2%.
Аугментация:
  +1h:      SBER 1h,  T=2% (все каузальные пивоты, без cap)
  +1h+10m:  SBER 1h + SBER 10m, T=2% (все каузальные, без cap)

Признаки: X = [z_i, y_i, y_{i-1}, ..., y_{i-p+1}]
  dim(p) = p + 1  (z + p разностей)
  ξ(p)   = max(15, 3·(p + 2))

Оптимизация: для каждого шага walk-forward строится X_full с P_MAX лагами
один раз, далее X_p = X_full[:, :p+1] без пересчёта.

CAUSALITY GATE (критично):
  для каждого вспомогательного ряда независимо:
    ce = searchsorted(dates_aux, current_date, side="left")
  Алгоритм не видит ни одного aux-пивота с датой >= даты текущего 1d-пивота.
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_BASE      = os.path.dirname(__file__)
DATA_1D    = os.path.join(_BASE, "../../data/candles/SBER/1d.json")
DATA_1H    = os.path.join(_BASE, "../../prototype/data/candles/SBER/1h.json")
DATA_10M   = os.path.join(_BASE, "../../prototype/data/candles/SBER/10m.json")
OUT_DIR    = _BASE

THRESHOLD  = 0.02
TRAIN_WIN  = 200
P_GRID     = [1, 2, 3, 4, 5, 6, 8, 10]
P_MAX      = max(P_GRID)          # 10
HORIZONS   = [1, 2, 3, 4]

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
    """Вектор признаков длины P_MAX+1: [z_i, y_i, ..., y_{i-P_MAX+1}].
    Требует i >= P_MAX + 1."""
    if i < P_MAX + 1:
        return None
    feats = [z[i]]
    for k in range(P_MAX):
        feats.append(z[i-k] - z[i-k-1])
    return np.array(feats, dtype=np.float64)

def xi_for(p):
    return max(15, 3 * (p + 2))

# ── LWR ───────────────────────────────────────────────────────────────────────

def lwr_predict(X_tr, y_tr, x_q, xi):
    N = len(X_tr); xi = min(xi, N - 1)
    mu = X_tr.mean(0); sigma = X_tr.std(0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    X_n = (X_tr - mu)/sigma; x_n = (x_q - mu)/sigma
    dists = np.sqrt(((X_n - x_n)**2).sum(1))
    idx   = np.argsort(dists)[:xi]
    bw    = dists[idx[-1]] + 1e-10
    w     = np.exp(-0.5*(dists[idx]/bw)**2)
    Xw    = np.column_stack([np.ones(xi), X_n[idx]]) * np.sqrt(w)[:,None]
    yw    = y_tr[idx] * np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    return float(coef[0] + coef[1:] @ x_n)

# ── walk-forward sweep ────────────────────────────────────────────────────────

def walk_forward_sweep(z1d, dt1d, dates1d, aug_configs):
    """
    aug_configs: list of (name, [(z_s, dt_s, dates_s), ...])
      CAUSALITY GATE: для каждого aux-ряда отдельно, без ограничения cap.

    Возвращает:
      results[aug_name][p][h] — массив предсказаний
      actuals[h]              — массив реальных значений
      train_sizes[aug_name]   — среднее число точек в обучающей выборке
    """
    N          = len(z1d)
    test_start = TRAIN_WIN + P_MAX + 2
    test_end   = N - max(HORIZONS)
    n_steps    = test_end - test_start

    # инициализация структур
    aug_names = [name for name, _ in aug_configs]
    results   = {
        name: {p: {h: np.full(n_steps, np.nan) for h in HORIZONS}
               for p in P_GRID}
        for name in aug_names
    }
    results_m0 = {h: np.full(n_steps, np.nan) for h in HORIZONS}
    actuals    = {h: np.full(n_steps, np.nan) for h in HORIZONS}
    train_sizes = {name: [] for name in aug_names}

    for step_idx, i in enumerate(range(test_start, test_end)):
        train_s      = i - TRAIN_WIN
        current_date = dates1d[i]

        # actuals и M0
        for h in HORIZONS:
            if i+h < N:
                actuals[h][step_idx]    = z1d[i+h]
                results_m0[h][step_idx] = z1d[i]

        x_q_full = build_feature_full(z1d, i)
        if x_q_full is None:
            continue

        # ── строим X для каждой aug-конфигурации ──────────────────────────
        for aug_name, aux_series in aug_configs:
            # 1d тренировочные точки
            X_rows, y_dict = [], {h: [] for h in HORIZONS}
            for j in range(train_s, i):
                xj = build_feature_full(z1d, j)
                if xj is None: continue
                X_rows.append(xj)
                for h in HORIZONS:
                    y_dict[h].append(z1d[j+h] if j+h < N else np.nan)

            # aux ряды (causality gate, NO cap)
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

            if n_train < 2:
                continue

            X_full = np.array(X_rows)   # (n_train, P_MAX+1)

            # ── для каждого p: срезаем и вызываем LWR ─────────────────────
            for p in P_GRID:
                dim = p + 1
                xi  = xi_for(p)
                X_p = X_full[:, :dim]
                x_q = x_q_full[:dim]

                for h in HORIZONS:
                    y_arr = np.array(y_dict[h])
                    mask  = ~np.isnan(y_arr)
                    if mask.sum() < xi + 1:
                        results[aug_name][p][h][step_idx] = z1d[i]
                        continue
                    pred = lwr_predict(X_p[mask], y_arr[mask], x_q, xi)
                    results[aug_name][p][h][step_idx] = pred

    return results, results_m0, actuals, train_sizes

# ── метрики ───────────────────────────────────────────────────────────────────

def rmae_arr(pred, actual, mean_y):
    mask = ~np.isnan(pred) & ~np.isnan(actual)
    return np.abs(pred[mask] - actual[mask]).mean() / mean_y

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Загрузка данных …")
    z1d,  dt1d,  d1d  = load_series(DATA_1D)
    z1h,  dt1h,  d1h  = load_series(DATA_1H)
    z10m, dt10m, d10m = load_series(DATA_10M)
    mean_y = np.mean(np.abs(np.diff(z1d)))

    print(f"  1d:  {len(z1d):4d} пивотов")
    print(f"  1h:  {len(z1h):4d} пивотов  (T=2%)")
    print(f"  10m: {len(z10m):4d} пивотов  (T=2%)")
    print(f"  P_MAX={P_MAX}, test_start={TRAIN_WIN+P_MAX+2}")

    aug_configs = [
        ("no_aug",   []),
        ("+1h",      [(z1h,  dt1h,  d1h)]),
        ("+1h+10m",  [(z1h,  dt1h,  d1h), (z10m, dt10m, d10m)]),
    ]

    print("\nWalk-forward sweep …", flush=True)
    results, res_m0, actuals, train_sizes = walk_forward_sweep(
        z1d, dt1d, d1d, aug_configs)

    # ── средний размер обучающей выборки ──────────────────────────────────
    print("\nСредний размер обучающей выборки (per step):")
    for name in [n for n,_ in aug_configs]:
        ts = train_sizes[name]
        if ts:
            print(f"  {name:12s}: mean={np.mean(ts):.0f}  "
                  f"min={np.min(ts):.0f}  max={np.max(ts):.0f}")

    # ── таблица rMAE ──────────────────────────────────────────────────────
    aug_names = [n for n,_ in aug_configs]
    rmae_m0 = {h: rmae_arr(res_m0[h], actuals[h], mean_y) for h in HORIZONS}

    rows = []
    for aug_name in aug_names:
        for p in P_GRID:
            row = {"aug": aug_name, "p": p}
            for h in HORIZONS:
                row[f"h{h}"] = rmae_arr(results[aug_name][p][h], actuals[h], mean_y)
            rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUT_DIR, "results"), exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "results", "zigzag_p_sweep.csv"), index=False)

    print(f"\n  M0   h=1:{rmae_m0[1]:.3f}  h=2:{rmae_m0[2]:.3f}")
    print()
    for h in [1, 2]:
        print(f"── rMAE h={h} ─────────────────────────────────────────")
        header = f"  {'p':>4}  " + "  ".join(f"{n:>12}" for n in aug_names)
        print(header)
        for p in P_GRID:
            vals = [df[(df.aug==n)&(df.p==p)][f"h{h}"].values[0]
                    for n in aug_names]
            best_idx = int(np.argmin(vals))
            row_str = f"  {p:>4}  "
            for idx, v in enumerate(vals):
                marker = " ←" if idx == best_idx else "  "
                row_str += f"{v:.3f}{marker}      "
            print(row_str)
        # лучший p по условию
        for n in aug_names:
            sub = df[(df.aug==n)].sort_values(f"h{h}")
            best_p = sub.iloc[0]["p"]
            best_v = sub.iloc[0][f"h{h}"]
            print(f"    {n}: лучший p={int(best_p)},  rMAE={best_v:.3f}  "
                  f"(vs M0: {(rmae_m0[h]-best_v)/rmae_m0[h]*100:+.1f}%)")
        print()

    # ── графики ───────────────────────────────────────────────────────────
    colors = {"no_aug": "#555555", "+1h": "#1f77b4", "+1h+10m": "#d62728"}
    markers = {"no_aug": "s", "+1h": "o", "+1h+10m": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Sweep размерности p — SBER 1d T=2%  |  A_p  |  walk-forward 200  "
        f"|  T_aug=2%  (no cap)",
        fontsize=11)

    for ax, h in zip(axes, [1, 2]):
        ax.axhline(rmae_m0[h], color="#bbbbbb", ls="--", lw=1.5,
                   label=f"M0  {rmae_m0[h]:.3f}")
        for aug_name in aug_names:
            sub = df[df.aug==aug_name].sort_values("p")
            vals = sub[f"h{h}"].values
            ps   = sub["p"].values
            ax.plot(ps, vals, color=colors[aug_name],
                    marker=markers[aug_name], ms=7, lw=1.8,
                    label=f"{aug_name}")
            # аннотация минимума
            best_i = int(np.argmin(vals))
            ax.annotate(f"p={int(ps[best_i])}\n{vals[best_i]:.3f}",
                        (ps[best_i], vals[best_i]),
                        textcoords="offset points", xytext=(0, -22),
                        fontsize=7.5, color=colors[aug_name], ha="center")

        ax.set_title(f"h = {h} пивот{'а' if h>1 else ''} вперёд", fontsize=10)
        ax.set_xlabel("n_lags  p", fontsize=9)
        ax.set_ylabel("rMAE", fontsize=9)
        ax.set_xticks(P_GRID)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ymin = min(rmae_m0[h], df[f"h{h}"].min()) * 0.96
        ymax = max(rmae_m0[h], df[f"h{h}"].max()) * 1.02
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "zigzag_p_sweep.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Сохранено: {out}")
    plt.close(fig)
    print("Готово.")


if __name__ == "__main__":
    main()
