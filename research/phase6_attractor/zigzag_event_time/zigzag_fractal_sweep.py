"""
Фрактальный sweep: влияние порога зигзага в аугментирующем ряду
на точность прогноза 1d пивотов.

Первичный ряд: SBER 1d, порог 2% (фиксировано).
Аугментация:
  - SBER 1h при порогах T ∈ THRESH_GRID
  - SBER 10m при порогах T ∈ THRESH_GRID

Для каждого (series, T): walk_forward(A_p2, 1d + series_T), rMAE h=1..4.
Causality gate: searchsorted(dates_aug < dates_1d[i]) для каждого шага.

Результат: кривые rMAE(T) для 1h и 10m — где минимум, там масштаб
фрактально близок к 1d 2%.
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_BASE        = os.path.dirname(__file__)
DATA_1D      = os.path.join(_BASE, "../../data/candles/SBER/1d.json")
DATA_1H      = os.path.join(_BASE, "../../prototype/data/candles/SBER/1h.json")
DATA_10M     = os.path.join(_BASE, "../../prototype/data/candles/SBER/10m.json")
OUT_DIR      = _BASE

THRESH_1D    = 0.02
THRESH_GRID  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
TRAIN_WIN    = 200
HORIZONS     = [1, 2, 3, 4]
N_LAGS       = 2          # A_p2
XI           = 15

# ── общие функции ─────────────────────────────────────────────────────────────

def load_candles(path):
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["close"] = pd.to_numeric(df["close"])
    df["date"]  = pd.to_datetime(df["begin"]).dt.normalize()
    return df

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
            if abs(v - ext_val) >= thr * ext_val:
                direction = 1 if v > ext_val else -1
                ext_val, ext_idx = v, i
        elif direction == 1:
            if v > ext_val: ext_val, ext_idx = v, i
            elif (ext_val - v) >= thr * ext_val:
                pivots.append(ext_idx); direction = -1; ext_val, ext_idx = v, i
        else:
            if v < ext_val: ext_val, ext_idx = v, i
            elif (v - ext_val) >= thr * ext_val:
                pivots.append(ext_idx); direction = 1; ext_val, ext_idx = v, i
    return pivots

def build_series(df, threshold):
    """Возвращает (z, dt, dates_piv) для заданного порога."""
    close   = df["close"].values.astype(np.float64)
    trend   = logtrend_causal(close)
    ratio   = close / trend
    pivots  = find_pivots(ratio, threshold)
    piv_arr = np.array(pivots)
    z       = ratio[piv_arr]
    dt_raw  = np.diff(piv_arr).astype(float)
    dt      = np.append(dt_raw[0], dt_raw)
    dates   = df["date"].values[piv_arr]
    return z, dt, dates

def build_feature(z, dt, i, n_lags):
    if i < n_lags + 1:
        return None
    feats = [z[i]]
    for k in range(n_lags):
        feats.append(z[i-k] - z[i-k-1])
    return np.array(feats, dtype=np.float64)

def lwr_predict(X_tr, y_tr, x_q, xi):
    N   = len(X_tr)
    xi  = min(xi, N - 1)
    mu  = X_tr.mean(0); sigma = X_tr.std(0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    X_n = (X_tr - mu) / sigma; x_n = (x_q - mu) / sigma
    dists = np.sqrt(((X_n - x_n)**2).sum(1))
    idx   = np.argsort(dists)[:xi]
    bw    = dists[idx[-1]] + 1e-10
    w     = np.exp(-0.5 * (dists[idx] / bw)**2)
    Xw    = np.column_stack([np.ones(xi), X_n[idx]]) * np.sqrt(w)[:, None]
    yw    = y_tr[idx] * np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    return float(coef[0] + coef[1:] @ x_n)

# ── walk-forward A_p2 с одним аугментирующим рядом ───────────────────────────

def walk_forward_a2(z1d, dt1d, dates1d, z_aug=None, dt_aug=None, dates_aug=None):
    """
    Прогон A_p2 (n_lags=2, ξ=15) на первичном 1d ряду.
    Если z_aug передан — добавляет каузально отфильтрованные пивоты аугментации.
    Возвращает (predictions_h1..h4, actuals_h1..h4).
    """
    N           = len(z1d)
    test_start  = TRAIN_WIN + N_LAGS + 2
    test_end    = N - max(HORIZONS)
    use_aug     = z_aug is not None

    preds   = {h: [] for h in HORIZONS}
    actuals = {h: [] for h in HORIZONS}

    for i in range(test_start, test_end):
        train_s      = i - TRAIN_WIN
        current_date = dates1d[i]

        for h in HORIZONS:
            actuals[h].append(z1d[i+h] if i+h < N else np.nan)

        x_q = build_feature(z1d, dt1d, i, N_LAGS)
        if x_q is None:
            for h in HORIZONS: preds[h].append(z1d[i])
            continue

        # 1d тренировочные точки
        X_list = []; y_dict = {h: [] for h in HORIZONS}
        for j in range(train_s, i):
            xj = build_feature(z1d, dt1d, j, N_LAGS)
            if xj is None: continue
            X_list.append(xj)
            for h in HORIZONS:
                y_dict[h].append(z1d[j+h] if j+h < N else np.nan)

        # аугментация (causally gated)
        if use_aug:
            ce   = int(np.searchsorted(dates_aug, current_date, side="left"))
            N_a  = len(z_aug)
            for k in range(N_LAGS + 1, ce):
                xk = build_feature(z_aug, dt_aug, k, N_LAGS)
                if xk is None: continue
                X_list.append(xk)
                for h in HORIZONS:
                    y_dict[h].append(z_aug[k+h] if k+h < N_a else np.nan)

        if len(X_list) < XI + 1:
            for h in HORIZONS: preds[h].append(z1d[i])
            continue

        X_tr = np.array(X_list)
        for h in HORIZONS:
            y_arr = np.array(y_dict[h]); mask = ~np.isnan(y_arr)
            if mask.sum() < XI + 1:
                preds[h].append(z1d[i]); continue
            preds[h].append(lwr_predict(X_tr[mask], y_arr[mask], x_q, XI))

    for h in HORIZONS:
        preds[h]   = np.array(preds[h])
        actuals[h] = np.array(actuals[h])
    return preds, actuals

def rmae(preds, actuals, z):
    mean_y = np.mean(np.abs(np.diff(z)))
    out = {}
    for h in HORIZONS:
        mask = ~np.isnan(actuals[h]) & ~np.isnan(preds[h])
        out[h] = np.abs(preds[h][mask] - actuals[h][mask]).mean() / mean_y
    return out

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── загрузка ──────────────────────────────────────────────────────────────
    print("Загрузка данных …")
    df_1d  = load_candles(DATA_1D)
    df_1h  = load_candles(DATA_1H)
    df_10m = load_candles(DATA_10M)

    z1d, dt1d, dates1d = build_series(df_1d, THRESH_1D)
    print(f"  1d  порог {THRESH_1D*100:.0f}%: {len(z1d)} пивотов")

    # ── baseline: без аугментации ─────────────────────────────────────────────
    print("\nBaseline (без aug) …", flush=True)
    p_base, act = walk_forward_a2(z1d, dt1d, dates1d)
    m0_preds    = {h: z1d[TRAIN_WIN + N_LAGS + 2 : len(z1d) - max(HORIZONS)]
                   for h in HORIZONS}
    # M0 нужен отдельно — вычислим из actuals
    mean_y = np.mean(np.abs(np.diff(z1d)))
    rmae_base = rmae(p_base, act, z1d)
    rmae_m0   = {h: np.abs(z1d[TRAIN_WIN+N_LAGS+2:len(z1d)-max(HORIZONS)]
                            - act[h][~np.isnan(act[h])]).mean() / mean_y
                 for h in HORIZONS}
    # проще: M0 = z1d[i] для каждого test i
    m0_vals = []
    for i in range(TRAIN_WIN+N_LAGS+2, len(z1d)-max(HORIZONS)):
        m0_vals.append(z1d[i])
    m0_arr = np.array(m0_vals)
    rmae_m0 = {}
    for h in HORIZONS:
        mask = ~np.isnan(act[h])
        rmae_m0[h] = np.abs(m0_arr[mask] - act[h][mask]).mean() / mean_y

    print(f"  M0  h=1: {rmae_m0[1]:.3f}")
    print(f"  A_p2 (no aug) h=1: {rmae_base[1]:.3f}")

    # ── sweep ─────────────────────────────────────────────────────────────────
    results = []   # list of dicts

    for series_name, df_aug in [("1h", df_1h), ("10m", df_10m)]:
        print(f"\nSweep {series_name} …")
        for T in THRESH_GRID:
            z_a, dt_a, dates_a = build_series(df_aug, T)
            n_piv = len(z_a)
            print(f"  {series_name} T={T*100:.0f}%: {n_piv} пивотов", end="  ", flush=True)
            p, _ = walk_forward_a2(z1d, dt1d, dates1d, z_a, dt_a, dates_a)
            r    = rmae(p, act, z1d)
            print(f"rMAE h=1={r[1]:.3f}")
            results.append({
                "series": series_name, "thresh": T,
                "n_pivots": n_piv,
                **{f"h{h}": r[h] for h in HORIZONS}
            })

    df_res = pd.DataFrame(results)

    # ── итоговая таблица ──────────────────────────────────────────────────────
    print("\nСводная таблица rMAE h=1:")
    print(f"{'':>12}  {'1h':>8}  {'10m':>8}")
    for T in THRESH_GRID:
        r1h  = df_res[(df_res.series=="1h")  & (df_res.thresh==T)]["h1"].values[0]
        r10m = df_res[(df_res.series=="10m") & (df_res.thresh==T)]["h1"].values[0]
        print(f"  T={T*100:4.1f}%:  {r1h:.3f}    {r10m:.3f}")
    print(f"\n  baseline A_p2 (no aug): {rmae_base[1]:.3f}")
    print(f"  M0:                     {rmae_m0[1]:.3f}")

    # ── графики ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Фрактальный sweep: rMAE(порог аугментации) — SBER 1d 2%  |  A_p2  |  walk-forward 200",
        fontsize=11)

    thresh_pct = [T * 100 for T in THRESH_GRID]

    for ax, h in zip(axes, [1, 2]):
        # reference lines
        ax.axhline(rmae_m0[h],   color="#aaaaaa", ls="--", lw=1.5,
                   label=f"M0 ({rmae_m0[h]:.3f})")
        ax.axhline(rmae_base[h], color="#333333", ls="--", lw=1.5,
                   label=f"A_p2 no aug ({rmae_base[h]:.3f})")

        for series_name, color, marker in [("1h", "#1f77b4", "o"), ("10m", "#d62728", "s")]:
            sub  = df_res[df_res.series == series_name].sort_values("thresh")
            vals = sub[f"h{h}"].values
            npiv = sub["n_pivots"].values
            ax.plot(thresh_pct, vals, color=color, marker=marker,
                    lw=1.6, ms=6, label=series_name)
            # аннотация количества пивотов у первой и последней точки
            for idx in [0, len(thresh_pct)-1]:
                ax.annotate(f"{npiv[idx]}p",
                            (thresh_pct[idx], vals[idx]),
                            textcoords="offset points", xytext=(0, 7),
                            fontsize=7, color=color, ha="center")

        ax.set_title(f"h = {h} пивота вперёд", fontsize=10)
        ax.set_xlabel("порог зигзага аугментации (%)", fontsize=9)
        ax.set_ylabel("rMAE", fontsize=9)
        ax.set_xticks(thresh_pct)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        # не обрезаем снизу — чтобы видеть улучшение относительно baseline
        ymin = min(rmae_m0[h], rmae_base[h],
                   df_res[f"h{h}"].min()) * 0.95
        ymax = max(rmae_m0[h], df_res[f"h{h}"].max()) * 1.03
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "zigzag_fractal_sweep.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nСохранено: {out}")
    plt.close(fig)

    # Дополнительный график: кол-во пивотов по порогам
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.suptitle("Количество пивотов по порогу и серии", fontsize=11)
    for series_name, color, marker in [("1h", "#1f77b4", "o"), ("10m", "#d62728", "s")]:
        sub = df_res[df_res.series == series_name].sort_values("thresh")
        ax2.plot(thresh_pct, sub["n_pivots"].values,
                 color=color, marker=marker, lw=1.4, ms=6, label=series_name)
    ax2.set_xlabel("порог (%)", fontsize=9)
    ax2.set_ylabel("число пивотов", fontsize=9)
    ax2.set_xticks(thresh_pct)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "zigzag_fractal_pivots.png")
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Сохранено: {out2}")
    plt.close(fig2)

    print("\nГотово.")

if __name__ == "__main__":
    main()
