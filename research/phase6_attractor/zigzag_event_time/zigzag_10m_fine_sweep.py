"""
Тонкий sweep порога зигзага на 10m аугментации.

Цель: найти, есть ли несколько локальных минимумов rMAE на 10m данных
(«фрактальные уровни»). Предыдущий грубый sweep {1%..8%} показал минимум
в районе 2% — проверяем структуру подробнее.

Первичный ряд: SBER 1d, порог 2% (фиксировано).
Аугментация: SBER 10m, пороги THRESH_GRID_10M.
Модель: A_p2 (n_lags=2, ξ=15), walk-forward с окном 200 пивотов.

Ограничение: MAX_AUG_PIVOTS — берётся не более последних N каузальных 10m пивотов,
чтобы тонкие пороги (0.3-0.5%) с тысячами пивотов не замедляли расчёт.
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_BASE        = os.path.dirname(__file__)
DATA_1D      = os.path.join(_BASE, "../../data/candles/SBER/1d.json")
DATA_10M     = os.path.join(_BASE, "../../prototype/data/candles/SBER/10m.json")
OUT_DIR      = _BASE

THRESH_1D    = 0.02

# мелкая сетка: от 0.3% до 10%
THRESH_GRID_10M = [
    0.003, 0.004, 0.005, 0.006, 0.007,
    0.008, 0.010, 0.012, 0.015,
    0.017, 0.020, 0.025, 0.030,
    0.040, 0.050, 0.060, 0.080, 0.100
]

TRAIN_WIN       = 200
HORIZONS        = [1, 2, 3, 4]
N_LAGS          = 2     # A_p2
XI              = 15
MAX_AUG_PIVOTS  = 3000  # не более последних N каузальных пивотов aug


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
    A_p2 на первичном 1d ряду + опциональная каузальная аугментация.
    Берёт не более MAX_AUG_PIVOTS последних каузально допустимых пивотов aug.
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

        X_list = []; y_dict = {h: [] for h in HORIZONS}

        # 1d тренировочные точки
        for j in range(train_s, i):
            xj = build_feature(z1d, dt1d, j, N_LAGS)
            if xj is None: continue
            X_list.append(xj)
            for h in HORIZONS:
                y_dict[h].append(z1d[j+h] if j+h < N else np.nan)

        # аугментация (causally gated + cap)
        if use_aug:
            ce = int(np.searchsorted(dates_aug, current_date, side="left"))
            # берём не более MAX_AUG_PIVOTS последних каузальных пивотов
            aug_start = max(N_LAGS + 1, ce - MAX_AUG_PIVOTS)
            N_a = len(z_aug)
            for k in range(aug_start, ce):
                xk = build_feature(z_aug, dt_aug, k, N_LAGS)
                if xk is None: continue
                X_list.append(xk)
                for h in HORIZONS:
                    y_dict[h].append(z_aug[k+h] if k+h < N_a else np.nan)

        if len(X_list) < XI + 1:
            for h in HORIZONS: preds[h].append(z1d[i]); continue

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


def calc_rmae(preds, actuals, mean_y):
    out = {}
    for h in HORIZONS:
        mask = ~np.isnan(actuals[h]) & ~np.isnan(preds[h])
        out[h] = np.abs(preds[h][mask] - actuals[h][mask]).mean() / mean_y
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Загрузка данных …")
    df_1d  = load_candles(DATA_1D)
    df_10m = load_candles(DATA_10M)

    z1d, dt1d, dates1d = build_series(df_1d, THRESH_1D)
    mean_y = np.mean(np.abs(np.diff(z1d)))
    print(f"  1d  {THRESH_1D*100:.0f}%: {len(z1d)} пивотов,  mean|Δz|={mean_y:.4f}")
    print(f"  10m записей: {len(df_10m)}")

    # ── baseline ──────────────────────────────────────────────────────────────
    print("\nBaseline (без aug) …", flush=True)
    p_base, act = walk_forward_a2(z1d, dt1d, dates1d)
    n_test = len(act[1])

    test_start = TRAIN_WIN + N_LAGS + 2
    m0_arr = z1d[test_start : test_start + n_test]
    rmae_m0 = {}
    for h in HORIZONS:
        mask = ~np.isnan(act[h])
        rmae_m0[h] = np.abs(m0_arr[:mask.sum()] - act[h][mask]).mean() / mean_y

    # пересчёт rmae_m0 корректно
    for h in HORIZONS:
        mask = ~np.isnan(act[h])
        act_h = act[h][mask]
        # M0 — предсказываем z1d[i] для каждой тестовой позиции i
        m0_vals = []
        for i in range(test_start, test_start + len(act[h])):
            m0_vals.append(z1d[i])
        m0_arr_h = np.array(m0_vals)[mask]
        rmae_m0[h] = np.abs(m0_arr_h - act_h).mean() / mean_y

    rmae_base = calc_rmae(p_base, act, mean_y)
    print(f"  M0      h=1: {rmae_m0[1]:.3f}")
    print(f"  A_p2 no aug h=1: {rmae_base[1]:.3f}")

    # ── sweep ─────────────────────────────────────────────────────────────────
    print(f"\nSweep 10m  (MAX_AUG_PIVOTS={MAX_AUG_PIVOTS}) …")
    print(f"  Всего порогов: {len(THRESH_GRID_10M)}")
    results = []

    for T in THRESH_GRID_10M:
        z_a, dt_a, dates_a = build_series(df_10m, T)
        n_piv = len(z_a)
        print(f"  T={T*100:.1f}%: {n_piv:5d} пивотов", end="  ", flush=True)
        p, _ = walk_forward_a2(z1d, dt1d, dates1d, z_a, dt_a, dates_a)
        r    = calc_rmae(p, act, mean_y)
        print(f"rMAE h=1={r[1]:.3f}  h=2={r[2]:.3f}")
        results.append({
            "thresh": T,
            "n_pivots": n_piv,
            **{f"h{h}": r[h] for h in HORIZONS}
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUT_DIR, "results", "zigzag_10m_fine_sweep.csv"), index=False)

    # ── итоговая таблица ──────────────────────────────────────────────────────
    print("\n── rMAE по порогам (SBER 10m аугментация) ──")
    print(f"{'T':>7}  {'n_piv':>6}  {'h=1':>6}  {'h=2':>6}  {'vs_base':>8}")
    for _, row in df_res.iterrows():
        diff = row.h1 - rmae_base[1]
        marker = " ←MIN" if row.h1 == df_res.h1.min() else ""
        print(f"  {row.thresh*100:4.1f}%  {row.n_pivots:6.0f}  "
              f"{row.h1:.3f}  {row.h2:.3f}  "
              f"{diff:+.3f}{marker}")
    print(f"\n  A_p2 no aug:  h=1={rmae_base[1]:.3f}  h=2={rmae_base[2]:.3f}")
    print(f"  M0:           h=1={rmae_m0[1]:.3f}  h=2={rmae_m0[2]:.3f}")

    # ── графики ───────────────────────────────────────────────────────────────
    thresh_pct = [T * 100 for T in THRESH_GRID_10M]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Тонкий sweep порога 10m аугментации — SBER  |  A_p2  |  1d 2%  |  walk-forward 200",
        fontsize=11)

    for ax, h in zip(axes.flat, HORIZONS):
        ax2r = ax.twinx()

        # правая ось: число пивотов (серый)
        ax2r.bar(thresh_pct, df_res.n_pivots.values,
                 width=[0.002 * 100] * len(thresh_pct),
                 color="#cccccc", alpha=0.5, label="n_pivots")
        ax2r.set_ylabel("число пивотов", fontsize=8, color="#aaaaaa")
        ax2r.tick_params(axis="y", labelcolor="#aaaaaa", labelsize=7)

        # левая ось: rMAE
        ax.plot(thresh_pct, df_res[f"h{h}"].values,
                color="#1f4e8c", marker="o", ms=5, lw=1.8, zorder=3)

        # подсвечиваем лучший порог
        best_idx = df_res[f"h{h}"].values.argmin()
        ax.scatter([thresh_pct[best_idx]], [df_res[f"h{h}"].values[best_idx]],
                   color="#d62728", s=60, zorder=4,
                   label=f"min {df_res[f'h{h}'].values[best_idx]:.3f} @ {thresh_pct[best_idx]:.1f}%")

        ax.axhline(rmae_m0[h],   color="#888888", ls="--", lw=1.2,
                   label=f"M0 {rmae_m0[h]:.3f}")
        ax.axhline(rmae_base[h], color="#444444", ls="--", lw=1.4,
                   label=f"no aug {rmae_base[h]:.3f}")
        ax.axvline(2.0, color="#999999", ls=":", lw=1.0, alpha=0.7)  # маркер предыдущего оптимума

        ax.set_title(f"h = {h} пивот{'а' if h>1 else ''} вперёд", fontsize=9)
        ax.set_xlabel("порог 10m зигзага (%)", fontsize=8)
        ax.set_ylabel("rMAE", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(thresh_pct)
        ax.set_xticklabels([f"{v:.1f}" for v in thresh_pct], fontsize=6.5, rotation=45)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2, zorder=0)

        ymin = min(rmae_m0[h], rmae_base[h], df_res[f"h{h}"].min()) * 0.97
        ymax = max(rmae_m0[h], df_res[f"h{h}"].max()) * 1.02
        ax.set_ylim(ymin, ymax)
        ax.set_zorder(ax2r.get_zorder() + 1)
        ax.patch.set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "zigzag_10m_fine_sweep.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nСохранено: {out}")
    plt.close(fig)
    print("Готово.")


if __name__ == "__main__":
    import sys, os as _os
    _os.makedirs(os.path.join(_BASE, "results"), exist_ok=True)
    main()
