#!/usr/bin/env python3
"""
zigzag_direct_forecast.py

Direct vs iterative S-map multi-step forecast. Tests SSA preprocessing.
Base params: theta=8, p=2, +1h+10m, T=2%, midprice=(O+C)/2.
SSA conditions: no_ssa / k=2 / k=3  (W=256, L=8).
Horizons h=1..4.
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THRESH    = 0.020
TRAIN_WIN = 200
THETA     = 8.0
P         = 2
H_MAX     = 4

SSA_W = 256
SSA_L = 8
SSA_K_LIST = [None, 2, 3]   # None = no SSA

DATA_DIR = Path("/home/kali/workspace/apps/sma/data/candles/SBER")
OUT_DIR  = Path("/home/kali/workspace/apps/sma/research/phase6_attractor")


def load_candles(path):
    with open(path) as f:
        data = json.load(f)
    closes = np.array([c["close"] for c in data], dtype=np.float64)
    opens  = np.array([c["open"]  for c in data], dtype=np.float64)
    dates  = np.array([c["begin"] for c in data])
    return opens, closes, dates


def midprice(opens, closes):
    return (opens + closes) / 2.0


def logtrend_ratio(prices):
    n  = len(prices)
    lp = np.log(prices)
    t  = np.arange(n, dtype=np.float64)
    N  = np.arange(1, n+1, dtype=np.float64)
    St, Sp   = np.cumsum(t), np.cumsum(lp)
    St2, Stp = np.cumsum(t**2), np.cumsum(t*lp)
    den = N*St2 - St**2
    b   = np.where(den > 1e-12, (N*Stp - St*Sp)/den, 0.0)
    a   = (Sp - b*St) / N
    return prices / np.exp(a + b*t)


def ssa_smooth(ratio, W, L, k):
    """Rolling causal SSA. Returns smoothed ratio (same length)."""
    N   = len(ratio)
    out = ratio.copy()
    K_m = W - L + 1
    if K_m < 2 or N < W:
        return out
    idx = np.arange(K_m)[:, None] + np.arange(L)[None, :]
    for t in range(W - 1, N):
        w      = ratio[t - W + 1 : t + 1]
        X      = w[idx]
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        nk     = min(k, len(s))
        out[t] = float(((U[:, :nk] * s[:nk]) @ Vt[:nk, :])[K_m - 1, L - 1])
    return out


def zigzag(ratio, thresh):
    pivots_i, pivots_v = [], []
    direction = None
    li, lv = 0, ratio[0]
    for i in range(1, len(ratio)):
        mv = (ratio[i] - lv) / lv
        if direction is None:
            if abs(mv) >= thresh:
                direction = 'up' if mv > 0 else 'down'
                pivots_i.append(li); pivots_v.append(lv)
                li, lv = i, ratio[i]
        elif direction == 'up':
            if ratio[i] > lv:
                li, lv = i, ratio[i]
            elif (lv - ratio[i]) / lv >= thresh:
                pivots_i.append(li); pivots_v.append(lv)
                direction = 'down'; li, lv = i, ratio[i]
        else:
            if ratio[i] < lv:
                li, lv = i, ratio[i]
            elif (ratio[i] - lv) / lv >= thresh:
                pivots_i.append(li); pivots_v.append(lv)
                direction = 'up'; li, lv = i, ratio[i]
    return np.array(pivots_i, dtype=int), np.array(pivots_v, dtype=np.float64)


def make_X(z, p):
    """X[i] = [z[i], z[i]-z[i-1], ...], NaN for i < p."""
    n = len(z)
    X = np.full((n, p), np.nan)
    for i in range(p, n):
        X[i, 0] = z[i]
        for k in range(p - 1):
            X[i, k+1] = z[i-k] - z[i-k-1]
    return X


def get_pool(step, h, train_start,
             X1d, z1d, ce1h, X1h, z1h, ce10m, X10m, z10m, p):
    """Causal training pool for direct h-step prediction."""
    parts_X, parts_y = [], []

    # 1d: j+h <= step → j in [train_start+p, step-h]
    j1d = np.arange(train_start + p, step - h + 1)
    if len(j1d):
        ok = ~np.any(np.isnan(X1d[j1d]), axis=1)
        if ok.any():
            parts_X.append(X1d[j1d[ok]])
            parts_y.append(z1d[j1d[ok] + h])

    # 1h: j+h < ce1h → j in [p, ce1h-h-1]
    j1h_end = ce1h - h
    if j1h_end > p:
        j1h = np.arange(p, j1h_end)
        ok = ~np.any(np.isnan(X1h[j1h]), axis=1)
        if ok.any():
            parts_X.append(X1h[j1h[ok]])
            parts_y.append(z1h[j1h[ok] + h])

    # 10m: j+h < ce10m → j in [p, ce10m-h-1]
    j10m_end = ce10m - h
    if j10m_end > p:
        j10m = np.arange(p, j10m_end)
        ok = ~np.any(np.isnan(X10m[j10m]), axis=1)
        if ok.any():
            parts_X.append(X10m[j10m[ok]])
            parts_y.append(z10m[j10m[ok] + h])

    if not parts_X:
        return None, None
    return np.vstack(parts_X), np.concatenate(parts_y)


def smap(X_n, y_n, x_q, theta):
    dists = np.sqrt(((X_n - x_q)**2).sum(1))
    d_bar = dists.mean()
    if theta == 0.0 or d_bar < 1e-12:
        w = np.ones(len(y_n))
    else:
        w = np.exp(-theta * dists / d_bar)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(len(y_n)), X_n]) * ws[:, None]
    b  = y_n * ws
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0] + coef[1:] @ x_q)


# ── Load raw data ──────────────────────────────────────────────────────────
print("Загрузка данных …")
o1d, c1d, d1d   = load_candles(DATA_DIR / "1d.json")
o1h, c1h, d1h   = load_candles(DATA_DIR / "1h.json")
o10m, c10m, d10m = load_candles(DATA_DIR / "10m.json")

print(f"  1d bars:{len(c1d)}  1h bars:{len(c1h)}  10m bars:{len(c10m)}")

mp1d  = midprice(o1d,  c1d)
mp1h  = midprice(o1h,  c1h)
mp10m = midprice(o10m, c10m)

# ── Results across SSA conditions ─────────────────────────────────────────
all_results = {}   # ssa_label → {h: rMAE_direct, rMAE_iter, rMAE_m0}

for ssa_k in SSA_K_LIST:
    label = "no_ssa" if ssa_k is None else f"ssa_k={ssa_k}"
    print(f"\n── SSA: {label} ──────────────────────────────────────────")

    # Build ratio per series
    r1d  = logtrend_ratio(mp1d)
    r1h  = logtrend_ratio(mp1h)
    r10m = logtrend_ratio(mp10m)

    if ssa_k is not None:
        print(f"  Применяю SSA(W={SSA_W}, L={SSA_L}, k={ssa_k}) …")
        r1d  = ssa_smooth(r1d,  SSA_W, SSA_L, ssa_k)
        r1h  = ssa_smooth(r1h,  SSA_W, SSA_L, ssa_k)
        r10m = ssa_smooth(r10m, SSA_W, SSA_L, ssa_k)

    pi1d,  z1d  = zigzag(r1d,  THRESH)
    pi1h,  z1h  = zigzag(r1h,  THRESH)
    pi10m, z10m = zigzag(r10m, THRESH)
    pd1d = d1d[pi1d]; pd1h = d1h[pi1h]; pd10m = d10m[pi10m]

    print(f"  пивоты  1d:{len(z1d)}  1h:{len(z1h)}  10m:{len(z10m)}")

    X1d  = make_X(z1d,  P)
    X1h  = make_X(z1h,  P)
    X10m = make_X(z10m, P)

    n1d        = len(z1d)
    test_start = TRAIN_WIN + P
    mean_dz    = np.mean(np.abs(np.diff(z1d)))

    err_direct = {h: [] for h in range(1, H_MAX+1)}
    err_iter   = {h: [] for h in range(1, H_MAX+1)}
    err_m0     = {h: [] for h in range(1, H_MAX+1)}

    for step in range(test_start, n1d - H_MAX):
        curr_date = pd1d[step]
        z_curr    = z1d[step]
        x_q       = X1d[step]
        if np.any(np.isnan(x_q)):
            continue

        ce1h  = int(np.searchsorted(pd1h,  curr_date, side="left"))
        ce10m = int(np.searchsorted(pd10m, curr_date, side="left"))
        ts    = max(0, step - TRAIN_WIN)

        X_h1, y_h1 = get_pool(step, 1, ts,
                               X1d, z1d, ce1h, X1h, z1h, ce10m, X10m, z10m, P)
        if X_h1 is None or len(X_h1) < 5:
            continue

        # ── Direct ──────────────────────────────────────────────────────
        for h in range(1, H_MAX+1):
            actual = z1d[step + h]
            err_m0[h].append(abs(z_curr - actual))

            if h == 1:
                X_h, y_h = X_h1, y_h1
            else:
                X_h, y_h = get_pool(step, h, ts,
                                     X1d, z1d, ce1h, X1h, z1h, ce10m, X10m, z10m, P)
                if X_h is None or len(X_h) < 5:
                    continue

            pred_d = smap(X_h, y_h, x_q, THETA)
            err_direct[h].append(abs(pred_d - actual))

        # ── Iterative ────────────────────────────────────────────────────
        preds = [z_curr]
        p1 = smap(X_h1, y_h1, x_q, THETA)
        preds.append(p1)
        err_iter[1].append(abs(p1 - z1d[step + 1]))

        for h in range(2, H_MAX+1):
            actual   = z1d[step + h]
            x_q_iter = np.array([preds[-1], preds[-1] - preds[-2]])
            pred_h   = smap(X_h1, y_h1, x_q_iter, THETA)
            preds.append(pred_h)
            err_iter[h].append(abs(pred_h - actual))

    # Compute rMAE
    res = {}
    for h in range(1, H_MAX+1):
        res[h] = {
            'direct': np.mean(err_direct[h]) / mean_dz if err_direct[h] else np.nan,
            'iter':   np.mean(err_iter[h])   / mean_dz if err_iter[h]   else np.nan,
            'm0':     np.mean(err_m0[h])      / mean_dz,
        }
    all_results[label] = res

    print(f"  {'h':>2}  {'Direct':>8}  {'Iter':>8}  {'M0':>8}  {'D vs M0':>9}  {'I vs M0':>9}")
    for h in range(1, H_MAX+1):
        d, it, m0 = res[h]['direct'], res[h]['iter'], res[h]['m0']
        print(f"  {h:>2}  {d:>8.3f}  {it:>8.3f}  {m0:>8.3f}  "
              f"{(d/m0-1)*100:>+9.1f}%  {(it/m0-1)*100:>+9.1f}%")

# ── Summary table: h=1 across SSA conditions ──────────────────────────────
print("\n── Сводка h=1: Direct rMAE по SSA-условию ──────────────────────")
print(f"  {'Условие':>12}  {'h=1':>7}  {'h=2':>7}  {'h=3':>7}  {'h=4':>7}")
for label, res in all_results.items():
    vals = "  ".join(f"{res[h]['direct']:>7.3f}" for h in range(1, H_MAX+1))
    print(f"  {label:>12}  {vals}")

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
hs = list(range(1, H_MAX+1))
colors = {'no_ssa': 'gray', 'ssa_k=2': 'steelblue', 'ssa_k=3': 'darkorange'}

for ax, method in zip(axes, ['direct', 'iter']):
    # M0 (same for all SSA conditions)
    m0_vals = [list(all_results.values())[0][h]['m0'] for h in hs]
    ax.plot(hs, m0_vals, 'k--o', lw=1.5, label='M0')

    for label, res in all_results.items():
        vals = [res[h][method] for h in hs]
        ax.plot(hs, vals, '-s', lw=2, color=colors.get(label, 'green'), label=label)

    title = 'Прямой S-map' if method == 'direct' else 'Итеративный S-map'
    ax.set_title(title)
    ax.set_xlabel('Горизонт h (пивотов)')
    ax.set_ylabel('rMAE')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(hs)

fig.suptitle(f'SSA preprocessing  (θ={THETA}, p={P}, midprice, +1h+10m, T={THRESH*100:.0f}%)')
plt.tight_layout()
fig_path = OUT_DIR / "zigzag_direct_forecast.png"
plt.savefig(fig_path, dpi=150)
print(f"\nСохранено: {fig_path}")
print("Готово.")
