"""
LP dimension sweep n=3, d от 15 до 100, чанковый SVD (память ~100MB).
Бинарный поиск нулевого перехода ACF(lag=1) с нижней границей d=15.
"""

import json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

DATA_PATH   = "data/candles/SBER/1d.json"
ACF_LAGS    = 40
N_ITER      = 3
CHUNK       = 600          # точек за один SVD-батч

# из предыдущего прогона (d=15..70) — для контекста на графике
PREV_D  = [15, 22, 30, 42, 55, 70]
PREV_A1 = [-0.1369, -0.1765, -0.2068, -0.2061, -0.2209, -0.2398]

COARSE_D    = [70, 85, 100, 120, 140, 165, 190]
BS_D_MIN    = 70           # нижняя граница бинарного поиска
BS_TOL      = 4
BS_MAX_ITER = 7

# ── данные ───────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.load(f)
close = np.array([c["close"] for c in raw], dtype=np.float64)
print(f"SBER 1d: {len(close)} баров")

def logtrend_causal(c):
    n = len(c); lc = np.log(np.maximum(c, 1e-10))
    t = np.arange(n, dtype=np.float64); cn = np.arange(1, n+1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    tr = np.exp(a + b * t); tr[:2] = c[:2]
    return tr

ratio = close / np.maximum(logtrend_causal(close), 1e-10)
ci = 1.96 / np.sqrt(len(ratio))

def acf_lag1(x):
    x = x - x.mean(); c0 = np.dot(x, x)
    if c0 < 1e-30: return 0.0
    return np.dot(x[:-1], x[1:]) / c0

def acf_full(x, lags):
    x = x - x.mean(); c0 = np.dot(x, x)
    if c0 < 1e-30: return [1.0] + [0.0]*lags
    return [1.0] + [np.dot(x[:len(x)-l], x[l:]) / c0 for l in range(1, lags+1)]

def run_lp(s, d, n_iter):
    """LP n_iter итераций, чанковый SVD — пик памяти CHUNK*k*m*8 байт."""
    m = 2*d + 3
    k = max(30, d + 15)
    if m >= len(s) - 1:
        return None, None, None
    k_eff = min(k, len(s) - m - 1)
    d_eff = min(d, m - 1)
    n_pts = len(s) - m + 1
    rows  = np.arange(n_pts)[:, None] + np.arange(m)[None, :]

    cur = s.copy()
    resid_last = None
    for _ in range(n_iter):
        X    = cur[rows]                         # (n_pts, m)
        tree = KDTree(X)
        X_proj = np.empty_like(X)
        resid_chunks = []

        for start in range(0, n_pts, CHUNK):
            end   = min(start + CHUNK, n_pts)
            Xc    = X[start:end]                 # (chunk, m)
            _, inds = tree.query(Xc, k=k_eff + 1)
            Xnn   = X[inds[:, 1:]]              # (chunk, k, m)
            cents = Xnn.mean(axis=1)             # (chunk, m)
            Xnn_c = Xnn - cents[:, None, :]
            _, _, Vt = np.linalg.svd(Xnn_c, full_matrices=False)
            Vd    = Vt[:, :d_eff, :].transpose(0, 2, 1)   # (chunk, m, d)
            xc    = Xc - cents
            coef  = np.einsum("nmd,nm->nd", Vd, xc)
            proj  = np.einsum("nmd,nd->nm", Vd, coef)
            X_proj[start:end] = cents + proj
            resid_chunks.append(np.linalg.norm(xc - proj, axis=1))

        resid_last = np.concatenate(resid_chunks).mean()
        result = np.zeros(len(s)); count = np.zeros(len(s), dtype=np.int32)
        for j in range(m):
            result[j:j+n_pts] += X_proj[:, j]
            count[j:j+n_pts]  += 1
        cur = result / np.maximum(count, 1)

    noise = s - cur
    return cur, acf_lag1(noise), resid_last

# ── грубый sweep ─────────────────────────────────────────────────────────────
print(f"\n── Грубый sweep (n={N_ITER}) ──────────────────────────────────")
print(f"{'d':>4}  {'m':>4}  {'k':>4}  {'ACF_lag1':>10}  {'resid':>10}  {'t,с':>6}")

sweep_d, sweep_a1, sweep_r = [], [], []
for d in COARSE_D:
    t0 = time.time()
    att, a1, res = run_lp(ratio, d, N_ITER)
    dt = time.time() - t0
    m = 2*d+3; k = max(30, d+15)
    print(f"{d:4d}  {m:4d}  {k:4d}  {a1:+10.4f}  {res:10.4e}  {dt:6.1f}")
    sweep_d.append(d); sweep_a1.append(a1); sweep_r.append(res)

# ── бинарный поиск ────────────────────────────────────────────────────────────
# Ищем: знаковый переход ACF_lag1 или, если нет, минимум |ACF_lag1|
bracket_lo, bracket_hi = None, None
for i in range(len(sweep_d) - 1):
    if sweep_a1[i] is not None and sweep_a1[i+1] is not None:
        if sweep_a1[i] * sweep_a1[i+1] < 0:
            lo_cand = max(sweep_d[i], BS_D_MIN)
            if sweep_d[i+1] > BS_D_MIN:
                bracket_lo, bracket_hi = lo_cand, sweep_d[i+1]
                break

bsearch_path = []
d_star = None

if bracket_lo is None:
    print("\nЗнакового перехода нет — берём минимум |ACF_lag1| при d >= BS_D_MIN")
    valid = [(d, a) for d, a in zip(sweep_d, sweep_a1)
             if a is not None and d >= BS_D_MIN]
    d_star = min(valid, key=lambda x: abs(x[1]))[0]
    print(f"  Минимум |ACF_lag1| = {min(abs(a) for _, a in valid):.4f}  при d={d_star}")
else:
    sign_lo = 1 if sweep_a1[sweep_d.index(bracket_lo)] > 0 else -1
    print(f"\n── Бинарный поиск ACF_lag1=0 в [{bracket_lo}, {bracket_hi}] ──")
    print(f"{'iter':>4}  {'lo':>5}  {'hi':>5}  {'mid':>5}  {'ACF1':>10}  {'t,с':>6}")
    lo, hi = bracket_lo, bracket_hi
    for it in range(BS_MAX_ITER):
        if hi - lo <= BS_TOL:
            break
        mid = (lo + hi) // 2
        t0 = time.time()
        _, a1_mid, _ = run_lp(ratio, mid, N_ITER)
        dt = time.time() - t0
        bsearch_path.append((mid, a1_mid))
        sign_mid = 1 if a1_mid > 0 else -1
        print(f"{it+1:4d}  {lo:5d}  {hi:5d}  {mid:5d}  {a1_mid:+10.4f}  {dt:6.1f}")
        if sign_mid == sign_lo:
            lo = mid
        else:
            hi = mid
    d_star = (lo + hi) // 2
    print(f"\nd* ≈ {d_star}  (переход в [{lo}, {hi}])")

# ── график ────────────────────────────────────────────────────────────────────
lags_x = np.arange(ACF_LAGS + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"LP sweep n={N_ITER}, d=70..190 + бинарный поиск  |  SBER 1d", fontsize=13)

# 1. ACF lag-1 vs d
ax = axes[0]
ax.plot(PREV_D, PREV_A1, "o--", color="gray", alpha=0.5, lw=1, label="предыдущий sweep")
ax.plot(sweep_d, sweep_a1, "o-", color="darkorange", label=f"n={N_ITER}")
if bsearch_path:
    bx, by = zip(*bsearch_path)
    ax.scatter(bx, by, marker="x", s=90, color="red", zorder=5, label="бин. поиск")
if d_star is not None:
    ax.axvline(d_star, color="purple", ls="--", lw=1.3, label=f"d*={d_star}")
ax.axhline(0,    color="black",  lw=0.7)
ax.axhline( ci,  color="gray",   ls=":", lw=0.8, label=f"95% CI ±{ci:.3f}")
ax.axhline(-ci,  color="gray",   ls=":", lw=0.8)
ax.axvline(BS_D_MIN, color="steelblue", ls=":", lw=1, label=f"BS граница d={BS_D_MIN}")
ax.set_xlabel("d"); ax.set_ylabel("ACF(noise) lag=1")
ax.set_title("ACF lag-1 vs d"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# 2. Полный ACF noise при разных d (n=3)
ax = axes[1]
show_d_full = [d for d in [15, 30, 55, 85, d_star] if d is not None and d in sweep_d or d == d_star]
show_d_full = sorted(set(show_d_full))
pal = plt.cm.viridis(np.linspace(0.1, 0.9, len(show_d_full)))
for d_show, col in zip(show_d_full, pal):
    att_s, _, _ = run_lp(ratio, d_show, N_ITER)
    if att_s is not None:
        lbl = f"d={d_show}" + (" ← d*" if d_show == d_star else "")
        lw  = 1.8 if d_show == d_star else 1.1
        ls  = "--" if d_show == d_star else "-"
        ax.plot(lags_x, acf_full(ratio - att_s, ACF_LAGS), color=col, lw=lw, ls=ls, label=lbl)
ax.axhline(0,    color="black", lw=0.6)
ax.axhline( ci,  color="gray",  ls=":", lw=0.7)
ax.axhline(-ci,  color="gray",  ls=":", lw=0.7)
ax.set_xlabel("Лаг"); ax.set_title(f"ACF(noise)  n={N_ITER}")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# 3. Projection residual vs d
ax = axes[2]
ax.plot(sweep_d, sweep_r, "s-", color="steelblue")
if d_star is not None:
    ax.axvline(d_star, color="purple", ls="--", lw=1.3, label=f"d*={d_star}")
ax.set_xlabel("d"); ax.set_ylabel("Средняя невязка проекции")
ax.set_title("Projection residual vs d"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "research/phase5_attractor/lp_dim_bsearch.png"
plt.savefig(out, dpi=130)
print(f"\nГрафик сохранён: {out}")
plt.show()
