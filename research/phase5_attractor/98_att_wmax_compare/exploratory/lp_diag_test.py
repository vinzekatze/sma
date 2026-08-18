"""
Диагностика LP-фильтра на SBER 1d: сравнение d=6, d=12, d=20.
Параметры по теории: m = 2d+3, k = 10d.
Векторизованный SVD (батчевый numpy) для приемлемого времени работы.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

DATA_PATH  = "data/candles/SBER/1d.json"
N_ITER_MAX = 60
ACF_LAGS   = 60
ACF_SNAPS  = [3, 10, 25, 50]

CONFIGS = [
    (6,  15,  60, "d=6  m=15  k=60"),
    (12, 27, 120, "d=12 m=27  k=120"),
    (20, 43, 200, "d=20 m=43  k=200"),
]

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

def acf(x, lags):
    x = x - x.mean(); c0 = np.dot(x, x)
    if c0 < 1e-30:
        return [1.0] + [0.0] * lags
    return [1.0] + [np.dot(x[:len(x)-lag], x[lag:]) / c0 for lag in range(1, lags+1)]

def lp_one_iter_vec(s, m, d_eff, k_eff):
    """Векторизованный LP: батчевый SVD вместо Python-цикла."""
    N = len(s); n_pts = N - m + 1
    rows = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
    X = s[rows]                                      # (n_pts, m)

    _, inds = KDTree(X).query(X, k=k_eff + 1)
    X_nn = X[inds[:, 1:]]                            # (n_pts, k, m)
    centroids = X_nn.mean(axis=1)                    # (n_pts, m)
    X_centered = X_nn - centroids[:, None, :]        # (n_pts, k, m)

    # батчевый SVD: Vt (n_pts, min(k,m), m)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V_d = Vt[:, :d_eff, :].transpose(0, 2, 1)       # (n_pts, m, d_eff)

    xc   = X - centroids                             # (n_pts, m)
    coef = np.einsum("nmd,nm->nd", V_d, xc)          # (n_pts, d_eff)
    proj = np.einsum("nmd,nd->nm", V_d, coef)        # (n_pts, m)
    X_proj = centroids + proj                        # (n_pts, m)

    resid_mean = np.linalg.norm(xc - proj, axis=1).mean()

    # реконструкция: скользящее суммирование по смещениям
    result = np.zeros(N); count = np.zeros(N, dtype=np.int32)
    for j in range(m):
        result[j:j + n_pts] += X_proj[:, j]
        count[j:j + n_pts]  += 1

    return result / np.maximum(count, 1), resid_mean

# ── прогон ───────────────────────────────────────────────────────────────────
results = {}
for (d_proj, m, k, label) in CONFIGS:
    k_eff = min(k, len(ratio) - m)
    d_eff = min(d_proj, m - 1)
    s = ratio.copy()
    conv_curve = []; resid_curve = []; acf_snaps = {}
    print(f"\nLP {label}")
    for n in range(1, N_ITER_MAX + 1):
        s_prev = s.copy()
        s, res = lp_one_iter_vec(s, m, d_eff, k_eff)
        rel = np.linalg.norm(s - s_prev) / (np.linalg.norm(s_prev) + 1e-12)
        conv_curve.append(rel); resid_curve.append(res)
        if n in ACF_SNAPS:
            acf_snaps[n] = s.copy()
        if n % 10 == 0 or n <= 3:
            print(f"  iter {n:3d}  conv={rel:.2e}  resid={res:.4e}")
    results[label] = dict(conv=conv_curve, resid=resid_curve,
                          att_final=s.copy(), acf_snaps=acf_snaps,
                          d=d_proj, m=m, k=k)

# ── график ───────────────────────────────────────────────────────────────────
COLORS = ["steelblue", "darkorange", "green"]
ci     = 1.96 / np.sqrt(len(ratio))
lags_x = np.arange(ACF_LAGS + 1)
iters  = np.arange(1, N_ITER_MAX + 1)
tail   = 300

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"LP диагностика SBER 1d  |  d=6 vs d=12 vs d=20  |  n до {N_ITER_MAX}", fontsize=13)

ax_conv  = axes[0, 0]
ax_resid = axes[0, 1]
ax_rat   = axes[0, 2]
ax_acf3  = axes[1, 0]   # ACF при n=3 для всех d
ax_acf25 = axes[1, 1]   # ACF при n=25
ax_acf50 = axes[1, 2]   # ACF при n=50

for (label, res), col in zip(results.items(), COLORS):
    ax_conv.semilogy(iters, res["conv"],  color=col, label=label)
    ax_resid.semilogy(iters, res["resid"], color=col, label=label)

for ax in (ax_conv, ax_resid):
    ax.axvline(3,  color="red", ls="--", lw=0.9, label="n=3")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
ax_conv.set_xlabel("Итерация n"); ax_conv.set_ylabel("||s_n − s_{n-1}|| / ||s_{n-1}||")
ax_conv.set_title("Кривая сходимости")
ax_resid.set_xlabel("Итерация n"); ax_resid.set_ylabel("Средняя невязка проекции")
ax_resid.set_title("Projection residual")

ax_rat.plot(ratio[-tail:], color="gray", lw=0.7, alpha=0.5, label="ratio")
for (label, res), col in zip(results.items(), COLORS):
    ax_rat.plot(res["att_final"][-tail:], color=col, lw=1.1, label=label[:4])
ax_rat.set_title(f"ratio vs att при n={N_ITER_MAX} (последние {tail} баров)")
ax_rat.legend(fontsize=7); ax_rat.grid(True, alpha=0.3)

for ax, n_snap in zip((ax_acf3, ax_acf25, ax_acf50), (3, 25, 50)):
    noise_dratio = acf(np.diff(ratio), ACF_LAGS - 1)
    ax.plot(range(ACF_LAGS), noise_dratio, color="gray", lw=0.8, ls=":", label="dratio raw")
    for (label, res), col in zip(results.items(), COLORS):
        snap = res["acf_snaps"].get(n_snap)
        if snap is not None:
            ax.plot(lags_x, acf(ratio - snap, ACF_LAGS), color=col, lw=1.2, label=label[:4])
    ax.axhline(0,    color="black", lw=0.6)
    ax.axhline( ci,  color="gray",  ls=":", lw=0.8)
    ax.axhline(-ci,  color="gray",  ls=":", lw=0.8)
    ax.set_title(f"ACF(noise)  n={n_snap}"); ax.set_xlabel("Лаг")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "research/phase5_attractor/lp_diag_d6_d12_d20.png"
plt.savefig(out, dpi=130)
print(f"\nГрафик сохранён: {out}")
plt.show()
