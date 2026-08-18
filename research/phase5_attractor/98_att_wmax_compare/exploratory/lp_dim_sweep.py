"""
Sweep d при n=1: ищем «истинную» размерность аттрактора.

При n=1 LP делает один проход проекции — быстро, можно тестировать
большие d. Смотрим при каком d:
  - projection residual перестаёт падать (изгиб «скри-плота»)
  - ACF(noise) lag-1 приближается к нулю

Параметры: m = 2d+3 (Такенс), k = max(30, d+15) (минимум для SVD).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

DATA_PATH = "data/candles/SBER/1d.json"
ACF_LAGS  = 40

D_VALUES = [2, 3, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30, 36, 42, 50, 60]

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

def acf_lag1(x):
    x = x - x.mean(); c0 = np.dot(x, x)
    if c0 < 1e-30: return 0.0
    return np.dot(x[:-1], x[1:]) / c0

def acf_full(x, lags):
    x = x - x.mean(); c0 = np.dot(x, x)
    if c0 < 1e-30: return [1.0] + [0.0]*lags
    return [1.0] + [np.dot(x[:len(x)-l], x[l:]) / c0 for l in range(1, lags+1)]

def lp_n1(s, m, d_eff, k_eff):
    """Один проход LP. Возвращает (att, proj_residual_mean)."""
    N = len(s); n_pts = N - m + 1
    rows = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
    X = s[rows]
    _, inds = KDTree(X).query(X, k=k_eff + 1)
    X_nn = X[inds[:, 1:]]                          # (n_pts, k, m)
    centroids = X_nn.mean(axis=1)                  # (n_pts, m)
    X_centered = X_nn - centroids[:, None, :]
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V_d = Vt[:, :d_eff, :].transpose(0, 2, 1)     # (n_pts, m, d_eff)
    xc   = X - centroids
    coef = np.einsum("nmd,nm->nd", V_d, xc)
    proj = np.einsum("nmd,nd->nm", V_d, coef)
    X_proj = centroids + proj
    resid_mean = np.linalg.norm(xc - proj, axis=1).mean()
    result = np.zeros(N); count = np.zeros(N, dtype=np.int32)
    for j in range(m):
        result[j:j+n_pts] += X_proj[:, j]
        count[j:j+n_pts]  += 1
    return result / np.maximum(count, 1), resid_mean

# ── sweep ─────────────────────────────────────────────────────────────────────
records = []
ci = 1.96 / np.sqrt(len(ratio))

print(f"\n{'d':>4}  {'m':>4}  {'k':>4}  {'resid':>10}  {'ACF_lag1':>10}  {'mem_MB':>8}")
for d in D_VALUES:
    m = 2*d + 3
    k = max(30, d + 15)
    if m >= len(ratio) - 1:
        print(f"{d:4d}  m={m} — не хватает данных, пропускаем"); continue
    k_eff = min(k, len(ratio) - m - 1)
    d_eff = min(d, m - 1)
    n_pts = len(ratio) - m + 1
    mem_mb = n_pts * k_eff * m * 8 / 1e6

    att, resid = lp_n1(ratio, m, d_eff, k_eff)
    noise = ratio - att
    a1 = acf_lag1(noise)
    print(f"{d:4d}  {m:4d}  {k_eff:4d}  {resid:10.4e}  {a1:+10.4f}  {mem_mb:8.0f}")
    records.append(dict(d=d, m=m, k=k_eff, resid=resid, acf1=a1,
                        att=att, noise=noise))

# ── график ────────────────────────────────────────────────────────────────────
ds     = [r["d"]    for r in records]
resids = [r["resid"] for r in records]
acf1s  = [r["acf1"]  for r in records]

# выбираем несколько d для показа полного ACF
show_d = [d for d in [3, 10, 20, 30, 50, 60] if d in ds]
show_recs = [r for r in records if r["d"] in show_d]
acf_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(show_recs)))
lags_x = np.arange(ACF_LAGS + 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("LP n=1 sweep по d  |  SBER 1d  |  m=2d+3, k=max(30,d+15)", fontsize=13)

# 1. Projection residual vs d  (скри-плот)
ax = axes[0]
ax.plot(ds, resids, "o-", color="steelblue")
ax.set_xlabel("d (размерность проекции)"); ax.set_ylabel("Средняя невязка проекции")
ax.set_title("Projection residual (n=1)")
ax.grid(True, alpha=0.3)

# 2. ACF lag-1 vs d
ax = axes[1]
ax.plot(ds, acf1s, "o-", color="darkorange")
ax.axhline(0,   color="black", lw=0.7)
ax.axhline( ci, color="gray",  ls="--", lw=0.8, label=f"+95% CI ({ci:.3f})")
ax.axhline(-ci, color="gray",  ls="--", lw=0.8, label=f"−95% CI")
ax.set_xlabel("d"); ax.set_ylabel("ACF(noise) lag=1")
ax.set_title("ACF lag-1 шума vs d (n=1)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 3. Полный ACF noise для выбранных d
ax = axes[2]
noise_raw = acf_full(np.diff(ratio), ACF_LAGS - 1)
ax.plot(range(ACF_LAGS), noise_raw, color="gray", lw=0.8, ls=":", label="dratio raw")
for rec, col in zip(show_recs, acf_colors):
    ax.plot(lags_x, acf_full(rec["noise"], ACF_LAGS),
            color=col, lw=1.2, label=f"d={rec['d']}")
ax.axhline(0,    color="black", lw=0.6)
ax.axhline( ci,  color="gray",  ls=":", lw=0.7)
ax.axhline(-ci,  color="gray",  ls=":", lw=0.7)
ax.set_xlabel("Лаг"); ax.set_title("ACF(noise) при разных d (n=1)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "research/phase5_attractor/lp_dim_sweep.png"
plt.savefig(out, dpi=130)
print(f"\nГрафик сохранён: {out}")
plt.show()
