"""
Диагностика: стабильность top-k соседей при итерациях LP.

Для текущего origin (последняя точка) запускаем LP итеративно и на каждой
итерации смотрим: изменился ли top-k список соседей?

Метрики:
  - Jaccard(n, n-1): сколько соседей совпадает с предыдущей итерацией
  - Jaccard(n, 0):   сколько совпадает с исходным raw-списком
  - mean_rank_shift: средний сдвиг ранга среди top-k raw-соседей

Тестируем d = [3, 6, 11] при p=300, k_neighbors=50.
"""

import json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

DATA_PATH  = "data/candles/SBER/1d.json"
P          = 300     # delay embedding dimension
K_NEIGH    = 50      # соседей для каскада
N_MAX      = 40      # максимум LP итераций
D_VALUES   = [3, 6, 11]
JACCARD_THR = 0.95   # порог стабильности

# ── данные ───────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.load(f)
close = np.array([c["close"] for c in raw], dtype=np.float64)
N = len(close)

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

# delay embedding helpers
n_pts = N - P + 1
rows  = np.arange(n_pts)[:, None] + np.arange(P)[None, :]

# origin — последняя точка; ищем соседей только в прошлом
ORIGIN_IDX = n_pts - 1

def get_neighbors(sig, origin_idx, k):
    """Top-k соседей origin в delay embedding сигнала sig (только прошлое)."""
    X      = sig[rows]
    x_q    = X[origin_idx]
    past_X = X[:origin_idx]
    _, inds = KDTree(past_X).query(x_q.reshape(1, -1), k=k)
    return set(inds[0].tolist()), inds[0]

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)

def mean_rank_shift(inds_ref, inds_cur):
    """Средний сдвиг ранга raw top-k в новом рейтинге."""
    rank_cur = {idx: r for r, idx in enumerate(inds_cur)}
    shifts = [abs(rank_cur.get(idx, K_NEIGH) - r) for r, idx in enumerate(inds_ref)]
    return np.mean(shifts)

def lp_one_iter_vec(s, m, d_eff, k_eff):
    """Один LP шаг (векторизованный)."""
    n_s   = len(s); n_p = n_s - m + 1
    rws   = np.arange(n_p)[:, None] + np.arange(m)[None, :]
    X     = s[rws]
    _, inds = KDTree(X).query(X, k=k_eff + 1)
    X_nn  = X[inds[:, 1:]]
    cents = X_nn.mean(axis=1)
    Xc    = X_nn - cents[:, None, :]
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vd    = Vt[:, :d_eff, :].transpose(0, 2, 1)
    xc    = X - cents
    proj  = np.einsum("nmd,nd->nm", Vd, np.einsum("nmd,nm->nd", Vd, xc))
    Xp    = cents + proj
    res   = np.zeros(n_s); cnt = np.zeros(n_s, dtype=np.int32)
    for j in range(m):
        res[j:j+n_p] += Xp[:, j]; cnt[j:j+n_p] += 1
    return res / np.maximum(cnt, 1)

# ── прогон для каждого d ─────────────────────────────────────────────────────
origin_date = raw[ORIGIN_IDX + P - 1]["begin"][:10]
print(f"Origin: {origin_date}  |  P={P}  K={K_NEIGH}  N_MAX={N_MAX}\n")

# raw baseline
neigh0_set, neigh0_inds = get_neighbors(ratio, ORIGIN_IDX, K_NEIGH)
print(f"Raw top-{K_NEIGH} соседей получены.\n")

results = {}

for d in D_VALUES:
    m_lp   = 2*d + 3
    k_lp   = max(30, d + 15)
    k_eff  = min(k_lp, N - m_lp - 1)
    d_eff  = min(d, m_lp - 1)

    print(f"── d={d}  m={m_lp}  k_lp={k_eff} ─────────────────")
    print(f"  {'n':>3}  {'J(n,n-1)':>10}  {'J(n,0)':>10}  {'rank_shift':>12}  {'t':>5}")

    sig      = ratio.copy()
    prev_set = neigh0_set
    prev_ind = neigh0_inds

    jac_consec = []   # J(n, n-1)
    jac_raw    = []   # J(n, 0)
    rank_sh    = []
    n_star     = None

    for n in range(1, N_MAX + 1):
        t0  = time.time()
        sig = lp_one_iter_vec(sig, m_lp, d_eff, k_eff)
        cur_set, cur_ind = get_neighbors(sig, ORIGIN_IDX, K_NEIGH)
        jc  = jaccard(cur_set, prev_set)
        jr  = jaccard(cur_set, neigh0_set)
        rs  = mean_rank_shift(neigh0_inds, cur_ind)
        jac_consec.append(jc); jac_raw.append(jr); rank_sh.append(rs)
        if n_star is None and jc >= JACCARD_THR:
            n_star = n
        print(f"  {n:3d}  {jc:10.3f}  {jr:10.3f}  {rs:12.1f}  {time.time()-t0:5.1f}s")
        prev_set, prev_ind = cur_set, cur_ind

    print(f"  → n* (J≥{JACCARD_THR}) = {n_star}\n")
    results[d] = dict(jac_consec=jac_consec, jac_raw=jac_raw,
                      rank_sh=rank_sh, n_star=n_star)

# ── график ───────────────────────────────────────────────────────────────────
COLORS = {3: "steelblue", 6: "darkorange", 11: "green"}
iters  = np.arange(1, N_MAX + 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Стабильность top-{K_NEIGH} соседей при LP итерациях  |  "
             f"SBER 1d  |  origin {origin_date}  |  p={P}", fontsize=12)

ax = axes[0]
for d, res in results.items():
    ax.plot(iters, res["jac_consec"], color=COLORS[d], lw=1.5, label=f"d={d}")
    if res["n_star"]:
        ax.axvline(res["n_star"], color=COLORS[d], ls=":", lw=1)
ax.axhline(JACCARD_THR, color="gray", ls="--", lw=1, label=f"порог {JACCARD_THR}")
ax.set_xlabel("Итерация n"); ax.set_ylabel("Jaccard(n, n-1)")
ax.set_title("Стабильность: J с предыдущей итерацией")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

ax = axes[1]
for d, res in results.items():
    ax.plot(iters, res["jac_raw"], color=COLORS[d], lw=1.5, label=f"d={d}")
ax.axhline(0.5, color="gray", ls="--", lw=1, label="50% совпадение")
ax.set_xlabel("Итерация n"); ax.set_ylabel("Jaccard(n, raw)")
ax.set_title("Дрейф от raw соседей")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

ax = axes[2]
for d, res in results.items():
    ax.plot(iters, res["rank_sh"], color=COLORS[d], lw=1.5, label=f"d={d}")
ax.set_xlabel("Итерация n"); ax.set_ylabel("Средний сдвиг ранга")
ax.set_title("Ранговый сдвиг raw top-k")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "research/phase5_attractor/local_lp_stability.png"
plt.savefig(out, dpi=130)
print(f"График сохранён: {out}")
plt.show()
