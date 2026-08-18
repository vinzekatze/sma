"""
Локальная LP-очистка пула соседей: тестовая диагностика.

Логика:
  origin → K ближайших соседей в P_MAX-пространстве → фиксированный пул
  SVD пула → d_local
  Итеративный LOCAL LP внутри пула (каждая точка проецируется
  на подпространство своих k' соседей из того же пула)

  Метрики по итерациям:
    - projection residual (сходимость)
    - d_eff пула @ 90% (геометрия аттрактора)
    - ACF P_FIT-срезов остатков (белый шум → 0?)

P_MAX = P_FIT × 4  (верхний уровень p-aligned каскада)
"""

import json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# ── параметры ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/candles/SBER/1d.json"
P_FIT       = 9          # нижний уровень каскада (LWR-окно)
P_MAX       = P_FIT * 4  # верхний уровень p-aligned (= 36)
K           = 50         # размер пула (соседей origin в P_MAX-пространстве)
K_PRIME     = 15         # соседей внутри пула для каждой локальной проекции
N_MAX_ITER  = 30         # максимум LP итераций
D_THRESH    = 0.90       # порог для d_local
VAR_THRESH  = [0.80, 0.90, 0.95, 0.99]
ACF_LAGS    = P_FIT - 1  # лагов ACF (максимум P_FIT-1 = 8)
SNAP_ITERS  = [1, 3, 5, 10, 20, 30]

# ── данные ────────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.load(f)
close = np.array([c["close"] for c in raw], dtype=np.float64)

def logtrend_causal(c):
    n = len(c); lc = np.log(np.maximum(c, 1e-10))
    t = np.arange(n, dtype=np.float64); cn = np.arange(1, n+1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t**2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    tr = np.exp(a + b * t); tr[:2] = c[:2]
    return tr

ratio  = close / np.maximum(logtrend_causal(close), 1e-10)
n_pts  = len(ratio) - P_MAX + 1
rows   = np.arange(n_pts)[:, None] + np.arange(P_MAX)[None, :]
X_full = ratio[rows]  # (n_pts, P_MAX)

ORIGIN = n_pts - 1
X_past = X_full[:ORIGIN]
x_orig = X_full[ORIGIN]
origin_date = raw[ORIGIN + P_MAX - 1]["begin"][:10]

print(f"SBER 1d: {len(ratio)} баров")
print(f"P_FIT={P_FIT}  P_MAX={P_MAX}  K={K}  K'={K_PRIME}  N_ITER={N_MAX_ITER}")
print(f"Origin: {origin_date}\n")

# ── шаг 1: пул K ближайших соседей ───────────────────────────────────────────
t0 = time.time()
_, inds = KDTree(X_past).query(x_orig.reshape(1, -1), k=K)
pool_inds    = inds[0]                        # (K,) — индексы в X_past
X_pool_orig  = X_past[pool_inds].copy()       # (K, P_MAX) — фиксированный оригинал

date_near = raw[pool_inds.max() + P_MAX - 1]["begin"][:10]
date_far  = raw[pool_inds.min() + P_MAX - 1]["begin"][:10]
print(f"Пул: индексы {pool_inds.min()}–{pool_inds.max()}  "
      f"(даты: {date_far}—{date_near})  [{time.time()-t0:.1f}s]")

# ── шаг 2: SVD исходного пула ─────────────────────────────────────────────────
def svd_info(X):
    Xc = X - X.mean(axis=0)
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    vf = sv**2 / (sv**2).sum()
    cv = np.cumsum(vf)
    d_at = {thr: int(np.searchsorted(cv, thr)) + 1 for thr in VAR_THRESH}
    return sv, vf, cv, d_at, Vt

sv0, vf0, cv0, d_at0, Vt0 = svd_info(X_pool_orig)
d_local = d_at0[D_THRESH]

print(f"\nSVD пула (оригинал):")
for thr in VAR_THRESH:
    print(f"  d@{int(thr*100)}% = {d_at0[thr]}")
print(f"\nd_local = {d_local} (порог {int(D_THRESH*100)}%)")

# ── шаг 3: функции ────────────────────────────────────────────────────────────
def lp_step(X, d, k_prime):
    """Один LP шаг внутри пула: каждая точка → k' соседей из пула → проекция."""
    k_eff = min(k_prime, len(X) - 1)
    d_eff = min(d, k_eff - 1)
    _, all_inds = KDTree(X).query(X, k=k_eff + 1)   # k+1: первый = self
    X_new   = np.empty_like(X)
    resid_v = np.empty(len(X))
    for i in range(len(X)):
        nn   = all_inds[i, 1:]
        Xnn  = X[nn]; cent = Xnn.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xnn - cent, full_matrices=False)
        Vd   = Vt[:d_eff]          # (d_eff, P_MAX)
        xc   = X[i] - cent
        proj = Vd.T @ (Vd @ xc)   # проекция на d-мерн. подпростр.
        X_new[i]   = cent + proj
        resid_v[i] = np.linalg.norm(xc - proj)
    return X_new, resid_v

def pfit_acf(X_cur, X_orig):
    """Среднее ACF (лаги 1..ACF_LAGS) по P_FIT-срезам остатков."""
    resid = (X_orig - X_cur)[:, :P_FIT]   # (K, P_FIT)
    vals  = np.zeros(ACF_LAGS)
    cnt   = 0
    for k in range(len(X_orig)):
        r  = resid[k] - resid[k].mean()
        c0 = np.dot(r, r)
        if c0 < 1e-30:
            continue
        for lag in range(1, ACF_LAGS + 1):
            if lag < P_FIT:
                vals[lag - 1] += np.dot(r[:P_FIT - lag], r[lag:]) / c0
        cnt += 1
    return vals / max(cnt, 1)

def project_point(x, X_ref, d, k_prime):
    """Проецирует точку x на d-мерн. подпростр. k' ближайших в X_ref."""
    k_eff = min(k_prime, len(X_ref) - 1)
    d_eff = min(d, k_eff - 1)
    _, inds = KDTree(X_ref).query(x.reshape(1, -1), k=k_eff)
    Xnn  = X_ref[inds[0]]; cent = Xnn.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xnn - cent, full_matrices=False)
    Vd = Vt[:d_eff]
    xc = x - cent
    return cent + Vd.T @ (Vd @ xc)

# ── шаг 4: итеративный local LP ───────────────────────────────────────────────
hist_resid    = []
hist_resid_max= []
hist_d_eff    = []
hist_acf1     = []
hist_acf_full = []
snap_pools    = {}

X_pool = X_pool_orig.copy()

print(f"\n{'iter':>4}  {'resid_mean':>12}  {'resid_max':>10}  "
      f"{'d@90%':>6}  {'ACF1(pfit)':>12}  {'t':>5}")

for it in range(1, N_MAX_ITER + 1):
    t0 = time.time()
    X_pool, rv = lp_step(X_pool, d_local, K_PRIME)
    _, _, cv_it, d_at_it, _ = svd_info(X_pool)
    acf_it = pfit_acf(X_pool, X_pool_orig)

    hist_resid.append(rv.mean())
    hist_resid_max.append(rv.max())
    hist_d_eff.append(d_at_it[D_THRESH])
    hist_acf1.append(acf_it[0])

    if it in SNAP_ITERS:
        hist_acf_full.append((it, acf_it.copy()))
        snap_pools[it]  = X_pool.copy()

    print(f"{it:4d}  {rv.mean():12.6f}  {rv.max():10.6f}  "
          f"{d_at_it[D_THRESH]:6d}  {acf_it[0]:+12.4f}  {time.time()-t0:.2f}s")

X_pool_final = X_pool.copy()

# проецируем origin на финальный очищенный пул
x_orig_clean = project_point(x_orig, X_pool_final, d_local, K_PRIME)
print(f"\nOrigin projected shift: "
      f"{np.linalg.norm(x_orig - x_orig_clean):.6f}")

# ── Рис. 1: анализ исходного пула ─────────────────────────────────────────────
cent0 = X_pool_orig.mean(axis=0)
pca2_orig = (X_pool_orig - cent0) @ Vt0[:2].T  # (K, 2) в пространстве пула
x_orig_pc = (x_orig - cent0) @ Vt0[:2].T

fig1, axes = plt.subplots(2, 3, figsize=(17, 10))
fig1.suptitle(
    f"Исходный пул соседей  |  SBER 1d  |  origin {origin_date}  |  "
    f"P_MAX={P_MAX}  K={K}  d_local={d_local}",
    fontsize=13, fontweight="bold"
)

# [0,0] Scree plot
ax = axes[0, 0]
nc = min(K, P_MAX)
ax.semilogy(np.arange(1, nc + 1), sv0[:nc], "o-", color="steelblue", ms=4, lw=1.5)
ax.axvline(d_local, ls="--", color="purple", lw=1.5,
           label=f"d_local={d_local} (@{int(D_THRESH*100)}%)")
for thr, col in zip(VAR_THRESH, ["#aaa", "steelblue", "darkorange", "#c00"]):
    ax.axvline(d_at0[thr], ls=":", color=col, lw=0.8,
               label=f"d@{int(thr*100)}%={d_at0[thr]}")
ax.set_xlabel("Компонента", fontsize=10)
ax.set_ylabel("Сингулярное число (log)", fontsize=10)
ax.set_title("Скри-плот (оригинальный пул)", fontsize=11)
ax.set_xlim(0.5, min(nc, 25) + 0.5)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# [0,1] Cumulative variance
ax = axes[0, 1]
ax.plot(np.arange(1, nc + 1), cv0[:nc] * 100, color="darkorange", lw=2)
for thr, col in zip(VAR_THRESH, ["#aaa", "steelblue", "darkorange", "#c00"]):
    d_t = d_at0[thr]
    ax.axhline(thr * 100, ls="--", color=col, lw=0.8)
    ax.axvline(d_t, ls=":", color=col, lw=0.8)
    ax.annotate(f"d={d_t}", xy=(d_t, thr * 100), xytext=(d_t + 0.5, thr * 100 - 3),
                fontsize=7, color=col)
ax.set_xlabel("Компонента", fontsize=10)
ax.set_ylabel("Кум. дисперсия %", fontsize=10)
ax.set_title("Кумулятивная дисперсия", fontsize=11)
ax.set_ylim(40, 103); ax.set_xlim(0.5, min(nc, 25) + 0.5)
ax.grid(True, alpha=0.3)

# [0,2] d_local по порогам (bar)
ax = axes[0, 2]
thr_labels = [f"{int(t*100)}%" for t in VAR_THRESH]
d_vals     = [d_at0[t] for t in VAR_THRESH]
bar_colors = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f"]
bars = ax.bar(thr_labels, d_vals, color=bar_colors, alpha=0.85, edgecolor="white")
for b, v in zip(bars, d_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v),
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.axhline(d_local, ls="--", color="purple", lw=1.5,
           label=f"d_local={d_local} (выбран)")
ax.set_ylabel("d", fontsize=10)
ax.set_title("d_local по порогам дисперсии", fontsize=11)
ax.set_ylim(0, max(d_vals) + 2)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# [1,0] 2D PCA scatter (цвет = время)
ax = axes[1, 0]
sc = ax.scatter(pca2_orig[:, 0], pca2_orig[:, 1],
                c=pool_inds, cmap="plasma", s=60,
                edgecolors="white", linewidths=0.4, zorder=3, label="соседи")
plt.colorbar(sc, ax=ax, label="индекс в истории")
ax.scatter(*x_orig_pc, marker="*", s=400, color="lime",
           edgecolor="black", lw=1, zorder=5, label="origin")
ax.set_xlabel("PC1", fontsize=10); ax.set_ylabel("PC2", fontsize=10)
ax.set_title("2D PCA пула (цвет = времени бара)", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# [1,1] Временно́е распределение соседей
ax = axes[1, 1]
hi_min = max(0, pool_inds.min() - 5)
hi_max = pool_inds.max() + 5
bins   = np.arange(hi_min, hi_max + 2) - 0.5
counts, bin_edges = np.histogram(pool_inds, bins=bins)
bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.bar(bin_mid, counts, width=1, color="steelblue", alpha=0.8, label="соседи")
ax.axvline(ORIGIN, ls="--", color="red", lw=1.5, label=f"origin (i={ORIGIN})")
ax.set_xlabel("Индекс бара в истории", fontsize=10)
ax.set_ylabel("Число соседей в бине", fontsize=10)
ax.set_title(f"Распределение {K} соседей во времени", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# [1,2] Heatmap пула K × P_MAX (z-norm по лагам)
ax = axes[1, 2]
Xn = (X_pool_orig - X_pool_orig.mean(axis=0)) / (X_pool_orig.std(axis=0) + 1e-10)
im = ax.imshow(Xn, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
               interpolation="nearest")
plt.colorbar(im, ax=ax, label="z-score по лагу")
ax.axvline(P_FIT - 0.5, color="yellow", lw=2, ls="--",
           label=f"P_FIT={P_FIT} граница")
ax.set_xlabel("Лаг (0 = самый свежий)", fontsize=10)
ax.set_ylabel("Сосед (ранг близости)", fontsize=10)
ax.set_title(f"Матрица пула K×P_MAX (z-norm)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out1 = "research/phase5_attractor/local_lp_pool_fig1.png"
fig1.savefig(out1, dpi=130)
print(f"\nФиг.1 сохранена: {out1}")

# ── Рис. 2: итерации LP ───────────────────────────────────────────────────────
iters_x    = np.arange(1, N_MAX_ITER + 1)
lags_x     = np.arange(1, ACF_LAGS + 1)
snap_snaps = [hs for hs in hist_acf_full if hs[0] in SNAP_ITERS]
snap_cols  = plt.cm.viridis(np.linspace(0.05, 0.95, len(snap_snaps)))
CI95       = 1.96 / np.sqrt(K * P_FIT)

fig2, axes = plt.subplots(2, 3, figsize=(17, 10))
fig2.suptitle(
    f"Итерации локального LP  |  d_local={d_local}  K'={K_PRIME}  "
    f"K={K}  P_MAX={P_MAX}→P_FIT={P_FIT}",
    fontsize=13, fontweight="bold"
)

# [0,0] Projection residual
ax = axes[0, 0]
ax.semilogy(iters_x, hist_resid,     "o-", color="steelblue",  lw=2, ms=4, label="mean")
ax.semilogy(iters_x, hist_resid_max, "s--", color="steelblue", lw=1, ms=3,
            alpha=0.5, label="max")
ax.set_xlabel("Итерация", fontsize=10)
ax.set_ylabel("Projection residual (log)", fontsize=10)
ax.set_title("Сходимость: projection residual", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# [0,1] d_eff@90% очищённого пула
ax = axes[0, 1]
ax.plot(iters_x, hist_d_eff, "s-", color="darkorange", lw=2, ms=5)
ax.axhline(d_local, ls="--", color="purple", lw=1.5, label=f"d_local={d_local}")
ax.set_xlabel("Итерация", fontsize=10)
ax.set_ylabel(f"d@{int(D_THRESH*100)}% очищённого пула", fontsize=10)
ax.set_title("Эффективная размерность пула", fontsize=11)
ax.set_ylim(0, max(hist_d_eff) + 2)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# [0,2] ACF lag-1 P_FIT-срезов
ax = axes[0, 2]
ax.plot(iters_x, hist_acf1, "^-", color="green", lw=2, ms=5)
ax.axhline(0,     color="black", lw=0.8)
ax.axhline( CI95, ls=":", color="gray", lw=1.2, label=f"95% CI ±{CI95:.3f}")
ax.axhline(-CI95, ls=":", color="gray", lw=1.2)
ax.set_xlabel("Итерация", fontsize=10)
ax.set_ylabel("ACF lag-1 (P_FIT-срезы)", fontsize=10)
ax.set_title("Качество очистки: ACF lag-1 остатков", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# [1,0] Полный ACF на снапшотах
ax = axes[1, 0]
for (it_s, acf_s), col in zip(snap_snaps, snap_cols):
    ax.plot(lags_x, acf_s, "o-", color=col, lw=1.8, ms=4, label=f"iter={it_s}")
ax.axhline(0,     color="black", lw=0.8)
ax.axhline( CI95, ls=":", color="gray", lw=1.2, label=f"±{CI95:.3f}")
ax.axhline(-CI95, ls=":", color="gray", lw=1.2)
ax.set_xlabel("Лаг", fontsize=10)
ax.set_ylabel("ACF", fontsize=10)
ax.set_title("Полный ACF P_FIT-срезов (снапшоты итераций)", fontsize=11)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# [1,1] 2D PCA: до и после очистки (стрелки смещений)
ax = axes[1, 1]
pool_final_for_plot = snap_pools.get(N_MAX_ITER, X_pool_final)
pca2_final = (pool_final_for_plot - cent0) @ Vt0[:2].T  # проецируем в то же пространство
x_orig_clean_pc = (x_orig_clean - cent0) @ Vt0[:2].T

# стрелки смещений
for i in range(K):
    ax.annotate("",
                xy=(pca2_final[i, 0], pca2_final[i, 1]),
                xytext=(pca2_orig[i, 0], pca2_orig[i, 1]),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.6, alpha=0.5))

ax.scatter(pca2_orig[:, 0],  pca2_orig[:, 1],  c="steelblue",  s=50, alpha=0.6,
           edgecolors="none", zorder=3, label="до LP")
ax.scatter(pca2_final[:, 0], pca2_final[:, 1], c="darkorange", s=55, alpha=0.9,
           edgecolors="white", lw=0.3, zorder=4, label=f"после {N_MAX_ITER}и")
ax.scatter(*x_orig_pc,       marker="*", s=350, color="lime",
           edgecolor="black", lw=1, zorder=6, label="origin (raw)")
ax.scatter(*x_orig_clean_pc, marker="D", s=100, color="yellow",
           edgecolor="black", lw=1, zorder=6, label="origin (projected)")

ax.set_xlabel("PC1", fontsize=10); ax.set_ylabel("PC2", fontsize=10)
ax.set_title("2D PCA: пул до и после LP (стрелки смещений)", fontsize=11)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# [1,2] P_FIT-срезы: orig | clean | residual (heatmap 3-блочный)
ax = axes[1, 2]
block_orig  = X_pool_orig[:, :P_FIT]
block_final = pool_final_for_plot[:, :P_FIT]
block_resid = block_orig - block_final

# нормируем по шкале оригинала
scale = np.percentile(np.abs(block_orig), 97) + 1e-10
separator = np.full((K, 1), np.nan)
combined   = np.hstack([block_orig, separator, block_final, separator, block_resid])
im2 = ax.imshow(combined / scale, aspect="auto", cmap="RdBu_r",
                vmin=-1.5, vmax=1.5, interpolation="nearest")
plt.colorbar(im2, ax=ax, label="нормированное значение")

w = P_FIT
ax.axvline(w + 0.5,     color="white", lw=2)
ax.axvline(2 * w + 1.5, color="white", lw=2)
ax.text(w / 2,         K + 1.5, "оригинал",  ha="center", fontsize=9, va="top",
        transform=ax.transData)
ax.text(w + 1 + w / 2, K + 1.5, f"LP×{N_MAX_ITER}",    ha="center", fontsize=9, va="top",
        transform=ax.transData)
ax.text(2*w+2 + w/2,   K + 1.5, "остаток",   ha="center", fontsize=9, va="top",
        transform=ax.transData)

ax.set_ylabel("Сосед (ранг близости)", fontsize=10)
ax.set_title(f"P_FIT-срезы (первые {P_FIT} лагов): orig | LP×{N_MAX_ITER} | residual",
             fontsize=11)

plt.tight_layout()
out2 = "research/phase5_attractor/local_lp_pool_fig2.png"
fig2.savefig(out2, dpi=130)
print(f"Фиг.2 сохранена: {out2}")

plt.show()
print("\nГотово.")
