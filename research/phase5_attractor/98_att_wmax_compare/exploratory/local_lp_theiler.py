"""
Theiler window + локальная LP-очистка пула: тестовая диагностика.

Theiler window W: при поиске K соседей origin исключаем точки
с |t_i - t_origin| < W.  При W=0 — стандартный поиск (temporal bias).
При W=P_MAX — соседи гарантированно не «сдвиговые копии» origin.

Сравниваем несколько W, затем для каждого W — локальный LP внутри пула.

Метрики:
  - временное распределение K соседей
  - d_local из SVD пула
  - сходимость LP (projection residual)
  - ACF P_FIT-срезов остатков (достаточность очистки)
"""

import json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# ── параметры ─────────────────────────────────────────────────────────────────
DATA_PATH    = "data/candles/SBER/1d.json"
P_FIT        = 9
P_MAX        = P_FIT * 4          # 36 — верхний уровень p-aligned каскада
K            = 50                  # размер пула
K_PRIME      = 15                  # соседей внутри пула для LP-проекции
N_MAX_ITER   = 30
D_THRESH     = 0.90
VAR_THRESH   = [0.80, 0.90, 0.95, 0.99]
ACF_LAGS     = P_FIT - 1           # 8 лагов
SNAP_ITERS   = [1, 3, 5, 10, 20, 30]

# Theiler window: набор значений для сравнения
THEILER_WS   = [0, P_FIT, P_MAX, P_MAX * 2]   # [0, 9, 36, 72]
THEILER_LBLS = ["W=0\n(нет)", f"W=P_FIT\n({P_FIT})", f"W=P_MAX\n({P_MAX})",
                f"W=2·P_MAX\n({P_MAX*2})"]
THEILER_COLS = ["#777777", "#4878d0", "#ee854a", "#d65f5f"]

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
X_full = ratio[rows]

ORIGIN      = n_pts - 1
X_past      = X_full[:ORIGIN]
x_orig      = X_full[ORIGIN]
all_past_t  = np.arange(len(X_past))   # временны́е индексы X_past
origin_date = raw[ORIGIN + P_MAX - 1]["begin"][:10]

print(f"SBER 1d: {len(ratio)} баров")
print(f"P_FIT={P_FIT}  P_MAX={P_MAX}  K={K}  K'={K_PRIME}  N_ITER={N_MAX_ITER}")
print(f"Origin: {origin_date}\n")

# ── функции ───────────────────────────────────────────────────────────────────
def find_pool(W):
    """K соседей origin с исключением |t_i - ORIGIN| < W."""
    mask = np.abs(all_past_t - ORIGIN) >= W
    if mask.sum() < K:
        raise ValueError(f"Theiler W={W}: меньше K={K} точек после фильтрации "
                         f"(осталось {mask.sum()})")
    X_cand  = X_past[mask]
    orig_idx = np.where(mask)[0]
    _, inds  = KDTree(X_cand).query(x_orig.reshape(1, -1), k=K)
    pool_inds = orig_idx[inds[0]]
    return pool_inds, X_past[pool_inds].copy()

def svd_info(X):
    Xc = X - X.mean(axis=0)
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    vf = sv**2 / (sv**2).sum()
    cv = np.cumsum(vf)
    d_at = {thr: int(np.searchsorted(cv, thr)) + 1 for thr in VAR_THRESH}
    return sv, vf, cv, d_at, Vt

def lp_step(X, d, k_prime):
    k_eff = min(k_prime, len(X) - 1)
    d_eff = min(d, k_eff - 1)
    _, all_inds = KDTree(X).query(X, k=k_eff + 1)
    X_new = np.empty_like(X); resid_v = np.empty(len(X))
    for i in range(len(X)):
        nn   = all_inds[i, 1:]
        Xnn  = X[nn]; cent = Xnn.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xnn - cent, full_matrices=False)
        Vd   = Vt[:d_eff]
        xc   = X[i] - cent
        proj = Vd.T @ (Vd @ xc)
        X_new[i]   = cent + proj
        resid_v[i] = np.linalg.norm(xc - proj)
    return X_new, resid_v

def pfit_acf(X_cur, X_orig):
    resid = (X_orig - X_cur)[:, :P_FIT]
    vals  = np.zeros(ACF_LAGS); cnt = 0
    for k in range(len(X_orig)):
        r  = resid[k] - resid[k].mean()
        c0 = np.dot(r, r)
        if c0 < 1e-30: continue
        for lag in range(1, ACF_LAGS + 1):
            if lag < P_FIT:
                vals[lag - 1] += np.dot(r[:P_FIT - lag], r[lag:]) / c0
        cnt += 1
    return vals / max(cnt, 1)

# ── прогон для каждого W ──────────────────────────────────────────────────────
results = {}   # W → dict с метриками

for W, lbl, col in zip(THEILER_WS, THEILER_LBLS, THEILER_COLS):
    print(f"══ Theiler W={W} ({'нет' if W==0 else f'исключ. {W} баров'}) ══")
    try:
        pool_inds, X_pool_orig = find_pool(W)
    except ValueError as e:
        print(f"  !! {e}\n"); continue

    sv0, vf0, cv0, d_at0, Vt0 = svd_info(X_pool_orig)
    d_local = d_at0[D_THRESH]

    date_near = raw[pool_inds.max() + P_MAX - 1]["begin"][:10]
    date_far  = raw[pool_inds.min() + P_MAX - 1]["begin"][:10]
    print(f"  Пул: {pool_inds.min()}–{pool_inds.max()}  ({date_far}—{date_near})")
    print(f"  d_local={d_local}  ", end="")
    for thr in VAR_THRESH:
        print(f"d@{int(thr*100)}%={d_at0[thr]}", end="  ")
    print()

    # LP итерации
    hist = dict(resid=[], resid_max=[], d_eff=[], acf1=[], acf_full=[])
    snap_pools = {}
    X_pool = X_pool_orig.copy()

    print(f"  {'iter':>4}  {'resid':>10}  {'d@90%':>6}  {'ACF1':>8}")
    for it in range(1, N_MAX_ITER + 1):
        X_pool, rv = lp_step(X_pool, d_local, K_PRIME)
        _, _, _, d_at_it, _ = svd_info(X_pool)
        acf_it = pfit_acf(X_pool, X_pool_orig)
        hist["resid"].append(rv.mean())
        hist["resid_max"].append(rv.max())
        hist["d_eff"].append(d_at_it[D_THRESH])
        hist["acf1"].append(acf_it[0])
        if it in SNAP_ITERS:
            hist["acf_full"].append((it, acf_it.copy()))
            snap_pools[it] = X_pool.copy()
        if it in [1, 3, 5, 10, 20, 30]:
            print(f"  {it:4d}  {rv.mean():10.6f}  {d_at_it[D_THRESH]:6d}  {acf_it[0]:+8.4f}")

    results[W] = dict(
        lbl=lbl, col=col,
        pool_inds=pool_inds, X_pool_orig=X_pool_orig, X_pool_final=X_pool,
        sv0=sv0, vf0=vf0, cv0=cv0, d_at0=d_at0, Vt0=Vt0,
        d_local=d_local, hist=hist, snap_pools=snap_pools
    )
    print()

# ── Рис. 1: сравнение пулов по W ─────────────────────────────────────────────
CI95 = 1.96 / np.sqrt(K * P_FIT)
iters_x = np.arange(1, N_MAX_ITER + 1)
lags_x  = np.arange(1, ACF_LAGS + 1)

n_w = len(results)
fig1, axes = plt.subplots(3, n_w, figsize=(5 * n_w, 13))
fig1.suptitle(
    f"Влияние Theiler window на пул соседей  |  SBER 1d  |  origin {origin_date}  |  "
    f"P_MAX={P_MAX}  K={K}",
    fontsize=13, fontweight="bold"
)

for col_i, (W, res) in enumerate(results.items()):
    pool_inds     = res["pool_inds"]
    sv0, cv0, Vt0 = res["sv0"], res["cv0"], res["Vt0"]
    d_local       = res["d_local"]
    col           = res["col"]
    lbl           = res["lbl"].replace("\n", " ")
    X_pool_orig   = res["X_pool_orig"]
    cent0         = X_pool_orig.mean(axis=0)
    pca2          = (X_pool_orig - cent0) @ Vt0[:2].T
    x_pc          = (x_orig      - cent0) @ Vt0[:2].T

    # строка 0: временно́е распределение
    ax = axes[0, col_i]
    lo = max(0, pool_inds.min() - 5); hi = pool_inds.max() + 5
    bins = np.arange(lo, hi + 2) - 0.5
    cnt_h, bin_e = np.histogram(pool_inds, bins=bins)
    ax.bar((bin_e[:-1] + bin_e[1:]) / 2, cnt_h, width=1, color=col, alpha=0.8)
    ax.axvline(ORIGIN, ls="--", color="red", lw=1.5, label=f"origin ({ORIGIN})")
    ax.set_title(f"Theiler {lbl}\nd_local={d_local}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Индекс бара"); ax.set_ylabel("Соседей в бине")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # строка 1: scree + cumvar
    ax = axes[1, col_i]
    nc  = min(K, P_MAX)
    ax2 = ax.twinx()
    ax.semilogy(np.arange(1, nc + 1), sv0[:nc], "o-", color=col,
                ms=3, lw=1.5, alpha=0.8, label="SV (лев.)")
    ax2.plot(np.arange(1, nc + 1), cv0[:nc] * 100, "s--", color="gray",
             ms=2, lw=1, alpha=0.6, label="cumvar % (прав.)")
    for thr, tc in zip(VAR_THRESH, ["#aaa", col, "darkorange", "#c00"]):
        d_t = res["d_at0"][thr]
        ax2.axhline(thr * 100, ls=":", color=tc, lw=0.7)
        ax2.axvline(d_t, ls=":", color=tc, lw=0.7)
        ax.annotate(f"d@{int(thr*100)}%={d_t}",
                    xy=(d_t, sv0[d_t - 1]),
                    xytext=(d_t + 0.3, sv0[d_t - 1] * 1.5),
                    fontsize=6, color=tc)
    ax.set_xlabel("Компонента"); ax.set_ylabel("SV (log)")
    ax2.set_ylabel("Cumvar %"); ax2.set_ylim(40, 103)
    ax.set_xlim(0.5, min(nc, 30) + 0.5)
    ax.set_title(f"Скри + Cumvar", fontsize=10)
    ax.grid(True, alpha=0.3)

    # строка 2: 2D PCA scatter (цвет = время)
    ax = axes[2, col_i]
    sc = ax.scatter(pca2[:, 0], pca2[:, 1],
                    c=pool_inds, cmap="plasma", s=55,
                    edgecolors="white", lw=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, label="индекс бара")
    ax.scatter(*x_pc, marker="*", s=350, color="lime",
               edgecolor="black", lw=1, zorder=5, label="origin")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("2D PCA (цвет = время)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = "research/phase5_attractor/local_lp_theiler_fig1.png"
fig1.savefig(out1, dpi=130)
print(f"Фиг.1 сохранена: {out1}")

# ── Рис. 2: LP итерации по W ──────────────────────────────────────────────────
fig2, axes = plt.subplots(2, 3, figsize=(17, 11))
fig2.suptitle(
    f"LP итерации по Theiler W  |  K'={K_PRIME}  N={N_MAX_ITER}  "
    f"P_MAX={P_MAX}→P_FIT={P_FIT}",
    fontsize=13, fontweight="bold"
)

# [0,0] Projection residual по W
ax = axes[0, 0]
for W, res in results.items():
    ax.semilogy(iters_x, res["hist"]["resid"],
                color=res["col"], lw=2, label=f"W={W}")
ax.set_xlabel("Итерация"); ax.set_ylabel("Mean projection residual (log)")
ax.set_title("Сходимость LP: residual", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# [0,1] d_eff@90% по W
ax = axes[0, 1]
for W, res in results.items():
    ax.plot(iters_x, res["hist"]["d_eff"],
            color=res["col"], lw=2, label=f"W={W}  d_local={res['d_local']}")
ax.set_xlabel("Итерация"); ax.set_ylabel(f"d@{int(D_THRESH*100)}% пула")
ax.set_title("Эффективная размерность пула", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# [0,2] ACF lag-1 по W
ax = axes[0, 2]
for W, res in results.items():
    ax.plot(iters_x, res["hist"]["acf1"],
            color=res["col"], lw=2, label=f"W={W}")
ax.axhline(0,     color="black", lw=0.8)
ax.axhline( CI95, ls=":", color="gray", lw=1.2, label=f"±CI95 {CI95:.3f}")
ax.axhline(-CI95, ls=":", color="gray", lw=1.2)
ax.set_xlabel("Итерация"); ax.set_ylabel("ACF lag-1 (P_FIT-срезы)")
ax.set_title("Качество очистки: ACF lag-1 остатков", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# [1,0] Полный ACF на финальной итерации по W
ax = axes[1, 0]
for W, res in results.items():
    snaps = res["hist"]["acf_full"]
    if snaps:
        it_s, acf_s = snaps[-1]
        ax.plot(lags_x, acf_s, "o-", color=res["col"], lw=1.8, ms=4,
                label=f"W={W} (iter={it_s})")
ax.axhline(0,     color="black", lw=0.8)
ax.axhline( CI95, ls=":", color="gray", lw=1.2, label=f"±CI95")
ax.axhline(-CI95, ls=":", color="gray", lw=1.2)
ax.set_xlabel("Лаг"); ax.set_ylabel("ACF")
ax.set_title("Финальный ACF P_FIT-срезов (после LP)", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# [1,1] Полный ACF: снапшоты для W=P_MAX (основной кандидат)
ax = axes[1, 1]
main_W = P_MAX
if main_W in results:
    snap_cols_w = plt.cm.viridis(np.linspace(0.05, 0.95,
                                              len(results[main_W]["hist"]["acf_full"])))
    for (it_s, acf_s), sc_col in zip(results[main_W]["hist"]["acf_full"], snap_cols_w):
        ax.plot(lags_x, acf_s, "o-", color=sc_col, lw=1.5, ms=3, label=f"iter={it_s}")
    ax.axhline(0,     color="black", lw=0.8)
    ax.axhline( CI95, ls=":", color="gray", lw=1.2, label=f"±CI95 {CI95:.3f}")
    ax.axhline(-CI95, ls=":", color="gray", lw=1.2)
    ax.set_title(f"ACF снапшоты для W=P_MAX={main_W}", fontsize=11)
ax.set_xlabel("Лаг"); ax.set_ylabel("ACF"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# [1,2] d_local по порогам: сравнение W (grouped bar)
ax = axes[1, 2]
thr_labels = [f"{int(t*100)}%" for t in VAR_THRESH]
n_thr = len(VAR_THRESH)
n_w_r = len(results)
width = 0.8 / n_w_r
x_pos = np.arange(n_thr)
for i, (W, res) in enumerate(results.items()):
    d_vals = [res["d_at0"][t] for t in VAR_THRESH]
    xoff   = (i - (n_w_r - 1) / 2) * width
    bars   = ax.bar(x_pos + xoff, d_vals, width,
                    color=res["col"], alpha=0.85, edgecolor="white",
                    label=f"W={W}")
    for b, v in zip(bars, d_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v),
                ha="center", va="bottom", fontsize=7, fontweight="bold")
ax.set_xticks(x_pos); ax.set_xticklabels(thr_labels)
ax.set_ylabel("d_local"); ax.set_title("d_local по порогам и W", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out2 = "research/phase5_attractor/local_lp_theiler_fig2.png"
fig2.savefig(out2, dpi=130)
print(f"Фиг.2 сохранена: {out2}")

# ── Рис. 3: до/после LP для каждого W (2D PCA + P_FIT heatmap) ───────────────
fig3, axes = plt.subplots(2, n_w, figsize=(5 * n_w, 11))
fig3.suptitle(
    f"Геометрия пула до/после LP×{N_MAX_ITER}  |  "
    f"P_MAX={P_MAX}→P_FIT={P_FIT}  K={K}",
    fontsize=13, fontweight="bold"
)

for col_i, (W, res) in enumerate(results.items()):
    X_orig  = res["X_pool_orig"]
    X_final = res["X_pool_final"]
    Vt0     = res["Vt0"]
    col     = res["col"]
    cent0   = X_orig.mean(axis=0)
    pca_o   = (X_orig  - cent0) @ Vt0[:2].T
    pca_f   = (X_final - cent0) @ Vt0[:2].T
    x_pc    = (x_orig  - cent0) @ Vt0[:2].T

    # строка 0: 2D PCA до/после со стрелками
    ax = axes[0, col_i]
    for i in range(K):
        ax.annotate("",
                    xy=(pca_f[i, 0], pca_f[i, 1]),
                    xytext=(pca_o[i, 0], pca_o[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.6, alpha=0.4))
    sc_o = ax.scatter(pca_o[:, 0], pca_o[:, 1],
                      c=res["pool_inds"], cmap="Blues", s=45, alpha=0.6,
                      edgecolors="none", zorder=3, label="до LP")
    sc_f = ax.scatter(pca_f[:, 0], pca_f[:, 1],
                      c=res["pool_inds"], cmap="Oranges", s=55, alpha=0.9,
                      edgecolors="white", lw=0.3, zorder=4, label=f"LP×{N_MAX_ITER}")
    ax.scatter(*x_pc, marker="*", s=350, color="lime",
               edgecolor="black", lw=1, zorder=6, label="origin")
    ax.set_title(f"W={W}  d_local={res['d_local']}\n2D PCA до/после LP",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # строка 1: P_FIT-срезы heatmap: orig | final | residual
    ax = axes[1, col_i]
    b_orig  = X_orig[:, :P_FIT]
    b_final = X_final[:, :P_FIT]
    b_resid = b_orig - b_final
    scale   = np.percentile(np.abs(b_orig), 97) + 1e-10
    sep     = np.full((K, 1), np.nan)
    comb    = np.hstack([b_orig, sep, b_final, sep, b_resid])
    im = ax.imshow(comb / scale, aspect="auto", cmap="RdBu_r",
                   vmin=-1.5, vmax=1.5, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="норм.")
    ax.axvline(P_FIT + 0.5,       color="white", lw=2)
    ax.axvline(2 * P_FIT + 1.5,   color="white", lw=2)
    for xi, lbl_h in [(P_FIT/2, "оригинал"),
                      (P_FIT + 1 + P_FIT/2, f"LP×{N_MAX_ITER}"),
                      (2*P_FIT + 2 + P_FIT/2, "остаток")]:
        ax.text(xi, -2, lbl_h, ha="center", va="bottom", fontsize=8,
                transform=ax.transData)
    ax.set_ylabel("Сосед (ранг)"); ax.set_title(f"P_FIT-срезы  W={W}", fontsize=10)

plt.tight_layout()
out3 = "research/phase5_attractor/local_lp_theiler_fig3.png"
fig3.savefig(out3, dpi=130)
print(f"Фиг.3 сохранена: {out3}")

plt.show()
print("\nГотово.")
