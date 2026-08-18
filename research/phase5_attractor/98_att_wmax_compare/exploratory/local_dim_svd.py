"""
Локальная размерность аттрактора через SVD соседского окна.

Для нескольких origin-точек: берём delay embedding p=300, находим k=50
ближайших соседей (только из прошлого), делаем SVD локального кластера,
строим скри-плот. Колено даёт локальный d.

Никакого LP, никакой глобальной обработки — чисто локальная геометрия
в той области фазового пространства, где работает каскад.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

DATA_PATH  = "data/candles/SBER/1d.json"
P          = 300    # окно задержки (как в каскаде)
K          = 50     # соседей для SVD
# проверяем несколько origin: смещения от конца ряда
ORIGIN_OFFSETS = [0, 50, 150, 300, 600]
VAR_THRESH = [0.80, 0.90, 0.95, 0.99]   # пороги кумулятивной дисперсии

# ── данные ───────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.load(f)
close = np.array([c["close"] for c in raw], dtype=np.float64)

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
N = len(ratio)

# ── LP-очищенный att (стандарт d=3 n=3) для сравнения ────────────────────────
LP_D, LP_N, LP_K = 3, 3, 30

def lp_clean(s, d, n_iter, k):
    m = 2*d + 3; k_eff = min(k, len(s)-m-1); d_eff = min(d, m-1)
    n_pts = len(s) - m + 1
    rows  = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
    cur   = s.copy()
    for _ in range(n_iter):
        X = cur[rows]
        _, inds = KDTree(X).query(X, k=k_eff+1)
        X_nn = X[inds[:, 1:]]
        cents = X_nn.mean(axis=1)
        Xc    = X_nn - cents[:, None, :]
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        Vd   = Vt[:, :d_eff, :].transpose(0, 2, 1)
        xc   = X - cents
        proj = np.einsum("nmd,nd->nm", Vd, np.einsum("nmd,nm->nd", Vd, xc))
        Xp   = cents + proj
        res  = np.zeros(len(s)); cnt = np.zeros(len(s), dtype=np.int32)
        for j in range(m):
            res[j:j+n_pts] += Xp[:, j]; cnt[j:j+n_pts] += 1
        cur  = res / np.maximum(cnt, 1)
    return cur

print(f"Вычисляю LP att (d={LP_D} n={LP_N} k={LP_K})...")
att = lp_clean(ratio, LP_D, LP_N, LP_K)
SIGNALS = {"raw ratio": ratio, f"LP att (d={LP_D},n={LP_N})": att}

# ── delay embedding и анализ для двух сигналов ───────────────────────────────
n_pts = N - P + 1
rows  = np.arange(n_pts)[:, None] + np.arange(P)[None, :]
print(f"SBER 1d: {N} баров  |  delay embedding: {n_pts} × {P}")

all_results = {}   # signal_name → list of dicts

for sig_name, sig in SIGNALS.items():
    X = sig[rows]
    print(f"\n── {sig_name} ─────────────────────────────────────────")
    print(f"{'date':>12}  ", end="")
    for thr in VAR_THRESH:
        print(f" d@{int(thr*100)}%", end="")
    print()

    res_list = []
    for offset in ORIGIN_OFFSETS:
        origin_idx = n_pts - 1 - offset
        if origin_idx < K + 1:
            continue
        x_q    = X[origin_idx]
        past_X = X[:origin_idx]
        _, inds = KDTree(past_X).query(x_q.reshape(1, -1), k=K)
        X_nn   = past_X[inds[0]]
        Xc     = X_nn - X_nn.mean(axis=0)
        _, sv, _ = np.linalg.svd(Xc, full_matrices=False)
        var_frac = sv**2 / (sv**2).sum()
        cumvar   = np.cumsum(var_frac)
        d_at = {thr: int(np.searchsorted(cumvar, thr)) + 1 for thr in VAR_THRESH}
        bar_date = raw[origin_idx + P - 1]["begin"][:10]
        print(f"{bar_date:>12}  ", end="")
        for thr in VAR_THRESH:
            print(f" {d_at[thr]:>5}", end="")
        print()
        res_list.append(dict(offset=offset, origin_idx=origin_idx, date=bar_date,
                             sv=sv, cumvar=cumvar, d_at=d_at))
    all_results[sig_name] = res_list

# ── графики ──────────────────────────────────────────────────────────────────
sig_names  = list(all_results.keys())
sig_styles = ["-", "--"]
orig_colors = plt.cm.tab10(np.linspace(0, 0.8, len(ORIGIN_OFFSETS)))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"Локальная SVD: raw vs LP att  |  SBER 1d  |  p={P}  k={K}", fontsize=13)

for row_i, (sig_name, res_list) in enumerate(all_results.items()):
    ax_scree = axes[row_i, 0]
    ax_cumv  = axes[row_i, 1]
    ax_bar   = axes[row_i, 2]

    for r, col in zip(res_list, orig_colors):
        ax_scree.semilogy(range(1, K+1), r["sv"], color=col, lw=1.2,
                          label=f"{r['date']}")
        ax_cumv.plot(range(1, K+1), r["cumvar"]*100, color=col, lw=1.2,
                     label=r["date"])

    ax_scree.set_xlabel("Компонента"); ax_scree.set_ylabel("SV (log)")
    ax_scree.set_title(f"Скри-плот  [{sig_name}]")
    ax_scree.legend(fontsize=7); ax_scree.grid(True, alpha=0.3)

    for thr in VAR_THRESH:
        ax_cumv.axhline(thr*100, color="gray", ls="--", lw=0.7)
        ax_cumv.text(K*0.97, thr*100+0.3, f"{int(thr*100)}%",
                     ha="right", fontsize=7, color="gray")
    ax_cumv.set_xlabel("Компонента"); ax_cumv.set_ylabel("Cumvar %")
    ax_cumv.set_title(f"Кум. дисперсия  [{sig_name}]")
    ax_cumv.legend(fontsize=7); ax_cumv.grid(True, alpha=0.3)

    x_pos = np.arange(len(res_list)); width = 0.18
    for i, thr in enumerate(VAR_THRESH):
        d_vals = [r["d_at"][thr] for r in res_list]
        ax_bar.bar(x_pos + i*width, d_vals, width,
                   label=f"{int(thr*100)}%", alpha=0.85)
    ax_bar.set_xticks(x_pos + width*1.5)
    ax_bar.set_xticklabels([r["date"][2:] for r in res_list],
                            rotation=25, fontsize=8)
    ax_bar.set_ylabel("d"); ax_bar.set_title(f"d по порогам  [{sig_name}]")
    ax_bar.legend(fontsize=7); ax_bar.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = "research/phase5_attractor/local_dim_svd.png"
plt.savefig(out, dpi=130)
print(f"\nГрафик сохранён: {out}")
plt.show()
