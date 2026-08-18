"""
Итеративная PCA-проекция: сходимость локального многообразия.

На каждом шаге:
  1. Текущий пул k соседей → SVD → проекция всей истории на d-мерное подпространство
  2. Переискать k соседей в d-мерном пространстве
  3. Jaccard(новые, предыдущие)
  until стабилизация

Тестируем d по разным порогам дисперсии (из local SVD: d@80, d@90, d@95, d@99).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

DATA_PATH   = "data/candles/SBER/1d.json"
P           = 300
K           = 50
N_MAX       = 20
JACCARD_THR = 0.95
# d значения из нашего анализа для origin 2026-06-04 (raw d@..%)
D_VALUES    = {"d@80(raw)": 6, "d@90(raw)": 11, "d@95(raw)": 23, "d@99(raw)": 41}

# ── данные ───────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw_data = json.load(f)
close = np.array([c["close"] for c in raw_data], dtype=np.float64)

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

ratio  = close / np.maximum(logtrend_causal(close), 1e-10)
n_pts  = len(ratio) - P + 1
rows   = np.arange(n_pts)[:, None] + np.arange(P)[None, :]
X_full = ratio[rows]                     # (n_pts, P) — вся история
ORIGIN = n_pts - 1
X_past = X_full[:ORIGIN]                 # только прошлое (каузально)
x_orig = X_full[ORIGIN]

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)

# ── итеративная PCA-проекция ──────────────────────────────────────────────────
print(f"Origin: {raw_data[ORIGIN + P - 1]['begin'][:10]}  |  "
      f"P={P}  K={K}  history={len(X_past)} points\n")

results = {}
COLORS  = plt.cm.tab10(np.linspace(0, 0.7, len(D_VALUES)))

for (label, d), color in zip(D_VALUES.items(), COLORS):
    d_eff = min(d, K - 1)   # не больше числа соседей

    # Шаг 0: соседи в raw 300D
    _, inds0 = KDTree(X_past).query(x_orig.reshape(1, -1), k=K)
    prev_set  = set(inds0[0].tolist())
    prev_inds = inds0[0]

    jac_hist   = []    # Jaccard(n, n-1)
    n_star     = None
    space_dim  = [P]   # размерность пространства на каждом шаге

    print(f"── {label} (d={d_eff}) ───────────────────────────")
    print(f"  {'iter':>4}  {'J(n,n-1)':>10}  {'#новых':>8}  {'dim':>6}")

    cur_inds = prev_inds
    for it in range(1, N_MAX + 1):
        # SVD по текущим k соседям
        X_nn     = X_past[cur_inds]                  # (K, P)
        centroid = X_nn.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_nn - centroid, full_matrices=False)
        V_d      = Vt[:d_eff]                        # (d_eff, P)

        # проецируем ВСЮ историю + origin на d-мерное подпространство
        X_proj   = (X_past  - centroid) @ V_d.T      # (n_past, d_eff)
        x_q_proj = (x_orig  - centroid) @ V_d.T      # (d_eff,)

        # переискиваем соседей в d-мерном пространстве
        _, new_inds = KDTree(X_proj).query(x_q_proj.reshape(1, -1), k=K)
        new_set     = set(new_inds[0].tolist())

        jc      = jaccard(new_set, prev_set)
        n_new   = len(new_set - prev_set)
        jac_hist.append(jc)
        space_dim.append(d_eff)

        if n_star is None and jc >= JACCARD_THR:
            n_star = it

        print(f"  {it:4d}  {jc:10.3f}  {n_new:8d}  {d_eff:6d}")

        prev_set, prev_inds = new_set, new_inds[0]
        cur_inds = new_inds[0]

        if jc >= JACCARD_THR and it >= 3:
            print(f"  → сошлось при n={it}")
            break

    if n_star is None:
        print(f"  → НЕ сошлось за {N_MAX} итераций")

    results[label] = dict(d=d_eff, jac=jac_hist, n_star=n_star,
                          final_inds=cur_inds, color=color)
    print()

# ── дополнительно: что изменилось от raw до финала ───────────────────────────
_, raw_inds = KDTree(X_past).query(x_orig.reshape(1, -1), k=K)
raw_set = set(raw_inds[0].tolist())
print("Jaccard(финал, raw) по каждому d:")
for label, res in results.items():
    j = jaccard(set(res["final_inds"].tolist()), raw_set)
    print(f"  {label:20s}  d={res['d']:2d}  J(final,raw)={j:.3f}  "
          f"{'сошлось n='+str(res['n_star']) if res['n_star'] else 'не сошлось'}")

# ── график ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Итеративная PCA-проекция: сходимость соседей  |  "
             f"SBER 1d  |  P={P}  K={K}", fontsize=12)

ax = axes[0]
for label, res in results.items():
    iters = range(1, len(res["jac"]) + 1)
    ax.plot(iters, res["jac"], "o-", color=res["color"], lw=1.5,
            label=f"{label} (d={res['d']})")
    if res["n_star"]:
        ax.axvline(res["n_star"], color=res["color"], ls=":", lw=1)
ax.axhline(JACCARD_THR, color="gray", ls="--", lw=1,
           label=f"порог {JACCARD_THR}")
ax.set_xlabel("Итерация"); ax.set_ylabel("Jaccard(n, n-1)")
ax.set_title("Сходимость PCA-итерации")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

ax = axes[1]
# сравниваем финальные наборы соседей — временны́е индексы
for label, res in results.items():
    dates = [ORIGIN - i for i in sorted(res["final_inds"])]
    ax.scatter(sorted(res["final_inds"]),
               [list(D_VALUES.keys()).index(label)] * K,
               c=[res["color"]], s=20, alpha=0.7)
# raw соседи
ax.scatter(sorted(raw_inds[0]), [-1]*K, c=["gray"], s=20, alpha=0.5)
labels_all = ["raw (300D)"] + list(D_VALUES.keys())
ax.set_yticks(range(-1, len(D_VALUES)))
ax.set_yticklabels(labels_all, fontsize=8)
ax.set_xlabel("Индекс соседа в истории (0 = самый старый)")
ax.set_title("Распределение финальных соседей во времени")
ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
out = "research/phase5_attractor/pca_iter_stability.png"
plt.savefig(out, dpi=130)
print(f"\nГрафик сохранён: {out}")
plt.show()
