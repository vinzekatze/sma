"""
25 — GMM-кластеризация в PCA-пространстве delay-матрицы.

Уровень 3 расслоения фазового пространства (phase_space_stratification.md):
прямая геометрическая кластеризация траектории в пространстве задержек —
без косвенных PE/Hurst прокси.

Гипотеза: GMM разбивает delay-матрицу X на «естественные зоны аттрактора».
При поиске соседей ограничиваем пул только кластером текущего query-вектора.
Если соседи внутри кластера геометрически ближе — прогноз улучшается.

Реализация:
  1. PCA(k_pca) обучается один раз на всём пуле (coordinate transform).
  2. GMM(k_gmm) обучается на каждом origin — только на доступной истории.
  3. Hard mask: ξ соседей ищем только внутри кластера query.
     Fallback: если кластер < xi_min соседей — используем полный пул.
  4. Soft weight: w = w_lwr × P(cluster_q | x_i) / max_prob
     (posterior вероятность принадлежности к кластеру query).

Тест: SBER 1d, p ∈ {5, 20}, k_gmm ∈ {3, 4, 5}, N=200 origins.
"""

import json, sys, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from sma.core.forecast.normalize import normalize
from sma.core.forecast.embedding import build_delay_matrix, last_vector

try:
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
except ImportError:
    print("Нужен sklearn: pip install scikit-learn")
    sys.exit(1)

# ── параметры ────────────────────────────────────────────────────────────────
TICKER    = "SBER"
INTERVAL  = "1d"
MA_WINDOW = 1000

P_LIST        = [5, 20]
K_GMM_LIST    = [3, 4, 5]
K_PCA         = 3          # компоненты PCA (одинаково для всех p)
_VAL_H        = 10
_N_EVAL       = 200
GMM_SEED      = 42


# ── LWR (baseline) ───────────────────────────────────────────────────────────
def _lwr(X: np.ndarray, y: np.ndarray, vec: np.ndarray,
         xi: int, mask: np.ndarray | None = None,
         gmm_prob: np.ndarray | None = None) -> float:
    """Один шаг LWR. mask ограничивает пул; gmm_prob — мягкий вес."""
    if mask is not None and mask.sum() >= xi:
        Xm, ym = X[mask], y[mask]
        prob_m = gmm_prob[mask] if gmm_prob is not None else None
    else:
        Xm, ym = X, y
        prob_m = gmm_prob

    dists  = np.linalg.norm(Xm - vec, axis=1)
    nn_idx = np.argpartition(dists, xi)[:xi]
    nn_dx  = dists[nn_idx]
    h_bw   = max(float(nn_dx.max()), 1e-10)
    w      = np.exp(-0.5 * (nn_dx / h_bw) ** 2)

    if prob_m is not None:
        w *= prob_m[nn_idx]
        w  = np.clip(w, 1e-30, None)

    A  = np.hstack([np.ones((xi, 1)), Xm[nn_idx]])
    sw = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * ym[nn_idx], rcond=None)
    return float(c[0] + vec @ c[1:])


def _eval_origin(X: np.ndarray, y: np.ndarray, vec: np.ndarray,
                 ratio: np.ndarray, val_origin: int, xi: int,
                 mask: np.ndarray | None, gmm_prob: np.ndarray | None) -> tuple[float, float]:
    """Walk-forward VAL_H шагов. Возвращает (mape, d_k_full)."""
    v   = vec.copy()
    hat = np.empty(_VAL_H)
    for h in range(_VAL_H):
        pred  = _lwr(X, y, v, xi, mask, gmm_prob)
        hat[h] = pred
        v = np.roll(v, -1); v[-1] = pred

    # d_k на полном пуле (без маски) — для диагностики
    dists_full = np.linalg.norm(X - vec, axis=1)
    d_k = float(np.sort(dists_full)[xi - 1])

    r0     = float(ratio[val_origin])
    r_hat  = r0 + np.cumsum(hat)
    actual = ratio[val_origin + 1: val_origin + 1 + _VAL_H]
    n_ = min(len(r_hat), len(actual))
    if n_ == 0:
        return np.nan, d_k
    mape = float(np.mean(
        np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)
    ))
    return mape, d_k


# ── данные ────────────────────────────────────────────────────────────────────
with open(DATA_DIR / TICKER / f"{INTERVAL}.json") as f:
    candles = json.load(f)
df     = normalize(candles, window=MA_WINDOW).dropna(subset=["ma"]).reset_index(drop=True)
ratio  = df["ratio"].values
dratio = np.diff(ratio)
print(f"{TICKER} {INTERVAL}: {len(dratio)} баров Δratio\n")


# ── sweep по p ────────────────────────────────────────────────────────────────
# RES[p][config] = {"mape": np.ndarray, "dk": np.ndarray, "fallbacks": int}
# config: "baseline", "hard_k3", "hard_k4", "hard_k5", "soft_k3", ...
RES: dict[int, dict[str, dict]] = {}

for p in P_LIST:
    xi = 3 * (p + 1)

    # PCA обучается один раз на всём dratio
    X_all, y_all = build_delay_matrix(dratio, p)
    pca = PCA(n_components=min(K_PCA, p)).fit(X_all)
    X_pca_all = pca.transform(X_all)

    min_orig = p + xi + 10
    max_orig = len(dratio) - _VAL_H
    origins  = np.arange(max(min_orig, max_orig - _N_EVAL), max_orig)

    configs  = ["baseline"] + [f"hard_k{k}" for k in K_GMM_LIST] \
                             + [f"soft_k{k}" for k in K_GMM_LIST]
    RES[p] = {c: {"mape": [], "dk": [], "fallbacks": 0} for c in configs}

    print(f"p={p}  ξ={xi}  k_pca={pca.n_components_}  "
          f"explained_var={pca.explained_variance_ratio_.sum():.2f}  "
          f"origins={len(origins)}")

    t0 = time.time()
    for vo in origins:
        pool = vo - p         # число строк X для этого origin
        if pool < xi + 2:
            continue
        X_pool   = X_all[:pool]
        y_pool   = y_all[:pool]
        X_pca_p  = X_pca_all[:pool]
        vec      = last_vector(dratio[:vo], p).copy()
        vec_pca  = pca.transform(vec.reshape(1, -1))[0]

        # baseline (без кластеризации)
        mape_b, dk_b = _eval_origin(X_pool, y_pool, vec, ratio, vo, xi, None, None)
        RES[p]["baseline"]["mape"].append(mape_b)
        RES[p]["baseline"]["dk"].append(dk_b)

        for k_gmm in K_GMM_LIST:
            gmm = GaussianMixture(
                n_components=k_gmm, random_state=GMM_SEED,
                max_iter=100, n_init=1,
            ).fit(X_pca_p)
            labels   = gmm.predict(X_pca_p)
            q_label  = int(gmm.predict(vec_pca.reshape(1, -1))[0])
            mask     = (labels == q_label)

            # posterior вероятности принадлежности к кластеру query
            probs    = gmm.predict_proba(X_pca_p)[:, q_label]

            # hard mask
            fallback = mask.sum() < xi
            if fallback:
                RES[p][f"hard_k{k_gmm}"]["fallbacks"] += 1
            mape_h, dk_h = _eval_origin(
                X_pool, y_pool, vec, ratio, vo, xi,
                mask if not fallback else None, None
            )
            RES[p][f"hard_k{k_gmm}"]["mape"].append(mape_h)
            RES[p][f"hard_k{k_gmm}"]["dk"].append(dk_h)

            # soft weight (posterior)
            mape_s, dk_s = _eval_origin(
                X_pool, y_pool, vec, ratio, vo, xi, None, probs
            )
            RES[p][f"soft_k{k_gmm}"]["mape"].append(mape_s)
            RES[p][f"soft_k{k_gmm}"]["dk"].append(dk_s)

    for c in configs:
        RES[p][c]["mape"] = np.array(RES[p][c]["mape"])
        RES[p][c]["dk"]   = np.array(RES[p][c]["dk"])

    print(f"  t={time.time()-t0:.1f}s\n")


# ── таблица ───────────────────────────────────────────────────────────────────
try:
    from scipy.stats import wilcoxon as _wlcx
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

print("=" * 80)
for p in P_LIST:
    xi   = 3 * (p + 1)
    bm   = float(np.nanmedian(RES[p]["baseline"]["mape"]))
    print(f"\np={p}  ξ={xi}  baseline_med={bm:.5f}")
    print(f"  {'Конфиг':<14}  {'med':>9}  {'Δ%':>7}  {'d_k med':>8}  "
          f"{'fallback':>8}  {'W-p':>8}")
    print("  " + "─" * 62)
    for c in [f"hard_k{k}" for k in K_GMM_LIST] \
           + [f"soft_k{k}" for k in K_GMM_LIST]:
        arr  = RES[p][c]["mape"]
        dk   = RES[p][c]["dk"]
        fb   = RES[p][c]["fallbacks"]
        med  = float(np.nanmedian(arr))
        dk_m = float(np.nanmedian(dk))
        dp   = (med - bm) / bm * 100
        wp   = "—"
        if _HAS_SCIPY:
            base = RES[p]["baseline"]["mape"]
            ok   = ~(np.isnan(base) | np.isnan(arr))
            diff = arr[ok] - base[ok]
            if ok.sum() >= 10 and not np.all(diff == 0):
                _, pv = _wlcx(diff)
                wp = f"{pv:.4f}"
        print(f"  {c:<14}  {med:>9.5f}  {dp:>+6.2f}%  {dk_m:>8.5f}  "
              f"{fb:>8}  {wp:>8}")
print("\n" + "=" * 80)


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"GMM-кластеризация в PCA-пространстве  |  {TICKER} {INTERVAL}  "
    f"k_pca={K_PCA}  N={_N_EVAL}",
    fontsize=11,
)

COLORS = {
    "baseline": "steelblue",
    "hard_k3":  "tomato",    "hard_k4": "orangered", "hard_k5": "darkred",
    "soft_k3":  "seagreen",  "soft_k4": "limegreen", "soft_k5": "darkgreen",
}

for pi, p in enumerate(P_LIST):
    bm = float(np.nanmedian(RES[p]["baseline"]["mape"]))

    ax_c = axes[pi][0]   # CDF MAPE
    ax_d = axes[pi][1]   # CDF d_k

    for c, arr in RES[p].items():
        med = float(np.nanmedian(arr["mape"]))
        s   = np.sort(arr["mape"][~np.isnan(arr["mape"])])
        lw  = 2.5 if c == "baseline" else 1.5
        lbl = f"{c}  (med={med:.4f})"
        ax_c.plot(np.linspace(0, 100, len(s)), s,
                  color=COLORS.get(c, "gray"), linewidth=lw, label=lbl)

        med_dk = float(np.nanmedian(arr["dk"]))
        s_dk   = np.sort(arr["dk"][~np.isnan(arr["dk"])])
        ax_d.plot(np.linspace(0, 100, len(s_dk)), s_dk,
                  color=COLORS.get(c, "gray"), linewidth=lw,
                  label=f"{c}  (d_k={med_dk:.5f})")

    ax_c.set_title(f"p={p}  CDF val_mape")
    ax_c.set_xlabel("Перцентиль")
    ax_c.set_ylabel("val_mape")
    ax_c.legend(fontsize=7)
    ax_c.grid(alpha=0.3)
    ax_c.set_ylim(0)

    ax_d.set_title(f"p={p}  CDF d_k (дистанция ξ-го соседа)")
    ax_d.set_xlabel("Перцентиль")
    ax_d.set_ylabel("d_k")
    ax_d.legend(fontsize=7)
    ax_d.grid(alpha=0.3)
    ax_d.set_ylim(0)

plt.tight_layout()
out_path = OUT_DIR / "25_gmm_pca_cluster.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
