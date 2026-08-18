"""
26 — GMM: sweep по k_pca при p=20.

Скрипт 25 показал: GMM soft_k4 при p=20, k_pca=3 даёт −5.4% (незначимо).
Но k_pca=3 объясняет лишь 17% дисперсии при p=20.

Гипотеза: при k_pca=8–10 GMM видит больше структуры фазового пространства
→ кластеры качественнее → эффект усиливается.

Sweep: k_pca ∈ {3, 5, 8, 10}, k_gmm ∈ {3, 4, 5}, p=20, SBER 1d, N=200.
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
    print("Нужен sklearn: pip install scikit-learn"); sys.exit(1)

# ── параметры ─────────────────────────────────────────────────────────────────
TICKER    = "SBER"
INTERVAL  = "1d"
MA_WINDOW = 1000

P         = 20
XI        = 3 * (P + 1)   # 63
K_PCA_LIST = [3, 5, 8, 10]
K_GMM_LIST = [3, 4, 5]
_VAL_H    = 10
_N_EVAL   = 200
GMM_SEED  = 42


# ── LWR ──────────────────────────────────────────────────────────────────────
def _lwr_step(Xm: np.ndarray, ym: np.ndarray, vec: np.ndarray,
              xi: int, prob: np.ndarray | None = None) -> float:
    dists  = np.linalg.norm(Xm - vec, axis=1)
    nn_idx = np.argpartition(dists, xi)[:xi]
    nn_dx  = dists[nn_idx]
    h_bw   = max(float(nn_dx.max()), 1e-10)
    w      = np.exp(-0.5 * (nn_dx / h_bw) ** 2)
    if prob is not None:
        w *= prob[nn_idx]
        w  = np.clip(w, 1e-30, None)
    A  = np.hstack([np.ones((xi, 1)), Xm[nn_idx]])
    sw = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * ym[nn_idx], rcond=None)
    return float(c[0] + vec @ c[1:])


def _mape(X: np.ndarray, y: np.ndarray, vec: np.ndarray,
          ratio: np.ndarray, val_origin: int, xi: int,
          mask: np.ndarray | None, prob: np.ndarray | None) -> float:
    if mask is not None and mask.sum() >= xi:
        Xu, yu, pu = X[mask], y[mask], (prob[mask] if prob is not None else None)
    else:
        Xu, yu, pu = X, y, prob

    v   = vec.copy()
    hat = np.empty(_VAL_H)
    for h in range(_VAL_H):
        hat[h] = _lwr_step(Xu, yu, v, xi, pu)
        v = np.roll(v, -1); v[-1] = hat[h]

    r0     = float(ratio[val_origin])
    r_hat  = r0 + np.cumsum(hat)
    actual = ratio[val_origin + 1: val_origin + 1 + _VAL_H]
    n_ = min(len(r_hat), len(actual))
    if n_ == 0:
        return np.nan
    return float(np.mean(
        np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)
    ))


# ── данные ────────────────────────────────────────────────────────────────────
with open(DATA_DIR / TICKER / f"{INTERVAL}.json") as f:
    candles = json.load(f)
df     = normalize(candles, window=MA_WINDOW).dropna(subset=["ma"]).reset_index(drop=True)
ratio  = df["ratio"].values
dratio = np.diff(ratio)
print(f"{TICKER} {INTERVAL}: {len(dratio)} баров Δratio")

X_all, y_all = build_delay_matrix(dratio, P)

min_orig = P + XI + 10
max_orig = len(dratio) - _VAL_H
origins  = np.arange(max(min_orig, max_orig - _N_EVAL), max_orig)
print(f"p={P}  ξ={XI}  origins={len(origins)}\n")


# ── baseline (один раз) ───────────────────────────────────────────────────────
print("Baseline...", end=" ", flush=True)
t0 = time.time()
base_mape = []
for vo in origins:
    pool = vo - P
    if pool < XI + 2: continue
    base_mape.append(_mape(
        X_all[:pool], y_all[:pool],
        last_vector(dratio[:vo], P).copy(),
        ratio, vo, XI, None, None
    ))
base_mape = np.array(base_mape)
print(f"med={np.nanmedian(base_mape):.5f}  t={time.time()-t0:.1f}s\n")


# ── sweep по k_pca ────────────────────────────────────────────────────────────
# RES[k_pca][(k_gmm, mode)] = np.ndarray(mape)
# mode: "hard" | "soft"
RES: dict[int, dict[tuple, np.ndarray]] = {}

for k_pca in K_PCA_LIST:
    pca     = PCA(n_components=k_pca).fit(X_all)
    X_pca_a = pca.transform(X_all)
    ev      = pca.explained_variance_ratio_.sum()

    RES[k_pca] = {}
    raw: dict[tuple, list] = {}
    for k in K_GMM_LIST:
        raw[(k, "hard")] = []
        raw[(k, "soft")] = []

    t1 = time.time()
    print(f"k_pca={k_pca}  explained_var={ev:.3f}", end="  ", flush=True)

    for vo in origins:
        pool = vo - P
        if pool < XI + 2: continue

        X_p   = X_all[:pool]
        y_p   = y_all[:pool]
        X_pca = X_pca_a[:pool]
        vec   = last_vector(dratio[:vo], P).copy()
        vpca  = pca.transform(vec.reshape(1, -1))[0]

        for k_gmm in K_GMM_LIST:
            gmm    = GaussianMixture(n_components=k_gmm, random_state=GMM_SEED,
                                     max_iter=100, n_init=1).fit(X_pca)
            labels = gmm.predict(X_pca)
            qlbl   = int(gmm.predict(vpca.reshape(1, -1))[0])
            mask   = (labels == qlbl)
            probs  = gmm.predict_proba(X_pca)[:, qlbl]

            raw[(k_gmm, "hard")].append(
                _mape(X_p, y_p, vec, ratio, vo, XI, mask, None)
            )
            raw[(k_gmm, "soft")].append(
                _mape(X_p, y_p, vec, ratio, vo, XI, None, probs)
            )

    for key in raw:
        RES[k_pca][key] = np.array(raw[key])

    dt = time.time() - t1
    best_key = min(RES[k_pca], key=lambda k: float(np.nanmedian(RES[k_pca][k])))
    bm = float(np.nanmedian(RES[k_pca][best_key]))
    bm_base = float(np.nanmedian(base_mape))
    print(f"best={best_key}  med={bm:.5f}  Δ={( bm-bm_base)/bm_base*100:+.2f}%  t={dt:.1f}s")


# ── сводная таблица ───────────────────────────────────────────────────────────
try:
    from scipy.stats import wilcoxon as _wlcx
    _HAS_SCI = True
except ImportError:
    _HAS_SCI = False

bm_base = float(np.nanmedian(base_mape))
print("\n" + "=" * 80)
print(f"baseline_med = {bm_base:.5f}  (p={P}, ξ={XI}, N={len(base_mape)})")
print(f"\n{'k_pca':>6}  {'ev':>5}  {'k_gmm':>5}  {'mode':>5}  "
      f"{'med':>9}  {'Δ%':>7}  {'W-p':>8}")
print("─" * 60)

pca_ev = {}
for k_pca in K_PCA_LIST:
    pca_tmp = PCA(n_components=k_pca).fit(X_all)
    pca_ev[k_pca] = pca_tmp.explained_variance_ratio_.sum()

best_overall = None
best_delta   = 0.0

for k_pca in K_PCA_LIST:
    ev = pca_ev[k_pca]
    for mode in ("hard", "soft"):
        for k_gmm in K_GMM_LIST:
            arr = RES[k_pca][(k_gmm, mode)]
            med = float(np.nanmedian(arr))
            dp  = (med - bm_base) / bm_base * 100
            wp  = "—"
            if _HAS_SCI:
                ok   = ~(np.isnan(base_mape) | np.isnan(arr))
                diff = arr[ok] - base_mape[ok]
                if ok.sum() >= 10 and not np.all(diff == 0):
                    _, pv = _wlcx(diff)
                    wp = f"{pv:.4f}"
            mark = " ◄" if dp < best_delta else ""
            if dp < best_delta:
                best_delta   = dp
                best_overall = (k_pca, k_gmm, mode)
                mark = " ◄ BEST"
            print(f"{k_pca:>6}  {ev:>5.3f}  {k_gmm:>5}  {mode:>5}  "
                  f"{med:>9.5f}  {dp:>+6.2f}%  {wp:>8}{mark}")
        print()
    print()

print("=" * 80)
if best_overall:
    k_pca, k_gmm, mode = best_overall
    print(f"Лучшая конфигурация: k_pca={k_pca}, k_gmm={k_gmm}, mode={mode}  "
          f"Δ={best_delta:+.2f}%")


# ── heatmap Δ% (k_pca × k_gmm) для лучшего mode ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"GMM k_pca sweep  |  {TICKER} {INTERVAL}  p={P}, ξ={XI}, N={_N_EVAL}",
    fontsize=11,
)

for ax, mode in zip(axes, ("hard", "soft")):
    mat = np.zeros((len(K_PCA_LIST), len(K_GMM_LIST)))
    for i, k_pca in enumerate(K_PCA_LIST):
        for j, k_gmm in enumerate(K_GMM_LIST):
            arr = RES[k_pca][(k_gmm, mode)]
            med = float(np.nanmedian(arr))
            mat[i, j] = (med - bm_base) / bm_base * 100

    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto",
                   vmin=-10, vmax=10)
    ax.set_xticks(range(len(K_GMM_LIST))); ax.set_xticklabels([f"k={k}" for k in K_GMM_LIST])
    ax.set_yticks(range(len(K_PCA_LIST)));
    ax.set_yticklabels([f"k_pca={k}\n(ev={pca_ev[k]:.2f})" for k in K_PCA_LIST])
    ax.set_xlabel("k_gmm")
    ax.set_title(f"mode={mode}  (Δ% vs baseline, зелёный = лучше)")
    plt.colorbar(im, ax=ax, label="Δ MAPE %")
    for i in range(len(K_PCA_LIST)):
        for j in range(len(K_GMM_LIST)):
            ax.text(j, i, f"{mat[i,j]:+.1f}%", ha="center", va="center",
                    fontsize=9, color="black")

plt.tight_layout()
out_path = OUT_DIR / "26_gmm_kpca_sweep.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")


# ── CDF лучших конфигов vs baseline ──────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.set_title(f"Лучшие конфиги GMM vs baseline  |  {TICKER} {INTERVAL}", fontsize=11)

s = np.sort(base_mape[~np.isnan(base_mape)])
ax2.plot(np.linspace(0, 100, len(s)), s, color="steelblue", lw=2.5,
         label=f"baseline  (med={bm_base:.5f})")

PALETTE = ["tomato", "seagreen", "darkorange", "purple",
           "brown", "teal", "navy", "olive"]
ci = 0
for k_pca in K_PCA_LIST:
    for mode in ("hard", "soft"):
        best_k = min(K_GMM_LIST,
                     key=lambda k: float(np.nanmedian(RES[k_pca][(k, mode)])))
        arr = RES[k_pca][(best_k, mode)]
        med = float(np.nanmedian(arr))
        dp  = (med - bm_base) / bm_base * 100
        if dp < -1.0:
            sv = np.sort(arr[~np.isnan(arr)])
            ax2.plot(np.linspace(0, 100, len(sv)), sv,
                     color=PALETTE[ci % len(PALETTE)], lw=1.5,
                     label=f"k_pca={k_pca} {mode} k_gmm={best_k}  "
                            f"(med={med:.5f} Δ={dp:+.1f}%)")
            ci += 1

ax2.set_xlabel("Перцентиль")
ax2.set_ylabel("val_mape")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylim(0)
plt.tight_layout()
out2 = OUT_DIR / "26_gmm_kpca_cdf.png"
plt.savefig(out2, dpi=140, bbox_inches="tight")
print(f"CDF: {out2}")

plt.show()
print("\n── Готово ──")
