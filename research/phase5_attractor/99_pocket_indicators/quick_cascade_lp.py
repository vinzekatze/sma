"""
Сравнение att: стандартный LP(n=3) vs поэтапный LP с убывающим d.

Идея: LP с убывающим d каждый раз проецирует на всё более жёсткое многообразие.
Вопрос: даёт ли это меньший d_L итогового att?

LP работает на ratio (сигнал), diff только в конце.
"""
import json, warnings, time
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[3]
_HERE   = Path(__file__).resolve().parent
DATADIR = ROOT / "data" / "candles"
TICKER  = "SBER"
K_NN    = 50
T_EXCL  = 50
LP_W    = 600

def logtrend_causal(close):
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n+1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2)
    cy = np.cumsum(lc); cty = np.cumsum(t*lc)
    den = cn*ct2 - ct**2
    b = np.where(den > 0, (cn*cty - ct*cy)/den, 0.0)
    a = (cy - b*ct)/cn
    tr = np.exp(a + b*t); tr[:2] = close[:2]
    return tr

def lp_pass(x, m, d, k):
    """Один проход LP, возвращает сигнал той же длины."""
    s = x.copy().astype(np.float64); N = len(s)
    M = N - m + 1
    X = np.lib.stride_tricks.sliding_window_view(s, m).copy()
    k_eff = min(k, M - 1); d_eff = min(d, k_eff - 1)
    D = cdist(X, X); np.fill_diagonal(D, np.inf)
    nn_idx = np.argpartition(D, k_eff, axis=1)[:, :k_eff]; del D
    nbrs = X[nn_idx]; centers = nbrs.mean(axis=1, keepdims=True)
    nbrs_c = nbrs - centers
    C = np.einsum("bki,bkj->bij", nbrs_c, nbrs_c)
    _, vecs = np.linalg.eigh(C); Vd = vecs[:, :, -d_eff:]
    xc = X - centers[:,0,:]; coef = np.einsum("bm,bmd->bd", xc, Vd)
    proj = np.einsum("bd,bmd->bm", coef, Vd); Xp = centers[:,0,:] + proj
    res = np.zeros(N); cnt = np.zeros(N, int)
    idx2d = np.arange(M)[:,None] + np.arange(m)[None,:]
    np.add.at(res, idx2d, Xp); np.add.at(cnt, idx2d, 1)
    return res / np.maximum(cnt, 1)

def twonn(sig, p_emb=9, k=K_NN, t_excl=T_EXCL):
    n = len(sig); n_lib = n - p_emb
    if n_lib < k + t_excl + 5: return np.nan
    idx   = np.arange(n_lib)[:,None] + np.arange(p_emb)[None,:]
    X_lib = sig[idx]; query = sig[-p_emb:]
    cand  = np.arange(n_lib - t_excl)
    dists = np.linalg.norm(X_lib[cand] - query, axis=1)
    k_eff = min(k, len(cand))
    nn    = cand[np.argpartition(dists, k_eff-1)[:k_eff]]
    X_nn  = X_lib[nn]
    D2 = np.sum((X_nn[:,None,:] - X_nn[None,:,:])**2, axis=2)
    np.fill_diagonal(D2, np.inf)
    top2 = np.partition(D2, 1, axis=1)[:, :2]
    r1 = np.sqrt(np.maximum(top2[:,0], 1e-30))
    r2 = np.sqrt(np.maximum(top2[:,1], 1e-30))
    mu = r2/r1; mu = mu[mu > 1.0]
    if len(mu) < 2: return np.nan
    return float((len(mu)-1) / np.sum(np.log(mu)))

def smoothness(att):
    """Среднее |второй разности| — мера гладкости (меньше = глаже)."""
    return float(np.mean(np.abs(np.diff(att, 2))))

def run_scheme(win, steps):
    """
    steps: список (d, m, k) — параметры каждого LP-прохода.
    Возвращает att = np.diff(финального сигнала).
    """
    s = win.copy()
    for d, m, k in steps:
        s = lp_pass(s, m=m, d=d, k=k)
    return np.diff(s)

# ── данные ────────────────────────────────────────────────────────────────────
close = np.array([c["close"] for c in
    json.loads((DATADIR / TICKER / "1d.json").read_text())], dtype=np.float64)
lt    = logtrend_causal(close)
ratio = close / np.maximum(lt, 1e-10)
N     = len(ratio)
t_orig = N - 60
win   = ratio[t_orig - LP_W + 2 : t_orig + 2]   # 600 баров ratio

print(f"Тикер: {TICKER},  окно LP_W={LP_W},  K_NN={K_NN},  T_EXCL={T_EXCL}\n")

# ── Схемы ─────────────────────────────────────────────────────────────────────
# m = 2d+3 (≥ Такенс 2d+1, +2 запас), k = max(30, 5d)
def params(d): return (d, max(9, 2*d+3), max(30, 5*d))

schemes = {
    "standard  LP(d=3, n=3)":      [params(3)]*3,
    "cascade   15→10→6→3":         [params(15), params(10), params(6),  params(3)],
    "cascade   15→8→4→3":          [params(15), params(8),  params(4),  params(3)],
    "cascade   12→7→3":            [params(12), params(7),  params(3)],
    "cascade   15→3 (2 шага)":     [params(15), params(3)],
    "single    LP(d=15, n=1)":     [params(15)],
}

print(f"{'Схема':<32}  {'d_L(9D)':>7}  {'smoothness':>11}  {'time':>6}")
print("─" * 62)

for name, steps in schemes.items():
    t0 = time.time()
    att = run_scheme(win, steps)
    dt  = time.time() - t0
    dL  = twonn(att, p_emb=9)
    sm  = smoothness(att)
    print(f"{name:<32}  {dL:>7.2f}  {sm:>11.6f}  {dt:>5.1f}s")

# Дополнительно: dratio без LP (baseline)
dratio = np.diff(win)
dL_raw = twonn(dratio, p_emb=9)
sm_raw = smoothness(dratio)
print(f"{'dratio (без LP)':<32}  {dL_raw:>7.2f}  {sm_raw:>11.6f}  {'—':>6}")

# ── Графики ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(schemes) + 1, 1, figsize=(14, 3*(len(schemes)+1)), sharex=True)
fig.suptitle(f"Сравнение att: стандартный LP vs поэтапный  ({TICKER})", fontsize=13)

att_dict = {}
for name, steps in schemes.items():
    att_dict[name] = run_scheme(win, steps)
att_dict["dratio (без LP)"] = np.diff(win)

for ax, (name, att) in zip(axes, att_dict.items()):
    dL = twonn(att, p_emb=9)
    sm = smoothness(att)
    ax.plot(att, lw=0.7, color="steelblue")
    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.set_ylabel(name, fontsize=8, labelpad=3)
    ax.set_title(f"d_L={dL:.2f}  smooth={sm:.6f}", fontsize=8, loc="right")
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel("бар")
plt.tight_layout()
out = _HERE / "figures" / "cascade_lp_compare.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nГрафик → {out}")
