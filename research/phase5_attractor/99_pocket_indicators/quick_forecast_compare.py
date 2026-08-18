"""
Визуальное сравнение прогнозов LWR для двух LP-схем:
  - standard: LP(d=3, n=3)
  - cascade:  LP(15→8→4→3)
"""
import json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[3]
_HERE   = Path(__file__).resolve().parent
DATADIR = ROOT / "data" / "candles"
TICKER  = "SBER"
LP_W    = 600
P_FIT   = 9
XI      = 35
LAMBDA  = 0.01
T_AHEAD = 15    # чуть длиннее для наглядности
N_SHOW  = 60    # баров истории на графике

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
    s = x.copy().astype(np.float64); N = len(s)
    M = N - m + 1
    X = np.lib.stride_tricks.sliding_window_view(s, m).copy()
    k_eff = min(k, M-1); d_eff = min(d, k_eff-1)
    D = cdist(X, X); np.fill_diagonal(D, np.inf)
    nn_idx = np.argpartition(D, k_eff, axis=1)[:, :k_eff]; del D
    nbrs = X[nn_idx]; centers = nbrs.mean(axis=1, keepdims=True)
    C = np.einsum("bki,bkj->bij", nbrs-centers, nbrs-centers)
    _, vecs = np.linalg.eigh(C); Vd = vecs[:, :, -d_eff:]
    xc = X - centers[:,0,:]
    Xp = centers[:,0,:] + np.einsum("bd,bmd->bm", np.einsum("bm,bmd->bd", xc, Vd), Vd)
    res = np.zeros(N); cnt = np.zeros(N, int)
    idx2d = np.arange(M)[:,None] + np.arange(m)[None,:]
    np.add.at(res, idx2d, Xp); np.add.at(cnt, idx2d, 1)
    return res / np.maximum(cnt, 1)

def make_att(win, steps):
    s = win.copy()
    for d, m, k in steps:
        s = lp_pass(s, m=m, d=d, k=k)
    return np.diff(s)

def p(d): return (d, max(9, 2*d+3), max(30, 5*d))

SCHEMES = {
    "standard (d=3, n=3)":  [p(3)]*3,
    "cascade 15→8→4→3":     [p(15), p(8), p(4), p(3)],
}

def _cosine_dist(A, b):
    nA = np.linalg.norm(A, axis=1); nb = float(np.linalg.norm(b))
    if nb < 1e-12: return np.ones(len(A))
    return 1.0 - np.clip((A @ b) / np.where(nA > 1e-12, nA, 1.0) / nb, -1.0, 1.0)

def forecast_lwr(att, p_fit=P_FIT, xi=XI, t_ahead=T_AHEAD):
    n = len(att); n_lib = n - p_fit
    if n_lib < xi + 5: return [att[-1]] * t_ahead
    idx   = np.arange(n_lib)[:,None] + np.arange(p_fit)[None,:]
    X_lib = att[idx]; y_lib = att[p_fit:p_fit + n_lib]
    acc_lib = np.zeros_like(X_lib)
    acc_lib[:, 2:] = X_lib[:, 2:] - 2*X_lib[:, 1:-1] + X_lib[:, :-2]

    context = list(att[-p_fit:])
    preds = []
    for _ in range(t_ahead):
        q = np.array(context[-p_fit:])
        acc_q = np.zeros(p_fit)
        if len(context) >= 3:
            c = np.array(context[-p_fit:])
            acc_q[2:] = c[2:] - 2*c[1:-1] + c[:-2]
        d_pos = np.linalg.norm(X_lib - q, axis=1)
        d_acc = _cosine_dist(acc_lib, acc_q)
        top   = np.argpartition(d_pos + LAMBDA*d_acc, xi-1)[:xi]
        X_nn  = X_lib[top]; y_nn = y_lib[top]
        w  = np.exp(-0.5 * (d_pos[top] / max(d_pos[top].max(), 1e-10))**2)
        sw = np.sqrt(np.maximum(w, 1e-30))
        A  = np.hstack([np.ones((xi,1)), X_nn])
        c, *_ = np.linalg.lstsq(sw[:,None]*A, sw*y_nn, rcond=None)
        pred = float(c[0] + q @ c[1:])
        preds.append(pred); context.append(pred)
    return preds

# ── данные ────────────────────────────────────────────────────────────────────
close  = np.array([c["close"] for c in
    json.loads((DATADIR / TICKER / "1d.json").read_text())], dtype=np.float64)
lt     = logtrend_causal(close)
ratio  = close / np.maximum(lt, 1e-10)
N      = len(ratio)

origins = [N - 60, N - 80, N - 100]   # 3 origin-а

fig, axes = plt.subplots(len(origins), 1, figsize=(14, 4*len(origins)))
fig.suptitle(f"Прогноз att: standard LP vs cascade  ({TICKER})", fontsize=13)

colors = {"standard (d=3, n=3)": "steelblue", "cascade 15→8→4→3": "darkorange"}

for ax, t_orig in zip(axes, origins):
    win = ratio[t_orig - LP_W + 2 : t_orig + 2]

    # att_actual из стандартного LP для каждого будущего бара
    actuals = []
    for h in range(1, T_AHEAD + 1):
        t_h = t_orig + h
        if t_h + 1 >= N: break
        w_h = ratio[t_h - LP_W + 2 : t_h + 2]
        att_h = make_att(w_h, [p(3)]*3)
        actuals.append(att_h[-1])

    for name, steps in SCHEMES.items():
        att = make_att(win, steps)
        hist = att[-N_SHOW:]
        preds = forecast_lwr(att)[:len(actuals)]

        x_hist = np.arange(-N_SHOW, 0)
        x_pred = np.arange(1, len(preds)+1)

        ax.plot(x_hist, hist, color=colors[name], lw=0.9, alpha=0.7, label=f"{name} (hist)")
        ax.plot(x_pred, preds, color=colors[name], lw=2.0, ls="--", label=f"{name} (forecast)")

    # actuals
    x_act = np.arange(1, len(actuals)+1)
    ax.plot(x_act, actuals, color="black", lw=1.5, marker="o", ms=3, label="att_actual")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.set_title(f"origin t={t_orig}", fontsize=9)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel("бары от origin")
plt.tight_layout()
out = _HERE / "figures" / "forecast_compare.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"График → {out}")
