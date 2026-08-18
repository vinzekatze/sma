"""
Влияние m (embedding dim LP) на качество att и прогноза.
Фиксируем d=3, k=30, n=3. Меняем только m: 9, 15, 25, 40, 60.
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
LP_D    = 3
LP_K    = 30
LP_N    = 3
P_FIT   = 9
XI      = 35
LAMBDA  = 0.01
T_AHEAD = 15
N_SHOW  = 60
K_NN    = 50
T_EXCL  = 50

M_VALUES = [9, 15, 25, 40, 60]

def logtrend_causal(close):
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64); cn = np.arange(1, n+1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2)
    cy = np.cumsum(lc); cty = np.cumsum(t*lc)
    den = cn*ct2 - ct**2
    b = np.where(den > 0, (cn*cty - ct*cy)/den, 0.0)
    a = (cy - b*ct)/cn
    tr = np.exp(a + b*t); tr[:2] = close[:2]; return tr

def make_att(win, m, d=LP_D, k=LP_K, n_iter=LP_N):
    s = win.copy().astype(np.float64); N = len(s)
    for _ in range(n_iter):
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
        s = res / np.maximum(cnt, 1)
    return np.diff(s)

def twonn(sig, p_emb=P_FIT, k=K_NN, t_excl=T_EXCL):
    n = len(sig); n_lib = n - p_emb
    if n_lib < k + t_excl + 5: return np.nan
    idx = np.arange(n_lib)[:,None] + np.arange(p_emb)[None,:]
    X_lib = sig[idx]; query = sig[-p_emb:]
    cand = np.arange(n_lib - t_excl)
    dists = np.linalg.norm(X_lib[cand] - query, axis=1)
    nn = cand[np.argpartition(dists, min(k,len(cand))-1)[:min(k,len(cand))]]
    X_nn = X_lib[nn]
    D2 = np.sum((X_nn[:,None,:]-X_nn[None,:,:])**2, axis=2)
    np.fill_diagonal(D2, np.inf)
    top2 = np.partition(D2, 1, axis=1)[:, :2]
    r1 = np.sqrt(np.maximum(top2[:,0], 1e-30)); r2 = np.sqrt(np.maximum(top2[:,1], 1e-30))
    mu = r2/r1; mu = mu[mu > 1.0]
    if len(mu) < 2: return np.nan
    return float((len(mu)-1) / np.sum(np.log(mu)))

def _cosine_dist(A, b):
    nA = np.linalg.norm(A, axis=1); nb = float(np.linalg.norm(b))
    if nb < 1e-12: return np.ones(len(A))
    return 1.0 - np.clip((A @ b) / np.where(nA > 1e-12, nA, 1.0) / nb, -1.0, 1.0)

def forecast_lwr(att):
    n = len(att); n_lib = n - P_FIT
    if n_lib < XI + 5: return [att[-1]] * T_AHEAD
    idx = np.arange(n_lib)[:,None] + np.arange(P_FIT)[None,:]
    X_lib = att[idx]; y_lib = att[P_FIT:P_FIT+n_lib]
    acc_lib = np.zeros_like(X_lib)
    acc_lib[:,2:] = X_lib[:,2:] - 2*X_lib[:,1:-1] + X_lib[:,:-2]
    context = list(att[-P_FIT:])
    preds = []
    for _ in range(T_AHEAD):
        q = np.array(context[-P_FIT:])
        acc_q = np.zeros(P_FIT)
        if len(context) >= 3:
            c = np.array(context[-P_FIT:]); acc_q[2:] = c[2:]-2*c[1:-1]+c[:-2]
        d_pos = np.linalg.norm(X_lib-q, axis=1)
        top = np.argpartition(d_pos + LAMBDA*_cosine_dist(acc_lib, acc_q), XI-1)[:XI]
        X_nn = X_lib[top]; y_nn = y_lib[top]
        w = np.exp(-0.5*(d_pos[top]/max(d_pos[top].max(),1e-10))**2)
        sw = np.sqrt(np.maximum(w,1e-30))
        A = np.hstack([np.ones((len(top),1)), X_nn])
        c2, *_ = np.linalg.lstsq(sw[:,None]*A, sw*y_nn, rcond=None)
        pred = float(c2[0] + q @ c2[1:]); preds.append(pred); context.append(pred)
    return preds

# ── данные ────────────────────────────────────────────────────────────────────
close  = np.array([c["close"] for c in
    json.loads((DATADIR / TICKER / "1d.json").read_text())], dtype=np.float64)
lt = logtrend_causal(close); ratio = close / np.maximum(lt, 1e-10); N = len(ratio)

# ── таблица метрик ────────────────────────────────────────────────────────────
origins_test = [N-60, N-80, N-100, N-120, N-140]
print(f"{'m':>5}  {'d_L(9D)':>7}  {'smooth':>10}  {'rMAE(5 orig)':>13}")
print("─" * 42)

metrics = {}
for m in M_VALUES:
    t_orig = N - 60
    win = ratio[t_orig - LP_W + 2 : t_orig + 2]
    att = make_att(win, m)
    dL  = twonn(att)
    sm  = float(np.mean(np.abs(np.diff(att, 2))))

    # rMAE по 5 origins
    rmaes = []
    for t_orig in origins_test:
        win = ratio[t_orig - LP_W + 2 : t_orig + 2]
        att = make_att(win, m)
        preds = forecast_lwr(att)
        actuals = []
        for h in range(1, T_AHEAD+1):
            t_h = t_orig + h
            if t_h + 1 >= N: break
            w_h = ratio[t_h - LP_W + 2 : t_h + 2]
            actuals.append(make_att(w_h, m)[-1])  # att_actual с тем же m
        actuals = np.array(actuals[:len(preds)])
        preds_a = np.array(preds[:len(actuals)])
        naive = np.full(len(actuals), att[-1])
        rmaes.append(np.mean(np.abs(preds_a-actuals)) / (np.mean(np.abs(naive-actuals))+1e-10))

    mean_rMAE = np.mean(rmaes)
    metrics[m] = (dL, sm, mean_rMAE)
    print(f"{m:>5}  {dL:>7.2f}  {sm:>10.6f}  {mean_rMAE:>13.3f}")

# ── графики прогнозов (3 origins, все m) ──────────────────────────────────────
plot_origins = [N-60, N-80, N-100]
cmap = plt.cm.plasma
colors = {m: cmap(i/len(M_VALUES)) for i, m in enumerate(M_VALUES)}

fig, axes = plt.subplots(len(plot_origins), 1, figsize=(14, 4*len(plot_origins)))
fig.suptitle(f"Прогноз att при разных m LP  ({TICKER}, d={LP_D}, k={LP_K}, n={LP_N})", fontsize=12)

for ax, t_orig in zip(axes, plot_origins):
    # actuals (m=9 как reference)
    actuals = []
    for h in range(1, T_AHEAD+1):
        t_h = t_orig + h
        if t_h + 1 >= N: break
        w_h = ratio[t_h - LP_W + 2 : t_h + 2]
        actuals.append(make_att(w_h, 9)[-1])
    x_act = np.arange(1, len(actuals)+1)
    ax.plot(x_act, actuals, color="black", lw=1.5, marker="o", ms=3, zorder=10, label="actual (m=9)")

    for m in M_VALUES:
        win = ratio[t_orig - LP_W + 2 : t_orig + 2]
        att = make_att(win, m)
        hist = att[-N_SHOW:]
        preds = forecast_lwr(att)[:len(actuals)]
        x_hist = np.arange(-N_SHOW, 0); x_pred = np.arange(1, len(preds)+1)
        ax.plot(x_hist, hist, color=colors[m], lw=0.8, alpha=0.5)
        ax.plot(x_pred, preds, color=colors[m], lw=2, ls="--", label=f"m={m}")

    ax.axvline(0, color="gray", lw=0.8, ls=":"); ax.axhline(0, color="gray", lw=0.4)
    ax.set_title(f"origin t={t_orig}", fontsize=9)
    ax.legend(fontsize=7, ncol=len(M_VALUES)+1); ax.grid(True, alpha=0.2)

axes[-1].set_xlabel("бары от origin")
plt.tight_layout()
out = _HERE / "figures" / "lp_m_compare.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nГрафик → {out}")
