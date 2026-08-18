import json, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[3]
DATADIR = ROOT / "data" / "candles"
TICKER  = "SBER"
K_NN    = 50
T_EXCL  = 50   # temporal exclusion

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

def twonn_local(sig, p_emb, k=K_NN, t_excl=T_EXCL):
    n = len(sig); n_lib = n - p_emb
    if n_lib < k + t_excl + 5:
        return np.nan, 0
    idx   = np.arange(n_lib)[:,None] + np.arange(p_emb)[None,:]
    X_lib = sig[idx]; query = sig[-p_emb:]
    cand  = np.arange(n_lib - t_excl)
    if len(cand) < k + 2:
        return np.nan, 0
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
    if len(mu) < 2:
        return np.nan, 0
    return float((len(mu)-1) / np.sum(np.log(mu))), len(cand)

# ── данные ────────────────────────────────────────────────────────────────────
close = np.array([c["close"] for c in
    json.loads((DATADIR / TICKER / "1d.json").read_text())], dtype=np.float64)
lt    = logtrend_causal(close)
ratio = close / np.maximum(lt, 1e-10)
dratio = np.diff(ratio)   # стационарный ряд

# берём последние 600 баров dratio как окно (аналог LP_W)
win = dratio[-600:]
print(f"Тикер: {TICKER},  dratio окно: {len(win)} pt,  K_NN={K_NN},  T_EXCL={T_EXCL}\n")

print(f"{'p_emb':>6}  {'n_cand':>7}  {'d_L':>6}")
print(f"{'─'*25}")
for p_emb in [256, 128, 64, 32, 16, 9]:
    d_L, n_cand = twonn_local(win, p_emb)
    print(f"{p_emb:>6}  {n_cand:>7}  {d_L:>6.2f}" if not np.isnan(d_L) else f"{p_emb:>6}  {n_cand:>7}  {'n/a':>6}")
