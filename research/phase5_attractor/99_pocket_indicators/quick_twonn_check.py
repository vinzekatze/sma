"""
Быстрый чек: TwoNN на 256D delay embedding att.
Вопрос: какую размерность видит TwoNN, если вложить att в 256D?
"""
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

ROOT    = Path(__file__).resolve().parents[3]
DATADIR = ROOT / "data" / "candles"

LP_W = 600
LP_M, LP_D, LP_K, LP_N = 9, 3, 30, 3
P_EMB = 256   # размерность вложения для поиска
K_NN  = 50    # соседей для TwoNN
TICKER = "SBER"

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

def lp_filter(x, m=LP_M, d=LP_D, k=LP_K, n_iter=LP_N):
    s = x.copy().astype(np.float64); N = len(s)
    for _ in range(n_iter):
        M = N - m + 1
        X = np.lib.stride_tricks.sliding_window_view(s, m).copy()
        k_eff = min(k, M-1); d_eff = min(d, k_eff-1)
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
        s = res / np.maximum(cnt, 1)
    return np.diff(s)

def twonn(X_nn):
    k = len(X_nn)
    if k < 3: return np.nan
    D = np.sum((X_nn[:,None,:] - X_nn[None,:,:])**2, axis=2)
    np.fill_diagonal(D, np.inf)
    top2 = np.partition(D, 1, axis=1)[:, :2]
    r1 = np.sqrt(np.maximum(top2[:,0], 1e-30))
    r2 = np.sqrt(np.maximum(top2[:,1], 1e-30))
    mu = r2/r1; mu = mu[mu > 1.0]
    if len(mu) < 2: return np.nan
    return float((len(mu)-1) / np.sum(np.log(mu)))

# ── Загружаем данные ──────────────────────────────────────────────────────────
close = np.array([c["close"] for c in
    json.loads((DATADIR / TICKER / "1d.json").read_text())], dtype=np.float64)
lt = logtrend_causal(close)
ratio = close / np.maximum(lt, 1e-10)
N = len(ratio)

# Берём origin в середине последних 100 баров
t_orig = N - 60
att_local = lp_filter(ratio[t_orig - LP_W + 2 : t_orig + 2])  # 599 pts
n = len(att_local)  # 599

print(f"att_local длина: {n}")
print(f"P_EMB={P_EMB}, n - P_EMB = {n - P_EMB} точек в библиотеке")

# ── Библиотека в P_EMB=256 ────────────────────────────────────────────────────
n_lib = n - P_EMB  # 343
idx   = np.arange(n_lib)[:,None] + np.arange(P_EMB)[None,:]
X_lib = att_local[idx]     # (343, 256)
query = att_local[-P_EMB:]  # (256,)

# Поиск K_NN ближайших в 256D
dists  = np.linalg.norm(X_lib - query, axis=1)
nn_idx = np.argpartition(dists, K_NN-1)[:K_NN]
X_nn   = X_lib[nn_idx]     # (50, 256)

d_L_256 = twonn(X_nn)
print(f"\nTwoNN на {K_NN} соседях в P_EMB={P_EMB}D: d_L = {d_L_256:.2f}")

# ── Для сравнения: те же соседи, но только последние p=8 измерений ─────────────
for p in [8, 16, 32, 64, 128]:
    X_sub = X_nn[:, -p:]   # последние p координат (самые свежие)
    d_sub = twonn(X_sub)
    print(f"  d_L на срезе последних {p:3d} координат: {d_sub:.2f}")

# ── А теперь TwoNN в p=8 пространстве на независимой библиотеке ──────────────
n_lib8 = n - 8
X_lib8 = att_local[np.arange(n_lib8)[:,None] + np.arange(8)[None,:]]
q8 = att_local[-8:]
d8 = np.linalg.norm(X_lib8 - q8, axis=1)
nn8 = np.argpartition(d8, K_NN-1)[:K_NN]
X_nn8 = X_lib8[nn8]
d_L_8 = twonn(X_nn8)
print(f"\nTwoNN на {K_NN} соседях в независимой p=8 библиотеке: d_L = {d_L_8:.2f}")
