"""
Проверка: падает ли d_L при поэтапном LP, начиная с att_local?
att_local = np.diff(LP(ratio, m=9,d=3,k=30,n=3)) — уже один LP-проход.
Дальше поэтапно добавляем LP с адаптивным d.
"""
import json, warnings
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[3]
DATADIR = ROOT / "data" / "candles"
LP_W    = 600
TICKER  = "SBER"
K_NN    = 50     # соседей для TwoNN
T_EXCL  = 50    # temporal exclusion (ближайшие по времени к query)
LP_M0, LP_D0, LP_K0, LP_N0 = 9, 3, 30, 3   # стандартные параметры LP

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

def lp_signal(x, m, d, k, n_iter=1):
    """LP: возвращает сигнал (не diff) той же длины."""
    s = x.copy().astype(np.float64); N = len(s)
    for _ in range(n_iter):
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
        s = res / np.maximum(cnt, 1)
    return s

def lp_att(x, m, d, k, n_iter=1):
    """LP + diff → att."""
    return np.diff(lp_signal(x, m, d, k, n_iter))

def twonn_local(sig, p_emb, k=K_NN, t_excl=T_EXCL):
    n = len(sig); n_lib = n - p_emb
    if n_lib < k + t_excl + 5:
        return np.nan, 0
    idx   = np.arange(n_lib)[:,None] + np.arange(p_emb)[None,:]
    X_lib = sig[idx]; query = sig[-p_emb:]
    cand  = np.arange(n_lib - t_excl)   # исключаем последние t_excl
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

# ── Загрузка и att_local (стандартный 1 LP-проход) ───────────────────────────
close = np.array([c["close"] for c in
    json.loads((DATADIR / TICKER / "1d.json").read_text())], dtype=np.float64)
lt = logtrend_causal(close)
ratio = close / np.maximum(lt, 1e-10)
N = len(ratio)
t_orig = N - 60
win = ratio[t_orig - LP_W + 2 : t_orig + 2]   # 600 баров

# Стандартный LP → att_local (599 значений)
att_local = lp_att(win, LP_M0, LP_D0, LP_K0, LP_N0)
print(f"Тикер: {TICKER},  att_local: {len(att_local)} pt  (1 LP-проход m={LP_M0},d={LP_D0},k={LP_K0},n={LP_N0})\n")

# ── Поэтапный каскад ─────────────────────────────────────────────────────────
print(f"{'─'*68}")
print(f"{'Шаг':>5}  {'p_emb':>6}  {'n_cand':>7}  {'d_L':>6}  {'→ LP params (для след. шага)':>30}")
print(f"{'─'*68}")

levels_p = [256, 128, 64, 32, 16, 9]
sig = att_local.copy()

for lev_i, p_emb in enumerate(levels_p):
    d_L, n_cand = twonn_local(sig, p_emb)

    if np.isnan(d_L):
        print(f"{lev_i+1:>5}  {p_emb:>6}  {'n/a':>7}  {'n/a':>6}")
        break

    d_int = max(2, round(d_L))
    m_lp  = max(2*d_int + 1, 9)         # минимум Такенса, не меньше 9
    m_lp  = min(m_lp, p_emb)            # не больше текущего p_emb
    k_lp  = max(15, 5 * d_int)

    is_last = (lev_i == len(levels_p) - 1)
    lp_str  = f"(финал: LWR с p_fit={d_int+1})" if is_last else f"LP(m={m_lp}, d={d_int}, k={k_lp})"
    print(f"{lev_i+1:>5}  {p_emb:>6}  {n_cand:>7}  {d_L:>6.2f}  {lp_str}")

    if not is_last:
        # Применяем один LP-проход к сигналу (возвращает diff!)
        next_p = levels_p[lev_i + 1]
        sig_lp = lp_signal(sig, m=m_lp, d=d_int, k=k_lp, n_iter=1)
        sig    = np.diff(sig_lp)   # diff → att следующего уровня
        # Убеждаемся, что хватает длины для следующего p_emb
        if len(sig) < next_p + K_NN + T_EXCL + 5:
            print(f"  [!] сигнал слишком короткий для p={next_p}: {len(sig)} pt")
            break

print(f"{'─'*68}")
