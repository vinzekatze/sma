"""
94_local_dim_aligned.py — Повтор скр.88 с p-aligned каскадом.

Проблема скр.88: P_REG_GRID=[4,6,8,10,12,14,16] + стандартный каскад [64→32→16→p].
Уровень 16 не кратен p=6,10,12,14 → нарушение нестинга → oracle_p смещён
в пользу p=4,8,16 (теорема Такенса предсказывает нечётные значения p).

Исправление:
  • P_REG_GRID = list(range(4, 17))  — 13 значений, включая нечётные
  • Aligned каскад:  p_fit × {1,2,4,...} ≤ P_MAX (нестинг гарантирован)
    p=9 → [36,18,9],  p=5 → [40,20,10,5],  p=8 → [64,32,16,8]

Гипотеза: r≈0 из скр.88 был артефактом смещённого oracle_p, а не
свойством аттрактора. С aligned каскадом oracle_p честнее отражает
оптимальную размерность и корреляция с d_local должна быть выше.

Четыре оценщика d_local: Levina-Bickel, TwoNN, local_corr, local_PCA.
Маппинг: p ≥ 2·d_local+1 (теорема Такенса), snap к ближайшему в [4..16].

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров 1d.
"""
from __future__ import annotations
import time
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "candles"
FIG_DIR  = ROOT / "research" / "figures"

TICKERS   = ["SBER", "MRKP", "LKOH", "NVTK", "CHMF", "NLMK", "MGNT", "VTBR"]
INTERVAL  = "1d"
N_ORIGINS = 40
STEP      = 5

P_MAX    = 64
LAMBDA   = 0.01
XI_FIXED = 51

LP_M = 9; LP_D = 3; LP_K = 30; LP_N = 3

P_REG_GRID = list(range(4, 17))   # [4,5,6,...,16], 13 значений
P_FIT_BASE = 16

K_LB    = 20
K_PCA   = 51
EXP_VAR = 0.90
N_RADII = 10

DIM_NAMES = ["d_lb", "d_twonn", "d_corr", "d_pca"]


# ── вспомогательные функции ───────────────────────────────────────────────────

def _logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n+1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2)
    cy = np.cumsum(lc); cty = np.cumsum(t*lc)
    denom = cn*ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn*cty - ct*cy)/denom, 0.0)
    a = (cy - b*ct)/cn
    trend = np.exp(a + b*t); trend[:2] = close[:2]
    return trend


def _lp_proj(ratio: np.ndarray, m: int, d: int, k: int, n_iter: int) -> np.ndarray:
    from scipy.spatial import KDTree
    s = ratio.copy().astype(np.float64)
    N = len(s); k_eff = min(k, N - m); d_eff = min(d, m - 1)
    for _ in range(n_iter):
        n_pts = N - m + 1
        if n_pts < k_eff + 1:
            break
        rows = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
        X = s[rows]; tree = KDTree(X)
        _, inds = tree.query(X, k=k_eff + 1)
        X_proj = np.empty_like(X)
        for i in range(n_pts):
            nn = inds[i, 1:]; X_nn = X[nn]
            centroid = X_nn.mean(axis=0)
            _, _, Vt = np.linalg.svd(X_nn - centroid, full_matrices=False)
            V_d = Vt[:d_eff].T; xc = X[i] - centroid
            X_proj[i] = centroid + V_d @ (V_d.T @ xc)
        result = np.zeros(N); count = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i+m] += X_proj[i]; count[i:i+m] += 1
        s = result / np.maximum(count, 1)
    return np.diff(s)


def _cosine_dist(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_A = np.linalg.norm(A, axis=1)
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-12:
        return np.ones(len(A))
    cos = np.where(norm_A > 1e-12, (A @ b) / (norm_A * norm_b), 0.0)
    return 1.0 - np.clip(cos, -1.0, 1.0)


def _lwr_predict(X_nn: np.ndarray, y_nn: np.ndarray, vec_f: np.ndarray) -> float:
    dists = np.linalg.norm(X_nn - vec_f, axis=1)
    h_bw  = max(float(dists.max()), 1e-10)
    w     = np.exp(-0.5 * (dists / h_bw) ** 2)
    A     = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw    = np.sqrt(np.maximum(w, 1e-30))
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vec_f @ c[1:])


def _lwr_loo_error(X_nn: np.ndarray, y_nn: np.ndarray, vec_f: np.ndarray) -> float:
    dists = np.linalg.norm(X_nn - vec_f, axis=1)
    h_bw  = max(float(dists.max()), 1e-10)
    w     = np.maximum(np.exp(-0.5 * (dists / h_bw) ** 2), 1e-30)
    A     = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw    = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    y_hat = A @ c; resid = y_nn - y_hat
    try:
        Aw       = sw[:, None] * A
        AtWA_inv = np.linalg.inv(Aw.T @ Aw + 1e-10 * np.eye(A.shape[1]))
        h_diag   = w * np.einsum("ij,ij->i", A @ AtWA_inv, A)
        denom    = 1.0 - h_diag
        e_loo    = np.where(np.abs(denom) > 1e-6, resid / denom, resid)
    except np.linalg.LinAlgError:
        e_loo = resid
    return float(np.mean(np.abs(e_loo)))


# ── генераторы уровней каскада ────────────────────────────────────────────────

def levels_aligned(p_fit: int, p_max: int = P_MAX) -> list[int]:
    """P-aligned каскад ×2: p_fit × {1,2,4,...} ≤ p_max (скр.93)."""
    levs = [p_fit]; p = p_fit
    while p * 2 <= p_max:
        p *= 2; levs.append(p)
    return list(reversed(levs))


def levels_two(p_fit: int, p_max: int = P_MAX) -> list[int]:
    """Двухуровневый: [p_max, p_fit] (для сравнения)."""
    return [p_fit] if p_max <= p_fit else [p_max, p_fit]


# ── контекст ──────────────────────────────────────────────────────────────────

class _Context:
    __slots__ = ("X_full", "y_base", "acc_hist", "vec_full", "n", "ok")

    def __init__(self, att: np.ndarray) -> None:
        n = len(att); self.n = n; self.ok = False
        if n - P_MAX - 1 < 3:
            return
        t_arr       = np.arange(P_MAX, n - 1)
        self.X_full = np.column_stack([att[t_arr - (P_MAX - 1 - j)] for j in range(P_MAX)])
        self.y_base = att[t_arr + 1]
        acc         = np.zeros(n)
        if n >= 3:
            acc[2:] = att[2:] - 2 * att[1:-1] + att[:-2]
        self.acc_hist = acc
        self.vec_full = att[-P_MAX:].copy()
        self.ok       = True


# ── ядро прогноза ─────────────────────────────────────────────────────────────

def _forecast_with_levels(ctx: _Context, p_reg: int,
                          cascade_levels: list[int]) -> tuple[float, float]:
    """Прогноз + LOO-ошибка с произвольной цепочкой уровней каскада."""
    X_full   = ctx.X_full; y_base = ctx.y_base
    vec_full = ctx.vec_full; acc_hist = ctx.acc_hist; n = ctx.n

    if len(X_full) < XI_FIXED + 1:
        return np.nan, np.nan

    t_arr   = np.arange(P_MAX, n - 1)
    X_acc   = np.column_stack([acc_hist[t_arr - (p_reg - 1 - j)] for j in range(p_reg)])
    vec_acc = acc_hist[-p_reg:].copy()

    cands = np.arange(len(X_full))
    for k, p_lvl in enumerate(cascade_levels):
        is_last = (k == len(cascade_levels) - 1)
        if not is_last:
            xi_lvl = min(XI_FIXED, len(cands))
            if len(cands) > xi_lvl:
                d = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full[-p_lvl:], axis=1)
                cands = cands[np.argpartition(d, xi_lvl - 1)[:xi_lvl]]
            p_next = cascade_levels[k + 1]
            radius = p_lvl - p_next
            if radius > 0:
                exp   = cands[:, None] - np.arange(radius + 1)[None, :]
                cands = np.unique(np.clip(exp, 0, len(X_full) - 1))

    if len(cands) > XI_FIXED:
        d_pos  = np.linalg.norm(X_full[cands, -p_reg:] - vec_full[-p_reg:], axis=1)
        d_acc  = _cosine_dist(X_acc[cands], vec_acc)
        cands  = cands[np.argpartition(d_pos + LAMBDA * d_acc, XI_FIXED - 1)[:XI_FIXED]]

    if len(cands) < p_reg + 2:
        return np.nan, np.nan

    X_nn = X_full[cands, -p_reg:]; y_nn = y_base[cands]
    query = vec_full[-p_reg:]
    return _lwr_predict(X_nn, y_nn, query), _lwr_loo_error(X_nn, y_nn, query)


def _forecast_fixed(ctx: _Context) -> float:
    pred, _ = _forecast_with_levels(ctx, P_FIT_BASE, levels_aligned(P_FIT_BASE))
    return pred


# ── оценщики локальной размерности ───────────────────────────────────────────

def dim_levina_bickel(sorted_dists: np.ndarray, k: int = K_LB) -> float:
    if len(sorted_dists) < k:
        return np.nan
    r_k = sorted_dists[k - 1]
    if r_k < 1e-12:
        return np.nan
    s = float(np.log(r_k / np.maximum(sorted_dists[:k - 1], 1e-12)).sum())
    return float((k - 2) / s) if s > 1e-10 else np.nan


def dim_twonn(X_lib: np.ndarray, n_sample: int = 200) -> float:
    n = len(X_lib)
    if n < 3:
        return np.nan
    idx = np.random.choice(n, size=min(n_sample, n), replace=False)
    mu_list: list[float] = []
    for i in idx:
        dists_i    = np.linalg.norm(X_lib - X_lib[i], axis=1)
        dists_i[i] = np.inf
        sd = np.sort(dists_i)
        r1, r2 = sd[0], sd[1]
        if r1 > 1e-12 and r2 > r1:
            mu_list.append(np.log(r2 / r1))
    if not mu_list:
        return np.nan
    mu = float(np.mean(mu_list))
    return float(np.log(2) / mu) if mu > 1e-10 else np.nan


def dim_local_corr(sorted_dists: np.ndarray) -> float:
    if len(sorted_dists) < 10:
        return np.nan
    eps_lo = float(np.percentile(sorted_dists, 5))
    eps_hi = float(np.percentile(sorted_dists, 60))
    if eps_lo < 1e-12 or eps_hi <= eps_lo:
        return np.nan
    radii  = np.geomspace(eps_lo, eps_hi, N_RADII)
    counts = np.searchsorted(sorted_dists, radii).astype(float)
    valid  = counts > 1
    if valid.sum() < 3:
        return np.nan
    return max(float(np.polyfit(np.log(radii[valid]), np.log(counts[valid]), 1)[0]), 0.0)


def dim_local_pca(X_nn: np.ndarray) -> float:
    if len(X_nn) < 3:
        return np.nan
    Xc = X_nn - X_nn.mean(axis=0)
    try:
        s = np.linalg.svd(Xc, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.nan
    var = s ** 2; tot = var.sum()
    if tot < 1e-12:
        return np.nan
    cumvar = np.cumsum(var) / tot
    return float(min(int(np.searchsorted(cumvar, EXP_VAR)) + 1, X_nn.shape[1]))


def compute_local_dims(ctx: _Context) -> dict[str, float]:
    """Все 4 оценки в точке query (полное P_MAX-мерное пространство)."""
    all_dists = np.linalg.norm(ctx.X_full - ctx.vec_full, axis=1)
    sorted_d  = np.sort(all_dists)
    nn_idx    = np.argpartition(all_dists, min(K_PCA, len(all_dists) - 1))[:K_PCA]
    return {
        "d_lb":    dim_levina_bickel(sorted_d),
        "d_twonn": dim_twonn(ctx.X_full),
        "d_corr":  dim_local_corr(sorted_d),
        "d_pca":   dim_local_pca(ctx.X_full[nn_idx]),
    }


def takens_to_p(d: float) -> int:
    """p ≥ 2d+1, snap к ближайшему в P_REG_GRID."""
    if np.isnan(d) or d <= 0:
        return P_FIT_BASE
    return min(P_REG_GRID, key=lambda p: abs(p - (2.0 * d + 1.0)))


# ── walk-forward ──────────────────────────────────────────────────────────────

def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend_causal(close)
    return _lp_proj(close / np.maximum(lt, 1e-10), LP_M, LP_D, LP_K, LP_N)


METHOD_KEYS = [
    "fixed_16",                         # baseline: acc_ang p=16 aligned cascade
    "oracle", "ensemble",               # верхний потолок и лучший реализуемый метод
    "aln_lb", "aln_twonn",             # aligned каскад до p*(Levina-Bickel / TwoNN)
    "aln_corr", "aln_pca",             # aligned каскад до p*(corr_dim / PCA)
    "two_corr",                         # двухуровневый P_MAX→p*(corr) для сравнения
]


def wf_ticker(att_full: np.ndarray) -> tuple[dict, list[dict]]:
    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP, n_total - 1, STEP))
    out  = {k: {"errors": [], "signs": [], "chosen_p": []} for k in METHOD_KEYS}
    rows: list[dict] = []

    for t_orig in origins:
        if t_orig + 1 >= n_total:
            continue
        hist = att_full[:t_orig + 1]
        true = att_full[t_orig + 1]

        ctx = _Context(hist)
        if not ctx.ok:
            continue

        # baseline fixed_16 с aligned каскадом
        pred_base = _forecast_fixed(ctx)
        if not np.isnan(pred_base):
            out["fixed_16"]["errors"].append(pred_base - true)
            out["fixed_16"]["signs"].append(int(np.sign(pred_base) == np.sign(true)))

        # aligned каскад для всех p_reg ∈ [4..16]
        preds:    dict[int, float] = {}
        loo_errs: dict[int, float] = {}
        for p_reg in P_REG_GRID:
            pred, loo = _forecast_with_levels(ctx, p_reg, levels_aligned(p_reg))
            preds[p_reg] = pred; loo_errs[p_reg] = loo

        valid_p = [p for p in P_REG_GRID
                   if not np.isnan(preds[p]) and not np.isnan(loo_errs[p])]

        if valid_p:
            # oracle (aligned)
            best_ora = min(valid_p, key=lambda p: abs(preds[p] - true))
            out["oracle"]["errors"].append(preds[best_ora] - true)
            out["oracle"]["signs"].append(int(np.sign(preds[best_ora]) == np.sign(true)))
            out["oracle"]["chosen_p"].append(best_ora)

            # ensemble 1/LOO (aligned)
            inv = np.array([1.0 / (loo_errs[p] + 1e-12) for p in valid_p])
            w   = inv / inv.sum()
            pe  = float(w @ np.array([preds[p] for p in valid_p]))
            out["ensemble"]["errors"].append(pe - true)
            out["ensemble"]["signs"].append(int(np.sign(pe) == np.sign(true)))

        # локальная размерность
        dims = compute_local_dims(ctx)

        def _record(key: str, d_val: float, lvl_fn) -> None:
            p_sel = takens_to_p(d_val)
            pred, _ = _forecast_with_levels(ctx, p_sel, lvl_fn(p_sel))
            if not np.isnan(pred):
                out[key]["errors"].append(pred - true)
                out[key]["signs"].append(int(np.sign(pred) == np.sign(true)))
                out[key]["chosen_p"].append(p_sel)

        _record("aln_lb",    dims["d_lb"],    levels_aligned)
        _record("aln_twonn", dims["d_twonn"], levels_aligned)
        _record("aln_corr",  dims["d_corr"],  levels_aligned)
        _record("aln_pca",   dims["d_pca"],   levels_aligned)
        _record("two_corr",  dims["d_corr"],  levels_two)

        if valid_p:
            best_ora = min(valid_p, key=lambda p: abs(preds[p] - true))
            rows.append({**dims, "oracle_p": float(best_ora)})

    return out, rows


# ── основной блок ─────────────────────────────────────────────────────────────

print("Загрузка + Local Projective (кэшируется)...")
t0 = time.time()
att_data: dict[str, np.ndarray] = {}
for tkr in TICKERS:
    att_data[tkr] = load_att(tkr)
    print(f"  {tkr}...")
std_all  = [float(np.std(att_data[tkr])) for tkr in TICKERS]
std_mean = float(np.mean(std_all))
print(f"Загрузка: {time.time()-t0:.1f}с")

print("\nWalk-forward (aligned каскад, p∈[4..16], 8 методов)...")
t0 = time.time()

all_res: dict[str, dict] = {k: {"errors": [], "signs": [], "chosen_p": []}
                            for k in METHOD_KEYS}
all_rows: list[dict] = []

for tkr in TICKERS:
    t1  = time.time()
    res, rows = wf_ticker(att_data[tkr])
    for key in METHOD_KEYS:
        for m in ("errors", "signs", "chosen_p"):
            all_res[key][m].extend(res[key][m])
    all_rows.extend(rows)
    print(f"  {tkr}: {time.time()-t1:.1f}с")
print(f"Walk-forward: {time.time()-t0:.1f}с  |  строк: {len(all_rows)}")


def rmae(key: str) -> float:
    e = all_res[key]["errors"]
    return float(np.mean(np.abs(e))) / std_mean if e else np.nan

def sacc(key: str) -> float:
    s = all_res[key]["signs"]
    return float(np.mean(s)) * 100.0 if s else np.nan

baseline = rmae("fixed_16")

# ── Фаза A: корреляции d_local vs oracle_p (aligned) ─────────────────────────

oracle_p_arr = np.array([r["oracle_p"] for r in all_rows])

print("\n── Корреляции d_local с oracle_p (aligned каскад) ──")
print(f"{'Оценщик':12s}  {'Pearson r':>10}  {'Spearman ρ':>10}  "
      f"{'p-value':>8}  {'range':>14}  {'median':>8}")
pearson_r_vals = []; spearman_r_vals = []
for dname in DIM_NAMES:
    vals = np.array([r[dname] for r in all_rows])
    mask = np.isfinite(vals) & np.isfinite(oracle_p_arr)
    if mask.sum() < 10:
        pearson_r_vals.append(0.0); spearman_r_vals.append(0.0)
        print(f"  {dname:12s}: нет данных")
        continue
    pr, pp = sp_stats.pearsonr(vals[mask], oracle_p_arr[mask])
    sr, _  = sp_stats.spearmanr(vals[mask], oracle_p_arr[mask])
    pearson_r_vals.append(pr); spearman_r_vals.append(sr)
    sig = "***" if pp < 0.001 else ("**" if pp < 0.01 else ("*" if pp < 0.05 else "ns"))
    print(f"  {dname:12s}: r={pr:+.3f}  ρ={sr:+.3f}  "
          f"p={pp:.4f}({sig})  "
          f"[{vals[mask].min():.1f},{vals[mask].max():.1f}]  "
          f"med={np.nanmedian(vals[mask]):.2f}")

# Сравнение скр.88 (biased): oracle_p на чётной сетке vs aligned
print(f"\nDistrib oracle_p aligned (все {len(all_rows)} origins):")
cnt_ora = Counter(int(p) for p in oracle_p_arr)
for p in P_REG_GRID:
    bar = "█" * round(cnt_ora.get(p, 0) / max(len(all_rows), 1) * 200)
    print(f"  p={p:2d}: {cnt_ora.get(p, 0):4d} ({cnt_ora.get(p,0)*100//max(len(all_rows),1):2d}%)  {bar}")

print("\nТакенс-маппинг: % совпадения p_takens с oracle_p:")
for dname in DIM_NAMES:
    vals = np.array([r[dname] for r in all_rows])
    mask = np.isfinite(vals) & np.isfinite(oracle_p_arr)
    if mask.sum() < 5:
        continue
    pt = np.array([takens_to_p(v) for v in vals[mask]])
    pct = float(np.mean(pt == oracle_p_arr[mask].astype(int))) * 100
    print(f"  {dname:12s}: {pct:.1f}%  (случайный ≈ {100/len(P_REG_GRID):.0f}%)")

# ── Фаза B: итоговая таблица walk-forward ─────────────────────────────────────

print("\n── Walk-forward: итоговая таблица ──")
print(f"{'Метод':16s}  {'rMAE':>7}  {'Δ vs baseline':>14}  "
      f"{'Δ vs 88_ensemble':>18}  {'SignAcc':>8}")
REF_88_ENSEMBLE = 0.0501   # −10.01% из скр.86 (стандартный каскад, чётная сетка)
for key in METHOD_KEYS:
    r  = rmae(key)
    d  = (r / baseline - 1) * 100 if not np.isnan(r) else np.nan
    d88 = (r / REF_88_ENSEMBLE - 1) * 100 if not np.isnan(r) else np.nan
    sa = sacc(key)
    print(f"  {key:16s}: {r:.4f}  ({d:+6.2f}%)  ({d88:+6.2f}% vs 88)  {sa:.1f}%")

print("\n── p-распределение oracle_p и dim-методов ──")
print(f"{'':12s}  " + "  ".join(f"p={p}" for p in P_REG_GRID))
for key in ["oracle", "aln_lb", "aln_corr", "aln_pca"]:
    cp = all_res[key]["chosen_p"]
    if not cp:
        continue
    cnt = Counter(cp)
    row = "  ".join(f"{cnt.get(p,0)*100//len(cp):4d}%" for p in P_REG_GRID)
    print(f"  {key:12s}: {row}")


# ── Фигуры ────────────────────────────────────────────────────────────────────

# Рис. A — Корреляции d_local с oracle_p: aligned vs скр.88 (old)
OLD_PEARSON = {"d_lb": 0.02, "d_twonn": -0.01, "d_corr": 0.03, "d_pca": 0.01}

x = np.arange(len(DIM_NAMES))
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.35
ax.bar(x - w/2, [OLD_PEARSON[d] for d in DIM_NAMES], w,
       color="#546e7a", label="скр.88 (стандартный каскад, чётная сетка)")
ax.bar(x + w/2, pearson_r_vals, w,
       color=["#42a5f5" if r >= 0 else "#ef5350" for r in pearson_r_vals],
       label="скр.94 (aligned каскад, p∈[4..16])")
ax.bar(x + w/2, spearman_r_vals, w, color="none",
       edgecolor="#ffa726", linewidth=1.5, linestyle="--", label="Spearman ρ (скр.94)")
ax.axhline(0, color="white", lw=0.8)
ax.axhline( 0.15, color="#66bb6a", ls=":", lw=1.2, label="±0.15")
ax.axhline(-0.15, color="#66bb6a", ls=":", lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(DIM_NAMES)
ax.set_ylabel("Корреляция с oracle_p")
ax.set_title(f"Рис.A  d_local vs oracle_p: aligned vs старый каскад  (N={len(all_rows)})")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIG_DIR / "94_A.png", dpi=120); plt.close()
print("\nРис. A сохранён")

# Рис. B — Scatter d_local vs oracle_p (aligned)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, dname, pr, sr in zip(axes, DIM_NAMES, pearson_r_vals, spearman_r_vals):
    vals = np.array([r[dname] for r in all_rows])
    mask = np.isfinite(vals) & np.isfinite(oracle_p_arr)
    ax.scatter(vals[mask], oracle_p_arr[mask], alpha=0.18, s=12, color="#42a5f5")
    if mask.sum() > 4:
        m_, b_ = np.polyfit(vals[mask], oracle_p_arr[mask], 1)
        xr = np.linspace(vals[mask].min(), vals[mask].max(), 80)
        ax.plot(xr, m_*xr+b_, color="#ef5350", lw=1.5)
    ax.set_xlabel(dname, fontsize=9); ax.set_ylabel("oracle_p (aligned)", fontsize=9)
    ax.set_title(f"r={pr:+.3f}  ρ={sr:+.3f}", fontsize=9)
plt.suptitle("Рис.B  d_local vs oracle_p aligned (scatter)", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "94_B.png", dpi=120); plt.close()
print("Рис. B сохранён")

# Рис. C — Walk-forward: Δ% по методам
SHOW = ["fixed_16", "oracle", "ensemble",
        "aln_lb", "aln_twonn", "aln_corr", "aln_pca", "two_corr"]
LBLS = ["fixed\np=16", "oracle\naln", "1/LOO\naln",
        "aln\nLB", "aln\nTwoNN", "aln\ncorr", "aln\nPCA", "2-lvl\ncorr"]
rmae_f  = [rmae(k) for k in SHOW]
delta_f = [(r / baseline - 1) * 100 if not np.isnan(r) else 0.0 for r in rmae_f]
c_list  = ["#78909c"] + ["#66bb6a" if d < 0 else "#ef5350" for d in delta_f[1:]]

fig, ax = plt.subplots(figsize=(13, 4))
bars = ax.bar(LBLS, delta_f, color=c_list, width=0.55)
ax.axhline(0, color="white", lw=0.8)
# Горизонталь ensemble из скр.93 (-12.41%) для сравнения
ax.axhline(-12.41, color="#ffa726", ls="--", lw=1.2, label="скр.93 ensemble aligned −12.41%")
for bar, v in zip(bars, delta_f):
    ax.text(bar.get_x() + bar.get_width() / 2, v + (0.3 if v >= 0 else -0.3),
            f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top",
            color="white", fontsize=9)
ax.set_ylabel("Δ rMAE vs fixed_16 (%)")
ax.set_title("Рис.C  Aligned каскад: dim-based p vs oracle/ensemble  (<0 = лучше baseline)")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "94_C.png", dpi=120); plt.close()
print("Рис. C сохранён")

# Рис. D — Oracle_p распределение (aligned vs скр.88 even-grid)
OLD_ORACLE_DIST_88 = {4: 17, 6: 8, 8: 22, 10: 9, 12: 10, 14: 9, 16: 25}  # примерные %
cnt_ora_aln = Counter(int(p) for p in oracle_p_arr)
total = max(len(oracle_p_arr), 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
y_aln = [cnt_ora_aln.get(p, 0) / total * 100 for p in P_REG_GRID]
ax.bar([str(p) for p in P_REG_GRID], y_aln, color="#42a5f5")
ax.set_title("Oracle_p aligned (скр.94)\np∈[4..16]")
ax.set_xlabel("p_reg"); ax.set_ylabel("% origins")
for i, (p, v) in enumerate(zip(P_REG_GRID, y_aln)):
    if v > 1:
        ax.text(i, v + 0.3, f"{v:.0f}%", ha="center", fontsize=8, color="white")

ax = axes[1]
for dim_key, color, label in [
        ("aln_lb",   "#ffa726", "Aligned LB"),
        ("aln_corr", "#66bb6a", "Aligned corr")]:
    cp = all_res[dim_key]["chosen_p"]
    if not cp:
        continue
    cnt_d = Counter(cp)
    y_d = [cnt_d.get(p, 0) / max(len(cp), 1) * 100 for p in P_REG_GRID]
    ax.plot([str(p) for p in P_REG_GRID], y_d, "o-", color=color, lw=1.5, label=label)
ax.plot([str(p) for p in P_REG_GRID], y_aln, "w--o", ms=5, lw=1.5, label="oracle_p")
ax.set_title("p-выбор dim-методов vs oracle_p")
ax.set_xlabel("p_reg"); ax.set_ylabel("% origins")
ax.legend(fontsize=8)

plt.suptitle("Рис.D  oracle_p и dim-based выбор p (aligned каскад)", y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "94_D.png", dpi=120); plt.close()
print("Рис. D сохранён")

print(f"\nГотово. Фигуры: 94_A...D.png")
