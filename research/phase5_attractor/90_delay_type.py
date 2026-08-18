"""
90_delay_type.py — Тип задержек: равномерные vs геометрические.

Гипотеза: аттрактор att многомасштабен. Геометрические лаги
[64, 32, 16, 8, 4, 2, 1, 0] охватывают все масштабы одновременно —
меньшим числом измерений, без каскада. Возможно, они эффективнее
равномерных лагов [0,1,2,...,15] для поиска соседей.

Тест:
  Равномерные (стандарт из скр.85):
    unif_fixed16  — p=16, октавный каскад ×2, acc_ang (baseline)
    unif_ensemble — ensemble 1/LOO, p ∈ {4…16}

  Геометрические (unified: X_search = X_fit = geom лаги, без каскада):
    geom_k5   — [64, 32, 16, 8, 0]           5 лагов
    geom_k7   — [64, 32, 16, 8, 4, 2, 0]     7 лагов
    geom_k8   — [64, 32, 16, 8, 4, 2, 1, 0]  8 лагов
    geom_k8_ang — geom_k8 + acc_ang в пространстве geom лагов

  Производные:
    geom_oracle   — oracle (лучший из k5/k7/k8 на каждый origin)
    geom_ensemble — 1/LOO взвешивание по k5/k7/k8
    adaptive_LOO  — per-origin LOO-выбор между unif_fixed16 и geom_k8

Примечание: unified=True обязательно. Иначе LWR → OLS при нерегулярных лагах.

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров 1d.
"""
from __future__ import annotations
import time
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "candles"
FIG_DIR  = ROOT / "research" / "figures"

TICKERS   = ["SBER", "MRKP", "LKOH", "NVTK", "CHMF", "NLMK", "MGNT", "VTBR"]
INTERVAL  = "1d"
N_ORIGINS = 40
STEP_WF   = 5

P_MAX    = 64
LAMBDA   = 0.01
XI_FIXED = 51

LP_M = 9; LP_D = 3; LP_K = 30; LP_N = 3

P_REG_GRID = [4, 6, 8, 10, 12, 14, 16]
P_FIT_BASE = 16

# Наборы геометрических лагов (descending, включая lag=0 = текущий момент)
def _make_geom_lags(k: int, max_lag: int, base: float = 2.0) -> list[int]:
    """max_lag / base^i для i=0..k-2, плюс 0. Дедупликация."""
    lags: set[int] = {0}
    for i in range(k - 1):
        lag = int(round(max_lag / (base ** i)))
        if lag > 0:
            lags.add(lag)
    return sorted(lags, reverse=True)

GEOM_SETS: dict[str, list[int]] = {
    "geom_k5": _make_geom_lags(5, P_MAX),   # [64,32,16,8,0]
    "geom_k7": _make_geom_lags(7, P_MAX),   # [64,32,16,8,4,2,0]
    "geom_k8": _make_geom_lags(9, P_MAX),   # [64,32,16,8,4,2,1,0]  (k=9 даёт 8 уник.)
}
GEOM_KEYS = list(GEOM_SETS.keys())

METHOD_KEYS = [
    "unif_fixed16", "unif_ensemble",
    "geom_k5", "geom_k7", "geom_k8", "geom_k8_ang",
    "geom_oracle", "geom_ensemble", "adaptive_LOO",
]


# ── утилиты ───────────────────────────────────────────────────────────────────

def _logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t); trend[:2] = close[:2]
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
            result[i:i + m] += X_proj[i]; count[i:i + m] += 1
        s = result / np.maximum(count, 1)
    return np.diff(s)


def _cosine_dist(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    nA = np.linalg.norm(A, axis=1); nb = float(np.linalg.norm(b))
    if nb < 1e-12: return np.ones(len(A))
    cos = np.where(nA > 1e-12, (A @ b) / (nA * nb), 0.0)
    return 1.0 - np.clip(cos, -1.0, 1.0)


def _lwr_predict(X_nn: np.ndarray, y_nn: np.ndarray, q: np.ndarray) -> float:
    d = np.linalg.norm(X_nn - q, axis=1); h = max(float(d.max()), 1e-10)
    w = np.exp(-0.5 * (d / h) ** 2)
    A = np.hstack([np.ones((len(X_nn), 1)), X_nn]); sw = np.sqrt(np.maximum(w, 1e-30))
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + q @ c[1:])


def _lwr_loo(X_nn: np.ndarray, y_nn: np.ndarray, q: np.ndarray) -> float:
    d = np.linalg.norm(X_nn - q, axis=1); h = max(float(d.max()), 1e-10)
    w = np.maximum(np.exp(-0.5 * (d / h) ** 2), 1e-30)
    A = np.hstack([np.ones((len(X_nn), 1)), X_nn]); sw = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    resid = y_nn - A @ c
    try:
        Aw = sw[:, None] * A
        inv = np.linalg.inv(Aw.T @ Aw + 1e-10 * np.eye(A.shape[1]))
        h_d = w * np.einsum("ij,ij->i", A @ inv, A)
        e_loo = np.where(np.abs(1 - h_d) > 1e-6, resid / (1 - h_d), resid)
    except np.linalg.LinAlgError:
        e_loo = resid
    return float(np.mean(np.abs(e_loo)))


# ── уровни каскада (только для uniform) ──────────────────────────────────────

def _octave_levels(p_fit: int, p_max: int = P_MAX, step: float = 2.0) -> list[int]:
    levels: list[int] = []; p = p_max
    while True:
        levels.append(p)
        p_next = int(p / step)
        if p_next < max(p_fit, 2): break
        p = p_next
    if levels[-1] != p_fit: levels.append(p_fit)
    return levels


# ── контекст: равномерные лаги ────────────────────────────────────────────────

class _CtxUnif:
    __slots__ = ("X_full", "y_base", "acc_hist", "vec_full", "n", "ok")

    def __init__(self, att: np.ndarray) -> None:
        n = len(att); self.n = n; self.ok = False
        if n - P_MAX - 1 < 3: return
        t_arr = np.arange(P_MAX, n - 1)
        self.X_full   = np.column_stack([att[t_arr - (P_MAX - 1 - j)] for j in range(P_MAX)])
        self.y_base   = att[t_arr + 1]
        acc = np.zeros(n)
        if n >= 3: acc[2:] = att[2:] - 2 * att[1:-1] + att[:-2]
        self.acc_hist = acc
        self.vec_full = att[-P_MAX:].copy()
        self.ok       = len(t_arr) >= XI_FIXED + 1


def _forecast_unif(ctx: _CtxUnif, p_reg: int,
                   levels: list[int], need_loo: bool) -> tuple[float, float]:
    """Uniform: каскадный поиск + acc_ang."""
    X_full = ctx.X_full; y_base = ctx.y_base
    vec_full = ctx.vec_full; acc_hist = ctx.acc_hist; n = ctx.n

    if len(X_full) < XI_FIXED + 1: return np.nan, np.nan

    t_arr   = np.arange(P_MAX, n - 1)
    X_acc   = np.column_stack([acc_hist[t_arr - (p_reg - 1 - j)] for j in range(p_reg)])
    vec_acc = acc_hist[-p_reg:].copy()
    query   = vec_full[-p_reg:]
    cands   = np.arange(len(X_full))

    for k, p_lvl in enumerate(levels):
        if k == len(levels) - 1: break
        xi_lvl = min(XI_FIXED, len(cands))
        if len(cands) > xi_lvl:
            d = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full[-p_lvl:], axis=1)
            cands = cands[np.argpartition(d, xi_lvl - 1)[:xi_lvl]]
        p_next = levels[k + 1]; radius = p_lvl - p_next
        if radius > 0:
            exp   = cands[:, None] - np.arange(radius + 1)[None, :]
            cands = np.unique(np.clip(exp, 0, len(X_full) - 1))

    if len(cands) > XI_FIXED:
        d_pos = np.linalg.norm(X_full[cands, -p_reg:] - query, axis=1)
        d_acc = _cosine_dist(X_acc[cands], vec_acc)
        cands = cands[np.argpartition(d_pos + LAMBDA * d_acc, XI_FIXED - 1)[:XI_FIXED]]

    if len(cands) < p_reg + 2: return np.nan, np.nan
    X_nn = X_full[cands, -p_reg:]; y_nn = y_base[cands]
    pred = _lwr_predict(X_nn, y_nn, query)
    loo  = _lwr_loo(X_nn, y_nn, query) if need_loo else np.nan
    return pred, loo


# ── контекст: геометрические лаги ────────────────────────────────────────────

class _CtxGeom:
    __slots__ = ("X", "y", "X_acc", "vec", "vec_acc", "lags", "ok")

    def __init__(self, att: np.ndarray, lags: list[int]) -> None:
        self.lags = lags; self.ok = False
        max_lag = max(lags); n = len(att)
        if n - max_lag - 1 < XI_FIXED + 1: return
        t_arr = np.arange(max_lag, n - 1)
        self.X   = np.column_stack([att[t_arr - lag] for lag in lags])
        self.y   = att[t_arr + 1]
        acc = np.zeros(n)
        if n >= 3: acc[2:] = att[2:] - 2 * att[1:-1] + att[:-2]
        self.X_acc   = np.column_stack([acc[t_arr - lag] for lag in lags])
        self.vec     = np.array([att[n - 1 - lag] for lag in lags])
        self.vec_acc = np.array([acc[n - 1 - lag] for lag in lags])
        self.ok      = True


def _forecast_geom(ctx_g: _CtxGeom,
                   use_acc: bool, need_loo: bool) -> tuple[float, float]:
    """Geom: прямой поиск в geom-пространстве (без каскада), unified."""
    if not ctx_g.ok: return np.nan, np.nan
    n_lib = len(ctx_g.X)
    if n_lib < len(ctx_g.lags) + 2: return np.nan, np.nan

    d_all = np.linalg.norm(ctx_g.X - ctx_g.vec, axis=1)
    xi    = min(XI_FIXED, n_lib)
    if n_lib > xi:
        cands = np.argpartition(d_all, xi - 1)[:xi]
    else:
        cands = np.arange(n_lib)

    if use_acc and len(cands) > len(ctx_g.lags) + 2:
        d_pos = d_all[cands]
        d_acc = _cosine_dist(ctx_g.X_acc[cands], ctx_g.vec_acc)
        xi2   = min(xi, len(cands))
        cands = cands[np.argpartition(d_pos + LAMBDA * d_acc, xi2 - 1)[:xi2]]

    if len(cands) < len(ctx_g.lags) + 2: return np.nan, np.nan
    X_nn = ctx_g.X[cands]; y_nn = ctx_g.y[cands]
    pred = _lwr_predict(X_nn, y_nn, ctx_g.vec)
    loo  = _lwr_loo(X_nn, y_nn, ctx_g.vec) if need_loo else np.nan
    return pred, loo


# ── загрузка ──────────────────────────────────────────────────────────────────

def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend_causal(close)
    return _lp_proj(close / np.maximum(lt, 1e-10), LP_M, LP_D, LP_K, LP_N)


# ── walk-forward ──────────────────────────────────────────────────────────────

def wf_ticker(att_full: np.ndarray) -> dict:
    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP_WF, n_total - 1, STEP_WF))
    out = {k: {"errors": [], "signs": []} for k in METHOD_KEYS}

    for t_orig in origins:
        if t_orig + 1 >= n_total: continue
        hist = att_full[:t_orig + 1]
        true = att_full[t_orig + 1]

        # ── равномерные ──
        ctx_u = _CtxUnif(hist)
        if not ctx_u.ok: continue

        # baseline fixed_16
        pred_f, loo_f = _forecast_unif(ctx_u, P_FIT_BASE,
                                        _octave_levels(P_FIT_BASE), need_loo=True)
        if not np.isnan(pred_f):
            out["unif_fixed16"]["errors"].append(pred_f - true)
            out["unif_fixed16"]["signs"].append(int(np.sign(pred_f) == np.sign(true)))

        # uniform ensemble
        u_preds: dict[int, float] = {}; u_loos: dict[int, float] = {}
        for p in P_REG_GRID:
            pr, lo = _forecast_unif(ctx_u, p, _octave_levels(p), need_loo=True)
            u_preds[p] = pr; u_loos[p] = lo
        valid_u = [p for p in P_REG_GRID
                   if not np.isnan(u_preds[p]) and not np.isnan(u_loos[p])]
        if valid_u:
            inv = np.array([1.0 / (u_loos[p] + 1e-12) for p in valid_u])
            w   = inv / inv.sum()
            pe  = float(w @ np.array([u_preds[p] for p in valid_u]))
            out["unif_ensemble"]["errors"].append(pe - true)
            out["unif_ensemble"]["signs"].append(int(np.sign(pe) == np.sign(true)))
            loo_ens_u = float(np.mean([u_loos[p] for p in valid_u]))
        else:
            loo_ens_u = np.nan

        # ── геометрические ──
        g_preds: dict[str, float] = {}; g_loos: dict[str, float] = {}
        for gk, lags in GEOM_SETS.items():
            ctx_g = _CtxGeom(hist, lags)
            if not ctx_g.ok:
                g_preds[gk] = np.nan; g_loos[gk] = np.nan; continue
            pr, lo = _forecast_geom(ctx_g, use_acc=False, need_loo=True)
            g_preds[gk] = pr; g_loos[gk] = lo
            # записываем отдельные geom методы
            if not np.isnan(pr):
                out[gk]["errors"].append(pr - true)
                out[gk]["signs"].append(int(np.sign(pr) == np.sign(true)))

        # geom_k8 + acc_ang
        ctx_g8 = _CtxGeom(hist, GEOM_SETS["geom_k8"])
        if ctx_g8.ok:
            pr_ang, lo_ang = _forecast_geom(ctx_g8, use_acc=True, need_loo=True)
        else:
            pr_ang, lo_ang = np.nan, np.nan
        if not np.isnan(pr_ang):
            out["geom_k8_ang"]["errors"].append(pr_ang - true)
            out["geom_k8_ang"]["signs"].append(int(np.sign(pr_ang) == np.sign(true)))

        # geom oracle (лучший из k5/k7/k8)
        valid_g = [k for k in GEOM_KEYS
                   if not np.isnan(g_preds[k]) and not np.isnan(g_loos[k])]
        if valid_g:
            best_gk = min(valid_g, key=lambda k: abs(g_preds[k] - true))
            out["geom_oracle"]["errors"].append(g_preds[best_gk] - true)
            out["geom_oracle"]["signs"].append(
                int(np.sign(g_preds[best_gk]) == np.sign(true)))

            # geom ensemble 1/LOO
            inv_g = np.array([1.0 / (g_loos[k] + 1e-12) for k in valid_g])
            w_g   = inv_g / inv_g.sum()
            pe_g  = float(w_g @ np.array([g_preds[k] for k in valid_g]))
            out["geom_ensemble"]["errors"].append(pe_g - true)
            out["geom_ensemble"]["signs"].append(int(np.sign(pe_g) == np.sign(true)))
            loo_geom_best = float(np.min([g_loos[k] for k in valid_g]))
        else:
            loo_geom_best = np.nan

        # adaptive LOO: выбор между unif_fixed16 и geom_k8
        pr_g8 = g_preds.get("geom_k8", np.nan)
        lo_g8 = g_loos.get("geom_k8", np.nan)
        if (not np.isnan(pred_f) and not np.isnan(loo_f)
                and not np.isnan(pr_g8) and not np.isnan(lo_g8)):
            chosen = pred_f if loo_f <= lo_g8 else pr_g8
            out["adaptive_LOO"]["errors"].append(chosen - true)
            out["adaptive_LOO"]["signs"].append(int(np.sign(chosen) == np.sign(true)))

    return out


# ── запуск ────────────────────────────────────────────────────────────────────

print("Загрузка + Local Projective...")
t0 = time.time()
att_data: dict[str, np.ndarray] = {}
for tkr in TICKERS:
    att_data[tkr] = load_att(tkr)
    print(f"  {tkr}...")
std_all  = [float(np.std(att_data[tkr])) for tkr in TICKERS]
std_mean = float(np.mean(std_all))
print(f"Загрузка: {time.time()-t0:.1f}с")

print("\nНаборы геометрических лагов:")
for gk, lags in GEOM_SETS.items():
    print(f"  {gk}: {lags}  ({len(lags)} лагов)")

print(f"\nWalk-forward ({len(TICKERS)} тикеров)...")
t0 = time.time()

agg = {k: {"errors": [], "signs": []} for k in METHOD_KEYS}
for tkr in TICKERS:
    t1  = time.time()
    res = wf_ticker(att_data[tkr])
    for k in METHOD_KEYS:
        agg[k]["errors"].extend(res[k]["errors"])
        agg[k]["signs"].extend(res[k]["signs"])
    print(f"  {tkr}: {time.time()-t1:.1f}с")
print(f"Walk-forward: {time.time()-t0:.1f}с")


def rmae(k: str) -> float:
    e = agg[k]["errors"]
    return float(np.mean(np.abs(e))) / std_mean if e else np.nan

def sacc(k: str) -> float:
    s = agg[k]["signs"]
    return float(np.mean(s)) * 100.0 if s else np.nan


baseline = rmae("unif_fixed16")
print(f"\nBaseline (unif_fixed16): rMAE = {baseline:.4f}")

print("\n── Итоговая таблица ──")
LABELS = {
    "unif_fixed16":  "Равномерные p=16 (baseline)",
    "unif_ensemble": "Равномерные ensemble",
    "geom_k5":       "Геом k5 [64,32,16,8,0]",
    "geom_k7":       "Геом k7 [64,32,16,8,4,2,0]",
    "geom_k8":       "Геом k8 [64,32,16,8,4,2,1,0]",
    "geom_k8_ang":   "Геом k8 + acc_ang",
    "geom_oracle":   "Геом oracle (k5/k7/k8)",
    "geom_ensemble": "Геом ensemble (1/LOO)",
    "adaptive_LOO":  "Adaptive: unif vs geom_k8",
}
print(f"{'Метод':40s}  {'rMAE':7s}  {'Δ%':8s}  {'SignAcc':8s}  N")
for k in METHOD_KEYS:
    r  = rmae(k); d = (r / baseline - 1) * 100 if not np.isnan(r) else np.nan
    sa = sacc(k); n = len(agg[k]["errors"])
    star = " ◄" if k in ("unif_ensemble", "geom_ensemble", "adaptive_LOO") else ""
    print(f"  {LABELS[k]:40s}  {r:.4f}  {d:+7.2f}%  {sa:.1f}%   {n}{star}")


# ── Фигуры ────────────────────────────────────────────────────────────────────

SHOW_KEYS = ["unif_fixed16", "unif_ensemble",
             "geom_k5", "geom_k7", "geom_k8", "geom_k8_ang",
             "geom_oracle", "geom_ensemble", "adaptive_LOO"]
SHOW_LBLS = ["unif\np=16", "unif\nens.", "geom\nk5", "geom\nk7",
             "geom\nk8", "geom\nk8+ang", "geom\noracle", "geom\nens.",
             "adapt.\nLOO"]

r_vals = [rmae(k) for k in SHOW_KEYS]
d_vals = [(r / baseline - 1) * 100 if not np.isnan(r) else 0.0 for r in r_vals]

# Рис. A — bar chart всех методов
colors = ["#78909c"] + ["#ef5350" if d > 0 else "#66bb6a" for d in d_vals[1:]]
fig, ax = plt.subplots(figsize=(13, 4))
bars = ax.bar(SHOW_LBLS, d_vals, color=colors, width=0.55)
ax.axhline(0, color="white", lw=0.8)
for bar, v in zip(bars, d_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            v + (0.2 if v >= 0 else -0.2),
            f"{v:+.1f}%", ha="center",
            va="bottom" if v >= 0 else "top", color="white", fontsize=9)
ax.set_ylabel("Δ rMAE vs unif_fixed16 (%)")
ax.set_title("Рис.A  Равномерные vs Геометрические задержки  (<0 = лучше baseline)")
plt.tight_layout()
fig.savefig(FIG_DIR / "90_A.png", dpi=120); plt.close()
print("\nРис. A сохранён")

# Рис. B — детальное сравнение: uniform vs geom лучших
COMPARE = ["unif_fixed16", "unif_ensemble", "geom_k8", "geom_k8_ang",
           "geom_ensemble", "adaptive_LOO"]
CLBLS   = ["unif\np=16", "unif\nens.", "geom\nk8", "geom\nk8\n+ang",
           "geom\nens.", "adapt.\nLOO"]
rc = [rmae(k) for k in COMPARE]
dc = [(r / baseline - 1) * 100 if not np.isnan(r) else 0.0 for r in rc]
colors2 = ["#78909c"] + ["#ef5350" if d > 0 else "#66bb6a" for d in dc[1:]]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
bars2 = axes[0].bar(CLBLS, dc, color=colors2, width=0.55)
axes[0].axhline(0, color="white", lw=0.8)
for bar, v in zip(bars2, dc):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 v + (0.15 if v >= 0 else -0.15),
                 f"{v:+.1f}%", ha="center",
                 va="bottom" if v >= 0 else "top", color="white", fontsize=9)
axes[0].set_ylabel("Δ rMAE vs baseline (%)")
axes[0].set_title("Рис.B.1  Ключевые методы")

# SignAcc
sa_vals = [sacc(k) for k in COMPARE]
axes[1].bar(CLBLS, sa_vals, color=colors2, width=0.55)
axes[1].axhline(sacc("unif_fixed16"), color="white", lw=1.2, ls="--", label="baseline")
axes[1].set_ylabel("SignAcc (%)")
axes[1].set_title("Рис.B.2  SignAcc")
axes[1].legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIG_DIR / "90_B.png", dpi=120); plt.close()
print("Рис. B сохранён")

print("\nГотово. Фигуры: 90_A.png, 90_B.png")
