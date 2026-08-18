"""
80 — Комбинирование победителей: split-budget и итоговая сводка.

Итоги скр.77-79:
  vel_eucl (auto)  : -1.4% rMAE,  +0.6% SignAcc
  vel_ang  (0.01)  : -8.5% rMAE,  +0.0% SignAcc   ← лучший rMAE скр.78
  acc_ang  (auto)  : -9.4% rMAE,  +1.5% SignAcc   ← лучший по обеим скр.79
  vel+acc additive : хуже vel_ang — конфликт метрик

Проблема аддитивного объединения: два угловых ограничения сразу
отбрасывают слишком много кандидатов. Решение — SPLIT BUDGET:
  Выбрать n_v соседей по vel_ang + n_a соседей по acc_ang → объединить.
  Общий бюджет = xi_lwr. Параметр: ratio r = n_v / xi_lwr ∈ [0, 1].
  r=0 → чистый acc_ang, r=1 → чистый vel_ang, r=0.5 → поровну.

Также тестируем optimal-λ grid для acc_ang_only (скр.79 использовал только auto).

Режимы:
  0. pos-only
  1. vel_ang (0.01)           ← лучший скр.78
  2. acc_ang (grid-opt)        ← лучший скр.79, λ оптимальный
  3. split r=0.25              ← 25% vel, 75% acc
  4. split r=0.5               ← поровну
  5. split r=0.75              ← 75% vel, 25% acc
  6. split r=0.25 + global_blend(pct=30)  ← каскад + глобал
  7. acc_ang_opt + global_blend(pct=30)

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров.
Att-фильтр: Local Projective (m=9, d=3, k=30, n=3) — стандарт.

Графики:
  A — rMAE vs split-ratio r (grid r=[0..1]), сравнение с одиночными методами
  B — SignAcc vs split-ratio r
  C — Финальная сводная таблица ВСЕХ методов скр.77-80 (rMAE + SignAcc)
  D — Grid λ_aa для acc_ang_only (без vel) — нахождение истинного оптимума
  E — Per-ticker heatmap: delta rMAE от baseline для лучших 4 методов
"""

import json, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

ROOT    = Path(__file__).resolve().parent.parent.parent
FIGDIR  = ROOT / "research" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATADIR = ROOT / "data" / "candles"
sys.path.insert(0, str(ROOT))

TICKERS  = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL = "1d"

LP_M = 9; LP_D = 3; LP_K = 30; LP_N = 3

P_FIT     = 16
P_MAX     = 64
XI_LWR    = 3 * (P_FIT + 1)
STEP_BASE = 2.0

LAM_VA_OPT = 0.01    # оптимум скр.78 для vel_ang
LAM_AA_GRID = [0.0, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.2]

SPLIT_RATIOS = [0.0, 0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9, 1.0]  # r = n_v / xi_lwr
GLOBAL_PCT   = 30    # % из каскада в global_blend

N_TEST  = 200
STEP_WF = 5

COLORS_TK = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
             "#9467bd","#8c564b","#e377c2","#17becf"]

# ── helpers ────────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64); cn = np.arange(1, n+1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2); cy = np.cumsum(lc); cty = np.cumsum(t*lc)
    denom = cn*ct2 - ct**2
    b = np.where(denom > 0, (cn*cty - ct*cy)/denom, 0.0)
    a = (cy - b*ct)/cn
    trend = np.exp(a + b*t); trend[:2] = close[:2]
    return trend


def local_projective(series: np.ndarray, m: int, d: int, k: int, n_iter: int) -> np.ndarray:
    s = series.copy().astype(np.float64); N = len(s)
    k_eff = min(k, N - m); d_eff = min(d, m - 1)
    for _ in range(n_iter):
        n_pts = N - m + 1
        rows  = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
        X     = s[rows]; tree = KDTree(X)
        _, inds = tree.query(X, k=k_eff + 1)
        X_proj = np.empty_like(X)
        for i in range(n_pts):
            nn = inds[i, 1:]; X_nn = X[nn]; c = X_nn.mean(0)
            _, _, Vt = np.linalg.svd(X_nn - c, full_matrices=False)
            Vd = Vt[:d_eff].T; xc = X[i] - c
            X_proj[i] = c + Vd @ (Vd.T @ xc)
        res = np.zeros(N); cnt = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            res[i:i+m] += X_proj[i]; cnt[i:i+m] += 1
        s = res / np.maximum(cnt, 1)
    return np.diff(s)


def load_att(ticker: str) -> np.ndarray:
    raw   = json.loads((DATADIR / ticker / f"{INTERVAL}.json").read_text())
    cands = raw["candles"] if isinstance(raw, dict) else raw
    close = np.array([c["close"] for c in cands], dtype=np.float64)
    trend = logtrend_causal(close)
    ratio = close / trend
    return local_projective(ratio, LP_M, LP_D, LP_K, LP_N)


def octave_levels(p_fit: int, p_max: int, step: float = 2.0) -> list[int]:
    levels: list[int] = []; p = p_max
    while True:
        levels.append(p)
        p_next = int(p / step)
        if p_next < max(p_fit, 2): break
        p = p_next
    if levels[-1] != p_fit: levels.append(p_fit)
    return levels


def lwr_fit(X_nn: np.ndarray, y_nn: np.ndarray, vec_f: np.ndarray, h_bw: float) -> float:
    dists = np.linalg.norm(X_nn - vec_f, axis=1)
    w = np.exp(-0.5 * (dists / h_bw) ** 2)
    if w.sum() < 1e-15:
        return float(y_nn.mean())
    try:
        A = np.column_stack([np.ones(len(X_nn)), X_nn])
        AtW = (A * w[:, None]).T
        coef = np.linalg.lstsq(AtW @ A, AtW @ y_nn, rcond=None)[0]
        return float(coef[0] + coef[1:] @ vec_f)
    except np.linalg.LinAlgError:
        return float((w @ y_nn) / w.sum())


def cosine_dist(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_A = np.linalg.norm(A, axis=1)
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-12:
        return np.ones(len(A))
    dot   = A @ b
    denom = norm_A * norm_b
    cos   = np.where(denom > 1e-12, dot / denom, 0.0)
    return 1.0 - np.clip(cos, -1.0, 1.0)


def _build_pool(att_hist: np.ndarray):
    """Строит матрицы и прогоняет каскад. Возвращает dict или None."""
    n  = len(att_hist)
    m  = n - P_MAX - 1
    if m < XI_LWR + 5:
        return None

    vel_hist = np.concatenate([[0.0], np.diff(att_hist)])
    acc_hist = np.concatenate([[0.0, 0.0], np.diff(vel_hist[1:])])

    t_arr   = np.arange(P_MAX, n - 1)
    X_pos_w = np.column_stack([att_hist[t_arr - (P_MAX-1-j)] for j in range(P_MAX)])
    X_pos   = np.column_stack([att_hist[t_arr - (P_FIT-1-j)] for j in range(P_FIT)])
    X_vel   = np.column_stack([vel_hist[t_arr - (P_FIT-1-j)] for j in range(P_FIT)])
    X_acc   = np.column_stack([acc_hist[t_arr - (P_FIT-1-j)] for j in range(P_FIT)])
    y_base  = att_hist[t_arr + 1]

    vec_posw = att_hist[-P_MAX:]
    vec_pos  = att_hist[-P_FIT:]
    vec_vel  = vel_hist[-P_FIT:]
    vec_acc  = acc_hist[-P_FIT:]

    levels = octave_levels(P_FIT, P_MAX, STEP_BASE)
    cands  = np.arange(m)
    for k, p_lvl in enumerate(levels):
        if k == len(levels) - 1: break
        xi_lvl = min(XI_LWR, len(cands))
        if len(cands) > xi_lvl:
            d = np.linalg.norm(X_pos_w[cands, -p_lvl:] - vec_posw[-p_lvl:], axis=1)
            cands = cands[np.argpartition(d, xi_lvl-1)[:xi_lvl]]
        p_next  = levels[k+1]; radius = p_lvl - p_next
        offsets = np.arange(radius + 1)
        exp     = cands[:, None] - offsets[None, :]
        cands   = np.unique(np.clip(exp, 0, m-1))

    return dict(
        cands=cands, m=m,
        X_pos=X_pos, X_vel=X_vel, X_acc=X_acc, y_base=y_base,
        vec_pos=vec_pos, vec_vel=vec_vel, vec_acc=vec_acc,
    )


def lwr_predict(pool: dict, cands_final: np.ndarray) -> float:
    X_nn = pool["X_pos"][cands_final]
    y_nn = pool["y_base"][cands_final]
    vec  = pool["vec_pos"]
    if len(cands_final) < P_FIT + 2:
        return np.nan
    h_bw = max(float(np.linalg.norm(X_nn - vec, axis=1).max()), 1e-10)
    return lwr_fit(X_nn, y_nn, vec, h_bw)


def select_by_metric(cands: np.ndarray, dist: np.ndarray, n: int) -> np.ndarray:
    """Отбирает n кандидатов с минимальным расстоянием."""
    n = min(n, len(cands))
    if len(cands) <= n:
        return cands.copy()
    return cands[np.argpartition(dist, n-1)[:n]]


def lwr_step_additive(att_hist: np.ndarray, lam_va: float, lam_aa: float) -> float:
    """pos + lam_va·d_ang(vel) + lam_aa·d_ang(acc)  (аддитивное)."""
    pool = _build_pool(att_hist)
    if pool is None: return np.nan
    cands   = pool["cands"]
    d_pos   = np.linalg.norm(pool["X_pos"][cands] - pool["vec_pos"], axis=1)
    d_comb  = d_pos.copy()
    if lam_va > 0: d_comb += lam_va * cosine_dist(pool["X_vel"][cands], pool["vec_vel"])
    if lam_aa > 0: d_comb += lam_aa * cosine_dist(pool["X_acc"][cands], pool["vec_acc"])
    sel = select_by_metric(np.arange(len(cands)), d_comb, XI_LWR)
    return lwr_predict(pool, cands[sel])


def lwr_step_split(att_hist: np.ndarray,
                   lam_va: float, lam_aa: float,
                   split_r: float) -> float:
    """Split-budget: n_v = round(r·ξ) соседей по vel_ang +
                    n_a = ξ - n_v соседей по acc_ang → объединение.
    split_r = n_v / xi_lwr.
    """
    pool = _build_pool(att_hist)
    if pool is None: return np.nan
    cands = pool["cands"]
    if len(cands) < P_FIT + 2: return np.nan

    n_v = int(round(split_r * XI_LWR))
    n_a = XI_LWR - n_v

    d_pos = np.linalg.norm(pool["X_pos"][cands] - pool["vec_pos"], axis=1)

    if n_v > 0 and lam_va > 0:
        d_v = d_pos + lam_va * cosine_dist(pool["X_vel"][cands], pool["vec_vel"])
        top_v = cands[select_by_metric(np.arange(len(cands)), d_v, n_v)]
    elif n_v > 0:
        top_v = cands[select_by_metric(np.arange(len(cands)), d_pos, n_v)]
    else:
        top_v = np.array([], dtype=np.intp)

    if n_a > 0 and lam_aa > 0:
        d_a = d_pos + lam_aa * cosine_dist(pool["X_acc"][cands], pool["vec_acc"])
        top_a = cands[select_by_metric(np.arange(len(cands)), d_a, n_a)]
    elif n_a > 0:
        top_a = cands[select_by_metric(np.arange(len(cands)), d_pos, n_a)]
    else:
        top_a = np.array([], dtype=np.intp)

    cands_final = np.unique(np.concatenate([top_v, top_a]))
    return lwr_predict(pool, cands_final)


def lwr_step_global_blend(att_hist: np.ndarray,
                          lam_va: float, lam_aa: float,
                          split_r: float,
                          global_pct: float) -> float:
    """Split-budget + global blend: берём pct% из каскадного пула,
    остаток ищем по всей истории по d_pos."""
    pool = _build_pool(att_hist)
    if pool is None: return np.nan
    cands = pool["cands"]
    if len(cands) < P_FIT + 2: return np.nan

    n_cas_total = max(P_FIT + 2, round(XI_LWR * global_pct / 100.0))
    n_cas_total = min(n_cas_total, len(cands))
    n_glob = XI_LWR - n_cas_total
    if n_glob <= 0:
        return lwr_step_split(att_hist, lam_va, lam_aa, split_r)

    # каскадная часть (split-budget внутри пула)
    n_v = int(round(split_r * n_cas_total))
    n_a = n_cas_total - n_v
    d_pos = np.linalg.norm(pool["X_pos"][cands] - pool["vec_pos"], axis=1)

    top_v_idx: np.ndarray
    top_a_idx: np.ndarray
    if n_v > 0 and lam_va > 0:
        d_v = d_pos + lam_va * cosine_dist(pool["X_vel"][cands], pool["vec_vel"])
        top_v_idx = select_by_metric(np.arange(len(cands)), d_v, n_v)
    elif n_v > 0:
        top_v_idx = select_by_metric(np.arange(len(cands)), d_pos, n_v)
    else:
        top_v_idx = np.array([], dtype=np.intp)

    if n_a > 0 and lam_aa > 0:
        d_a = d_pos + lam_aa * cosine_dist(pool["X_acc"][cands], pool["vec_acc"])
        top_a_idx = select_by_metric(np.arange(len(cands)), d_a, n_a)
    elif n_a > 0:
        top_a_idx = select_by_metric(np.arange(len(cands)), d_pos, n_a)
    else:
        top_a_idx = np.array([], dtype=np.intp)

    top_cas = np.unique(np.concatenate([cands[top_v_idx], cands[top_a_idx]]))

    # глобальная часть: поиск по всему X_pos по d_pos
    m = pool["m"]
    d_glob = np.linalg.norm(pool["X_pos"][:m] - pool["vec_pos"], axis=1)
    n_g = min(n_glob, m)
    top_glob = np.argpartition(d_glob, n_g-1)[:n_g] if n_g > 0 else np.array([], dtype=np.intp)

    cands_final = np.unique(np.concatenate([top_cas, top_glob]))
    return lwr_predict(pool, cands_final)


# ── загрузка данных ────────────────────────────────────────────────────────────

print("Загрузка + Local Projective...")
atts: dict[str, np.ndarray] = {}
t_load = time.time()
for tk in TICKERS:
    print(f"  {tk}...", end=" ", flush=True)
    t1 = time.time(); atts[tk] = load_att(tk)
    print(f"{time.time()-t1:.1f}с")
print(f"Загрузка итого: {time.time()-t_load:.1f}с")

# auto-λ_aa по SBER (из скр.79 = 0.0156, вычислим снова)
att_sber = atts["SBER"]
n_demo   = len(att_sber) - N_TEST // 2
pool_d   = _build_pool(att_sber[:n_demo+1])
cands_d  = pool_d["cands"]
d_pos_d  = np.linalg.norm(pool_d["X_pos"][cands_d] - pool_d["vec_pos"], axis=1)
d_aa_d   = cosine_dist(pool_d["X_acc"][cands_d], pool_d["vec_acc"])
auto_lam_aa = float(np.median(d_pos_d)) / (float(np.median(d_aa_d)) + 1e-12)
print(f"auto λ_aa: {auto_lam_aa:.4f}")

# ── grid λ_aa для acc_ang_only ─────────────────────────────────────────────────

LAM_AA_GRID_EXT = LAM_AA_GRID + [auto_lam_aa]
LAM_AA_LABELS   = [str(x) for x in LAM_AA_GRID] + [f"auto\n({auto_lam_aa:.3f})"]

res_grid_aa = {li: {tk: {"mae": [], "sign": []} for tk in TICKERS}
               for li in range(len(LAM_AA_GRID_EXT))}

t0 = time.time()
print("\nGrid λ_aa (acc_ang_only)...")
for tk in TICKERS:
    att = atts[tk]; n_att = len(att)
    origins = list(range(n_att - N_TEST, n_att - 1, STEP_WF))
    for t_orig in origins:
        if t_orig + 1 >= n_att: continue
        actual = att[t_orig + 1]; hist = att[:t_orig+1]
        for li, laa in enumerate(LAM_AA_GRID_EXT):
            pred = lwr_step_additive(hist, lam_va=0.0, lam_aa=laa)
            if np.isnan(pred): continue
            res_grid_aa[li][tk]["mae"].append(abs(pred - actual))
            res_grid_aa[li][tk]["sign"].append(1 if np.sign(pred) == np.sign(actual) else 0)
print(f"Grid λ_aa завершён за {time.time()-t0:.1f}с")

# найти оптимальный λ_aa по среднему rMAE
rmae_grid_aa = {}
for li in range(len(LAM_AA_GRID_EXT)):
    mean_r = []
    for tk in TICKERS:
        std_att = float(np.std(atts[tk]))
        mv = res_grid_aa[li][tk]["mae"]
        if mv: mean_r.append(float(np.mean(mv)) / std_att)
    rmae_grid_aa[li] = float(np.nanmean(mean_r)) if mean_r else np.nan

opt_li  = int(np.argmin([rmae_grid_aa[li] for li in range(len(LAM_AA_GRID_EXT))]))
opt_laa = LAM_AA_GRID_EXT[opt_li]
print(f"Оптимальный λ_aa = {opt_laa:.4f} (rMAE={rmae_grid_aa[opt_li]:.4f})")

# ── split-ratio grid ───────────────────────────────────────────────────────────

res_split = {si: {tk: {"mae": [], "sign": []} for tk in TICKERS}
             for si in range(len(SPLIT_RATIOS))}

t0 = time.time()
print(f"\nSplit-ratio grid (λ_va={LAM_VA_OPT}, λ_aa={opt_laa:.4f})...")
for tk in TICKERS:
    att = atts[tk]; n_att = len(att)
    origins = list(range(n_att - N_TEST, n_att - 1, STEP_WF))
    for t_orig in origins:
        if t_orig + 1 >= n_att: continue
        actual = att[t_orig + 1]; hist = att[:t_orig+1]
        for si, sr in enumerate(SPLIT_RATIOS):
            pred = lwr_step_split(hist, lam_va=LAM_VA_OPT, lam_aa=opt_laa, split_r=sr)
            if np.isnan(pred): continue
            res_split[si][tk]["mae"].append(abs(pred - actual))
            res_split[si][tk]["sign"].append(1 if np.sign(pred) == np.sign(actual) else 0)
print(f"Split-ratio grid завершён за {time.time()-t0:.1f}с")

# найти оптимальный split_r
rmae_split = {}; sacc_split = {}
for si in range(len(SPLIT_RATIOS)):
    mr = []; ms = []
    for tk in TICKERS:
        std_att = float(np.std(atts[tk]))
        mv = res_split[si][tk]["mae"]; sv = res_split[si][tk]["sign"]
        if mv: mr.append(float(np.mean(mv)) / std_att); ms.append(float(np.mean(sv)) * 100)
    rmae_split[si] = float(np.nanmean(mr)) if mr else np.nan
    sacc_split[si] = float(np.nanmean(ms)) if ms else np.nan

opt_si    = int(np.argmin([rmae_split[si] for si in range(len(SPLIT_RATIOS))]))
opt_split = SPLIT_RATIOS[opt_si]
print(f"Оптимальный split_r = {opt_split} (rMAE={rmae_split[opt_si]:.4f})")

# ── финальные 8 режимов walk-forward ──────────────────────────────────────────

MODES_FINAL = [
    ("pos-only",             dict(lam_va=0.0,        lam_aa=0.0,        split_r=1.0, glob=0)),
    ("vel_ang(0.01)",        dict(lam_va=LAM_VA_OPT, lam_aa=0.0,        split_r=1.0, glob=0)),
    ("acc_ang(opt)",         dict(lam_va=0.0,        lam_aa=opt_laa,    split_r=0.0, glob=0)),
    ("acc_ang(auto)",        dict(lam_va=0.0,        lam_aa=auto_lam_aa,split_r=0.0, glob=0)),
    (f"split(r={opt_split})",dict(lam_va=LAM_VA_OPT, lam_aa=opt_laa,    split_r=opt_split, glob=0)),
    ("split(r=0.5)",         dict(lam_va=LAM_VA_OPT, lam_aa=opt_laa,    split_r=0.5, glob=0)),
    (f"acc_ang+glob{GLOBAL_PCT}%", dict(lam_va=0.0, lam_aa=opt_laa,    split_r=0.0, glob=GLOBAL_PCT)),
    (f"split(opt)+glob{GLOBAL_PCT}%", dict(lam_va=LAM_VA_OPT, lam_aa=opt_laa, split_r=opt_split, glob=GLOBAL_PCT)),
]

res_final = {mi: {tk: {"mae": [], "sign": []} for tk in TICKERS}
             for mi in range(len(MODES_FINAL))}

t0 = time.time()
print(f"\nФинальный walk-forward ({len(MODES_FINAL)} режимов)...")
for tk in TICKERS:
    att = atts[tk]; n_att = len(att)
    origins = list(range(n_att - N_TEST, n_att - 1, STEP_WF))
    print(f"  {tk}: {len(origins)} origins", flush=True)
    for t_orig in origins:
        if t_orig + 1 >= n_att: continue
        actual = att[t_orig + 1]; hist = att[:t_orig+1]
        for mi, (name, p) in enumerate(MODES_FINAL):
            if p["glob"] > 0:
                pred = lwr_step_global_blend(hist, p["lam_va"], p["lam_aa"], p["split_r"], p["glob"])
            else:
                pred = lwr_step_split(hist, p["lam_va"], p["lam_aa"], p["split_r"])
            if np.isnan(pred): continue
            res_final[mi][tk]["mae"].append(abs(pred - actual))
            res_final[mi][tk]["sign"].append(1 if np.sign(pred) == np.sign(actual) else 0)
print(f"Финальный walk-forward завершён за {time.time()-t0:.1f}с")

# ── агрегация ──────────────────────────────────────────────────────────────────

def agg_r(res_d):
    out = {}
    for li in res_d:
        mr = []; ms = []
        for tk in TICKERS:
            std_att = float(np.std(atts[tk]))
            mv = res_d[li][tk]["mae"]; sv = res_d[li][tk]["sign"]
            if mv: mr.append(float(np.mean(mv)) / std_att); ms.append(float(np.mean(sv)) * 100)
        out[li] = (float(np.nanmean(mr)) if mr else np.nan,
                   float(np.nanmean(ms)) if ms else np.nan)
    return out

final_agg = agg_r(res_final)
base_rmae = final_agg[0][0]

print("\nФинальная сводка:")
for mi, (name, _) in enumerate(MODES_FINAL):
    mr, ms = final_agg[mi]
    d = (mr - base_rmae) / base_rmae * 100
    pref = "+" if d >= 0 else ""
    print(f"  {name:30s}: rMAE={mr:.4f} ({pref}{d:.2f}%)  SignAcc={ms:.1f}%")

# ── графики ────────────────────────────────────────────────────────────────────

# A: rMAE vs split_r ────────────────────────────────────────────────────────────

fig_a, ax = plt.subplots(figsize=(10, 5))
ax.set_title(f"80-A: rMAE vs split_r  (λ_va={LAM_VA_OPT}, λ_aa={opt_laa:.4f})", fontsize=11)
mr_vals = [rmae_split[si] for si in range(len(SPLIT_RATIOS))]
ax.plot(SPLIT_RATIOS, mr_vals, color="black", lw=2.0, marker="D", ms=7, label="split-budget СРЕДНЕЕ")
ax.axhline(0.0572, color="#d62728",  lw=1.0, ls="--", label="vel_ang(0.01) скр.78")
ax.axhline(final_agg[2][0], color="#2ca02c", lw=1.0, ls="--", label=f"acc_ang(opt={opt_laa:.3f})")
ax.axhline(base_rmae, color="#1f77b4", lw=0.8, ls=":", label="pos-only")
for i, tk in enumerate(TICKERS):
    vals = []
    for si in range(len(SPLIT_RATIOS)):
        std_att = float(np.std(atts[tk]))
        mv = res_split[si][tk]["mae"]
        vals.append(float(np.mean(mv)) / std_att if mv else np.nan)
    ax.plot(SPLIT_RATIOS, vals, color=COLORS_TK[i], lw=0.8, alpha=0.35)
for x, v in zip(SPLIT_RATIOS, mr_vals):
    ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
ax.set_xlabel("split_r = n_vel / ξ  (0=чистый acc, 1=чистый vel)"); ax.set_ylabel("rMAE")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig_a.tight_layout(); fig_a.savefig(FIGDIR / "80_A_split_rmae.png", dpi=130); plt.close(fig_a)
print("Рис. A сохранён")

# B: SignAcc vs split_r ────────────────────────────────────────────────────────

fig_b, ax = plt.subplots(figsize=(10, 5))
ax.set_title(f"80-B: SignAcc vs split_r", fontsize=11)
ms_vals = [sacc_split[si] for si in range(len(SPLIT_RATIOS))]
ax.plot(SPLIT_RATIOS, ms_vals, color="black", lw=2.0, marker="D", ms=7, label="split-budget СРЕДНЕЕ")
ax.axhline(91.6, color="#d62728",  lw=1.0, ls="--", label="vel_ang(0.01)")
ax.axhline(final_agg[2][1], color="#2ca02c", lw=1.0, ls="--", label=f"acc_ang(opt)")
ax.axhline(91.6, color="#1f77b4", lw=0.8, ls=":", label="pos-only")
for x, v in zip(SPLIT_RATIOS, ms_vals):
    ax.annotate(f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
ax.set_xlabel("split_r"); ax.set_ylabel("SignAcc %")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig_b.tight_layout(); fig_b.savefig(FIGDIR / "80_B_split_signacc.png", dpi=130); plt.close(fig_b)
print("Рис. B сохранён")

# C: финальная сводная таблица ─────────────────────────────────────────────────

mode_names = [m[0] for m in MODES_FINAL]
mr_fin = [final_agg[mi][0] for mi in range(len(MODES_FINAL))]
ms_fin = [final_agg[mi][1] for mi in range(len(MODES_FINAL))]

# цвет баров: лучший rMAE — тёмно-зелёный, остальные — серые с градиентом
max_drop = min(mr_fin)
cols = ["#4CAF50" if v == max_drop else "#90a4ae" for v in mr_fin]

fig_c, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig_c.suptitle("80-C: Итоговая сводка методов отбора соседей (все скр.77-80)", fontsize=11)

bars1 = ax1.bar(mode_names, mr_fin, color=cols, width=0.6, alpha=0.9)
ax1.set_ylabel("rMAE"); ax1.set_title("rMAE (меньше лучше)")
for bar, v in zip(bars1, mr_fin):
    d = (v - base_rmae) / base_rmae * 100
    pref = "+" if d >= 0 else ""
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.0002,
             f"{v:.4f}\n({pref}{d:.1f}%)", ha="center", va="bottom", fontsize=8)
ax1.set_ylim(min(mr_fin)*0.984, max(mr_fin)*1.015)
ax1.set_xticks(range(len(mode_names))); ax1.set_xticklabels(mode_names, rotation=25, ha="right", fontsize=8)
ax1.grid(axis="y", alpha=0.3)

cols2 = ["#4CAF50" if v == max(ms_fin) else "#90a4ae" for v in ms_fin]
bars2 = ax2.bar(mode_names, ms_fin, color=cols2, width=0.6, alpha=0.9)
ax2.set_ylabel("SignAcc %"); ax2.set_title("SignAcc % (больше лучше)")
for bar, v in zip(bars2, ms_fin):
    d = v - ms_fin[0]
    pref = "+" if d >= 0 else ""
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
             f"{v:.1f}%\n({pref}{d:.1f}pp)", ha="center", va="bottom", fontsize=8)
ax2.set_ylim(min(ms_fin)*0.995, max(ms_fin)*1.005)
ax2.set_xticks(range(len(mode_names))); ax2.set_xticklabels(mode_names, rotation=25, ha="right", fontsize=8)
ax2.grid(axis="y", alpha=0.3)

fig_c.tight_layout(); fig_c.savefig(FIGDIR / "80_C_final_summary.png", dpi=130, bbox_inches="tight")
plt.close(fig_c)
print("Рис. C сохранён")

# D: grid λ_aa для acc_ang_only ────────────────────────────────────────────────

x_ticks = list(range(len(LAM_AA_GRID_EXT)))
fig_d, ax = plt.subplots(figsize=(12, 5))
ax.set_title("80-D: rMAE vs λ_aa  [pos + λ_aa·d_ang(acc), без vel]", fontsize=11)
for i, tk in enumerate(TICKERS):
    vals = []
    for li in range(len(LAM_AA_GRID_EXT)):
        std_att = float(np.std(atts[tk]))
        mv = res_grid_aa[li][tk]["mae"]
        vals.append(float(np.mean(mv)) / std_att if mv else np.nan)
    ax.plot(x_ticks, vals, color=COLORS_TK[i], lw=0.9, marker="o", ms=3, alpha=0.55, label=tk)
mean_r2 = [rmae_grid_aa[li] for li in range(len(LAM_AA_GRID_EXT))]
ax.plot(x_ticks, mean_r2, color="black", lw=2.2, marker="D", ms=7, label="СРЕДНЕЕ")
ax.axhline(0.0572, color="crimson", lw=1.0, ls="--", label="скр.78 vel_ang(0.01)")
ax.set_xticks(x_ticks); ax.set_xticklabels(LAM_AA_LABELS, fontsize=7)
ax.set_xlabel("λ_aa"); ax.set_ylabel("rMAE")
for xi, v in zip(x_ticks, mean_r2):
    ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=0.3)
fig_d.tight_layout(); fig_d.savefig(FIGDIR / "80_D_grid_laa.png", dpi=130); plt.close(fig_d)
print("Рис. D сохранён")

# E: per-ticker heatmap (delta rMAE) ───────────────────────────────────────────

# выбираем 6 лучших методов по среднему rMAE
sorted_mi = sorted(range(len(MODES_FINAL)), key=lambda i: final_agg[i][0])
top_mi    = sorted_mi[:6]
top_names = [MODES_FINAL[mi][0] for mi in top_mi]

# матрица delta rMAE (%)
matrix = np.zeros((len(top_mi), len(TICKERS)))
for row, mi in enumerate(top_mi):
    for col, tk in enumerate(TICKERS):
        std_att = float(np.std(atts[tk]))
        mv_base = res_final[0][tk]["mae"]
        mv      = res_final[mi][tk]["mae"]
        if mv and mv_base:
            r_base = float(np.mean(mv_base)) / std_att
            r_mi   = float(np.mean(mv)) / std_att
            matrix[row, col] = (r_mi - r_base) / r_base * 100

fig_e, ax = plt.subplots(figsize=(12, 4))
ax.set_title("80-E: Δ rMAE от baseline (%) — per-ticker, топ-6 методов", fontsize=11)
im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
               vmin=min(-15, matrix.min()), vmax=max(5, matrix.max()))
plt.colorbar(im, ax=ax, label="Δ rMAE (%)")
ax.set_xticks(range(len(TICKERS))); ax.set_xticklabels(TICKERS, fontsize=9)
ax.set_yticks(range(len(top_names))); ax.set_yticklabels(top_names, fontsize=9)
for row in range(len(top_mi)):
    for col in range(len(TICKERS)):
        val = matrix[row, col]
        ax.text(col, row, f"{val:.1f}%", ha="center", va="center", fontsize=8,
                color="white" if abs(val) > 8 else "black")
fig_e.tight_layout(); fig_e.savefig(FIGDIR / "80_E_heatmap.png", dpi=130); plt.close(fig_e)
print("Рис. E сохранён")

print("\nГотово. Фигуры: 80_A...E.png")
