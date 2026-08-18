"""
78 — Угловой фильтр скорости (cosine dissimilarity).

Гипотеза: косинусное расстояние между vel-векторами соседей и запроса
лучше, чем евклидово (скр.77), потому что фильтрует по НАПРАВЛЕНИЮ
траектории, а не по комбинации направления и амплитуды. На аттракторе
траектории разной «скорости», но одного направления — ближайшие соседи.

Метрики:
  d_pos  = ||Δatt[-p_fit:]||          (позиция, евклидово)
  d_vel  = ||Δvel[-p_vel:]||          (скорость, евклидово — из скр.77)
  d_ang  = 1 - cos_sim(vel_q, vel_i)  (угловое расстояние, [0, 2])
         cos_sim = dot / (||q||·||i||)

Сравниваемые режимы (финальный отбор после каскада):
  0. pos-only       : d = d_pos
  1. pos+vel_eucl   : d = d_pos + λ_v·d_vel   (λ_v = auto ≈ 2.33 из скр.77)
  2. pos+vel_ang    : d = d_pos + λ_a·d_ang    (grid λ_a)
  3. pos+vel_e+a    : d = d_pos + λ_v·d_vel + λ_a·d_ang  (оба на auto-λ)

auto-λ_ang: выравнивает медианный вклад d_pos и d_ang по пулу кандидатов
  → λ_a_auto = median(d_pos_pool) / median(d_ang_pool)  (per-ticker)

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров (как скр.76-77).
Att-фильтр: Local Projective (m=9, d=3, k=30, n=3) — стандарт.

Графики:
  A — att / vel / instantaneous angle (SBER, 150 баров)
  B — rMAE vs λ_ang (pos+vel_ang)
  C — SignAcc vs λ_ang
  D — Scatter d_pos vs d_ang для пула каскада (SBER);
      отмечены pos-only, pos+vel_eucl_auto, pos+vel_ang_auto
  E — Сводная таблица: 4 режима × rMAE + SignAcc
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
P_VEL     = P_FIT
XI_LWR    = 3 * (P_FIT + 1)
STEP_BASE = 2.0
LAM_V_AUTO = 2.33        # из скр.77

N_TEST  = 200
STEP_WF = 5

LAM_ANG_GRID   = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
LAM_ANG_LABELS = ["0\n(pos)", "0.005", "0.01", "0.02", "0.05",
                   "0.1", "0.2", "0.5", "1.0", "2.0", "auto"]
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
    """1 - cos_sim между строками A и вектором b; nan → 1.0."""
    norm_A = np.linalg.norm(A, axis=1)
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-12:
        return np.ones(len(A))
    dot = A @ b
    denom = norm_A * norm_b
    cos = np.where(denom > 1e-12, dot / denom, 0.0)
    cos = np.clip(cos, -1.0, 1.0)
    return 1.0 - cos


def _cascade_pool(att_hist: np.ndarray):
    """Возвращает (cands, X_pos, X_vel, y_base, vec_pos, vec_vel, m)
    после полного прогона каскада (последний уровень НЕ фильтрует)."""
    n  = len(att_hist)
    m  = n - P_MAX - 1
    if m < XI_LWR + 5:
        return None

    vel_hist = np.concatenate([[0.0], np.diff(att_hist)])
    t_arr    = np.arange(P_MAX, n - 1)
    X_pos    = np.column_stack([att_hist[t_arr - (P_MAX-1-j)] for j in range(P_MAX)])
    X_vel    = np.column_stack([vel_hist[t_arr - (P_VEL-1-j)] for j in range(P_VEL)])
    y_base   = att_hist[t_arr + 1]
    vec_pos  = att_hist[-P_MAX:]
    vec_vel  = vel_hist[-P_VEL:]

    levels = octave_levels(P_FIT, P_MAX, STEP_BASE)
    cands  = np.arange(m)

    for k, p_lvl in enumerate(levels):
        is_last = (k == len(levels) - 1)
        if not is_last:
            xi_lvl = min(XI_LWR, len(cands))
            if len(cands) > xi_lvl:
                d = np.linalg.norm(X_pos[cands, -p_lvl:] - vec_pos[-p_lvl:], axis=1)
                cands = cands[np.argpartition(d, xi_lvl-1)[:xi_lvl]]
            p_next  = levels[k+1]
            radius  = p_lvl - p_next
            offsets = np.arange(radius + 1)
            exp     = cands[:, None] - offsets[None, :]
            cands   = np.unique(np.clip(exp, 0, m-1))

    return cands, X_pos, X_vel, y_base, vec_pos, vec_vel, m


def lwr_step_ang(att_hist: np.ndarray,
                 lam_ang: float,
                 lam_vel: float = 0.0,
                 auto_lam_ang: float | None = None
                 ) -> tuple[float, dict]:
    """h=1 LWR с d = d_pos + lam_vel·d_vel + lam_ang·d_ang.

    Если lam_ang='auto', используется auto_lam_ang.
    Возвращает (prediction, диагностика).
    """
    res = _cascade_pool(att_hist)
    if res is None:
        return np.nan, {}

    cands, X_pos, X_vel, y_base, vec_pos, vec_vel, m = res

    d_pos_all = np.linalg.norm(X_pos[cands, -P_FIT:] - vec_pos[-P_FIT:], axis=1)
    d_vel_all = np.linalg.norm(X_vel[cands, -P_VEL:] - vec_vel[-P_VEL:], axis=1)
    d_ang_all = cosine_dist(X_vel[cands, -P_VEL:], vec_vel[-P_VEL:])

    # auto-λ_ang: выравниваем медианы d_pos и d_ang
    if auto_lam_ang is None:
        med_ang = float(np.median(d_ang_all))
        med_pos = float(np.median(d_pos_all))
        auto_lam_ang = med_pos / (med_ang + 1e-12)

    lam_a = auto_lam_ang if lam_ang < 0 else lam_ang

    # pos-only
    if len(cands) > XI_LWR:
        sel0 = np.argpartition(d_pos_all, XI_LWR-1)[:XI_LWR]
        cands_pos = cands[sel0]
    else:
        cands_pos = cands.copy()

    if len(cands) < P_FIT + 2:
        return np.nan, {}

    # финальный отбор
    d_combined = d_pos_all.copy()
    if lam_vel > 0.0:
        d_combined = d_combined + lam_vel * d_vel_all
    if lam_a > 0.0:
        d_combined = d_combined + lam_a * d_ang_all

    if len(cands) > XI_LWR and (lam_vel > 0.0 or lam_a > 0.0):
        sel = np.argpartition(d_combined, XI_LWR-1)[:XI_LWR]
        cands_final = cands[sel]
    else:
        cands_final = cands_pos

    X_nn  = X_pos[cands_final, -P_FIT:]
    y_nn  = y_base[cands_final]
    vec_f = vec_pos[-P_FIT:]
    h_bw  = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
    val   = lwr_fit(X_nn, y_nn, vec_f, h_bw)

    diag = dict(
        cands_pool=cands, cands_pos=cands_pos, cands_final=cands_final,
        d_pos=d_pos_all, d_vel=d_vel_all, d_ang=d_ang_all,
        auto_lam_ang=auto_lam_ang,
    )
    return val, diag


# ── загрузка данных ────────────────────────────────────────────────────────────

print("Загрузка + Local Projective...")
atts: dict[str, np.ndarray] = {}
t_load = time.time()
for tk in TICKERS:
    print(f"  {tk}...", end=" ", flush=True)
    t1 = time.time(); atts[tk] = load_att(tk)
    print(f"{time.time()-t1:.1f}с")

# auto-λ_ang: вычисляем по SBER (репрезентативный)
att_sber_full = atts["SBER"]
n_demo = len(att_sber_full) - N_TEST // 2
_, diag_demo = lwr_step_ang(att_sber_full[:n_demo+1], lam_ang=0.0, auto_lam_ang=None)
auto_lam_ang_global = float(diag_demo.get("auto_lam_ang", 0.1))
print(f"auto λ_ang (SBER): {auto_lam_ang_global:.4f}")
print(f"Загрузка итого: {time.time()-t_load:.1f}с")

# полный grid: LAM_ANG_GRID + auto
LAM_ANG_ALL = LAM_ANG_GRID + [auto_lam_ang_global]
N_LAM = len(LAM_ANG_ALL)

# 4 режима для финальной сводки
MODES = [
    ("pos-only",        0.0,        0.0),
    ("pos+vel_eucl",    0.0,        LAM_V_AUTO),
    ("pos+vel_ang",     -1,         0.0),          # lam_ang=-1 → auto
    ("pos+e+a",         -1,         LAM_V_AUTO),   # оба auto
]

# ── walk-forward по grid λ_ang ─────────────────────────────────────────────────

# results_grid[lam_idx][ticker] = {"mae": [], "sign": []}
results_grid = {li: {tk: {"mae": [], "sign": []} for tk in TICKERS}
                for li in range(N_LAM)}

t0 = time.time()
print("\nWalk-forward (grid λ_ang)...")
for tk in TICKERS:
    att   = atts[tk]
    n_att = len(att)
    origins = list(range(n_att - N_TEST, n_att - 1, STEP_WF))
    print(f"  {tk}: {len(origins)} origins", flush=True)

    for t_orig in origins:
        if t_orig + 1 >= n_att:
            continue
        actual = att[t_orig + 1]
        hist   = att[:t_orig+1]

        for li, lam_a in enumerate(LAM_ANG_ALL):
            # pos + ang, без vel
            pred, _ = lwr_step_ang(hist, lam_ang=lam_a,
                                   lam_vel=0.0, auto_lam_ang=auto_lam_ang_global)
            if np.isnan(pred):
                continue
            results_grid[li][tk]["mae"].append(abs(pred - actual))
            results_grid[li][tk]["sign"].append(1 if np.sign(pred) == np.sign(actual) else 0)

print(f"Walk-forward grid завершён за {time.time()-t0:.1f}с")

# ── walk-forward по 4 режимам ──────────────────────────────────────────────────

results_modes = {mi: {tk: {"mae": [], "sign": []} for tk in TICKERS}
                 for mi in range(len(MODES))}

t0 = time.time()
print("\nWalk-forward (4 режима)...")
for tk in TICKERS:
    att   = atts[tk]
    n_att = len(att)
    origins = list(range(n_att - N_TEST, n_att - 1, STEP_WF))
    for t_orig in origins:
        if t_orig + 1 >= n_att:
            continue
        actual = att[t_orig + 1]
        hist   = att[:t_orig+1]
        for mi, (name, la, lv) in enumerate(MODES):
            pred, _ = lwr_step_ang(hist, lam_ang=la,
                                   lam_vel=lv, auto_lam_ang=auto_lam_ang_global)
            if np.isnan(pred):
                continue
            results_modes[mi][tk]["mae"].append(abs(pred - actual))
            results_modes[mi][tk]["sign"].append(1 if np.sign(pred) == np.sign(actual) else 0)

print(f"Walk-forward режимы завершён за {time.time()-t0:.1f}с")

# ── агрегация ──────────────────────────────────────────────────────────────────

def agg(results_d):
    rmae_d = {}; sacc_d = {}
    for li in results_d:
        rmae_d[li] = {}; sacc_d[li] = {}
        for tk in TICKERS:
            std_att = float(np.std(atts[tk]))
            mv = results_d[li][tk]["mae"]
            sv = results_d[li][tk]["sign"]
            rmae_d[li][tk] = float(np.mean(mv)) / std_att if mv else np.nan
            sacc_d[li][tk] = float(np.mean(sv)) * 100     if sv else np.nan
    return rmae_d, sacc_d

rmae_grid, sacc_grid = agg(results_grid)
rmae_modes, sacc_modes = agg(results_modes)

base_rmae = float(np.nanmean([rmae_grid[0][tk] for tk in TICKERS]))

print("\nrMAE / SignAcc по λ_ang (pos+vel_ang):")
for li, lbl in enumerate(LAM_ANG_LABELS):
    mr  = float(np.nanmean([rmae_grid[li][tk] for tk in TICKERS]))
    ms  = float(np.nanmean([sacc_grid[li][tk] for tk in TICKERS]))
    d   = (mr - base_rmae) / base_rmae * 100
    pref = "+" if d >= 0 else ""
    print(f"  λ_ang={lbl:6s}: rMAE={mr:.4f} ({pref}{d:.2f}%)  SignAcc={ms:.1f}%")

print("\nСводка 4 режима:")
for mi, (name, _, _) in enumerate(MODES):
    mr = float(np.nanmean([rmae_modes[mi][tk] for tk in TICKERS]))
    ms = float(np.nanmean([sacc_modes[mi][tk] for tk in TICKERS]))
    d  = (mr - base_rmae) / base_rmae * 100
    pref = "+" if d >= 0 else ""
    print(f"  {name:20s}: rMAE={mr:.4f} ({pref}{d:.2f}%)  SignAcc={ms:.1f}%")

# ── графики ────────────────────────────────────────────────────────────────────

# A: att, vel, cosine angle ────────────────────────────────────────────────────

SHOW = 150
att_s  = atts["SBER"]
vel_s  = np.concatenate([[0.0], np.diff(att_s)])
n_sber = len(att_s)
sl     = slice(n_sber - SHOW, n_sber)

# мгновенный угол между соседними vel-векторами
EPS = 1e-12
v1  = vel_s[:-1]; v2 = vel_s[1:]
cos_inst = v1 * v2 / (np.abs(v1) * np.abs(v2) + EPS)
cos_inst = np.clip(cos_inst, -1.0, 1.0)
ang_inst  = np.degrees(np.arccos(cos_inst))   # 0=то же направление, 180=разворот

fig_a, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
fig_a.suptitle("78-A: att / vel / angle (SBER, последние 150 баров)", fontsize=11)
axes[0].plot(att_s[sl],  color="seagreen",  lw=1.2, label="att"); axes[0].axhline(0, color="k", lw=0.4, ls="--"); axes[0].set_ylabel("att"); axes[0].legend(fontsize=8)
axes[1].plot(vel_s[sl],  color="steelblue", lw=1.0, label="vel = diff(att)"); axes[1].axhline(0, color="k", lw=0.4, ls="--"); axes[1].set_ylabel("vel"); axes[1].legend(fontsize=8)
axes[2].plot(ang_inst[(n_sber-SHOW):(n_sber-1)], color="darkorange", lw=0.9, label="инст. угол (°)")
axes[2].axhline(90, color="k", lw=0.4, ls="--"); axes[2].set_ylabel("угол, °"); axes[2].set_xlabel("бар"); axes[2].legend(fontsize=8)

fig_a.tight_layout(); fig_a.savefig(FIGDIR / "78_A_signals.png", dpi=130); plt.close(fig_a)
print("Рис. A сохранён")

# B: rMAE vs λ_ang ─────────────────────────────────────────────────────────────

x_ticks = list(range(N_LAM))
fig_b, ax = plt.subplots(figsize=(12, 5))
ax.set_title("78-B: rMAE vs λ_ang  [pos + λ_ang·d_ang(cosine)]", fontsize=11)
for i, tk in enumerate(TICKERS):
    vals = [rmae_grid[li][tk] for li in range(N_LAM)]
    ax.plot(x_ticks, vals, color=COLORS_TK[i], lw=0.9, marker="o", ms=3, alpha=0.55, label=tk)
mean_r = [float(np.nanmean([rmae_grid[li][tk] for tk in TICKERS])) for li in range(N_LAM)]
ax.plot(x_ticks, mean_r, color="black", lw=2.2, marker="D", ms=7, label="СРЕДНЕЕ")
# baseline из скр.77 (pos+vel_eucl auto)
ax.axhline(0.0616, color="crimson", lw=1.0, ls="--", label="скр.77 pos+vel_eucl auto")
ax.set_xticks(x_ticks); ax.set_xticklabels(LAM_ANG_LABELS, fontsize=8)
ax.set_xlabel("λ_ang"); ax.set_ylabel("rMAE")
for xi, v in zip(x_ticks, mean_r):
    ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=0.3)
fig_b.tight_layout(); fig_b.savefig(FIGDIR / "78_B_rmae.png", dpi=130); plt.close(fig_b)
print("Рис. B сохранён")

# C: SignAcc vs λ_ang ──────────────────────────────────────────────────────────

fig_c, ax = plt.subplots(figsize=(12, 5))
ax.set_title("78-C: SignAcc vs λ_ang  [pos + λ_ang·d_ang(cosine)]", fontsize=11)
for i, tk in enumerate(TICKERS):
    vals = [sacc_grid[li][tk] for li in range(N_LAM)]
    ax.plot(x_ticks, vals, color=COLORS_TK[i], lw=0.9, marker="o", ms=3, alpha=0.55, label=tk)
mean_s = [float(np.nanmean([sacc_grid[li][tk] for tk in TICKERS])) for li in range(N_LAM)]
ax.plot(x_ticks, mean_s, color="black", lw=2.2, marker="D", ms=7, label="СРЕДНЕЕ")
ax.axhline(92.2, color="crimson", lw=1.0, ls="--", label="скр.77 pos+vel_eucl auto")
ax.set_xticks(x_ticks); ax.set_xticklabels(LAM_ANG_LABELS, fontsize=8)
ax.set_xlabel("λ_ang"); ax.set_ylabel("SignAcc %")
for xi, v in zip(x_ticks, mean_s):
    ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=0.3)
fig_c.tight_layout(); fig_c.savefig(FIGDIR / "78_C_signacc.png", dpi=130); plt.close(fig_c)
print("Рис. C сохранён")

# D: Scatter d_pos vs d_ang (SBER) ────────────────────────────────────────────

_, diag = lwr_step_ang(att_sber_full[:n_demo+1], lam_ang=-1,
                       lam_vel=0.0, auto_lam_ang=auto_lam_ang_global)
cands_pool = diag["cands_pool"]
cands_pos  = diag["cands_pos"]
# pos+ang auto
_, diag_a = lwr_step_ang(att_sber_full[:n_demo+1], lam_ang=-1,
                         lam_vel=0.0, auto_lam_ang=auto_lam_ang_global)
cands_ang  = diag_a["cands_final"]
# pos+vel_eucl auto (lam_vel=2.33, lam_ang=0)
_, diag_v = lwr_step_ang(att_sber_full[:n_demo+1], lam_ang=0.0,
                         lam_vel=LAM_V_AUTO, auto_lam_ang=auto_lam_ang_global)
cands_vel  = diag_v["cands_final"]

d_pos_pool = diag["d_pos"]
d_ang_pool = diag["d_ang"]

fig_d, ax = plt.subplots(figsize=(8, 6))
ax.set_title(f"78-D: Scatter d_pos vs d_ang (SBER, пул каскада={len(cands_pool)})", fontsize=11)
ax.scatter(d_pos_pool, d_ang_pool, s=12, color="silver", alpha=0.5, label=f"весь пул ({len(cands_pool)})")

# pos-only
mask_pos = np.isin(cands_pool, cands_pos)
ax.scatter(d_pos_pool[mask_pos], d_ang_pool[mask_pos], s=35, color="#1f77b4",
           alpha=0.8, label=f"pos-only ({mask_pos.sum()})", zorder=3)

# pos+vel_eucl auto
mask_vel = np.isin(cands_pool, cands_vel)
ax.scatter(d_pos_pool[mask_vel], d_ang_pool[mask_vel], s=35, color="crimson",
           alpha=0.8, marker="^", label=f"pos+vel_eucl auto ({mask_vel.sum()})", zorder=4)

# pos+ang auto
mask_ang = np.isin(cands_pool, cands_ang)
ax.scatter(d_pos_pool[mask_ang], d_ang_pool[mask_ang], s=35, color="#2ca02c",
           alpha=0.8, marker="s", label=f"pos+ang auto ({mask_ang.sum()})", zorder=5)

ax.set_xlabel("d_pos (евклидово)"); ax.set_ylabel("d_ang (cosine dissimilarity)")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig_d.tight_layout(); fig_d.savefig(FIGDIR / "78_D_scatter.png", dpi=130); plt.close(fig_d)
print("Рис. D сохранён")

# E: Сводная таблица 4 режима ─────────────────────────────────────────────────

mode_names = [m[0] for m in MODES]
mr_modes = [float(np.nanmean([rmae_modes[mi][tk] for tk in TICKERS])) for mi in range(len(MODES))]
ms_modes = [float(np.nanmean([sacc_modes[mi][tk] for tk in TICKERS])) for mi in range(len(MODES))]
base_r = mr_modes[0]; base_s = ms_modes[0]

fig_e, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig_e.suptitle("78-E: Сводка 4 режимов (8 тикеров, walk-forward)", fontsize=11)

colors_mode = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

bars1 = ax1.bar(mode_names, mr_modes, color=colors_mode, width=0.5, alpha=0.8)
ax1.set_ylabel("rMAE"); ax1.set_title("rMAE (меньше лучше)")
for bar, v in zip(bars1, mr_modes):
    d = (v - base_r) / base_r * 100
    pref = "+" if d >= 0 else ""
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.0003, f"{v:.4f}\n({pref}{d:.1f}%)",
             ha="center", va="bottom", fontsize=9)
ax1.set_ylim(min(mr_modes)*0.985, max(mr_modes)*1.02)
ax1.set_xticklabels(mode_names, rotation=15, ha="right", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

bars2 = ax2.bar(mode_names, ms_modes, color=colors_mode, width=0.5, alpha=0.8)
ax2.set_ylabel("SignAcc %"); ax2.set_title("SignAcc % (больше лучше)")
for bar, v in zip(bars2, ms_modes):
    d = v - base_s
    pref = "+" if d >= 0 else ""
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.1f}%\n({pref}{d:.1f}pp)",
             ha="center", va="bottom", fontsize=9)
ax2.set_ylim(min(ms_modes)*0.995, max(ms_modes)*1.005)
ax2.set_xticklabels(mode_names, rotation=15, ha="right", fontsize=9)
ax2.grid(axis="y", alpha=0.3)

# per-ticker таблица под графиком
per_ticker_text = []
for tk in TICKERS:
    row = f"{tk}: "
    parts = []
    for mi, (name, _, _) in enumerate(MODES):
        r = rmae_modes[mi][tk]; s = sacc_modes[mi][tk]
        parts.append(f"{name[:8]}={r:.4f}/{s:.0f}%")
    row += "  ".join(parts)
    per_ticker_text.append(row)
fig_e.text(0.01, -0.02, "\n".join(per_ticker_text), fontsize=7,
           family="monospace", va="top")

fig_e.tight_layout(); fig_e.savefig(FIGDIR / "78_E_summary.png", dpi=130, bbox_inches="tight")
plt.close(fig_e)
print("Рис. E сохранён")

print("\nГотово. Фигуры: 78_A...E.png")
