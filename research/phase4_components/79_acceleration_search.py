"""
79 — Поиск с учётом ускорения (2-я производная аттрактора).

Итог скр.78: угловое расстояние по скорости (-8.5% rMAE при λ=0.01) >>
евклидово (-1.4%). Направление траектории важнее амплитуды.

Гипотеза: добавление кривизны траектории (acc = diff(vel)) дополнительно
уточняет отбор соседей. «То же направление И та же кривизна» — ещё
точнее соответствует фазовому портрету аттрактора.

acc[t] = vel[t] - vel[t-1] = att[t] - 2·att[t-1] + att[t-2]  (каузальная)

Тестируемые метрики ускорения:
  d_acc_eucl = ||acc_q - acc_i||        (евклидово)
  d_acc_ang  = 1 - cos_sim(acc_q, acc_i) (угловое, как vel в скр.78)

Режимы:
  0. baseline       : pos-only
  1. vel_ang_opt    : pos + λ_va·d_ang(vel)          λ_va=0.01 (оптимум скр.78)
  2. + acc_eucl     : pos + λ_va·d_ang(vel) + λ_ae·d_eucl(acc)
  3. + acc_ang      : pos + λ_va·d_ang(vel) + λ_aa·d_ang(acc)
  4. pos + acc_ang  : pos + λ_aa·d_ang(acc)           (acc без vel, контроль)

auto-λ для acc компонент: та же формула, что в скр.78 (выравнивание медиан).

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров.
Att-фильтр: Local Projective (m=9, d=3, k=30, n=3) — стандарт.

Графики:
  A — att / vel / acc (SBER, 150 баров): характер 3 сигналов
  B — rMAE по режимам (bar chart, 8 тикеров + среднее)
  C — SignAcc по режимам
  D — grid λ_aa (pos + vel_ang_opt + acc_ang): оптимальный λ_aa
  E — Scatter d_ang(vel) vs d_ang(acc) для пула (SBER);
      выделены соседи каждого режима
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
LAM_VA_OPT = 0.01      # оптимум скр.78 для vel_ang

N_TEST  = 200
STEP_WF = 5

COLORS_TK = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
             "#9467bd","#8c564b","#e377c2","#17becf"]

# grid для λ_aa (угловое ускорение)
LAM_AA_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

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
    """1 - cos_sim(строки A, вектор b); при нулевых нормах → 1.0."""
    norm_A = np.linalg.norm(A, axis=1)
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-12:
        return np.ones(len(A))
    dot   = A @ b
    denom = norm_A * norm_b
    cos   = np.where(denom > 1e-12, dot / denom, 0.0)
    return 1.0 - np.clip(cos, -1.0, 1.0)


def _build_matrices(att_hist: np.ndarray):
    """Строит все матрицы задержек + проводит каскад.
    Возвращает словарь данных либо None при недостатке истории.
    """
    n  = len(att_hist)
    m  = n - P_MAX - 1
    if m < XI_LWR + 5:
        return None

    vel_hist = np.concatenate([[0.0], np.diff(att_hist)])
    acc_hist = np.concatenate([[0.0, 0.0], np.diff(vel_hist[1:])])   # д.б. длина n

    t_arr   = np.arange(P_MAX, n - 1)
    X_pos   = np.column_stack([att_hist[t_arr - (P_FIT-1-j)] for j in range(P_FIT)])
    X_pos_w = np.column_stack([att_hist[t_arr - (P_MAX-1-j)] for j in range(P_MAX)])
    X_vel   = np.column_stack([vel_hist[t_arr - (P_FIT-1-j)] for j in range(P_FIT)])
    X_acc   = np.column_stack([acc_hist[t_arr - (P_FIT-1-j)] for j in range(P_FIT)])
    y_base  = att_hist[t_arr + 1]

    vec_pos  = att_hist[-P_FIT:]
    vec_posw = att_hist[-P_MAX:]
    vec_vel  = vel_hist[-P_FIT:]
    vec_acc  = acc_hist[-P_FIT:]

    # Каскад по позиции (ширина P_MAX для уровней, P_FIT для финала)
    levels = octave_levels(P_FIT, P_MAX, STEP_BASE)
    cands  = np.arange(m)

    for k, p_lvl in enumerate(levels):
        is_last = (k == len(levels) - 1)
        if not is_last:
            xi_lvl = min(XI_LWR, len(cands))
            if len(cands) > xi_lvl:
                d = np.linalg.norm(X_pos_w[cands, -p_lvl:] - vec_posw[-p_lvl:], axis=1)
                cands = cands[np.argpartition(d, xi_lvl-1)[:xi_lvl]]
            p_next  = levels[k+1]
            radius  = p_lvl - p_next
            offsets = np.arange(radius + 1)
            exp     = cands[:, None] - offsets[None, :]
            cands   = np.unique(np.clip(exp, 0, m-1))

    return dict(
        cands=cands, m=m,
        X_pos=X_pos, X_vel=X_vel, X_acc=X_acc, y_base=y_base,
        vec_pos=vec_pos, vec_vel=vec_vel, vec_acc=vec_acc,
        vel_hist=vel_hist, acc_hist=acc_hist,
    )


def lwr_step(att_hist: np.ndarray,
             lam_va: float = 0.0,
             lam_ae: float = 0.0,
             lam_aa: float = 0.0,
             ) -> float:
    """LWR h=1 с метрикой d_pos + lam_va·d_ang(vel) + lam_ae·d_eucl(acc) + lam_aa·d_ang(acc)."""
    data = _build_matrices(att_hist)
    if data is None:
        return np.nan

    cands   = data["cands"]
    X_pos   = data["X_pos"]; X_vel = data["X_vel"]; X_acc = data["X_acc"]
    y_base  = data["y_base"]
    vec_pos = data["vec_pos"]; vec_vel = data["vec_vel"]; vec_acc = data["vec_acc"]

    d_pos = np.linalg.norm(X_pos[cands] - vec_pos, axis=1)

    d_combined = d_pos.copy()
    if lam_va > 0.0:
        d_combined = d_combined + lam_va * cosine_dist(X_vel[cands], vec_vel)
    if lam_ae > 0.0:
        d_combined = d_combined + lam_ae * np.linalg.norm(X_acc[cands] - vec_acc, axis=1)
    if lam_aa > 0.0:
        d_combined = d_combined + lam_aa * cosine_dist(X_acc[cands], vec_acc)

    if len(cands) > XI_LWR and (lam_va > 0 or lam_ae > 0 or lam_aa > 0):
        sel   = np.argpartition(d_combined, XI_LWR-1)[:XI_LWR]
        cands = cands[sel]
    elif len(cands) > XI_LWR:
        sel   = np.argpartition(d_pos, XI_LWR-1)[:XI_LWR]
        cands = cands[sel]

    if len(cands) < P_FIT + 2:
        return np.nan

    X_nn = X_pos[cands]; y_nn = y_base[cands]
    h_bw = max(float(np.linalg.norm(X_nn - vec_pos, axis=1).max()), 1e-10)
    return lwr_fit(X_nn, y_nn, vec_pos, h_bw)


def lwr_step_diag(att_hist: np.ndarray,
                  lam_va: float = 0.0, lam_aa: float = 0.0):
    """Как lwr_step, но возвращает диагностические дистанции для графика E."""
    data = _build_matrices(att_hist)
    if data is None:
        return np.nan, {}

    cands   = data["cands"]
    X_pos   = data["X_pos"]; X_vel = data["X_vel"]; X_acc = data["X_acc"]
    y_base  = data["y_base"]
    vec_pos = data["vec_pos"]; vec_vel = data["vec_vel"]; vec_acc = data["vec_acc"]

    d_pos = np.linalg.norm(X_pos[cands] - vec_pos, axis=1)
    d_va  = cosine_dist(X_vel[cands], vec_vel)
    d_aa  = cosine_dist(X_acc[cands], vec_acc)

    # pos-only
    if len(cands) > XI_LWR:
        sel_pos   = np.argpartition(d_pos, XI_LWR-1)[:XI_LWR]
        cands_pos = cands[sel_pos]
    else:
        cands_pos = cands.copy()

    # vel_ang_opt
    d_comb_v = d_pos + lam_va * d_va
    if len(cands) > XI_LWR:
        sel_v   = np.argpartition(d_comb_v, XI_LWR-1)[:XI_LWR]
        cands_v = cands[sel_v]
    else:
        cands_v = cands.copy()

    # vel_ang_opt + acc_ang
    d_comb_va = d_pos + lam_va * d_va + lam_aa * d_aa
    if len(cands) > XI_LWR:
        sel_va   = np.argpartition(d_comb_va, XI_LWR-1)[:XI_LWR]
        cands_va = cands[sel_va]
    else:
        cands_va = cands.copy()

    diag = dict(
        cands_pool=cands, cands_pos=cands_pos, cands_v=cands_v, cands_va=cands_va,
        d_va_pool=d_va, d_aa_pool=d_aa,
    )

    X_nn = X_pos[cands_va]; y_nn = y_base[cands_va]
    h_bw = max(float(np.linalg.norm(X_nn - vec_pos, axis=1).max()), 1e-10)
    return lwr_fit(X_nn, y_nn, vec_pos, h_bw), diag


# ── загрузка данных ────────────────────────────────────────────────────────────

print("Загрузка + Local Projective...")
atts: dict[str, np.ndarray] = {}
t_load = time.time()
for tk in TICKERS:
    print(f"  {tk}...", end=" ", flush=True)
    t1 = time.time(); atts[tk] = load_att(tk)
    print(f"{time.time()-t1:.1f}с")
print(f"Загрузка итого: {time.time()-t_load:.1f}с")

# ── auto-λ для acc (по SBER) ───────────────────────────────────────────────────

att_sber = atts["SBER"]
n_demo   = len(att_sber) - N_TEST // 2
data_demo = _build_matrices(att_sber[:n_demo+1])
cands_d   = data_demo["cands"]
d_pos_d   = np.linalg.norm(data_demo["X_pos"][cands_d] - data_demo["vec_pos"], axis=1)
d_aa_d    = cosine_dist(data_demo["X_acc"][cands_d], data_demo["vec_acc"])
med_d_pos = float(np.median(d_pos_d)); med_d_aa = float(np.median(d_aa_d))
auto_lam_aa = med_d_pos / (med_d_aa + 1e-12)
print(f"auto λ_aa (SBER): {auto_lam_aa:.4f}  (med_pos={med_d_pos:.5f}, med_aa={med_d_aa:.4f})")

LAM_AA_ALL = LAM_AA_GRID + [auto_lam_aa]
LAM_AA_LABELS = [str(x) for x in LAM_AA_GRID] + [f"auto\n({auto_lam_aa:.3f})"]

# ── 5 режимов walk-forward ─────────────────────────────────────────────────────

MODES = [
    ("pos-only",       0.0,         0.0,    0.0),
    ("vel_ang_opt",    LAM_VA_OPT,  0.0,    0.0),
    ("+acc_eucl_auto", LAM_VA_OPT,  auto_lam_aa, 0.0),
    ("+acc_ang_auto",  LAM_VA_OPT,  0.0,    auto_lam_aa),
    ("acc_ang_only",   0.0,         0.0,    auto_lam_aa),
]
N_MODES = len(MODES)

results_modes = {mi: {tk: {"mae": [], "sign": []} for tk in TICKERS}
                 for mi in range(N_MODES)}

t0 = time.time()
print("\nWalk-forward (5 режимов)...")
for tk in TICKERS:
    att   = atts[tk]
    n_att = len(att)
    origins = list(range(n_att - N_TEST, n_att - 1, STEP_WF))
    print(f"  {tk}: {len(origins)} origins", flush=True)
    for t_orig in origins:
        if t_orig + 1 >= n_att: continue
        actual = att[t_orig + 1]
        hist   = att[:t_orig+1]
        for mi, (name, lva, lae, laa) in enumerate(MODES):
            pred = lwr_step(hist, lam_va=lva, lam_ae=lae, lam_aa=laa)
            if np.isnan(pred): continue
            results_modes[mi][tk]["mae"].append(abs(pred - actual))
            results_modes[mi][tk]["sign"].append(1 if np.sign(pred) == np.sign(actual) else 0)
print(f"Walk-forward режимы завершён за {time.time()-t0:.1f}с")

# ── grid λ_aa walk-forward (pos + vel_ang_opt + acc_ang) ───────────────────────

results_grid = {li: {tk: {"mae": [], "sign": []} for tk in TICKERS}
                for li in range(len(LAM_AA_ALL))}

t0 = time.time()
print("\nWalk-forward (grid λ_aa)...")
for tk in TICKERS:
    att   = atts[tk]
    n_att = len(att)
    origins = list(range(n_att - N_TEST, n_att - 1, STEP_WF))
    for t_orig in origins:
        if t_orig + 1 >= n_att: continue
        actual = att[t_orig + 1]
        hist   = att[:t_orig+1]
        for li, laa in enumerate(LAM_AA_ALL):
            pred = lwr_step(hist, lam_va=LAM_VA_OPT, lam_ae=0.0, lam_aa=laa)
            if np.isnan(pred): continue
            results_grid[li][tk]["mae"].append(abs(pred - actual))
            results_grid[li][tk]["sign"].append(1 if np.sign(pred) == np.sign(actual) else 0)
print(f"Walk-forward grid завершён за {time.time()-t0:.1f}с")

# ── агрегация ──────────────────────────────────────────────────────────────────

def agg(results_d, tickers):
    rmae_d = {}; sacc_d = {}
    for li in results_d:
        rmae_d[li] = {}; sacc_d[li] = {}
        for tk in tickers:
            std_att = float(np.std(atts[tk]))
            mv = results_d[li][tk]["mae"]
            sv = results_d[li][tk]["sign"]
            rmae_d[li][tk] = float(np.mean(mv)) / std_att if mv else np.nan
            sacc_d[li][tk] = float(np.mean(sv)) * 100     if sv else np.nan
    return rmae_d, sacc_d

rmae_m, sacc_m = agg(results_modes, TICKERS)
rmae_g, sacc_g = agg(results_grid,  TICKERS)

base_rmae = float(np.nanmean([rmae_m[0][tk] for tk in TICKERS]))

print("\nСводка 5 режимов:")
for mi, (name, *_) in enumerate(MODES):
    mr = float(np.nanmean([rmae_m[mi][tk] for tk in TICKERS]))
    ms = float(np.nanmean([sacc_m[mi][tk] for tk in TICKERS]))
    d  = (mr - base_rmae) / base_rmae * 100
    pref = "+" if d >= 0 else ""
    print(f"  {name:22s}: rMAE={mr:.4f} ({pref}{d:.2f}%)  SignAcc={ms:.1f}%")

print("\nGrid λ_aa (pos + vel_ang_opt + acc_ang):")
for li, lbl in enumerate(LAM_AA_LABELS):
    mr  = float(np.nanmean([rmae_g[li][tk] for tk in TICKERS]))
    ms  = float(np.nanmean([sacc_g[li][tk] for tk in TICKERS]))
    d   = (mr - base_rmae) / base_rmae * 100
    pref = "+" if d >= 0 else ""
    print(f"  λ_aa={lbl:10s}: rMAE={mr:.4f} ({pref}{d:.2f}%)  SignAcc={ms:.1f}%")

# ── графики ────────────────────────────────────────────────────────────────────

# A: att / vel / acc (SBER) ────────────────────────────────────────────────────

SHOW = 150
att_s = att_sber
n_s   = len(att_s)
vel_s = np.concatenate([[0.0], np.diff(att_s)])
acc_s = np.concatenate([[0.0, 0.0], np.diff(vel_s[1:])])
sl    = slice(n_s - SHOW, n_s)

fig_a, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
fig_a.suptitle("79-A: att / vel / acc (SBER, последние 150 баров)", fontsize=11)
axes[0].plot(att_s[sl],  color="seagreen",   lw=1.2, label="att"); axes[0].axhline(0, color="k", lw=0.4, ls="--"); axes[0].set_ylabel("att"); axes[0].legend(fontsize=8)
axes[1].plot(vel_s[sl],  color="steelblue",  lw=1.0, label="vel = diff(att)"); axes[1].axhline(0, color="k", lw=0.4, ls="--"); axes[1].set_ylabel("vel"); axes[1].legend(fontsize=8)
axes[2].plot(acc_s[sl],  color="darkorange", lw=0.9, label="acc = diff(vel)"); axes[2].axhline(0, color="k", lw=0.4, ls="--"); axes[2].set_ylabel("acc"); axes[2].set_xlabel("бар"); axes[2].legend(fontsize=8)

fig_a.tight_layout(); fig_a.savefig(FIGDIR / "79_A_signals.png", dpi=130); plt.close(fig_a)
print("Рис. A сохранён")

# B: rMAE по режимам (bar chart) ───────────────────────────────────────────────

mode_names = [m[0] for m in MODES]
mr_modes = [float(np.nanmean([rmae_m[mi][tk] for tk in TICKERS])) for mi in range(N_MODES)]
ms_modes = [float(np.nanmean([sacc_m[mi][tk] for tk in TICKERS])) for mi in range(N_MODES)]

colors_mode = ["#1f77b4","#d62728","#2ca02c","#9467bd","#8c564b"]

fig_b, ax = plt.subplots(figsize=(11, 5))
ax.set_title("79-B: rMAE по режимам (8 тикеров, walk-forward)", fontsize=11)
bars = ax.bar(mode_names, mr_modes, color=colors_mode, width=0.5, alpha=0.85)
for bar, v in zip(bars, mr_modes):
    d = (v - base_rmae) / base_rmae * 100
    pref = "+" if d >= 0 else ""
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.0002,
            f"{v:.4f}\n({pref}{d:.1f}%)", ha="center", va="bottom", fontsize=9)
ax.set_ylim(min(mr_modes)*0.985, max(mr_modes)*1.02)
ax.set_xticklabels(mode_names, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("rMAE"); ax.grid(axis="y", alpha=0.3)
fig_b.tight_layout(); fig_b.savefig(FIGDIR / "79_B_rmae_modes.png", dpi=130); plt.close(fig_b)
print("Рис. B сохранён")

# C: SignAcc по режимам ────────────────────────────────────────────────────────

fig_c, ax = plt.subplots(figsize=(11, 5))
ax.set_title("79-C: SignAcc по режимам", fontsize=11)
bars2 = ax.bar(mode_names, ms_modes, color=colors_mode, width=0.5, alpha=0.85)
for bar, v in zip(bars2, ms_modes):
    d = v - ms_modes[0]
    pref = "+" if d >= 0 else ""
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
            f"{v:.1f}%\n({pref}{d:.1f}pp)", ha="center", va="bottom", fontsize=9)
ax.set_ylim(min(ms_modes)*0.995, max(ms_modes)*1.005)
ax.set_xticklabels(mode_names, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("SignAcc %"); ax.grid(axis="y", alpha=0.3)
fig_c.tight_layout(); fig_c.savefig(FIGDIR / "79_C_signacc_modes.png", dpi=130); plt.close(fig_c)
print("Рис. C сохранён")

# D: grid λ_aa ─────────────────────────────────────────────────────────────────

x_ticks = list(range(len(LAM_AA_ALL)))
fig_d, ax = plt.subplots(figsize=(12, 5))
ax.set_title("79-D: rMAE vs λ_aa  [pos + vel_ang(0.01) + λ_aa·d_ang(acc)]", fontsize=11)
for i, tk in enumerate(TICKERS):
    vals = [rmae_g[li][tk] for li in range(len(LAM_AA_ALL))]
    ax.plot(x_ticks, vals, color=COLORS_TK[i], lw=0.9, marker="o", ms=3, alpha=0.55, label=tk)
mean_r = [float(np.nanmean([rmae_g[li][tk] for tk in TICKERS])) for li in range(len(LAM_AA_ALL))]
ax.plot(x_ticks, mean_r, color="black", lw=2.2, marker="D", ms=7, label="СРЕДНЕЕ")
ax.axhline(0.0572, color="crimson", lw=1.0, ls="--", label="скр.78 vel_ang(0.01)")
ax.set_xticks(x_ticks); ax.set_xticklabels(LAM_AA_LABELS, fontsize=8)
ax.set_xlabel("λ_aa"); ax.set_ylabel("rMAE")
for xi, v in zip(x_ticks, mean_r):
    ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=0.3)
fig_d.tight_layout(); fig_d.savefig(FIGDIR / "79_D_grid_acc.png", dpi=130); plt.close(fig_d)
print("Рис. D сохранён")

# E: Scatter d_ang(vel) vs d_ang(acc) для пула ────────────────────────────────

_, diag = lwr_step_diag(att_sber[:n_demo+1], lam_va=LAM_VA_OPT, lam_aa=auto_lam_aa)

fig_e, ax = plt.subplots(figsize=(8, 6))
ax.set_title(f"79-E: Scatter d_ang(vel) vs d_ang(acc) (SBER, пул={len(diag['cands_pool'])})", fontsize=11)

d_va_p = diag["d_va_pool"]; d_aa_p = diag["d_aa_pool"]
pool   = diag["cands_pool"]

ax.scatter(d_va_p, d_aa_p, s=12, color="silver", alpha=0.5, label=f"весь пул ({len(pool)})")

mask_pos = np.isin(pool, diag["cands_pos"])
ax.scatter(d_va_p[mask_pos], d_aa_p[mask_pos], s=35, color="#1f77b4", alpha=0.8,
           label=f"pos-only ({mask_pos.sum()})", zorder=3)

mask_v = np.isin(pool, diag["cands_v"])
ax.scatter(d_va_p[mask_v], d_aa_p[mask_v], s=35, color="#d62728", alpha=0.8, marker="^",
           label=f"vel_ang_opt ({mask_v.sum()})", zorder=4)

mask_va = np.isin(pool, diag["cands_va"])
ax.scatter(d_va_p[mask_va], d_aa_p[mask_va], s=35, color="#2ca02c", alpha=0.8, marker="s",
           label=f"vel_ang+acc_ang ({mask_va.sum()})", zorder=5)

ax.set_xlabel("d_ang(vel) — угловое расстояние по скорости")
ax.set_ylabel("d_ang(acc) — угловое расстояние по ускорению")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig_e.tight_layout(); fig_e.savefig(FIGDIR / "79_E_scatter.png", dpi=130); plt.close(fig_e)
print("Рис. E сохранён")

print("\nГотово. Фигуры: 79_A...E.png")
