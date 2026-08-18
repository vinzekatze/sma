"""
77 — Поиск в пространстве позиция + скорость.

Гипотеза: добавление vel[t] = att[t] - att[t-1] (скорость изменения аттрактора)
в метрику отбора соседей позволяет различать точки, движущиеся в одном направлении,
от точек, оказавшихся рядом по положению, но с другой динамикой.

Архитектура:
  1. Каскад (как обычно) отбирает кандидатов по позиционным координатам.
  2. Финальный отбор xi_lwr соседей: комбинированное расстояние
       d = d_pos + λ · d_vel
     где d_pos = ||Δatt[-p_fit:]||,  d_vel = ||Δvel[-p_fit:]||,
         vel[t] = att[t] - att[t-1]  (каузальная разность).
  3. LWR-аппроксимация только по позиционным координатам (p_fit).

λ = 0         — чистая позиция (базовый случай)
λ = 0.25/0.5/1.0/2.0 — ручное масштабирование
λ = 'auto'    — std(att) / std(vel), уравнивает вклады позиции и скорости

p_vel = p_fit (те же лаги, что у позиции).
Att-фильтр: Local Projective (m=9, d=3, k=30, n=3) — стандарт.

Протокол: идентичен скр.76 — h=1, последние 200 баров, шаг 5, 8 тикеров.

Графики:
  A — att и vel (SBER, 150 баров) — характер сигналов
  B — rMAE vs λ (все тикеры + среднее)
  C — SignAcc vs λ
  D — Scatter: pos_dist vs vel_dist для кандидатов каскада;
      отмечены финально отобранные pos-only (синий) и pos+vel auto (зелёный)
  E — Фазовый портрет pos+vel: [att[t], vel[t]] для всей библиотеки;
      соседи pos-only vs pos+vel auto
"""

import json, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.spatial import KDTree

ROOT    = Path(__file__).resolve().parent.parent.parent
FIGDIR  = ROOT / "research" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATADIR = ROOT / "data" / "candles"
sys.path.insert(0, str(ROOT))

TICKERS   = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL  = "1d"

LP_M = 9; LP_D = 3; LP_K = 30; LP_N = 3   # Local Projective стандарт

P_FIT     = 16
P_MAX     = 64
P_VEL     = P_FIT        # число лагов скорости = числу лагов позиции
XI_LWR    = 3 * (P_FIT + 1)
STEP_BASE = 2.0

N_TEST  = 200
STEP_WF = 5

LAM_FIXED  = [0.0, 0.25, 0.5, 1.0, 2.0]   # 0.0 = baseline pos-only
LAM_LABELS = ["0 (pos-only)", "0.25", "0.5", "1.0", "2.0", "auto"]
COLORS_LAM = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
COLORS_TK  = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#17becf"]

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


def lwr_step_vel(att_hist: np.ndarray, lam: float
                 ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """h=1 LWR с комбинированной метрикой pos + λ·vel.

    Возвращает: (prediction, cands_pos_only, cands_vel, (d_pos, d_vel) для cands каскада).
    """
    n   = len(att_hist)
    m   = n - P_MAX - 1
    if m < XI_LWR + 5:
        return np.nan, np.array([]), np.array([]), (np.array([]), np.array([]))

    # Скорость: каузальная разность att (att[0]=0 → vel[0]=0)
    vel_hist = np.concatenate([[0.0], np.diff(att_hist)])   # длина n

    t_arr  = np.arange(P_MAX, n - 1)
    X_pos  = np.column_stack([att_hist[t_arr - (P_MAX-1-j)] for j in range(P_MAX)])
    X_vel  = np.column_stack([vel_hist[t_arr - (P_VEL-1-j)] for j in range(P_VEL)])
    y_base = att_hist[t_arr + 1]

    vec_pos = att_hist[-P_MAX:]
    vec_vel = vel_hist[-P_VEL:]

    # ── Каскад (позиция) + финальный отбор по pos+vel ─────────────────────────
    # Все уровни кроме последнего: фильтрация по позиции + расширение.
    # Последний уровень: НЕ фильтрует сам — передаёт расширенный пул
    # на совместный отбор pos+λ·vel. Это позволяет скорости реально влиять
    # на выбор (иначе пул уже == xi_lwr и перебора нет).
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
            exp     = np.clip(exp, 0, m-1)
            cands   = np.unique(exp)
        # Последний уровень: не фильтруем здесь

    # Дистанции для всего пула, поступившего на последний уровень
    d_pos_all = np.linalg.norm(X_pos[cands, -P_FIT:] - vec_pos[-P_FIT:], axis=1)
    d_vel_all = np.linalg.norm(X_vel[cands, -P_VEL:] - vec_vel[-P_VEL:], axis=1)

    # Pos-only: отбор только по позиции (базовый случай)
    if len(cands) > XI_LWR:
        sel_pos = np.argpartition(d_pos_all, XI_LWR-1)[:XI_LWR]
        cands_pos_only = cands[sel_pos]
    else:
        cands_pos_only = cands.copy()

    if len(cands) < P_FIT + 2:
        return np.nan, cands_pos_only, cands_pos_only, (d_pos_all, d_vel_all)

    # ── Финальный отбор с учётом скорости ─────────────────────────────────────
    if lam > 0.0 and len(cands) > XI_LWR:
        d_combined = d_pos_all + lam * d_vel_all
        cands = cands[np.argpartition(d_combined, XI_LWR-1)[:XI_LWR]]
    else:
        cands = cands_pos_only

    # ── LWR аппроксимация (позиция) ───────────────────────────────────────────
    X_nn  = X_pos[cands, -P_FIT:]
    y_nn  = y_base[cands]
    vec_f = vec_pos[-P_FIT:]
    h_bw  = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
    val   = lwr_fit(X_nn, y_nn, vec_f, h_bw)

    return val, cands_pos_only, cands, (d_pos_all, d_vel_all)

# ── загрузка данных ────────────────────────────────────────────────────────────

print("Загрузка + Local Projective...")
atts: dict[str, np.ndarray] = {}
t_load = time.time()
for tk in TICKERS:
    print(f"  {tk}...", end=" ", flush=True)
    t1 = time.time(); atts[tk] = load_att(tk)
    print(f"{time.time()-t1:.1f}с  ({len(atts[tk])} баров)")

# auto λ: std(att) / std(vel)  — усредняем по тикерам
auto_lam_per_tk = {}
for tk in TICKERS:
    att = atts[tk]
    vel = np.concatenate([[0.0], np.diff(att)])
    auto_lam_per_tk[tk] = float(np.std(att) / (np.std(vel) + 1e-12))
auto_lam_mean = float(np.mean(list(auto_lam_per_tk.values())))
print(f"auto λ по тикерам: {', '.join(f'{tk}={v:.2f}' for tk, v in auto_lam_per_tk.items())}")
print(f"auto λ среднее: {auto_lam_mean:.2f}")
print(f"Загрузка итого: {time.time()-t_load:.1f}с")

# Все тестируемые λ (включая auto)
LAM_ALL = LAM_FIXED + [auto_lam_mean]

# ── walk-forward ───────────────────────────────────────────────────────────────

# results[lam_idx][ticker] = {"mae": [], "sign": []}
results = {li: {tk: {"mae": [], "sign": []} for tk in TICKERS}
           for li in range(len(LAM_ALL))}

t0 = time.time()
for tk in TICKERS:
    att   = atts[tk]
    n_att = len(att)
    origins = list(range(n_att - N_TEST, n_att - 1, STEP_WF))
    print(f"\n{tk}: {len(origins)} origins...", flush=True)

    for t_orig in origins:
        if t_orig + 1 >= n_att:
            continue
        actual = att[t_orig + 1]

        for li, lam in enumerate(LAM_ALL):
            pred, _, _, _ = lwr_step_vel(att[:t_orig+1], lam)
            if np.isnan(pred):
                continue
            results[li][tk]["mae"].append(abs(pred - actual))
            results[li][tk]["sign"].append(1 if np.sign(pred) == np.sign(actual) else 0)

print(f"\nWalk-forward завершён за {time.time()-t0:.1f}с")

# ── агрегация ──────────────────────────────────────────────────────────────────

rmae   = {li: {} for li in range(len(LAM_ALL))}
sigacc = {li: {} for li in range(len(LAM_ALL))}

for li in range(len(LAM_ALL)):
    for tk in TICKERS:
        std_att = float(np.std(atts[tk]))
        m_vals  = results[li][tk]["mae"]
        s_vals  = results[li][tk]["sign"]
        rmae[li][tk]   = float(np.mean(m_vals)) / std_att if m_vals else np.nan
        sigacc[li][tk] = float(np.mean(s_vals)) * 100     if s_vals else np.nan

print("\nrMAE / SignAcc по λ:")
base_rmae = np.nanmean([rmae[0][tk] for tk in TICKERS])
for li, lbl in enumerate(LAM_LABELS):
    mr   = np.nanmean([rmae[li][tk] for tk in TICKERS])
    ms   = np.nanmean([sigacc[li][tk] for tk in TICKERS])
    d    = (mr - base_rmae) / base_rmae * 100
    pref = "+" if d >= 0 else ""
    print(f"  λ={lbl:12s}: rMAE={mr:.4f} ({pref}{d:.2f}%)  SignAcc={ms:.1f}%")

# ── диагностика для SBER (последний origin тестового окна) ────────────────────

att_sber = atts["SBER"]
n_sber   = len(att_sber)
t_demo   = n_sber - N_TEST // 2

_, cands_pos, cands_vel_auto, (d_pos_all, d_vel_all) = \
    lwr_step_vel(att_sber[:t_demo+1], lam=auto_lam_mean)

vel_sber = np.concatenate([[0.0], np.diff(att_sber)])

# ── графики ────────────────────────────────────────────────────────────────────

# ── A: сигналы att и vel ───────────────────────────────────────────────────────

SHOW = 150
fig_a, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
fig_a.suptitle("77-A: att и vel (SBER, последние 150 баров)", fontsize=11)
sl = slice(n_sber - SHOW, n_sber)

axes[0].plot(att_sber[sl], color="seagreen", lw=1.2, label="att (Local Projective)")
axes[0].axhline(0, color="k", lw=0.4, ls="--"); axes[0].set_ylabel("att"); axes[0].legend(fontsize=8)

axes[1].plot(vel_sber[sl], color="steelblue", lw=1.0, label="vel = diff(att)")
axes[1].axhline(0, color="k", lw=0.4, ls="--"); axes[1].set_ylabel("vel")
axes[1].set_xlabel("бар (отн. окна)"); axes[1].legend(fontsize=8)

fig_a.tight_layout()
fig_a.savefig(FIGDIR / "77_A_signals.png", dpi=130); plt.close(fig_a)
print("Рис. A сохранён")

# ── B: rMAE vs λ ─────────────────────────────────────────────────────────────

x_ticks = list(range(len(LAM_ALL)))
fig_b, ax = plt.subplots(figsize=(10, 5))
ax.set_title("77-B: rMAE по λ  [pos + λ·vel, финальный отбор]", fontsize=11)
for i, tk in enumerate(TICKERS):
    vals = [rmae[li][tk] for li in range(len(LAM_ALL))]
    ax.plot(x_ticks, vals, color=COLORS_TK[i], lw=1.0, marker="o", ms=4, alpha=0.6, label=tk)
mean_r = [np.nanmean([rmae[li][tk] for tk in TICKERS]) for li in range(len(LAM_ALL))]
ax.plot(x_ticks, mean_r, color="black", lw=2.2, marker="D", ms=7, label="СРЕДНЕЕ")
ax.set_xticks(x_ticks); ax.set_xticklabels(LAM_LABELS, fontsize=9)
ax.set_xlabel("λ"); ax.set_ylabel("rMAE")
for xi, v in zip(x_ticks, mean_r):
    ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                xytext=(0, 9), fontsize=7, ha="center")
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
fig_b.tight_layout()
fig_b.savefig(FIGDIR / "77_B_rmae.png", dpi=130); plt.close(fig_b)
print("Рис. B сохранён")

# ── C: SignAcc vs λ ──────────────────────────────────────────────────────────

fig_c, ax = plt.subplots(figsize=(10, 5))
ax.set_title("77-C: SignAcc по λ", fontsize=11)
for i, tk in enumerate(TICKERS):
    vals = [sigacc[li][tk] for li in range(len(LAM_ALL))]
    ax.plot(x_ticks, vals, color=COLORS_TK[i], lw=1.0, marker="o", ms=4, alpha=0.6, label=tk)
mean_s = [np.nanmean([sigacc[li][tk] for tk in TICKERS]) for li in range(len(LAM_ALL))]
ax.plot(x_ticks, mean_s, color="black", lw=2.2, marker="D", ms=7, label="СРЕДНЕЕ")
ax.axhline(50, color="grey", ls="--", lw=0.8, label="случайный угадыватель")
ax.set_xticks(x_ticks); ax.set_xticklabels(LAM_LABELS, fontsize=9)
ax.set_xlabel("λ"); ax.set_ylabel("SignAcc, %")
for xi, v in zip(x_ticks, mean_s):
    ax.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points",
                xytext=(0, 9), fontsize=7, ha="center")
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
fig_c.tight_layout()
fig_c.savefig(FIGDIR / "77_C_signacc.png", dpi=130); plt.close(fig_c)
print("Рис. C сохранён")

# ── D: Scatter d_pos vs d_vel для кандидатов каскада ─────────────────────────

fig_d, ax = plt.subplots(figsize=(7, 7))
ax.set_title(f"77-D: Кандидаты каскада: d_pos vs d_vel\nSBER, origin t={t_demo}", fontsize=10)

if len(d_pos_all):
    ax.scatter(d_pos_all, d_vel_all, s=18, color="lightgray", alpha=0.7,
               zorder=1, label=f"Все кандидаты каскада ({len(d_pos_all)})")

    # Соседи pos-only: ближайшие по d_pos
    if len(cands_pos) > XI_LWR:
        idx_p = np.argpartition(d_pos_all, XI_LWR-1)[:XI_LWR]
    else:
        idx_p = np.arange(len(d_pos_all))
    ax.scatter(d_pos_all[idx_p], d_vel_all[idx_p], s=45, color="steelblue", alpha=0.85,
               zorder=2, label=f"Отобраны pos-only ({len(idx_p)})")

    # Соседи pos+vel auto: ближайшие по d_pos + auto_lam * d_vel
    d_comb = d_pos_all + auto_lam_mean * d_vel_all
    if len(d_comb) > XI_LWR:
        idx_v = np.argpartition(d_comb, XI_LWR-1)[:XI_LWR]
    else:
        idx_v = np.arange(len(d_comb))
    ax.scatter(d_pos_all[idx_v], d_vel_all[idx_v], s=45, color="seagreen", alpha=0.85,
               zorder=3, label=f"Отобраны pos+vel auto ({len(idx_v)})")

ax.set_xlabel("d_pos (||Δatt[-p_fit:]||)"); ax.set_ylabel("d_vel (||Δvel[-p_fit:]||)")
ax.legend(fontsize=8); ax.grid(alpha=0.2)
fig_d.tight_layout()
fig_d.savefig(FIGDIR / "77_D_scatter.png", dpi=130); plt.close(fig_d)
print("Рис. D сохранён")

# ── E: Фазовый портрет [att, vel] с подсветкой соседей ───────────────────────

fig_e, ax = plt.subplots(figsize=(8, 8))
ax.set_title("77-E: Фазовый портрет [att[t], vel[t]] — SBER\n"
             "синий=pos-only, зелёный=pos+vel auto, ★=запрос", fontsize=10)

att_use = att_sber[:t_demo+1]
vel_use = vel_sber[:t_demo+1]
n_lib   = len(att_use) - P_MAX - 1
t_arr_d = np.arange(P_MAX, P_MAX + n_lib)

if n_lib > 0:
    lib_att = att_use[t_arr_d]
    lib_vel = vel_use[t_arr_d]
    ax.scatter(lib_att, lib_vel, s=1.5, color="lightgray", alpha=0.3, zorder=1)

    if len(cands_pos):
        idx = cands_pos[cands_pos < n_lib]
        ax.scatter(lib_att[idx], lib_vel[idx], s=45, color="steelblue",
                   alpha=0.85, zorder=2, label=f"pos-only ({len(idx)})")

    if len(cands_vel_auto):
        idx = cands_vel_auto[cands_vel_auto < n_lib]
        ax.scatter(lib_att[idx], lib_vel[idx], s=45, color="seagreen",
                   alpha=0.85, zorder=3, label=f"pos+vel auto ({len(idx)})")

q_att = att_use[-1]; q_vel = vel_use[-1]
ax.scatter([q_att], [q_vel], s=220, color="gold", marker="*",
           zorder=4, label="Запрос")
ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
ax.set_xlabel("att[t]  (позиция)"); ax.set_ylabel("vel[t]  (скорость)")
ax.legend(fontsize=8)
fig_e.tight_layout()
fig_e.savefig(FIGDIR / "77_E_phase_vel.png", dpi=130); plt.close(fig_e)
print("Рис. E сохранён")

# ── итоговая таблица ──────────────────────────────────────────────────────────

print("\n══════════════════════════════════════════════════════════")
print(f"{'λ':>14} | {'rMAE':>8} | {'ΔrMAE%':>8} | {'SignAcc':>9}")
print("──────────────────────────────────────────────────────────")
for li, (lbl, mr, ms) in enumerate(zip(LAM_LABELS, mean_r, mean_s)):
    d = (mr - base_rmae) / base_rmae * 100
    pref = "+" if d >= 0 else ""
    print(f"{lbl:>14} | {mr:>8.4f} | {pref}{d:>7.2f}% | {ms:>8.1f}%")
print("══════════════════════════════════════════════════════════")
print(f"\n(auto λ среднее = {auto_lam_mean:.2f})")
print("\nГотово. Графики:")
for ltr, fn in zip("ABCDE", ["77_A_signals.png","77_B_rmae.png","77_C_signacc.png",
                              "77_D_scatter.png","77_E_phase_vel.png"]):
    print(f"  {ltr}: {FIGDIR/fn}")
