"""
89_cascade_step.py — Шаг каскада: поиск оптимального множителя.

Текущий стандарт: октавный ×2 (P_MAX=64 → 32 → 16).
Гипотеза: аттрактор att имеет "естественные масштабы", не обязательно
кратные 2. Иной шаг даёт лучшее выравнивание промежуточных уровней
с реальной спектральной структурой аттрактора.

Тест: шаги ×√2 / ×1.5 / ×2.0 / ×2.5 / ×3.0 / ×4.0
при фиксированном P_MAX=64, P_FIT=16.

Два режима для каждого шага:
  fixed_16  — p_reg=16, каскад от P_MAX до 16 данным шагом
  ensemble  — p_reg ∈ {4,6,8,10,12,14,16}, честный каскад P_MAX→p_reg,
              1/LOO взвешивание (xi=51 фиксировано)

Дополнительно: oracle per-step (лучший p_reg при данном шаге) и
per-(step, p_reg) тепловая карта — где именно проявляется эффект.

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

P_REG_GRID  = [4, 6, 8, 10, 12, 14, 16]
P_FIT_BASE  = 16

STEPS_GRID  = [math.sqrt(2), 1.5, 2.0, 2.5, 3.0, 4.0]
STEP_LABELS = ["×√2", "×1.5", "×2.0*", "×2.5", "×3.0", "×4.0"]


# ── вспомогательные функции ───────────────────────────────────────────────────

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
    norm_A = np.linalg.norm(A, axis=1); norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-12:
        return np.ones(len(A))
    cos = np.where(norm_A > 1e-12, (A @ b) / (norm_A * norm_b), 0.0)
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
        h_diag = w * np.einsum("ij,ij->i", A @ inv, A)
        denom = 1.0 - h_diag
        e_loo = np.where(np.abs(denom) > 1e-6, resid / denom, resid)
    except np.linalg.LinAlgError:
        e_loo = resid
    return float(np.mean(np.abs(e_loo)))


# ── генератор уровней каскада ─────────────────────────────────────────────────

def levels_by_step(p_fit: int, step: float, p_max: int = P_MAX) -> list[int]:
    """Каскад P_MAX→…→p_fit с заданным множителем step."""
    levels: list[int] = []; p = p_max
    while True:
        levels.append(p)
        p_next = int(p / step)
        if p_next < max(p_fit, 2):
            break
        p = p_next
    if levels[-1] != p_fit:
        levels.append(p_fit)
    return levels


# ── контекст ──────────────────────────────────────────────────────────────────

class _Context:
    __slots__ = ("X_full", "y_base", "acc_hist", "vec_full", "n", "ok")

    def __init__(self, att: np.ndarray) -> None:
        n = len(att); self.n = n; self.ok = False
        if n - P_MAX - 1 < 3:
            return
        t_arr = np.arange(P_MAX, n - 1)
        self.X_full   = np.column_stack([att[t_arr - (P_MAX - 1 - j)] for j in range(P_MAX)])
        self.y_base   = att[t_arr + 1]
        acc = np.zeros(n)
        if n >= 3:
            acc[2:] = att[2:] - 2 * att[1:-1] + att[:-2]
        self.acc_hist = acc
        self.vec_full = att[-P_MAX:].copy()
        self.ok       = True


# ── ядро прогноза ─────────────────────────────────────────────────────────────

def _forecast(ctx: _Context, p_reg: int,
              cascade_levels: list[int],
              need_loo: bool = True) -> tuple[float, float]:
    """Прогноз с acc_ang и произвольной цепочкой уровней.
    Возвращает (pred, loo_err). loo_err=nan если need_loo=False."""
    X_full   = ctx.X_full; y_base = ctx.y_base
    vec_full = ctx.vec_full; acc_hist = ctx.acc_hist; n = ctx.n

    if len(X_full) < XI_FIXED + 1:
        return np.nan, np.nan

    t_arr   = np.arange(P_MAX, n - 1)
    X_acc   = np.column_stack([acc_hist[t_arr - (p_reg - 1 - j)] for j in range(p_reg)])
    vec_acc = acc_hist[-p_reg:].copy()
    query   = vec_full[-p_reg:]
    cands   = np.arange(len(X_full))

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
        d_pos = np.linalg.norm(X_full[cands, -p_reg:] - query, axis=1)
        d_acc = _cosine_dist(X_acc[cands], vec_acc)
        cands = cands[np.argpartition(d_pos + LAMBDA * d_acc, XI_FIXED - 1)[:XI_FIXED]]

    if len(cands) < p_reg + 2:
        return np.nan, np.nan

    X_nn = X_full[cands, -p_reg:]; y_nn = y_base[cands]
    pred = _lwr_predict(X_nn, y_nn, query)
    loo  = _lwr_loo(X_nn, y_nn, query) if need_loo else np.nan
    return pred, loo


# ── загрузка данных ───────────────────────────────────────────────────────────

def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend_causal(close)
    return _lp_proj(close / np.maximum(lt, 1e-10), LP_M, LP_D, LP_K, LP_N)


# ── walk-forward для одного тикера ────────────────────────────────────────────

def wf_ticker(att_full: np.ndarray) -> dict:
    """
    Возвращает словарь:
      "fixed"   : {step_name: [errors]}
      "ensemble": {step_name: [errors]}
      "oracle"  : {step_name: [errors]}
      "per_preg": {step_name: {p_reg: [errors]}}   ← для тепловой карты
    """
    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP_WF, n_total - 1, STEP_WF))

    res: dict = {
        "fixed":    {n: [] for n in STEP_LABELS},
        "ensemble": {n: [] for n in STEP_LABELS},
        "oracle":   {n: [] for n in STEP_LABELS},
        "per_preg": {n: {p: [] for p in P_REG_GRID} for n in STEP_LABELS},
    }

    for t_orig in origins:
        if t_orig + 1 >= n_total:
            continue
        hist = att_full[:t_orig + 1]
        true = att_full[t_orig + 1]

        ctx = _Context(hist)
        if not ctx.ok:
            continue

        for step_val, step_lbl in zip(STEPS_GRID, STEP_LABELS):
            # fixed_16: каскад до p=16 данным шагом, без LOO
            lvls_fixed = levels_by_step(P_FIT_BASE, step_val)
            pred_fixed, _ = _forecast(ctx, P_FIT_BASE, lvls_fixed, need_loo=False)
            if not np.isnan(pred_fixed):
                res["fixed"][step_lbl].append(pred_fixed - true)

            # per-p_reg: честный каскад P_MAX→p_reg данным шагом + LOO
            preds: dict[int, float] = {}
            loos:  dict[int, float] = {}
            for p_reg in P_REG_GRID:
                lvls = levels_by_step(p_reg, step_val)
                pred, loo = _forecast(ctx, p_reg, lvls, need_loo=True)
                preds[p_reg] = pred; loos[p_reg] = loo

            valid = [p for p in P_REG_GRID
                     if not np.isnan(preds[p]) and not np.isnan(loos[p])]
            if not valid:
                continue

            # oracle (лучший p для данного шага)
            best_p = min(valid, key=lambda p: abs(preds[p] - true))
            res["oracle"][step_lbl].append(preds[best_p] - true)

            # ensemble 1/LOO
            inv = np.array([1.0 / (loos[p] + 1e-12) for p in valid])
            w   = inv / inv.sum()
            pe  = float(w @ np.array([preds[p] for p in valid]))
            res["ensemble"][step_lbl].append(pe - true)

            # per-p_reg
            for p in valid:
                res["per_preg"][step_lbl][p].append(preds[p] - true)

    return res


# ── агрегация ─────────────────────────────────────────────────────────────────

print("Загрузка + Local Projective...")
t0 = time.time()
att_data: dict[str, np.ndarray] = {}
for tkr in TICKERS:
    att_data[tkr] = load_att(tkr)
    print(f"  {tkr}...")
std_all  = [float(np.std(att_data[tkr])) for tkr in TICKERS]
std_mean = float(np.mean(std_all))
print(f"Загрузка: {time.time()-t0:.1f}с")

# Показываем уровни каскада до запуска walk-forward
print("\n── Уровни каскада для каждого шага ──")
for step_val, step_lbl in zip(STEPS_GRID, STEP_LABELS):
    lvls_demo = levels_by_step(P_FIT_BASE, step_val)
    print(f"  {step_lbl:6s}: {lvls_demo}  ({len(lvls_demo)} уровней)")

print(f"\nWalk-forward ({len(STEPS_GRID)} шагов × 7 p_reg, {len(TICKERS)} тикеров)...")
t0 = time.time()

agg: dict = {
    "fixed":    {n: [] for n in STEP_LABELS},
    "ensemble": {n: [] for n in STEP_LABELS},
    "oracle":   {n: [] for n in STEP_LABELS},
    "per_preg": {n: {p: [] for p in P_REG_GRID} for n in STEP_LABELS},
}

for tkr in TICKERS:
    t1  = time.time()
    res = wf_ticker(att_data[tkr])
    for mode in ("fixed", "ensemble", "oracle"):
        for lbl in STEP_LABELS:
            agg[mode][lbl].extend(res[mode][lbl])
    for lbl in STEP_LABELS:
        for p in P_REG_GRID:
            agg["per_preg"][lbl][p].extend(res["per_preg"][lbl][p])
    print(f"  {tkr}: {time.time()-t1:.1f}с")

print(f"Walk-forward: {time.time()-t0:.1f}с")


def rmae_from(errs: list) -> float:
    return float(np.mean(np.abs(errs))) / std_mean if errs else np.nan


# baseline = ×2.0 fixed_16 (совпадает с acc_ang baseline из скр.80/85)
baseline = rmae_from(agg["fixed"]["×2.0*"])

print(f"\nBaseline (×2.0 fixed_16): rMAE = {baseline:.4f}")

print("\n── Итоговая таблица ──")
print(f"{'Шаг':8s}  {'Уровни':20s}  {'fixed_16 Δ%':12s}  {'ensemble Δ%':12s}  {'oracle Δ%':12s}")
for step_val, step_lbl in zip(STEPS_GRID, STEP_LABELS):
    lvls = levels_by_step(P_FIT_BASE, step_val)
    r_f  = rmae_from(agg["fixed"][step_lbl])
    r_e  = rmae_from(agg["ensemble"][step_lbl])
    r_o  = rmae_from(agg["oracle"][step_lbl])
    d_f  = (r_f / baseline - 1) * 100
    d_e  = (r_e / baseline - 1) * 100
    d_o  = (r_o / baseline - 1) * 100
    lvls_str = "→".join(str(l) for l in lvls)
    print(f"  {step_lbl:8s}  {lvls_str:20s}  {d_f:+8.2f}%      {d_e:+8.2f}%      {d_o:+8.2f}%")

print("\n── Лучший шаг по режиму ──")
for mode, label in [("fixed", "fixed_16"), ("ensemble", "ensemble")]:
    best_lbl = min(STEP_LABELS, key=lambda l: rmae_from(agg[mode][l]))
    best_r   = rmae_from(agg[mode][best_lbl])
    best_d   = (best_r / baseline - 1) * 100
    print(f"  {label}: лучший шаг = {best_lbl}  rMAE={best_r:.4f}  Δ={best_d:+.2f}%")

# ── per-(step, p_reg) тепловая карта ─────────────────────────────────────────
print("\n── Per-(шаг, p_reg) rMAE Δ% ──")
header = "шаг\\p_reg   " + "  ".join(f"p={p:2d}" for p in P_REG_GRID)
print(f"  {header}")
heat = np.zeros((len(STEP_LABELS), len(P_REG_GRID)))
for si, step_lbl in enumerate(STEP_LABELS):
    row = []
    for pi, p in enumerate(P_REG_GRID):
        r = rmae_from(agg["per_preg"][step_lbl][p])
        d = (r / baseline - 1) * 100 if not np.isnan(r) else np.nan
        heat[si, pi] = d if not np.isnan(d) else 0.0
        row.append(f"{d:+5.1f}%" if not np.isnan(d) else "  NaN ")
    print(f"  {step_lbl:8s}:  " + "  ".join(row))


# ── Фигуры ────────────────────────────────────────────────────────────────────

# Рис. A — fixed_16 по шагам
r_fixed = [rmae_from(agg["fixed"][l]) for l in STEP_LABELS]
d_fixed = [(r / baseline - 1) * 100 for r in r_fixed]

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#ef5350" if d > 0 else "#66bb6a" for d in d_fixed]
bars = ax.bar(STEP_LABELS, d_fixed, color=colors, width=0.55)
ax.axhline(0, color="white", lw=0.8)
for bar, v in zip(bars, d_fixed):
    ax.text(bar.get_x() + bar.get_width() / 2,
            v + (0.15 if v >= 0 else -0.15),
            f"{v:+.2f}%", ha="center",
            va="bottom" if v >= 0 else "top", color="white", fontsize=10)
ax.set_ylabel("Δ rMAE vs baseline ×2.0 (%)")
ax.set_title("Рис.A  fixed_16: эффект шага каскада  (<0 = лучше)")
plt.tight_layout()
fig.savefig(FIG_DIR / "89_A.png", dpi=120); plt.close()
print("\nРис. A сохранён")

# Рис. B — ensemble по шагам
r_ens = [rmae_from(agg["ensemble"][l]) for l in STEP_LABELS]
d_ens = [(r / baseline - 1) * 100 for r in r_ens]

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#ef5350" if d > 0 else "#66bb6a" for d in d_ens]
bars = ax.bar(STEP_LABELS, d_ens, color=colors, width=0.55)
ax.axhline(0, color="white", lw=0.8)
for bar, v in zip(bars, d_ens):
    ax.text(bar.get_x() + bar.get_width() / 2,
            v + (0.15 if v >= 0 else -0.15),
            f"{v:+.2f}%", ha="center",
            va="bottom" if v >= 0 else "top", color="white", fontsize=10)
ax.set_ylabel("Δ rMAE vs baseline ×2.0 (%)")
ax.set_title("Рис.B  ensemble 1/LOO: эффект шага каскада  (<0 = лучше)")
plt.tight_layout()
fig.savefig(FIG_DIR / "89_B.png", dpi=120); plt.close()
print("Рис. B сохранён")

# Рис. C — тепловая карта per-(step, p_reg)
fig, ax = plt.subplots(figsize=(10, 5))
vmax = max(abs(heat).max(), 1.0)
im = ax.imshow(heat, cmap="RdYlGn_r", aspect="auto",
               vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(P_REG_GRID)))
ax.set_xticklabels([f"p={p}" for p in P_REG_GRID])
ax.set_yticks(range(len(STEP_LABELS)))
ax.set_yticklabels(STEP_LABELS)
for i in range(len(STEP_LABELS)):
    for j in range(len(P_REG_GRID)):
        ax.text(j, i, f"{heat[i,j]:+.1f}", ha="center", va="center",
                fontsize=8, color="black")
plt.colorbar(im, ax=ax, label="Δ rMAE % vs baseline")
ax.set_title("Рис.C  Per-(шаг, p_reg) Δ rMAE % (красный = хуже, зелёный = лучше)")
plt.tight_layout()
fig.savefig(FIG_DIR / "89_C.png", dpi=120); plt.close()
print("Рис. C сохранён")

# Рис. D — сравнение fixed vs ensemble для каждого шага
x   = np.arange(len(STEP_LABELS)); w = 0.35
fig, ax = plt.subplots(figsize=(10, 4))
b1 = ax.bar(x - w / 2, d_fixed, width=w, label="fixed_16",
            color=["#ef5350" if d > 0 else "#66bb6a" for d in d_fixed], alpha=0.8)
b2 = ax.bar(x + w / 2, d_ens,   width=w, label="ensemble",
            color=["#e53935" if d > 0 else "#43a047" for d in d_ens], alpha=0.8)
for bars, vals in [(b1, d_fixed), (b2, d_ens)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.1 if v >= 0 else -0.1),
                f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", color="white", fontsize=8)
ax.axhline(0, color="white", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(STEP_LABELS)
ax.set_ylabel("Δ rMAE vs baseline ×2.0 (%)")
ax.set_title("Рис.D  fixed_16 vs ensemble по шагам каскада")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "89_D.png", dpi=120); plt.close()
print("Рис. D сохранён")

print("\nГотово. Фигуры: 89_A...D.png")
