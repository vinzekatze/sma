"""
92_full_preg_grid.py — Расширенная сетка p_reg: все целые [4..16].

В скр.83-87 использовалась только чётная сетка [4,6,8,10,12,14,16].
Наблюдение: p=9 иногда лучше p=8 → нечётные значения могут быть оптимальными.

Вопросы:
  1. Растёт ли oracle ceiling при полной целой сетке?
  2. Улучшается ли ensemble (больше разнообразия)?
  3. Каково распределение oracle_p по полной сетке — равномерное или нет?
  4. LOO-select на полной сетке лучше/хуже, чем на чётной?

Сравнение:
  fixed_16       — стандарт p=16, каскад ×2 (baseline)
  oracle_even    — лучший p из [4,6,8,10,12,14,16]  (7 значений, скр.85)
  ensemble_even  — 1/LOO по [4,6,8,10,12,14,16]      (−10%, скр.85)
  loo_even       — LOO-select по чётной сетке         (−6.5%, скр.85)
  oracle_full    — лучший p из [4,5,6,...,16]         (13 значений, новый)
  ensemble_full  — 1/LOO по [4,5,6,...,16]            (новый)
  loo_full       — LOO-select по полной сетке          (новый)

Условия: xi=51 фиксирован для всех p_reg (как в скр.85 → сопоставимые LOO).
Каскад: levels_by_step(p_reg, 2.0) — честный, от P_MAX=64 до p_reg.

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров 1d.
"""
from __future__ import annotations
import time
import json
from pathlib import Path
from collections import Counter
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

P_FIT_BASE  = 16
P_GRID_EVEN = [4, 6, 8, 10, 12, 14, 16]
P_GRID_FULL = list(range(4, 17))          # [4, 5, 6, ..., 16]

METHOD_KEYS = ["fixed_16",
               "oracle_even", "ensemble_even", "loo_even",
               "oracle_full", "ensemble_full", "loo_full"]


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


# ── каскад ────────────────────────────────────────────────────────────────────

def levels_by_step(p_fit: int, step: float = 2.0, p_max: int = P_MAX) -> list[int]:
    levels: list[int] = []; p = p_max
    while True:
        levels.append(p)
        p_next = int(p / step)
        if p_next < max(p_fit, 2): break
        p = p_next
    if levels[-1] != p_fit: levels.append(p_fit)
    return levels


# ── контекст ──────────────────────────────────────────────────────────────────

class _Context:
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


def _forecast(ctx: _Context, p_reg: int) -> tuple[float, float]:
    """Прогноз p_reg с честным каскадом ×2 и acc_ang. Возвращает (pred, loo)."""
    X_full = ctx.X_full; y_base = ctx.y_base
    vec_full = ctx.vec_full; acc_hist = ctx.acc_hist; n = ctx.n

    if len(X_full) < XI_FIXED + 1: return np.nan, np.nan

    levels  = levels_by_step(p_reg)
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
    return _lwr_predict(X_nn, y_nn, query), _lwr_loo(X_nn, y_nn, query)


# ── загрузка ──────────────────────────────────────────────────────────────────

def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend_causal(close)
    return _lp_proj(close / np.maximum(lt, 1e-10), LP_M, LP_D, LP_K, LP_N)


# ── walk-forward ──────────────────────────────────────────────────────────────

def _agg_grid(preds: dict[int, float], loos: dict[int, float],
              grid: list[int], true: float) -> dict:
    """Вычисляет oracle, ensemble, loo_select для заданной сетки."""
    valid = [p for p in grid
             if not np.isnan(preds[p]) and not np.isnan(loos[p])]
    if not valid:
        return {"oracle": np.nan, "ensemble": np.nan, "loo_select": np.nan,
                "oracle_p": None}

    best_p  = min(valid, key=lambda p: abs(preds[p] - true))
    oracle  = preds[best_p]

    inv = np.array([1.0 / (loos[p] + 1e-12) for p in valid])
    w   = inv / inv.sum()
    ens = float(w @ np.array([preds[p] for p in valid]))

    best_loo_p = min(valid, key=lambda p: loos[p])
    loo_sel    = preds[best_loo_p]

    return {"oracle": oracle, "ensemble": ens, "loo_select": loo_sel,
            "oracle_p": best_p}


def wf_ticker(att_full: np.ndarray) -> tuple[dict, list[dict]]:
    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP_WF, n_total - 1, STEP_WF))
    out     = {k: {"errors": [], "signs": []} for k in METHOD_KEYS}
    rows: list[dict] = []

    for t_orig in origins:
        if t_orig + 1 >= n_total: continue
        hist = att_full[:t_orig + 1]
        true = att_full[t_orig + 1]

        ctx = _Context(hist)
        if not ctx.ok: continue

        # baseline fixed_16
        pred_f, _ = _forecast(ctx, P_FIT_BASE)
        if not np.isnan(pred_f):
            out["fixed_16"]["errors"].append(pred_f - true)
            out["fixed_16"]["signs"].append(int(np.sign(pred_f) == np.sign(true)))

        # вычисляем все p_reg из полной сетки (чётная ⊂ полная)
        preds: dict[int, float] = {}; loos: dict[int, float] = {}
        for p in P_GRID_FULL:
            preds[p], loos[p] = _forecast(ctx, p)

        # агрегация по чётной сетке
        r_even = _agg_grid(preds, loos, P_GRID_EVEN, true)
        for mk, rk in [("oracle_even",   r_even["oracle"]),
                       ("ensemble_even", r_even["ensemble"]),
                       ("loo_even",      r_even["loo_select"])]:
            if not np.isnan(rk):
                out[mk]["errors"].append(rk - true)
                out[mk]["signs"].append(int(np.sign(rk) == np.sign(true)))

        # агрегация по полной сетке
        r_full = _agg_grid(preds, loos, P_GRID_FULL, true)
        for mk, rk in [("oracle_full",   r_full["oracle"]),
                       ("ensemble_full", r_full["ensemble"]),
                       ("loo_full",      r_full["loo_select"])]:
            if not np.isnan(rk):
                out[mk]["errors"].append(rk - true)
                out[mk]["signs"].append(int(np.sign(rk) == np.sign(true)))

        rows.append({
            "oracle_p_even": r_even["oracle_p"],
            "oracle_p_full": r_full["oracle_p"],
        })

    return out, rows


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

print(f"\nСетка p_reg:")
print(f"  Чётная ({len(P_GRID_EVEN)} значений): {P_GRID_EVEN}")
print(f"  Полная ({len(P_GRID_FULL)} значений): {P_GRID_FULL}")

print(f"\nWalk-forward ({len(TICKERS)} тикеров)...")
t0 = time.time()

agg  = {k: {"errors": [], "signs": []} for k in METHOD_KEYS}
all_rows: list[dict] = []

for tkr in TICKERS:
    t1  = time.time()
    res, rows = wf_ticker(att_data[tkr])
    for k in METHOD_KEYS:
        agg[k]["errors"].extend(res[k]["errors"])
        agg[k]["signs"].extend(res[k]["signs"])
    all_rows.extend(rows)
    print(f"  {tkr}: {time.time()-t1:.1f}с")
print(f"Walk-forward: {time.time()-t0:.1f}с  |  origins: {len(all_rows)}")


def rmae(k: str) -> float:
    e = agg[k]["errors"]
    return float(np.mean(np.abs(e))) / std_mean if e else np.nan

def sacc(k: str) -> float:
    s = agg[k]["signs"]
    return float(np.mean(s)) * 100.0 if s else np.nan


baseline = rmae("fixed_16")
print(f"\nBaseline: rMAE = {baseline:.4f}")

# ── таблица ───────────────────────────────────────────────────────────────────
LABELS = {
    "fixed_16":      "fixed p=16 (baseline)",
    "oracle_even":   "oracle  чётная [4,6..16]",
    "ensemble_even": "ensemble чётная [4,6..16]",
    "loo_even":      "LOO-select чётная [4,6..16]",
    "oracle_full":   "oracle  полная [4..16]",
    "ensemble_full": "ensemble полная [4..16]",
    "loo_full":      "LOO-select полная [4..16]",
}
print("\n── Итоговая таблица ──")
print(f"{'Метод':38s}  {'rMAE':7s}  {'Δ%':8s}  {'SignAcc':8s}")
for k in METHOD_KEYS:
    r  = rmae(k); d = (r / baseline - 1) * 100 if not np.isnan(r) else np.nan
    sa = sacc(k)
    print(f"  {LABELS[k]:38s}  {r:.4f}  {d:+7.2f}%  {sa:.1f}%")

# ── сравнение oracle ceiling ──────────────────────────────────────────────────
r_oe = rmae("oracle_even"); r_of = rmae("oracle_full")
d_oe = (r_oe / baseline - 1) * 100; d_of = (r_of / baseline - 1) * 100
print(f"\nOracle ceiling:")
print(f"  Чётная сетка: {d_oe:+.2f}%")
print(f"  Полная сетка: {d_of:+.2f}%")
print(f"  Прирост от расширения: {d_of - d_oe:+.2f}%")

r_ee = rmae("ensemble_even"); r_ef = rmae("ensemble_full")
d_ee = (r_ee / baseline - 1) * 100; d_ef = (r_ef / baseline - 1) * 100
print(f"\nEnsemble:")
print(f"  Чётная сетка: {d_ee:+.2f}%")
print(f"  Полная сетка: {d_ef:+.2f}%")
print(f"  Прирост от расширения: {d_ef - d_ee:+.2f}%")

# ── распределение oracle_p ────────────────────────────────────────────────────
print("\n── Распределение oracle_p (полная сетка [4..16]) ──")
op_full = [r["oracle_p_full"] for r in all_rows if r["oracle_p_full"] is not None]
op_even = [r["oracle_p_even"] for r in all_rows if r["oracle_p_even"] is not None]
cnt_f   = Counter(op_full); cnt_e = Counter(op_even)
print(f"  p:  " + "  ".join(f"{p:3d}" for p in range(4, 17)))
print(f"  N:  " + "  ".join(f"{cnt_f.get(p,0):3d}" for p in range(4, 17)))
pct_f   = [cnt_f.get(p, 0) / max(len(op_full), 1) * 100 for p in range(4, 17)]
print(f"  %:  " + "  ".join(f"{v:3.0f}" for v in pct_f))


# ── Фигуры ────────────────────────────────────────────────────────────────────

# Рис. A — bar chart: все методы
SHOW = ["fixed_16", "oracle_even", "ensemble_even", "loo_even",
        "oracle_full", "ensemble_full", "loo_full"]
SLBL = ["fixed\np=16", "oracle\nчётн.", "ens.\nчётн.", "LOO\nчётн.",
        "oracle\nполн.", "ens.\nполн.", "LOO\nполн."]
rv = [rmae(k) for k in SHOW]
dv = [(r / baseline - 1) * 100 if not np.isnan(r) else 0.0 for r in rv]
cols = ["#78909c"] + ["#ef5350" if d > 0 else "#66bb6a" for d in dv[1:]]

fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.bar(SLBL, dv, color=cols, width=0.55)
ax.axhline(0, color="white", lw=0.8)
# граница между чётной и полной сетками
ax.axvline(3.5, color="#ffa726", lw=1.5, ls="--", alpha=0.7)
ax.text(1.5, min(dv) * 0.6, "чётная\n[4,6..16]", ha="center",
        color="#ffa726", fontsize=9)
ax.text(5.0, min(dv) * 0.6, "полная\n[4..16]", ha="center",
        color="#ffa726", fontsize=9)
for bar, v in zip(bars, dv):
    ax.text(bar.get_x() + bar.get_width() / 2,
            v + (0.5 if v >= 0 else -0.5),
            f"{v:+.1f}%", ha="center",
            va="bottom" if v >= 0 else "top", color="white", fontsize=9)
ax.set_ylabel("Δ rMAE vs fixed_16 (%)")
ax.set_title("Рис.A  Чётная vs Полная сетка p_reg  (<0 = лучше baseline)")
plt.tight_layout()
fig.savefig(FIG_DIR / "92_A.png", dpi=120); plt.close()
print("\nРис. A сохранён")

# Рис. B — распределение oracle_p по полной сетке vs чётной
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
x_full = range(4, 17); y_full = [cnt_f.get(p, 0) for p in x_full]
x_even = P_GRID_EVEN;  y_even = [cnt_e.get(p, 0) for p in x_even]
uniform_full = len(op_full) / len(P_GRID_FULL)
uniform_even = len(op_even) / len(P_GRID_EVEN)

axes[0].bar([str(p) for p in x_full], y_full, color="#42a5f5")
axes[0].axhline(uniform_full, color="#ef5350", lw=1.5, ls="--",
                label=f"равномерный={uniform_full:.0f}")
axes[0].set_title(f"Полная сетка [4..16]  (oracle={d_of:+.1f}%)")
axes[0].set_xlabel("p_reg"); axes[0].set_ylabel("origins")
axes[0].legend(fontsize=8)

axes[1].bar([str(p) for p in x_even], y_even, color="#66bb6a")
axes[1].axhline(uniform_even, color="#ef5350", lw=1.5, ls="--",
                label=f"равномерный={uniform_even:.0f}")
axes[1].set_title(f"Чётная сетка [4,6..16]  (oracle={d_oe:+.1f}%)")
axes[1].set_xlabel("p_reg"); axes[1].set_ylabel("origins")
axes[1].legend(fontsize=8)

plt.suptitle("Рис.B  Распределение oracle_p", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "92_B.png", dpi=120); plt.close()
print("Рис. B сохранён")


print("\nГотово. Фигуры: 92_A.png, 92_B.png")
