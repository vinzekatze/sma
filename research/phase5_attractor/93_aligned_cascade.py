"""
93_aligned_cascade.py — P-aligned каскад: уровни согласованы с p_fit.

Идея: для p_fit уровни каскада = p_fit × {1, 2, 4, 8, ...} ≤ P_MAX.
При равномерных лагах выполняется свойство вложенности: p-D ⊂ 2p-D ⊂ 4p-D...
→ сходство на уровне 2p ГАРАНТИРУЕТ сходство в p-мерном подпространстве.
Стандартный каскад [64,32,16,8,p] нарушает это для нечётных p (8D не вложено в 5D).

Примеры уровней (сверху вниз):
  p=4  → [64, 32, 16,  8,  4]   aligned = standard ✓
  p=5  → [40, 20, 10,  5]        vs [64, 32, 16, 8, 5]
  p=6  → [48, 24, 12,  6]        vs [64, 32, 16, 8, 6]
  p=7  → [56, 28, 14,  7]        vs [64, 32, 16, 8, 7]
  p=8  → [64, 32, 16,  8]        aligned = standard ✓
  p=9  → [36, 18,  9]            vs [64, 32, 16, 9]
  p=16 → [64, 32, 16]            aligned = standard ✓

Три варианта:
  std      — стандарт: P_MAX→…→p_fit делением на 2 (из скр.85/92)
  aligned  — чистый p-aligned: max_harmonic(p)→…→p_fit
  extended — [P_MAX, max_harmonic→…→p_fit]: глобальный старт + aligned

Сравнение для полной сетки p ∈ [4..16]:
  oracle, ensemble, LOO-select для каждого из трёх вариантов каскада.

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
P_FIT_BASE = 16
P_GRID     = list(range(4, 17))   # [4..16], 13 значений

CASCADE_VARIANTS = ["std", "aligned", "extended"]
METHOD_KEYS = [f"{m}_{v}"
               for v in CASCADE_VARIANTS
               for m in ("oracle", "ensemble", "loo")]
METHOD_KEYS.insert(0, "fixed_16")   # baseline первым


# ── генераторы каскадов ───────────────────────────────────────────────────────

def levels_std(p_fit: int, p_max: int = P_MAX) -> list[int]:
    """Стандарт: P_MAX / 2^k → … → p_fit (нисходящий, из скр.85/92)."""
    levels: list[int] = []; p = p_max
    while True:
        levels.append(p)
        p_next = p // 2
        if p_next < max(p_fit, 2): break
        p = p_next
    if levels[-1] != p_fit: levels.append(p_fit)
    return levels


def levels_aligned(p_fit: int, p_max: int = P_MAX) -> list[int]:
    """P-aligned: p_fit × 2^k ≤ p_max, нисходящий."""
    levs: list[int] = [p_fit]
    p = p_fit
    while p * 2 <= p_max:
        p *= 2
        levs.append(p)
    return list(reversed(levs))


def levels_extended(p_fit: int, p_max: int = P_MAX) -> list[int]:
    """P-aligned с глобальным стартом P_MAX: [P_MAX] + aligned, без дублей."""
    al = levels_aligned(p_fit, p_max)
    if al[0] == p_max:
        return al
    return [p_max] + al


def show_cascade_table():
    print("── Уровни каскада для каждого p_fit ──")
    print(f"  {'p':3s}  {'std':28s}  {'aligned':24s}  {'extended':28s}")
    for p in P_GRID:
        s = "→".join(str(l) for l in levels_std(p))
        a = "→".join(str(l) for l in levels_aligned(p))
        e = "→".join(str(l) for l in levels_extended(p))
        tag = " ←same" if levels_std(p) == levels_aligned(p) == levels_extended(p) else ""
        print(f"  {p:3d}  {s:28s}  {a:24s}  {e:28s}{tag}")


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
        if n_pts < k_eff + 1: break
        rows = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
        X = s[rows]; tree = KDTree(X)
        _, inds = tree.query(X, k=k_eff + 1)
        X_proj = np.empty_like(X)
        for i in range(n_pts):
            nn = inds[i, 1:]; X_nn = X[nn]; centroid = X_nn.mean(axis=0)
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


def _forecast(ctx: _Context, p_reg: int, levels: list[int]) -> tuple[float, float]:
    """Прогноз с произвольным каскадом + acc_ang. Возвращает (pred, loo)."""
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
    return _lwr_predict(X_nn, y_nn, query), _lwr_loo(X_nn, y_nn, query)


# ── загрузка ──────────────────────────────────────────────────────────────────

def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend_causal(close)
    return _lp_proj(close / np.maximum(lt, 1e-10), LP_M, LP_D, LP_K, LP_N)


# ── walk-forward ──────────────────────────────────────────────────────────────

LEVEL_FNS = {
    "std":      levels_std,
    "aligned":  levels_aligned,
    "extended": levels_extended,
}

def _agg_grid(preds: dict, loos: dict, grid: list[int], true: float) -> dict:
    valid = [p for p in grid if not np.isnan(preds[p]) and not np.isnan(loos[p])]
    if not valid:
        return {"oracle": np.nan, "ensemble": np.nan, "loo": np.nan, "oracle_p": None}
    best_p  = min(valid, key=lambda p: abs(preds[p] - true))
    oracle  = preds[best_p]
    inv = np.array([1.0 / (loos[p] + 1e-12) for p in valid])
    ens = float((inv / inv.sum()) @ np.array([preds[p] for p in valid]))
    loo_sel = preds[min(valid, key=lambda p: loos[p])]
    return {"oracle": oracle, "ensemble": ens, "loo": loo_sel, "oracle_p": best_p}


def wf_ticker(att_full: np.ndarray) -> tuple[dict, list]:
    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP_WF, n_total - 1, STEP_WF))
    out     = {k: {"errors": [], "signs": []} for k in METHOD_KEYS}
    rows    = []

    for t_orig in origins:
        if t_orig + 1 >= n_total: continue
        hist = att_full[:t_orig + 1]; true = att_full[t_orig + 1]
        ctx  = _Context(hist)
        if not ctx.ok: continue

        # baseline fixed_16 std
        pred_f, _ = _forecast(ctx, P_FIT_BASE, levels_std(P_FIT_BASE))
        if not np.isnan(pred_f):
            out["fixed_16"]["errors"].append(pred_f - true)
            out["fixed_16"]["signs"].append(int(np.sign(pred_f) == np.sign(true)))

        row: dict = {}
        for var, lvl_fn in LEVEL_FNS.items():
            preds: dict[int, float] = {}; loos: dict[int, float] = {}
            for p in P_GRID:
                preds[p], loos[p] = _forecast(ctx, p, lvl_fn(p))

            agg = _agg_grid(preds, loos, P_GRID, true)
            for mk in ("oracle", "ensemble", "loo"):
                key = f"{mk}_{var}"; v = agg[mk]
                if not np.isnan(v):
                    out[key]["errors"].append(v - true)
                    out[key]["signs"].append(int(np.sign(v) == np.sign(true)))
            row[f"oracle_p_{var}"] = agg["oracle_p"]
        rows.append(row)

    return out, rows


# ── запуск ────────────────────────────────────────────────────────────────────

show_cascade_table()

print("\nЗагрузка + Local Projective...")
t0 = time.time()
att_data: dict[str, np.ndarray] = {}
for tkr in TICKERS:
    att_data[tkr] = load_att(tkr); print(f"  {tkr}...")
std_all  = [float(np.std(att_data[tkr])) for tkr in TICKERS]
std_mean = float(np.mean(std_all))
print(f"Загрузка: {time.time()-t0:.1f}с")

print(f"\nWalk-forward ({len(TICKERS)} тикеров × 3 варианта каскада × 13 p)...")
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

# ── итоговая таблица ──────────────────────────────────────────────────────────
print("\n── Итоговая таблица ──")
MLABELS = {
    "fixed_16":        "fixed p=16 std (baseline)",
    "oracle_std":      "oracle    стандартный каскад",
    "ensemble_std":    "ensemble  стандартный каскад",
    "loo_std":         "LOO-sel   стандартный каскад",
    "oracle_aligned":  "oracle    p-aligned каскад",
    "ensemble_aligned":"ensemble  p-aligned каскад",
    "loo_aligned":     "LOO-sel   p-aligned каскад",
    "oracle_extended": "oracle    extended каскад",
    "ensemble_extended":"ensemble extended каскад",
    "loo_extended":    "LOO-sel   extended каскад",
}
print(f"{'Метод':42s}  {'rMAE':7s}  {'Δ%':8s}  {'SignAcc':8s}")
for k in METHOD_KEYS:
    r  = rmae(k); d = (r / baseline - 1) * 100 if not np.isnan(r) else np.nan
    sa = sacc(k)
    lbl = MLABELS.get(k, k)
    sep = "  ─" if k in ("oracle_aligned", "oracle_extended") else ""
    print(f"  {lbl:42s}  {r:.4f}  {d:+7.2f}%  {sa:.1f}%{sep}")

# ── ключевое сравнение ────────────────────────────────────────────────────────
print("\n── Сравнение oracle и ensemble ──")
for metric in ("oracle", "ensemble", "loo"):
    rs = rmae(f"{metric}_std")
    ra = rmae(f"{metric}_aligned")
    re = rmae(f"{metric}_extended")
    ds = (rs / baseline - 1) * 100
    da = (ra / baseline - 1) * 100
    de = (re / baseline - 1) * 100
    best = min([(ds, "std"), (da, "aligned"), (de, "extended")], key=lambda x: x[0])
    print(f"  {metric:8s}: std={ds:+.2f}%  aligned={da:+.2f}%  extended={de:+.2f}%"
          f"  → лучший: {best[1]}")

# ── распределение oracle_p ────────────────────────────────────────────────────
print("\n── Распределение oracle_p по вариантам каскада ──")
print(f"  {'p':3s}  " + "  ".join(f"{p:3d}" for p in P_GRID))
for var in CASCADE_VARIANTS:
    ops = [r[f"oracle_p_{var}"] for r in all_rows if r[f"oracle_p_{var}"] is not None]
    cnt = Counter(ops)
    pct = [cnt.get(p, 0) / max(len(ops), 1) * 100 for p in P_GRID]
    print(f"  {var:8s}: " + "  ".join(f"{v:3.0f}" for v in pct) + "%")


# ── Фигуры ────────────────────────────────────────────────────────────────────

# Рис. A — bar chart: oracle и ensemble по трём вариантам
metrics = ["oracle", "ensemble", "loo"]
variants = CASCADE_VARIANTS
x = np.arange(len(variants)); w = 0.25
colors_m = {"oracle": "#42a5f5", "ensemble": "#66bb6a", "loo": "#ffa726"}

fig, ax = plt.subplots(figsize=(10, 4))
for i, metric in enumerate(metrics):
    dv = [(rmae(f"{metric}_{v}") / baseline - 1) * 100 for v in variants]
    bars = ax.bar(x + (i - 1) * w, dv, width=w,
                  color=colors_m[metric], label=metric, alpha=0.85)
    for bar, v in zip(bars, dv):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v - 0.5, f"{v:+.1f}", ha="center", va="top",
                color="white", fontsize=8)
ax.axhline(0, color="white", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(["Стандартный\n[P_MAX→…→p]",
                                       "P-aligned\n[max_harm→…→p]",
                                       "Extended\n[P_MAX + aligned]"])
ax.set_ylabel("Δ rMAE vs fixed_16 (%)")
ax.set_title("Рис.A  Стандартный vs P-aligned vs Extended каскад  (<0 = лучше)")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "93_A.png", dpi=120); plt.close()
print("\nРис. A сохранён")

# Рис. B — per-p: oracle_aligned vs oracle_std (разница по p_reg)
per_p_std  = {}; per_p_aln  = {}; per_p_ext  = {}
# (нужно пересчитать per-p из all_rows через хранение индивидуальных ошибок)
# Упрощение: показываем только распределения oracle_p
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, var, title, color in [
        (axes[0], "std",      "Стандартный",  "#42a5f5"),
        (axes[1], "aligned",  "P-aligned",    "#66bb6a"),
        (axes[2], "extended", "Extended",     "#ffa726")]:
    ops = [r[f"oracle_p_{var}"] for r in all_rows if r[f"oracle_p_{var}"] is not None]
    cnt = Counter(ops)
    y   = [cnt.get(p, 0) for p in P_GRID]
    ax.bar([str(p) for p in P_GRID], y, color=color)
    r_v = rmae(f"oracle_{var}")
    d_v = (r_v / baseline - 1) * 100
    ax.set_title(f"{title}\noracle {d_v:+.1f}%")
    ax.set_xlabel("oracle p_reg"); ax.set_ylabel("origins")
    ax.axhline(len(ops) / len(P_GRID), color="white", ls="--", lw=1, label="равном.")
    ax.legend(fontsize=7)
plt.suptitle("Рис.B  Распределение oracle_p по типу каскада", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "93_B.png", dpi=120); plt.close()
print("Рис. B сохранён")

print("\nГотово. Фигуры: 93_A.png, 93_B.png")
