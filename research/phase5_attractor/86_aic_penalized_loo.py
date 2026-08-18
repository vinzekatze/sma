"""
86_aic_penalized_loo.py — AIC-штраф к LOO для отбора p_reg.

Проблема скр.85: LOO-select смещён к малым p (p=4: 41%), потому что
переопределённые модели (51 точка / 5 параметров) структурно имеют
ниже LOO-дисперсию без связи с качеством прогноза.
Oracle-оптимальное p_reg распределено равномерно (p=4: 18%, p=16: 20%).
Совпадение LOO vs Oracle было 15% (r≈0).

Исправление: добавить штраф за сложность модели:
  penalized_LOO(p) = LOO(p) × (1 + k × (p+1)/xi)

При xi=51: p=4 → ×(1+0.098k), p=16 → ×(1+0.333k).
Штраф смещает отбор от малых p к оракул-распределению.

Также тестируем AICc-поправку (корректировка конечной выборки):
  loo_aicc(p) = LOO(p) × n/(n - p - 1)
где n = xi = 51.

Методы в финальном сравнении:
  fixed_16   — baseline (скр.80-85)
  oracle     — потолок
  loo_raw    — без штрафа (≡ скр.85 loo_select)
  loo_aicc   — AICc-поправка
  loo_best_k — лучший k из сетки
  ensemble   — взвешенное среднее (≡ скр.85)

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
STEP      = 5

P_MAX    = 64
LAMBDA   = 0.01
XI_FIXED = 51

LP_M = 9; LP_D = 3; LP_K = 30; LP_N = 3

P_REG_GRID = [4, 6, 8, 10, 12, 14, 16]
P_FIT_BASE = 16

# Сетка коэффициентов штрафа
K_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


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


def _uniform_octave_levels(p_fit: int, p_max: int, step: float = 2.0) -> list[int]:
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


def _forecast_at_preg(ctx: _Context, p_reg: int) -> tuple[float, float]:
    """Честный каскад до p_reg, xi=XI_FIXED. Возвращает (pred, loo)."""
    levels   = _uniform_octave_levels(p_reg, P_MAX, 2.0)
    X_full   = ctx.X_full; y_base = ctx.y_base
    vec_full = ctx.vec_full; acc_hist = ctx.acc_hist; n = ctx.n

    if len(X_full) < XI_FIXED + 1:
        return np.nan, np.nan

    t_arr   = np.arange(P_MAX, n - 1)
    X_acc   = np.column_stack([acc_hist[t_arr - (p_reg - 1 - j)] for j in range(p_reg)])
    vec_acc = acc_hist[-p_reg:].copy()
    query   = vec_full[-p_reg:]

    cands = np.arange(len(X_full))
    for k, p_lvl in enumerate(levels):
        is_last = (k == len(levels) - 1)
        if not is_last:
            xi_lvl = min(XI_FIXED, len(cands))
            if len(cands) > xi_lvl:
                d = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full[-p_lvl:], axis=1)
                cands = cands[np.argpartition(d, xi_lvl - 1)[:xi_lvl]]
            p_next = levels[k + 1]; radius = p_lvl - p_next
            exp    = cands[:, None] - np.arange(radius + 1)[None, :]
            cands  = np.unique(np.clip(exp, 0, len(X_full) - 1))

    if len(cands) > XI_FIXED:
        d_pos  = np.linalg.norm(X_full[cands, -p_reg:] - query, axis=1)
        d_acc  = _cosine_dist(X_acc[cands], vec_acc)
        d_comb = d_pos + LAMBDA * d_acc
        sel    = np.argpartition(d_comb, XI_FIXED - 1)[:XI_FIXED]
        cands  = cands[sel]

    if len(cands) < p_reg + 2:
        return np.nan, np.nan

    X_nn = X_full[cands, -p_reg:]
    y_nn = y_base[cands]
    return _lwr_predict(X_nn, y_nn, query), _lwr_loo_error(X_nn, y_nn, query)


def _penalized_loo(loo: float, p_reg: int, k_pen: float) -> float:
    """LOO × (1 + k_pen × (p+1)/xi)."""
    return loo * (1.0 + k_pen * (p_reg + 1) / XI_FIXED)


def _aicc_loo(loo: float, p_reg: int) -> float:
    """AICc-поправка: LOO × n/(n - p - 1)."""
    n   = XI_FIXED
    dof = n - (p_reg + 1)
    if dof <= 0:
        return np.inf
    return loo * n / dof


# ── walk-forward ──────────────────────────────────────────────────────────────

def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend_causal(close)
    ratio = close / np.maximum(lt, 1e-10)
    return _lp_proj(ratio, LP_M, LP_D, LP_K, LP_N)


def wf_ticker(att_full: np.ndarray) -> dict:
    """
    Возвращает словарь:
      results[method_key] = {"errors": [], "signs": [], "chosen_p": []}
    Методы: fixed_16, oracle, loo_raw, loo_aicc, loo_k{v} для v в K_GRID, ensemble.
    """
    all_keys = (["fixed_16", "oracle", "loo_raw", "loo_aicc", "ensemble"] +
                [f"loo_k{k}" for k in K_GRID])
    out: dict[str, dict] = {k: {"errors": [], "signs": [], "chosen_p": []}
                            for k in all_keys}

    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP, n_total - 1, STEP))

    for t_orig in origins:
        if t_orig + 1 >= n_total:
            continue
        hist = att_full[:t_orig + 1]
        true = att_full[t_orig + 1]

        ctx = _Context(hist)
        if not ctx.ok:
            continue

        # baseline
        pred_base, _ = _forecast_at_preg(ctx, P_FIT_BASE)
        if not np.isnan(pred_base):
            out["fixed_16"]["errors"].append(pred_base - true)
            out["fixed_16"]["signs"].append(int(np.sign(pred_base) == np.sign(true)))

        # собираем pred и LOO для всех p_reg
        preds:    dict[int, float] = {}
        loo_errs: dict[int, float] = {}
        for p_reg in P_REG_GRID:
            pred, loo = _forecast_at_preg(ctx, p_reg)
            preds[p_reg]    = pred
            loo_errs[p_reg] = loo

        valid_p = [p for p in P_REG_GRID
                   if not np.isnan(preds[p]) and not np.isnan(loo_errs[p])]
        if not valid_p:
            continue

        def _record(key: str, p: int) -> None:
            out[key]["errors"].append(preds[p] - true)
            out[key]["signs"].append(int(np.sign(preds[p]) == np.sign(true)))
            out[key]["chosen_p"].append(p)

        # oracle
        best_ora = min(valid_p, key=lambda p: abs(preds[p] - true))
        _record("oracle", best_ora)

        # loo_raw (без штрафа)
        best_raw = min(valid_p, key=lambda p: loo_errs[p])
        _record("loo_raw", best_raw)

        # loo_aicc
        best_aicc = min(valid_p, key=lambda p: _aicc_loo(loo_errs[p], p))
        _record("loo_aicc", best_aicc)

        # loo_k{v} для каждого k в K_GRID
        for k_val in K_GRID:
            best_k = min(valid_p, key=lambda p: _penalized_loo(loo_errs[p], p, k_val))
            _record(f"loo_k{k_val}", best_k)

        # ensemble 1/LOO
        inv_loo  = np.array([1.0 / (loo_errs[p] + 1e-12) for p in valid_p])
        w        = inv_loo / inv_loo.sum()
        pred_ens = float(w @ np.array([preds[p] for p in valid_p]))
        out["ensemble"]["errors"].append(pred_ens - true)
        out["ensemble"]["signs"].append(int(np.sign(pred_ens) == np.sign(true)))

    return out


# ── основной блок ─────────────────────────────────────────────────────────────

print("Загрузка + Local Projective...")
t0 = time.time()
att_data: dict[str, np.ndarray] = {}
for tkr in TICKERS:
    att_data[tkr] = load_att(tkr)
    print(f"  {tkr}... готово")
std_all  = [float(np.std(att_data[tkr])) for tkr in TICKERS]
std_mean = float(np.mean(std_all))
print(f"Загрузка итого: {time.time()-t0:.1f}с")

print(f"\nWalk-forward: xi={XI_FIXED}, {len(K_GRID)} значений k...")
t0 = time.time()

all_keys = (["fixed_16", "oracle", "loo_raw", "loo_aicc", "ensemble"] +
            [f"loo_k{k}" for k in K_GRID])
all_res: dict[str, dict] = {k: {"errors": [], "signs": [], "chosen_p": []}
                            for k in all_keys}

for tkr in TICKERS:
    t1  = time.time()
    res = wf_ticker(att_data[tkr])
    for key in all_keys:
        for m in ("errors", "signs", "chosen_p"):
            all_res[key][m].extend(res[key][m])
    print(f"  {tkr}: {time.time()-t1:.1f}с")
print(f"Walk-forward завершён за {time.time()-t0:.1f}с")


def rmae(key: str) -> float:
    e = all_res[key]["errors"]
    return float(np.mean(np.abs(e))) / std_mean if e else np.nan

def sacc(key: str) -> float:
    s = all_res[key]["signs"]
    return float(np.mean(s)) * 100.0 if s else np.nan

baseline = rmae("fixed_16")
ora_arr  = np.array(all_res["oracle"]["chosen_p"])

print("\nИтоговая таблица:")
for key in ["fixed_16", "oracle", "loo_raw", "loo_aicc", "ensemble"] + [f"loo_k{k}" for k in K_GRID]:
    r  = rmae(key); d = (r / baseline - 1) * 100
    sa = sacc(key)
    cp = all_res[key]["chosen_p"]
    dist_str = ""
    if cp:
        cnt = Counter(cp)
        dist_str = "  p=(" + " ".join(f"{p}:{cnt.get(p,0)*100//len(cp)}%" for p in P_REG_GRID) + ")"
        # LOO vs oracle correlation
        loo_arr = np.array(cp[:len(ora_arr)])
        if len(loo_arr) == len(ora_arr) and len(loo_arr) > 1:
            match = float(np.mean(ora_arr == loo_arr)) * 100
            r_cor = float(np.corrcoef(ora_arr, loo_arr)[0, 1]) if len(loo_arr) > 2 else 0.0
            dist_str += f"  match={match:.0f}% r={r_cor:.3f}"
    print(f"  {key:15s}: rMAE={r:.4f} ({d:+.2f}%)  SignAcc={sa:.1f}%{dist_str}")

# Найти лучший k по rMAE
k_rmae = {k: rmae(f"loo_k{k}") for k in K_GRID}
best_k = min(K_GRID, key=lambda k: k_rmae[k])
print(f"\nЛучший k по rMAE: k={best_k}  → rMAE={k_rmae[best_k]:.4f} ({(k_rmae[best_k]/baseline-1)*100:+.2f}%)")


# ── фигуры ────────────────────────────────────────────────────────────────────

# Рис. A — кривая rMAE vs k (сетка штрафов)
k_vals    = K_GRID
rmae_k    = [rmae(f"loo_k{k}") for k in k_vals]
delta_k   = [(r / baseline - 1) * 100 for r in rmae_k]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(k_vals, delta_k, "o-", color="#42a5f5", lw=2, label="LOO-select (k-штраф)")
ax.axhline((rmae("loo_aicc") / baseline - 1) * 100, color="#ffa726", ls="--", lw=1.5,
           label=f"LOO-AICc  ({(rmae('loo_aicc')/baseline-1)*100:+.1f}%)")
ax.axhline((rmae("ensemble") / baseline - 1) * 100, color="#ab47bc", ls="--", lw=1.5,
           label=f"Ensemble  ({(rmae('ensemble')/baseline-1)*100:+.1f}%)")
ax.axhline(0, color="#90a4ae", ls=":", lw=1, label="fixed_16 (baseline)")
ax.axhline((rmae("oracle") / baseline - 1) * 100, color="#ef5350", ls=":", lw=1,
           label=f"Oracle  ({(rmae('oracle')/baseline-1)*100:+.1f}%)")
ax.set_xlabel("k (коэффициент штрафа)")
ax.set_ylabel("Δ rMAE vs fixed acc_ang (%)")
ax.set_title(f"Рис.A  rMAE vs штраф AIC  (8 тикеров, {N_ORIGINS}×STEP={STEP})")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "86_A.png", dpi=120)
plt.close()
print("Рис. A сохранён")

# Рис. B — распределения p_reg для oracle, loo_raw, loo_aicc, loo_best_k
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
cases = [("oracle", "#ef5350"), ("loo_raw", "#66bb6a"),
         ("loo_aicc", "#ffa726"), (f"loo_k{best_k}", "#42a5f5")]
for ax, (key, color) in zip(axes, cases):
    cp  = all_res[key]["chosen_p"]
    cnt = Counter(cp)
    y   = [cnt.get(p, 0) / max(len(cp), 1) * 100 for p in P_REG_GRID]
    ax.bar([str(p) for p in P_REG_GRID], y, color=color)
    ax.set_xlabel("p_reg"); ax.set_ylabel("% origins")
    d   = (rmae(key) / baseline - 1) * 100
    ax.set_title(f"{key}\n{d:+.1f}%")
plt.suptitle("Рис.B  Распределение выбранного p_reg по критериям", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "86_B.png", dpi=120)
plt.close()
print("Рис. B сохранён")

# Рис. C — сводная таблица по всем методам (bar chart)
final_keys = ["fixed_16", "oracle", "loo_raw", "loo_aicc",
              f"loo_k{best_k}", "ensemble"]
final_labels = ["fixed\np=16", "oracle", "LOO\nraw", "LOO\nAICc",
                f"LOO\nk={best_k}", "ensemble"]
rmae_f  = [rmae(k) for k in final_keys]
delta_f = [(r / baseline - 1) * 100 for r in rmae_f]
c_list  = ["#78909c"] + ["#66bb6a" if d < 0 else "#ef5350" for d in delta_f[1:]]

fig, ax = plt.subplots(figsize=(11, 4))
bars = ax.bar(final_labels, delta_f, color=c_list, width=0.5)
ax.axhline(0, color="white", lw=0.8)
for bar, v in zip(bars, delta_f):
    ax.text(bar.get_x() + bar.get_width() / 2, v + (0.5 if v >= 0 else -0.5),
            f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top",
            color="white", fontsize=10)
ax.set_ylabel("Δ rMAE vs fixed acc_ang p=16 (%)")
ax.set_title("Рис.C  Итог: LOO с разными штрафами  (<0 = лучше baseline)")
plt.tight_layout()
fig.savefig(FIG_DIR / "86_C.png", dpi=120)
plt.close()
print("Рис. C сохранён")

print(f"\nГотово. Фигуры: 86_A...C.png")
