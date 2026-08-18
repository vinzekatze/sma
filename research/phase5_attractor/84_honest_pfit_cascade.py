"""
84_honest_pfit_cascade.py — Честный per-p_reg каскад.

Проблема скр.83: каскад строился с фиксированным P_FIT=16, поэтому соседи
отбирались под 16-мерное расстояние. При регрессии с p_reg=4 они оказывались
"лучшими" не потому что p=4 точнее, а потому что несогласованный отбор
систематически смещал результат.

Исправление: для каждого кандидата p_reg строим НЕЗАВИСИМЫЙ каскад
от P_MAX=64 до этого p_reg. Соседи отобраны под то же измерение,
с которым работает LWR.

Дополнительно: ансамблирование предсказаний по всем p_reg,
взвешенное обратно пропорционально LOO-ошибке.

Методы:
  fixed_16    — acc_ang, каскад до P_FIT=16 (baseline скр.80-83)
  oracle      — лучший p_reg per origin (знает истину)
  loo_select  — выбор p_reg по LOO (честный каскад)
  ensemble    — взвешенное среднее всех p_reg (веса ∝ 1/LOO)

Разделение ответственности:
  _build_context(att) — строит X_full (P_MAX колонок) и acc один раз
  _cascade_for_preg(ctx, p_reg) — независимый каскад + acc_ang + LWR+LOO

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров 1d (≡ скр.80-84).
Метрики: rMAE = MAE/std(att), SignAcc = % правильных знаков.
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

P_MAX  = 64    # верхний уровень каскада (общий для всех p_reg)
LAMBDA = 0.01  # acc_ang λ

LP_M = 9; LP_D = 3; LP_K = 30; LP_N = 3

# Сетка кандидатов для p_reg; xi_lwr = 3*(p+1) для каждого
P_REG_GRID = [4, 6, 8, 10, 12, 14, 16]

# Единый baseline P_FIT для fixed_16
P_FIT_BASE = 16
XI_BASE    = 3 * (P_FIT_BASE + 1)   # 51


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
    """LOO-CV через hat matrix WLS: O(n·p²), без перебора."""
    dists = np.linalg.norm(X_nn - vec_f, axis=1)
    h_bw  = max(float(dists.max()), 1e-10)
    w     = np.maximum(np.exp(-0.5 * (dists / h_bw) ** 2), 1e-30)
    A     = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw    = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    y_hat = A @ c
    resid = y_nn - y_hat
    try:
        Aw       = sw[:, None] * A
        AtWA_inv = np.linalg.inv(Aw.T @ Aw + 1e-10 * np.eye(A.shape[1]))
        AiM      = A @ AtWA_inv
        h_diag   = w * np.einsum("ij,ij->i", AiM, A)
        denom    = 1.0 - h_diag
        safe     = np.abs(denom) > 1e-6
        e_loo    = np.where(safe, resid / denom, resid)
    except np.linalg.LinAlgError:
        e_loo = resid
    return float(np.mean(np.abs(e_loo)))


def _uniform_octave_levels(p_fit: int, p_max: int, step: float = 2.0) -> list[int]:
    if step >= 9999.0:
        return [p_fit]
    if step >= 999.0:
        return [p_max] if p_max == p_fit else [p_max, p_fit]
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


# ── ядро прогноза ─────────────────────────────────────────────────────────────

class _Context:
    """Разделяемый контекст: X_full и acc для одного origin."""
    __slots__ = ("X_full", "y_base", "acc_hist", "vec_full", "n", "ok")

    def __init__(self, att: np.ndarray) -> None:
        n = len(att)
        self.n = n; self.ok = False
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
    """
    Честный каскад от P_MAX до p_reg + acc_ang + LWR.
    Возвращает (pred, loo_error). Если данных мало — (nan, nan).
    """
    xi     = 3 * (p_reg + 1)
    levels = _uniform_octave_levels(p_reg, P_MAX, 2.0)

    n      = ctx.n
    X_full = ctx.X_full
    y_base = ctx.y_base

    if len(X_full) < xi + 1:
        return np.nan, np.nan

    vec_full = ctx.vec_full
    acc_hist = ctx.acc_hist

    # X_acc и vec_acc для p_reg-масштаба acc_ang
    t_arr = np.arange(P_MAX, n - 1)
    X_acc = np.column_stack([acc_hist[t_arr - (p_reg - 1 - j)] for j in range(p_reg)])
    vec_acc = acc_hist[-p_reg:].copy()
    query   = vec_full[-p_reg:]

    # Каскад (открытый: последний уровень без финального фильтра)
    cands = np.arange(len(X_full))
    for k, p_lvl in enumerate(levels):
        is_last = (k == len(levels) - 1)
        if not is_last:
            xi_lvl = min(xi, len(cands))
            if len(cands) > xi_lvl:
                d = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full[-p_lvl:], axis=1)
                cands = cands[np.argpartition(d, xi_lvl - 1)[:xi_lvl]]
            p_next = levels[k + 1]; radius = p_lvl - p_next
            exp    = cands[:, None] - np.arange(radius + 1)[None, :]
            cands  = np.unique(np.clip(exp, 0, len(X_full) - 1))

    # acc_ang: финальный отбор xi соседей
    if len(cands) > xi:
        d_pos  = np.linalg.norm(X_full[cands, -p_reg:] - query, axis=1)
        d_acc  = _cosine_dist(X_acc[cands], vec_acc)
        d_comb = d_pos + LAMBDA * d_acc
        sel    = np.argpartition(d_comb, xi - 1)[:xi]
        cands  = cands[sel]

    if len(cands) < p_reg + 2:
        return np.nan, np.nan

    X_nn  = X_full[cands, -p_reg:]
    y_nn  = y_base[cands]

    pred    = _lwr_predict(X_nn, y_nn, query)
    loo_err = _lwr_loo_error(X_nn, y_nn, query)
    return pred, loo_err


def _forecast_fixed_baseline(ctx: _Context) -> float:
    """Baseline: каскад до P_FIT_BASE=16, acc_ang, xi=51."""
    xi     = XI_BASE
    levels = _uniform_octave_levels(P_FIT_BASE, P_MAX, 2.0)

    X_full   = ctx.X_full
    y_base   = ctx.y_base
    vec_full = ctx.vec_full
    acc_hist = ctx.acc_hist
    n        = ctx.n

    if len(X_full) < xi + 1:
        return np.nan

    t_arr   = np.arange(P_MAX, n - 1)
    X_acc   = np.column_stack([acc_hist[t_arr - (P_FIT_BASE - 1 - j)]
                                for j in range(P_FIT_BASE)])
    vec_acc = acc_hist[-P_FIT_BASE:].copy()
    query   = vec_full[-P_FIT_BASE:]

    cands = np.arange(len(X_full))
    for k, p_lvl in enumerate(levels):
        is_last = (k == len(levels) - 1)
        if not is_last:
            xi_lvl = min(xi, len(cands))
            if len(cands) > xi_lvl:
                d = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full[-p_lvl:], axis=1)
                cands = cands[np.argpartition(d, xi_lvl - 1)[:xi_lvl]]
            p_next = levels[k + 1]; radius = p_lvl - p_next
            exp    = cands[:, None] - np.arange(radius + 1)[None, :]
            cands  = np.unique(np.clip(exp, 0, len(X_full) - 1))

    if len(cands) > xi:
        d_pos  = np.linalg.norm(X_full[cands, -P_FIT_BASE:] - query, axis=1)
        d_acc  = _cosine_dist(X_acc[cands], vec_acc)
        d_comb = d_pos + LAMBDA * d_acc
        sel    = np.argpartition(d_comb, xi - 1)[:xi]
        cands  = cands[sel]

    if len(cands) < P_FIT_BASE + 2:
        return np.nan

    return _lwr_predict(X_full[cands, -P_FIT_BASE:], y_base[cands], query)


# ── walk-forward ──────────────────────────────────────────────────────────────

def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend_causal(close)
    ratio = close / np.maximum(lt, 1e-10)
    return _lp_proj(ratio, LP_M, LP_D, LP_K, LP_N)


def wf_ticker(att_full: np.ndarray) -> tuple[dict, list[int], list[int]]:
    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP, n_total - 1, STEP))
    out = {k: {"errors": [], "signs": []} for k in
           ("fixed_16", "oracle", "loo_select", "ensemble")}
    oracle_p_hist: list[int] = []
    loo_p_hist:    list[int] = []

    for t_orig in origins:
        if t_orig + 1 >= n_total:
            continue
        hist = att_full[:t_orig + 1]
        true = att_full[t_orig + 1]

        ctx = _Context(hist)
        if not ctx.ok:
            continue

        # baseline
        pred_base = _forecast_fixed_baseline(ctx)
        if not np.isnan(pred_base):
            out["fixed_16"]["errors"].append(pred_base - true)
            out["fixed_16"]["signs"].append(int(np.sign(pred_base) == np.sign(true)))

        # честные каскады для всех p_reg
        preds:    dict[int, float] = {}
        loo_errs: dict[int, float] = {}
        for p_reg in P_REG_GRID:
            pred, loo = _forecast_at_preg(ctx, p_reg)
            preds[p_reg]    = pred
            loo_errs[p_reg] = loo

        # oracle: знает истину
        valid_p = [p for p in P_REG_GRID if not np.isnan(preds[p])]
        if valid_p:
            best_p_ora = min(valid_p, key=lambda p: abs(preds[p] - true))
            pred_ora   = preds[best_p_ora]
            out["oracle"]["errors"].append(pred_ora - true)
            out["oracle"]["signs"].append(int(np.sign(pred_ora) == np.sign(true)))
            oracle_p_hist.append(best_p_ora)

        # loo_select: минимальная LOO-ошибка
        valid_loo = [p for p in P_REG_GRID
                     if not np.isnan(preds[p]) and not np.isnan(loo_errs[p])]
        if valid_loo:
            best_p_loo = min(valid_loo, key=lambda p: loo_errs[p])
            pred_loo   = preds[best_p_loo]
            out["loo_select"]["errors"].append(pred_loo - true)
            out["loo_select"]["signs"].append(int(np.sign(pred_loo) == np.sign(true)))
            loo_p_hist.append(best_p_loo)

        # ensemble: взвешено 1/loo_err
        ens_p  = [p for p in P_REG_GRID
                  if not np.isnan(preds[p]) and not np.isnan(loo_errs[p])
                     and loo_errs[p] > 0]
        if ens_p:
            inv_loo = np.array([1.0 / (loo_errs[p] + 1e-12) for p in ens_p])
            w       = inv_loo / inv_loo.sum()
            pred_ens = float(w @ np.array([preds[p] for p in ens_p]))
            out["ensemble"]["errors"].append(pred_ens - true)
            out["ensemble"]["signs"].append(int(np.sign(pred_ens) == np.sign(true)))

    return out, oracle_p_hist, loo_p_hist


# ── основной блок ─────────────────────────────────────────────────────────────

print("Загрузка + Local Projective...")
t0 = time.time()
att_data: dict[str, np.ndarray] = {}
for tkr in TICKERS:
    t1 = time.time()
    att_data[tkr] = load_att(tkr)
    print(f"  {tkr}... {time.time()-t1:.1f}с")
std_all  = [float(np.std(att_data[tkr])) for tkr in TICKERS]
std_mean = float(np.mean(std_all))
print(f"Загрузка итого: {time.time()-t0:.1f}с")

print(f"\nWalk-forward: {len(P_REG_GRID)} честных каскадов на origin...")
t0 = time.time()

all_res: dict[str, dict] = {k: {"errors": [], "signs": []} for k in
                             ("fixed_16", "oracle", "loo_select", "ensemble")}
oracle_p_all: list[int] = []
loo_p_all:    list[int] = []

for tkr in TICKERS:
    t1  = time.time()
    att = att_data[tkr]
    res, op, lp = wf_ticker(att)
    for key in all_res:
        for m in ("errors", "signs"):
            all_res[key][m].extend(res[key][m])
    oracle_p_all.extend(op)
    loo_p_all.extend(lp)
    print(f"  {tkr}: {time.time()-t1:.1f}с")

print(f"Walk-forward завершён за {time.time()-t0:.1f}с")


def rmae(key: str) -> float:
    e = all_res[key]["errors"]
    return float(np.mean(np.abs(e))) / std_mean if e else np.nan

def sacc(key: str) -> float:
    s = all_res[key]["signs"]
    return float(np.mean(s)) * 100.0 if s else np.nan

baseline = rmae("fixed_16")

print("\nИтоговая таблица:")
KEYS = ["fixed_16", "oracle", "loo_select", "ensemble"]
for key in KEYS:
    r = rmae(key); d = (r / baseline - 1) * 100
    print(f"  {key:12s}: rMAE={r:.4f} ({d:+.2f}%)  SignAcc={sacc(key):.1f}%")

print("\nOracle: распределение p_reg:")
for p, cnt in sorted(Counter(oracle_p_all).items()):
    print(f"  p={p:2d}: {cnt:4d} ({cnt/max(len(oracle_p_all),1)*100:.1f}%)")

print("\nLOO-select: распределение p_reg:")
for p, cnt in sorted(Counter(loo_p_all).items()):
    print(f"  p={p:2d}: {cnt:4d} ({cnt/max(len(loo_p_all),1)*100:.1f}%)")


# ── фигуры ────────────────────────────────────────────────────────────────────

LABELS  = ["fixed\nacc_ang\np=16", "oracle\n(best p)", "LOO-\nselect", "ensemble\n1/LOO"]
rmae_v  = [rmae(k) for k in KEYS]
delta_v = [(r / baseline - 1) * 100 for r in rmae_v]
colors  = ["#78909c"] + ["#66bb6a" if d < 0 else "#ef5350" for d in delta_v[1:]]

# Рис. A — сравнение методов
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(LABELS, delta_v, color=colors, width=0.5)
ax.axhline(0, color="white", lw=0.8)
for bar, v in zip(bars, delta_v):
    ax.text(bar.get_x() + bar.get_width() / 2, v + (0.3 if v >= 0 else -0.3),
            f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top",
            color="white", fontsize=10)
ax.set_ylabel("Δ rMAE vs fixed acc_ang p=16 (%)")
ax.set_title(f"Рис.A  Честный per-p каскад  (8 тикеров, {N_ORIGINS}×STEP={STEP})")
plt.tight_layout()
fig.savefig(FIG_DIR / "84_A.png", dpi=120)
plt.close()
print("Рис. A сохранён")

# Рис. B — распределения выбранного p_reg
x   = P_REG_GRID
y_o = [Counter(oracle_p_all).get(p, 0) / max(len(oracle_p_all), 1) * 100 for p in x]
y_l = [Counter(loo_p_all).get(p, 0)   / max(len(loo_p_all),   1) * 100 for p in x]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar([str(p) for p in x], y_o, color="#ffa726")
axes[0].set_xlabel("p_reg"); axes[0].set_ylabel("% origins")
axes[0].set_title(f"Oracle — p_reg\nrMAE={rmae('oracle'):.4f} ({delta_v[1]:+.1f}%)")
axes[1].bar([str(p) for p in x], y_l, color="#42a5f5")
axes[1].set_xlabel("p_reg"); axes[1].set_ylabel("% origins")
axes[1].set_title(f"LOO-select — p_reg\nrMAE={rmae('loo_select'):.4f} ({delta_v[2]:+.1f}%)")
plt.tight_layout()
fig.savefig(FIG_DIR / "84_B.png", dpi=120)
plt.close()
print("Рис. B сохранён")

# Рис. C — per-ticker по всем методам
ticker_rows: dict[str, dict[str, float]] = {tkr: {} for tkr in TICKERS}
print("\nPer-ticker (второй проход)...")
for tkr in TICKERS:
    att = att_data[tkr]; std = float(np.std(att))
    res, _, _ = wf_ticker(att)
    for key in KEYS:
        e = res[key]["errors"]
        ticker_rows[tkr][key] = float(np.mean(np.abs(e))) / std if e else np.nan

x_t = np.arange(len(TICKERS)); w = 0.2
fig, ax = plt.subplots(figsize=(13, 4))
method_colors = {"oracle": "#ffa726", "loo_select": "#42a5f5", "ensemble": "#ab47bc"}
for i, (key, color) in enumerate([("oracle", "#ffa726"), ("loo_select", "#42a5f5"),
                                    ("ensemble", "#ab47bc")]):
    base_t = np.array([ticker_rows[tkr]["fixed_16"] for tkr in TICKERS])
    val_t  = np.array([ticker_rows[tkr][key]        for tkr in TICKERS])
    delta_t = (val_t / base_t - 1) * 100
    ax.bar(x_t + (i - 1) * w, delta_t, width=w, color=color, label=key)
ax.axhline(0, color="white", lw=0.8)
ax.set_xticks(x_t); ax.set_xticklabels(TICKERS)
ax.set_ylabel("Δ rMAE vs fixed acc_ang (%)")
ax.set_title("Рис.C  Per-ticker: oracle / LOO / ensemble vs fixed  (<0 = лучше)")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "84_C.png", dpi=120)
plt.close()
print("Рис. C сохранён")

print(f"\nГотово. Фигуры: 84_A...C.png")
