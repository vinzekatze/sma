"""
91_spec_cascade.py — Спектр-адаптивный шаг каскада.

Гипотеза: аттрактор att имеет локальную доминирующую периодичность T,
меняющуюся от origin к origin. Если шаг каскада подобрать так, чтобы
промежуточный уровень попал вблизи T, фильтрация соседей будет точнее.

Подход: ACF(att[-W:]) → первый значимый локальный максимум = T_local.
Шаг каскада: P_MAX / T_local → снэп к сетке {√2, 1.5, 2.0, 2.5, 3.0, 4.0}.
При P_MAX=64: T=32 → ×2.0; T=20 → ×3.0 (64/3≈21); T=40 → ×1.5 (64/1.5≈42).

Дополнительно тестируем PSD-подход (Welch), чтобы сравнить с ACF.

Методы:
  fixed_2          — стандарт ×2, p=16 (baseline)
  ensemble_2       — ensemble 1/LOO ×2 (reference −10%)
  spec_acf_p16     — ACF-адаптивный шаг, p_reg=16 fixed
  spec_psd_p16     — PSD-адаптивный шаг, p_reg=16 fixed
  spec_acf_ens     — ACF-адаптивный шаг, ensemble (все p_reg)
  oracle_step_p16  — oracle: лучший шаг из сетки по p=16 на каждый origin
                     (потолок для spec-adaptive при фиксированном p)

Диагностика: распределение выбранных шагов vs oracle шагов.

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров 1d.
"""
from __future__ import annotations
import time
import json
import math
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.signal import welch, find_peaks
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
STEP_NAMES  = ["√2", "1.5", "2.0", "2.5", "3.0", "4.0"]

ACF_WINDOW  = 128   # баров для оценки локального периода
MIN_PERIOD  = 4     # минимальный период att (LP-граница ≈8, но возьмём с запасом)
MAX_PERIOD  = P_MAX # ограничиваем сверху


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


# ── оценка локального периода ─────────────────────────────────────────────────

def period_via_acf(att_window: np.ndarray) -> float | None:
    """
    Первый локальный максимум нормированного ACF за пределами min_period.
    Возвращает период в барах или None если нет чёткого пика.
    """
    x = att_window - att_window.mean()
    std = x.std()
    if std < 1e-12: return None
    x = x / std; n = len(x)
    max_lag = min(MAX_PERIOD, n // 2)
    acf = np.array([float(np.dot(x[:n-lag], x[lag:])) / (n - lag)
                    for lag in range(max_lag + 1)])
    acf /= max(float(acf[0]), 1e-12)

    for i in range(MIN_PERIOD, max_lag - 1):
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.05:
            return float(i)
    return None


def period_via_psd(att_window: np.ndarray) -> float | None:
    """
    Доминирующий период через PSD (Welch). Возвращает период в барах или None.
    """
    n = len(att_window)
    if n < 16: return None
    nperseg = min(n // 2, 64)
    try:
        freqs, psd = welch(att_window, nperseg=nperseg)
    except Exception:
        return None
    # убираем DC и слишком высокие частоты
    valid = (freqs > 1.0 / MAX_PERIOD) & (freqs < 1.0 / MIN_PERIOD)
    if valid.sum() < 3: return None
    f_valid = freqs[valid]; p_valid = psd[valid]
    peak_idx = int(np.argmax(p_valid))
    f_peak = f_valid[peak_idx]
    if f_peak < 1e-6: return None
    return float(1.0 / f_peak)


def period_to_step(T: float | None, fallback: float = 2.0) -> float:
    """
    T (доминирующий период att) → шаг каскада.
    Логика: хотим, чтобы первый уровень ниже P_MAX попал вблизи T.
    ratio = P_MAX / T, снэп к сетке.
    """
    if T is None or T < MIN_PERIOD: return fallback
    T = min(T, MAX_PERIOD - 1)
    ratio = P_MAX / T
    if ratio < STEPS_GRID[0]: return STEPS_GRID[0]
    if ratio > STEPS_GRID[-1]: return STEPS_GRID[-1]
    return min(STEPS_GRID, key=lambda s: abs(s - ratio))


def step_name(s: float) -> str:
    return min(STEP_NAMES, key=lambda n: abs(float(n) - s))


# ── уровни каскада ────────────────────────────────────────────────────────────

def levels_by_step(p_fit: int, step: float, p_max: int = P_MAX) -> list[int]:
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


def _forecast(ctx: _Context, p_reg: int,
              levels: list[int], need_loo: bool = True) -> tuple[float, float]:
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


# ── загрузка ──────────────────────────────────────────────────────────────────

def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend_causal(close)
    return _lp_proj(close / np.maximum(lt, 1e-10), LP_M, LP_D, LP_K, LP_N)


# ── walk-forward ──────────────────────────────────────────────────────────────

METHOD_KEYS = ["fixed_2", "ensemble_2",
               "spec_acf_p16", "spec_psd_p16", "spec_acf_ens",
               "oracle_step_p16"]

def wf_ticker(att_full: np.ndarray) -> tuple[dict, list[dict]]:
    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP_WF, n_total - 1, STEP_WF))
    out  = {k: {"errors": [], "signs": []} for k in METHOD_KEYS}
    rows: list[dict] = []   # для диагностики: chosen_step, oracle_step, T

    for t_orig in origins:
        if t_orig + 1 >= n_total: continue
        hist = att_full[:t_orig + 1]
        true = att_full[t_orig + 1]

        ctx = _Context(hist)
        if not ctx.ok: continue

        # ── оценка локального периода (causal: только hist) ──
        window = hist[-min(ACF_WINDOW, len(hist)):]
        T_acf  = period_via_acf(window)
        T_psd  = period_via_psd(window)
        step_acf = period_to_step(T_acf)
        step_psd = period_to_step(T_psd)

        # ── baseline: fixed ×2, p=16 ──
        lvls_std = levels_by_step(P_FIT_BASE, 2.0)
        pred_f, loo_f = _forecast(ctx, P_FIT_BASE, lvls_std, need_loo=True)
        if not np.isnan(pred_f):
            out["fixed_2"]["errors"].append(pred_f - true)
            out["fixed_2"]["signs"].append(int(np.sign(pred_f) == np.sign(true)))

        # ── ensemble ×2: все p_reg ──
        preds_2: dict[int, float] = {}; loos_2: dict[int, float] = {}
        for p in P_REG_GRID:
            pr, lo = _forecast(ctx, p, levels_by_step(p, 2.0), need_loo=True)
            preds_2[p] = pr; loos_2[p] = lo
        valid_2 = [p for p in P_REG_GRID
                   if not np.isnan(preds_2[p]) and not np.isnan(loos_2[p])]
        if valid_2:
            inv = np.array([1.0 / (loos_2[p] + 1e-12) for p in valid_2])
            w   = inv / inv.sum()
            pe  = float(w @ np.array([preds_2[p] for p in valid_2]))
            out["ensemble_2"]["errors"].append(pe - true)
            out["ensemble_2"]["signs"].append(int(np.sign(pe) == np.sign(true)))

        # ── spec_acf_p16 ──
        pred_acf, _ = _forecast(ctx, P_FIT_BASE, levels_by_step(P_FIT_BASE, step_acf),
                                 need_loo=False)
        if not np.isnan(pred_acf):
            out["spec_acf_p16"]["errors"].append(pred_acf - true)
            out["spec_acf_p16"]["signs"].append(int(np.sign(pred_acf) == np.sign(true)))

        # ── spec_psd_p16 ──
        pred_psd, _ = _forecast(ctx, P_FIT_BASE, levels_by_step(P_FIT_BASE, step_psd),
                                 need_loo=False)
        if not np.isnan(pred_psd):
            out["spec_psd_p16"]["errors"].append(pred_psd - true)
            out["spec_psd_p16"]["signs"].append(int(np.sign(pred_psd) == np.sign(true)))

        # ── spec_acf_ens: ensemble поверх ACF-адаптивного шага ──
        preds_acf: dict[int, float] = {}; loos_acf: dict[int, float] = {}
        for p in P_REG_GRID:
            pr, lo = _forecast(ctx, p, levels_by_step(p, step_acf), need_loo=True)
            preds_acf[p] = pr; loos_acf[p] = lo
        valid_acf = [p for p in P_REG_GRID
                     if not np.isnan(preds_acf[p]) and not np.isnan(loos_acf[p])]
        if valid_acf:
            inv_a = np.array([1.0 / (loos_acf[p] + 1e-12) for p in valid_acf])
            w_a   = inv_a / inv_a.sum()
            pe_a  = float(w_a @ np.array([preds_acf[p] for p in valid_acf]))
            out["spec_acf_ens"]["errors"].append(pe_a - true)
            out["spec_acf_ens"]["signs"].append(int(np.sign(pe_a) == np.sign(true)))

        # ── oracle_step_p16: лучший шаг из сетки по p=16 ──
        best_pred = None; best_step = None; best_err = np.inf
        for step_val in STEPS_GRID:
            pr, _ = _forecast(ctx, P_FIT_BASE, levels_by_step(P_FIT_BASE, step_val),
                               need_loo=False)
            if not np.isnan(pr):
                err = abs(pr - true)
                if err < best_err:
                    best_err = err; best_pred = pr; best_step = step_val
        if best_pred is not None:
            out["oracle_step_p16"]["errors"].append(best_pred - true)
            out["oracle_step_p16"]["signs"].append(
                int(np.sign(best_pred) == np.sign(true)))

        rows.append({
            "T_acf":      T_acf if T_acf is not None else np.nan,
            "T_psd":      T_psd if T_psd is not None else np.nan,
            "step_acf":   step_acf,
            "step_psd":   step_psd,
            "oracle_step": best_step if best_step is not None else np.nan,
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


baseline = rmae("fixed_2")
print(f"\nBaseline (fixed_2): rMAE = {baseline:.4f}")

# ── Результаты ────────────────────────────────────────────────────────────────
LABELS = {
    "fixed_2":         "×2.0 фиксированный p=16 (baseline)",
    "ensemble_2":      "×2.0 ensemble p∈[4..16]",
    "spec_acf_p16":    "ACF-адапт. шаг, p=16",
    "spec_psd_p16":    "PSD-адапт. шаг, p=16",
    "spec_acf_ens":    "ACF-адапт. шаг, ensemble",
    "oracle_step_p16": "Oracle шаг (знает истину), p=16",
}
print("\n── Итоговая таблица ──")
print(f"{'Метод':45s}  {'rMAE':7s}  {'Δ%':8s}  {'SignAcc':8s}")
for k in METHOD_KEYS:
    r  = rmae(k); d = (r / baseline - 1) * 100 if not np.isnan(r) else np.nan
    sa = sacc(k)
    print(f"  {LABELS[k]:45s}  {r:.4f}  {d:+7.2f}%  {sa:.1f}%")

# ── Диагностика: распределение шагов ──────────────────────────────────────────
step_acf_chosen   = [r["step_acf"]    for r in all_rows]
step_oracle_chosen = [r["oracle_step"] for r in all_rows
                      if not np.isnan(r["oracle_step"])]
T_acf_vals = [r["T_acf"] for r in all_rows if not np.isnan(r["T_acf"])]

print("\n── Диагностика шагов ──")
print(f"ACF период T: медиана={np.nanmedian(T_acf_vals):.1f}б  "
      f"диапазон=[{np.nanmin(T_acf_vals):.0f}, {np.nanmax(T_acf_vals):.0f}]б  "
      f"None={sum(1 for r in all_rows if np.isnan(r['T_acf']))}/{len(all_rows)}")

cnt_acf = Counter(round(s, 2) for s in step_acf_chosen)
cnt_ora = Counter(round(s, 2) for s in step_oracle_chosen)
print(f"\nРаспределение шагов (ACF-адапт.):")
for sn, sv in zip(STEP_NAMES, STEPS_GRID):
    print(f"  ×{sn}: ACF={cnt_acf.get(round(sv, 2), 0):3d}  "
          f"oracle={cnt_ora.get(round(sv, 2), 0):3d}")

# Accuracy: как часто ACF угадывает oracle step
if step_oracle_chosen:
    match_rate = sum(1 for a, o in zip(step_acf_chosen[:len(step_oracle_chosen)],
                                        step_oracle_chosen)
                     if abs(a - o) < 0.01) / len(step_oracle_chosen)
    print(f"\nACF vs oracle совпадение: {match_rate*100:.1f}%  "
          f"(случайный ≈ {100/len(STEPS_GRID):.0f}%)")


# ── Фигуры ────────────────────────────────────────────────────────────────────

SHOW = ["fixed_2", "ensemble_2", "spec_acf_p16", "spec_psd_p16",
        "spec_acf_ens", "oracle_step_p16"]
SLBL = ["×2.0\np=16", "×2.0\nens.", "ACF\np=16", "PSD\np=16",
        "ACF\nens.", "oracle\nstep"]
r_v  = [rmae(k) for k in SHOW]
d_v  = [(r / baseline - 1) * 100 if not np.isnan(r) else 0.0 for r in r_v]
cols = ["#78909c"] + ["#ef5350" if d > 0 else "#66bb6a" for d in d_v[1:]]

# Рис. A — rMAE сравнение
fig, ax = plt.subplots(figsize=(11, 4))
bars = ax.bar(SLBL, d_v, color=cols, width=0.55)
ax.axhline(0, color="white", lw=0.8)
for bar, v in zip(bars, d_v):
    ax.text(bar.get_x() + bar.get_width() / 2,
            v + (0.2 if v >= 0 else -0.2),
            f"{v:+.1f}%", ha="center",
            va="bottom" if v >= 0 else "top", color="white", fontsize=10)
ax.set_ylabel("Δ rMAE vs fixed_2 (%)")
ax.set_title("Рис.A  Спектр-адаптивный шаг каскада  (<0 = лучше)")
plt.tight_layout()
fig.savefig(FIG_DIR / "91_A.png", dpi=120); plt.close()
print("\nРис. A сохранён")

# Рис. B — распределение ACF-шагов vs oracle
sv_labels = [f"×{n}" for n in STEP_NAMES]
cnt_acf_v = [cnt_acf.get(round(sv, 2), 0) for sv in STEPS_GRID]
cnt_ora_v = [cnt_ora.get(round(sv, 2), 0) for sv in STEPS_GRID]
x = np.arange(len(STEPS_GRID)); w = 0.35
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w / 2, cnt_acf_v, width=w, color="#42a5f5", label="ACF-выбор")
ax.bar(x + w / 2, cnt_ora_v, width=w, color="#ef5350", label="Oracle-выбор", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(sv_labels)
ax.set_ylabel("Количество origins")
ax.set_title("Рис.B  Распределение шагов: ACF vs Oracle")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "91_B.png", dpi=120); plt.close()
print("Рис. B сохранён")

# Рис. C — гистограмма оценённых периодов T_acf
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(T_acf_vals, bins=range(MIN_PERIOD, MAX_PERIOD + 2),
        color="#42a5f5", edgecolor="none", alpha=0.8)
ax.axvline(np.nanmedian(T_acf_vals), color="#ef5350", lw=2, ls="--",
           label=f"медиана={np.nanmedian(T_acf_vals):.0f}б")
ax.set_xlabel("T_acf (баров)"); ax.set_ylabel("Количество origins")
ax.set_title("Рис.C  Распределение локального периода T (ACF)")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "91_C.png", dpi=120); plt.close()
print("Рис. C сохранён")

print("\nГотово. Фигуры: 91_A...C.png")
