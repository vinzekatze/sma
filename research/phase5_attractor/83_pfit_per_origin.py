"""
83_pfit_per_origin.py — Адаптивный p_fit на уровне регрессии.

Гипотеза: оптимальное измерение LWR варьируется от origin к origin
(разные режимы аттрактора, волатильность, тренд). Фиксированный p_fit=16
может быть неоптимальным на отдельных участках.

Эксперимент: каскад + acc_ang выполняется с фиксированным P_FIT=16 (как
обычно). Меняется только измерение LWR-регрессии (p_reg), которое определяет,
сколько последних координат вектора использует LWR-аппроксимация.

Методы:
  pos_only    — baseline (только позиция)
  acc_ang_16  — acc_ang λ=0.01, p_reg=16 (стандарт, скр.80-81)
  oracle      — best p_reg per origin (cheating — верхняя граница)
  loo_select  — p_reg выбирается по LOO внутри пула соседей

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров 1d (≡ скр.80-83).
Метрики: rMAE = MAE/std(att), SignAcc = % правильных знаков.

Дизайн-решение: каскад остаётся с P_FIT=16, чтобы изолировать эффект
размерности регрессии от эффекта поиска соседей.
"""
from __future__ import annotations
import time
import json
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
STEP      = 5

P_FIT  = 16        # измерение каскада и acc_ang — не меняется
P_MAX  = 64
XI_LWR = 3 * (P_FIT + 1)   # 51
LAMBDA = 0.01

LP_M = 9; LP_D = 3; LP_K = 30; LP_N = 3

# Сетка регрессионных измерений p_reg
P_REG_GRID = [4, 6, 8, 10, 12, 14, 16]


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
    """LWR: Gaussian weights by distance to vec_f."""
    dists = np.linalg.norm(X_nn - vec_f, axis=1)
    h_bw  = max(float(dists.max()), 1e-10)
    w     = np.exp(-0.5 * (dists / h_bw) ** 2)
    A     = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw    = np.sqrt(np.maximum(w, 1e-30))
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vec_f @ c[1:])


def _lwr_loo_error(X_nn: np.ndarray, y_nn: np.ndarray, vec_f: np.ndarray) -> float:
    """
    LOO-CV ошибка для WLS через hat matrix (O(n·p²), без перебора).

    Формула: e_loo_i = e_i / (1 - h_ii)
    где h_ii = w_i · x_i^T (X^T W X)^{-1} x_i — leverage WLS.
    """
    dists = np.linalg.norm(X_nn - vec_f, axis=1)
    h_bw  = max(float(dists.max()), 1e-10)
    w     = np.maximum(np.exp(-0.5 * (dists / h_bw) ** 2), 1e-30)

    A  = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw = np.sqrt(w)
    Aw = sw[:, None] * A                        # (n, p+1) weighted design
    yw = sw * y_nn

    c, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    y_hat = A @ c
    resid = y_nn - y_hat

    # leverage h_ii = w_i * x_i^T (A^T W A)^{-1} x_i
    try:
        AtWA_inv = np.linalg.inv(Aw.T @ Aw + 1e-10 * np.eye(A.shape[1]))
    except np.linalg.LinAlgError:
        return np.mean(np.abs(resid))
    # Вычисляем строку за строкой: h_ii = w_i * (A[i] @ AtWA_inv @ A[i])
    AiM = A @ AtWA_inv                           # (n, p+1)
    h_diag = w * np.einsum("ij,ij->i", AiM, A)  # w_i * x_i^T M x_i

    # Защита: если h_ii ≥ 1 (вырожденный случай) — используем обычный residual
    denom  = 1.0 - h_diag
    safe   = np.abs(denom) > 1e-6
    e_loo  = np.where(safe, resid / denom, resid)
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


LEVELS = _uniform_octave_levels(P_FIT, P_MAX, 2.0)
P_TOP  = LEVELS[0]


# ── ядро прогноза ─────────────────────────────────────────────────────────────

def _get_pool_and_context(att: np.ndarray):
    """
    Возвращает (X_full, y_base, X_nn, y_nn, query_pfit, ok).
    ok=False если данных мало.
    Каскад + acc_ang выполняется с фиксированным P_FIT=16.
    """
    n = len(att)
    if n - P_TOP - 1 < XI_LWR + 1:
        return None, None, None, None, None, False

    t_arr  = np.arange(P_TOP, n - 1)
    X_full = np.column_stack([att[t_arr - (P_TOP - 1 - j)] for j in range(P_TOP)])
    y_base = att[t_arr + 1]

    acc_hist = np.zeros(n)
    if n >= 3:
        acc_hist[2:] = att[2:] - 2 * att[1:-1] + att[:-2]
    X_acc = np.column_stack([acc_hist[t_arr - (P_FIT - 1 - j)] for j in range(P_FIT)])

    vec_full   = att[-P_TOP:].copy()
    vec_acc    = acc_hist[-P_FIT:].copy()
    query_pfit = vec_full[-P_FIT:]

    # каскад (открытый, без финального фильтра)
    cands = np.arange(len(X_full))
    for k, p_lvl in enumerate(LEVELS):
        is_last = (k == len(LEVELS) - 1)
        if not is_last:
            xi_lvl = min(XI_LWR, len(cands))
            if len(cands) > xi_lvl:
                d = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full[-p_lvl:], axis=1)
                cands = cands[np.argpartition(d, xi_lvl - 1)[:xi_lvl]]
            p_next = LEVELS[k + 1]; radius = p_lvl - p_next
            exp = cands[:, None] - np.arange(radius + 1)[None, :]
            cands = np.unique(np.clip(exp, 0, len(X_full) - 1))

    # acc_ang отбор
    if len(cands) > XI_LWR:
        d_pos  = np.linalg.norm(X_full[cands, -P_FIT:] - query_pfit, axis=1)
        d_acc  = _cosine_dist(X_acc[cands], vec_acc)
        d_comb = d_pos + LAMBDA * d_acc
        sel    = np.argpartition(d_comb, XI_LWR - 1)[:XI_LWR]
        cands  = cands[sel]

    if len(cands) < P_FIT + 2:
        return None, None, None, None, None, False

    X_nn = X_full[cands]   # (≤xi_lwr, P_TOP)
    y_nn = y_base[cands]
    return X_full, y_base, X_nn, y_nn, query_pfit, True


def forecast_fixed(att: np.ndarray) -> float:
    """acc_ang с фиксированным p_reg=P_FIT=16."""
    _, _, X_nn, y_nn, query_pfit, ok = _get_pool_and_context(att)
    if not ok:
        return np.nan
    return _lwr_predict(X_nn[:, -P_FIT:], y_nn, query_pfit)


def forecast_oracle(att: np.ndarray, true_next: float) -> tuple[float, int]:
    """Oracle: возвращает (best_pred, best_p_reg). Знает истинное следующее значение."""
    _, _, X_nn, y_nn, query_pfit, ok = _get_pool_and_context(att)
    if not ok:
        return np.nan, -1
    best_err = np.inf; best_pred = np.nan; best_p = P_REG_GRID[-1]
    for p_reg in P_REG_GRID:
        if X_nn.shape[1] < p_reg or len(X_nn) < p_reg + 2:
            continue
        pred = _lwr_predict(X_nn[:, -p_reg:], y_nn, query_pfit[-p_reg:])
        err  = abs(pred - true_next)
        if err < best_err:
            best_err = err; best_pred = pred; best_p = p_reg
    return best_pred, best_p


def forecast_loo(att: np.ndarray) -> tuple[float, int]:
    """LOO: выбирает p_reg с минимальной LOO-ошибкой. Возвращает (pred, selected_p_reg)."""
    _, _, X_nn, y_nn, query_pfit, ok = _get_pool_and_context(att)
    if not ok:
        return np.nan, -1
    best_loo = np.inf; best_pred = np.nan; best_p = P_REG_GRID[-1]
    for p_reg in P_REG_GRID:
        if X_nn.shape[1] < p_reg or len(X_nn) < p_reg + 2:
            continue
        X_r  = X_nn[:, -p_reg:]
        v_r  = query_pfit[-p_reg:]
        loo  = _lwr_loo_error(X_r, y_nn, v_r)
        if loo < best_loo:
            best_loo = loo
            best_pred = _lwr_predict(X_r, y_nn, v_r)
            best_p = p_reg
    return best_pred, best_p


def forecast_pos_only(att: np.ndarray) -> float:
    """Baseline: каскад без acc_ang, p_reg=P_FIT."""
    n = len(att)
    if n - P_TOP - 1 < XI_LWR + 1:
        return np.nan
    t_arr  = np.arange(P_TOP, n - 1)
    X_full = np.column_stack([att[t_arr - (P_TOP - 1 - j)] for j in range(P_TOP)])
    y_base = att[t_arr + 1]
    vec_full   = att[-P_TOP:].copy()
    query_pfit = vec_full[-P_FIT:]
    cands = np.arange(len(X_full))
    for k, p_lvl in enumerate(LEVELS):
        xi_lvl = min(XI_LWR, len(cands))
        if len(cands) > xi_lvl:
            d = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full[-p_lvl:], axis=1)
            cands = cands[np.argpartition(d, xi_lvl - 1)[:xi_lvl]]
        if k < len(LEVELS) - 1:
            p_next = LEVELS[k + 1]; radius = p_lvl - p_next
            exp = cands[:, None] - np.arange(radius + 1)[None, :]
            cands = np.unique(np.clip(exp, 0, len(X_full) - 1))
    if len(cands) < P_FIT + 2:
        return np.nan
    return _lwr_predict(X_full[cands, -P_FIT:], y_base[cands], query_pfit)


# ── загрузка + walk-forward ───────────────────────────────────────────────────

def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend_causal(close)
    ratio = close / np.maximum(lt, 1e-10)
    return _lp_proj(ratio, LP_M, LP_D, LP_K, LP_N)


def wf_ticker(att_full: np.ndarray) -> dict:
    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP, n_total - 1, STEP))
    out = {k: {"errors": [], "signs": []} for k in ("pos_only", "fixed", "oracle", "loo")}
    oracle_p_hist: list[int] = []
    loo_p_hist:    list[int] = []

    for t_orig in origins:
        if t_orig + 1 >= n_total:
            continue
        hist = att_full[:t_orig + 1]
        true = att_full[t_orig + 1]

        for key, pred in [("pos_only", forecast_pos_only(hist)),
                           ("fixed",    forecast_fixed(hist))]:
            if not np.isnan(pred):
                out[key]["errors"].append(pred - true)
                out[key]["signs"].append(int(np.sign(pred) == np.sign(true)))

        pred_ora, p_ora = forecast_oracle(hist, true)
        if not np.isnan(pred_ora):
            out["oracle"]["errors"].append(pred_ora - true)
            out["oracle"]["signs"].append(int(np.sign(pred_ora) == np.sign(true)))
            oracle_p_hist.append(p_ora)

        pred_loo, p_loo = forecast_loo(hist)
        if not np.isnan(pred_loo):
            out["loo"]["errors"].append(pred_loo - true)
            out["loo"]["signs"].append(int(np.sign(pred_loo) == np.sign(true)))
            loo_p_hist.append(p_loo)

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

print("\nWalk-forward...")
t0 = time.time()

all_res: dict[str, dict] = {k: {"errors": [], "signs": []} for k in ("pos_only", "fixed", "oracle", "loo")}
oracle_p_all: list[int] = []
loo_p_all:    list[int] = []

for tkr in TICKERS:
    att = att_data[tkr]
    res, op, lp = wf_ticker(att)
    for key in all_res:
        for m in ("errors", "signs"):
            all_res[key][m].extend(res[key][m])
    oracle_p_all.extend(op)
    loo_p_all.extend(lp)
    print(f"  {tkr}: oracle_p распределение = {sorted(set(op))}")

print(f"Walk-forward завершён за {time.time()-t0:.1f}с")

def rmae(key: str) -> float:
    e = all_res[key]["errors"]
    return float(np.mean(np.abs(e))) / std_mean if e else np.nan

def sacc(key: str) -> float:
    s = all_res[key]["signs"]
    return float(np.mean(s)) * 100.0 if s else np.nan

baseline = rmae("pos_only")

print("\nИтоговая таблица:")
for key in ("pos_only", "fixed", "oracle", "loo"):
    r = rmae(key); d = (r / baseline - 1) * 100
    print(f"  {key:12s}: rMAE={r:.4f} ({d:+.2f}%)  SignAcc={sacc(key):.1f}%")

# Распределение выбранных p_reg
from collections import Counter
print("\nOracle: распределение выбранного p_reg:")
for p, cnt in sorted(Counter(oracle_p_all).items()):
    print(f"  p={p:2d}: {cnt:4d} раз ({cnt/len(oracle_p_all)*100:.1f}%)")

print("\nLOO: распределение выбранного p_reg:")
for p, cnt in sorted(Counter(loo_p_all).items()):
    print(f"  p={p:2d}: {cnt:4d} раз ({cnt/len(loo_p_all)*100:.1f}%)")


# ── фигуры ────────────────────────────────────────────────────────────────────

KEYS   = ["pos_only", "fixed", "oracle", "loo"]
LABELS = ["pos_only", "acc_ang\n(p=16)", "oracle\n(best p)", "LOO-select"]
rmae_v  = [rmae(k) for k in KEYS]
delta_v = [(r / baseline - 1) * 100 for r in rmae_v]
colors  = ["#78909c"] + ["#66bb6a" if d < 0 else "#ef5350" for d in delta_v[1:]]

# Рис. A — сравнение методов
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(LABELS, delta_v, color=colors)
ax.axhline(0, color="white", lw=0.8)
for bar, v in zip(bars, delta_v):
    ax.text(bar.get_x() + bar.get_width() / 2, v + (0.2 if v >= 0 else -0.2),
            f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top",
            color="white", fontsize=10)
ax.set_ylabel("Δ rMAE vs pos_only (%)")
ax.set_title(f"Рис.A  Per-origin p_fit (регрессия)  "
             f"(8 тикеров, {N_ORIGINS}×STEP={STEP})")
plt.tight_layout()
fig.savefig(FIG_DIR / "83_A.png", dpi=120)
plt.close()
print("Рис. A сохранён")

# Рис. B — гистограмма oracle-выбранного p_reg
cnt_ora = Counter(oracle_p_all)
cnt_loo = Counter(loo_p_all)
x = P_REG_GRID
y_ora = [cnt_ora.get(p, 0) / max(len(oracle_p_all), 1) * 100 for p in x]
y_loo = [cnt_loo.get(p, 0) / max(len(loo_p_all),   1) * 100 for p in x]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar([str(p) for p in x], y_ora, color="#ffa726")
axes[0].set_xlabel("p_reg")
axes[0].set_ylabel("% origins")
axes[0].set_title(f"Oracle — выбранный p_reg\n"
                  f"(rMAE={rmae('oracle'):.4f}, {delta_v[2]:+.1f}%)")
axes[1].bar([str(p) for p in x], y_loo, color="#42a5f5")
axes[1].set_xlabel("p_reg")
axes[1].set_ylabel("% origins")
axes[1].set_title(f"LOO — выбранный p_reg\n"
                  f"(rMAE={rmae('loo'):.4f}, {delta_v[3]:+.1f}%)")
for ax in axes:
    ax.axvline(x.index(P_FIT) - 0.5 + 0.5, color="white", ls="--", lw=1,
               label=f"default p={P_FIT}")
plt.tight_layout()
fig.savefig(FIG_DIR / "83_B.png", dpi=120)
plt.close()
print("Рис. B сохранён")

# Рис. C — per-ticker: oracle потенциал vs LOO реализация
from collections import defaultdict
ticker_deltas_ora: dict[str, float] = {}
ticker_deltas_loo: dict[str, float] = {}
for tkr in TICKERS:
    att = att_data[tkr]; std = float(np.std(att))
    res, _, _ = wf_ticker(att)
    def _rd(key):
        e = res[key]["errors"]
        return float(np.mean(np.abs(e))) / std if e else np.nan
    base_t = _rd("fixed")
    d_ora  = (_rd("oracle") / base_t - 1) * 100 if base_t else np.nan
    d_loo  = (_rd("loo")    / base_t - 1) * 100 if base_t else np.nan
    ticker_deltas_ora[tkr] = d_ora
    ticker_deltas_loo[tkr] = d_loo

x_t = np.arange(len(TICKERS)); w = 0.35
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(x_t - w/2, [ticker_deltas_ora[t] for t in TICKERS], width=w,
       color="#ffa726", label="oracle")
ax.bar(x_t + w/2, [ticker_deltas_loo[t] for t in TICKERS], width=w,
       color="#42a5f5", label="LOO-select")
ax.axhline(0, color="white", lw=0.8)
ax.set_xticks(x_t); ax.set_xticklabels(TICKERS)
ax.set_ylabel("Δ rMAE vs acc_ang p=16 (%)")
ax.set_title("Рис.C  Per-ticker: oracle и LOO vs fixed acc_ang  (<0 = лучше)")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "83_C.png", dpi=120)
plt.close()
print("Рис. C сохранён")

print(f"\nГотово. Фигуры: 83_A...C.png")
