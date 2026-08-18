"""
82_adaptive_lambda_temporal_decay.py — Два направления улучшения acc_ang.

Гипотезы:
  1. Адаптивный λ (per-origin): вычислять λ = σ_pos/σ_acc по пулу каскада →
     нормировка обеих метрик к одному масштабу без ручной настройки.
  2. Temporal decay в LWR: w_i *= exp(-α·age_i) → снижать вклад устаревших
     аналогов, учитывая нестационарность рынка.
  3. Комбинация: adaptive_λ + temporal decay.

Методы:
  pos_only           — baseline (скр.80-81)
  acc_ang            — λ=0.01 fixed, no decay (лучший скр.80: −10.8%)
  adaptive_λ         — λ = σ_pos/σ_acc per-origin, no decay
  acc_ang_td         — λ=0.01 fixed + temporal decay α=best
  adaptive_λ_td      — adaptive λ + temporal decay α=best

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров 1d (≡ скр.80-81).
Метрики: rMAE = MAE/std(att), SignAcc = % правильных знаков.
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

# ── параметры ─────────────────────────────────────────────────────────────────
TICKERS   = ["SBER", "MRKP", "LKOH", "NVTK", "CHMF", "NLMK", "MGNT", "VTBR"]
INTERVAL  = "1d"
N_ORIGINS = 40
STEP      = 5
H         = 1

P_FIT  = 16
P_MAX  = 64
XI_LWR = 3 * (P_FIT + 1)   # 51
LAMBDA = 0.01               # фиксированный acc_ang λ (оптимум скр.80)

LP_M = 9; LP_D = 3; LP_K = 30; LP_N = 3

# Сетка temporal decay α
# T_half = ln(2)/α: 0.001→693б, 0.002→347б, 0.005→139б, 0.01→69б, 0.02→35б, 0.05→14б
ALPHA_GRID = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


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
    """1 - cos_sim(строки A, вектор b) ∈ [0,2]."""
    norm_A = np.linalg.norm(A, axis=1)
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-12:
        return np.ones(len(A))
    cos = np.where(norm_A > 1e-12, (A @ b) / (norm_A * norm_b), 0.0)
    return 1.0 - np.clip(cos, -1.0, 1.0)


def _lwr(X_nn: np.ndarray, y_nn: np.ndarray, vec_f: np.ndarray,
         ages: np.ndarray | None = None, alpha: float = 0.0) -> float:
    """LWR с опциональным temporal decay.

    ages — возраст каждого соседа (в барах от origin); None → нет decay.
    alpha > 0 → w_i *= exp(-alpha * age_i).
    """
    dists = np.linalg.norm(X_nn - vec_f, axis=1)
    h_bw  = max(float(dists.max()), 1e-10)
    w     = np.exp(-0.5 * (dists / h_bw) ** 2)
    if alpha > 0.0 and ages is not None:
        w *= np.exp(-alpha * ages.astype(np.float64))
    A  = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw = np.sqrt(np.maximum(w, 1e-30))
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vec_f @ c[1:])


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


# ── одношаговый прогноз ───────────────────────────────────────────────────────

def _forecast_one(att: np.ndarray,
                  lambda_mode: str = "fixed",
                  alpha: float = 0.0) -> float:
    """
    Один шаг h=1 прогноза.

    lambda_mode:
      "fixed"    — λ=LAMBDA (=0.01, оптимум скр.80)
      "adaptive" — λ = σ_pos/σ_acc вычисляется по пулу каскада
      "pos_only" — без acc_ang (baseline)

    alpha: temporal decay (0.0 = отключён).
    """
    n = len(att)
    if n - P_TOP - 1 < XI_LWR + 1:
        return np.nan

    t_arr  = np.arange(P_TOP, n - 1)          # shape: (m,)
    X_full = np.column_stack([att[t_arr - (P_TOP - 1 - j)] for j in range(P_TOP)])
    y_base = att[t_arr + 1]

    acc_hist = np.zeros(n)
    if n >= 3:
        acc_hist[2:] = att[2:] - 2 * att[1:-1] + att[:-2]
    X_acc = np.column_stack([acc_hist[t_arr - (P_FIT - 1 - j)] for j in range(P_FIT)])

    vec_full   = att[-P_TOP:].copy()
    vec_acc    = acc_hist[-P_FIT:].copy()
    query_pfit = vec_full[-P_FIT:]

    # Возраст каждого вектора в X_full (в барах от текущего origin)
    # t_arr[i] = P_TOP + i; последний t = n-2; age = (n-2) - t_arr[i]
    ages_all = (n - 2) - t_arr   # shape: (m,), неотрицательные

    # ── каскад ────────────────────────────────────────────────────────────────
    def _cascade() -> np.ndarray:
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
        return cands

    def _cascade_open() -> np.ndarray:
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
        return cands

    def _select_acc_ang(pool: np.ndarray, lam: float) -> np.ndarray:
        if len(pool) <= XI_LWR:
            return pool
        d_pos  = np.linalg.norm(X_full[pool, -P_FIT:] - query_pfit, axis=1)
        d_acc  = _cosine_dist(X_acc[pool], vec_acc)
        d_comb = d_pos + lam * d_acc
        sel    = np.argpartition(d_comb, XI_LWR - 1)[:XI_LWR]
        return pool[sel]

    # ── отбор кандидатов ──────────────────────────────────────────────────────
    if lambda_mode == "pos_only":
        cands = _cascade()
    else:
        pool = _cascade_open()

        if lambda_mode == "fixed":
            lam = LAMBDA
        else:  # "adaptive"
            d_pos_pool = np.linalg.norm(X_full[pool, -P_FIT:] - query_pfit, axis=1)
            d_acc_pool = _cosine_dist(X_acc[pool], vec_acc)
            sigma_pos  = float(np.std(d_pos_pool)) + 1e-12
            sigma_acc  = float(np.std(d_acc_pool)) + 1e-12
            lam        = sigma_pos / sigma_acc

        cands = _select_acc_ang(pool, lam)

    # ── LWR ───────────────────────────────────────────────────────────────────
    if len(cands) < P_FIT + 2:
        return np.nan

    X_nn  = X_full[cands, -P_FIT:]
    y_nn  = y_base[cands]
    vec_f = query_pfit
    ages  = ages_all[cands] if alpha > 0.0 else None
    return _lwr(X_nn, y_nn, vec_f, ages=ages, alpha=alpha)


# ── walk-forward ──────────────────────────────────────────────────────────────

def wf_ticker(att_full: np.ndarray, methods_cfg: list[tuple]) -> dict:
    """
    methods_cfg: список (key, lambda_mode, alpha).
    Протокол ≡ скр.80-81.
    """
    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP, n_total - 1, STEP))
    results: dict[str, dict] = {k: {"errors": [], "signs": []} for k, *_ in methods_cfg}

    for t_orig in origins:
        if t_orig + 1 >= n_total:
            continue
        hist = att_full[:t_orig + 1]
        true = att_full[t_orig + 1]
        for key, lambda_mode, alpha in methods_cfg:
            pred = _forecast_one(hist, lambda_mode, alpha)
            if np.isnan(pred):
                continue
            results[key]["errors"].append(pred - true)
            results[key]["signs"].append(int(np.sign(pred) == np.sign(true)))

    return results


# ── загрузка данных ───────────────────────────────────────────────────────────

def load_att(ticker: str) -> np.ndarray:
    path = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw  = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend_causal(close)
    ratio = close / np.maximum(lt, 1e-10)
    return _lp_proj(ratio, LP_M, LP_D, LP_K, LP_N)


def compute_metrics(errors: list[float], signs: list[int],
                    std_att: float) -> tuple[float, float]:
    if not errors:
        return np.nan, np.nan
    mae  = float(np.mean(np.abs(errors)))
    rmae = mae / std_att if std_att > 0 else np.nan
    sacc = float(np.mean(signs)) * 100.0
    return rmae, sacc


# ── основной блок ─────────────────────────────────────────────────────────────

print("Загрузка + Local Projective...")
t0 = time.time()
att_data: dict[str, np.ndarray] = {}
for tkr in TICKERS:
    t1 = time.time()
    att_data[tkr] = load_att(tkr)
    print(f"  {tkr}... {time.time()-t1:.1f}с")
std_all = [float(np.std(att_data[tkr])) for tkr in TICKERS]
std_mean = float(np.mean(std_all))
print(f"Загрузка итого: {time.time()-t0:.1f}с")


# ── фаза 1: базовые методы (pos_only, acc_ang, adaptive_λ) ────────────────────
print("\nФаза 1: pos_only / acc_ang / adaptive_λ ...")
t0 = time.time()

PHASE1_CFG: list[tuple] = [
    ("pos_only",     "pos_only",  0.0),
    ("acc_ang",      "fixed",     0.0),
    ("adaptive_lam", "adaptive",  0.0),
]

ph1_res: dict[str, dict] = {k: {"errors": [], "signs": []} for k, *_ in PHASE1_CFG}
ph1_per_ticker: dict[str, dict[str, float]] = {}

for tkr in TICKERS:
    att = att_data[tkr]; std = float(np.std(att))
    res = wf_ticker(att, PHASE1_CFG)
    trow: dict[str, float] = {}
    for key, *_ in PHASE1_CFG:
        for m in ("errors", "signs"):
            ph1_res[key][m].extend(res[key][m])
        rmae, _ = compute_metrics(res[key]["errors"], res[key]["signs"], std)
        trow[key] = rmae
    ph1_per_ticker[tkr] = trow
    print(f"  {tkr}: {N_ORIGINS} origins")

print(f"Фаза 1 завершена за {time.time()-t0:.1f}с")

def rmae_ph1(key: str) -> float:
    return float(np.mean(np.abs(ph1_res[key]["errors"]))) / std_mean

def sacc_ph1(key: str) -> float:
    s = ph1_res[key]["signs"]
    return float(np.mean(s)) * 100.0 if s else np.nan

baseline = rmae_ph1("pos_only")
acc_ang_rmae = rmae_ph1("acc_ang")

print("\nФаза 1 — итог:")
for key, *_ in PHASE1_CFG:
    r = rmae_ph1(key); d = (r / baseline - 1) * 100
    print(f"  {key:20s}: rMAE={r:.4f} ({d:+.2f}%)  SignAcc={sacc_ph1(key):.1f}%")


# ── фаза 2: сетка temporal decay α ────────────────────────────────────────────
print("\nФаза 2: temporal decay — сетка α ...")
t0 = time.time()

grid_fixed_td:    list[float] = []   # acc_ang (λ=0.01) + decay
grid_adaptive_td: list[float] = []   # adaptive_λ       + decay

for alpha in ALPHA_GRID:
    cfg = [
        ("fixed_td",    "fixed",    alpha),
        ("adaptive_td", "adaptive", alpha),
    ]
    e_fixed: list[float] = []; e_adap: list[float] = []
    for tkr in TICKERS:
        res = wf_ticker(att_data[tkr], cfg)
        e_fixed.extend(res["fixed_td"]["errors"])
        e_adap.extend(res["adaptive_td"]["errors"])
    r_fixed = float(np.mean(np.abs(e_fixed))) / std_mean if e_fixed else np.nan
    r_adap  = float(np.mean(np.abs(e_adap)))  / std_mean if e_adap  else np.nan
    grid_fixed_td.append(r_fixed)
    grid_adaptive_td.append(r_adap)
    t_half_str = f"T½≈{round(0.693/alpha):d}б" if alpha > 0 else "∞"
    print(f"  α={alpha:.3f} ({t_half_str:>10s}):  fixed={r_fixed:.4f} ({(r_fixed/baseline-1)*100:+.1f}%)  "
          f"adaptive={r_adap:.4f} ({(r_adap/baseline-1)*100:+.1f}%)")

print(f"Фаза 2 завершена за {time.time()-t0:.1f}с")

# Лучший α для каждой линии
best_alpha_fixed    = ALPHA_GRID[int(np.nanargmin(grid_fixed_td))]
best_alpha_adaptive = ALPHA_GRID[int(np.nanargmin(grid_adaptive_td))]
print(f"\nЛучший α для acc_ang(fixed):    {best_alpha_fixed}")
print(f"Лучший α для adaptive_λ:        {best_alpha_adaptive}")


# ── фаза 3: финальная таблица 2×2 ────────────────────────────────────────────
print("\nФаза 3: финальная таблица (2×2) ...")
t0 = time.time()

PHASE3_CFG: list[tuple] = [
    ("pos_only",              "pos_only",  0.0),
    ("acc_ang",               "fixed",     0.0),
    ("adaptive_λ",            "adaptive",  0.0),
    ("acc_ang_td",            "fixed",     best_alpha_fixed),
    ("adaptive_λ_td",         "adaptive",  best_alpha_adaptive),
]

ph3_res: dict[str, dict] = {k: {"errors": [], "signs": []} for k, *_ in PHASE3_CFG}
ph3_per_ticker: dict[str, dict[str, float]] = {}

for tkr in TICKERS:
    att = att_data[tkr]; std = float(np.std(att))
    res = wf_ticker(att, PHASE3_CFG)
    trow: dict[str, float] = {}
    for key, *_ in PHASE3_CFG:
        for m in ("errors", "signs"):
            ph3_res[key][m].extend(res[key][m])
        rmae, _ = compute_metrics(res[key]["errors"], res[key]["signs"], std)
        trow[key] = rmae
    ph3_per_ticker[tkr] = trow
    print(f"  {tkr}: {N_ORIGINS} origins")

print(f"Фаза 3 завершена за {time.time()-t0:.1f}с")

def rmae_ph3(key: str) -> float:
    return float(np.mean(np.abs(ph3_res[key]["errors"]))) / std_mean

def sacc_ph3(key: str) -> float:
    s = ph3_res[key]["signs"]
    return float(np.mean(s)) * 100.0 if s else np.nan

print("\nФинальная таблица:")
for key, lm, alpha in PHASE3_CFG:
    r = rmae_ph3(key); d = (r / baseline - 1) * 100
    tag = f"λ={'adaptive' if lm=='adaptive' else '0.01'} α={alpha}"
    print(f"  {key:25s}: rMAE={r:.4f} ({d:+.2f}%)  SignAcc={sacc_ph3(key):.1f}%  [{tag}]")


# ── фигуры ────────────────────────────────────────────────────────────────────

names   = [k for k, *_ in PHASE3_CFG]
rmae_v  = [rmae_ph3(k) for k in names]
delta_v = [(r / baseline - 1) * 100 for r in rmae_v]
colors  = ["#78909c"] + ["#66bb6a" if d < 0 else "#ef5350" for d in delta_v[1:]]

# Рис. A — итоговое сравнение методов
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(names, delta_v, color=colors)
ax.axvline(0, color="white", lw=0.8)
for bar, v in zip(bars, delta_v):
    ax.text(v + (0.15 if v >= 0 else -0.15), bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}%", va="center", ha="left" if v >= 0 else "right",
            color="white", fontsize=9)
ax.set_xlabel("Δ rMAE vs pos_only (%)")
ax.set_title(f"Рис.A  Adaptive λ + Temporal decay  (8 тикеров, {N_ORIGINS}×STEP={STEP})")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(FIG_DIR / "82_A.png", dpi=120)
plt.close()
print("Рис. A сохранён")

# Рис. B — сетка α для обоих методов
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ALPHA_GRID, grid_fixed_td,    "o-", color="#42a5f5",
        label="acc_ang (λ=0.01) + decay")
ax.plot(ALPHA_GRID, grid_adaptive_td, "s-", color="#ab47bc",
        label="adaptive_λ + decay")
ax.axhline(acc_ang_rmae, color="#66bb6a", ls="--", lw=1.2,
           label=f"acc_ang (no decay) {acc_ang_rmae:.4f}")
ax.axhline(baseline, color="#90a4ae", ls=":", lw=1.2,
           label=f"pos_only {baseline:.4f}")
ax.set_xlabel("α (temporal decay)")
ax.set_ylabel("rMAE")
ax.set_title("Рис.B  rMAE vs α (temporal decay rate)")
ax.legend(fontsize=9)
ax.set_xticks(ALPHA_GRID)
ax.set_xticklabels([str(a) for a in ALPHA_GRID])
plt.tight_layout()
fig.savefig(FIG_DIR / "82_B.png", dpi=120)
plt.close()
print("Рис. B сохранён")

# Рис. C — per-ticker: acc_ang vs adaptive_λ_td (лучший combo)
best_combo = min(names[1:], key=lambda k: rmae_ph3(k))
print(f"Лучший метод: {best_combo}")

ticker_rmae_ref  = []
ticker_rmae_best = []
for tkr in TICKERS:
    att = att_data[tkr]; std = float(np.std(att))
    r1 = wf_ticker(att, [("ref",  "fixed",  0.0),
                          ("best", *next((lm, al) for k, lm, al in PHASE3_CFG if k == best_combo))])
    ticker_rmae_ref.append(float(np.mean(np.abs(r1["ref"]["errors"]))) / std)
    ticker_rmae_best.append(float(np.mean(np.abs(r1["best"]["errors"]))) / std)

delta_per = [(b / r - 1) * 100 for r, b in zip(ticker_rmae_ref, ticker_rmae_best)]

fig, ax = plt.subplots(figsize=(9, 4))
c_list = ["#66bb6a" if d < 0 else "#ef5350" for d in delta_per]
bars   = ax.bar(TICKERS, delta_per, color=c_list)
ax.axhline(0, color="white", lw=0.8)
for bar, v in zip(bars, delta_per):
    ax.text(bar.get_x() + bar.get_width() / 2, v + (0.2 if v >= 0 else -0.2),
            f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top",
            color="white", fontsize=9)
ax.set_ylabel(f"Δ rMAE {best_combo} vs acc_ang (%)")
ax.set_title(f"Рис.C  Per-ticker: {best_combo} vs acc_ang  (<0 = лучше)")
plt.tight_layout()
fig.savefig(FIG_DIR / "82_C.png", dpi=120)
plt.close()
print("Рис. C сохранён")

print(f"\nГотово. Фигуры: 82_A...C.png")
