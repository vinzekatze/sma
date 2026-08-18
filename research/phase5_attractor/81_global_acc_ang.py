"""
81_global_acc_ang.py — Глобальный поиск соседей с angular acceleration.

Гипотеза: деградация global_blend (скр.80: +8.6%) вызвана тем, что
глобальные соседи отбираются чистым d_pos — неоднородный пул рушит LWR.
Решение: применить acc_ang и к глобальному поиску (d_pos + λ·d_ang(acc)).

Методы:
  pos_only       — baseline: каскад, только позиция
  acc_ang        — cascade_open + acc_ang (лучший скр.80, −10.8%)
  gpos(r)        — acc_ang каскад + (1-r)·xi_lwr из глобал по d_pos     [воспроизводим скр.80]
  gacc(r)        — acc_ang каскад + (1-r)·xi_lwr из глобал по acc_ang   [НОВОЕ]

r = blend_frac: доля xi_lwr из каскадного пула (0.0=весь глобал, 1.0=нет глобала).

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров 1d.
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
LAMBDA = 0.01               # acc_ang λ (оптимум скр.80)

LP_M = 9; LP_D = 3; LP_K = 30; LP_N = 3


# ── алгоритмические функции ───────────────────────────────────────────────────

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


def _lwr(X_nn: np.ndarray, y_nn: np.ndarray, vec_f: np.ndarray) -> float:
    h_bw = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
    w = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_f, axis=1) / h_bw) ** 2)
    A = np.hstack([np.ones((len(X_nn), 1)), X_nn]); sw = np.sqrt(w)
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
P_TOP  = LEVELS[0]   # = P_MAX в каскаде


# ── одношаговый прогноз ───────────────────────────────────────────────────────

def _forecast_one(att: np.ndarray,
                  method: str,
                  blend_frac: float = 1.0) -> float:
    """
    Один шаг h=1 прогноза. Возвращает предсказанное значение att[n].

    method:
      'pos_only'   — стандартный каскад, только позиция
      'acc_ang'    — cascade_open + acc_ang (без глобала)
      'gpos'       — acc_ang каскад + глобал по d_pos
      'gacc'       — acc_ang каскад + глобал по acc_ang

    blend_frac: доля ξ из каскадного пула (1.0 = весь каскад, нет глобала).
    При blend_frac=1.0 gpos/gacc ≡ acc_ang.
    """
    n = len(att)
    if n - P_TOP - 1 < XI_LWR + 1:
        return np.nan

    t_arr  = np.arange(P_TOP, n - 1)
    X_full = np.column_stack([att[t_arr - (P_TOP - 1 - j)] for j in range(P_TOP)])
    y_base = att[t_arr + 1]

    acc_hist = np.zeros(n)
    if n >= 3:
        acc_hist[2:] = att[2:] - 2 * att[1:-1] + att[:-2]
    X_acc = np.column_stack([acc_hist[t_arr - (P_FIT - 1 - j)] for j in range(P_FIT)])

    vec_full    = att[-P_TOP:].copy()
    vec_acc     = acc_hist[-P_FIT:].copy()
    query_pfit  = vec_full[-P_FIT:]

    # каскад: возвращает xi_lwr кандидатов по позиции
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

    # каскад без фильтрации на последнем уровне → расширенный пул для acc_ang
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

    # финальный отбор xi_lwr из пула по d_pos + λ·d_ang(acc)
    def _acc_ang_select(pool: np.ndarray) -> np.ndarray:
        if len(pool) <= XI_LWR:
            return pool
        d_pos  = np.linalg.norm(X_full[pool, -P_FIT:] - query_pfit, axis=1)
        d_acc  = _cosine_dist(X_acc[pool], vec_acc)
        d_comb = d_pos + LAMBDA * d_acc
        sel    = np.argpartition(d_comb, XI_LWR - 1)[:XI_LWR]
        return pool[sel]

    # ── отбор кандидатов по методу ────────────────────────────────────────────
    if method == "pos_only":
        cands = _cascade()

    elif method == "acc_ang":
        pool  = _cascade_open()
        cands = _acc_ang_select(pool)

    else:
        # gpos / gacc: acc_ang из каскада + глобальный пул
        pool      = _cascade_open()
        cas_sel   = _acc_ang_select(pool)   # все xi_lwr из каскада

        n_cas  = max(1, round(XI_LWR * blend_frac))
        n_cas  = min(n_cas, len(cas_sel))
        n_glob = XI_LWR - n_cas

        if n_glob <= 0:
            cands = cas_sel
        else:
            # отбор n_cas из каскадного пула по d_pos (они уже acc_ang-отобраны)
            d_cas   = np.linalg.norm(X_full[cas_sel, -P_FIT:] - query_pfit, axis=1)
            top_cas = cas_sel[np.argpartition(d_cas, min(n_cas - 1, len(d_cas) - 1))[:n_cas]]

            # глобальный поиск
            if method == "gpos":
                # pos-only (воспроизводим скр.80)
                d_glob  = np.linalg.norm(X_full[:, -P_FIT:] - query_pfit, axis=1)
            else:  # "gacc" — acc_ang по всей истории
                d_pos_g = np.linalg.norm(X_full[:, -P_FIT:] - query_pfit, axis=1)
                d_acc_g = _cosine_dist(X_acc, vec_acc)
                d_glob  = d_pos_g + LAMBDA * d_acc_g

            n_g_eff  = min(n_glob, len(X_full))
            top_glob = np.argpartition(d_glob, n_g_eff - 1)[:n_g_eff]
            cands    = np.unique(np.concatenate([top_cas, top_glob]))

    # ── LWR ───────────────────────────────────────────────────────────────────
    if len(cands) < P_FIT + 2:
        return np.nan
    X_nn  = X_full[cands, -P_FIT:]
    y_nn  = y_base[cands]
    vec_f = query_pfit
    return _lwr(X_nn, y_nn, vec_f)


# ── walk-forward для одного тикера ────────────────────────────────────────────

def wf_ticker(att_full: np.ndarray, methods_cfg: list[tuple]) -> dict:
    """
    att_full — полный att-ряд тикера.
    methods_cfg — список (key, method, blend_frac).
    Протокол идентичен скр.80: origins = range(n-N*STEP, n-1, STEP),
    hist = att_full[:t_orig+1], actual = att_full[t_orig+1].
    """
    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP, n_total - 1, STEP))

    results: dict[str, dict] = {k: {"errors": [], "signs": []} for k, *_ in methods_cfg}

    for t_orig in origins:
        if t_orig + 1 >= n_total:
            continue
        hist = att_full[:t_orig + 1]   # включает t_orig (как в скр.80)
        true = att_full[t_orig + 1]
        for key, method, blend_frac in methods_cfg:
            pred = _forecast_one(hist, method, blend_frac)
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


# ── метрики ───────────────────────────────────────────────────────────────────

def compute_metrics(errors: list[float], signs: list[int],
                    std_att: float) -> tuple[float, float]:
    if not errors:
        return np.nan, np.nan
    mae  = float(np.mean(np.abs(errors)))
    rmae = mae / std_att if std_att > 0 else np.nan
    sacc = float(np.mean(signs)) * 100.0
    return rmae, sacc


# ── основной блок ─────────────────────────────────────────────────────────────

# Сетка blend_frac для кривых A
BLEND_GRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]

# Фиксированные конфигурации для финальной таблицы
FIXED_METHODS: list[tuple] = [
    ("pos_only",     "pos_only",  1.0),
    ("acc_ang",      "acc_ang",   1.0),
    ("gpos_r30",     "gpos",      0.70),   # 70% каскад, 30% глобал-pos  → как в скр.80
    ("gacc_r30",     "gacc",      0.70),   # 70% каскад, 30% глобал-acc
    ("gacc_r50",     "gacc",      0.50),   # 50/50
    ("gacc_r20",     "gacc",      0.80),   # 80% каскад, 20% глобал-acc
    ("gacc_r10",     "gacc",      0.90),   # 90% каскад, 10% глобал-acc
]

print("Загрузка + Local Projective...")
t0 = time.time()
att_data: dict[str, np.ndarray] = {}
for tkr in TICKERS:
    t1 = time.time()
    att_data[tkr] = load_att(tkr)
    print(f"  {tkr}... {time.time()-t1:.1f}с")
print(f"Загрузка итого: {time.time()-t0:.1f}с")

# ── фаза 1: финальная таблица по фиксированным методам ───────────────────────
print("\nФинальный walk-forward (фиксированные методы)...")
t0 = time.time()

all_res: dict[str, dict] = {k: {"errors": [], "signs": []} for k, *_ in FIXED_METHODS}
per_ticker: dict[str, dict[str, float]] = {}

for tkr in TICKERS:
    att  = att_data[tkr]
    std  = float(np.std(att))
    res  = wf_ticker(att, FIXED_METHODS)
    trow: dict[str, float] = {}
    print(f"  {tkr}: {N_ORIGINS} origins")
    for key, *_ in FIXED_METHODS:
        for metric in ("errors", "signs"):
            all_res[key][metric].extend(res[key][metric])
        rmae, sacc = compute_metrics(res[key]["errors"], res[key]["signs"], std)
        trow[key] = rmae
    per_ticker[tkr] = trow

print(f"Walk-forward завершён за {time.time()-t0:.1f}с")

# ── фаза 2: сетка blend_frac для gpos и gacc ─────────────────────────────────
print("\nСетка blend_frac для gpos / gacc...")
t0 = time.time()

grid_gpos: list[float] = []
grid_gacc: list[float] = []

std_all = [float(np.std(att_data[tkr])) for tkr in TICKERS]

for bf in BLEND_GRID:
    cfg = [("gpos", "gpos", bf), ("gacc", "gacc", bf)]
    errs_gpos: list[float] = []; errs_gacc: list[float] = []
    stds_rep:  list[float] = []
    for tkr, std in zip(TICKERS, std_all):
        res = wf_ticker(att_data[tkr], cfg)
        errs_gpos.extend(res["gpos"]["errors"])
        errs_gacc.extend(res["gacc"]["errors"])
        stds_rep.extend([std] * len(res["gpos"]["errors"]))
    # rMAE через средний std (нормировка как в скр.78-80)
    std_mean = float(np.mean(std_all))
    g_gpos = float(np.mean(np.abs(errs_gpos))) / std_mean if errs_gpos else np.nan
    g_gacc = float(np.mean(np.abs(errs_gacc))) / std_mean if errs_gacc else np.nan
    grid_gpos.append(g_gpos)
    grid_gacc.append(g_gacc)
    print(f"  blend_frac={bf:.1f}  gpos={g_gpos:.4f}  gacc={g_gacc:.4f}")

print(f"Сетка завершена за {time.time()-t0:.1f}с")

# ── итоговая таблица ──────────────────────────────────────────────────────────
std_mean = float(np.mean(std_all))

def rmae_all(key: str) -> float:
    return float(np.mean(np.abs(all_res[key]["errors"]))) / std_mean

def sacc_all(key: str) -> float:
    s = all_res[key]["signs"]
    return float(np.mean(s)) * 100.0 if s else np.nan

baseline = rmae_all("pos_only")

print("\nИтоговая таблица:")
for key, *_ in FIXED_METHODS:
    rmae = rmae_all(key); sacc = sacc_all(key)
    delta = (rmae / baseline - 1) * 100
    print(f"  {key:20s}: rMAE={rmae:.4f} ({delta:+.2f}%)  SignAcc={sacc:.1f}%")

# ── фигуры ────────────────────────────────────────────────────────────────────
names   = [k for k, *_ in FIXED_METHODS]
rmae_v  = [rmae_all(k) for k in names]
delta_v = [(r / baseline - 1) * 100 for r in rmae_v]
colors  = ["#90a4ae" if d >= 0 else "#66bb6a" for d in delta_v]
colors[0] = "#78909c"

# Рис. A — сравнение фиксированных методов
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(names, delta_v, color=colors)
ax.axvline(0, color="white", lw=0.8)
for bar, v in zip(bars, delta_v):
    ax.text(v + (0.3 if v >= 0 else -0.3), bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}%", va="center", ha="left" if v >= 0 else "right",
            color="white", fontsize=9)
ax.set_xlabel("Δ rMAE vs pos_only (%)")
ax.set_title(f"Рис.A  Глобальный acc_ang vs pos-only  (8 тикеров, {N_ORIGINS}×STEP={STEP})")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(FIG_DIR / "81_A.png", dpi=120)
plt.close()
print("Рис. A сохранён")

# Рис. B — кривые gpos / gacc по blend_frac
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(BLEND_GRID, grid_gpos, "o-", color="#ef5350", label="gpos (глобал по d_pos)")
ax.plot(BLEND_GRID, grid_gacc, "s-", color="#66bb6a", label="gacc (глобал по acc_ang)")
ax.axhline(rmae_all("acc_ang"),  color="#42a5f5", ls="--", lw=1.2,
           label=f"acc_ang (без глобала) {rmae_all('acc_ang'):.4f}")
ax.axhline(baseline, color="#90a4ae", ls=":", lw=1.2,
           label=f"pos_only baseline {baseline:.4f}")
ax.set_xlabel("blend_frac (доля ξ из каскада)")
ax.set_ylabel("rMAE")
ax.set_title("Рис.B  rMAE vs blend_frac")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "81_B.png", dpi=120)
plt.close()
print("Рис. B сохранён")

# Рис. C — per-ticker heatmap: acc_ang vs gacc_best
best_bf_idx = int(np.nanargmin(grid_gacc))
best_bf     = BLEND_GRID[best_bf_idx]

ticker_rmae_acc  = []
ticker_rmae_gacc = []

for tkr in TICKERS:
    att = att_data[tkr]; std = float(np.std(att))
    r1  = wf_ticker(att, [("acc_ang", "acc_ang", 1.0),
                          ("gacc_best", "gacc", best_bf)])
    ticker_rmae_acc.append(
        float(np.mean(np.abs(r1["acc_ang"]["errors"]))) / std)
    ticker_rmae_gacc.append(
        float(np.mean(np.abs(r1["gacc_best"]["errors"]))) / std)

delta_per = [(gacc / acc - 1) * 100
             for acc, gacc in zip(ticker_rmae_acc, ticker_rmae_gacc)]

fig, ax = plt.subplots(figsize=(9, 4))
c_list  = ["#66bb6a" if d < 0 else "#ef5350" for d in delta_per]
bars    = ax.bar(TICKERS, delta_per, color=c_list)
ax.axhline(0, color="white", lw=0.8)
for bar, v in zip(bars, delta_per):
    ax.text(bar.get_x() + bar.get_width() / 2, v + (0.2 if v >= 0 else -0.2),
            f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top",
            color="white", fontsize=9)
ax.set_ylabel("Δ rMAE gacc vs acc_ang (%)")
ax.set_title(f"Рис.C  Per-ticker: gacc(bf={best_bf}) vs acc_ang  (<0 = лучше)")
plt.tight_layout()
fig.savefig(FIG_DIR / "81_C.png", dpi=120)
plt.close()
print("Рис. C сохранён")

print(f"\nГотово. Оптимальный blend_frac для gacc: {best_bf}  (rMAE={grid_gacc[best_bf_idx]:.4f})")
print("Фигуры: 81_A...C.png")
