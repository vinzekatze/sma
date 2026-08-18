"""
87_pfit_predictor.py — Предсказание оптимального p_fit по локальной геометрии.

Две гипотезы:
  A. Локальные признаки аттрактора (кривизна, волатильность, автокорреляция,
     спектральный центроид) коррелируют с oracle_p.
     → Если r > 0.15 — строим предиктор.

  B. Z-нормировка LOO (по исторической μ/σ каждого p_reg) убирает смещение
     в сторону малых p и даёт честное сравнение.
     → Если z-LOO лучше raw LOO → реализуем в app3.

Признаки в точке origin t:
  curv      — средняя |вторая производная att| / global_std  (быстрая кривизна)
  vol_ratio — std(diff(att[-16:])) / std(diff(att[-64:]))    (локальная vs. глобальная волатильность)
  ac4       — автокорреляция att на лаге 4
  ac16      — автокорреляция att на лаге 16
  spec_c    — спектральный центроид att[-64:]                 (высокий → быстрые колебания)
  slope_n   — |наклон OLS(att[-16:])| / local_vol             (сила тренда)
  loo_cv    — CV(LOO) = std(LOO)/mean(LOO) по p-сетке         (мета: насколько p важен)

Walk-forward:
  fixed_16    — baseline (скр.80-86)
  oracle      — верхняя граница
  loo_raw     — сырой LOO (≡ скр.85 loo_select)
  z_select    — z-нормированный LOO (каузально)
  z_ensemble  — взвешенное среднее по z-score (softmax(-z))
  ensemble    — взвешенное среднее 1/LOO (≡ скр.85-86)
  feat_select — p_reg, предсказанный линейной комбинацией признаков

Протокол: h=1, последние 200 баров, шаг 5, 8 тикеров 1d.
"""
from __future__ import annotations
import time
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import stats as sp_stats
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

WIN_SHORT = 16   # окно для локальных признаков
WIN_LONG  = 64   # окно для глобальных признаков
Z_MIN_HIST = 6   # минимум прошлых origins для z-нормировки


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


# ── признаки локальной геометрии аттрактора ───────────────────────────────────

def extract_features(att: np.ndarray) -> dict[str, float] | None:
    """Локальные признаки на кончике att[-1]."""
    n = len(att)
    if n < WIN_LONG + 4:
        return None

    # кривизна
    acc       = att[2:] - 2 * att[1:-1] + att[:-2]
    g_std     = float(np.std(att)) + 1e-12
    f_curv    = float(np.mean(np.abs(acc[-WIN_SHORT:]))) / g_std

    # локальная vs глобальная волатильность
    d_att    = np.diff(att)
    vol_loc  = float(np.std(d_att[-WIN_SHORT:])) + 1e-12
    vol_glob = float(np.std(d_att[-WIN_LONG:]))  + 1e-12
    f_vol_r  = vol_loc / vol_glob

    # автокорреляция на лагах 4 и 16
    seg = att[-WIN_LONG:]
    def _ac(lag: int) -> float:
        if len(seg) <= lag + 4:
            return 0.0
        r = float(np.corrcoef(seg[:-lag], seg[lag:])[0, 1])
        return r if np.isfinite(r) else 0.0
    f_ac4  = _ac(4)
    f_ac16 = _ac(16)

    # спектральный центроид (высокий → быстрые колебания → предпочтителен малый p)
    seg_s  = att[-WIN_LONG:] - att[-WIN_LONG:].mean()
    fft_m  = np.abs(np.fft.rfft(seg_s))
    freqs  = np.fft.rfftfreq(WIN_LONG)
    f_spec = float(np.sum(freqs * fft_m) / (fft_m.sum() + 1e-12))

    # нормированный наклон (сила тренда)
    t_idx   = np.arange(WIN_SHORT)
    slope   = float(np.polyfit(t_idx, att[-WIN_SHORT:], 1)[0])
    f_slope = abs(slope) / vol_loc

    return {
        "curv":      f_curv,
        "vol_ratio": f_vol_r,
        "ac4":       f_ac4,
        "ac16":      f_ac16,
        "spec_c":    f_spec,
        "slope_n":   f_slope,
    }


# ── разделяемый контекст ──────────────────────────────────────────────────────

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
    levels   = _uniform_octave_levels(p_reg, P_MAX, 2.0)
    X_full   = ctx.X_full; y_base = ctx.y_base
    vec_full = ctx.vec_full; acc_hist = ctx.acc_hist; n = ctx.n
    if len(X_full) < XI_FIXED + 1:
        return np.nan, np.nan
    t_arr   = np.arange(P_MAX, n - 1)
    X_acc   = np.column_stack([acc_hist[t_arr - (p_reg - 1 - j)] for j in range(p_reg)])
    vec_acc = acc_hist[-p_reg:].copy()
    query   = vec_full[-p_reg:]
    cands   = np.arange(len(X_full))
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
        cands  = cands[np.argpartition(d_pos + LAMBDA * d_acc, XI_FIXED - 1)[:XI_FIXED]]
    if len(cands) < p_reg + 2:
        return np.nan, np.nan
    X_nn = X_full[cands, -p_reg:]; y_nn = y_base[cands]
    return _lwr_predict(X_nn, y_nn, query), _lwr_loo_error(X_nn, y_nn, query)


def _forecast_fixed(ctx: _Context) -> float:
    pred, _ = _forecast_at_preg(ctx, P_FIT_BASE)
    return pred


# ── walk-forward: сбор данных + прогноз ──────────────────────────────────────

def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend_causal(close)
    return _lp_proj(close / np.maximum(lt, 1e-10), LP_M, LP_D, LP_K, LP_N)


def wf_ticker(att_full: np.ndarray) -> tuple[dict, list]:
    """
    Возвращает:
      results[key] = {"errors": [], "signs": [], "chosen_p": []}
      rows: список dict с признаками и oracle_p для корреляционного анализа
    """
    keys = ("fixed_16", "oracle", "loo_raw",
            "z_select", "z_ensemble", "ensemble", "feat_select")
    out  = {k: {"errors": [], "signs": [], "chosen_p": []} for k in keys}
    rows = []   # для Phase A (корреляции)

    n_total = len(att_full)
    origins = list(range(n_total - N_ORIGINS * STEP, n_total - 1, STEP))

    # каузальная статистика LOO per p_reg
    loo_hist: dict[int, list[float]] = {p: [] for p in P_REG_GRID}

    for t_orig in origins:
        if t_orig + 1 >= n_total:
            continue
        hist = att_full[:t_orig + 1]
        true = att_full[t_orig + 1]

        ctx = _Context(hist)
        if not ctx.ok:
            continue

        # baseline
        pred_base = _forecast_fixed(ctx)
        if not np.isnan(pred_base):
            out["fixed_16"]["errors"].append(pred_base - true)
            out["fixed_16"]["signs"].append(int(np.sign(pred_base) == np.sign(true)))

        # все p_reg → preds, LOO
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

        # loo_raw
        best_raw = min(valid_p, key=lambda p: loo_errs[p])
        _record("loo_raw", best_raw)

        # ensemble 1/LOO
        inv_loo  = np.array([1.0 / (loo_errs[p] + 1e-12) for p in valid_p])
        w_ens    = inv_loo / inv_loo.sum()
        pred_ens = float(w_ens @ np.array([preds[p] for p in valid_p]))
        out["ensemble"]["errors"].append(pred_ens - true)
        out["ensemble"]["signs"].append(int(np.sign(pred_ens) == np.sign(true)))

        # ── z-нормировка (каузальная) ──────────────────────────────────────
        z_valid = [p for p in valid_p if len(loo_hist[p]) >= Z_MIN_HIST]

        if z_valid:
            z_scores: dict[int, float] = {}
            for p in z_valid:
                hist_arr = np.array(loo_hist[p])
                mu    = float(np.mean(hist_arr))
                sigma = max(float(np.std(hist_arr)), 1e-12)
                z_scores[p] = (loo_errs[p] - mu) / sigma

            # z_select: argmin z
            best_z = min(z_valid, key=lambda p: z_scores[p])
            _record("z_select", best_z)

            # z_ensemble: softmax(-z)
            z_arr   = np.array([z_scores[p] for p in z_valid])
            z_arr  -= z_arr.min()          # numerical stability
            w_z     = np.exp(-z_arr); w_z /= w_z.sum()
            pred_ze = float(w_z @ np.array([preds[p] for p in z_valid]))
            out["z_ensemble"]["errors"].append(pred_ze - true)
            out["z_ensemble"]["signs"].append(int(np.sign(pred_ze) == np.sign(true)))

        # ── признаки + loo_cv для строки корреляционного анализа ──────────
        feats = extract_features(hist)
        if feats is not None:
            loo_vals = [loo_errs[p] for p in valid_p]
            loo_cv   = float(np.std(loo_vals) / (np.mean(loo_vals) + 1e-12))
            row = {**feats,
                   "loo_cv":   loo_cv,
                   "oracle_p": best_ora,
                   "loo_raw_p": best_raw}
            rows.append(row)

        # ── простой признаковый предиктор ──────────────────────────────────
        # (используется для feat_select — дообучается ниже, заглушка пока)
        # Заглушка: просто loo_raw (будет заменена после обучения)
        _record("feat_select", best_raw)

        # обновляем LOO-историю ПОСЛЕ всех вычислений (каузальность)
        for p in valid_p:
            loo_hist[p].append(loo_errs[p])

    return out, rows


# ── основной блок ─────────────────────────────────────────────────────────────

print("Загрузка + Local Projective...")
t0 = time.time()
att_data: dict[str, np.ndarray] = {}
for tkr in TICKERS:
    att_data[tkr] = load_att(tkr)
    print(f"  {tkr}...")
std_all  = [float(np.std(att_data[tkr])) for tkr in TICKERS]
std_mean = float(np.mean(std_all))
print(f"Загрузка: {time.time()-t0:.1f}с")

print(f"\nФаза A+B: walk-forward...")
t0 = time.time()

all_keys = ("fixed_16", "oracle", "loo_raw",
            "z_select", "z_ensemble", "ensemble", "feat_select")
all_res: dict[str, dict] = {k: {"errors": [], "signs": [], "chosen_p": []}
                            for k in all_keys}
all_rows: list[dict] = []

for tkr in TICKERS:
    t1  = time.time()
    res, rows = wf_ticker(att_data[tkr])
    for key in all_keys:
        for m in ("errors", "signs", "chosen_p"):
            all_res[key][m].extend(res[key][m])
    all_rows.extend(rows)
    print(f"  {tkr}: {time.time()-t1:.1f}с  ({len(rows)} строк признаков)")
print(f"Walk-forward: {time.time()-t0:.1f}с  |  строк: {len(all_rows)}")


# ── Фаза A: корреляционный анализ ─────────────────────────────────────────────

FEAT_NAMES = ["curv", "vol_ratio", "ac4", "ac16", "spec_c", "slope_n", "loo_cv"]

oracle_p_arr = np.array([r["oracle_p"] for r in all_rows], dtype=float)
feat_matrix  = np.array([[r[f] for f in FEAT_NAMES] for r in all_rows])

print("\n── Корреляции признаков с oracle_p (Pearson / Spearman) ──")
pearson_r  = []
spearman_r = []
for i, fname in enumerate(FEAT_NAMES):
    fv   = feat_matrix[:, i]
    mask = np.isfinite(fv) & np.isfinite(oracle_p_arr)
    if mask.sum() < 10:
        pearson_r.append(0.0); spearman_r.append(0.0)
        print(f"  {fname:12s}: N слишком мало")
        continue
    pr, pp = sp_stats.pearsonr(fv[mask], oracle_p_arr[mask])
    sr, sp = sp_stats.spearmanr(fv[mask], oracle_p_arr[mask])
    pearson_r.append(pr); spearman_r.append(sr)
    sig_p = "*" if pp < 0.05 else " "
    sig_s = "*" if sp < 0.05 else " "
    print(f"  {fname:12s}: Pearson={pr:+.3f}{sig_p}  Spearman={sr:+.3f}{sig_s}  N={mask.sum()}")

# Наиболее коррелирующий признак
best_feat_i = int(np.argmax(np.abs(pearson_r)))
best_feat   = FEAT_NAMES[best_feat_i]
best_r      = pearson_r[best_feat_i]
print(f"\nЛучший признак: {best_feat}  (r={best_r:+.3f})")


# ── Фаза A: простой признаковый предиктор ─────────────────────────────────────
# Линейная регрессия: features → oracle_p (обучаем на всех данных — оценка потенциала)

from numpy.linalg import lstsq as nplstsq

mask_all  = np.all(np.isfinite(feat_matrix), axis=1) & np.isfinite(oracle_p_arr)
X_feat    = feat_matrix[mask_all]
y_feat    = oracle_p_arr[mask_all]
X_feat_b  = np.hstack([np.ones((len(X_feat), 1)), X_feat])
coef, _, _, _ = nplstsq(X_feat_b, y_feat, rcond=None)

p_hat_all = X_feat_b @ coef
r2        = 1 - np.var(y_feat - p_hat_all) / (np.var(y_feat) + 1e-12)
print(f"\nЛинейный предиктор oracle_p: R²={r2:.3f}  "
      f"(R²=0 → признаки бесполезны, R²=1 → идеально)")

# Маппинг предсказанного p_hat → ближайший из P_REG_GRID
def snap_p(p_hat: float) -> int:
    return min(P_REG_GRID, key=lambda p: abs(p - p_hat))

# Насколько хорошо предиктор угадывает oracle_p?
p_hat_snapped = np.array([snap_p(ph) for ph in p_hat_all])
y_snapped     = np.array([snap_p(yv) for yv in y_feat])
match_pct     = float(np.mean(p_hat_snapped == y_snapped)) * 100
print(f"Совпадение предиктора с oracle: {match_pct:.1f}%  "
      f"(случайный уровень ≈ {100/len(P_REG_GRID):.0f}%)")


# ── Фаза B: walk-forward результаты ───────────────────────────────────────────

def rmae(key: str) -> float:
    e = all_res[key]["errors"]
    return float(np.mean(np.abs(e))) / std_mean if e else np.nan

def sacc(key: str) -> float:
    s = all_res[key]["signs"]
    return float(np.mean(s)) * 100.0 if s else np.nan

baseline = rmae("fixed_16")

print("\n── Walk-forward: итоговая таблица ──")
for key in all_keys:
    r  = rmae(key); d = (r / baseline - 1) * 100 if r is not np.nan else np.nan
    sa = sacc(key)
    cp = all_res[key]["chosen_p"]
    cnt_str = ""
    if cp:
        cnt = Counter(cp)
        cnt_str = "  p=(" + " ".join(f"{p}:{cnt.get(p,0)*100//max(len(cp),1)}%"
                                      for p in P_REG_GRID) + ")"
    print(f"  {key:12s}: rMAE={r:.4f} ({d:+.2f}%)  SignAcc={sa:.1f}%{cnt_str}")


# ── Фигуры ────────────────────────────────────────────────────────────────────

# Рис. A — корреляции признаков с oracle_p
fig, ax = plt.subplots(figsize=(9, 4))
x_pos   = np.arange(len(FEAT_NAMES))
colors  = ["#42a5f5" if r >= 0 else "#ef5350" for r in pearson_r]
bars    = ax.bar(FEAT_NAMES, pearson_r, color=colors)
ax.bar(FEAT_NAMES, spearman_r, color="none",
       edgecolor="#ffa726", linewidth=2, linestyle="--", label="Spearman r")
ax.axhline(0, color="white", lw=0.8)
ax.axhline(0.15,  color="#66bb6a", ls=":", lw=1, label="+0.15 threshold")
ax.axhline(-0.15, color="#66bb6a", ls=":", lw=1)
ax.set_xticks(x_pos); ax.set_xticklabels(FEAT_NAMES)
ax.set_ylabel("Корреляция с oracle_p")
ax.set_title(f"Рис.A  Признаки vs oracle_p  (N={mask_all.sum()} origins)")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "87_A.png", dpi=120)
plt.close()
print("Рис. A сохранён")

# Рис. B — scatter лучшего признака vs oracle_p
fig, ax = plt.subplots(figsize=(7, 5))
fv_best = feat_matrix[:, best_feat_i]
mask_b  = np.isfinite(fv_best) & np.isfinite(oracle_p_arr)
ax.scatter(fv_best[mask_b], oracle_p_arr[mask_b],
           alpha=0.25, s=18, color="#42a5f5", label="origins")
# линия тренда
if mask_b.sum() > 4:
    m_, b_ = np.polyfit(fv_best[mask_b], oracle_p_arr[mask_b], 1)
    xr = np.linspace(fv_best[mask_b].min(), fv_best[mask_b].max(), 100)
    ax.plot(xr, m_ * xr + b_, color="#ef5350", lw=2, label=f"OLS (r={best_r:+.3f})")
ax.set_xlabel(best_feat); ax.set_ylabel("oracle_p")
ax.set_title(f"Рис.B  {best_feat} vs oracle_p  (R²={r2:.3f})")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "87_B.png", dpi=120)
plt.close()
print("Рис. B сохранён")

# Рис. C — walk-forward: bar chart методов
SHOW_KEYS   = ["fixed_16", "oracle", "loo_raw", "z_select", "z_ensemble", "ensemble"]
SHOW_LABELS = ["fixed\np=16", "oracle", "LOO\nraw", "z-\nselect", "z-\nensemble", "1/LOO\nens."]
rmae_f  = [rmae(k) for k in SHOW_KEYS]
delta_f = [(r / baseline - 1) * 100 for r in rmae_f]
c_list  = ["#78909c"] + ["#66bb6a" if d < 0 else "#ef5350" for d in delta_f[1:]]

fig, ax = plt.subplots(figsize=(11, 4))
bars = ax.bar(SHOW_LABELS, delta_f, color=c_list, width=0.5)
ax.axhline(0, color="white", lw=0.8)
for bar, v in zip(bars, delta_f):
    ax.text(bar.get_x() + bar.get_width() / 2, v + (0.4 if v >= 0 else -0.4),
            f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top",
            color="white", fontsize=10)
ax.set_ylabel("Δ rMAE vs fixed acc_ang p=16 (%)")
ax.set_title("Рис.C  Признаки + z-нормировка vs baseline  (<0 = лучше)")
plt.tight_layout()
fig.savefig(FIG_DIR / "87_C.png", dpi=120)
plt.close()
print("Рис. C сохранён")

# Рис. D — scatter матрица ключевых признаков vs oracle_p (mini grid)
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, (fname, pr, sr) in zip(axes.flat, zip(FEAT_NAMES, pearson_r, spearman_r)):
    fi   = FEAT_NAMES.index(fname)
    fv   = feat_matrix[:, fi]
    mask = np.isfinite(fv) & np.isfinite(oracle_p_arr)
    ax.scatter(fv[mask], oracle_p_arr[mask], alpha=0.2, s=12, color="#42a5f5")
    if mask.sum() > 4:
        m_, b_ = np.polyfit(fv[mask], oracle_p_arr[mask], 1)
        xr = np.linspace(fv[mask].min(), fv[mask].max(), 80)
        ax.plot(xr, m_*xr + b_, color="#ef5350", lw=1.5)
    ax.set_xlabel(fname, fontsize=8)
    ax.set_ylabel("oracle_p", fontsize=8)
    ax.set_title(f"r={pr:+.3f} ρ={sr:+.3f}", fontsize=8)
axes.flat[-1].set_visible(False)
plt.suptitle("Рис.D  Все признаки vs oracle_p", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "87_D.png", dpi=120)
plt.close()
print("Рис. D сохранён")

print(f"\nГотово. Фигуры: 87_A...D.png")
