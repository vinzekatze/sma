"""
EXPERIMENT_ID : 95a_oracle_distribution
VERSION       : 1.0
ФАЗА          : 5 — исследование аттрактора

Диагностический эксперимент: распределение oracle_p при расширении
сетки до p_reg ∈ [2..80] с p-aligned каскадом и P_MAX=300.

Запуск:
  python 95a_oracle_distribution.py --mode test   # ~3-5 мин
  python 95a_oracle_distribution.py --mode full   # ~20-60 мин

Выходные данные пишутся онлайн (flush после каждого origin).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ── Идентификаторы ────────────────────────────────────────────────────────────

EXPERIMENT_ID = "95a_oracle_distribution"
VERSION       = "1.0"

# ── Пути (env-переменные позволяют переопределить из Docker) ──────────────────

_HERE    = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR",    str(_HERE.parent.parent.parent / "data" / "candles")))
OUT_DIR  = Path(os.environ.get("RESULTS_DIR", str(_HERE / "results")))
FIG_DIR  = Path(os.environ.get("FIGURES_DIR", str(_HERE / "figures")))
INTERVAL = "1d"

# ── Фиксированные параметры ───────────────────────────────────────────────────

P_MAX  = 300    # размерность поиска; X_full хранит P_MAX последних значений att
LAMBDA = 0.01   # вес acc_ang в метрике поиска соседей

LP_M, LP_D, LP_K, LP_N = 9, 3, 30, 3   # параметры Local Projective фильтра

# ── Конфигурация режимов ──────────────────────────────────────────────────────

MODES: dict[str, dict] = {
    "test": dict(
        tickers   = ["SBER"],
        n_origins = 5,
        step      = 40,
        p_grid    = [2, 5, 10, 20, 40, 60, 80],
        desc      = "ТЕСТОВЫЙ ПРОГОН  (1 тикер · 5 origins · 7 значений p)",
    ),
    "full": dict(
        tickers   = ["SBER"],
        n_origins = 20,
        step      = 10,
        p_grid    = list(range(2, 81)),
        desc      = "ПОЛНЫЙ ПРОГОН    (1 тикер · 20 origins · p∈[2..80])",
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Математические вспомогательные функции
# ══════════════════════════════════════════════════════════════════════════════

def _logtrend(close: np.ndarray) -> np.ndarray:
    """Причинный OLS логтренд: ratio = close / exp(a + b·t)."""
    n  = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);    ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc);   cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a  = (cy - b * ct) / cn
    tr = np.exp(a + b * t)
    tr[:2] = close[:2]
    return tr


def _lp_proj(ratio: np.ndarray, m: int, d: int, k: int, n_iter: int) -> np.ndarray:
    """Local Projective фильтр → возвращает diff(att), то есть Δatt."""
    from scipy.spatial import KDTree
    s = ratio.copy().astype(np.float64)
    N = len(s)
    k_eff = min(k, N - m)
    d_eff = min(d, m - 1)
    for _ in range(n_iter):
        n_pts = N - m + 1
        if n_pts < k_eff + 1:
            break
        rows = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
        X = s[rows]
        tree = KDTree(X)
        _, inds = tree.query(X, k=k_eff + 1)
        X_proj = np.empty_like(X)
        for i in range(n_pts):
            nn     = inds[i, 1:]
            X_nn   = X[nn]
            cen    = X_nn.mean(axis=0)
            _, _, Vt = np.linalg.svd(X_nn - cen, full_matrices=False)
            V_d    = Vt[:d_eff].T
            xc     = X[i] - cen
            X_proj[i] = cen + V_d @ (V_d.T @ xc)
        result = np.zeros(N)
        count  = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i + m] += X_proj[i]
            count[i:i + m]  += 1
        s = result / np.maximum(count, 1)
    return np.diff(s)


def _cosine_dist(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Косинусное расстояние между строками A и вектором b ∈ [0, 2]."""
    nA = np.linalg.norm(A, axis=1)
    nb = float(np.linalg.norm(b))
    if nb < 1e-12:
        return np.ones(len(A))
    cos = np.where(nA > 1e-12, (A @ b) / (nA * nb), 0.0)
    return 1.0 - np.clip(cos, -1.0, 1.0)


def _lwr_predict(X_nn: np.ndarray, y_nn: np.ndarray, q: np.ndarray) -> float:
    """LWR прогноз: МНК с гауссовыми весами по расстоянию от q."""
    d  = np.linalg.norm(X_nn - q, axis=1)
    h  = max(float(d.max()), 1e-10)
    w  = np.exp(-0.5 * (d / h) ** 2)
    A  = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw = np.sqrt(np.maximum(w, 1e-30))
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + q @ c[1:])


def _lwr_loo(X_nn: np.ndarray, y_nn: np.ndarray, q: np.ndarray) -> float:
    """LOO-ошибка LWR через hat-матрицу (аналитически)."""
    d  = np.linalg.norm(X_nn - q, axis=1)
    h  = max(float(d.max()), 1e-10)
    w  = np.maximum(np.exp(-0.5 * (d / h) ** 2), 1e-30)
    A  = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    resid  = y_nn - A @ c
    try:
        Aw       = sw[:, None] * A
        AtWA_inv = np.linalg.inv(Aw.T @ Aw + 1e-10 * np.eye(A.shape[1]))
        h_diag   = w * np.einsum("ij,ij->i", A @ AtWA_inv, A)
        denom    = 1.0 - h_diag
        e_loo    = np.where(np.abs(denom) > 1e-6, resid / denom, resid)
    except np.linalg.LinAlgError:
        e_loo = resid
    return float(np.mean(np.abs(e_loo)))


# ══════════════════════════════════════════════════════════════════════════════
# Каскад
# ══════════════════════════════════════════════════════════════════════════════

def levels_aligned(p_fit: int, p_max: int = P_MAX) -> list[int]:
    """
    P-aligned каскад ×2 (скр.93): p_fit × {1, 2, 4, ...} ≤ p_max.
    Возвращает уровни в порядке убывания (от крупного к мелкому).
    Гарантирует вложенность: соседи в 2p-D близки и в p-D.
    """
    levs = [p_fit]
    p    = p_fit
    while p * 2 <= p_max:
        p *= 2
        levs.append(p)
    return list(reversed(levs))


# ══════════════════════════════════════════════════════════════════════════════
# Контекст одного origin (матрицы, общие для всех p_reg)
# ══════════════════════════════════════════════════════════════════════════════

class _Ctx:
    """Предвычисленные матрицы для одного origin. Строятся один раз."""

    __slots__ = ("X_full", "X_acc", "y_base", "vec_full", "vec_acc", "n_lib", "ok")

    def __init__(self, att: np.ndarray) -> None:
        self.ok = False
        n = len(att)
        if n - P_MAX - 1 < 3:
            return

        # acc = вторая разность att (ускорение / кривизна)
        acc      = np.zeros(n)
        acc[2:]  = att[2:] - 2 * att[1:-1] + att[:-2]

        # X_full[i] = att[i+1 : i+1+P_MAX]  (view, без копирования)
        # Строка i соответствует времени t_arr[i] = P_MAX + i
        att_wins = sliding_window_view(att[:-1], P_MAX)   # (n-1-P_MAX+1, P_MAX)
        acc_wins = sliding_window_view(acc[:-1], P_MAX)
        # t_arr = range(P_MAX, n-1)  →  wins[t_arr - P_MAX] = wins[range(0, n-P_MAX-1)]
        self.X_full = np.asarray(att_wins[:n - P_MAX - 1])  # (N_lib, P_MAX)
        self.X_acc  = np.asarray(acc_wins[:n - P_MAX - 1])  # (N_lib, P_MAX)
        self.y_base = att[P_MAX + 1:n].copy()               # (N_lib,)
        self.n_lib  = len(self.y_base)

        # Вектора запроса
        self.vec_full = att[n - P_MAX:n].copy()   # последние P_MAX значений att
        self.vec_acc  = acc[n - P_MAX:n].copy()

        self.ok = True


# ══════════════════════════════════════════════════════════════════════════════
# Прогноз для одного p_reg
# ══════════════════════════════════════════════════════════════════════════════

def forecast_p(ctx: _Ctx, p_reg: int) -> tuple[float, float]:
    """
    Возвращает (pred, loo_error) для p_reg с p-aligned каскадом.
    ξ = 3·(p_reg+1)+5  (адаптивный).
    """
    xi     = 3 * (p_reg + 1) + 5
    levels = levels_aligned(p_reg)
    cands  = np.arange(ctx.n_lib)

    # ── Каскадная фильтрация ──────────────────────────────────────────────────
    for k, p_lvl in enumerate(levels):
        is_last = (k == len(levels) - 1)
        if is_last:
            break
        # Оставляем top-ξ по расстоянию в p_lvl-D подпространстве
        xi_lvl = min(xi, len(cands))
        if len(cands) > xi_lvl:
            d = np.linalg.norm(
                ctx.X_full[cands, -p_lvl:] - ctx.vec_full[-p_lvl:], axis=1
            )
            cands = cands[np.argpartition(d, xi_lvl - 1)[:xi_lvl]]
        # Временно́е расширение: ±radius шагов
        radius = p_lvl - levels[k + 1]
        if radius > 0:
            exp   = cands[:, None] - np.arange(radius + 1)[None, :]
            cands = np.unique(np.clip(exp, 0, ctx.n_lib - 1))

    # ── Финальный acc_ang отбор ───────────────────────────────────────────────
    if len(cands) > xi:
        d_pos = np.linalg.norm(
            ctx.X_full[cands, -p_reg:] - ctx.vec_full[-p_reg:], axis=1
        )
        d_acc = _cosine_dist(ctx.X_acc[cands, -p_reg:], ctx.vec_acc[-p_reg:])
        idx   = np.argpartition(d_pos + LAMBDA * d_acc, xi - 1)[:xi]
        cands = cands[idx]

    if len(cands) < p_reg + 2:
        return np.nan, np.nan

    X_nn = ctx.X_full[cands, -p_reg:]
    y_nn = ctx.y_base[cands]
    q    = ctx.vec_full[-p_reg:]

    return _lwr_predict(X_nn, y_nn, q), _lwr_loo(X_nn, y_nn, q)


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка данных
# ══════════════════════════════════════════════════════════════════════════════

def load_att(ticker: str) -> np.ndarray:
    path  = DATA_DIR / ticker / f"{INTERVAL}.json"
    raw   = json.loads(path.read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lt    = _logtrend(close)
    ratio = close / np.maximum(lt, 1e-10)
    att   = _lp_proj(ratio, LP_M, LP_D, LP_K, LP_N)
    return att


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward для одного тикера
# ══════════════════════════════════════════════════════════════════════════════

def run_ticker(
    ticker: str,
    att_full: np.ndarray,
    p_grid: list[int],
    n_origins: int,
    step: int,
    out_file: Path,
    std_att: float,
) -> None:
    n_total = len(att_full)
    origins = list(range(n_total - n_origins * step, n_total - 1, step))

    for idx_o, t_orig in enumerate(origins):
        if t_orig + 1 >= n_total:
            continue
        t0_o = time.time()

        hist = att_full[: t_orig + 1]
        true = float(att_full[t_orig + 1])

        ctx = _Ctx(hist)
        if not ctx.ok:
            print(f"    origin {t_orig}: пропущен (короткий ряд)")
            continue

        per_p: dict[str, dict] = {}
        for p in p_grid:
            pred, loo = forecast_p(ctx, p)
            per_p[str(p)] = {
                "pred":   float(pred) if not np.isnan(pred) else None,
                "loo":    float(loo)  if not np.isnan(loo)  else None,
                "xi":     3 * (p + 1) + 5,
                "levels": levels_aligned(p),
            }

        record = {
            "exp_id":   EXPERIMENT_ID,
            "version":  VERSION,
            "ticker":   ticker,
            "t_orig":   int(t_orig),
            "true_att": true,
            "std_att":  float(std_att),
            "per_p":    per_p,
        }

        # Онлайн-запись: flush сразу после каждого origin
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        elapsed = time.time() - t0_o
        print(
            f"    origin {idx_o + 1}/{len(origins)}  t={t_orig}"
            f"  true={true:+.5f}  [{elapsed:.1f}с]",
            flush=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Анализ результатов и фигуры
# ══════════════════════════════════════════════════════════════════════════════

def analyse(jsonl_path: Path, p_grid: list[int], mode: str) -> dict:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not rows:
        print("Нет данных для анализа.")
        return {}

    std_mean = float(np.mean([r["std_att"] for r in rows]))

    # Сбор ошибок per-p
    errors_by_p: dict[int, list[float]] = {p: [] for p in p_grid}
    loo_by_p:    dict[int, list[float]] = {p: [] for p in p_grid}
    oracle_ps:   list[int]              = []
    ensemble_errors: list[float]        = []
    true_vals:   list[float]            = []

    for r in rows:
        true = r["true_att"]
        true_vals.append(true)
        preds:    dict[int, float] = {}
        loos:     dict[int, float] = {}
        for p in p_grid:
            v = r["per_p"].get(str(p), {})
            pred_v = v.get("pred")
            loo_v  = v.get("loo")
            if pred_v is not None and not np.isnan(pred_v):
                preds[p] = pred_v
                errors_by_p[p].append(pred_v - true)
            if loo_v is not None and not np.isnan(loo_v):
                loos[p] = loo_v
                loo_by_p[p].append(loo_v)

        valid = [p for p in p_grid if p in preds]
        if not valid:
            continue

        # Oracle
        best_p = min(valid, key=lambda p: abs(preds[p] - true))
        oracle_ps.append(best_p)

        # Ensemble 1/LOO
        valid_loo = [p for p in valid if p in loos and loos[p] > 0]
        if valid_loo:
            inv = np.array([1.0 / loos[p] for p in valid_loo])
            w   = inv / inv.sum()
            ens = float(w @ np.array([preds[p] for p in valid_loo]))
            ensemble_errors.append(ens - true)

    # rMAE per-p
    rmae_by_p = {
        p: float(np.mean(np.abs(errors_by_p[p]))) / std_mean
        if errors_by_p[p] else np.nan
        for p in p_grid
    }

    # Baseline: фиксированный p=16 (ближайший в сетке к стандарту)
    p_base = min(p_grid, key=lambda p: abs(p - 16))
    base_rmae = rmae_by_p.get(p_base, np.nan)

    oracle_rmae  = float(np.mean(np.abs([
        errors_by_p[op][i] - 0
        for i, op in enumerate(oracle_ps)
        if oracle_ps[i] in errors_by_p and i < len(errors_by_p[oracle_ps[i]])
    ]))) if oracle_ps else np.nan

    # Считаем oracle rMAE правильно
    oracle_abs_errs = []
    row_idx = 0
    for r in rows:
        true = r["true_att"]
        preds = {
            p: r["per_p"][str(p)]["pred"]
            for p in p_grid
            if r["per_p"].get(str(p), {}).get("pred") is not None
        }
        valid = list(preds.keys())
        if not valid:
            continue
        best_p = min(valid, key=lambda p: abs(preds[p] - true))
        oracle_abs_errs.append(abs(preds[best_p] - true))
        row_idx += 1

    oracle_rmae = float(np.mean(oracle_abs_errs)) / std_mean if oracle_abs_errs else np.nan
    ens_rmae    = float(np.mean(np.abs(ensemble_errors))) / std_mean if ensemble_errors else np.nan

    summary = {
        "exp_id":         EXPERIMENT_ID,
        "version":        VERSION,
        "mode":           mode,
        "n_origins":      len(rows),
        "p_grid":         p_grid,
        "std_mean":       std_mean,
        "base_p":         p_base,
        "base_rmae":      base_rmae,
        "oracle_rmae":    oracle_rmae,
        "ensemble_rmae":  ens_rmae,
        "oracle_p_dist":  dict(Counter(oracle_ps)),
        "rmae_by_p":      {str(p): v for p, v in rmae_by_p.items()},
    }

    summary_path = OUT_DIR / f"summary_{mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nСводка записана: {summary_path}")

    # ── Печать ключевых результатов ───────────────────────────────────────────

    print(f"\n{'─'*60}")
    print(f"  base (p={p_base}) rMAE : {base_rmae:.4f}")
    print(f"  oracle rMAE       : {oracle_rmae:.4f}"
          f"  ({(oracle_rmae/base_rmae - 1)*100:+.1f}%)" if not np.isnan(base_rmae) else "")
    print(f"  ensemble rMAE     : {ens_rmae:.4f}"
          f"  ({(ens_rmae/base_rmae - 1)*100:+.1f}%)" if not np.isnan(base_rmae) else "")
    print(f"{'─'*60}")

    if oracle_ps:
        cnt = Counter(oracle_ps)
        print(f"\nОраклов p-распределение (топ-15):")
        for p, c in sorted(cnt.items(), key=lambda x: -x[1])[:15]:
            bar = "█" * round(c / len(oracle_ps) * 40)
            print(f"  p={p:3d}: {c:3d} ({c*100//len(oracle_ps):2d}%)  {bar}")
        print(f"\n  median oracle_p : {int(np.median(oracle_ps))}")
        print(f"  mean  oracle_p  : {np.mean(oracle_ps):.1f}")
        print(f"  % oracle_p > 16 : {sum(p > 16 for p in oracle_ps)*100/len(oracle_ps):.1f}%")
        print(f"  % oracle_p > 32 : {sum(p > 32 for p in oracle_ps)*100/len(oracle_ps):.1f}%")

    # ── Фигуры ────────────────────────────────────────────────────────────────

    _plot_oracle_dist(oracle_ps, p_grid, mode)
    _plot_rmae_by_p(rmae_by_p, p_grid, p_base, base_rmae, oracle_rmae, ens_rmae, mode)
    _plot_cumulative(oracle_ps, p_grid, mode)

    return summary


def _plot_oracle_dist(oracle_ps: list[int], p_grid: list[int], mode: str) -> None:
    if not oracle_ps:
        return
    cnt = Counter(oracle_ps)
    ps  = list(range(min(p_grid), max(p_grid) + 1))
    ys  = [cnt.get(p, 0) / len(oracle_ps) * 100 for p in ps]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(ps, ys, color="#42a5f5", width=0.8)
    ax.axvline(16, color="#ffa726", ls="--", lw=1.5, label="p=16 (прошлый потолок)")
    ax.axvline(float(np.median(oracle_ps)), color="#66bb6a", ls="-", lw=1.5,
               label=f"медиана={int(np.median(oracle_ps))}")
    ax.set_xlabel("p_reg")
    ax.set_ylabel("% origins")
    ax.set_title(f"95a  Распределение oracle_p  (N={len(oracle_ps)}, режим={mode})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / f"95a_oracle_dist_{mode}.png"
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"Рис. oracle_dist → {out.name}")


def _plot_rmae_by_p(
    rmae_by_p: dict[int, float],
    p_grid: list[int],
    p_base: int,
    base_rmae: float,
    oracle_rmae: float,
    ens_rmae: float,
    mode: str,
) -> None:
    ps   = [p for p in p_grid if not np.isnan(rmae_by_p.get(p, np.nan))]
    vals = [rmae_by_p[p] for p in ps]
    if not ps:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ps, vals, color="#42a5f5", lw=1.5, marker="o", ms=3, label="rMAE(p)")
    ax.axvline(p_base, color="#ffa726", ls="--", lw=1.2, label=f"p_base={p_base}")
    if not np.isnan(oracle_rmae):
        ax.axhline(oracle_rmae, color="#66bb6a", ls=":", lw=1.5,
                   label=f"oracle rMAE={oracle_rmae:.4f}")
    if not np.isnan(ens_rmae):
        ax.axhline(ens_rmae, color="#ef5350", ls=":", lw=1.5,
                   label=f"ensemble rMAE={ens_rmae:.4f}")
    ax.set_xlabel("p_reg")
    ax.set_ylabel("rMAE")
    ax.set_title(f"95a  rMAE как функция p_reg  (режим={mode})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / f"95a_rmae_by_p_{mode}.png"
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"Рис. rmae_by_p → {out.name}")


def _plot_cumulative(oracle_ps: list[int], p_grid: list[int], mode: str) -> None:
    if not oracle_ps:
        return
    sorted_p = sorted(oracle_ps)
    cdf_x    = sorted_p
    cdf_y    = np.arange(1, len(sorted_p) + 1) / len(sorted_p) * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(cdf_x, cdf_y, color="#42a5f5", lw=2, where="post")
    ax.axvline(16, color="#ffa726", ls="--", lw=1.5, label="p=16")
    ax.axhline(50, color="white",   ls=":",  lw=0.8)
    ax.set_xlabel("p_reg")
    ax.set_ylabel("% origins с oracle_p ≤ p")
    ax.set_title(f"95a  CDF oracle_p  (режим={mode})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / f"95a_cumulative_{mode}.png"
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"Рис. cumulative → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_ID} v{VERSION}")
    parser.add_argument("--mode", choices=["test", "full"], default="full")
    parser.add_argument("--n_origins", type=int, default=None,
                        help="Переопределить количество origins из режима")
    parser.add_argument("--step", type=int, default=None,
                        help="Переопределить шаг между origins")
    args = parser.parse_args()
    cfg  = dict(MODES[args.mode])   # копия, чтобы не менять константу
    if args.n_origins is not None:
        cfg["n_origins"] = args.n_origins
    if args.step is not None:
        cfg["step"] = args.step

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Баннер ────────────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  EXPERIMENT : {EXPERIMENT_ID}")
    print(f"  VERSION    : {VERSION}")
    print(f"  РЕЖИМ      : {cfg['desc']}")
    print(f"  DATA_DIR   : {DATA_DIR}")
    print(f"  OUT_DIR    : {OUT_DIR}")
    print(f"  P_MAX      : {P_MAX}")
    print(f"  p_grid     : [{cfg['p_grid'][0]}..{cfg['p_grid'][-1]}]"
          f"  ({len(cfg['p_grid'])} значений)")
    print(f"  ξ(p)       : 3·(p+1)+5")
    print("=" * 65)

    t_global = time.time()

    for ticker in cfg["tickers"]:
        print(f"\n── {ticker} ─────────────────────────────────────────────────")
        att = load_att(ticker)
        std_att = float(np.std(att))
        print(f"   att: {len(att)} баров  std={std_att:.5f}  N_lib≈{len(att)-P_MAX}")

        out_file = OUT_DIR / f"origins_{ticker}_{args.mode}.jsonl"
        if out_file.exists():
            out_file.unlink()   # начинаем чисто

        run_ticker(
            ticker    = ticker,
            att_full  = att,
            p_grid    = cfg["p_grid"],
            n_origins = cfg["n_origins"],
            step      = cfg["step"],
            out_file  = out_file,
            std_att   = std_att,
        )

        print(f"\n  Анализ результатов {ticker}...")
        analyse(out_file, cfg["p_grid"], args.mode)

    total = time.time() - t_global
    print(f"\n{'='*65}")
    print(f"  Готово за {total/60:.1f} мин")
    print(f"  Результаты: {OUT_DIR}")
    print(f"  Фигуры:     {FIG_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
