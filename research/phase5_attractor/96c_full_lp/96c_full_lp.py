"""
EXPERIMENT_ID : 96c_full_lp
VERSION       : 1.0
ФАЗА          : 5 — исследование аттрактора

LP-коррекция на всех 200 origins из 96a + реконструкция реальных цен.

Метрики:
  att-уровень : rMAE = |att_hat - att_true| / std(att_causal)
  цена-уровень: MAPE = |price_hat - price_true| / price_true × 100%

Реконструкция цены из att-прогноза:
  att[i] = lp_ratio[i+1] − lp_ratio[i]
  lp_ratio_hat[t+h] = lp_ratio[t_orig+1] + cumsum(att_hat[:h])
  price_hat[t+h]    = lp_ratio_hat[t+h]  × logtrend[t+h]

  att[t_orig+1..t_orig+H] → price bars [t_orig+2..t_orig+H+1]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import KDTree

EXPERIMENT_ID = "96c_full_lp"
VERSION       = "1.0"

_HERE    = Path(__file__).parent
DATA_DIR = _HERE.parent.parent.parent / "data" / "candles"
OUT_DIR  = _HERE / "results"
FIG_DIR  = _HERE / "figures"

PATH_96A_TAU1 = _HERE.parent / "96_tau_delay" / "results" / "origins_tau1_full.jsonl"

TICKER   = "SBER"
INTERVAL = "1d"

P_REG  = 16
H_MAX  = 20
K_PROJ = 30
D_PROJ = 3
M_LP   = 9
LAMBDA = 0.01
P_MAX  = 300

LP_M, LP_D, LP_K, LP_N = 9, 3, 30, 3


# ══════════════════════════════════════════════════════════════════════════════
# Данные
# ══════════════════════════════════════════════════════════════════════════════

def _logtrend(close: np.ndarray) -> np.ndarray:
    n = len(close); lc = np.log(np.maximum(close, 1e-10)); t = np.arange(n, dtype=float)
    cn = np.arange(1, n + 1, dtype=float); ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc); denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    tr = np.exp(a + b * t); tr[:2] = close[:2]
    return tr


def _lp_proj_with_ratio(ratio: np.ndarray, m: int, d: int, k: int, n_iter: int):
    """Возвращает (att, lp_ratio) — диффрат и LP-проекцию ratio."""
    s = ratio.copy().astype(float); N = len(s)
    k_eff = min(k, N - m); d_eff = min(d, m - 1)
    for _ in range(n_iter):
        n_pts = N - m + 1
        if n_pts < k_eff + 1:
            break
        rows = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
        X = s[rows]; tree = KDTree(X); _, inds = tree.query(X, k=k_eff + 1)
        Xp = np.empty_like(X)
        for i in range(n_pts):
            nn = inds[i, 1:]; Xnn = X[nn]; cen = Xnn.mean(0)
            _, _, Vt = np.linalg.svd(Xnn - cen, full_matrices=False)
            Vd = Vt[:d_eff].T; xc = X[i] - cen; Xp[i] = cen + Vd @ (Vd.T @ xc)
        res = np.zeros(N); cnt = np.zeros(N, int)
        for i in range(n_pts):
            res[i:i + m] += Xp[i]; cnt[i:i + m] += 1
        s = res / np.maximum(cnt, 1)
    return np.diff(s), s   # (att, lp_ratio)


def load_full_data(ticker: str):
    """Загружает OHLCV и возвращает (att, lp_ratio, logtrend, close).

    Все массивы кроме att имеют длину n_close.
    att имеет длину n_close−1.
    att[i] = lp_ratio[i+1] − lp_ratio[i].
    """
    raw   = json.loads((DATA_DIR / ticker / f"{INTERVAL}.json").read_text())
    close = np.array([c["close"] for c in raw], dtype=float)
    lt    = _logtrend(close)
    ratio = close / np.maximum(lt, 1e-10)
    att, lp_ratio = _lp_proj_with_ratio(ratio, LP_M, LP_D, LP_K, LP_N)
    return att, lp_ratio, lt, close


# ══════════════════════════════════════════════════════════════════════════════
# Библиотека (causal)
# ══════════════════════════════════════════════════════════════════════════════

class _Lib:
    __slots__ = ("X_full", "X_acc", "y_base", "n_lib", "X_lib_m", "ok")

    def __init__(self, att: np.ndarray) -> None:
        self.ok = False
        n = len(att)
        if n - P_MAX - 1 < 3:
            return
        att_wins    = sliding_window_view(att[:-1], P_MAX)
        n_lib       = n - P_MAX - 1
        self.X_full = np.asarray(att_wins[:n_lib])
        acc_full    = np.zeros(n)
        acc_full[2:] = att[2:] - 2 * att[1:-1] + att[:-2]
        acc_wins    = sliding_window_view(acc_full[:-1], P_MAX)
        self.X_acc  = np.asarray(acc_wins[:n_lib])
        self.y_base = att[P_MAX + 1:n].copy()
        self.n_lib  = n_lib
        self.X_lib_m = self.X_full[:, -M_LP:]
        self.ok = True


# ══════════════════════════════════════════════════════════════════════════════
# LWR + каскад (τ=1)
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_dist(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    nA = np.linalg.norm(A, axis=1); nb = float(np.linalg.norm(b))
    if nb < 1e-12:
        return np.ones(len(A))
    return 1.0 - np.clip((A @ b) / (np.where(nA > 1e-12, nA, 1.) * nb), -1., 1.)


def _lwr_predict(X_nn, y_nn, q):
    d = np.linalg.norm(X_nn - q, axis=1); h = max(float(d.max()), 1e-10)
    w = np.exp(-0.5 * (d / h) ** 2)
    A = np.hstack([np.ones((len(X_nn), 1)), X_nn]); sw = np.sqrt(np.maximum(w, 1e-30))
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + q @ c[1:])


def _levels(p_reg: int) -> list[int]:
    levs = [p_reg]; p = p_reg
    while p * 2 <= P_MAX:
        p *= 2; levs.append(p)
    return list(reversed(levs))


def _lwr_step(lib: _Lib, vec_full: np.ndarray, vec_acc: np.ndarray, p_reg: int) -> float:
    xi     = 3 * (p_reg + 1) + 5
    levels = _levels(p_reg)
    cands  = np.arange(lib.n_lib)
    for k_lev, p_lvl in enumerate(levels):
        if k_lev == len(levels) - 1:
            break
        xi_lvl = min(xi, len(cands))
        if len(cands) > xi_lvl:
            cols  = P_MAX - 1 - np.arange(p_lvl)
            d_arr = np.linalg.norm(lib.X_full[cands][:, cols] - vec_full[cols], axis=1)
            cands = cands[np.argpartition(d_arr, xi_lvl - 1)[:xi_lvl]]
        radius = p_lvl - levels[k_lev + 1]
        if radius > 0:
            exp   = cands[:, None] - np.arange(radius + 1)[None, :]
            cands = np.unique(np.clip(exp, 0, lib.n_lib - 1))
    cols_p = P_MAX - 1 - np.arange(p_reg)
    if len(cands) > xi:
        d_pos = np.linalg.norm(lib.X_full[cands][:, cols_p] - vec_full[cols_p], axis=1)
        d_acc = _cosine_dist(lib.X_acc[cands][:, cols_p], vec_acc[cols_p])
        cands = cands[np.argpartition(d_pos + LAMBDA * d_acc, xi - 1)[:xi]]
    if len(cands) < p_reg + 2:
        return np.nan
    X_nn = lib.X_full[cands][:, cols_p]; y_nn = lib.y_base[cands]; q = vec_full[cols_p]
    return _lwr_predict(X_nn, y_nn, q)


# ══════════════════════════════════════════════════════════════════════════════
# LP-коррекция точки
# ══════════════════════════════════════════════════════════════════════════════

def _lp_project_point(v_pred: np.ndarray, X_lib_m: np.ndarray, k: int, d: int) -> float:
    """Проецирует v_pred на локальное d-мерное подпространство аттрактора.
    v_pred[-1] — самое свежее значение (att_hat); возвращает скорректированное.
    """
    dists  = np.linalg.norm(X_lib_m - v_pred, axis=1)
    k_eff  = min(k, len(X_lib_m) - 1)
    idx    = np.argpartition(dists, k_eff)[:k_eff]
    X_nn   = X_lib_m[idx]
    center = X_nn.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_nn - center, full_matrices=False)
    Vd     = Vt[:min(d, len(Vt))].T
    v_c    = v_pred - center
    return float((center + Vd @ (Vd.T @ v_c))[-1])


# ══════════════════════════════════════════════════════════════════════════════
# Рекурсивный прогноз
# ══════════════════════════════════════════════════════════════════════════════

def forecast_recursive(
    att_full: np.ndarray,
    t_orig: int,
    lib: _Lib,
    lp_correct: bool,
) -> list[float]:
    context = list(att_full[t_orig - P_MAX + 1 : t_orig + 1])
    preds: list[float] = []
    for _ in range(H_MAX):
        vec_full = np.array(context[-P_MAX:])
        vec_acc  = np.zeros(P_MAX)
        vec_acc[2:] = vec_full[2:] - 2 * vec_full[1:-1] + vec_full[:-2]
        pred_raw = _lwr_step(lib, vec_full, vec_acc, P_REG)
        if np.isnan(pred_raw):
            pred_raw = context[-1]
        if lp_correct:
            v_lp = np.array(context[-(M_LP - 1):] + [pred_raw])
            pred_final = _lp_project_point(v_lp, lib.X_lib_m, K_PROJ, D_PROJ)
        else:
            pred_final = pred_raw
        preds.append(pred_final)
        context.append(pred_final)
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# Реконструкция цены
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_price(
    att_pred: list[float],
    t_orig: int,
    lp_ratio_full: np.ndarray,
    logtrend_full: np.ndarray,
) -> np.ndarray:
    """att_hat[t_orig+1..t_orig+H] → price_hat[t_orig+2..t_orig+H+1].

    Базовое значение lp_ratio — точка t_orig+1 (известна из наблюдённых данных).
    lp_ratio_hat[t_orig+h+1] = lp_ratio[t_orig+1] + cumsum(att_pred[:h])
    price_hat[t_orig+h+1]    = lp_ratio_hat[t_orig+h+1] × logtrend[t_orig+h+1]
    """
    base       = lp_ratio_full[t_orig + 1]
    ratio_hat  = base + np.cumsum(att_pred)                         # len=H
    price_idx  = np.arange(t_orig + 2, t_orig + 2 + len(att_pred)) # close indices
    lt_future  = logtrend_full[price_idx]
    return ratio_hat * lt_future


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка origins из 96a
# ══════════════════════════════════════════════════════════════════════════════

def load_all_origins(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return sorted(records, key=lambda r: r["t_orig"])


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{'='*70}")
    print(f"  {EXPERIMENT_ID}  v{VERSION}")
    print(f"  P_REG={P_REG}  H={H_MAX}  200 origins  LP: k={K_PROJ} d={D_PROJ} m={M_LP}")
    print(f"{'='*70}\n")

    att_full, lp_ratio_full, logtrend_full, close_full = load_full_data(TICKER)
    print(f"att_full len={len(att_full)}, close={len(close_full)}")

    all_recs = load_all_origins(PATH_96A_TAU1)
    print(f"Origins из 96a: {len(all_recs)}\n")

    results   = []
    rmae_lwr  = np.zeros(H_MAX)
    rmae_lp   = np.zeros(H_MAX)
    mape_lwr  = np.zeros(H_MAX)
    mape_lp   = np.zeros(H_MAX)
    n_valid   = 0

    t_start = time.time()

    for idx, rec in enumerate(all_recs):
        t_orig  = int(rec["t_orig"])
        std_att = float(rec["std_att"])

        if t_orig + H_MAX + 1 >= len(close_full):
            continue

        hist = att_full[:t_orig + 1]
        lib  = _Lib(hist)
        if not lib.ok:
            continue

        actual_att   = att_full[t_orig + 1 : t_orig + H_MAX + 1]
        actual_price = close_full[t_orig + 2 : t_orig + H_MAX + 2]

        preds_lwr = forecast_recursive(att_full, t_orig, lib, lp_correct=False)
        preds_lp  = forecast_recursive(att_full, t_orig, lib, lp_correct=True)

        price_lwr = reconstruct_price(preds_lwr, t_orig, lp_ratio_full, logtrend_full)
        price_lp  = reconstruct_price(preds_lp,  t_orig, lp_ratio_full, logtrend_full)

        rmae_h_lwr = np.abs(np.array(preds_lwr) - actual_att) / (std_att + 1e-12)
        rmae_h_lp  = np.abs(np.array(preds_lp)  - actual_att) / (std_att + 1e-12)
        mape_h_lwr = np.abs(price_lwr - actual_price) / (actual_price + 1e-6) * 100.0
        mape_h_lp  = np.abs(price_lp  - actual_price) / (actual_price + 1e-6) * 100.0

        rmae_lwr += rmae_h_lwr;  rmae_lp += rmae_h_lp
        mape_lwr += mape_h_lwr;  mape_lp += mape_h_lp
        n_valid  += 1

        # Улучшение в att-rMAE на H=20 для отбора кейсов
        delta_h20 = float(rmae_h_lwr[19] - rmae_h_lp[19])

        results.append({
            "t_orig":       t_orig,
            "std_att":      std_att,
            "actual_att":   actual_att.tolist(),
            "actual_price": actual_price.tolist(),
            "preds_lwr":    preds_lwr,
            "preds_lp":     preds_lp,
            "price_lwr":    price_lwr.tolist(),
            "price_lp":     price_lp.tolist(),
            "rmae_lwr":     rmae_h_lwr.tolist(),
            "rmae_lp":      rmae_h_lp.tolist(),
            "mape_lwr":     mape_h_lwr.tolist(),
            "mape_lp":      mape_h_lp.tolist(),
            "delta_rmae_h20": delta_h20,
        })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1:3d}/200] elapsed={time.time()-t_start:.1f}s  "
                  f"mean rMAE LWR H=1={rmae_lwr[0]/n_valid:.4f}  LP={rmae_lp[0]/n_valid:.4f}")

    mean_rmae_lwr = rmae_lwr / n_valid;  mean_rmae_lp = rmae_lp / n_valid
    mean_mape_lwr = mape_lwr / n_valid;  mean_mape_lp = mape_lp / n_valid
    pct_rmae = (mean_rmae_lwr - mean_rmae_lp) / (mean_rmae_lwr + 1e-12) * 100.0
    pct_mape = (mean_mape_lwr - mean_mape_lp) / (mean_mape_lwr + 1e-12) * 100.0

    key_H = [0, 4, 9, 19]
    print(f"\n{'─'*72}")
    print(f"{'H':>4}  {'rMAE LWR':>10}  {'rMAE LP':>10}  {'Δ rMAE%':>8}  "
          f"{'MAPE LWR%':>10}  {'MAPE LP%':>10}  {'Δ MAPE%':>8}")
    print(f"{'─'*72}")
    for h in key_H:
        print(f"{h+1:>4}  {mean_rmae_lwr[h]:>10.4f}  {mean_rmae_lp[h]:>10.4f}  "
              f"{pct_rmae[h]:>+8.2f}%  {mean_mape_lwr[h]:>10.3f}  {mean_mape_lp[h]:>10.3f}  "
              f"{pct_mape[h]:>+8.2f}%")
    print(f"{'─'*72}")
    print(f"  n_valid={n_valid}  elapsed={time.time()-t_start:.1f}s\n")

    # Сохранение
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "per_origin.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "experiment_id": EXPERIMENT_ID, "version": VERSION,
        "ticker": TICKER, "interval": INTERVAL,
        "P_REG": P_REG, "H_MAX": H_MAX, "n_valid": n_valid,
        "K_PROJ": K_PROJ, "D_PROJ": D_PROJ, "M_LP": M_LP,
        "mean_rmae_lwr": mean_rmae_lwr.tolist(), "mean_rmae_lp": mean_rmae_lp.tolist(),
        "mean_mape_lwr": mean_mape_lwr.tolist(), "mean_mape_lp": mean_mape_lp.tolist(),
        "pct_rmae_improvement": pct_rmae.tolist(),
        "pct_mape_improvement": pct_mape.tolist(),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Рисунки ────────────────────────────────────────────────────────────
    FIG_DIR.mkdir(exist_ok=True)
    H_axis = np.arange(1, H_MAX + 1)

    # 1. rMAE(H) — att уровень
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(H_axis, mean_rmae_lwr, "b-o", ms=4, label=f"LWR (p={P_REG})")
    ax.plot(H_axis, mean_rmae_lp,  "r-s", ms=4, label=f"LWR + LP-корр.")
    ax.set_xlabel("Шаг H"); ax.set_ylabel("rMAE (att)")
    ax.set_title("att-уровень: rMAE(H)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(H_axis, mean_mape_lwr, "b-o", ms=4, label=f"LWR (p={P_REG})")
    ax.plot(H_axis, mean_mape_lp,  "r-s", ms=4, label=f"LWR + LP-корр.")
    ax.set_xlabel("Шаг H"); ax.set_ylabel("MAPE % (цена)")
    ax.set_title("Цена-уровень: MAPE(H)"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("LP-коррекция: 200 origins SBER 1d", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "96c_rmae_mape_H.png", dpi=120)
    plt.close(fig)

    # 2. % улучшения
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, data, label in [
        (axes[0], pct_rmae, "Δ% rMAE (att)"),
        (axes[1], pct_mape, "Δ% MAPE (цена)"),
    ]:
        colors = np.where(data >= 0, "green", "red")
        ax.bar(H_axis, data, color=colors, alpha=0.7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Шаг H"); ax.set_ylabel(label)
        ax.set_title(f"LP-коррекция: {label}\n(>0 = LP лучше)")
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "96c_improvement_pct.png", dpi=120)
    plt.close(fig)

    # 3. Ценовые траектории — топ-6 по улучшению и топ-3 где LP хуже
    results_sorted_best  = sorted(results, key=lambda r: -r["delta_rmae_h20"])
    results_sorted_worst = sorted(results, key=lambda r:  r["delta_rmae_h20"])

    def _price_plot(r, ax, label):
        t0 = r["t_orig"]
        H_ax = np.arange(1, H_MAX + 1)
        ax.plot(H_ax, r["actual_price"], "k-",  lw=2,   label="Реальная цена")
        ax.plot(H_ax, r["price_lwr"],    "b--", lw=1.5, label="LWR")
        ax.plot(H_ax, r["price_lp"],     "r-.", lw=1.5, label="LWR+LP")
        ax.set_title(f"{label}  t_orig={t0}\n"
                     f"H=1: LWR MAPE={r['mape_lwr'][0]:.1f}%  LP={r['mape_lp'][0]:.1f}%  |  "
                     f"H=20: LWR MAPE={r['mape_lwr'][-1]:.1f}%  LP={r['mape_lp'][-1]:.1f}%",
                     fontsize=8)
        ax.set_ylabel("SBER, руб."); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Лучшие 6 кейсов (где LP максимально помогает)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, r in enumerate(results_sorted_best[:6]):
        _price_plot(r, axes[i], f"Best #{i+1}: Δ={r['delta_rmae_h20']:+.3f}")
    axes[-1].set_xlabel("Шаг H")
    fig.suptitle("Ценовые траектории — топ-6 origins, где LP-коррекция помогает", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "96c_price_best6.png", dpi=120)
    plt.close(fig)

    # Худшие 3 кейса (где LP хуже)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, r in enumerate(results_sorted_worst[:3]):
        _price_plot(r, axes[i], f"Worst #{i+1}: Δ={r['delta_rmae_h20']:+.3f}")
        axes[i].set_xlabel("Шаг H")
    fig.suptitle("Ценовые траектории — топ-3 origins, где LP-коррекция ХУЖЕ", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "96c_price_worst3.png", dpi=120)
    plt.close(fig)

    # 4. Распределение улучшения по origins (при H=5 и H=20)
    deltas_h5  = [r["rmae_lwr"][4]  - r["rmae_lp"][4]  for r in results]
    deltas_h20 = [r["rmae_lwr"][19] - r["rmae_lp"][19] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, h_label in [(axes[0], deltas_h5, "H=5"), (axes[1], deltas_h20, "H=20")]:
        ax.hist(data, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", lw=1.5, linestyle="--")
        ax.axvline(np.mean(data), color="green", lw=1.5, label=f"mean={np.mean(data):.3f}")
        pct_pos = 100 * sum(1 for d in data if d > 0) / len(data)
        ax.set_title(f"LP улучшение rMAE {h_label}\n"
                     f"LP лучше в {pct_pos:.0f}% origins, mean Δ={np.mean(data):.4f}")
        ax.set_xlabel("Δ rMAE (LWR−LP)  >0 = LP лучше")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "96c_delta_distribution.png", dpi=120)
    plt.close(fig)

    print(f"Рисунки → {FIG_DIR}")
    # LP лучше H=1 в скольки origins
    lp_better_h1  = sum(1 for r in results if r["rmae_lp"][0]  < r["rmae_lwr"][0])
    lp_better_h20 = sum(1 for r in results if r["rmae_lp"][19] < r["rmae_lwr"][19])
    print(f"\nLP лучше (rMAE, att):")
    print(f"  H=1:  {lp_better_h1}/{n_valid} ({100*lp_better_h1/n_valid:.0f}%)")
    print(f"  H=20: {lp_better_h20}/{n_valid} ({100*lp_better_h20/n_valid:.0f}%)")


if __name__ == "__main__":
    main()
