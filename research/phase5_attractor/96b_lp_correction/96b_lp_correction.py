"""
EXPERIMENT_ID : 96b_lp_correction
VERSION       : 1.0
ФАЗА          : 5 — исследование аттрактора

LP-коррекция рекурсивного LWR-прогноза.

После каждого шага LWR предсказанная точка проецируется на аттрактор
через Local Projective проекцию (1 итерация, k=30 соседей, d=3):

  pred_raw  = LWR(context[-p_reg:])
  v_lp      = context[-(m-1):] + [pred_raw]   # m-мерный вектор, новое → v[-1]
  pred_corr = LP_project(v_lp, library_m)[−1]  # первая координата ≡ att[t+h]

Тестируется на топ-40 «худших» origins из 96a (τ=1, p=16, ранжирование по 1-step rMAE).

Запуск:
  python 96b_lp_correction.py
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

EXPERIMENT_ID = "96b_lp_correction"
VERSION       = "1.0"

_HERE    = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "candles"
OUT_DIR  = _HERE / "results"
FIG_DIR  = _HERE / "figures"

PATH_96A_TAU1 = (
    _HERE.parent / "96_tau_delay" / "results" / "origins_tau1_full.jsonl"
)

TICKER   = "SBER"
INTERVAL = "1d"

P_REG   = 16
H_MAX   = 20
N_WORST = 40
K_PROJ  = 30
D_PROJ  = 3
M_LP    = 9
LAMBDA  = 0.01
P_MAX   = 300

LP_M, LP_D, LP_K, LP_N = 9, 3, 30, 3


# ══════════════════════════════════════════════════════════════════════════════
# LP-пайплайн (att из ratio)
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


def _lp_proj(ratio: np.ndarray, m: int, d: int, k: int, n_iter: int) -> np.ndarray:
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
    return np.diff(s)


def load_att(ticker: str) -> np.ndarray:
    raw   = json.loads((DATA_DIR / ticker / f"{INTERVAL}.json").read_text())
    close = np.array([c["close"] for c in raw], dtype=float)
    lt    = _logtrend(close)
    ratio = close / np.maximum(lt, 1e-10)
    return _lp_proj(ratio, LP_M, LP_D, LP_K, LP_N)


# ══════════════════════════════════════════════════════════════════════════════
# Библиотека (causal, до t_orig включительно)
# ══════════════════════════════════════════════════════════════════════════════

class _Lib:
    """Causal library embedding для t_orig."""
    __slots__ = ("X_full", "X_acc", "y_base", "n_lib", "X_lib_m", "ok")

    def __init__(self, att: np.ndarray) -> None:
        self.ok = False
        n = len(att)
        if n - P_MAX - 1 < 3:
            return
        att_wins   = sliding_window_view(att[:-1], P_MAX)   # att[0..n-2] windowed
        n_lib      = n - P_MAX - 1                          # = n - P_MAX - 1
        self.X_full = np.asarray(att_wins[:n_lib])          # (n_lib, P_MAX)
        # acc: Δ²att, causal
        acc_full   = np.zeros(n)
        acc_full[2:] = att[2:] - 2 * att[1:-1] + att[:-2]
        acc_wins   = sliding_window_view(acc_full[:-1], P_MAX)
        self.X_acc  = np.asarray(acc_wins[:n_lib])
        self.y_base = att[P_MAX + 1:n].copy()               # (n_lib,)
        self.n_lib  = n_lib
        # LP correction library: last M_LP columns of X_full
        self.X_lib_m = self.X_full[:, -M_LP:]               # (n_lib, M_LP)
        self.ok = True


# ══════════════════════════════════════════════════════════════════════════════
# LWR + cascade (τ=1)
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_dist(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    nA = np.linalg.norm(A, axis=1); nb = float(np.linalg.norm(b))
    if nb < 1e-12:
        return np.ones(len(A))
    cos = np.where(nA > 1e-12, (A @ b) / (nA * nb), 0.0)
    return 1.0 - np.clip(cos, -1.0, 1.0)


def _lwr_predict(X_nn: np.ndarray, y_nn: np.ndarray, q: np.ndarray) -> float:
    d = np.linalg.norm(X_nn - q, axis=1); h = max(float(d.max()), 1e-10)
    w = np.exp(-0.5 * (d / h) ** 2)
    A = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw = np.sqrt(np.maximum(w, 1e-30))
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + q @ c[1:])


def _levels_tau1(p_reg: int) -> list[int]:
    """Octave cascade levels для τ=1."""
    levs = [p_reg]; p = p_reg
    while p * 2 <= P_MAX:
        p *= 2; levs.append(p)
    return list(reversed(levs))


def _lwr_step(lib: _Lib, vec_full: np.ndarray, vec_acc: np.ndarray, p_reg: int) -> float:
    """Один шаг LWR с каскадом и acc_ang (τ=1)."""
    xi     = 3 * (p_reg + 1) + 5
    levels = _levels_tau1(p_reg)
    cands  = np.arange(lib.n_lib)

    for k_lev, p_lvl in enumerate(levels):
        if k_lev == len(levels) - 1:
            break
        xi_lvl = min(xi, len(cands))
        if len(cands) > xi_lvl:
            cols  = P_MAX - 1 - np.arange(p_lvl)  # τ=1
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
        idx   = np.argpartition(d_pos + LAMBDA * d_acc, xi - 1)[:xi]
        cands = cands[idx]

    if len(cands) < p_reg + 2:
        return np.nan

    X_nn = lib.X_full[cands][:, cols_p]
    y_nn = lib.y_base[cands]
    q    = vec_full[cols_p]
    return _lwr_predict(X_nn, y_nn, q)


# ══════════════════════════════════════════════════════════════════════════════
# LP-коррекция точки
# ══════════════════════════════════════════════════════════════════════════════

def _lp_project_point(v_pred: np.ndarray, X_lib_m: np.ndarray,
                       k: int, d: int) -> float:
    """Проецирует m-мерный вектор v_pred на локальный аттрактор.

    v_pred: shape (m,), v_pred[-1] = att_hat[t+h] (самое свежее значение).
    Возвращает скорректированное att_hat (последняя координата проекции).
    """
    dists = np.linalg.norm(X_lib_m - v_pred, axis=1)
    k_eff = min(k, len(X_lib_m) - 1)
    idx   = np.argpartition(dists, k_eff)[:k_eff]
    X_nn  = X_lib_m[idx]
    center = X_nn.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_nn - center, full_matrices=False)
    d_eff = min(d, len(Vt))
    Vd    = Vt[:d_eff].T                 # (m, d)
    v_c   = v_pred - center
    v_proj = center + Vd @ (Vd.T @ v_c)
    return float(v_proj[-1])             # последняя координата = att_hat скорр.


# ══════════════════════════════════════════════════════════════════════════════
# Рекурсивный прогноз H шагов
# ══════════════════════════════════════════════════════════════════════════════

def forecast_recursive(
    att_full: np.ndarray,
    t_orig: int,
    lib: _Lib,
    p_reg: int,
    H: int,
    lp_correct: bool,
) -> list[float]:
    """H-шаговый рекурсивный прогноз.

    Контекстный буфер начинается с P_MAX реальных значений att[t_orig-P_MAX+1..t_orig].
    После каждого шага пополняется предсказанным (или скорр.) значением.
    """
    context = list(att_full[t_orig - P_MAX + 1 : t_orig + 1])   # length P_MAX

    preds: list[float] = []

    for _ in range(H):
        # vec_full: последние P_MAX значений контекста (старое → новое)
        vec_full = np.array(context[-P_MAX:])

        # acc (Δ²) от текущего контекста
        vec_acc = np.zeros(P_MAX)
        vec_acc[2:] = vec_full[2:] - 2 * vec_full[1:-1] + vec_full[:-2]

        # LWR шаг
        pred_raw = _lwr_step(lib, vec_full, vec_acc, p_reg)
        if np.isnan(pred_raw):
            pred_raw = context[-1]  # fallback: повторить последнее

        if lp_correct:
            # m-мерный вектор: [context[-(m-1):], pred_raw], oldest→newest
            v_lp = np.array(context[-(M_LP - 1):] + [pred_raw])
            pred_final = _lp_project_point(v_lp, lib.X_lib_m, K_PROJ, D_PROJ)
        else:
            pred_final = pred_raw

        preds.append(pred_final)
        context.append(pred_final)

    return preds


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка худших origins из 96a
# ══════════════════════════════════════════════════════════════════════════════

def load_worst_origins(path: Path, n_worst: int, p_ref: int) -> list[dict]:
    """Топ-N origins по 1-step rMAE при p_ref из 96a JSONL."""
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        per_p = d["per_p"]
        p_key = str(p_ref)
        if p_key not in per_p:
            continue
        true_val = float(d["true_att"])
        std_att  = float(d["std_att"])
        pred_val = float(per_p[p_key]["pred"])
        rmae     = abs(pred_val - true_val) / std_att if std_att > 0 else 0.0
        records.append({"t_orig": int(d["t_orig"]), "rmae_1step": rmae, "std_att": std_att})

    records.sort(key=lambda r: -r["rmae_1step"])
    return records[:n_worst]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{'='*70}")
    print(f"  {EXPERIMENT_ID}  v{VERSION}")
    print(f"  P_REG={P_REG}  H={H_MAX}  N_WORST={N_WORST}")
    print(f"  LP-коррекция: k={K_PROJ}  d={D_PROJ}  m={M_LP}")
    print(f"{'='*70}\n")

    att_full = load_att(TICKER)
    print(f"att_full len={len(att_full)}")

    worst = load_worst_origins(PATH_96A_TAU1, N_WORST, P_REG)
    print(f"Худших origins: {len(worst)}")
    print(f"  worst[0]: t_orig={worst[0]['t_orig']}, 1-step rMAE={worst[0]['rmae_1step']:.4f}")
    print(f"  worst[-1]: t_orig={worst[-1]['t_orig']}, 1-step rMAE={worst[-1]['rmae_1step']:.4f}\n")

    results = []
    rmae_lwr_acc = np.zeros(H_MAX)   # накопление по origins
    rmae_lp_acc  = np.zeros(H_MAX)
    n_valid = 0

    t_start = time.time()

    for rank, rec in enumerate(worst):
        t_orig  = rec["t_orig"]
        std_att = rec["std_att"]

        # Проверяем, что есть H_MAX шагов будущего
        if t_orig + H_MAX >= len(att_full):
            print(f"  [skip] t_orig={t_orig}: недостаточно будущего")
            continue

        # Строим библиотеку (causal до t_orig)
        hist = att_full[:t_orig + 1]
        lib  = _Lib(hist)
        if not lib.ok:
            print(f"  [skip] t_orig={t_orig}: lib not ok")
            continue

        # Реальные значения H шагов вперёд
        actual = att_full[t_orig + 1 : t_orig + H_MAX + 1]

        # Рекурсивный прогноз: LWR без коррекции
        preds_lwr = forecast_recursive(att_full, t_orig, lib, P_REG, H_MAX, lp_correct=False)
        # Рекурсивный прогноз: LWR + LP-коррекция
        preds_lp  = forecast_recursive(att_full, t_orig, lib, P_REG, H_MAX, lp_correct=True)

        # rMAE по шагам
        rmae_lwr = np.abs(np.array(preds_lwr) - actual) / (std_att + 1e-12)
        rmae_lp  = np.abs(np.array(preds_lp)  - actual) / (std_att + 1e-12)

        rmae_lwr_acc += rmae_lwr
        rmae_lp_acc  += rmae_lp
        n_valid += 1

        results.append({
            "rank": rank + 1,
            "t_orig": t_orig,
            "rmae_1step_base": rec["rmae_1step"],
            "std_att": std_att,
            "actual":    actual.tolist(),
            "preds_lwr": preds_lwr,
            "preds_lp":  preds_lp,
            "rmae_lwr":  rmae_lwr.tolist(),
            "rmae_lp":   rmae_lp.tolist(),
        })

        # Прогресс каждые 10 origins
        if (rank + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  [{rank+1:3d}/{len(worst)}] elapsed={elapsed:.1f}s")

    if n_valid == 0:
        print("Нет валидных origins, выход.")
        return

    mean_lwr = rmae_lwr_acc / n_valid
    mean_lp  = rmae_lp_acc  / n_valid
    pct_diff = (mean_lwr - mean_lp) / (mean_lwr + 1e-12) * 100.0

    # Ключевые H-точки
    key_H = [0, 4, 9, 19]  # H=1,5,10,20 (0-indexed)
    print(f"\n{'─'*60}")
    print(f"{'H':>4}  {'rMAE LWR':>10}  {'rMAE LP':>10}  {'Δ%':>8}")
    print(f"{'─'*60}")
    for h in key_H:
        print(f"{h+1:>4}  {mean_lwr[h]:>10.4f}  {mean_lp[h]:>10.4f}  {pct_diff[h]:>+8.2f}%")
    print(f"{'─'*60}\n")

    # Сохраняем результаты
    OUT_DIR.mkdir(exist_ok=True)

    with open(OUT_DIR / "per_origin.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "ticker": TICKER,
        "interval": INTERVAL,
        "P_REG": P_REG,
        "H_MAX": H_MAX,
        "N_WORST": N_WORST,
        "K_PROJ": K_PROJ,
        "D_PROJ": D_PROJ,
        "M_LP": M_LP,
        "n_valid": n_valid,
        "mean_rmae_lwr": mean_lwr.tolist(),
        "mean_rmae_lp": mean_lp.tolist(),
        "pct_improvement": pct_diff.tolist(),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Результаты → {OUT_DIR}")

    # ── Рисунки ────────────────────────────────────────────────────────────
    FIG_DIR.mkdir(exist_ok=True)
    H_axis = np.arange(1, H_MAX + 1)

    # 1. rMAE(H) — среднее по 40 origins
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(H_axis, mean_lwr, "b-o", markersize=4, label=f"LWR (p={P_REG})")
    ax.plot(H_axis, mean_lp,  "r-s", markersize=4, label=f"LWR + LP-корр. (k={K_PROJ},d={D_PROJ},m={M_LP})")
    ax.set_xlabel("Шаг прогноза H")
    ax.set_ylabel("rMAE (среднее по топ-40 worst)")
    ax.set_title("LP-коррекция рекурсивного прогноза — средний rMAE(H)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "96b_rmae_by_H.png", dpi=120)
    plt.close(fig)

    # 2. Δ% по H
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = np.where(pct_diff >= 0, "green", "red")
    ax.bar(H_axis, pct_diff, color=colors, alpha=0.7)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Шаг прогноза H")
    ax.set_ylabel("% улучшения rMAE (>0 = LP лучше)")
    ax.set_title("LP-коррекция: % снижения rMAE vs LWR без коррекции")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "96b_improvement_pct.png", dpi=120)
    plt.close(fig)

    # 3. Траектории топ-5 худших origins
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=False)
    for i, r in enumerate(results[:5]):
        ax = axes[i]
        H_ax = np.arange(1, H_MAX + 1)
        ax.plot(H_ax, r["actual"],    "k-",   linewidth=2,   label="Реальный att")
        ax.plot(H_ax, r["preds_lwr"], "b--",  linewidth=1.5, label="LWR")
        ax.plot(H_ax, r["preds_lp"],  "r-.",  linewidth=1.5, label="LWR+LP")
        ax.axhline(r["actual"][0], color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(
            f"Rank #{r['rank']}  t_orig={r['t_orig']}  "
            f"1-step rMAE={r['rmae_1step_base']:.3f}"
        )
        ax.set_ylabel("att (Δratio)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Шаг H")
    fig.suptitle("Траектории прогноза — топ-5 худших origins", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "96b_trajectories_top5.png", dpi=120)
    plt.close(fig)

    print(f"Рисунки → {FIG_DIR}")
    print(f"\nВремя: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
