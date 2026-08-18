"""
97 — Параметры LP-фильтра: систематический перебор (d_proj, m) по FNN-критерию.

Цель: посмотреть, как FNN-кривая dratio_filtered меняется при росте m
для каждого d_proj. Разведка структуры, не поиск победителя.

Параметры LP:
  d_proj ∈ [2..15]           — размерность локального подпространства аттрактора
  m ∈ [2*d_proj+1 .. 2*d_proj+11]  — размерность вложения (11 значений на d_proj)
  k = 3*m                    — соседей для локальной PCA (k >> m гарантирует стаб. SVD)
  n_iter = 3                 — итерации LP (совпадает со стандартом фазы 5)

Параметры FNN (на dratio_filtered):
  D_MAX   = 20  — максимальная проверяемая размерность
  tau     = 1   — задержка
  Theiler = m   — исключает m баров вокруг точки (LP вносит авток. на ~m баров)
  r_tol   = 10.0, a_tol = 2.0  — пороги критериев Кеннела
  FNN_THR = 0.05 — порог для p_opt

Диагностика:
  var_ratio = var(dratio_filtered) / var(dratio_raw)
  Если < 0.01 — конфиг чрезмерно сглажен.

Выход:
  results/fnn_grid.csv
  figures/97_fnn_d{d:02d}.png   (14 рисунков, один на d_proj)
  figures/97_heatmap_popt.png
  figures/97_heatmap_varratio.png

Запуск:
  python 97_lp_fnn_params.py          # полный прогон
  python 97_lp_fnn_params.py --test   # 2 тикера, d=2..5, m_excess=0..5
"""

import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

ROOT    = Path(__file__).resolve().parents[3]
DATADIR = ROOT / "data" / "candles"
OUTDIR  = Path(__file__).resolve().parent
RESDIR  = OUTDIR / "results"
FIGDIR  = OUTDIR / "figures"
RESDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
TICKERS_TEST = ["SBER", "CHMF"]
INTERVAL     = "1d"

D_PROJ_RANGE   = range(2, 16)   # 2..15 включительно
M_EXCESS_RANGE = range(0, 11)   # 0..10  → m = 2*d+1+excess
N_ITER         = 3

FNN_DMAX    = 20
FNN_RTOL    = 10.0
FNN_ATOL    = 2.0
FNN_THR     = 0.05
FNN_MAX_N   = 2000   # используем последние N точек для FNN (достаточно для оценки)

TEST_MODE = "--test" in sys.argv


# ── Данные ────────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n   = len(close)
    lc  = np.log(np.maximum(close, 1e-10))
    t   = np.arange(n, dtype=np.float64)
    cn  = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    den = cn * ct2 - ct ** 2
    b   = np.where(den > 0, (cn * cty - ct * cy) / den, 0.0)
    a   = (cy - b * ct) / cn
    tr  = np.exp(a + b * t)
    tr[:2] = close[:2]
    return tr


def load_ratio(ticker: str) -> np.ndarray:
    path  = DATADIR / ticker / f"{INTERVAL}.json"
    data  = json.loads(path.read_text())
    close = np.array([c["close"] for c in data], dtype=np.float64)
    return close / logtrend_causal(close)


# ── LP-фильтр ─────────────────────────────────────────────────────────────────

def local_projective(x: np.ndarray, m: int, d: int, k: int, n_iter: int = 3) -> np.ndarray:
    """Local Projective noise reduction (Grassberger-Hegger 1993).

    Батчевая реализация: PCA через eigh на свёрнутой ковариационной матрице
    (M, m, m) вместо M отдельных SVD-вызовов (~40× быстрее).
    """
    s = x.copy().astype(np.float64)
    N = len(s)
    for _ in range(n_iter):
        M     = N - m + 1
        X     = np.lib.stride_tricks.sliding_window_view(s, m).copy()  # (M, m)
        k_eff = min(k, M - 1)
        d_eff = min(d, k_eff)

        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)
        nn_idx = np.argpartition(D, k_eff, axis=1)[:, :k_eff]  # (M, k_eff)
        del D

        # Батчевая PCA: (M, k_eff, m) → ковариация (M, m, m) → eigh
        nbrs     = X[nn_idx]                                # (M, k_eff, m)
        mu       = nbrs.mean(axis=1, keepdims=True)         # (M, 1, m)
        centered = nbrs - mu                                # (M, k_eff, m)
        del nbrs
        C        = centered.transpose(0, 2, 1) @ centered   # (M, m, m)
        del centered
        _, eigvecs = np.linalg.eigh(C)                      # (M, m, m), восх. порядок
        del C
        B = eigvecs[:, :, -d_eff:].transpose(0, 2, 1)      # (M, d_eff, m)
        del eigvecs

        # Батчевая проекция
        mu0  = mu[:, 0, :]                                  # (M, m)
        Xmu  = X - mu0
        coef = np.einsum('im,idm->id', Xmu, B)             # (M, d_eff)
        xr   = np.einsum('id,idm->im', coef, B) + mu0      # (M, m)

        # Диагональное усреднение
        x_new = np.zeros(N)
        cnt   = np.zeros(N)
        for i in range(M):
            x_new[i:i + m] += xr[i]
            cnt[i:i + m]   += 1
        s = x_new / np.maximum(cnt, 1)
    return s


# ── FNN ───────────────────────────────────────────────────────────────────────

def fnn_fracs(x: np.ndarray, theiler: int) -> np.ndarray:
    """FNN-фракции для размерностей 1..FNN_DMAX (критерии Кеннела).

    theiler: исключаем |i-j| <= theiler из поиска соседей.
    Возвращает массив длины FNN_DMAX (последний элемент всегда nan).
    """
    N   = len(x)
    sig = x.std()
    out = np.full(FNN_DMAX, np.nan)
    idx = np.arange  # alias

    for d in range(1, FNN_DMAX + 1):
        M = N - d
        if M < 30:
            break

        # Вложение размерности d (tau=1)
        X_d = np.column_stack([x[j:j + M] for j in range(d)])  # (M, d)
        Dd  = cdist(X_d, X_d)

        # Маска Теилера (векторизована)
        ii = np.arange(M)
        Dd[np.abs(ii[:, None] - ii[None, :]) <= theiler] = np.inf
        # np.fill_diagonal уже покрыт (theiler >= 0 включает диагональ)

        nn_j = Dd.argmin(axis=1)            # индекс ближайшего соседа
        r_d  = Dd[ii, nn_j]                 # расстояние в d-мерном пространстве
        valid = r_d < np.inf

        if d < FNN_DMAX and valid.any():
            x_i_next = x[d:d + M]            # (d+1)-я координата точки i
            x_j_next = x[d:d + M][nn_j]      # (d+1)-я координата соседа j

            r_d1  = np.sqrt(r_d ** 2 + (x_i_next - x_j_next) ** 2)
            safe  = np.where(r_d > 0, r_d, np.inf)
            crit1 = r_d1 / safe > FNN_RTOL
            crit2 = np.abs(x_i_next - x_j_next) / max(sig, 1e-12) > FNN_ATOL
            flag  = (crit1 | crit2) & valid
            out[d - 1] = flag[valid].mean()

    return out


# ── Обработка одного тикера ───────────────────────────────────────────────────

def process_ticker(ticker: str) -> list:
    t0 = time.time()
    print(f"[{ticker}] start", flush=True)

    ratio   = load_ratio(ticker)
    dratio  = np.diff(ratio)
    var_raw = float(dratio.var())

    d_range = list(D_PROJ_RANGE)
    m_exc_range = list(M_EXCESS_RANGE)
    if TEST_MODE:
        d_range     = d_range[:4]      # d = 2..5
        m_exc_range = m_exc_range[:6]  # excess = 0..5

    rows    = []
    n_total = len(d_range) * len(m_exc_range)
    done    = 0

    for d_proj in d_range:
        for m_exc in m_exc_range:
            m     = 2 * d_proj + 1 + m_exc
            k     = 3 * m
            try:
                ratio_f  = local_projective(ratio, m, d_proj, k, N_ITER)
                dratio_f = np.diff(ratio_f)
                var_f    = float(dratio_f.var())
                var_ratio = var_f / max(var_raw, 1e-30)

                # FNN только на последних FNN_MAX_N точках
                dr_fnn = dratio_f[-FNN_MAX_N:] if len(dratio_f) > FNN_MAX_N else dratio_f
                fnn = fnn_fracs(dr_fnn, theiler=m)

                below = np.where(fnn < FNN_THR)[0]
                p_opt = int(below[0]) + 1 if len(below) else 999

                row = {
                    "ticker":    ticker,
                    "d_proj":    d_proj,
                    "m":         m,
                    "m_excess":  m_exc,
                    "k":         k,
                    "n_iter":    N_ITER,
                    "theiler":   m,
                    "var_ratio": round(var_ratio, 6),
                    "p_opt":     p_opt,
                }
                for i, v in enumerate(fnn, 1):
                    row[f"fnn_{i:02d}"] = "" if np.isnan(v) else round(float(v), 4)
                rows.append(row)

            except Exception as e:
                print(f"[{ticker}] d={d_proj} m={m} ERROR: {e}", flush=True)

            done += 1
            if done % 10 == 0 or done == n_total:
                print(f"[{ticker}] {done}/{n_total} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[{ticker}] DONE {time.time()-t0:.1f}s", flush=True)
    return rows


# ── Фигуры ────────────────────────────────────────────────────────────────────

def plot_fnn_per_d(all_rows: list) -> None:
    """14 рисунков — по одному на каждое d_proj.
    8 панелей (тикеры), 11 кривых FNN (раскрашены по m_excess).
    """
    tickers  = sorted({r["ticker"] for r in all_rows})
    d_vals   = sorted({r["d_proj"] for r in all_rows})
    exc_vals = sorted({r["m_excess"] for r in all_rows})
    cmap     = plt.cm.plasma

    for d in d_vals:
        n_tick = len(tickers)
        ncols  = min(4, n_tick)
        nrows  = (n_tick + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
        axes_list = list(np.array(axes).flat)  # список, не итератор
        fig.suptitle(f"FNN(dratio_filtered)  d_proj={d}  [n_iter={N_ITER}, k=3m, Theiler=m]",
                     fontsize=12)

        for ax, ticker in zip(axes_list, tickers):
            rows_d = sorted(
                [r for r in all_rows if r["ticker"] == ticker and r["d_proj"] == d],
                key=lambda r: r["m_excess"]
            )
            n_exc = len(exc_vals)
            for r in rows_d:
                curve = [
                    float(r[f"fnn_{i:02d}"]) if r.get(f"fnn_{i:02d}") != "" else np.nan
                    for i in range(1, FNN_DMAX + 1)
                ]
                color = cmap(r["m_excess"] / max(n_exc - 1, 1))
                ax.plot(range(1, FNN_DMAX + 1), curve, color=color, lw=1.3,
                        alpha=0.85, label=f"+{r['m_excess']} (m={r['m']})")
            ax.axhline(FNN_THR, color="red", ls=":", lw=1.0, alpha=0.7)
            ax.set_title(ticker, fontsize=10)
            ax.set_xlabel("d_fnn", fontsize=8)
            ax.set_ylabel("FNN fraction", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(1, FNN_DMAX)
            ax.tick_params(labelsize=7)

        # Скрыть лишние оси
        for ax in axes_list[n_tick:]:
            ax.set_visible(False)

        # Легенда на последней видимой оси
        axes_list[n_tick - 1].legend(fontsize=6, loc="upper right", ncol=2)
        fig.tight_layout()
        path = FIGDIR / f"97_fnn_d{d:02d}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Saved {path.name}", flush=True)


def plot_heatmaps(all_rows: list) -> None:
    """Два heatmap: p_opt(d_proj, m_excess) и var_ratio(d_proj, m_excess), 8 тикеров."""
    tickers  = sorted({r["ticker"] for r in all_rows})
    d_vals   = sorted({r["d_proj"] for r in all_rows})
    exc_vals = sorted({r["m_excess"] for r in all_rows})
    n_tick   = len(tickers)
    ncols    = min(4, n_tick)
    nrows    = (n_tick + ncols - 1) // ncols

    for metric, title, cmap_name, fmt in [
        ("p_opt",     "p_opt — первый d_fnn < 0.05 (меньше = чище)",    "viridis_r", ".0f"),
        ("var_ratio", "var_ratio = var(dratio_filt)/var(dratio_raw)",    "magma",     ".3f"),
    ]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        fig.suptitle(f"97: {title}  [n_iter={N_ITER}]", fontsize=12)
        axes_list = list(np.array(axes).flat)

        for ax, ticker in zip(axes_list, tickers):
            Z = np.full((len(d_vals), len(exc_vals)), np.nan)
            for r in all_rows:
                if r["ticker"] != ticker:
                    continue
                di = d_vals.index(r["d_proj"])
                ei = exc_vals.index(r["m_excess"])
                v  = r.get(metric, "")
                if v != "":
                    Z[di, ei] = float(v)

            im = ax.imshow(Z, aspect="auto", cmap=cmap_name,
                           origin="lower", interpolation="nearest")
            ax.set_xticks(range(len(exc_vals)))
            ax.set_xticklabels([f"+{e}" for e in exc_vals], fontsize=7)
            ax.set_yticks(range(len(d_vals)))
            ax.set_yticklabels(d_vals, fontsize=7)
            ax.set_xlabel("m_excess", fontsize=8)
            ax.set_ylabel("d_proj", fontsize=8)
            ax.set_title(ticker, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.75, format=f"%{fmt}")

        for ax in axes_list[n_tick:]:
            ax.set_visible(False)

        fig.tight_layout()
        path = FIGDIR / f"97_heatmap_{metric}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Saved {path.name}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mode = "TEST" if TEST_MODE else "FULL"
    print(f"=== 97_lp_fnn_params v1.0 | {mode} | {time.strftime('%Y-%m-%d %H:%M:%S')} ===",
          flush=True)
    tickers   = TICKERS_TEST if TEST_MODE else TICKERS
    n_workers = min(len(tickers), max(1, (os.cpu_count() or 4) - 1))
    print(f"Тикеры: {tickers}  воркеры: {n_workers}", flush=True)
    d_cnt = len(list(D_PROJ_RANGE)[:4] if TEST_MODE else D_PROJ_RANGE)
    e_cnt = len(list(M_EXCESS_RANGE)[:6] if TEST_MODE else M_EXCESS_RANGE)
    print(f"Конфигов на тикер: {d_cnt * e_cnt}  итого: {d_cnt * e_cnt * len(tickers)}",
          flush=True)

    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(process_ticker, tickers)

    all_rows = [r for ticker_rows in results for r in ticker_rows]
    print(f"Всего строк: {len(all_rows)}  время: {time.time()-t0:.1f}s", flush=True)

    if not all_rows:
        print("Нет данных для записи.", flush=True)
        return

    csv_path = RESDIR / "fnn_grid.csv"
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"CSV: {csv_path}", flush=True)

    print("Генерация рисунков...", flush=True)
    plot_fnn_per_d(all_rows)
    plot_heatmaps(all_rows)

    print(f"=== ГОТОВО {time.time()-t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
