"""
T03_diag.py — диагностика механики каскада для одного origin.

Цель: понять ЧТО происходит в пуле кандидатов на каждом шаге каскада —
не строить выводы о геометрии, а просто увидеть механику.

Параметры:
  TICKER  = "SBER", интервал 1d
  ORIGIN  = -50   (50 баров от конца ряда)
  P_FIT   = 9, P_MAX = 288, XI_LWR = 30
  att = diff(LP(ratio, m=9, d=3, k=30, n=3))

Вывод:
  - Таблица: на каждом шаге каскада pool_before / pool_after / pool_after_expand
  - Рисунок: временной ряд att + подсветка кандидатов по уровням
  - Рисунок: гистограмма временны́х индексов кандидатов по уровням
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ── параметры ──────────────────────────────────────────────────────────────────
TICKER   = "SBER"
ORIGIN   = -50      # отсчёт от конца ряда (включительно)
P_FIT    = 9
P_MAX    = 288
XI_LWR   = 30
LP_M     = 9
LP_D     = 3
LP_K     = 30
LP_N     = 3

DATA_DIR   = Path(__file__).resolve().parents[3] / "data" / "candles"
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "results"
FIG_DIR    = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# ── утилиты ───────────────────────────────────────────────────────────────────

def load_close(ticker: str) -> np.ndarray:
    data = json.loads((DATA_DIR / ticker / "1d.json").read_text())
    return np.array([c["close"] for c in data], dtype=float)


def logtrend_causal(close: np.ndarray) -> np.ndarray:
    N   = len(close)
    lc  = np.log(np.maximum(close, 1e-10))
    t   = np.arange(N, dtype=float)
    st  = np.cumsum(t);  st2 = np.cumsum(t * t)
    sy  = np.cumsum(lc); sty = np.cumsum(t * lc)
    n   = np.arange(1, N + 1, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = n * st2 - st ** 2
        b   = np.where(den > 0, (n * sty - st * sy) / den, 0.0)
        a   = np.where(den > 0, (sy - b * st) / n, lc)
    return np.exp(a + b * t)


def lp_smooth(signal: np.ndarray, m: int, d: int, k: int, n_iter: int = 3) -> np.ndarray:
    x    = signal.copy().astype(float)
    half = m // 2
    k_use = min(k + 1, len(x))
    for _ in range(n_iter):
        N   = len(x)
        pad = np.pad(x, (half, half), mode="edge")
        X   = np.lib.stride_tricks.sliding_window_view(pad, m)[:N].copy()
        nn  = NearestNeighbors(n_neighbors=k_use, algorithm="ball_tree")
        nn.fit(X)
        _, idx = nn.kneighbors(X)
        patches  = X[idx]
        center   = patches.mean(axis=1, keepdims=True)
        centered = patches - center
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        Vd       = Vt[:, :d, :]
        delta    = X - center[:, 0, :]
        coeff    = np.einsum("ni,ndi->nd", delta, Vd)
        proj_d   = np.einsum("nd,ndi->ni", coeff, Vd)
        x        = (center[:, 0, :] + proj_d)[:, half]
    return x


def levels_aligned(p_fit: int, p_max: int) -> list[int]:
    levs = [p_fit]; p = p_fit
    while p * 2 <= p_max:
        p *= 2; levs.append(p)
    return list(reversed(levs))


# ── каскад с трассировкой ─────────────────────────────────────────────────────

def run_cascade_trace(X_full: np.ndarray, vec_full: np.ndarray,
                      p_fit: int, p_max: int, xi_lwr: int,
                      ) -> list[dict]:
    """
    Запускает каскад и возвращает трассу на каждом шаге:
      level, p_lvl, before (size), after_cut (size + indices), after_expand (size + indices)

    Ничего не вычисляет про геометрию — только записывает какие индексы
    выжили после обрезки и после расширения.
    """
    levels  = levels_aligned(p_fit, p_max)
    xi_search = xi_lwr
    trace: list[dict] = []

    cands = np.arange(len(X_full))

    for k, p_lvl in enumerate(levels):
        is_last  = (k == len(levels) - 1)
        xi_here  = xi_lwr if is_last else xi_search
        xi_clip  = min(xi_here, len(cands))

        before_idx = cands.copy()

        if len(cands) > xi_clip:
            dists  = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full[-p_lvl:], axis=1)
            top    = np.argpartition(dists, xi_clip - 1)[:xi_clip]
            cands  = cands[top]

        after_cut_idx = cands.copy()

        if not is_last:
            p_next   = levels[k + 1]
            radius   = p_lvl - p_next
            offsets  = np.arange(radius + 1)
            expanded = cands[:, None] - offsets[None, :]
            cands    = np.unique(np.clip(expanded, 0, len(X_full) - 1))

        after_exp_idx = cands.copy() if not is_last else after_cut_idx.copy()

        trace.append({
            "level":         k,
            "p_lvl":         p_lvl,
            "is_last":       is_last,
            "n_before":      len(before_idx),
            "n_after_cut":   len(after_cut_idx),
            "n_after_expand": len(after_exp_idx),
            "idx_after_cut":  after_cut_idx,
            "idx_after_expand": after_exp_idx,
            "radius":        p_lvl - levels[k + 1] if not is_last else 0,
        })

    return trace


# ── визуализация ──────────────────────────────────────────────────────────────

LEVEL_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3", "#a65628"]


def plot_timeline(att: np.ndarray, trace: list[dict],
                  origin_idx: int, p_max: int) -> None:
    """
    Временной ряд att (последние 600 баров) с подсветкой кандидатов.
    Показывает итоговый пул после обрезки (idx_after_cut) на каждом уровне.
    Origin отмечен вертикальной линией.

    X_full строится на att[0:origin_idx], индексы в X_full → индексы в att
    через смещение p_max (т.к. первые p_max-1 баров att не входят в X_full).
    """
    WINDOW = 600
    att_show = att[max(0, origin_idx - WINDOW): origin_idx + 1]
    t_offset = max(0, origin_idx - WINDOW)

    # att-индекс точки X_full[j] = j + p_max  (X_full[j] = att[j .. j+p_max-1])
    lib_offset = p_max

    fig, axes = plt.subplots(len(trace), 1, figsize=(14, 2.5 * len(trace)),
                             sharex=True)
    if len(trace) == 1:
        axes = [axes]

    for ax, step, color in zip(axes, trace, LEVEL_COLORS):
        x_time = np.arange(len(att_show)) + t_offset
        ax.plot(x_time, att_show, color="lightgray", linewidth=0.7, zorder=1)

        idx_cut = step["idx_after_cut"]
        att_idx_cut = idx_cut + lib_offset          # индекс в att
        in_win      = (att_idx_cut >= t_offset) & (att_idx_cut < origin_idx + 1)
        ax.scatter(att_idx_cut[in_win], att[att_idx_cut[in_win]],
                   color=color, s=18, zorder=3, label="after_cut")

        idx_exp = step["idx_after_expand"]
        if not step["is_last"]:
            att_idx_exp = idx_exp + lib_offset
            in_win_e    = (att_idx_exp >= t_offset) & (att_idx_exp < origin_idx + 1)
            ax.scatter(att_idx_exp[in_win_e], att[att_idx_exp[in_win_e]],
                       color=color, s=5, alpha=0.25, zorder=2, label="after_expand")

        ax.axvline(origin_idx, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(
            f"Level {step['level']}  p={step['p_lvl']}  "
            f"before={step['n_before']}  →cut→ {step['n_after_cut']}  "
            f"→expand(±{step['radius']})→ {step['n_after_expand']}",
            fontsize=9
        )
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("bar index")
    plt.suptitle(f"{TICKER}  origin={origin_idx}  P_FIT={P_FIT} P_MAX={P_MAX}  "
                 f"xi={XI_LWR}  LP(m={LP_M},d={LP_D})", fontsize=10)
    plt.tight_layout()
    out = FIG_DIR / f"cascade_timeline_{TICKER}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  Сохранён: {out.name}")


def plot_hist(att: np.ndarray, trace: list[dict],
              origin_idx: int, p_max: int) -> None:
    """Гистограмма временны́х индексов кандидатов (after_cut) на каждом уровне."""
    lib_offset = p_max
    n_levels   = len(trace)
    fig, axes  = plt.subplots(1, n_levels, figsize=(3.5 * n_levels, 4), sharey=False)
    if n_levels == 1:
        axes = [axes]

    for ax, step, color in zip(axes, trace, LEVEL_COLORS):
        idx_cut  = step["idx_after_cut"]
        att_idx  = idx_cut + lib_offset
        bins     = np.linspace(0, origin_idx, 40)
        ax.hist(att_idx, bins=bins, color=color, alpha=0.8, edgecolor="white", lw=0.3)
        ax.axvline(origin_idx, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"p={step['p_lvl']}\n n={step['n_after_cut']}", fontsize=9)
        ax.set_xlabel("bar index", fontsize=8)

    axes[0].set_ylabel("count")
    plt.suptitle(f"{TICKER} — кандидаты (after_cut) по уровням каскада", fontsize=10)
    plt.tight_layout()
    out = FIG_DIR / f"cascade_hist_{TICKER}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  Сохранён: {out.name}")


def print_trace(trace: list[dict]) -> None:
    print()
    print(f"{'Level':>6} {'p_lvl':>6} {'before':>8} {'→cut':>6} {'radius':>7} "
          f"{'→expand':>9}  {'is_last'}")
    print("-" * 65)
    for s in trace:
        print(f"{s['level']:>6} {s['p_lvl']:>6} {s['n_before']:>8} "
              f"{s['n_after_cut']:>6} {s['radius']:>7} "
              f"{s['n_after_expand']:>9}  {'✓' if s['is_last'] else ''}")
    print()
    final = trace[-1]
    final_att = final["idx_after_cut"] + P_MAX
    print(f"Финальные {len(final['idx_after_cut'])} кандидатов (att-индексы):")
    print(sorted(final["idx_after_cut"].tolist()))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Загрузка {TICKER}...", flush=True)
    close  = load_close(TICKER)
    N      = len(close)

    # ORIGIN: нормализуем в абсолютный индекс
    origin_close = N + ORIGIN if ORIGIN < 0 else ORIGIN
    close_hist   = close[:origin_close + 1]
    N_hist       = len(close_hist)
    print(f"  Баров истории до origin: {N_hist}  (origin close idx = {origin_close})")

    print("  logtrend + LP...", flush=True)
    ratio  = close_hist / np.maximum(logtrend_causal(close_hist), 1e-10)
    att    = np.diff(lp_smooth(ratio, LP_M, LP_D, LP_K, LP_N))
    N_att  = len(att)

    # X_full: p_max-мерная матрица задержек att
    # строка j ↔ att[j .. j+P_MAX-1], целевой элемент att[j+P_MAX]
    m = N_att - P_MAX - 1
    if m < XI_LWR:
        print(f"Недостаточно данных: m={m} < xi={XI_LWR}")
        return

    t_arr  = np.arange(P_MAX, N_att - 1)
    X_full = np.column_stack([att[t_arr - (P_MAX - 1 - j)] for j in range(P_MAX)])
    # vec_full: вектор запроса (последние P_MAX значений att)
    vec_full = att[-P_MAX:]

    print(f"  X_full: {X_full.shape}   xi={XI_LWR}", flush=True)
    print("  Трассировка каскада...", flush=True)

    trace = run_cascade_trace(X_full, vec_full, P_FIT, P_MAX, XI_LWR)
    print_trace(trace)

    # att-индекс origin в att: последний элемент (len(att)-1)
    origin_in_att = N_att - 1
    print(f"\nРисунки...", flush=True)
    plot_timeline(att, trace, origin_in_att, P_MAX)
    plot_hist(att, trace, origin_in_att, P_MAX)

    print("\nГотово.", flush=True)


if __name__ == "__main__":
    main()
