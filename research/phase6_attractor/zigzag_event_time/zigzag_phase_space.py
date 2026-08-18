"""
Exploratory: фазовое пространство зигзага в событийном времени.

Три визуализации × два пространства (ratio / log-ratio) × три порога (1%, 2%, 5%):
  A — фазовый портрет (z_i, y_i) где y_i = z_i − z_{i-1}
  B — return map (y_i, y_{i+1}) — качание vs следующее качание
  C — 3D вложение (z_i, z_{i-1}, z_{i-2}) → отдельный HTML (plotly)

Запуск: python research/phase6_attractor/zigzag_phase_space.py
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats

# ── данные ────────────────────────────────────────────────────────────────────

_BASE = os.path.dirname(__file__)
DATASETS = [
    ("SBER", "1d",  os.path.join(_BASE, "../../data/candles/SBER/1d.json")),
    ("SBER", "1h",  os.path.join(_BASE, "../../prototype/data/candles/SBER/1h.json")),
    ("SBER", "10m", os.path.join(_BASE, "../../prototype/data/candles/SBER/10m.json")),
]
DATASETS_SHORT = [d for d in DATASETS if d[1] != "10m"]
OUT_DIR = _BASE

THRESHOLDS    = [0.01, 0.02, 0.05]
THRESH_LABELS = ["1 %", "2 %", "5 %"]


def load_close(path: str) -> np.ndarray:
    with open(path) as f:
        candles = json.load(f)
    return np.array([c["close"] for c in candles], dtype=np.float64)


# ── logtrend ──────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n  = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t);   ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc);  cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t)
    trend[:2] = close[:2]
    return trend


# ── зигзаг ────────────────────────────────────────────────────────────────────

def find_pivots(ratio: np.ndarray, threshold: float) -> list[int]:
    """Каузальный зигзаг: возвращает индексы подтверждённых пивотов."""
    n = len(ratio)
    if n < 3 or threshold <= 0:
        return [0]
    pivots: list[int] = [0]
    direction = 0
    ext_val, ext_idx = ratio[0], 0
    for i in range(1, n):
        v = ratio[i]
        if direction == 0:
            if abs(v - ext_val) >= threshold * ext_val:
                direction = 1 if v > ext_val else -1
                ext_val, ext_idx = v, i
        elif direction == 1:
            if v > ext_val:
                ext_val, ext_idx = v, i
            elif (ext_val - v) >= threshold * ext_val:
                pivots.append(ext_idx)
                direction = -1
                ext_val, ext_idx = v, i
        else:
            if v < ext_val:
                ext_val, ext_idx = v, i
            elif (v - ext_val) >= threshold * ext_val:
                pivots.append(ext_idx)
                direction = 1
                ext_val, ext_idx = v, i
    return pivots


# ── построение векторов ───────────────────────────────────────────────────────

def pivot_series(signal: np.ndarray, pivots: list[int]) -> np.ndarray:
    """Значения сигнала в точках пивотов (z_0, z_1, ...)."""
    return signal[np.array(pivots)]


def swing_series(z: np.ndarray) -> np.ndarray:
    """y_i = z_i − z_{i-1} (знаковая амплитуда качания)."""
    return np.diff(z)


def duration_series(pivots: list[int]) -> np.ndarray:
    """Δt_i = pivots[i] − pivots[i-1] — число баров i-го качания."""
    p = np.array(pivots)
    return np.diff(p)


def find_pivots_abs(signal: np.ndarray, threshold: float) -> list[int]:
    """Каузальный зигзаг с абсолютным порогом (для сигналов вблизи нуля)."""
    n = len(signal)
    if n < 3 or threshold <= 0:
        return [0]
    pivots: list[int] = [0]
    direction = 0
    ext_val, ext_idx = signal[0], 0
    for i in range(1, n):
        v = signal[i]
        if direction == 0:
            if abs(v - ext_val) >= threshold:
                direction = 1 if v > ext_val else -1
                ext_val, ext_idx = v, i
        elif direction == 1:
            if v > ext_val:
                ext_val, ext_idx = v, i
            elif (ext_val - v) >= threshold:
                pivots.append(ext_idx)
                direction = -1
                ext_val, ext_idx = v, i
        else:
            if v < ext_val:
                ext_val, ext_idx = v, i
            elif (v - ext_val) >= threshold:
                pivots.append(ext_idx)
                direction = 1
                ext_val, ext_idx = v, i
    return pivots


# ── matplotlib A+B ────────────────────────────────────────────────────────────

def plot_ab(ratio: np.ndarray, log_ratio: np.ndarray, out_path: str,
           title_suffix: str = ""):
    n_thresh = len(THRESHOLDS)
    fig, axes = plt.subplots(4, n_thresh, figsize=(5 * n_thresh, 16))
    fig.suptitle(f"Фазовое пространство зигзага — {title_suffix}\n"
                 "Строки: A-ratio | B-ratio | A-log | B-log", fontsize=13)

    for col, (thr, tlabel) in enumerate(zip(THRESHOLDS, THRESH_LABELS)):
        for row, (sig, sname) in enumerate([(ratio, "ratio"), (log_ratio, "log-ratio")]):
            pivots = find_pivots(ratio, thr)   # пивоты всегда по ratio
            z = pivot_series(sig, pivots)
            y = swing_series(z)                # y_i = z_i − z_{i-1}

            n_pts  = len(y)
            colors = cm.plasma(np.linspace(0, 1, n_pts))

            # ── A: фазовый портрет (z_i, y_i) ─────────────────────────
            ax_a = axes[row * 2, col]
            ax_a.scatter(z[1:], y, c=colors, s=12, alpha=0.7, linewidths=0)
            # траектория
            for k in range(n_pts - 1):
                ax_a.annotate("", xy=(z[k + 2], y[k + 1]),
                              xytext=(z[k + 1], y[k]),
                              arrowprops=dict(arrowstyle="->",
                                             color=colors[k], lw=0.5, alpha=0.4))
            ax_a.axhline(0, color="gray", lw=0.5, ls="--")
            ax_a.set_title(f"A  {sname}  порог {tlabel}  ({len(pivots)} пивотов)",
                           fontsize=9)
            ax_a.set_xlabel(f"z_i ({sname})", fontsize=8)
            ax_a.set_ylabel("y_i = z_i − z_{{i−1}}", fontsize=8)

            # ── B: return map (y_i, y_{i+1}) ───────────────────────────
            ax_b = axes[row * 2 + 1, col]
            if len(y) >= 2:
                ax_b.scatter(y[:-1], y[1:], c=colors[:-1], s=12, alpha=0.7,
                             linewidths=0)
                for k in range(n_pts - 2):
                    ax_b.annotate("", xy=(y[k + 1], y[k + 2]),
                                  xytext=(y[k], y[k + 1]),
                                  arrowprops=dict(arrowstyle="->",
                                                 color=colors[k], lw=0.5, alpha=0.4))

                # корреляция амплитуд |y_i| vs |y_{i+1}|
                ay, ay1 = np.abs(y[:-1]), np.abs(y[1:])
                r_p, p_p = stats.pearsonr(ay, ay1)
                r_s, _   = stats.spearmanr(ay, ay1)
                p_str = "p<0.001" if p_p < 0.001 else f"p={p_p:.3f}"
                ax_b.text(0.04, 0.96,
                          f"r(|y|,|y'|) = {r_p:.3f} ({p_str})\nρ = {r_s:.3f}",
                          transform=ax_b.transAxes, fontsize=7.5, va="top",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="white", alpha=0.85, edgecolor="gray"))

            ax_b.axhline(0, color="gray", lw=0.5, ls="--")
            ax_b.axvline(0, color="gray", lw=0.5, ls="--")
            lim = max(np.abs(y).max() * 1.05, 1e-9)
            ax_b.plot([-lim, lim], [lim, -lim], color="red", lw=0.6,
                      ls=":", alpha=0.5, label="y'=−y")
            ax_b.set_xlim(-lim, lim)
            ax_b.set_ylim(-lim, lim)
            ax_b.set_title(f"B  {sname}  порог {tlabel}", fontsize=9)
            ax_b.set_xlabel("y_i", fontsize=8)
            ax_b.set_ylabel("y_{{i+1}}", fontsize=8)
            ax_b.legend(fontsize=7)

    # цветовая шкала = время
    sm = cm.ScalarMappable(cmap="plasma",
                           norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.4, pad=0.02)
    cbar.set_label("время (нач → кон)", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


# ── амплитудная корреляция log-log ───────────────────────────────────────────

def plot_amplitude_corr(ratio: np.ndarray, log_ratio: np.ndarray, out_path: str,
                        title_suffix: str = ""):
    """
    |y_i| vs |y_{i+1}| в log-log масштабе + OLS-прямая.
    Показывает: прогнозируема ли амплитуда следующего качания по текущему.
    """
    n_thresh = len(THRESHOLDS)
    fig, axes = plt.subplots(2, n_thresh, figsize=(5 * n_thresh, 9))
    fig.suptitle(f"|y_i| vs |y_{{i+1}}| (амплитуды последовательных качаний) — log-log\n"
                 f"{title_suffix}", fontsize=12)

    for col, (thr, tlabel) in enumerate(zip(THRESHOLDS, THRESH_LABELS)):
        for row, (sig, sname) in enumerate([(ratio, "ratio"),
                                            (log_ratio, "log-ratio")]):
            pivots = find_pivots(ratio, thr)
            z = pivot_series(sig, pivots)
            y = swing_series(z)

            ay  = np.abs(y[:-1])
            ay1 = np.abs(y[1:])
            mask = (ay > 0) & (ay1 > 0)
            ay, ay1 = ay[mask], ay1[mask]

            t_color = np.linspace(0, 1, len(ay))
            ax = axes[row, col]
            ax.scatter(ay, ay1, c=t_color, cmap="plasma", s=10, alpha=0.5,
                       linewidths=0)

            # OLS в log-log: log(ay1) = a + b·log(ay)
            lx = np.log(ay)
            ly = np.log(ay1)
            slope, intercept, r_val, p_val, _ = stats.linregress(lx, ly)
            xfit = np.array([ay.min(), ay.max()])
            yfit = np.exp(intercept) * xfit ** slope
            p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            ax.plot(xfit, yfit, color="crimson", lw=1.5,
                    label=f"slope={slope:.3f}  R²={r_val**2:.3f}\n{p_str}")

            # диагональ |y'| = |y| (идеальный повтор амплитуды)
            lim = max(ay.max(), ay1.max()) * 1.1
            ax.plot([0, lim], [0, lim], color="gray", lw=0.8, ls="--",
                    alpha=0.5, label="|y'|=|y|")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{sname}  порог {tlabel}  (n={len(ay)})", fontsize=9)
            ax.set_xlabel("|y_i|", fontsize=8)
            ax.set_ylabel("|y_{i+1}|", fontsize=8)
            ax.legend(fontsize=7.5)

    sm = cm.ScalarMappable(cmap="plasma",
                           norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.4, pad=0.02)
    cbar.set_label("время (нач → кон)", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


# ── де-тавтологизированные визуализации ──────────────────────────────────────

def plot_clean(ratio: np.ndarray, log_ratio: np.ndarray, out_path: str,
               title_suffix: str = ""):
    """
    Три де-тавтологизированных графика × 2 пространства × 3 порога.

    A-clean : (z_{i-1}, y_i)   — откуда стартовало качание vs его амплитуда
    B1-clean: (|y_i|, |y_{i+1}|) — абсолютные амплитуды + OLS
    B2-clean: (y_i, y_{i+2})   — качание vs качание того же направления (через одно)
    """
    n_thresh = len(THRESHOLDS)
    fig, axes = plt.subplots(6, n_thresh, figsize=(5 * n_thresh, 22))
    fig.suptitle(
        f"Де-тавтологизированное фазовое пространство зигзага — {title_suffix}\n"
        "A: (z_{{i-1}}, y_i)  |  B1: (|y_i|, |y_{{i+1}}|)  |  B2: (y_i, y_{{i+2}})",
        fontsize=12,
    )

    for col, (thr, tlabel) in enumerate(zip(THRESHOLDS, THRESH_LABELS)):
        for row_off, (sig, sname) in enumerate([(ratio, "ratio"),
                                                (log_ratio, "log-ratio")]):
            pivots = find_pivots(ratio, thr)
            z = pivot_series(sig, pivots)
            y = swing_series(z)
            n_pts  = len(y)
            colors = cm.plasma(np.linspace(0, 1, n_pts))

            # ── A-clean: (z_{i-1}, y_i) ────────────────────────────────
            ax_a = axes[row_off * 3, col]
            ax_a.scatter(z[:-1], y, c=colors, s=10, alpha=0.6, linewidths=0)
            ax_a.axhline(0, color="gray", lw=0.5, ls="--")

            r_p, p_p = stats.pearsonr(z[:-1], y)
            r_s, _   = stats.spearmanr(z[:-1], y)
            p_str = "p<0.001" if p_p < 0.001 else f"p={p_p:.3f}"
            ax_a.text(0.04, 0.96,
                      f"r={r_p:.3f} ({p_str})\nρ={r_s:.3f}",
                      transform=ax_a.transAxes, fontsize=7.5, va="top",
                      bbox=dict(boxstyle="round,pad=0.3",
                                facecolor="white", alpha=0.85, edgecolor="gray"))
            ax_a.set_title(f"A-clean  {sname}  {tlabel}  ({len(pivots)} пивотов)",
                           fontsize=9)
            ax_a.set_xlabel(f"z_{{i-1}} ({sname})", fontsize=8)
            ax_a.set_ylabel("y_i = z_i − z_{{i-1}}", fontsize=8)

            # ── B1-clean: (|y_i|, |y_{i+1}|) ──────────────────────────
            ax_b1 = axes[row_off * 3 + 1, col]
            ay  = np.abs(y[:-1])
            ay1 = np.abs(y[1:])
            ax_b1.scatter(ay, ay1, c=colors[:-1], s=10, alpha=0.6, linewidths=0)

            slope, intercept, r_val, p_val, _ = stats.linregress(ay, ay1)
            xfit = np.array([ay.min(), ay.max()])
            ax_b1.plot(xfit, intercept + slope * xfit, color="crimson", lw=1.5)
            r_s2, _ = stats.spearmanr(ay, ay1)
            p_str2  = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            ax_b1.text(0.04, 0.96,
                       f"r={r_val:.3f} ({p_str2})\nρ={r_s2:.3f}  slope={slope:.3f}",
                       transform=ax_b1.transAxes, fontsize=7.5, va="top",
                       bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="white", alpha=0.85, edgecolor="gray"))
            lim_b1 = ay.max() * 1.05
            ax_b1.plot([0, lim_b1], [0, lim_b1], color="gray", lw=0.8,
                       ls="--", alpha=0.5, label="|y'|=|y|")
            ax_b1.set_title(f"B1-clean  {sname}  {tlabel}", fontsize=9)
            ax_b1.set_xlabel("|y_i|", fontsize=8)
            ax_b1.set_ylabel("|y_{{i+1}}|", fontsize=8)
            ax_b1.legend(fontsize=7)

            # ── B2-clean: (y_i, y_{i+2}) ───────────────────────────────
            ax_b2 = axes[row_off * 3 + 2, col]
            if n_pts >= 3:
                y0 = y[:-2]
                y2 = y[2:]
                ax_b2.scatter(y0, y2, c=colors[:-2], s=10, alpha=0.6, linewidths=0)
                ax_b2.axhline(0, color="gray", lw=0.5, ls="--")
                ax_b2.axvline(0, color="gray", lw=0.5, ls="--")

                r_p2, p_p2 = stats.pearsonr(y0, y2)
                r_s2, _    = stats.spearmanr(y0, y2)
                p_str2 = "p<0.001" if p_p2 < 0.001 else f"p={p_p2:.3f}"
                ax_b2.text(0.04, 0.96,
                           f"r={r_p2:.3f} ({p_str2})\nρ={r_s2:.3f}",
                           transform=ax_b2.transAxes, fontsize=7.5, va="top",
                           bbox=dict(boxstyle="round,pad=0.3",
                                     facecolor="white", alpha=0.85, edgecolor="gray"))

                lim_b2 = np.abs(y).max() * 1.05
                ax_b2.plot([-lim_b2, lim_b2], [-lim_b2, lim_b2], color="gray",
                           lw=0.8, ls="--", alpha=0.5, label="y_{{i+2}}=y_i")
                ax_b2.set_xlim(-lim_b2, lim_b2)
                ax_b2.set_ylim(-lim_b2, lim_b2)

            ax_b2.set_title(f"B2-clean  {sname}  {tlabel}", fontsize=9)
            ax_b2.set_xlabel("y_i", fontsize=8)
            ax_b2.set_ylabel("y_{{i+2}}", fontsize=8)
            ax_b2.legend(fontsize=7)

    sm = cm.ScalarMappable(cmap="plasma",
                           norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.4, pad=0.02)
    cbar.set_label("время (нач → кон)", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


def plot_3d_clean(ratio: np.ndarray, log_ratio: np.ndarray, out_path: str,
                  title_suffix: str = ""):
    """
    3D вложение в амплитудном пространстве: (log|y_i|, log|y_{i-1}|, log|y_{i-2}|).
    Убирает линейную зависимость между уровнями и ценовой дрейф.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly не установлен — пропускаю 3D-clean")
        return

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "scatter3d"}] * 3] * 2,
        subplot_titles=[f"|y| ratio  {tl}" for tl in THRESH_LABELS] +
                       [f"|y| log-ratio  {tl}" for tl in THRESH_LABELS],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    for col, thr in enumerate(THRESHOLDS, start=1):
        for row, (sig, sname) in enumerate([(ratio, "ratio"),
                                            (log_ratio, "log-ratio")], start=1):
            pivots = find_pivots(ratio, thr)
            z = pivot_series(sig, pivots)
            y = swing_series(z)
            ay = np.abs(y)
            mask = ay > 0
            ay = ay[mask]
            if len(ay) < 3:
                continue
            # log-амплитудное вложение с лагом
            ly = np.log(ay)
            x0, x1, x2 = ly[2:], ly[1:-1], ly[:-2]
            t = np.linspace(0, 1, len(x0))

            fig.add_trace(
                go.Scatter3d(
                    x=x0, y=x1, z=x2,
                    mode="markers",
                    marker=dict(size=2, color=t, colorscale="Plasma",
                                showscale=(col == 3 and row == 2)),
                    name=f"{sname} {thr*100:.0f}%",
                    showlegend=False,
                ),
                row=row, col=col,
            )
            fig.update_scenes(
                dict(xaxis_title="log|y_i|",
                     yaxis_title="log|y_{{i-1}}|",
                     zaxis_title="log|y_{{i-2}}|"),
                row=row, col=col,
            )

    fig.update_layout(
        title=f"3D амплитудное вложение зигзага — {title_suffix}",
        height=900,
    )
    fig.write_html(out_path)
    print(f"  сохранено: {out_path}")


# ── продолжительность качаний ────────────────────────────────────────────────

def _annotate(ax, x, y, extra=""):
    """Pearson r + Spearman ρ на subplot."""
    if len(x) < 3:
        return
    r_p, p_p = stats.pearsonr(x, y)
    r_s, _   = stats.spearmanr(x, y)
    p_str = "p<0.001" if p_p < 0.001 else f"p={p_p:.3f}"
    txt = f"r={r_p:.3f} ({p_str})\nρ={r_s:.3f}"
    if extra:
        txt += f"\n{extra}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="gray"))


def plot_duration(ratio: np.ndarray, out_path: str, title_suffix: str = ""):
    """
    Фазовое пространство продолжительности качаний.

    Строки (× 3 порога):
      0: (Δt_i, |y_i|)                     — длина vs амплитуда
      1: (Δt_i, Δt_{i+1})                  — память продолжительности
      2: (|y_i|/Δt_i, |y_{i+1}|/Δt_{i+1}) — память «скорости» качания
      3: (Δt_i, y_{i+2})                   — длина предсказывает следующий же-направленный свинг?
    """
    n_thresh = len(THRESHOLDS)
    fig, axes = plt.subplots(4, n_thresh, figsize=(5 * n_thresh, 16))
    fig.suptitle(f"Продолжительность качаний (событийное время) — {title_suffix}",
                 fontsize=12)

    for col, (thr, tlabel) in enumerate(zip(THRESHOLDS, THRESH_LABELS)):
        pivots = find_pivots(ratio, thr)
        z  = pivot_series(ratio, pivots)
        y  = swing_series(z)
        dt = duration_series(pivots)       # len = len(y)
        n  = len(dt)
        colors = cm.plasma(np.linspace(0, 1, n))

        # 0: (Δt_i, |y_i|)
        ax = axes[0, col]
        ay = np.abs(y)
        ax.scatter(dt, ay, c=colors, s=10, alpha=0.6, linewidths=0)
        # лог-лог OLS
        mask = (dt > 0) & (ay > 0)
        if mask.sum() > 2:
            sl, ic, rv, pv, _ = stats.linregress(np.log(dt[mask]),
                                                  np.log(ay[mask]))
            xf = np.array([dt[mask].min(), dt[mask].max()])
            ax.plot(xf, np.exp(ic) * xf ** sl, color="crimson", lw=1.5)
            _annotate(ax, dt[mask], ay[mask],
                      extra=f"log-log slope={sl:.3f} R²={rv**2:.3f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"(Δt, |y|)  {tlabel}", fontsize=9)
        ax.set_xlabel("Δt_i (баров)", fontsize=8)
        ax.set_ylabel("|y_i|", fontsize=8)

        # 1: (Δt_i, Δt_{i+1}) — память длины
        ax = axes[1, col]
        if n >= 2:
            ax.scatter(dt[:-1], dt[1:], c=colors[:-1], s=10, alpha=0.6,
                       linewidths=0)
            _annotate(ax, dt[:-1], dt[1:])
            lim = dt.max() * 1.05
            ax.plot([0, lim], [0, lim], color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"(Δt_i, Δt_{{i+1}})  {tlabel}", fontsize=9)
        ax.set_xlabel("Δt_i", fontsize=8); ax.set_ylabel("Δt_{{i+1}}", fontsize=8)

        # 2: скорость = |y|/Δt — память скорости
        ax = axes[2, col]
        vel = np.abs(y) / np.maximum(dt, 1)
        if n >= 2:
            ax.scatter(vel[:-1], vel[1:], c=colors[:-1], s=10, alpha=0.6,
                       linewidths=0)
            _annotate(ax, vel[:-1], vel[1:])
            lim = vel.max() * 1.05
            ax.plot([0, lim], [0, lim], color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"(|y|/Δt_i, |y|/Δt_{{i+1}})  скорость  {tlabel}", fontsize=9)
        ax.set_xlabel("|y_i|/Δt_i", fontsize=8)
        ax.set_ylabel("|y_{{i+1}}|/Δt_{{i+1}}", fontsize=8)

        # 3: (Δt_i, y_{i+2}) — длина предсказывает следующий одно-направленный свинг?
        ax = axes[3, col]
        if n >= 3:
            ax.scatter(dt[:-2], y[2:], c=colors[:-2], s=10, alpha=0.6,
                       linewidths=0)
            _annotate(ax, dt[:-2], y[2:])
            ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(f"(Δt_i, y_{{i+2}})  {tlabel}", fontsize=9)
        ax.set_xlabel("Δt_i", fontsize=8)
        ax.set_ylabel("y_{{i+2}}", fontsize=8)

    sm = cm.ScalarMappable(cmap="plasma", norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.4, pad=0.02)
    cbar.set_label("время (нач → кон)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


# ── dratio A: dratio в тех же пивотах что и ratio ────────────────────────────

def plot_dratio_a(ratio: np.ndarray, out_path: str, title_suffix: str = ""):
    """
    dratio-значения в точках пивотов зигзага (на ratio).
    Пивоты фиксированы — те же что для ratio; смотрим: есть ли структура
    в bar-by-bar изменениях именно в моменты разворотов?

    Строки × 3 порога: A-clean, B1-clean (|·|), B2-clean (через одно).
    """
    dratio = np.diff(ratio)           # len = N-1
    n_thresh = len(THRESHOLDS)
    fig, axes = plt.subplots(3, n_thresh, figsize=(5 * n_thresh, 12))
    fig.suptitle(
        f"dratio в пивотах ratio-зигзага — {title_suffix}\n"
        "A: (d_{{i-1}}, Δd_i)  |  B1: (|Δd_i|, |Δd_{{i+1}}|)  |  B2: (Δd_i, Δd_{{i+2}})",
        fontsize=11,
    )

    for col, (thr, tlabel) in enumerate(zip(THRESHOLDS, THRESH_LABELS)):
        pivots = find_pivots(ratio, thr)
        # dratio в каждом пивоте (клип на len-1)
        idx = np.array(pivots)
        idx = np.clip(idx, 0, len(dratio) - 1)
        d  = dratio[idx]               # значение dratio в пивоте
        yd = np.diff(d)                # "качание" dratio между пивотами
        n  = len(yd)
        colors = cm.plasma(np.linspace(0, 1, n))

        # A-clean: (d_{i-1}, yd_i)
        ax = axes[0, col]
        ax.scatter(d[:-1], yd, c=colors, s=10, alpha=0.6, linewidths=0)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        _annotate(ax, d[:-1], yd)
        ax.set_title(f"A-clean dratio  {tlabel}  ({len(pivots)} пивотов)", fontsize=9)
        ax.set_xlabel("d_{{i-1}} (dratio в пивоте)", fontsize=8)
        ax.set_ylabel("Δd_i", fontsize=8)

        # B1-clean: (|yd_i|, |yd_{i+1}|)
        ax = axes[1, col]
        ayd = np.abs(yd)
        if n >= 2:
            ax.scatter(ayd[:-1], ayd[1:], c=colors[:-1], s=10, alpha=0.6,
                       linewidths=0)
            sl, ic, rv, pv, _ = stats.linregress(ayd[:-1], ayd[1:])
            xf = np.array([ayd[:-1].min(), ayd[:-1].max()])
            ax.plot(xf, ic + sl * xf, color="crimson", lw=1.5)
            _annotate(ax, ayd[:-1], ayd[1:], extra=f"slope={sl:.3f}")
            lim = ayd.max() * 1.05
            ax.plot([0, lim], [0, lim], color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"B1-clean dratio  {tlabel}", fontsize=9)
        ax.set_xlabel("|Δd_i|", fontsize=8); ax.set_ylabel("|Δd_{{i+1}}|", fontsize=8)

        # B2-clean: (yd_i, yd_{i+2})
        ax = axes[2, col]
        if n >= 3:
            ax.scatter(yd[:-2], yd[2:], c=colors[:-2], s=10, alpha=0.6,
                       linewidths=0)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.axvline(0, color="gray", lw=0.5, ls="--")
            _annotate(ax, yd[:-2], yd[2:])
            lim = np.abs(yd).max() * 1.05
            ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8,
                    ls="--", alpha=0.5)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"B2-clean dratio  {tlabel}", fontsize=9)
        ax.set_xlabel("Δd_i", fontsize=8); ax.set_ylabel("Δd_{{i+2}}", fontsize=8)

    sm = cm.ScalarMappable(cmap="plasma", norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.4, pad=0.02)
    cbar.set_label("время (нач → кон)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


# ── dratio B: зигзаг прямо на dratio ─────────────────────────────────────────

def plot_dratio_b(ratio: np.ndarray, out_path: str, title_suffix: str = ""):
    """
    Зигзаг строится прямо на dratio (абсолютный порог = k·σ(dratio)).
    Показывает: есть ли в bar-by-bar изменениях своя событийная структура?
    Пороги: 0.5σ, 1.0σ, 2.0σ.
    """
    dratio = np.diff(ratio)
    sigma  = dratio.std()
    thr_abs    = [0.5 * sigma, 1.0 * sigma, 2.0 * sigma]
    thr_labels = ["0.5σ", "1.0σ", "2.0σ"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        f"Зигзаг на dratio (абс. порог k·σ) — {title_suffix}\n"
        "A-clean: (d_{{i-1}}, yd_i)  |  B1: (|yd_i|, |yd_{{i+1}}|)  |  B2: (yd_i, yd_{{i+2}})",
        fontsize=11,
    )

    for col, (thr, tlabel) in enumerate(zip(thr_abs, thr_labels)):
        pivots = find_pivots_abs(dratio, thr)
        d  = dratio[np.array(pivots)]
        yd = np.diff(d)
        n  = len(yd)
        colors = cm.plasma(np.linspace(0, 1, max(n, 1)))

        # A-clean
        ax = axes[0, col]
        if n >= 1:
            ax.scatter(d[:-1], yd, c=colors, s=10, alpha=0.6, linewidths=0)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.axvline(0, color="gray", lw=0.5, ls="--")
            _annotate(ax, d[:-1], yd)
        ax.set_title(f"A-clean  порог {tlabel}  ({len(pivots)} пивотов)", fontsize=9)
        ax.set_xlabel("d_{{i-1}}", fontsize=8); ax.set_ylabel("Δd_i", fontsize=8)

        # B1-clean
        ax = axes[1, col]
        if n >= 2:
            ayd = np.abs(yd)
            ax.scatter(ayd[:-1], ayd[1:], c=colors[:-1], s=10, alpha=0.6,
                       linewidths=0)
            sl, ic, rv, pv, _ = stats.linregress(ayd[:-1], ayd[1:])
            xf = np.array([ayd[:-1].min(), ayd[:-1].max()])
            ax.plot(xf, ic + sl * xf, color="crimson", lw=1.5)
            _annotate(ax, ayd[:-1], ayd[1:], extra=f"slope={sl:.3f}")
            lim = ayd.max() * 1.05
            ax.plot([0, lim], [0, lim], color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"B1-clean  {tlabel}", fontsize=9)
        ax.set_xlabel("|Δd_i|", fontsize=8); ax.set_ylabel("|Δd_{{i+1}}|", fontsize=8)

        # B2-clean
        ax = axes[2, col]
        if n >= 3:
            ax.scatter(yd[:-2], yd[2:], c=colors[:-2], s=10, alpha=0.6,
                       linewidths=0)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.axvline(0, color="gray", lw=0.5, ls="--")
            _annotate(ax, yd[:-2], yd[2:])
            lim = np.abs(yd).max() * 1.05
            ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8,
                    ls="--", alpha=0.5)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"B2-clean  {tlabel}", fontsize=9)
        ax.set_xlabel("Δd_i", fontsize=8); ax.set_ylabel("Δd_{{i+2}}", fontsize=8)

    sm = cm.ScalarMappable(cmap="plasma", norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.4, pad=0.02)
    cbar.set_label("время (нач → кон)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


# ── проверка тавтологии B1: ACF |Δd| и |y| ──────────────────────────────────

def plot_tautology_check(ratio: np.ndarray, out_path: str, title_suffix: str = ""):
    """
    Проверка тавтологии dratio B1 через автокорреляцию абсолютных амплитуд.

    Для |Δd_i| (dratio) и |y_i| (ratio) считаем Pearson r(lag) для lag=1..MAX_LAG.

    Если r(lag=1) >> r(lag=2) → тавтология через общий элемент (d_i входит
    и в Δd_i и в Δd_{i+1}).
    Если все r(lag) примерно одинаковы и высокие → тавтология через постоянную
    амплитуду (все |Δd| ≈ 2σ, так как d чередует знак).
    Если r убывает плавно → настоящая память.
    """
    dratio  = np.diff(ratio)
    MAX_LAG = 8

    # строки: ratio | dratio-A | dratio-B(0.5σ) | dratio-B(1σ) | dratio-B(2σ)
    sigma = dratio.std()
    thr_b = [0.5 * sigma, 1.0 * sigma, 2.0 * sigma]
    thr_b_labels = ["dratio-B 0.5σ", "dratio-B 1.0σ", "dratio-B 2.0σ"]

    n_rows = 2 + len(thr_b)           # ratio + dratio-A + 3×dratio-B
    n_cols = len(THRESHOLDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    fig.suptitle(
        f"Проверка тавтологии B1: r(|ampl_i|, |ampl_{{i+k}}|) по лагам — {title_suffix}\n"
        "Тавтология = r(1)>>r(2) или r flat&high; Память = плавное убывание",
        fontsize=11,
    )

    def acf_bar(ax, series: np.ndarray, label: str, n_pivots: int):
        """Барчарт r(lag) + пунктир уровня значимости."""
        lags = np.arange(1, MAX_LAG + 1)
        rs   = []
        for k in lags:
            if len(series) > k + 1:
                r, _ = stats.pearsonr(series[:-k], series[k:])
            else:
                r = np.nan
            rs.append(r)
        rs = np.array(rs)

        colors_bar = ["crimson" if k == 1 else "steelblue" for k in lags]
        ax.bar(lags, rs, color=colors_bar, alpha=0.8)
        # 95% significance bound ≈ 2/√n
        sig = 2 / np.sqrt(n_pivots) if n_pivots > 0 else 0.1
        ax.axhline(sig, color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(-sig, color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(label, fontsize=8.5)
        ax.set_xlabel("lag k", fontsize=8)
        ax.set_ylabel("r", fontsize=8)
        ax.set_xticks(lags)
        ax.set_ylim(-0.2, 1.05)
        # аннотация lag=1 и lag=2
        if len(rs) >= 2 and not np.isnan(rs[0]):
            ax.text(1, rs[0] + 0.03, f"{rs[0]:.2f}", ha="center", fontsize=7,
                    color="crimson", fontweight="bold")
        if len(rs) >= 2 and not np.isnan(rs[1]):
            ax.text(2, rs[1] + 0.03, f"{rs[1]:.2f}", ha="center", fontsize=7,
                    color="steelblue")

    for col, (thr, tlabel) in enumerate(zip(THRESHOLDS, THRESH_LABELS)):
        # ── ratio |y| ──────────────────────────────────────────────────
        pivots_r = find_pivots(ratio, thr)
        z  = pivot_series(ratio, pivots_r)
        ay = np.abs(swing_series(z))
        acf_bar(axes[0, col], ay,
                f"ratio |y|  {tlabel}  ({len(pivots_r)} пивотов)", len(ay))

        # ── dratio-A |Δd| (пивоты ratio) ───────────────────────────────
        idx = np.clip(np.array(pivots_r), 0, len(dratio) - 1)
        d   = dratio[idx]
        ayd = np.abs(np.diff(d))
        acf_bar(axes[1, col], ayd,
                f"dratio-A |Δd|  {tlabel}  ({len(pivots_r)} пивотов)", len(ayd))

        # ── dratio-B по каждому порогу ──────────────────────────────────
        for row_b, (thr_b_val, tbl) in enumerate(zip(thr_b, thr_b_labels)):
            pivots_b = find_pivots_abs(dratio, thr_b_val)
            db  = dratio[np.array(pivots_b)]
            ayb = np.abs(np.diff(db))
            acf_bar(axes[2 + row_b, col], ayb,
                    f"{tbl}  ({len(pivots_b)} пивотов)", len(ayb))

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


# ── проверка velocity memory: ACF + дискретность ────────────────────────────

def plot_velocity_check(ratio: np.ndarray, out_path: str, title_suffix: str = ""):
    """
    Проверяет, не вызвана ли velocity memory дискретностью баров.

    Строки (× 3 порога):
      0: ACF velocity и |y| (лаги 1..8) — сравнение паттернов убывания
      1: распределение Δt + доля свингов с Δt=1,2,3
      2: scatter (Δt_i, velocity_i) — видна ли гиперболическая граница threshold/Δt
      3: r(v_i, v_{i+1}) по группам Δt: [1], [2–5], [>5]
    """
    MAX_LAG  = 8
    n_thresh = len(THRESHOLDS)
    fig, axes = plt.subplots(4, n_thresh, figsize=(5 * n_thresh, 16))
    fig.suptitle(
        f"Проверка velocity memory: ACF + дискретность баров — {title_suffix}\n"
        "Если r высокий только в группе Δt=1–2 → дискретность, не память",
        fontsize=11,
    )

    for col, (thr, tlabel) in enumerate(zip(THRESHOLDS, THRESH_LABELS)):
        pivots = find_pivots(ratio, thr)
        z   = pivot_series(ratio, pivots)
        y   = swing_series(z)
        dt  = duration_series(pivots)
        vel = np.abs(y) / np.maximum(dt, 1)
        ay  = np.abs(y)
        n   = len(vel)
        thr_floor = thr  # минимальная amplitude ≈ thr * ratio ≈ thr (ratio ~ 1)

        # ── 0: ACF velocity vs ACF |y| ──────────────────────────────────
        ax = axes[0, col]
        lags = np.arange(1, MAX_LAG + 1)
        rs_vel, rs_ay = [], []
        for k in lags:
            if n > k + 1:
                rs_vel.append(stats.pearsonr(vel[:-k], vel[k:])[0])
                rs_ay.append(stats.pearsonr(ay[:-k], ay[k:])[0])
            else:
                rs_vel.append(np.nan); rs_ay.append(np.nan)

        x_vel = lags - 0.2
        x_ay  = lags + 0.2
        ax.bar(x_vel, rs_vel, width=0.35, color="steelblue",  alpha=0.8, label="velocity")
        ax.bar(x_ay,  rs_ay,  width=0.35, color="darkorange", alpha=0.8, label="|y|")
        sig = 2 / np.sqrt(n)
        ax.axhline(sig,  color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0,    color="black", lw=0.5)
        ax.set_xticks(lags); ax.set_ylim(-0.15, 1.0)
        ax.legend(fontsize=7)
        ax.set_title(f"ACF  {tlabel}  ({len(pivots)} пивотов)", fontsize=9)
        ax.set_xlabel("lag k", fontsize=8); ax.set_ylabel("r", fontsize=8)
        # аннотация lag=1 и lag=2
        if not np.isnan(rs_vel[0]):
            ax.text(x_vel[0], rs_vel[0]+0.03, f"{rs_vel[0]:.2f}",
                    ha="center", fontsize=7, color="steelblue")
        if len(rs_vel) > 1 and not np.isnan(rs_vel[1]):
            ax.text(x_vel[1], rs_vel[1]+0.03, f"{rs_vel[1]:.2f}",
                    ha="center", fontsize=7, color="steelblue")

        # ── 1: распределение Δt ─────────────────────────────────────────
        ax = axes[1, col]
        max_dt_show = int(np.percentile(dt, 95))
        bins = np.arange(0.5, max_dt_show + 1.5, 1)
        ax.hist(dt, bins=bins, color="steelblue", alpha=0.7, edgecolor="white")
        for cutoff, c in [(1, "red"), (2, "orange"), (5, "green")]:
            frac = (dt <= cutoff).mean()
            ax.axvline(cutoff + 0.5, color=c, lw=1.2, ls="--",
                       label=f"Δt≤{cutoff}: {frac:.1%}")
        ax.set_xlim(0, max_dt_show + 1)
        ax.legend(fontsize=7)
        ax.set_title(f"Распред. Δt  {tlabel}", fontsize=9)
        ax.set_xlabel("Δt (баров)", fontsize=8); ax.set_ylabel("count", fontsize=8)

        # ── 2: scatter (Δt, velocity) с гиперболической границей ────────
        ax = axes[2, col]
        colors_sc = cm.plasma(np.linspace(0, 1, n))
        ax.scatter(dt, vel, c=colors_sc, s=8, alpha=0.5, linewidths=0)
        # теоретический пол: velocity_min ≈ thr / Δt
        dt_range = np.linspace(1, dt.max(), 200)
        ax.plot(dt_range, thr_floor / dt_range, color="red", lw=1.2, ls="--",
                label=f"floor = {thr_floor:.2f}/Δt")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.set_title(f"(Δt, velocity)  {tlabel}", fontsize=9)
        ax.set_xlabel("Δt (баров)", fontsize=8); ax.set_ylabel("|y|/Δt", fontsize=8)

        # ── 3: r(v_i, v_{i+1}) по группам Δt ───────────────────────────
        ax = axes[3, col]
        groups = {"Δt=1":  dt == 1,
                  "Δt=2–5": (dt >= 2) & (dt <= 5),
                  "Δt>5": dt > 5}
        bar_colors = ["red", "orange", "green"]
        bar_rs, bar_ns, bar_labels = [], [], []
        for (gname, gmask), bc in zip(groups.items(), bar_colors):
            # нужны пары: оба соседних свинга в группе
            idx = np.where(gmask)[0]
            # пары (i, i+1) где оба в группе
            pairs_i   = idx[idx < n - 1]
            pairs_i1  = pairs_i + 1
            in_both   = gmask[pairs_i] & gmask[pairs_i1]
            vi  = vel[pairs_i[in_both]]
            vi1 = vel[pairs_i1[in_both]]
            if len(vi) > 5:
                r, p = stats.pearsonr(vi, vi1)
                bar_rs.append(r)
                bar_labels.append(f"{gname}\nn={len(vi)}")
            else:
                bar_rs.append(0.0)
                bar_labels.append(f"{gname}\nn<5")
            bar_ns.append(len(vi) if len(vi) > 5 else 0)

        bars = ax.bar(range(len(bar_rs)), bar_rs,
                      color=bar_colors, alpha=0.8)
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, fontsize=7.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(2/np.sqrt(max(n,1)), color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylim(-0.2, 1.0)
        for i, (b, r) in enumerate(zip(bars, bar_rs)):
            if bar_ns[i] > 5:
                ax.text(b.get_x() + b.get_width()/2, r + 0.03,
                        f"{r:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title(f"r(v_i, v_{{i+1}}) по группам Δt  {tlabel}", fontsize=9)
        ax.set_ylabel("r", fontsize=8)

    sm = cm.ScalarMappable(cmap="plasma", norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.3, pad=0.02)
    cbar.set_label("время", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


# ── GARCH-гипотеза: velocity vs барная волатильность ─────────────────────────

def plot_garch_check(ratio: np.ndarray, out_path: str, title_suffix: str = ""):
    """
    Проверяет: не является ли velocity memory просто GARCH-эффектом.

    Строки (× 3 порога):
      0: ACF |dratio| в барном времени vs ACF velocity в событийном (lag масштаб = mean Δt)
      1: scatter (velocity_i, mean|dratio|_в_свинге) — r=?
         если r≈1: velocity ≈ прокси барной волатильности
      2: ACF residual velocity = velocity_i − β·vol_bars_i
         если ACF остатка умирает → память целиком объяснена GARCH
         если ACF остатка жива → в событийном времени есть дополнительная структура
    """
    dratio   = np.diff(ratio)
    abs_dr   = np.abs(dratio)
    MAX_LAG  = 12
    n_thresh = len(THRESHOLDS)

    fig, axes = plt.subplots(3, n_thresh, figsize=(5 * n_thresh, 12))
    fig.suptitle(
        f"GARCH-гипотеза: velocity vs барная волатильность — {title_suffix}\n"
        "Если остаточная ACF умирает → velocity = GARCH; если жива → доп. структура",
        fontsize=11,
    )

    # ACF |dratio| в барном времени (общий, не зависит от порога)
    acf_bar_lags = np.arange(1, MAX_LAG + 1)
    acf_bar_vals = []
    for k in acf_bar_lags:
        r, _ = stats.pearsonr(abs_dr[:-k], abs_dr[k:])
        acf_bar_vals.append(r)
    acf_bar_vals = np.array(acf_bar_vals)

    for col, (thr, tlabel) in enumerate(zip(THRESHOLDS, THRESH_LABELS)):
        pivots = find_pivots(ratio, thr)
        z   = pivot_series(ratio, pivots)
        y   = swing_series(z)
        dt  = duration_series(pivots)
        vel = np.abs(y) / np.maximum(dt, 1)
        n   = len(vel)

        # барная волатильность внутри каждого свинга
        vol_bars = np.array([
            abs_dr[pivots[i]:pivots[i + 1]].mean()
            if pivots[i + 1] > pivots[i] else abs_dr[pivots[i]]
            for i in range(len(pivots) - 1)
        ])

        mean_dt = dt.mean()
        sig_n   = 2 / np.sqrt(n)

        # ── 0: ACF |dratio| (bar) vs ACF velocity (event) ──────────────
        ax = axes[0, col]

        # ACF velocity в событийном времени
        lags_ev = np.arange(1, MAX_LAG + 1)
        acf_vel = []
        for k in lags_ev:
            if n > k + 1:
                r, _ = stats.pearsonr(vel[:-k], vel[k:])
            else:
                r = np.nan
            acf_vel.append(r)
        acf_vel = np.array(acf_vel)

        # масштаб оси X для барного ACF: lag_bars = k * mean_dt
        bar_lags_scaled = acf_bar_lags * mean_dt

        ax.plot(bar_lags_scaled, acf_bar_vals, "o-", color="darkorange",
                lw=1.5, ms=4, label=f"|dratio| bars (×{mean_dt:.1f}б/свинг)")
        ax.plot(lags_ev, acf_vel, "s-", color="steelblue",
                lw=1.5, ms=4, label="velocity event-time")
        ax.axhline(0,     color="black", lw=0.5)
        ax.axhline(sig_n, color="gray",  lw=0.8, ls="--", alpha=0.7)
        ax.set_title(f"ACF сравнение  {tlabel}", fontsize=9)
        ax.set_xlabel("lag (событийн. или бары×mean_Δt)", fontsize=8)
        ax.set_ylabel("r", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_ylim(-0.1, 0.8)

        # ── 1: scatter (velocity_i, vol_bars_i) ────────────────────────
        ax = axes[1, col]
        colors_sc = cm.plasma(np.linspace(0, 1, n))
        ax.scatter(vol_bars, vel, c=colors_sc, s=10, alpha=0.5, linewidths=0)

        r_vv, p_vv = stats.pearsonr(vol_bars, vel)
        sl, ic, _, _, _ = stats.linregress(vol_bars, vel)
        xf = np.array([vol_bars.min(), vol_bars.max()])
        ax.plot(xf, ic + sl * xf, color="crimson", lw=1.5)
        p_str = "p<0.001" if p_vv < 0.001 else f"p={p_vv:.3f}"
        ax.text(0.04, 0.96,
                f"r={r_vv:.3f} ({p_str})\nslope={sl:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="gray"))
        # диагональ velocity = vol_bars (ожидание если velocity ≡ bar vol)
        lim = max(vol_bars.max(), vel.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="gray", lw=0.8, ls="--",
                alpha=0.5, label="v = vol_bars")
        ax.legend(fontsize=7)
        ax.set_title(f"velocity vs mean|dratio|_свинга  {tlabel}", fontsize=9)
        ax.set_xlabel("mean|dratio| в свинге", fontsize=8)
        ax.set_ylabel("velocity = |y|/Δt", fontsize=8)

        # ── 2: ACF residual velocity ────────────────────────────────────
        ax = axes[2, col]

        # OLS: velocity = β·vol_bars + ε
        sl_r, ic_r, _, _, _ = stats.linregress(vol_bars, vel)
        resid = vel - (ic_r + sl_r * vol_bars)

        acf_res = []
        for k in lags_ev:
            if len(resid) > k + 1:
                r, _ = stats.pearsonr(resid[:-k], resid[k:])
            else:
                r = np.nan
            acf_res.append(r)
        acf_res = np.array(acf_res)

        x_v   = lags_ev - 0.2
        x_r   = lags_ev + 0.2
        ax.bar(x_v, acf_vel, width=0.35, color="steelblue",  alpha=0.8,
               label="velocity")
        ax.bar(x_r, acf_res, width=0.35, color="tomato",     alpha=0.8,
               label="residual")
        ax.axhline(0,     color="black", lw=0.5)
        ax.axhline(sig_n, color="gray",  lw=0.8, ls="--", alpha=0.7)
        ax.set_xticks(lags_ev)
        ax.set_ylim(-0.2, 0.9)
        ax.legend(fontsize=7)
        ax.set_title(f"ACF velocity vs остаток  {tlabel}", fontsize=9)
        ax.set_xlabel("lag k (событийн.)", fontsize=8)
        ax.set_ylabel("r", fontsize=8)
        # аннотация lag=1
        if not np.isnan(acf_vel[0]):
            ax.text(x_v[0], acf_vel[0]+0.03, f"{acf_vel[0]:.2f}",
                    ha="center", fontsize=7, color="steelblue")
        if not np.isnan(acf_res[0]):
            ax.text(x_r[0], acf_res[0]+0.03, f"{acf_res[0]:.2f}",
                    ha="center", fontsize=7, color="tomato")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


# ── plotly 3D ─────────────────────────────────────────────────────────────────

def plot_3d(ratio: np.ndarray, log_ratio: np.ndarray, out_path: str,
           title_suffix: str = ""):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly не установлен — пропускаю 3D (pip install plotly)")
        return

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "scatter3d"}] * 3] * 2,
        subplot_titles=[
            f"ratio  {tl}" for tl in THRESH_LABELS
        ] + [
            f"log-ratio  {tl}" for tl in THRESH_LABELS
        ],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    for col, thr in enumerate(THRESHOLDS, start=1):
        for row, (sig, sname) in enumerate([(ratio, "ratio"),
                                            (log_ratio, "log-ratio")], start=1):
            pivots = find_pivots(ratio, thr)
            z = pivot_series(sig, pivots)
            if len(z) < 3:
                continue
            x0 = z[2:]
            x1 = z[1:-1]
            x2 = z[:-2]
            t  = np.linspace(0, 1, len(x0))

            fig.add_trace(
                go.Scatter3d(
                    x=x0, y=x1, z=x2,
                    mode="lines+markers",
                    marker=dict(size=2.5, color=t, colorscale="Plasma",
                                showscale=(col == 3 and row == 2)),
                    line=dict(color=t, colorscale="Plasma", width=1.5),
                    name=f"{sname} {thr*100:.0f}%",
                    showlegend=False,
                ),
                row=row, col=col,
            )
            ax_label = f"z_i ({sname})"
            fig.update_scenes(
                dict(
                    xaxis_title="z_i",
                    yaxis_title="z_{i-1}",
                    zaxis_title="z_{i-2}",
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title=f"3D вложение зигзага (событийное время) — {title_suffix}",
        height=900,
    )
    fig.write_html(out_path)
    print(f"  сохранено: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    for ticker, interval, path in DATASETS:
        print(f"\n{'='*50}")
        print(f"Загрузка {ticker} {interval} …")
        close     = load_close(path)
        trend     = logtrend_causal(close)
        ratio     = close / trend
        log_ratio = np.log(ratio)

        print(f"  баров: {len(close)}")
        for thr, tl in zip(THRESHOLDS, THRESH_LABELS):
            p = find_pivots(ratio, thr)
            print(f"  порог {tl}: {len(p)} пивотов")

        tag = f"{ticker}_{interval}"

        print(f"  Рисую A+B …")
        plot_ab(ratio, log_ratio,
                os.path.join(OUT_DIR, f"zigzag_phase_ab_{tag}.png"),
                title_suffix=f"{ticker} {interval}")

        print(f"  Рисую амплитудную корреляцию …")
        plot_amplitude_corr(ratio, log_ratio,
                            os.path.join(OUT_DIR, f"zigzag_amplitude_corr_{tag}.png"),
                            title_suffix=f"{ticker} {interval}")

        print(f"  Рисую 3D …")
        plot_3d(ratio, log_ratio,
                os.path.join(OUT_DIR, f"zigzag_phase_3d_{tag}.html"),
                title_suffix=f"{ticker} {interval}")

        print(f"  Рисую clean (де-тавтологизированный) …")
        plot_clean(ratio, log_ratio,
                   os.path.join(OUT_DIR, f"zigzag_clean_{tag}.png"),
                   title_suffix=f"{ticker} {interval}")

        print(f"  Рисую 3D-clean (амплитудное вложение) …")
        plot_3d_clean(ratio, log_ratio,
                      os.path.join(OUT_DIR, f"zigzag_3d_clean_{tag}.html"),
                      title_suffix=f"{ticker} {interval}")

    # ── duration + dratio только для 1d и 1h ──────────────────────────────
    for ticker, interval, path in DATASETS_SHORT:
        print(f"\n{'='*50}")
        print(f"Duration + dratio: {ticker} {interval} …")
        close     = load_close(path)
        trend     = logtrend_causal(close)
        ratio     = close / trend
        tag       = f"{ticker}_{interval}"

        print(f"  Рисую duration …")
        plot_duration(ratio,
                      os.path.join(OUT_DIR, f"zigzag_duration_{tag}.png"),
                      title_suffix=f"{ticker} {interval}")

        print(f"  Рисую dratio-A …")
        plot_dratio_a(ratio,
                      os.path.join(OUT_DIR, f"zigzag_dratio_a_{tag}.png"),
                      title_suffix=f"{ticker} {interval}")

        print(f"  Рисую dratio-B …")
        plot_dratio_b(ratio,
                      os.path.join(OUT_DIR, f"zigzag_dratio_b_{tag}.png"),
                      title_suffix=f"{ticker} {interval}")

        print(f"  Рисую GARCH-проверку …")
        plot_garch_check(ratio,
                         os.path.join(OUT_DIR, f"zigzag_garch_check_{tag}.png"),
                         title_suffix=f"{ticker} {interval}")

        print(f"  Рисую проверку velocity memory …")
        plot_velocity_check(ratio,
                            os.path.join(OUT_DIR, f"zigzag_velocity_check_{tag}.png"),
                            title_suffix=f"{ticker} {interval}")

        print(f"  Рисую проверку тавтологии …")
        plot_tautology_check(ratio,
                             os.path.join(OUT_DIR, f"zigzag_tautology_check_{tag}.png"),
                             title_suffix=f"{ticker} {interval}")

    print("\nГотово.")


if __name__ == "__main__":
    main()
