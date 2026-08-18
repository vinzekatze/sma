"""
DET (Determinism) из Recurrence Quantification Analysis — SBER 1d.

Алгоритм:
  1. Скользящее окно W баров → delay-матрица размерности m
  2. Матрица близости R[i,j] = 1 если dist(x_i, x_j) < ε
     ε выбирается так, чтобы RR (recurrence rate) ≈ target_rr
  3. DET = доля точек на диагоналях длиной ≥ L_min от всех recurrence-точек
     DET → 1: детерминированная система
     DET → 0: случайный процесс

Считаем для dratio (ряд LA) и ratio — сравниваем с rolling Hurst.
"""

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sma.core.forecast.normalize  import normalize
from sma.core.forecast.hurst      import rolling_hurst
from sma.core.forecast.embedding  import build_delay_matrix

DATA_FILE = ROOT / "data/candles/SBER/1d.json"

# ── параметры ────────────────────────────────────────────────────────────────
MA_WINDOW  = 1000
M_DRATIO   = 5    # embedding dim для dratio (около FNN-минимума)
M_RATIO    = 4    # embedding dim для ratio  (FNN → p=4)
TAU        = 1
WIN        = 200  # окно RQA (баров)
STEP       = 20   # шаг скольжения
TARGET_RR  = 0.10 # целевой recurrence rate для подбора ε
L_MIN      = 2    # минимальная длина диагонали для DET


# ── RQA ─────────────────────────────────────────────────────────────────────

def compute_det(series: np.ndarray, m: int, tau: int = 1,
                target_rr: float = TARGET_RR, l_min: int = L_MIN
                ) -> dict[str, float]:
    """
    Вычислить DET и RR для отрезка series.
    ε подбирается адаптивно под target_rr.
    """
    try:
        X, _ = build_delay_matrix(series, m, tau)
    except ValueError:
        return {"DET": np.nan, "RR": np.nan, "eps": np.nan}

    N = len(X)
    if N < l_min + 2:
        return {"DET": np.nan, "RR": np.nan, "eps": np.nan}

    # попарные расстояния (нижний треугольник)
    diff = X[:, None, :] - X[None, :, :]          # (N, N, m)
    D    = np.sqrt((diff ** 2).sum(axis=-1))       # (N, N)

    # ε: процентиль расстояний (исключая диагональ)
    mask   = ~np.eye(N, dtype=bool)
    d_flat = D[mask]
    eps    = np.percentile(d_flat, target_rr * 100)

    R = (D < eps) & mask                           # recurrence matrix (без диагонали)
    total_rec = int(R.sum())
    if total_rec == 0:
        return {"DET": 0.0, "RR": 0.0, "eps": float(eps)}

    RR = total_rec / (N * (N - 1))

    # DET: считаем точки на диагоналях длиной ≥ l_min
    det_points = 0
    for k in list(range(1, N)) + list(range(-N + 1, 0)):
        d = np.diag(R.astype(np.int8), k)
        if len(d) < l_min:
            continue
        # runs of 1s
        padded = np.concatenate([[0], d, [0]])
        diff_  = np.diff(padded)
        starts = np.where(diff_ == 1)[0]
        ends   = np.where(diff_ == -1)[0]
        lengths = ends - starts
        det_points += int(lengths[lengths >= l_min].sum())

    DET = det_points / total_rec

    return {"DET": float(DET), "RR": float(RR), "eps": float(eps)}


def rolling_det(series: np.ndarray, m: int, window: int, step: int,
                tau: int = 1, target_rr: float = TARGET_RR, l_min: int = L_MIN
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Скользящий DET. Возвращает (центры окон, DET-значения).
    """
    centers, dets = [], []
    for start in range(0, len(series) - window - (m - 1) * tau, step):
        seg = series[start: start + window]
        r   = compute_det(seg, m, tau, target_rr, l_min)
        centers.append(start + window // 2)
        dets.append(r["DET"])
    return np.array(centers), np.array(dets)


# ── данные ───────────────────────────────────────────────────────────────────
with open(DATA_FILE) as f:
    candles = json.load(f)

df     = normalize(candles, window=MA_WINDOW)
df     = df.dropna(subset=["ma"]).reset_index(drop=True)
ratio  = df["ratio"].values
dratio = np.diff(ratio)
dates  = df["begin"].values

print(f"Баров: {len(df)}  ({df['begin'].iloc[0].date()} … {df['begin'].iloc[-1].date()})")
print(f"Параметры: WIN={WIN}  STEP={STEP}  target_RR={TARGET_RR:.0%}  L_min={L_MIN}")
print(f"  dratio: m={M_DRATIO}  |  ratio: m={M_RATIO}")
print("Считаю rolling DET…")

# ── расчёт ───────────────────────────────────────────────────────────────────
idx_dr, det_dr = rolling_det(dratio, M_DRATIO, WIN, STEP)
idx_rt, det_rt = rolling_det(ratio,  M_RATIO,  WIN, STEP)

# rolling Hurst для сравнения (уже есть в коде)
h_idx, h_vals  = rolling_hurst(dratio, window=WIN, step=STEP)

# перевод индексов в даты (dratio смещён на 1)
def idx_to_date(idx_arr, offset=0):
    clipped = np.clip(idx_arr + offset, 0, len(dates) - 1)
    return dates[clipped.astype(int)]

dates_dr = idx_to_date(idx_dr, offset=1)   # dratio[i] → ratio[i+1]
dates_rt = idx_to_date(idx_rt)
dates_h  = idx_to_date(h_idx)

print(f"  dratio DET: {len(det_dr)} точек  median={np.nanmedian(det_dr):.3f}")
print(f"  ratio  DET: {len(det_rt)} точек  median={np.nanmedian(det_rt):.3f}")

# ── график ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(4, 1, hspace=0.45)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax2 = fig.add_subplot(gs[2], sharex=ax0)
ax3 = fig.add_subplot(gs[3], sharex=ax0)

# цена
ax0.plot(dates, df["close"].values, color="black", lw=0.8)
ax0.set_ylabel("close"); ax0.set_title("SBER 1d  (MA=1000)")
ax0.grid(alpha=0.2)

# DET dratio
ax1.plot(dates_dr, det_dr, color="steelblue", lw=1, label=f"DET dratio m={M_DRATIO}")
ax1.axhline(np.nanmedian(det_dr), color="steelblue", lw=0.8, ls="--", alpha=0.6)
ax1.set_ylabel("DET"); ax1.set_ylim(0, 1); ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

# DET ratio
ax2.plot(dates_rt, det_rt, color="seagreen", lw=1, label=f"DET ratio m={M_RATIO}")
ax2.axhline(np.nanmedian(det_rt), color="seagreen", lw=0.8, ls="--", alpha=0.6)
ax2.set_ylabel("DET"); ax2.set_ylim(0, 1); ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

# Hurst
ax3.plot(dates_h, h_vals, color="darkorange", lw=1, label="rolling Hurst (dratio)")
ax3.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
ax3.axhline(np.nanmedian(h_vals), color="darkorange", lw=0.8, ls="--", alpha=0.6)
ax3.set_ylabel("H"); ax3.legend(fontsize=8); ax3.grid(alpha=0.2)

fig.suptitle(f"RQA DET vs Hurst — SBER 1d  WIN={WIN} STEP={STEP} RR={TARGET_RR:.0%}",
             fontsize=12)

out = ROOT / "research/figures/11_det_rolling.png"
plt.savefig(out, dpi=140)
print(f"\nГрафик: {out}")

# ── корреляция DET с Hurst ───────────────────────────────────────────────────
# интерполируем на общую сетку дат
common_dates = dates_dr
h_interp = np.interp(
    [pd.Timestamp(d).timestamp() for d in common_dates],
    [pd.Timestamp(d).timestamp() for d in dates_h],
    h_vals,
)
valid = ~(np.isnan(det_dr) | np.isnan(h_interp))
if valid.sum() > 10:
    r = np.corrcoef(det_dr[valid], h_interp[valid])[0, 1]
    print(f"\nКорреляция DET(dratio) ↔ Hurst: r={r:.3f}")

rt_interp = np.interp(
    [pd.Timestamp(d).timestamp() for d in common_dates],
    [pd.Timestamp(d).timestamp() for d in dates_rt],
    det_rt,
)
valid2 = ~(np.isnan(det_dr) | np.isnan(rt_interp))
if valid2.sum() > 10:
    r2 = np.corrcoef(det_dr[valid2], rt_interp[valid2])[0, 1]
    print(f"Корреляция DET(dratio) ↔ DET(ratio): r={r2:.3f}")

plt.show()
