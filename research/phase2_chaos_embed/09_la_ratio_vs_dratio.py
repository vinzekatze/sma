"""
Walk-forward сравнение двух режимов LA:
  A) dratio — текущий подход: LA на Δratio, реконструкция через cumsum
  B) ratio  — LA напрямую на ratio = close/MA, реконструкция ratio_hat * MA

FNN показал:
  - ratio  → аттрактор p=4
  - dratio → стохастический (FNN не сходится)

Тест: 100 точек отсчёта из хвоста SBER 1d, горизонт 5 баров.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sma.core.forecast.normalize  import normalize
from sma.core.forecast.embedding  import build_delay_matrix, last_vector

DATA_FILE = ROOT / "data/candles/SBER/1d.json"

# ── параметры теста ──────────────────────────────────────────────────────────
MA_WINDOW   = 1000
P_LIST      = [3, 4, 5, 6, 7]   # перебираем несколько p
N_NEIGHBORS = 20                 # соседей Ξ
HORIZON     = 5                  # баров вперёд
N_ORIGINS   = 150                # точек отсчёта (берём из хвоста)
WARMUP      = 300                # минимальная история до origin


# ── LA на произвольном ряду ──────────────────────────────────────────────────
def _forecast_la(series: np.ndarray, origin_k: int, p: int,
                 n_neighbors: int, horizon: int) -> np.ndarray | None:
    """
    Iterative LA forecast on `series` starting at origin_k.
    Returns array of shape (horizon,) — predicted values of `series`.
    """
    history = series[:origin_k + 1]
    if len(history) < p + n_neighbors + 5:
        return None

    try:
        X, y = build_delay_matrix(history, p)
    except ValueError:
        return None

    if len(X) < n_neighbors:
        return None

    vec = last_vector(history, p).copy()
    out = np.empty(horizon)

    for h in range(horizon):
        dists = np.linalg.norm(X - vec, axis=1)
        idx   = np.argpartition(dists, n_neighbors)[:n_neighbors]
        A     = np.hstack([np.ones((n_neighbors, 1)), X[idx]])
        coeffs, _, _, _ = np.linalg.lstsq(A, y[idx], rcond=None)
        val     = float(coeffs[0] + vec @ coeffs[1:])
        out[h]  = val
        vec     = np.roll(vec, -1)
        vec[-1] = val

    return out


# ── загрузка данных ──────────────────────────────────────────────────────────
with open(DATA_FILE) as f:
    candles = json.load(f)

df    = normalize(candles, window=MA_WINDOW)
df    = df.dropna(subset=["ma"]).reset_index(drop=True)
close = df["close"].values
ma    = df["ma"].values
ratio = df["ratio"].values
dratio = np.diff(ratio)  # len = len(ratio) - 1

print(f"Баров после warm-up MA: {len(df)}")
print(f"Точек отсчёта: {N_ORIGINS}, горизонт: {HORIZON}, MA: {MA_WINDOW}")
print(f"Перебор p: {P_LIST}, соседей: {N_NEIGHBORS}")

# точки отсчёта — равномерно из хвоста, оставляем HORIZON баров для валидации
last_valid = len(df) - HORIZON - 1
origins = np.linspace(WARMUP + max(P_LIST) + 5, last_valid, N_ORIGINS, dtype=int)

# ── walk-forward ─────────────────────────────────────────────────────────────
results = []  # list of dicts

for origin_k in origins:
    actual_close = close[origin_k + 1: origin_k + 1 + HORIZON]
    if len(actual_close) < HORIZON:
        continue

    ma_origin = float(ma[origin_k])
    ratio_origin = float(ratio[origin_k])

    for p in P_LIST:
        # ── режим A: LA на dratio ────────────────────────────────────────────
        # dratio[i] = ratio[i+1] - ratio[i], поэтому origin в dratio = origin_k-1
        dratio_origin_k = origin_k - 1
        dratio_hat = _forecast_la(dratio, dratio_origin_k, p, N_NEIGHBORS, HORIZON)

        if dratio_hat is not None:
            ratio_hat_A  = ratio_origin + np.cumsum(dratio_hat)
            price_hat_A  = ratio_hat_A * ma_origin
            mape_A = float(np.mean(np.abs(price_hat_A - actual_close) / actual_close))
        else:
            mape_A = np.nan

        # ── режим B: LA на ratio ─────────────────────────────────────────────
        ratio_hat_B = _forecast_la(ratio, origin_k, p, N_NEIGHBORS, HORIZON)

        if ratio_hat_B is not None:
            price_hat_B = ratio_hat_B * ma_origin
            mape_B = float(np.mean(np.abs(price_hat_B - actual_close) / actual_close))
        else:
            mape_B = np.nan

        results.append({"origin_k": int(origin_k), "p": p,
                        "mape_A": mape_A, "mape_B": mape_B})

df_res = pd.DataFrame(results)

# ── сводка по p ──────────────────────────────────────────────────────────────
print(f"\n{'p':>3}  {'MAPE dratio':>12}  {'MAPE ratio':>12}  {'ratio лучше?':>14}")
print("-" * 50)
for p in P_LIST:
    sub = df_res[df_res["p"] == p]
    a   = sub["mape_A"].median()
    b   = sub["mape_B"].median()
    better = "ДА" if b < a else "нет"
    print(f"{p:>3}  {a:>11.4%}  {b:>11.4%}  {better:>14}  (ratio/dratio={b/a:.2f})")

# ── общий итог ───────────────────────────────────────────────────────────────
print()
all_A = df_res["mape_A"].median()
all_B = df_res["mape_B"].median()
print(f"Медиана по всем p:  dratio={all_A:.4%}  ratio={all_B:.4%}  ratio/dratio={all_B/all_A:.3f}")

# ── распределение MAPE (лучший p по dratio и ratio) ─────────────────────────
best_p_A = df_res.groupby("p")["mape_A"].median().idxmin()
best_p_B = df_res.groupby("p")["mape_B"].median().idxmin()
print(f"Лучший p: dratio→p={best_p_A}, ratio→p={best_p_B}")

# ── графики ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"LA: dratio vs ratio  (SBER 1d, h={HORIZON}, Ξ={N_NEIGHBORS})", fontsize=13)

# левый: медиана MAPE по p
ax = axes[0]
mape_by_p_A = [df_res[df_res["p"]==p]["mape_A"].median()*100 for p in P_LIST]
mape_by_p_B = [df_res[df_res["p"]==p]["mape_B"].median()*100 for p in P_LIST]
x = np.arange(len(P_LIST))
w = 0.35
ax.bar(x - w/2, mape_by_p_A, w, label="dratio (текущий)", color="steelblue", alpha=0.8)
ax.bar(x + w/2, mape_by_p_B, w, label="ratio (новый)",    color="seagreen",  alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(P_LIST)
ax.set_xlabel("p (размерность вложения)")
ax.set_ylabel("медиана MAPE, %")
ax.set_title("Медиана MAPE по p")
ax.legend()

# правый: гистограмма MAPE для лучшего p каждого режима
ax = axes[1]
best_A = df_res[df_res["p"]==best_p_A]["mape_A"].dropna() * 100
best_B = df_res[df_res["p"]==best_p_B]["mape_B"].dropna() * 100
cap = np.percentile(pd.concat([best_A, best_B]), 97)
bins = np.linspace(0, cap, 40)
ax.hist(best_A, bins=bins, alpha=0.6, label=f"dratio p={best_p_A}", color="steelblue")
ax.hist(best_B, bins=bins, alpha=0.6, label=f"ratio  p={best_p_B}", color="seagreen")
ax.axvline(best_A.median(), color="steelblue", linestyle="--", linewidth=1.5)
ax.axvline(best_B.median(), color="seagreen",  linestyle="--", linewidth=1.5)
ax.set_xlabel("MAPE, %")
ax.set_ylabel("число прогнозов")
ax.set_title("Распределение MAPE (лучший p)")
ax.legend()

plt.tight_layout()
out = ROOT / "research/figures/09_la_ratio_vs_dratio.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=140)
print(f"\nГрафик: {out}")
plt.show()
