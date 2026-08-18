"""
Один прогноз: сравнение dratio (текущий, из API) vs ratio (новый, p=4 по FNN).

Берём последний прогноз SBER 1d из http://10.0.2.3:8000, воспроизводим
origin и запускаем LA напрямую на ряде ratio.
"""

import json, sys
from pathlib import Path

import subprocess
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sma.core.forecast.normalize import normalize
from sma.core.forecast.embedding import build_delay_matrix, last_vector

API = "http://10.0.2.3:8000"

def get(path):
    out = subprocess.check_output(["curl", "-s", f"{API}{path}"])
    return json.loads(out)

# ── 1. последний прогноз SBER 1d ────────────────────────────────────────────
forecasts = get("/forecasts?ticker=SBER&interval=1d&limit=1")

fid = forecasts[0]["id"]
print(f"Используем прогноз #{fid}  origin={forecasts[0]['origin_ts']}")

fc = get(f"/forecasts/{fid}")

result  = fc["result"]
params  = fc["params"]
origin_ts = result["origin_ts"]
ma_window = result["ma_window"]
val_horizon = params["val_horizon"]
horizon     = params["horizon"]

print(f"  ma_window={ma_window}  val_horizon={val_horizon}  horizon={horizon}")
print(f"  val_origin_ts={result['val_origin_ts']}")

# текущий лучший кандидат (dratio)
best_dratio = result["candidates"][0]
print(f"\nЛучший кандидат dratio: p={best_dratio['p']}, pca_k={best_dratio['pca_k']}, "
      f"val_mape={best_dratio['mape']:.4%}")

# ── 2. свечи ────────────────────────────────────────────────────────────────
raw_candles = get("/candles?ticker=SBER&interval=1d")

candles = [{"begin": c["begin"], "open": c["open"], "high": c["high"],
            "low": c["low"], "close": c["close"], "volume": c["volume"]}
           for c in raw_candles]

# ── 3. нормализация ──────────────────────────────────────────────────────────
import pandas as pd
df    = normalize(candles, window=ma_window)
df    = df.dropna(subset=["ma"]).reset_index(drop=True)
ratio = df["ratio"].values
ma    = df["ma"].values
close = df["close"].values

# origin_k: индекс бара origin_ts
origin_k = int(np.searchsorted(df["begin"].values, pd.Timestamp(origin_ts), side="right")) - 1
val_origin_k = origin_k - val_horizon

print(f"\norigin_k={origin_k}  val_origin_k={val_origin_k}")
print(f"close[origin_k]={close[origin_k]:.2f}  ratio[origin_k]={ratio[origin_k]:.6f}")

# ── 4. LA на ratio, p=4 (FNN) ────────────────────────────────────────────────
P_RATIO     = 4
N_NEIGHBORS = 20   # Ξ ≥ 3(p+1) → 15; берём 20

def la_forecast(series, origin_k, p, n_neighbors, horizon):
    """Iterative LA на произвольном ряду; возвращает (val_pred, fwd_pred)."""
    # val: запускаем из val_origin_k, предсказываем val_horizon шагов
    val_ok = origin_k - horizon - p - 5
    results = []
    for start_k in [val_origin_k, origin_k]:
        history = series[:start_k + 1]
        X, y = build_delay_matrix(history, p)
        vec  = last_vector(history, p).copy()
        h_n  = val_horizon if start_k == val_origin_k else horizon
        out  = np.empty(h_n)
        for h in range(h_n):
            dists  = np.linalg.norm(X - vec, axis=1)
            idx    = np.argpartition(dists, n_neighbors)[:n_neighbors]
            A      = np.hstack([np.ones((n_neighbors, 1)), X[idx]])
            coeffs, _, _, _ = np.linalg.lstsq(A, y[idx], rcond=None)
            val    = float(coeffs[0] + vec @ coeffs[1:])
            out[h] = val
            vec    = np.roll(vec, -1)
            vec[-1] = val
        results.append(out)
    return results[0], results[1]

ratio_val_pred, ratio_fwd_pred = la_forecast(ratio, origin_k, P_RATIO, N_NEIGHBORS, horizon)

# реконструкция в цены
ma_origin = float(ma[origin_k])
price_val_ratio  = ratio_val_pred  * ma_origin
price_fwd_ratio  = ratio_fwd_pred  * ma_origin

# val_mape для ratio
actual_val_close = close[val_origin_k + 1 : val_origin_k + 1 + val_horizon]
val_mape_ratio   = float(np.mean(np.abs(price_val_ratio - actual_val_close) / actual_val_close))

print(f"\nLA ratio (p={P_RATIO}, Ξ={N_NEIGHBORS}):")
print(f"  val_mape = {val_mape_ratio:.4%}")
print(f"  val_price  = {[round(v,2) for v in price_val_ratio]}")
print(f"  forecast   = {[round(v,2) for v in price_fwd_ratio]}")

print(f"\nLA dratio (p={best_dratio['p']}, pca_k={best_dratio['pca_k']}, из API):")
print(f"  val_mape   = {best_dratio['mape']:.4%}")
print(f"  val_price  = {[round(v,2) for v in best_dratio['val_price']]}")
print(f"  forecast   = {[round(v,2) for v in best_dratio['forecast_price']]}")

print(f"\nActual val_close: {[round(v,2) for v in actual_val_close]}")

# ── 5. график ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle(f"LA dratio vs ratio — SBER 1d  origin={origin_ts[:10]}", fontsize=12)

# история: 60 баров до origin
hist_start = max(0, origin_k - 60)
hist_x     = np.arange(hist_start, origin_k + 1)
ax.plot(hist_x, close[hist_start: origin_k + 1], color="black", lw=1.5, label="close (факт)")

# val-окно (факт)
val_x = np.arange(val_origin_k + 1, val_origin_k + 1 + val_horizon)
ax.plot(val_x, actual_val_close, "o--", color="gray", lw=1, ms=4, label="val (факт)")

# val-прогноз dratio
ax.plot(val_x, best_dratio["val_price"], "s:", color="steelblue", lw=1.5, ms=5,
        label=f"val dratio p={best_dratio['p']} ({best_dratio['mape']:.3%})")

# val-прогноз ratio
ax.plot(val_x, price_val_ratio, "^:", color="seagreen", lw=1.5, ms=5,
        label=f"val ratio p={P_RATIO} ({val_mape_ratio:.3%})")

# прогноз dratio
fwd_x = np.arange(origin_k + 1, origin_k + 1 + horizon)
ax.plot(fwd_x, best_dratio["forecast_price"], "s--", color="steelblue", lw=1.5, ms=5,
        label=f"forecast dratio", alpha=0.8)

# прогноз ratio
ax.plot(fwd_x, price_fwd_ratio, "^--", color="seagreen", lw=1.5, ms=5,
        label=f"forecast ratio", alpha=0.8)

# маркеры
ax.axvline(val_origin_k, color="gray", lw=0.8, linestyle=":")
ax.axvline(origin_k, color="black", lw=1, linestyle="--", label="origin")
ax.set_xlabel("bar index")
ax.set_ylabel("price")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
out = ROOT / "research/figures/09_ratio_oneshot.png"
plt.savefig(out, dpi=140)
print(f"\nГрафик: {out}")
plt.show()
