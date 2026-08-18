"""
Честное сравнение dratio vs ratio:
  - dratio: берём готовый результат из API (уже посчитан с p_max=70)
  - ratio:  запускаем тот же sweep (p=2..70, pca_k=2..p-1) на ratio напрямую

Один origin, один прогноз — чтобы увидеть разницу в принципе.
"""

import json, sys, os
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sma.core.forecast.normalize import normalize

API = "http://10.0.2.3:8000"

def api_get(path):
    out = subprocess.check_output(["curl", "-s", f"{API}{path}"])
    return json.loads(out)


# ── 1. берём прогноз и свечи из API ─────────────────────────────────────────
forecasts = api_get("/forecasts?ticker=SBER&interval=1d&limit=1")
fid = forecasts[0]["id"]
fc  = api_get(f"/forecasts/{fid}")

result   = fc["result"]
params   = fc["params"]
origin_ts   = result["origin_ts"]
ma_window   = result["ma_window"]
val_horizon = params["val_horizon"]
horizon     = params["horizon"]
use_lwr     = params["use_lwr"]
p_max       = params["p_max"]
top_n       = params["top_n"]

print(f"Прогноз #{fid}  SBER 1d  origin={origin_ts}")
print(f"ma={ma_window}  val_h={val_horizon}  h={horizon}  p_max={p_max}  lwr={use_lwr}")

raw_candles = api_get("/candles?ticker=SBER&interval=1d")
candles = [{"begin": c["begin"], "open": c["open"], "high": c["high"],
            "low": c["low"], "close": c["close"], "volume": c["volume"]}
           for c in raw_candles]

# нормализация
df    = normalize(candles, window=ma_window)
df    = df.dropna(subset=["ma"]).reset_index(drop=True)
ratio  = df["ratio"].values
ma_arr = df["ma"].values
close  = df["close"].values

origin_k     = int(np.searchsorted(df["begin"].values, pd.Timestamp(origin_ts), side="right")) - 1
val_origin_k = origin_k - val_horizon
ma_origin    = float(ma_arr[origin_k])

print(f"origin_k={origin_k}  val_origin_k={val_origin_k}  close={close[origin_k]:.2f}")

# dratio из API (уже готов)
best_dratio = result["candidates"][0]
print(f"\nLучший dratio: p={best_dratio['p']} pca_k={best_dratio['pca_k']} "
      f"val_mape={best_dratio['mape']:.4%}")


# ── 2. sweep ratio ─────────────────────────────────────────────────────────
# Shared state для воркеров

_g_ratio = None

def _worker_init_ratio(ratio_arr):
    global _g_ratio
    _g_ratio = ratio_arr


def _eval_p_ratio(origin_k, val_origin_k, val_horizon, horizon, p, use_lwr):
    """
    Sweep pca_k=2..p-1 для одного p, ряд = ratio (не dratio).

    val_mape считается прямо в ratio-пространстве (без cumsum).
    """
    from sma.core.forecast.embedding import build_delay_matrix, last_vector
    from sma.core.forecast.la1 import _apply_pca

    ratio = _g_ratio
    if ratio is None:
        return []

    if val_origin_k < p + 5:
        return []

    total_h     = val_horizon + horizon
    n_neighbors = 3 * (p + 1)

    # история до val_origin_k включительно
    history = ratio[:val_origin_k + 1]
    try:
        X, y = build_delay_matrix(history, p)
        vec0 = last_vector(history, p).copy()
    except Exception:
        return []

    if len(X) < n_neighbors + 1:
        return []

    # полный PCA один раз
    mean_   = X.mean(axis=0)
    Xc      = X - mean_
    cov     = Xc.T @ Xc / max(len(Xc) - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    V_all   = vecs[:, ::-1].T      # (p, p)
    X_proj  = Xc @ V_all.T         # (N, p)

    actual_val = ratio[val_origin_k + 1 : val_origin_k + 1 + val_horizon]

    results = []
    for pca_k in range(2, p):
        X_cmp = X_proj[:, :pca_k]
        pca_V = V_all[:pca_k]
        vec   = vec0.copy()
        out   = np.empty(total_h)

        try:
            if use_lwr:
                for h in range(total_h):
                    q        = _apply_pca(vec, mean_, pca_V)
                    dists    = np.linalg.norm(X_cmp - q, axis=1)
                    nn_idx   = np.argpartition(dists, n_neighbors)[:n_neighbors]
                    nn_dists = dists[nn_idx]
                    h_bw     = max(float(nn_dists.max()), 1e-10)
                    weights  = np.exp(-0.5 * (nn_dists / h_bw) ** 2)
                    X_nn     = X[nn_idx]
                    A_nn     = np.hstack([np.ones((n_neighbors, 1)), X_nn])
                    w_sq     = np.sqrt(weights)
                    coeffs, _, _, _ = np.linalg.lstsq(
                        w_sq[:, None] * A_nn, w_sq * y[nn_idx], rcond=None)
                    val    = float(coeffs[0] + vec @ coeffs[1:])
                    out[h] = val
                    vec    = np.roll(vec, -1); vec[-1] = val
            else:
                for h in range(total_h):
                    q      = _apply_pca(vec, mean_, pca_V)
                    dists  = np.linalg.norm(X_cmp - q, axis=1)
                    idx    = np.argpartition(dists, n_neighbors)[:n_neighbors]
                    A      = np.hstack([np.ones((n_neighbors, 1)), X[idx]])
                    coeffs, _, _, _ = np.linalg.lstsq(A, y[idx], rcond=None)
                    val    = float(coeffs[0] + vec @ coeffs[1:])
                    out[h] = val
                    vec    = np.roll(vec, -1); vec[-1] = val
        except Exception:
            continue

        # val_mape в ratio-пространстве напрямую
        ratio_hat_val = out[:val_horizon]
        n = min(len(ratio_hat_val), len(actual_val))
        if n == 0:
            continue
        mape = float(np.mean(
            np.abs(ratio_hat_val[:n] - actual_val[:n]) / (np.abs(actual_val[:n]) + 1e-12)
        ))
        if np.isnan(mape):
            continue

        results.append((p, pca_k, mape, out.tolist()))

    return results


def run_ratio_sweep(ratio, origin_k, val_origin_k, val_horizon, horizon,
                    p_max, use_lwr, top_n):
    p_values = [p for p in range(2, p_max + 1) if val_origin_k >= p + 5]
    n_workers = max(1, (os.cpu_count() or 4) // 4)
    print(f"Ratio sweep: {len(p_values)} p-задач, {n_workers} воркеров…")

    all_results = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init_ratio,
        initargs=(ratio,),
    ) as executor:
        futures = {executor.submit(
            _eval_p_ratio, origin_k, val_origin_k, val_horizon, horizon, p, use_lwr
        ): p for p in p_values}

        done_n = 0
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    batch = f.result() or []
                    all_results.extend(batch)
                except Exception as e:
                    pass
                done_n += 1
                if done_n % 10 == 0 or done_n == len(p_values):
                    print(f"  {done_n}/{len(p_values)} p готово, "
                          f"кандидатов пока: {len(all_results)}")

    all_results.sort(key=lambda r: r[2])
    return all_results[:top_n]


if __name__ == "__main__":
    top_ratio = run_ratio_sweep(
        ratio, origin_k, val_origin_k, val_horizon, horizon,
        p_max, use_lwr, top_n
    )

    # ── вывод ───────────────────────────────────────────────────────────────
    print(f"\n{'':=<55}")
    print(f"{'Метод':<20} {'p':>4} {'pca_k':>6} {'val_mape':>10}")
    print(f"{'':=<55}")

    for i, c in enumerate(result["candidates"]):
        marker = " ← лучший" if i == 0 else ""
        print(f"dratio{marker:<14} {c['p']:>4} {c['pca_k']:>6} {c['mape']:>10.4%}")

    print(f"{'':─<55}")
    for i, (p, pca_k, mape, _) in enumerate(top_ratio):
        marker = " ← лучший" if i == 0 else ""
        print(f"ratio{marker:<15} {p:>4} {pca_k:>6} {mape:>10.4%}")

    # ── реконструкция прогнозов в цены ──────────────────────────────────────
    best_r = top_ratio[0]
    p_r, pca_k_r, mape_r, out_r = best_r
    out_r = np.array(out_r)

    # ratio → price
    price_val_ratio = out_r[:val_horizon] * ma_origin
    price_fwd_ratio = out_r[val_horizon:] * ma_origin

    # dratio → price (из API)
    dratio_best  = result["candidates"][0]
    price_val_dr = np.array(dratio_best["val_price"])
    price_fwd_dr = np.array(dratio_best["forecast_price"])

    actual_val_close = close[val_origin_k + 1 : val_origin_k + 1 + val_horizon]

    mape_val_dr = float(np.mean(np.abs(price_val_dr - actual_val_close) / actual_val_close))
    mape_val_rt = float(np.mean(np.abs(price_val_ratio - actual_val_close) / actual_val_close))

    print(f"\nval_mape в ценах:")
    print(f"  dratio (p={dratio_best['p']}, pca_k={dratio_best['pca_k']}): {mape_val_dr:.4%}")
    print(f"  ratio  (p={p_r}, pca_k={pca_k_r}):                          {mape_val_rt:.4%}")

    # ── график ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(f"LA dratio vs ratio (sweep p_max={p_max}) — SBER 1d  origin={origin_ts[:10]}", fontsize=12)

    hist_start = max(0, origin_k - 60)
    hist_x = np.arange(hist_start, origin_k + 1)
    ax.plot(hist_x, close[hist_start: origin_k + 1], color="black", lw=1.5, label="close (факт)")

    val_x = np.arange(val_origin_k + 1, val_origin_k + 1 + val_horizon)
    ax.plot(val_x, actual_val_close, "o--", color="gray", lw=1, ms=4, label="val (факт)")

    ax.plot(val_x, price_val_dr, "s:", color="steelblue", lw=1.5, ms=5,
            label=f"val dratio p={dratio_best['p']}/k={dratio_best['pca_k']} ({mape_val_dr:.3%})")
    ax.plot(val_x, price_val_ratio, "^:", color="seagreen", lw=1.5, ms=5,
            label=f"val ratio  p={p_r}/k={pca_k_r} ({mape_val_rt:.3%})")

    fwd_x = np.arange(origin_k + 1, origin_k + 1 + horizon)
    ax.plot(fwd_x, price_fwd_dr, "s--", color="steelblue", lw=1.5, ms=5,
            label="forecast dratio", alpha=0.8)
    ax.plot(fwd_x, price_fwd_ratio, "^--", color="seagreen", lw=1.5, ms=5,
            label="forecast ratio", alpha=0.8)

    ax.axvline(val_origin_k, color="gray", lw=0.8, linestyle=":")
    ax.axvline(origin_k, color="black", lw=1, linestyle="--", label="origin")
    ax.set_xlabel("bar index")
    ax.set_ylabel("price")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_file = ROOT / "research/figures/09_ratio_sweep.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=140)
    print(f"\nГрафик: {out_file}")
    plt.show()
