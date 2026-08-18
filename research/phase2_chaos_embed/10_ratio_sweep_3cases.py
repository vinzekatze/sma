"""
dratio vs ratio sweep — три случая: успешный / средний / фатальный прогноз.

Для каждого:
  - dratio: лучший кандидат из API (уже посчитан)
  - ratio:  запускаем тот же sweep p=2..70, отбираем top-5 по val_mape
  - считаем фактическую точность на горизонте (из доступных свечей)
"""

import json, sys, os
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sma.core.forecast.normalize import normalize

API = "http://10.0.2.3:8000"

CASES = [
    {"label": "Успешный",    "fid": 38,  "color": "green"},
    {"label": "Средний",     "fid": 30,  "color": "orange"},
    {"label": "Фатальный",   "fid": 3,   "color": "red"},
]


def api_get(path):
    out = subprocess.check_output(["curl", "-s", f"{API}{path}"])
    return json.loads(out)


# ── ratio sweep worker ───────────────────────────────────────────────────────

_g_ratio = None

def _worker_init(ratio_arr):
    global _g_ratio
    _g_ratio = ratio_arr


def _eval_p_ratio(origin_k, val_origin_k, val_horizon, horizon, p, use_lwr):
    from sma.core.forecast.embedding import build_delay_matrix, last_vector
    from sma.core.forecast.la1 import _apply_pca

    ratio = _g_ratio
    if ratio is None or val_origin_k < p + 5:
        return []

    total_h     = val_horizon + horizon
    n_neighbors = 3 * (p + 1)
    history     = ratio[:val_origin_k + 1]

    try:
        X, y = build_delay_matrix(history, p)
        vec0 = last_vector(history, p).copy()
    except Exception:
        return []

    if len(X) < n_neighbors + 1:
        return []

    mean_   = X.mean(axis=0)
    Xc      = X - mean_
    cov     = Xc.T @ Xc / max(len(Xc) - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    V_all   = vecs[:, ::-1].T
    X_proj  = Xc @ V_all.T

    actual_val = ratio[val_origin_k + 1: val_origin_k + 1 + val_horizon]

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
                    A_nn     = np.hstack([np.ones((n_neighbors, 1)), X[nn_idx]])
                    w_sq     = np.sqrt(weights)
                    coeffs, _, _, _ = np.linalg.lstsq(
                        w_sq[:, None] * A_nn, w_sq * y[nn_idx], rcond=None)
                    val = float(coeffs[0] + vec @ coeffs[1:])
                    out[h] = val; vec = np.roll(vec, -1); vec[-1] = val
            else:
                for h in range(total_h):
                    q     = _apply_pca(vec, mean_, pca_V)
                    dists = np.linalg.norm(X_cmp - q, axis=1)
                    idx   = np.argpartition(dists, n_neighbors)[:n_neighbors]
                    A     = np.hstack([np.ones((n_neighbors, 1)), X[idx]])
                    coeffs, _, _, _ = np.linalg.lstsq(A, y[idx], rcond=None)
                    val = float(coeffs[0] + vec @ coeffs[1:])
                    out[h] = val; vec = np.roll(vec, -1); vec[-1] = val
        except Exception:
            continue

        ratio_hat_val = out[:val_horizon]
        n = min(len(ratio_hat_val), len(actual_val))
        if n == 0:
            continue
        mape = float(np.mean(
            np.abs(ratio_hat_val[:n] - actual_val[:n]) / (np.abs(actual_val[:n]) + 1e-12)
        ))
        if not np.isnan(mape):
            results.append((p, pca_k, mape, out.tolist()))

    return results


def run_ratio_sweep(ratio, origin_k, val_origin_k, val_horizon, horizon,
                    p_max, use_lwr, top_n, label=""):
    p_values  = [p for p in range(2, p_max + 1) if val_origin_k >= p + 5]
    n_workers = max(1, (os.cpu_count() or 4) // 4)
    print(f"  [{label}] ratio sweep: {len(p_values)} p-задач, {n_workers} воркеров…")

    all_res = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(ratio,),
    ) as ex:
        futures = {ex.submit(
            _eval_p_ratio, origin_k, val_origin_k, val_horizon, horizon, p, use_lwr
        ) for p in p_values}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when="FIRST_COMPLETED")
            for f in done:
                try:
                    all_res.extend(f.result() or [])
                except Exception:
                    pass

    all_res.sort(key=lambda r: r[2])
    print(f"  [{label}] кандидатов: {len(all_res)}, лучший: "
          f"p={all_res[0][0]} k={all_res[0][1]} mape={all_res[0][2]:.4%}" if all_res else "нет")
    return all_res[:top_n]


# ── загрузить свечи один раз ─────────────────────────────────────────────────
print("Загружаю свечи…")
raw_candles = api_get("/candles?ticker=SBER&interval=1d")
print(f"  {len(raw_candles)} свечей  ({raw_candles[0]['begin'][:10]} … {raw_candles[-1]['begin'][:10]})")

# ── обработка каждого кейса ──────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("LA dratio vs ratio (sweep p_max=70) — SBER 1d", fontsize=13)

summary_rows = []

for ax, case in zip(axes, CASES):
    label = case["label"]
    fid   = case["fid"]
    color = case["color"]
    print(f"\n{'='*55}")
    print(f"{label}  (forecast #{fid})")

    fc      = api_get(f"/forecasts/{fid}")
    result  = fc["result"]
    params  = fc["params"]

    origin_ts   = result["origin_ts"]
    ma_window   = result["ma_window"]
    val_horizon = params["val_horizon"]
    horizon     = params["horizon"]
    use_lwr     = params["use_lwr"]
    p_max       = params["p_max"]
    top_n       = params["top_n"]

    print(f"  origin={origin_ts[:10]}  ma={ma_window}  val_h={val_horizon}  h={horizon}")

    # нормализация с нужным MA-окном
    candles = [{"begin": c["begin"], "open": c["open"], "high": c["high"],
                "low": c["low"],  "close": c["close"], "volume": c["volume"]}
               for c in raw_candles]
    df    = normalize(candles, window=ma_window)
    df    = df.dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values
    ma_arr = df["ma"].values
    close  = df["close"].values
    dates  = df["begin"].values

    origin_k     = int(np.searchsorted(dates, pd.Timestamp(origin_ts), side="right")) - 1
    val_origin_k = origin_k - val_horizon
    ma_origin    = float(ma_arr[origin_k])

    # dratio — из API
    best_dr      = result["candidates"][0]
    price_val_dr = np.array(best_dr["val_price"])
    price_fwd_dr = np.array(best_dr["forecast_price"])

    # ratio sweep
    top_ratio = run_ratio_sweep(
        ratio, origin_k, val_origin_k, val_horizon, horizon,
        p_max, use_lwr, top_n, label=label
    )

    best_r = top_ratio[0]
    out_r  = np.array(best_r[3])
    price_val_rt  = out_r[:val_horizon] * ma_origin
    price_fwd_rt  = out_r[val_horizon:] * ma_origin

    # фактические цены: val + forward
    actual_val = close[val_origin_k + 1: val_origin_k + 1 + val_horizon]
    actual_fwd = close[origin_k + 1:     origin_k + 1 + horizon]

    def mape_price(pred, actual):
        n = min(len(pred), len(actual))
        if n == 0: return float("nan")
        return float(np.mean(np.abs(pred[:n] - actual[:n]) / (np.abs(actual[:n]) + 1e-12)))

    vm_dr  = mape_price(price_val_dr, actual_val)
    vm_rt  = mape_price(price_val_rt, actual_val)
    fm_dr  = mape_price(price_fwd_dr, actual_fwd)
    fm_rt  = mape_price(price_fwd_rt, actual_fwd)

    print(f"  val_mape:    dratio={vm_dr:.4%}  ratio={vm_rt:.4%}")
    print(f"  fwd_mape:    dratio={fm_dr:.4%}  ratio={fm_rt:.4%}")

    summary_rows.append({
        "label": label, "fid": fid, "origin": origin_ts[:10],
        "dratio_p": best_dr["p"], "dratio_k": best_dr["pca_k"],
        "ratio_p":  best_r[0],   "ratio_k":  best_r[1],
        "vm_dr": vm_dr, "vm_rt": vm_rt,
        "fm_dr": fm_dr, "fm_rt": fm_rt,
    })

    # ── график ───────────────────────────────────────────────────────────────
    hist_start = max(0, origin_k - 60)
    hist_x     = np.arange(hist_start, origin_k + 1)
    val_x      = np.arange(val_origin_k + 1, val_origin_k + 1 + val_horizon)
    fwd_x      = np.arange(origin_k + 1,     origin_k + 1 + min(horizon, len(actual_fwd)))

    ax.plot(hist_x, close[hist_start: origin_k + 1], color="black", lw=1.5, label="close")
    ax.plot(val_x,  actual_val, "o--", color="gray",   lw=1, ms=3, label="val факт")
    ax.plot(fwd_x,  actual_fwd[:len(fwd_x)], "o-",
            color="black", lw=1.2, ms=3, alpha=0.5, label="fwd факт")

    ax.plot(val_x,  price_val_dr, "s:", color="steelblue", lw=1.5, ms=4,
            label=f"val dratio p={best_dr['p']}/k={best_dr['pca_k']} vm={vm_dr:.3%} fm={fm_dr:.3%}")
    ax.plot(val_x,  price_val_rt, "^:", color="seagreen",  lw=1.5, ms=4,
            label=f"val ratio  p={best_r[0]}/k={best_r[1]} vm={vm_rt:.3%} fm={fm_rt:.3%}")
    ax.plot(fwd_x,  price_fwd_dr[:len(fwd_x)], "s--", color="steelblue", lw=1.5, ms=4, alpha=0.8)
    ax.plot(fwd_x,  price_fwd_rt[:len(fwd_x)], "^--", color="seagreen",  lw=1.5, ms=4, alpha=0.8)

    ax.axvline(val_origin_k, color="gray",  lw=0.8, linestyle=":")
    ax.axvline(origin_k,     color="black", lw=1,   linestyle="--")
    ax.set_title(f"{label}  #{fid}  origin={origin_ts[:10]}  ma={ma_window}", color=color)
    ax.set_ylabel("price")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("bar index")
plt.tight_layout()
out_file = ROOT / "research/figures/10_ratio_3cases.png"
out_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_file, dpi=140)

# ── итоговая таблица ─────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"{'Кейс':<12} {'origin':>12}  {'dr val':>8} {'rt val':>8}  {'dr fwd':>8} {'rt fwd':>8}  {'fwd: ratio лучше?':>18}")
print(f"{'─'*75}")
for r in summary_rows:
    better = "ДА" if r["fm_rt"] < r["fm_dr"] else "нет"
    print(f"{r['label']:<12} {r['origin']:>12}  "
          f"{r['vm_dr']:>8.3%} {r['vm_rt']:>8.3%}  "
          f"{r['fm_dr']:>8.3%} {r['fm_rt']:>8.3%}  {better:>18}")

print(f"\nГрафик: {out_file}")
plt.show()
