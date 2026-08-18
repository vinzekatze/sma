#!/usr/bin/env python3
"""
Benchmark LA1 / LWR across parameter grid.

Outputs: scripts/benchmark_results.jsonl  (one JSON per line, appended live)
         scripts/benchmark_summary.txt     (human-readable, written at end)

Run from project root:
    /home/kali/.venvs/sma/bin/python scripts/benchmark.py
"""

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from forcaster.forecast.normalize import normalize
from forcaster.forecast.embedding import build_delay_matrix, last_vector
from forcaster.forecast.la import find_neighbors, fit_la1, apply_la1, _huber_irls
from forcaster.forecast.lwr import forecast_lwr
from forcaster.forecast.hurst import rolling_hurst
from forcaster.forecast.adaptive import hurst_percentile_threshold, regime_mask_for_pool
from forcaster.forecast.phase import pool_phase_labels, query_phase_label, PHASE_NAMES

RESULTS  = ROOT / "scripts" / "benchmark_results.jsonl"
SUMMARY  = ROOT / "scripts" / "benchmark_summary.txt"

# ── параметры сетки ────────────────────────────────────────────────────────────

TICKERS     = ["SBER", "MRKP"]
MA_WINDOWS  = [500, 1000]
P_VALS      = [3, 5, 7, 10, 13, 15, 20]
HORIZON     = 20         # шагов прогноза
N_ORIGINS   = 20         # точек старта на тикер×MA
HURST_PCT   = 30
PHASE_DN    = 0.98
PHASE_UP    = 1.02

FILTER_COMBOS = [
    dict(norm_vecs=False, use_huber=False, use_phase=False, use_regime=False, label="baseline"),
    dict(norm_vecs=True,  use_huber=False, use_phase=False, use_regime=False, label="norm"),
    dict(norm_vecs=True,  use_huber=True,  use_phase=False, use_regime=False, label="norm+huber"),
    dict(norm_vecs=True,  use_huber=False, use_phase=True,  use_regime=False, label="norm+phase"),
    dict(norm_vecs=True,  use_huber=False, use_phase=False, use_regime=True,  label="norm+regime"),
    dict(norm_vecs=True,  use_huber=True,  use_phase=True,  use_regime=True,  label="all"),
]

MODELS = ["LA1", "LWR"]


# ── вспомогательные функции ────────────────────────────────────────────────────

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted) / (np.abs(actual) + 1e-10)))

def dir_acc(actual_delta: np.ndarray, pred_delta: np.ndarray) -> float:
    return float(np.mean(np.sign(actual_delta) == np.sign(pred_delta)))

def compute_metrics(dratio_hat, dratio_actual, ratio0):
    out = {}
    for h in [5, 10, 20]:
        if h > len(dratio_hat) or h > len(dratio_actual):
            continue
        r_hat = ratio0 + np.cumsum(dratio_hat[:h])
        r_act = ratio0 + np.cumsum(dratio_actual[:h])
        out[f"mape_{h}"] = mape(r_act, r_hat)
        out[f"rmse_{h}"] = float(np.sqrt(np.mean((r_hat - r_act) ** 2)))
        out[f"dir_{h}"]  = dir_acc(dratio_actual[:h], dratio_hat[:h])
    return out


def forecast_la1_run(dratio, origin_k, p, xi, horizon, norm_vecs, use_huber, mask):
    history = dratio[:origin_k]
    X, y    = build_delay_matrix(history, p)
    vec     = last_vector(history, p).copy()
    out     = np.empty(horizon)

    if mask is not None and mask.sum() >= xi:
        Xs, ys = X[mask], y[mask]
    else:
        Xs, ys = X, y

    for h in range(horizon):
        idx    = find_neighbors(Xs, vec, xi, norm_vecs=norm_vecs)
        coeffs = fit_la1(Xs[idx], ys[idx], use_huber=use_huber)
        val    = apply_la1(vec, coeffs)
        out[h] = val
        vec    = np.roll(vec, -1)
        vec[-1] = val
    return out


def get_neighbors_step1(dratio, origin_k, p, xi, norm_vecs, mask):
    """Return global delay-matrix indices of first-step neighbours."""
    history = dratio[:origin_k]
    X, y    = build_delay_matrix(history, p)
    vec     = last_vector(history, p).copy()

    pool_idx = np.arange(len(X))
    if mask is not None and mask.sum() >= xi:
        pool_idx = pool_idx[mask]

    Xs = X[pool_idx]
    local_nn = find_neighbors(Xs, vec, xi, norm_vecs=norm_vecs)
    return pool_idx[local_nn]


def build_mask(ratio, dratio, origin_k, p, xi, bar_hurst, hurst_thr,
               use_regime, use_phase):
    mask = None
    if use_regime:
        pool_size = origin_k - p
        if pool_size >= 1:
            rm = regime_mask_for_pool(bar_hurst, pool_size, p, hurst_thr)
            mask = rm
    if use_phase:
        pm = (pool_phase_labels(ratio, dratio[:origin_k], p, PHASE_DN, PHASE_UP)
              == query_phase_label(ratio, dratio[:origin_k], origin_k, PHASE_DN, PHASE_UP))
        mask = (mask & pm) if mask is not None else pm
    return mask


# ── основная функция ───────────────────────────────────────────────────────────

def run():
    RESULTS.write_text("")   # очистить файл
    all_rows = []
    run_id = 0
    t_global = time.time()

    for ticker in TICKERS:
        data_path = ROOT / "data" / "candles" / ticker / "1h.json"
        if not data_path.exists():
            print(f"[skip] нет данных: {data_path}")
            continue
        candles = json.loads(data_path.read_text())

        for ma_w in MA_WINDOWS:
            norm = normalize(candles, window=ma_w)
            valid = norm.dropna(subset=["ma"]).reset_index(drop=True)
            ratio  = valid["ratio"].values
            dratio = np.diff(ratio)
            dates  = valid["begin"].values

            print(f"\n=== {ticker}  MA={ma_w}  N={len(valid):,} ===")

            # Hurst для режимного фильтра
            _, h_vals = rolling_hurst(dratio, window=200, step=20)
            valid_h   = h_vals[~np.isnan(h_vals)]
            hurst_thr = hurst_percentile_threshold(valid_h, HURST_PCT) if len(valid_h) else 0.60
            bar_hurst = np.full(len(valid), np.nan)
            from forcaster.forecast.hurst import rolling_hurst as _rh
            h_idx, h_v = _rh(dratio, window=200, step=20)
            for ci, hv in zip(h_idx, h_v):
                if not np.isnan(hv):
                    bar_hurst[int(ci): int(ci) + 20] = hv

            # origin-точки: равномерно в диапазоне [15%..80%]
            n = len(valid)
            starts = int(n * 0.15)
            ends   = int(n * 0.80) - HORIZON - 5
            origins = np.linspace(starts, ends, N_ORIGINS, dtype=int)

            for origin_k in origins:
                origin_date = str(pd.Timestamp(dates[origin_k]))[:16]
                ratio0      = float(ratio[origin_k])
                dratio_actual = dratio[origin_k: origin_k + HORIZON]
                if len(dratio_actual) < HORIZON:
                    continue
                phase_now = PHASE_NAMES[
                    query_phase_label(ratio, dratio[:origin_k], origin_k, PHASE_DN, PHASE_UP)
                ]

                for p in P_VALS:
                    if origin_k < p + 50:
                        continue
                    xi = 3 * (p + 1)

                    # предвычислить маски (переиспользуются всеми фильтрами)
                    _masks = {}
                    for fc in FILTER_COMBOS:
                        key = (fc["use_regime"], fc["use_phase"])
                        if key not in _masks:
                            _masks[key] = build_mask(
                                ratio, dratio, origin_k, p, xi,
                                bar_hurst, hurst_thr,
                                fc["use_regime"], fc["use_phase"]
                            )

                    for fc in FILTER_COMBOS:
                        mask = _masks[(fc["use_regime"], fc["use_phase"])]
                        pool_n = int(mask.sum()) if mask is not None else (origin_k - p)

                        for model in MODELS:
                            run_id += 1
                            t0 = time.time()
                            try:
                                if model == "LA1":
                                    dhat = forecast_la1_run(
                                        dratio, origin_k, p, xi, HORIZON,
                                        fc["norm_vecs"], fc["use_huber"], mask)
                                else:
                                    dhat = forecast_lwr(
                                        dratio, origin_k, p, xi, HORIZON,
                                        regime_mask=mask,
                                        norm_vecs=fc["norm_vecs"],
                                        use_huber=fc["use_huber"])
                            except Exception as e:
                                print(f"  ERR run {run_id}: {e}")
                                continue
                            elapsed = time.time() - t0

                            metrics = compute_metrics(dhat, dratio_actual, ratio0)
                            row = dict(
                                id=run_id, ticker=ticker, ma_window=ma_w,
                                origin_k=int(origin_k), origin_date=origin_date,
                                phase=phase_now,
                                p=p, xi=xi, model=model,
                                filter=fc["label"],
                                pool_n=pool_n,
                                **metrics,
                                elapsed_s=round(elapsed, 3),
                            )
                            all_rows.append(row)

                            with open(RESULTS, "a") as f:
                                f.write(json.dumps(row, default=lambda x: int(x) if hasattr(x, '__index__') else float(x)) + "\n")

                            m20 = metrics.get("mape_20", 999)
                            d20 = metrics.get("dir_20",  0)
                            marker = " ★" if m20 < 0.003 else (" ✓" if m20 < 0.006 else "")
                            print(f"  {ticker} MA={ma_w} ok={origin_k} p={p:2d} {model} "
                                  f"[{fc['label']:12s}] "
                                  f"mape5={metrics.get('mape_5',0):.4f} "
                                  f"mape20={m20:.4f} dir={d20:.2f}"
                                  f"{marker}")

    elapsed_total = time.time() - t_global
    print(f"\nИтого: {run_id} прогонов за {elapsed_total:.0f}с")

    # ── сохраняем сводку ────────────────────────────────────────────────────────
    write_summary(all_rows, elapsed_total)
    print(f"Сводка → {SUMMARY}")


def write_summary(rows, elapsed_total):
    df = pd.DataFrame(rows)
    if df.empty:
        SUMMARY.write_text("Нет данных.\n")
        return

    lines = []
    lines.append(f"BENCHMARK  {len(df)} прогонов  {elapsed_total:.0f}с\n")
    lines.append("=" * 70)

    # ── 1. средний MAPE по модели × фильтру ───────────────────────────────────
    lines.append("\n## 1. Средний MAPE@20 по модели × фильтру (все тикеры, все p)\n")
    piv = (df.groupby(["model", "filter"])["mape_20"]
              .mean().unstack("filter").round(5))
    lines.append(piv.to_string())

    # ── 2. оптимальный p по тикеру × окну × origin ────────────────────────────
    lines.append("\n\n## 2. Оптимальный p по точке старта (norm, LA1)\n")
    sub = df[(df["filter"] == "norm") & (df["model"] == "LA1")]
    best_p = (sub.loc[sub.groupby(["ticker", "ma_window", "origin_k"])["mape_20"].idxmin()]
                 [["ticker", "ma_window", "origin_k", "origin_date", "phase",
                   "p", "mape_20", "dir_20"]])
    best_p = best_p.sort_values(["ticker", "ma_window", "origin_k"])
    lines.append(best_p.to_string(index=False))

    # ── 3. лучшие 20 прогнозов абсолютно ──────────────────────────────────────
    lines.append("\n\n## 3. Топ-20 прогнозов (mape_20)\n")
    top = df.nsmallest(20, "mape_20")[
        ["ticker", "ma_window", "origin_date", "phase",
         "p", "model", "filter", "mape_20", "dir_20", "pool_n"]
    ]
    lines.append(top.to_string(index=False))

    # ── 4. лучшие прогнозы по фазе ────────────────────────────────────────────
    lines.append("\n\n## 4. Медианный MAPE@20 по фазе (norm, LA1)\n")
    sub2 = df[(df["filter"] == "norm") & (df["model"] == "LA1")]
    phase_stat = (sub2.groupby("phase")["mape_20"]
                      .agg(["median", "mean", "count"]).round(5)
                      .sort_values("median"))
    lines.append(phase_stat.to_string())

    # ── 5. MAPE по p (агрегировано) ───────────────────────────────────────────
    lines.append("\n\n## 5. MAPE@20 по p (norm, LA1, все origin и тикеры)\n")
    sub3 = df[(df["filter"] == "norm") & (df["model"] == "LA1")]
    p_stat = (sub3.groupby("p")["mape_20"]
                  .agg(["median", "mean", "min", "count"]).round(5))
    lines.append(p_stat.to_string())

    # ── 6. фильтр vs baseline: прирост/потеря ─────────────────────────────────
    lines.append("\n\n## 6. Delta MAPE@20 фильтра vs baseline (LA1, norm)\n")
    sub4 = df[df["model"] == "LA1"]
    pivot_filt = (sub4.groupby(["origin_k", "ticker", "ma_window", "p", "filter"])["mape_20"]
                      .mean().unstack("filter"))
    if "baseline" in pivot_filt.columns:
        for col in pivot_filt.columns:
            if col != "baseline":
                pivot_filt[f"Δ_{col}"] = pivot_filt[col] - pivot_filt["baseline"]
        delta_cols = [c for c in pivot_filt.columns if c.startswith("Δ_")]
        lines.append(pivot_filt[delta_cols].mean().round(6).to_string())

    # ── 7. neighbor-анализ (топ-30 прогонов) ──────────────────────────────────
    lines.append("\n\n## 7. Neighbour analysis (топ-30 прогонов по mape_20)\n")
    top30 = df.nsmallest(30, "mape_20")[
        (df.nsmallest(30, "mape_20")["model"] == "LA1") |
        (df.nsmallest(30, "mape_20")["mape_20"] < df["mape_20"].quantile(0.05))
    ].copy()

    # загружаем данные один раз для анализа соседей
    # (по тикерам из топ-30)
    neighbor_lines = []
    data_cache = {}
    for _, row in top30.iterrows():
        ticker = row["ticker"]
        ma_w   = int(row["ma_window"])
        key    = (ticker, ma_w)
        if key not in data_cache:
            dp = ROOT / "data" / "candles" / ticker / "1h.json"
            if not dp.exists():
                continue
            candles = json.loads(dp.read_text())
            norm    = normalize(candles, window=ma_w)
            valid   = norm.dropna(subset=["ma"]).reset_index(drop=True)
            data_cache[key] = valid
        valid  = data_cache[key]
        ratio  = valid["ratio"].values
        dratio = np.diff(ratio)
        ok     = int(row["origin_k"])
        p      = int(row["p"])
        xi     = int(row["xi"])

        try:
            nn_idx = get_neighbors_step1(dratio, ok, p, xi,
                                          norm_vecs=True, mask=None)
        except Exception:
            continue

        nn_dates   = [str(pd.Timestamp(valid["begin"].values[i + p]))[:10]
                      for i in nn_idx if i + p < len(valid)]
        nn_ages    = [ok - int(i) for i in nn_idx]   # баров назад
        nn_ratios  = [round(float(ratio[i + p]), 5) for i in nn_idx if i + p < len(valid)]

        neighbor_lines.append(
            f"  origin={row['origin_date']}  p={p}  mape20={row['mape_20']:.5f}\n"
            f"    возраст соседей (баров): min={min(nn_ages)} max={max(nn_ages)} "
            f"med={int(np.median(nn_ages))}\n"
            f"    даты соседей: {', '.join(nn_dates[:8])} ...\n"
            f"    ratio у соседей: {nn_ratios[:8]}\n"
        )

    lines.append("\n".join(neighbor_lines) if neighbor_lines else "  нет данных")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
