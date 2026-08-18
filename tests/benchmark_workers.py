#!/usr/bin/env python3
"""
Benchmark: run full forecast pipeline with varying worker counts.

Results are written to /output/benchmark_results.json.
Run via Docker — see tests/Dockerfile and tests/run.bat.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

DATA_PATH  = Path("/app/data/candles/SBER/1d.json")
OUTPUT_DIR = Path("/output")
ORIGIN_TS  = "2026-05-17"

PARAMS = dict(
    p_max       = 70,
    val_horizon = 5,
    horizon     = 15,
    auto_ma     = True,
    ma_fit_bars = 10,
    use_lwr     = True,
    top_n       = 5,
)


def run_single(candles: list[dict], n_workers: int) -> tuple[float, object]:
    from sma.core.pipeline import run_forecast
    from sma.core.models import ForecastParams

    params = ForecastParams(**PARAMS, max_workers=n_workers)

    last_pct = [0]
    t0 = time.perf_counter()

    def cb(done: int, total: int) -> None:
        pct = done * 100 // total
        if pct >= last_pct[0] + 10:
            last_pct[0] = pct
            elapsed = time.perf_counter() - t0
            print(f"    {pct:3d}%  ({done}/{total})  {elapsed:.1f}s", flush=True)

    out = run_forecast(candles, "SBER", "1d", ORIGIN_TS, params, progress_cb=cb)
    return time.perf_counter() - t0, out


def worker_plan(cpu_n: int) -> list[int]:
    """Return ascending list of worker counts to test, capped at cpu_n."""
    candidates = [1, 2, 4, 8, 16, 32]
    plan = sorted({w for w in candidates if w <= cpu_n})
    if cpu_n not in plan:
        plan.append(cpu_n)
    # also always include cpu_n // 2 if it's meaningfully different
    half = max(1, cpu_n // 2)
    if half not in plan:
        plan.append(half)
    return sorted(set(plan))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH) as f:
        candles = json.load(f)

    cpu_n = os.cpu_count() or 4
    workers = worker_plan(cpu_n)

    print(f"Host CPU count : {cpu_n}")
    print(f"Testing workers: {workers}")
    print(f"Origin         : {ORIGIN_TS}")
    print(f"p_max={PARAMS['p_max']}, val_horizon={PARAMS['val_horizon']}, "
          f"horizon={PARAMS['horizon']}, top_n={PARAMS['top_n']}")
    print()

    summary: list[dict] = []
    base_time: float | None = None

    for n in workers:
        print(f"─── workers = {n} ──────────────────────────────────────")
        sys.stdout.flush()

        elapsed, out = run_single(candles, n)

        if base_time is None:
            base_time = elapsed

        speedup = base_time / elapsed if elapsed > 0 else float("inf")

        row = {
            "workers"    : n,
            "elapsed_s"  : round(elapsed, 2),
            "speedup"    : round(speedup, 2),
            "ma_window"  : out.ma_window,
            "hurst"      : round(out.hurst_at_origin, 3),
            "close"      : out.close_at_origin,
            "top1_p"     : out.candidates[0].p,
            "top1_pca_k" : out.candidates[0].pca_k,
            "top1_mape"  : round(out.candidates[0].mape * 100, 4),
            "mean_price" : round(out.mean_price[-1], 2),
            "std_price"  : round(out.std_price[-1], 2),
        }
        summary.append(row)

        print(
            f"    done in {elapsed:.1f}s  (×{speedup:.2f})  |  "
            f"top-1: p={row['top1_p']} k={row['top1_pca_k']} "
            f"MAPE={row['top1_mape']:.4f}%  |  "
            f"forecast: {row['mean_price']} ± {row['std_price']} руб."
        )
        print()

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "origin_ts"  : ORIGIN_TS,
                "cpu_count"  : cpu_n,
                "params"     : PARAMS,
                "results"    : summary,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"Results saved → {out_path}")

    # ── print table ───────────────────────────────────────────────────────────
    print()
    print("╔══════════╦═══════════╦══════════╦═══════════════════════════════╗")
    print("║  workers ║  time (s) ║  speedup ║  forecast (mean ± std)        ║")
    print("╠══════════╬═══════════╬══════════╬═══════════════════════════════╣")
    for r in summary:
        print(
            f"║  {r['workers']:>6}  ║  {r['elapsed_s']:>7.1f}  "
            f"║  {r['speedup']:>5.2f}×  ║  "
            f"{r['mean_price']:>8.2f} ± {r['std_price']:>5.2f} руб.          ║"
        )
    print("╚══════════╩═══════════╩══════════╩═══════════════════════════════╝")


if __name__ == "__main__":
    main()
