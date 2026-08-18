"""
Walk-forward validation of LA1 forecast quality.

Usage:
    python scripts/walkforward.py
    python scripts/walkforward.py --ma 5000 --horizon 40 --step 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from forcaster.forecast.normalize import normalize
from forcaster.forecast.la import forecast_la1, reconstruct_price


def run(
    ticker: str = "SBER",
    interval: str = "1h",
    ma_window: int = 5000,
    p_vals: list[int] | None = None,
    horizon: int = 40,
    step: int = 20,
    zones: list[tuple[str, int, int]] | None = None,
    data_dir: str = "data/candles",
) -> None:
    if p_vals is None:
        p_vals = [3, 4, 5, 6, 7, 8]

    path = Path(data_dir) / ticker / f"{interval}.json"
    candles = json.loads(path.read_text(encoding="utf-8"))

    norm   = normalize(candles, window=ma_window)
    valid  = norm.dropna(subset=["ma"]).reset_index(drop=True)
    dratio = np.diff(valid["ratio"].values)
    closes = valid["close"].values

    n = len(valid)
    if zones is None:
        third = n // 3
        zones = [
            ("Period 1", third,         2 * third),
            ("Period 2", 2 * third,     n - horizon - 5),
        ]

    print(f"Ticker: {ticker} [{interval}]  MA={ma_window}  horizon={horizon}  step={step}")

    for zone_name, z_start, z_end in zones:
        origins = range(z_start, z_end, step)
        results = {p: {"rmse": [], "dir": []} for p in p_vals}

        for origin_k in origins:
            actual = closes[origin_k + 1: origin_k + 1 + horizon]
            if len(actual) < horizon:
                continue
            actual_dir = np.sign(actual[-1] - closes[origin_k])

            for p in p_vals:
                xi = 3 * (p + 1)
                try:
                    dhat = forecast_la1(dratio, origin_k, p, xi, horizon)
                    phat = reconstruct_price(
                        dhat,
                        float(valid["ratio"].iloc[origin_k]),
                        float(valid["ma"].iloc[origin_k]),
                    )
                    results[p]["rmse"].append(float(np.sqrt(np.mean((phat - actual) ** 2))))
                    pred_dir = np.sign(phat[-1] - closes[origin_k])
                    results[p]["dir"].append(int(pred_dir == actual_dir))
                except Exception:
                    pass

        print(f"\n  {zone_name}  "
              f"({valid['begin'].iloc[z_start]:%Y-%m-%d} — "
              f"{valid['begin'].iloc[min(z_end, n-1)]:%Y-%m-%d})")
        print(f"  {'p':>3} | {'RMSE':>7} ±{'std':>6} | {'Dir%':>6} | n")
        print("  " + "-" * 36)
        for p in p_vals:
            r, d = results[p]["rmse"], results[p]["dir"]
            if r:
                print(f"  {p:>3} | {np.mean(r):>7.3f} ±{np.std(r):>6.3f} | "
                      f"{100 * np.mean(d):>5.1f}% | {len(r)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker",   default="SBER")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--ma",       type=int, default=5000)
    ap.add_argument("--horizon",  type=int, default=40)
    ap.add_argument("--step",     type=int, default=20)
    ap.add_argument("--p",        nargs="+", type=int, default=[3, 4, 5, 6, 7, 8])
    args = ap.parse_args()
    run(args.ticker, args.interval, args.ma, args.p, args.horizon, args.step)
