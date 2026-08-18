#!/usr/bin/env python3
"""
17i_linear_correction.py — линейная коррекция систематического смещения
калиброванного S-map (недооценка крупных плеч, см. Stage 1g).

Коррекция: actual_lr ≈ a + b·predicted_lr (в лог-возвратах), OLS.
Если b>1 — подтверждает shrinkage-bias (модель регрессирует к среднему
плечу пула, недооценивая выбросы) и коррекция должна помочь.

КАУЗАЛЬНО: (a,b) на шаге i подгоняются ТОЛЬКО по шагам j<i (расширяющееся
окно) — иначе коррекция сама станет источником утечки. MIN_FIT=15 шагов
до первого применения коррекции (шаги до этого — некорректируемый прогон).

Для сравнения — некаузальный (по всей выборке) fit: показывает потолок
метода и подтверждает/опровергает саму гипотезу о смещении, но НЕ является
честной оценкой качества (in-sample, оптимистично).
"""
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("blend", HERE / "17h_persistence_blend.py")
blend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blend)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

MIN_FIT = 15


def ols(x, y):
    """actual = a + b*pred, обычный МНК."""
    x = np.asarray(x); y = np.asarray(y)
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def main():
    print("=== 17i_linear_correction ===")
    target_data = blend.exp17.load_ticker(blend.TARGET)
    blend.exp17.PEERS = [t for t in blend.exp17.UNIVERSE if t != blend.TARGET]
    peer_data = {t: blend.exp17.load_ticker(t) for t in blend.exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = blend.exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, full_dirs = blend.exp17.build_zigzag(lh, ll, dates, blend.T_BIG)
    t_frac = blend.T_RATIO * blend.T_BIG

    steps = blend.build_steps(target_data, peer_data, rankings, checkpoints, blend.T_BIG, blend.M, t_frac, blend.ARM, full_lp, full_conf)
    for s in steps:
        s["actual_lr"] = float(np.log(s["actual_price"]) - s["cur_lp"])
    print(f"Шагов: {len(steps)}  (MIN_FIT={MIN_FIT} до первой коррекции)\n")

    # ── некаузальный fit (только для проверки гипотезы о смещении, не для метрики) ──
    all_pred = [s["smap_lr"] for s in steps]
    all_act  = [s["actual_lr"] for s in steps]
    a_full, b_full = ols(all_pred, all_act)
    print(f"Некаузальный (in-sample, оптимистично!) fit: a={a_full:.4f}  b={b_full:.4f}")
    print(f"  b>1 означало бы недооценку амплитуды — {'ПОДТВЕРЖДЕНО' if b_full > 1 else 'НЕ подтверждено'} на всей выборке\n")

    # ── каузальная walk-forward коррекция ──
    errs_raw, errs_corr, dz = [], [], []
    trace = []
    for idx, s in enumerate(steps):
        row = {"i": s["i"], "smap_lr": s["smap_lr"], "actual_lr": s["actual_lr"]}
        pred_raw_price = float(np.exp(s["cur_lp"] + s["smap_lr"]))
        err_raw = abs(pred_raw_price - s["actual_price"])

        if idx >= MIN_FIT:
            prior_pred = [steps[j]["smap_lr"] for j in range(idx)]
            prior_act  = [steps[j]["actual_lr"] for j in range(idx)]
            a, b = ols(prior_pred, prior_act)
            corr_lr = a + b * s["smap_lr"]
            pred_corr_price = float(np.exp(s["cur_lp"] + corr_lr))
            err_corr = abs(pred_corr_price - s["actual_price"])
            errs_raw.append(err_raw); errs_corr.append(err_corr); dz.append(s["pers_err"])
            row.update({"a": a, "b": b, "err_raw": err_raw, "err_corr": err_corr})
        trace.append(row)

    rmae_raw  = float(np.mean(errs_raw) / np.mean(dz))
    rmae_corr = float(np.mean(errs_corr) / np.mean(dz))
    print(f"Каузальная walk-forward оценка (n={len(errs_raw)}, после burn-in {MIN_FIT}):")
    print(f"  без коррекции:  rMAE={rmae_raw:.4f}")
    print(f"  с коррекцией:   rMAE={rmae_corr:.4f}")
    print(f"  Δ = {(rmae_corr-rmae_raw)/rmae_raw*100:+.1f}%")

    b_hist = [r["b"] for r in trace if "b" in r]
    print(f"\n  b по шагам: min={min(b_hist):.3f} max={max(b_hist):.3f} последний={b_hist[-1]:.3f}")

    pd.DataFrame(trace).to_csv(RESULTS / "linear_correction.csv", index=False, float_format="%.5f")
    print(f"\nСохранено: {RESULTS / 'linear_correction.csv'}")


if __name__ == "__main__":
    main()
