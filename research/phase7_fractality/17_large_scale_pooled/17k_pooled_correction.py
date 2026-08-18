#!/usr/bin/env python3
"""
17k_pooled_correction.py — линейная коррекция амплитуды (17i), но (a,b)
подгоняются не по 42 шагам одного тикера, а по ОБЪЕДИНЁННОЙ, каузально
отсортированной по датам истории нескольких целевых тикеров (как в Stage 2).

17i показал: гипотеза о смещении верна (b=1.134>1 in-sample), но каузальная
подгонка на 15-40 своих точках слишком шумная — коррекция портит, а не чинит
(+4.2%). Тот же корень проблемы, что топил идею #2 (val-window) — на одном
тикере мало данных для мета-уровня. Здесь тот же трюк, что уже решил проблему
для базового прогноза (кросс-тикерный пул), применяется к самой коррекции.

Параметры S-map калибровки НЕ переоткалиброваны per-ticker — используются
параметры, найденные на SBER в 17f (m=3, θ=25.697, T_ratio=0.8987, arm=
D_allpeers), как и в Stage 2 (там тоже параметры не тюнились per-ticker).

Каузальность: на шаге i тикера X подгонка (a,b) использует ВСЕ шаги (любого
из 7 тикеров), чья confirm_date строго раньше confirm_date шага i.
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

TARGET_LIST = ["SBER", "LKOH", "CHMF", "NVTK", "MGNT", "VTBR", "NLMK"]
MIN_FIT = 15
blend.THETA = 25.697  # калиброванный на SBER (17f), используется для всех тикеров


def ols(x, y):
    A = np.column_stack([np.ones(len(x)), np.asarray(x)])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y), rcond=None)
    return float(coef[0]), float(coef[1])


def main():
    print("=== 17k_pooled_correction ===")
    all_data = {t: blend.exp17.load_ticker(t) for t in blend.exp17.UNIVERSE}

    master = []  # все шаги всех тикеров, с датами, для каузальной кросс-тикерной подгонки
    per_ticker_steps = {}

    for target in TARGET_LIST:
        blend.exp17.PEERS = [t for t in blend.exp17.UNIVERSE if t != target]
        target_data = all_data[target]
        years = sorted(set(int(d[:4]) for d in target_data["dates"]))
        checkpoints = np.array([f"{y}-01-01" for y in years])
        rankings = blend.exp17.compute_peer_rankings(target_data, all_data, checkpoints)

        lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
        full_lp, full_conf, full_dirs = blend.exp17.build_zigzag(lh, ll, dates, blend.T_BIG)
        t_frac = blend.T_RATIO * blend.T_BIG

        steps = blend.build_steps(target_data, all_data, rankings, checkpoints, blend.T_BIG, blend.M, t_frac, blend.ARM, full_lp, full_conf)
        for s in steps:
            s["confirm_date"] = str(full_conf[s["i"]])
            s["actual_lr"] = float(np.log(s["actual_price"]) - s["cur_lp"])
            s["ticker"] = target
        per_ticker_steps[target] = steps
        master.extend(steps)
        print(f"  {target}: {len(steps)} шагов")

    master.sort(key=lambda s: s["confirm_date"])
    print(f"\nВсего в объединённом пуле: {len(master)} шагов\n")

    # ── некаузальный fit на всём объединённом пуле (только проверка гипотезы) ──
    a_full, b_full = ols([s["smap_lr"] for s in master], [s["actual_lr"] for s in master])
    print(f"Некаузальный (in-sample) fit на объединённом пуле: a={a_full:.4f} b={b_full:.4f}\n")

    # ── каузальная кросс-тикерная коррекция ──
    results = []
    for target in TARGET_LIST:
        steps = sorted(per_ticker_steps[target], key=lambda s: s["confirm_date"])
        errs_raw, errs_corr, dz = [], [], []
        b_hist = []
        for s in steps:
            prior = [m for m in master if m["confirm_date"] < s["confirm_date"]]
            pred_raw_price = float(np.exp(s["cur_lp"] + s["smap_lr"]))
            err_raw = abs(pred_raw_price - s["actual_price"])
            if len(prior) >= MIN_FIT:
                a, b = ols([p["smap_lr"] for p in prior], [p["actual_lr"] for p in prior])
                corr_lr = a + b * s["smap_lr"]
                pred_corr_price = float(np.exp(s["cur_lp"] + corr_lr))
                err_corr = abs(pred_corr_price - s["actual_price"])
                errs_raw.append(err_raw); errs_corr.append(err_corr); dz.append(s["pers_err"])
                b_hist.append(b)

        if not errs_raw:
            print(f"  {target}: недостаточно данных для оценки (< {MIN_FIT} предшествующих шагов)")
            continue
        rmae_raw = float(np.mean(errs_raw) / np.mean(dz))
        rmae_corr = float(np.mean(errs_corr) / np.mean(dz))
        results.append({"ticker": target, "n": len(errs_raw), "rMAE_raw": rmae_raw, "rMAE_corr": rmae_corr,
                        "delta_pct": (rmae_corr - rmae_raw) / rmae_raw * 100,
                        "b_mean": float(np.mean(b_hist)), "b_last": b_hist[-1]})
        print(f"  {target:5s} n={len(errs_raw):3d}  raw={rmae_raw:.4f}  corr={rmae_corr:.4f}  "
              f"Δ={(rmae_corr-rmae_raw)/rmae_raw*100:+.1f}%  b_mean={np.mean(b_hist):.3f}")

    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"Среднее по {len(df)} тикерам: raw={df['rMAE_raw'].mean():.4f}  corr={df['rMAE_corr'].mean():.4f}  "
          f"Δ={(df['rMAE_corr'].mean()-df['rMAE_raw'].mean())/df['rMAE_raw'].mean()*100:+.1f}%")
    wins = (df["rMAE_corr"] < df["rMAE_raw"]).sum()
    print(f"Коррекция лучше raw на {wins}/{len(df)} тикерах")

    out = RESULTS / "pooled_correction.csv"
    df.to_csv(out, index=False, float_format="%.5f")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
