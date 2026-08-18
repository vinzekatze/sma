#!/usr/bin/env python3
"""
17o_efficiency_ratio.py — коррелирует ли Kaufman Efficiency Ratio (классическая
мера "трендовости" из ТА, не связанная с зигзаг-признаками) с тем, насколько
модель недооценивает/переоценивает амплитуду движения?

ER = |close[t]-close[t-N]| / Σ|close[k]-close[k-1]|, k=t-N+1..t.
ER→1: чистый директивный тренд (путь ≈ смещению). ER→0: шум/боковик (путь >>
смещение). Считается по БАРАМ (не по зигзаг-пивотам) — принципиально другой
источник информации, чем qvec (17l, trend=qvec[0], там сигнала не нашлось).

Каузально: ER на шаге i считается по N барам ДО confirm_date(i) включительно.

Сначала — просто корреляция ER с log(|actual_leg| / |smap_lr|) (насколько
модель недо-/переоценила амплитуду) на объединённом пуле 7 тикеров. Если
корреляции нет — не строить коррекцию (17l research уже показал, что лишний
шумный параметр только вредит).
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
N_ER = 60   # ~3 месяца торговых дней
blend.THETA = 25.697


def efficiency_ratio(close, end_idx, n):
    """ER по n барам, заканчивающимся на end_idx включительно (каузально)."""
    lo = end_idx - n
    if lo < 0:
        return np.nan
    window = close[lo:end_idx + 1]
    net = abs(window[-1] - window[0])
    path = np.sum(np.abs(np.diff(window)))
    return net / path if path > 1e-12 else np.nan


def main():
    print("=== 17o_efficiency_ratio ===")
    all_data = {t: blend.exp17.load_ticker(t) for t in blend.exp17.UNIVERSE}

    master = []
    for target in TARGET_LIST:
        blend.exp17.PEERS = [t for t in blend.exp17.UNIVERSE if t != target]
        target_data = all_data[target]
        years = sorted(set(int(d[:4]) for d in target_data["dates"]))
        checkpoints = np.array([f"{y}-01-01" for y in years])
        rankings = blend.exp17.compute_peer_rankings(target_data, all_data, checkpoints)

        lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
        close = np.exp(target_data["lc"])
        full_lp, full_conf, full_dirs = blend.exp17.build_zigzag(lh, ll, dates, blend.T_BIG)
        t_frac = blend.T_RATIO * blend.T_BIG

        steps = blend.build_steps(target_data, all_data, rankings, checkpoints, blend.T_BIG, blend.M, t_frac, blend.ARM, full_lp, full_conf)
        for s in steps:
            i = s["i"]
            conf_idx = int(np.searchsorted(dates, full_conf[i], side="right")) - 1
            er = efficiency_ratio(close, conf_idx, N_ER)
            actual_lr = float(np.log(s["actual_price"]) - s["cur_lp"])
            s.update({"ticker": target, "ER": er, "actual_lr": actual_lr,
                      "log_ratio": float(np.log(max(abs(actual_lr), 1e-6) / max(abs(s["smap_lr"]), 1e-6)))})
        master.extend(steps)

    df = pd.DataFrame(master).dropna(subset=["ER"])
    print(f"Точек с валидным ER (N={N_ER}): {len(df)} из {len(master)}\n")

    corr_lr = df["ER"].corr(df["log_ratio"])
    corr_abs_act = df["ER"].corr(df["actual_lr"].abs())
    corr_smap = df["ER"].corr(df["smap_lr"].abs())
    print(f"corr(ER, log(|actual|/|pred|))  = {corr_lr:+.3f}   (>0 значило бы: высокий ER -> модель недооценивает)")
    print(f"corr(ER, |actual_lr|)           = {corr_abs_act:+.3f}   (ожидаемо >0 — при высоком ER реальные плечи крупнее)")
    print(f"corr(ER, |smap_lr|)             = {corr_smap:+.3f}   (ловит ли МОДЕЛЬ уже сама этот эффект через пул?)")

    print("\nПо квартилям ER — средний |actual_lr| и |smap_lr| и their ratio:")
    df["ER_q"] = pd.qcut(df["ER"], 4, labels=["Q1(шум)", "Q2", "Q3", "Q4(тренд)"])
    g = df.groupby("ER_q", observed=True).agg(
        n=("ER", "size"), ER_mean=("ER", "mean"),
        actual_abs=("actual_lr", lambda x: x.abs().mean()),
        smap_abs=("smap_lr", lambda x: x.abs().mean()),
    )
    g["ratio_actual_smap"] = g["actual_abs"] / g["smap_abs"]
    print(g.round(4).to_string())

    df.to_csv(RESULTS / "efficiency_ratio.csv", index=False, float_format="%.5f")
    print(f"\nСохранено: {RESULTS / 'efficiency_ratio.csv'}")


if __name__ == "__main__":
    main()
