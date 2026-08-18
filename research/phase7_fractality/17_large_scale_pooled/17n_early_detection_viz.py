#!/usr/bin/env python3
"""
17n_early_detection_viz.py — есть ли что ловить между прогнозом и его
подтверждением?

Уточнение постановки: интересующее окно — НЕ между истинным экстремумом
пивота i и его собственным подтверждением (это уже было в Stage 1g), а
между ТОЧКОЙ ПРОГНОЗА (conf_i — момент, когда пивот i подтвердился, и мы
выдаём прогноз на пивот i+1) и ТОЧКОЙ ПОДТВЕРЖДЕНИЯ прогноза (conf_{i+1}
или ext_{i+1} — когда цель фактически достигнута). Это и есть "живое" окно
прогноза (в Stage 1g было 53-419 дней на конкретных примерах) — сейчас оно
просто ждётся статично до следующего T_big-события. Внутри него более
мелкий зигзаг (T_tiny << T_frac) уже непрерывно подтверждается и показывает
направление — можно ли использовать эти промежуточные точки, чтобы
отслеживать/уточнять прогноз по ходу, а не ждать вслепую?

Чисто визуальная разведка перед тем, как строить механизм: видно ли на
глаз в окне прогноза что-то полезное, или мелкий зигзаг просто шумит.

T_tiny = 0.15 × T_big = 3% (нижний конец сетки ratio из Stage 1, там он был
худшим для пула — но, возможно, как раз подходящим масштабом для трекинга).
"""
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
spec17 = importlib.util.spec_from_file_location("exp17", HERE / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

TARGET = "SBER"
T_BIG = 0.20
T_TINY = 0.15 * T_BIG   # 3%
STEPS_TO_SHOW = [63, 46, 50]   # те же примеры, что в Stage 1g — для прямой сравнимости


def build_zigzag_idx(lh, ll, dates, thr):
    lp, ext_idx, conf, conf_idx, dirs = [], [], [], [], []
    cur = 0
    ext = (lh[0] + ll[0]) / 2.0
    ei = 0
    for i in range(len(lh)):
        if cur == 0:
            if lh[i] - ext >= thr:
                cur = 1; ext = lh[i]; ei = i
            elif ext - ll[i] >= thr:
                cur = -1; ext = ll[i]; ei = i
        elif cur == 1:
            if lh[i] > ext:
                ext = lh[i]; ei = i
            elif ext - ll[i] >= thr:
                lp.append(ext); ext_idx.append(ei)
                conf.append(dates[i]); conf_idx.append(i); dirs.append(1)
                cur = -1; ext = ll[i]; ei = i
        else:
            if ll[i] < ext:
                ext = ll[i]; ei = i
            elif lh[i] - ext >= thr:
                lp.append(ext); ext_idx.append(ei)
                conf.append(dates[i]); conf_idx.append(i); dirs.append(-1)
                cur = 1; ext = lh[i]; ei = i
    return (np.array(lp), np.array(ext_idx), np.array(conf),
            np.array(conf_idx), np.array(dirs, dtype=np.int8))


def main():
    target_data = exp17.load_ticker(TARGET)
    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    close = np.exp(target_data["lc"])

    zz_lp, zz_ext_idx, zz_conf, zz_conf_idx, zz_dirs = build_zigzag_idx(lh, ll, dates, T_BIG)
    tiny_lp, tiny_ext_idx, tiny_conf, tiny_conf_idx, tiny_dirs = build_zigzag_idx(lh, ll, dates, T_TINY)

    fig, axes = plt.subplots(len(STEPS_TO_SHOW), 1, figsize=(12, 4.5 * len(STEPS_TO_SHOW)))

    for ax, i in zip(axes, STEPS_TO_SHOW):
        conf_i = int(zz_conf_idx[i])           # точка прогноза (пивот i подтверждён, выдаём прогноз на i+1)
        ext_n  = int(zz_ext_idx[i + 1])         # цель фактически достигнута
        conf_n = int(zz_conf_idx[i + 1])        # официальное подтверждение i+1

        lo = max(0, conf_i - 10); hi = min(len(dates), conf_n + 10)
        x = np.arange(lo, hi)
        ax.plot(x, close[lo:hi], color="black", lw=1, alpha=0.5, label="цена закрытия")

        in_big = (zz_ext_idx >= lo) & (zz_ext_idx <= hi)
        ax.plot(zz_ext_idx[in_big], np.exp(zz_lp[in_big]), color="darkorange", lw=1.8, marker="o", ms=5,
                label="зигзаг T_big=20%")

        in_tiny = (tiny_ext_idx >= lo) & (tiny_ext_idx <= hi)
        ax.plot(tiny_ext_idx[in_tiny], np.exp(tiny_lp[in_tiny]), color="steelblue", lw=1, marker=".", ms=4,
                alpha=0.8, label=f"зигзаг T_tiny={T_TINY*100:.0f}%")

        ax.axvspan(conf_i, ext_n, color="green", alpha=0.10, label="прогноз → цель фактически достигнута")
        ax.axvspan(ext_n, conf_n, color="red", alpha=0.10, label="цель достигнута → официально подтверждено")
        ax.axvline(conf_i, color="blue", ls="--", lw=1, alpha=0.7)
        ax.axvline(ext_n,  color="green", ls="--", lw=1, alpha=0.7)
        ax.axvline(conf_n, color="red", ls="--", lw=1, alpha=0.7)

        n_tiny_in_window = int(((tiny_conf_idx > conf_i) & (tiny_conf_idx <= conf_n)).sum())
        days_to_target = (np.datetime64(dates[ext_n][:10]) - np.datetime64(dates[conf_i][:10])).astype(int)
        days_to_conf    = (np.datetime64(dates[conf_n][:10]) - np.datetime64(dates[conf_i][:10])).astype(int)

        ax.set_title(f"step {i}: прогноз {dates[conf_i][:10]} → цель {dates[ext_n][:10]} ({days_to_target} дн.) "
                     f"→ офиц. подтв. {dates[conf_n][:10]} ({days_to_conf} дн.)  |  "
                     f"внутри окна подтвердилось {n_tiny_in_window} T_tiny-пивотов")
        ax.legend(fontsize=8, loc="best")
        ax.set_ylabel("цена, ₽")

    fig.suptitle(f"{TARGET} — мелкий зигзаг (T_tiny={T_TINY*100:.0f}%) внутри слепой зоны T_big=20%", fontsize=11)
    fig.tight_layout()
    out = FIGURES / "early_detection_blindzone.png"
    fig.savefig(out, dpi=130)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
