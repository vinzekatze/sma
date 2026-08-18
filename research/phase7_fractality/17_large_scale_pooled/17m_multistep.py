#!/usr/bin/env python3
"""
17m_multistep.py — итеративный 2-3-шаговый прогноз калиброванной S-map:
не для rMAE (заведомо хуже на дальних шагах), а чтобы увидеть, насколько
быстро прогноз "схлопывается" к persistence-логике (плечо ≈ такого же
размера, что и последнее известное) — то, что пользователь уже видел в
прототипе.

Механика: шаг 1 — обычный прогноз (qvec из известных пивотов). Шаг 2 —
предсказанное плечо шага 1 подставляется в вектор запроса (сдвиг окна),
направление разворачивается (зигзаг чередует), тот же пул (уже построенный
на момент i, новых данных не добавляется) фильтруется по новому направлению.
Шаг 3 — аналогично от шага 2.

Диагностика: |lr_2|/|qvec[0]| и |lr_3|/|qvec[0]| — если кластеруются около
1.0 с низким разбросом, прогноз выродился в "следующее плечо = последнее
известное" (персистенс), а не использует пул содержательно.

Точка: SBER, T_big=20%, ratio=0.85, D_allpeers, θ=25.697 (калибровка 17f).
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

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)

TARGET = "SBER"
T_BIG = 0.20
RATIO = 0.85
ARM = "D_allpeers"
M, THETA = 3, 25.697
N_STEPS = 3
SEED = 20260701


def build_unfiltered_pool(target_data, peer_data, rankings, checkpoints, t_big, m, t_frac, arm, i, full_lp, full_conf):
    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    confirm_date = full_conf[i]
    cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))

    t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]
    own_big_lp, _, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_big)
    if len(own_big_lp) < m + 1 or own_big_lp[-1] != full_lp[i]:
        return None

    qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
    if not np.all(np.isfinite(qvec)):
        return None
    q_dir = int(own_big_dir[-1])

    own_big_feats, own_big_tgts, own_big_dirs = exp17.build_pool_rows(own_big_lp, own_big_dir, m)
    pool_feats = [own_big_feats]; pool_tgts = [own_big_tgts]; pool_dirs = [own_big_dirs]

    own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
    f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, m)
    pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

    for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, arm):
        p_dates = peer_data[peer]["dates"]
        p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
        if p_cutoff < m + 2:
            continue
        p_lh = peer_data[peer]["lh"][:p_cutoff]
        p_ll = peer_data[peer]["ll"][:p_cutoff]
        p_dt = p_dates[:p_cutoff]
        p_lp, _, p_dir = exp17.build_zigzag(p_lh, p_ll, p_dt, t_frac)
        f, tg, dd = exp17.build_pool_rows(p_lp, p_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

    feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts); dirs = np.concatenate(pool_dirs)
    cur_lp = float(own_big_lp[-1])
    return feats, tgts, dirs, qvec, q_dir, cur_lp


def iterate_smap(feats, tgts, dirs, qvec1, qdir1, m, theta, n_steps):
    lrs = []
    qvec = qvec1.copy()
    qdir = qdir1
    for _ in range(n_steps):
        mask = dirs == qdir
        fd, td = feats[mask], tgts[mask]
        if len(fd):
            dd = np.linalg.norm(fd - qvec, axis=1)
            dup = dd < exp17.DUP_EPS
            if dup.any():
                fd, td = fd[~dup], td[~dup]
        lr = exp17._smap(qvec, fd, td, m + 2, theta)
        lrs.append(lr)
        if not np.isfinite(lr):
            break
        qvec = np.concatenate([[lr], qvec[:-1]])
        qdir = -qdir
    return lrs


def main():
    print("=== 17m_multistep ===")
    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, full_dirs = exp17.build_zigzag(lh, ll, dates, T_BIG)
    n_big = len(full_lp)
    t_frac = RATIO * T_BIG

    records = []
    for i in range(exp17.MIN_HIST, n_big - 1 - N_STEPS):  # нужны i+1..i+N_STEPS фактических пивота
        res = build_unfiltered_pool(target_data, peer_data, rankings, checkpoints, T_BIG, M, t_frac, ARM, i, full_lp, full_conf)
        if res is None:
            continue
        feats, tgts, dirs, qvec, q_dir, cur_lp = res
        lrs = iterate_smap(feats, tgts, dirs, qvec, q_dir, M, THETA, N_STEPS)
        if len(lrs) < N_STEPS or not all(np.isfinite(lrs)):
            continue

        last_known_leg = abs(qvec[0])
        actual_legs = [full_lp[i + k + 1] - full_lp[i + k] for k in range(N_STEPS)]

        row = {"step": i, "last_known_leg": last_known_leg}
        for k in range(N_STEPS):
            row[f"lr_{k+1}"] = lrs[k]
            row[f"ratio_{k+1}"] = abs(lrs[k]) / last_known_leg if last_known_leg > 1e-9 else np.nan
            row[f"actual_leg_{k+1}"] = actual_legs[k]
            row[f"actual_ratio_{k+1}"] = abs(actual_legs[k]) / last_known_leg if last_known_leg > 1e-9 else np.nan
        records.append(row)

    import pandas as pd
    df = pd.DataFrame(records)
    print(f"Шагов с полными {N_STEPS} горизонтами: {len(df)}\n")

    print("Отношение |предсказанное плечо_k| / |последнее известное плечо| "
          "(1.0 = точь-в-точь persistence-continuation):")
    for k in range(1, N_STEPS + 1):
        r = df[f"ratio_{k}"]
        ar = df[f"actual_ratio_{k}"]
        print(f"  шаг {k}: прогноз mean={r.mean():.3f} median={r.median():.3f} std={r.std():.3f}   "
              f"| факт mean={ar.mean():.3f} median={ar.median():.3f} std={ar.std():.3f}")

    # rMAE на каждом горизонте (только для контекста, не главная цель)
    print("\n(для контекста, не основная цель) rMAE по горизонтам относительно persistence h=1:")
    dz1 = df["actual_leg_1"].abs().mean()
    for k in range(1, N_STEPS + 1):
        err = (df[f"lr_{k}"] - df[f"actual_leg_{k}"]).abs().mean() if k == 1 else None
        # для k>1 сравниваем НАКОПЛЕННЫЙ прогноз с накопленным фактом
        cum_pred = df[[f"lr_{j}" for j in range(1, k+1)]].sum(axis=1)
        cum_act  = df[[f"actual_leg_{j}" for j in range(1, k+1)]].sum(axis=1)
        cum_err = (cum_pred - cum_act).abs().mean()
        print(f"  горизонт h={k}: mean|cum_err|={cum_err:.4f}  (rMAE~{cum_err/dz1:.3f} vs h=1 persistence denom)")

    df.to_csv(RESULTS / "multistep.csv", index=False, float_format="%.5f")

    # ── визуализация 2 случайных примеров ──
    rng = np.random.RandomState(SEED)
    close = np.exp(target_data["lc"])
    picked = rng.choice(len(df), size=min(2, len(df)), replace=False)

    fig, axes = plt.subplots(len(picked), 1, figsize=(11, 4.5 * len(picked)))
    if len(picked) == 1:
        axes = [axes]
    for ax, pi in zip(axes, picked):
        row = df.iloc[pi]
        i = int(row["step"])
        cur_lp = float(full_lp[i])
        idx_lo = max(0, i - 5)  # приближённо, используем позиции пивотов как есть для контекста

        # ищем позиции в баровом времени для рисования
        cutoff_i  = int(np.searchsorted(dates, full_conf[i], side="right")) - 1
        cutoff_ip = [int(np.searchsorted(dates, full_conf[i + k], side="right")) - 1 for k in range(1, N_STEPS + 1)]
        lo = max(0, cutoff_i - 20); hi = min(len(dates), cutoff_ip[-1] + 15)
        x = np.arange(lo, hi)
        ax.plot(x, close[lo:hi], color="black", lw=1, alpha=0.6, label="цена закрытия")

        cum_lr = 0.0
        pred_x = [cutoff_i]; pred_y = [float(np.exp(cur_lp))]
        for k in range(1, N_STEPS + 1):
            cum_lr += row[f"lr_{k}"]
            pred_x.append(cutoff_ip[k-1]); pred_y.append(float(np.exp(cur_lp + cum_lr)))
        ax.plot(pred_x, pred_y, color="red", marker="x", ls="--", lw=1.5, label="итеративный прогноз S-map")

        act_x = [cutoff_i]; act_y = [float(np.exp(cur_lp))]
        cum_act = 0.0
        for k in range(1, N_STEPS + 1):
            cum_act += row[f"actual_leg_{k}"]
            act_x.append(cutoff_ip[k-1]); act_y.append(float(np.exp(cur_lp + cum_act)))
        ax.plot(act_x, act_y, color="green", marker="*", ls="-", lw=1.5, ms=10, label="факт (реальные пивоты)")

        pers_x = [cutoff_i]; pers_y = [float(np.exp(cur_lp))]
        leg = row["last_known_leg"]; sign = 1 if row["lr_1"] > 0 else -1
        cum_pers = 0.0
        for k in range(1, N_STEPS + 1):
            cum_pers += sign * leg
            sign = -sign
            pers_x.append(cutoff_ip[k-1]); pers_y.append(float(np.exp(cur_lp + cum_pers)))
        ax.plot(pers_x, pers_y, color="gray", marker="o", ls=":", lw=1.2, label="наивный persistence-continuation")

        ax.set_title(f"step {i}: ratio_1={row['ratio_1']:.2f} ratio_2={row['ratio_2']:.2f} ratio_3={row['ratio_3']:.2f}")
        ax.legend(fontsize=8); ax.set_ylabel("цена, ₽")

    fig.suptitle(f"{TARGET} — итеративный {N_STEPS}-шаговый прогноз vs persistence-continuation", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "multistep_examples.png", dpi=130)
    print(f"\nСохранено: {RESULTS / 'multistep.csv'}, {FIGURES / 'multistep_examples.png'}")


if __name__ == "__main__":
    main()
