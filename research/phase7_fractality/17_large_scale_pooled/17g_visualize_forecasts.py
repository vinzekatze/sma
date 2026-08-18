#!/usr/bin/env python3
"""
17g_visualize_forecasts.py — визуализация случайных прогнозов калиброванной
S-map (m=3, θ=25.697, T_ratio=0.8987, T_big=20%, arm=D_allpeers, из 17f) на
ценовом графике SBER с наложенным зигзагом.

v2: зигзаг строится с учётом ИСТИННОГО индекса экстремума (не только даты
подтверждения) — исходный build_zigzag хранит только значение пивота и дату
ПОДТВЕРЖДЕНИЯ (которая всегда позже, чем сам экстремум), поэтому в v1 точки
"плавали" над линией цены, а не сидели на зигзаге. Здесь явно разделены:
  - экстремум пивота (истинная точка разворота, где реально стоит маркер)
  - дата подтверждения (момент, когда это стало ИЗВЕСТНО и можно действовать)
На график нанесены обе — вертикальные линии с датами показывают, когда
именно решение стало доступно и сколько времени реально прошло до цели.

Каузальный контракт прогноза не меняется (тот же build_all_pools, что и в
17f) — визуализируются честные walk-forward прогнозы, дорисовка "будущего"
(факт, экстремум следующего пивота) только для отображения, не для прогноза.

Выбор шагов — случайный (seed фиксирован), НЕ подобранный по качеству.
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
ARM = "D_allpeers"
M, THETA, T_RATIO = 3, 25.697, 0.8987   # калиброванный S-map из 17f
N_SAMPLES = 3
SEED = 20260701


def build_zigzag_idx(lh, ll, dates, thr):
    """Как exp17.build_zigzag, но дополнительно хранит индекс бара экстремума
    (не только дату подтверждения) — нужно для отрисовки зигзага НА цене."""
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


def build_all_pools_with_idx(target_data, peer_data, rankings, checkpoints, t_big, m, t_frac, arm, full_lp, full_conf):
    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    n_big = len(full_lp)
    out = []
    for i in range(exp17.MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))

        t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]
        own_big_lp, _, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_big)
        if len(own_big_lp) < m + 1 or own_big_lp[-1] != full_lp[i]:
            continue

        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue
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
        mask = dirs == q_dir
        feats_d, tgts_d = feats[mask], tgts[mask]

        if len(feats_d):
            d = np.linalg.norm(feats_d - qvec, axis=1)
            dup_mask = d < exp17.DUP_EPS
            if dup_mask.any():
                feats_d, tgts_d = feats_d[~dup_mask], tgts_d[~dup_mask]
        cur_lp = float(own_big_lp[-1])
        out.append((i, qvec, feats_d, tgts_d, cur_lp))
    return out


def main():
    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    close = np.exp(target_data["lc"])

    full_lp, full_conf, full_dirs = exp17.build_zigzag(lh, ll, dates, T_BIG)
    # тот же зигзаг, но с индексами экстремумов — для отрисовки и для контекстных пивотов
    zz_lp, zz_ext_idx, zz_conf, zz_conf_idx, zz_dirs = build_zigzag_idx(lh, ll, dates, T_BIG)
    assert len(zz_lp) == len(full_lp) and np.allclose(zz_lp, full_lp), "рассинхрон зигзагов"

    t_frac = T_RATIO * T_BIG
    pools = build_all_pools_with_idx(target_data, peer_data, rankings, checkpoints, T_BIG, M, t_frac, ARM, full_lp, full_conf)
    print(f"Доступно шагов: {len(pools)}")

    rng = np.random.RandomState(SEED)
    picked = rng.choice(len(pools), size=min(N_SAMPLES, len(pools)), replace=False)

    fig, axes = plt.subplots(len(picked), 1, figsize=(12, 5 * len(picked)))
    if len(picked) == 1:
        axes = [axes]

    for ax, pi in zip(axes, picked):
        i, qvec, feats_d, tgts_d, cur_lp = pools[pi]

        lr = exp17._smap(qvec, feats_d, tgts_d, M + 2, THETA)
        pred_price = float(np.exp(cur_lp + lr))
        actual_price = float(np.exp(full_lp[i + 1]))
        cur_price = float(np.exp(cur_lp))
        pers_price = float(np.exp(full_lp[i - 1]))

        ext_i   = int(zz_ext_idx[i])       # истинный бар экстремума текущего пивота
        conf_i  = int(zz_conf_idx[i])      # бар, где это ПОДТВЕРДИЛОСЬ (здесь мы узнаём и можем действовать)
        ext_n   = int(zz_ext_idx[i + 1])   # истинный бар экстремума СЛЕДУЮЩЕГО (факт)
        conf_n  = int(zz_conf_idx[i + 1])  # бар, где следующий пивот подтвердился

        idx_lo = max(0, ext_i - 30)
        idx_hi = min(len(dates), ext_n + 15)
        x = np.arange(idx_lo, idx_hi)
        ax.plot(x, close[idx_lo:idx_hi], color="black", lw=1, alpha=0.6, label="цена закрытия (1d)")

        # зигзаг T_big в окне (контекстные пивоты + текущий + следующий)
        in_win = (zz_ext_idx >= idx_lo) & (zz_ext_idx <= idx_hi)
        zz_x = zz_ext_idx[in_win]; zz_y = np.exp(zz_lp[in_win])
        ax.plot(zz_x, zz_y, color="darkorange", lw=1.8, marker="o", ms=4, alpha=0.9, label="зигзаг T_big=20% (истинные развороты)")

        entry_price = float(close[conf_i])   # реальная цена в момент, когда мы УЗНАЁМ о пивоте (не экстремум!)

        ax.scatter([ext_i], [cur_price], color="blue", zorder=6, s=90, alpha=0.5, label="экстремум пивота (известен только задним числом)")
        ax.scatter([conf_i], [entry_price], color="cyan", edgecolor="black", zorder=7, s=110, marker="D",
                   label="реальная цена входа (в момент подтверждения)")
        ax.scatter([ext_n], [actual_price], color="green", zorder=6, s=140, marker="*", label="факт (экстремум след. пивота)")
        ax.scatter([conf_n], [pred_price], color="red", zorder=6, s=70, marker="x", label="прогноз S-map (цена; момент — неизвестен)")
        ax.hlines(pers_price, ext_i, ext_n, colors="gray", linestyles="dotted", lw=1, label="persistence (пред. пивот)")

        entry_gap_pct = (entry_price - cur_price) / cur_price * 100

        # вертикальные линии: когда стало известно
        ax.axvline(conf_i, color="blue", ls="--", lw=1, alpha=0.7)
        ax.axvline(conf_n, color="green", ls="--", lw=1, alpha=0.7)
        ymin, ymax = ax.get_ylim()
        ax.text(conf_i, ymax, f"  подтверждено\n  {dates[conf_i][:10]}\n  (можно входить)",
                va="top", ha="left", fontsize=7.5, color="blue")
        ax.text(conf_n, ymax, f"  подтверждено\n  {dates[conf_n][:10]}",
                va="top", ha="left", fontsize=7.5, color="green")

        days_conf_to_target = (np.datetime64(dates[ext_n][:10]) - np.datetime64(dates[conf_i][:10])).astype(int)
        days_conf_to_conf   = (np.datetime64(dates[conf_n][:10]) - np.datetime64(dates[conf_i][:10])).astype(int)

        err_price = abs(pred_price - actual_price)
        err_pct = err_price / actual_price * 100
        pers_err = abs(pers_price - actual_price)
        step_rmae = err_price / pers_err if pers_err > 1e-9 else float("nan")

        ax.set_title(
            f"step {i}   вход {dates[conf_i][:10]} (цена входа {entry_price:.2f}₽, "
            f"уже {entry_gap_pct:+.1f}% от экстремума {cur_price:.2f}₽) → цель достигнута {dates[ext_n][:10]} "
            f"({days_conf_to_target} дн.), офиц. подтверждено {dates[conf_n][:10]} ({days_conf_to_conf} дн.)\n"
            f"факт={actual_price:.2f}₽  прогноз={pred_price:.2f}₽  ошибка={err_price:.2f}₽ ({err_pct:.1f}%)  step_rMAE={step_rmae:.3f}",
            fontsize=9.5
        )
        ax.legend(loc="lower left", fontsize=7.5)
        ax.set_ylabel("цена, ₽")

    fig.suptitle(f"{TARGET}  T_big=20%  S-map калиброванный (m={M}, θ={THETA}, T_ratio={T_RATIO})  "
                 f"— {len(picked)} случайных шага (seed={SEED})", fontsize=11)
    fig.tight_layout()
    out = FIGURES / "random_forecasts_smap_calibrated_v2.png"
    fig.savefig(out, dpi=130)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
