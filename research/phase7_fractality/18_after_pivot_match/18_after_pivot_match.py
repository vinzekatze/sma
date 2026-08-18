#!/usr/bin/env python3
"""
18_after_pivot_match.py — проверка гипотезы разворота: возможно, прогноз
калиброванной S-map (эксп.17f, T_big=20%, D_allpeers, θ=25.697, T_ratio=0.8987)
на самом деле лучше совпадает не с целевым T_big-пивотом, а с одним из мелких
пивотов (F — тот же порог, что у пула; G — ещё мельче, T_big/7), которые
происходят ПОСЛЕ него, чем с самим T_big-пивотом.

Модель и пул НЕ меняются — переиспользуется механизм итеративного
многошагового прогноза из research/reference/smap_ref.py (--steps N) /
17m_multistep.py: шаг h>1 получается сдвигом вектора запроса на предсказанное
плечо шага h-1, разворотом направления, переотбором пула по новому
направлению.

F/G-зигзаги строятся ОДИН РАЗ на ПОЛНОМ (необрезанном) ряду — они используются
только для разметки фактических точек сравнения (тот же приём, что уже
применяется для full_lp/full_conf самого T_big в эксп.17 — не утечка в
модель, полный ряд не участвует в прогнозе).

Каузальный контракт (не меняется относительно эксп.17/17m):
целевой ряд и каждый пир физически обрезаются (`arr[:cutoff_idx]`) до
построения зигзага пула на каждом шаге walk-forward.

Для каждого origin i и горизонта h=1..N_STEPS:
  pred_price_h  = exp(cur_lp + cumsum(lr_1..lr_h))           — прогноз (не меняется)
  actual_BIG_h  = exp(full_lp[i+h])                          — фактический T_big-пивот h
  pers_BIG_h    = exp(full_lp[i+h-2])                        — persistence того же уровня

  Для level ∈ {F, G} и k=1..K_level:
    j = первый индекс level-зигзага строго после даты подтверждения full_conf[i+h] + (k-1)
    actual_level_k = exp(level_lp[j])
    pers_level_k   = exp(level_lp[j-2])                      — тот же принцип persistence

  Три варианта окна по датам (ограничение того, что считается "после него"):
    immediate — level-пивот должен быть раньше full_conf[i+h+1]
    extended  — раньше full_conf[i+h+2]
    unbounded — без ограничения (первые K_level пивотов уровня после c_h)

rMAE = mean(|pred-actual|) / mean(|pers-actual|)  — по каждой группе (h,level,k,window) отдельно
       + среднее по k внутри уровня (суммарно) + win-rate (какая цель ближе к прогнозу).
"""
import importlib.util
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
EXP17_DIR = HERE.parents[0] / "17_large_scale_pooled"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

spec17 = importlib.util.spec_from_file_location("exp17", EXP17_DIR / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

# ── Параметры (калибровка эксп.17f, точка T_big=20%, D_allpeers) ─────────────
TARGET = "SBER"
T_BIG = 0.20
ARM = "D_allpeers"
M = 3
THETA = 25.697
T_RATIO_POOL = 0.8987          # = уровень F
T_FRAC = T_RATIO_POOL * T_BIG
G_RATIO = 1.0 / 7.0            # уровень G — заметно мельче
T_TINY = T_BIG * G_RATIO

N_STEPS = 3                    # h = 1..3, многошаговый прогноз
K_F = 2
K_G = 5
WINDOWS = ["immediate", "extended", "unbounded"]

MIN_HIST = exp17.MIN_HIST


# ── Пул для одного origin (копия механики 17m, без изменений) ───────────────

def build_pool_for_origin(target_data, peer_data, rankings, checkpoints, i, full_lp, full_conf):
    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    confirm_date = full_conf[i]
    cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))

    t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]
    own_big_lp, _, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, T_BIG)
    if len(own_big_lp) < M + 1 or own_big_lp[-1] != full_lp[i]:
        return None

    qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(M)])
    if not np.all(np.isfinite(qvec)):
        return None
    q_dir = int(own_big_dir[-1])

    own_big_feats, own_big_tgts, own_big_dirs = exp17.build_pool_rows(own_big_lp, own_big_dir, M)
    pool_feats = [own_big_feats]; pool_tgts = [own_big_tgts]; pool_dirs = [own_big_dirs]

    own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, T_FRAC)
    f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, M)
    pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

    for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
        p_dates = peer_data[peer]["dates"]
        p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
        if p_cutoff < M + 2:
            continue
        p_lh = peer_data[peer]["lh"][:p_cutoff]
        p_ll = peer_data[peer]["ll"][:p_cutoff]
        p_dt = p_dates[:p_cutoff]
        p_lp, _, p_dir = exp17.build_zigzag(p_lh, p_ll, p_dt, T_FRAC)
        f, tg, dd = exp17.build_pool_rows(p_lp, p_dir, M)
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


# ── Поиск фактических F/G пивотов "после" big-пивота h ───────────────────────

def find_level_matches(level_lp, level_conf, after_date, upper_date, k_max):
    """Первые k_max пивотов уровня строго после after_date, опционально раньше upper_date."""
    j0 = int(np.searchsorted(level_conf, after_date, side="right"))
    out = []
    for k in range(1, k_max + 1):
        idx = j0 + k - 1
        if idx >= len(level_lp) or idx < 2:
            break
        if upper_date is not None and level_conf[idx] >= upper_date:
            break
        out.append((k, idx))
    return out


def main():
    t0 = time.time()
    print("=== 18_after_pivot_match ===")
    print(f"Target={TARGET} T_BIG={T_BIG} ARM={ARM} M={M} THETA={THETA}")
    print(f"T_FRAC(F)={T_FRAC:.4f} (ratio={T_RATIO_POOL})  T_TINY(G)={T_TINY:.4f} (ratio={G_RATIO:.4f})")
    print(f"N_STEPS={N_STEPS}  K_F={K_F}  K_G={K_G}  WINDOWS={WINDOWS}")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, full_dirs = exp17.build_zigzag(lh, ll, dates, T_BIG)
    fF_lp, fF_conf, _ = exp17.build_zigzag(lh, ll, dates, T_FRAC)
    fG_lp, fG_conf, _ = exp17.build_zigzag(lh, ll, dates, T_TINY)
    n_big = len(full_lp)
    print(f"Пивотов: T_big={n_big}  F={len(fF_lp)}  G={len(fG_lp)}")

    # нужны фактические big-пивоты до i+N_STEPS+2 (extended-окно на последнем h)
    i_hi = n_big - 1 - N_STEPS - 2
    print(f"Диапазон origin: [{MIN_HIST}, {i_hi})  n≈{max(0, i_hi - MIN_HIST)}")

    long_rows = []   # (origin,h,level,k,window,pred_price,actual_price,pers_price)
    for i in range(MIN_HIST, i_hi):
        res = build_pool_for_origin(target_data, peer_data, rankings, checkpoints, i, full_lp, full_conf)
        if res is None:
            continue
        feats, tgts, dirs, qvec, q_dir, cur_lp = res
        lrs = iterate_smap(feats, tgts, dirs, qvec, q_dir, M, THETA, N_STEPS)
        if len(lrs) < N_STEPS or not all(np.isfinite(lrs)):
            continue

        cum_lr = 0.0
        for h in range(1, N_STEPS + 1):
            cum_lr += lrs[h - 1]
            pred_price = float(np.exp(cur_lp + cum_lr))

            # baseline: сам T_big-пивот h
            actual_big = float(np.exp(full_lp[i + h]))
            pers_big = float(np.exp(full_lp[i + h - 2])) if i + h - 2 >= 0 else np.nan
            long_rows.append((i, h, "BIG", 1, "unbounded", pred_price, actual_big, pers_big))

            after_date = full_conf[i + h]
            date_imm = full_conf[i + h + 1]
            date_ext = full_conf[i + h + 2]

            for level_name, level_lp, level_conf, k_max in [
                ("F", fF_lp, fF_conf, K_F), ("G", fG_lp, fG_conf, K_G),
            ]:
                for window, upper in [
                    ("immediate", date_imm), ("extended", date_ext), ("unbounded", None),
                ]:
                    matches = find_level_matches(level_lp, level_conf, after_date, upper, k_max)
                    for k, idx in matches:
                        actual_lvl = float(np.exp(level_lp[idx]))
                        pers_lvl = float(np.exp(level_lp[idx - 2]))
                        long_rows.append((i, h, level_name, k, window, pred_price, actual_lvl, pers_lvl))

    df = pd.DataFrame(long_rows, columns=["origin", "h", "level", "k", "window",
                                           "pred_price", "actual_price", "pers_price"])
    df["abs_err"] = (df["pred_price"] - df["actual_price"]).abs()
    df["pers_err"] = (df["pers_price"] - df["actual_price"]).abs()
    df.to_csv(RESULTS / "long_table.csv", index=False, float_format="%.5f")
    n_origins = df["origin"].nunique()
    print(f"\nШагов (origin) с полным {N_STEPS}-шаговым прогнозом: {n_origins}")

    # ── rMAE по группам (h, level, k, window) ──
    def group_rmae(g):
        dz = g["pers_err"].mean()
        if len(g) < 5 or dz < 1e-12:
            return np.nan
        return g["abs_err"].mean() / dz

    grp = df.groupby(["h", "level", "k", "window"]).apply(
        lambda g: pd.Series({"n": len(g), "rMAE": group_rmae(g)})
    ).reset_index()
    grp.to_csv(RESULTS / "group_rmae.csv", index=False, float_format="%.5f")

    print("\n=== rMAE по (h, level, k, window) — window='unbounded' ===")
    print(grp[grp["window"] == "unbounded"].to_string(index=False))

    # ── суммарно: среднее rMAE по k внутри (h,level,window) ──
    summary = grp.groupby(["h", "level", "window"])["rMAE"].mean().reset_index()
    summary.to_csv(RESULTS / "summary_by_level.csv", index=False, float_format="%.5f")
    print("\n=== Среднее rMAE по k (суммарно на уровень), window='unbounded' ===")
    print(summary[summary["window"] == "unbounded"].to_string(index=False))

    # ── win-rate: BIG vs лучшая F/G-альтернатива (unbounded), по каждому origin×h ──
    win_rows = []
    sub = df[df["window"] == "unbounded"]
    for (o, h), g in sub.groupby(["origin", "h"]):
        big_row = g[g["level"] == "BIG"]
        alt_rows = g[g["level"] != "BIG"]
        if big_row.empty or alt_rows.empty:
            continue
        big_err = float(big_row["abs_err"].iloc[0])
        best_alt = alt_rows.loc[alt_rows["abs_err"].idxmin()]
        win_rows.append({
            "origin": o, "h": h, "big_err": big_err,
            "best_alt_level": best_alt["level"], "best_alt_k": int(best_alt["k"]),
            "best_alt_err": float(best_alt["abs_err"]),
            "alt_wins": bool(best_alt["abs_err"] < big_err),
        })
    win_df = pd.DataFrame(win_rows)
    win_df.to_csv(RESULTS / "win_rate.csv", index=False, float_format="%.5f")

    print("\n=== Win-rate: доля origin×h, где F/G-альтернатива ближе к прогнозу, чем сам BIG-пивот ===")
    for h in range(1, N_STEPS + 1):
        gh = win_df[win_df["h"] == h]
        if len(gh) == 0:
            continue
        rate = gh["alt_wins"].mean()
        level_counts = gh[gh["alt_wins"]]["best_alt_level"].value_counts().to_dict()
        print(f"  h={h}: n={len(gh)}  win_rate={rate:.3f}  победившие уровни={level_counts}")

    print(f"\nСохранено: {RESULTS}/long_table.csv, group_rmae.csv, summary_by_level.csv, win_rate.csv")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
