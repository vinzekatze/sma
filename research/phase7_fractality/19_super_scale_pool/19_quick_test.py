#!/usr/bin/env python3
"""
19_quick_test.py — Фаза A: быстрый тест гипотезы «пул крупнее» на ТЕКУЩЕЙ
калибровке (эксп.17f: m=3, θ=25.697, T_ratio_frac=0.8987, SBER, D_allpeers).

Идея (пользователь): калиброванная S-map систематически недооценивает
крупные движения (Stage 1g), а три попытки постфактум-коррекции амплитуды
(17i/17k/17l/17o) не нашли эксплуатируемого сигнала. Гипотеза здесь другая:
дело не в статистике готового прогноза, а в нехватке в пуле физических
примеров с большими плечами (T_big-пивотов мало — то же H1 из эксп.17).
Добавляем в пул события КРУПНЕЕ T_big — T_super — свои + кросс-тикерные
(та же корреляционная ранжировка D_allpeers, порог-независимая).

T_super = T_big / T_ratio_super. «Зеркальное» значение T_ratio_super=0.8987
(то же, что T_ratio_frac) математически красиво, но не обязано быть верным —
поэтому здесь сетка, не одна точка.

Дедуп: единый проход по ОБЪЕДИНЁННОМУ (после конкатенации и фильтра
направления) пулу, d_to_query < DUP_EPS — тот же механизм, что в эксп.17,
автоматически покрывает дубли из любого источника (own_big/frac/super,
peers_frac/peers_super). Считаем n_dup с разбивкой по источнику для
диагностики.

m, θ, T_ratio_frac — ФИКСИРОВАНЫ на калибровке эксп.17f (не пересчитываются
здесь). Полная совместная калибровка — Фаза B (19b_calibrator.py).
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

TARGET = "SBER"
T_BIG = 0.20
ARM = "D_allpeers"
M = 3
THETA = 25.697
T_RATIO_FRAC = 0.8987
T_FRAC = T_RATIO_FRAC * T_BIG

# сетка T_ratio_super — от «сильно уродливее зеркала» до «почти T_big»
T_RATIO_SUPER_GRID = [0.60, 0.70, 0.80, 0.8987, 0.95]

MIN_HIST = exp17.MIN_HIST


def build_pool_with_super(target_data, peer_data, rankings, checkpoints, i, full_lp, full_conf,
                            t_frac, t_super):
    """Возвращает (feats_d, tgts_d, cur_lp, actual_price, pers_price, n_dup_by_src) для origin i,
    либо None. t_super=None → без T_super-пула (базовая линия эксп.17f)."""
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

    pool_feats, pool_tgts, pool_dirs, pool_src = [], [], [], []

    def add(lp, dirs, src):
        f, tg, dd = exp17.build_pool_rows(lp, dirs, M)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)
        pool_src.append(np.array([src] * len(tg)))

    big_f, big_tg, big_dd = exp17.build_pool_rows(own_big_lp, own_big_dir, M)
    pool_feats.append(big_f); pool_tgts.append(big_tg); pool_dirs.append(big_dd)
    pool_src.append(np.array(["big"] * len(big_tg)))

    own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
    add(own_frac_lp, own_frac_dir, "frac")
    for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
        p_dates = peer_data[peer]["dates"]
        p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
        if p_cutoff < M + 2:
            continue
        p_lp, _, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                             peer_data[peer]["ll"][:p_cutoff],
                                             p_dates[:p_cutoff], t_frac)
        add(p_lp, p_dir, "frac")

    if t_super is not None:
        own_super_lp, _, own_super_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_super)
        add(own_super_lp, own_super_dir, "super")
        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < M + 2:
                continue
            p_lp, _, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                                 peer_data[peer]["ll"][:p_cutoff],
                                                 p_dates[:p_cutoff], t_super)
            add(p_lp, p_dir, "super")

    feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts)
    dirs = np.concatenate(pool_dirs); src = np.concatenate(pool_src)

    mask = dirs == q_dir
    feats_d, tgts_d, src_d = feats[mask], tgts[mask], src[mask]

    n_dup_by_src = {"big": 0, "frac": 0, "super": 0}
    if len(feats_d):
        d_to_query = np.linalg.norm(feats_d - qvec, axis=1)
        dup_mask = d_to_query < exp17.DUP_EPS
        if dup_mask.any():
            for s in np.unique(src_d[dup_mask]):
                n_dup_by_src[s] = int((src_d[dup_mask] == s).sum())
            feats_d, tgts_d, src_d = feats_d[~dup_mask], tgts_d[~dup_mask], src_d[~dup_mask]

    cur_lp = float(own_big_lp[-1])
    actual_price = float(np.exp(full_lp[i + 1]))
    pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan

    return feats_d, tgts_d, qvec, cur_lp, actual_price, pers_price, n_dup_by_src, dict(
        (s, int((src_d == s).sum())) for s in ["big", "frac", "super"]
    )


def walk_forward(target_data, peer_data, rankings, checkpoints, full_lp, full_conf, t_frac, t_super, label):
    n_big = len(full_lp)
    records = []
    for i in range(MIN_HIST, n_big - 1):
        res = build_pool_with_super(target_data, peer_data, rankings, checkpoints, i, full_lp, full_conf,
                                     t_frac, t_super)
        if res is None:
            continue
        feats_d, tgts_d, qvec, cur_lp, actual_price, pers_price, n_dup_by_src, n_pool_by_src = res
        lr = exp17._smap(qvec, feats_d, tgts_d, M + 2, THETA)
        pers_err = abs(actual_price - pers_price)
        row = {"step": i, "pers_err": pers_err,
               "n_pool": len(tgts_d),
               "n_pool_frac": n_pool_by_src["frac"], "n_pool_super": n_pool_by_src["super"],
               "n_dup_frac": n_dup_by_src["frac"], "n_dup_super": n_dup_by_src["super"]}
        if np.isfinite(lr):
            row["abs_err"] = abs(float(np.exp(cur_lp + lr)) - actual_price)
        else:
            row["abs_err"] = np.nan
        records.append(row)
    df = pd.DataFrame(records)
    if df.empty:
        return {"label": label, "n_steps": 0, "rMAE": np.nan}
    valid = df["abs_err"].dropna()
    dz = df["pers_err"].mean()
    rmae = float(valid.mean() / dz) if len(valid) > 5 and dz > 1e-12 else np.nan
    return {
        "label": label, "n_steps": len(df), "n_valid": len(valid), "rMAE": rmae,
        "n_pool_avg": float(df["n_pool"].mean()),
        "n_pool_frac_avg": float(df["n_pool_frac"].mean()),
        "n_pool_super_avg": float(df["n_pool_super"].mean()),
        "n_dup_frac_avg": float(df["n_dup_frac"].mean()),
        "n_dup_super_avg": float(df["n_dup_super"].mean()),
    }


def main():
    t0 = time.time()
    print("=== 19_quick_test — Фаза A: сетка T_ratio_super, калибровка эксп.17f зафиксирована ===")
    print(f"Target={TARGET} T_BIG={T_BIG} M={M} THETA={THETA} T_RATIO_FRAC={T_RATIO_FRAC}")
    print(f"T_RATIO_SUPER_GRID={T_RATIO_SUPER_GRID}")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, _ = exp17.build_zigzag(lh, ll, dates, T_BIG)
    print(f"Пивотов T_big: {len(full_lp)}")

    rows = []
    print("\n[baseline] без T_super-пула (= эксп.17f)...")
    r = walk_forward(target_data, peer_data, rankings, checkpoints, full_lp, full_conf, T_FRAC, None, "baseline_no_super")
    rows.append(r)
    print(f"  {r}")

    for tr in T_RATIO_SUPER_GRID:
        t_super = T_BIG / tr
        label = f"T_ratio_super={tr:.4f} (T_super={t_super*100:.2f}%)"
        print(f"\n[{label}]...")
        r = walk_forward(target_data, peer_data, rankings, checkpoints, full_lp, full_conf, T_FRAC, t_super, label)
        r["T_ratio_super"] = tr
        r["T_super"] = t_super
        rows.append(r)
        print(f"  {r}")
        pd.DataFrame(rows).to_csv(RESULTS / "phaseA_sweep.csv", index=False, float_format="%.5f")

    print(f"\nСохранено: {RESULTS / 'phaseA_sweep.csv'}")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
