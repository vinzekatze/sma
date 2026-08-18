#!/usr/bin/env python3
"""
17p_nn_approximator.py — небольшой MLP вместо локальной аппроксимации.

Мотивация пользователя: размерность признаков маленькая (m=3, +направление
=4), сеть может быть мелкой (2-3 слоя) — должно быть дёшево. В отличие от
LA0/LWR/Simplex/S-map (все — локальные/взвешенные методы вокруг запроса),
MLP — глобальная нелинейная регрессия по всему пулу; в отличие от RBF (17j,
взорвался на плотном пуле из-за вырождения грам-матрицы), MLP не требует
обращения матрицы размера пула — обучение по мини-батчам, устойчивее.

Направление объединено в ОДНУ сеть (не раздельно, как в остальных методах):
направление подаётся как признак (+1/-1), что даёт вдвое больше обучающих
примеров на каждом шаге — прямая попытка закрыть проблему "мало данных",
которая топила коррекции весь день (17i/17k/17l).

Каузально: сеть переобучается заново на каждом шаге walk-forward, только на
пуле, доступном к этому моменту (тот же контракт, что и везде в этом
эксперименте).

Точка: SBER, T_big=20%, ratio=0.85, arm=D_allpeers (сравнение с 17d/17e/17f).
"""
import importlib.util
import time
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
spec17 = importlib.util.spec_from_file_location("exp17", HERE / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGET = "SBER"
T_BIG = 0.20
RATIO = 0.85
ARM = "D_allpeers"
M = 3
HIDDEN = (16, 8)   # 2 скрытых слоя — "2-3 шага" из просьбы
ALPHA_REG = 0.01   # L2-регуляризация посильнее — данных немного
SEED = 42


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


def run_config(target_data, peer_data, rankings, checkpoints, full_lp, full_conf, n_big, t_frac, hidden, alpha_reg):
    t0 = time.time()
    errs, dz = [], []
    for i in range(exp17.MIN_HIST, n_big - 1):
        res = build_unfiltered_pool(target_data, peer_data, rankings, checkpoints, T_BIG, M, t_frac, ARM, i, full_lp, full_conf)
        if res is None:
            continue
        feats, tgts, dirs, qvec, q_dir, cur_lp = res

        X = np.column_stack([feats, dirs.astype(float)])
        y = tgts
        if len(y) < 30:
            continue

        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        Xq = scaler.transform(np.concatenate([qvec, [q_dir]]).reshape(1, -1))

        mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", alpha=alpha_reg,
                           max_iter=2000, random_state=SEED, early_stopping=True,
                           validation_fraction=0.15, n_iter_no_change=15)
        mlp.fit(Xs, y)
        lr = float(mlp.predict(Xq)[0])

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1]))
        pred_price = float(np.exp(cur_lp + lr))
        errs.append(abs(pred_price - actual_price))
        dz.append(abs(pers_price - actual_price))

    elapsed = time.time() - t0
    rmae = float(np.mean(errs) / np.mean(dz))
    return rmae, len(errs), elapsed


def main():
    print("=== 17p_nn_approximator: сетка архитектур/регуляризации ===")
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

    HIDDEN_GRID = [(8,), (16,), (16, 8), (32, 16), (8, 4)]
    ALPHA_GRID = [0.01, 0.1, 1.0, 3.0]

    results = []
    for hidden in HIDDEN_GRID:
        for alpha_reg in ALPHA_GRID:
            rmae, n, elapsed = run_config(target_data, peer_data, rankings, checkpoints, full_lp, full_conf, n_big, t_frac, hidden, alpha_reg)
            results.append((hidden, alpha_reg, rmae, n, elapsed))
            print(f"  hidden={str(hidden):<12} alpha={alpha_reg:<5} rMAE={rmae:.4f}  n={n}  ({elapsed:.1f}s)")

    best = min(results, key=lambda r: r[2])
    print(f"\nЛучшая конфигурация: hidden={best[0]} alpha={best[1]} rMAE={best[2]:.4f}")

    print("\nРеференс (та же точка, T_big=20% ratio=0.85 D_allpeers):")
    print("  LA0_K50=0.6432  LWR_K30=0.6389  Simplex_K4=0.6630  Smap(θ=1)=0.6738  Smap_калиброванный(θ=25.7)=0.6049")


if __name__ == "__main__":
    main()
