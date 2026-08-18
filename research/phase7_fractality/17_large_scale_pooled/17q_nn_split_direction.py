#!/usr/bin/env python3
"""
17q_nn_split_direction.py — MLP отдельно на UP-> LOW и DOWN->HIGH (как
LA0/LWR/Simplex/S-map устроены с самого начала), вместо одной сети на оба
направления с направлением как признаком (17p).

Последний тест в этой серии — раздельная калибровка (S-map) отклонена как
рискованная на малых n (Stage 1 памяти), но для MLP это дёшево проверить
(секунды), раз уж 17p уже показал, что MLP жизнеспособен по скорости.
"""
import importlib.util
import time
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
specp = importlib.util.spec_from_file_location("nnp", HERE / "17p_nn_approximator.py")
nnp = importlib.util.module_from_spec(specp)
specp.loader.exec_module(nnp)

exp17 = nnp.exp17
TARGET, T_BIG, RATIO, ARM, M, SEED = nnp.TARGET, nnp.T_BIG, nnp.RATIO, nnp.ARM, nnp.M, nnp.SEED


def run_split(target_data, peer_data, rankings, checkpoints, full_lp, full_conf, n_big, t_frac, hidden, alpha_reg, min_n):
    t0 = time.time()
    errs, dz = [], []
    errs_up, dz_up, errs_dn, dz_dn = [], [], [], []
    skipped = 0
    for i in range(exp17.MIN_HIST, n_big - 1):
        res = nnp.build_unfiltered_pool(target_data, peer_data, rankings, checkpoints, T_BIG, M, t_frac, ARM, i, full_lp, full_conf)
        if res is None:
            continue
        feats, tgts, dirs, qvec, q_dir, cur_lp = res

        mask = dirs == q_dir
        X = feats[mask]; y = tgts[mask]
        if len(y) < min_n:
            skipped += 1
            continue

        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        Xq = scaler.transform(qvec.reshape(1, -1))

        mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", alpha=alpha_reg,
                           max_iter=2000, random_state=SEED, early_stopping=True,
                           validation_fraction=0.15, n_iter_no_change=15)
        mlp.fit(Xs, y)
        lr = float(mlp.predict(Xq)[0])

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1]))
        pred_price = float(np.exp(cur_lp + lr))
        e = abs(pred_price - actual_price); d = abs(pers_price - actual_price)
        errs.append(e); dz.append(d)
        if q_dir == 1:
            errs_up.append(e); dz_up.append(d)
        else:
            errs_dn.append(e); dz_dn.append(d)

    elapsed = time.time() - t0
    rmae = float(np.mean(errs) / np.mean(dz)) if errs else float("nan")
    rmae_up = float(np.mean(errs_up) / np.mean(dz_up)) if errs_up else float("nan")
    rmae_dn = float(np.mean(errs_dn) / np.mean(dz_dn)) if errs_dn else float("nan")
    return rmae, rmae_up, rmae_dn, len(errs), len(errs_up), len(errs_dn), skipped, elapsed


def main():
    print("=== 17q_nn_split_direction ===")
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

    CONFIGS = [
        ((16, 8), 3.0, 15),
        ((16, 8), 1.0, 15),
        ((8,), 1.0, 15),
        ((8,), 3.0, 10),
    ]
    for hidden, alpha_reg, min_n in CONFIGS:
        rmae, rmae_up, rmae_dn, n, n_up, n_dn, skipped, elapsed = run_split(
            target_data, peer_data, rankings, checkpoints, full_lp, full_conf, n_big, t_frac, hidden, alpha_reg, min_n)
        print(f"hidden={str(hidden):<10} alpha={alpha_reg:<4} min_n={min_n:<3} "
              f"rMAE={rmae:.4f}  UP(n={n_up})={rmae_up:.4f}  DOWN(n={n_dn})={rmae_dn:.4f}  "
              f"пропущено(мало данных)={skipped}  ({elapsed:.1f}s)")

    print("\nРеференс:")
    print("  17p (объединённый MLP, лучший): 0.6635")
    print("  Калиброванный S-map по направлениям: UP=0.5696 DOWN=0.6256 (общий 0.6049)")
    print("  LA0_K50=0.6432  LWR_K30=0.6389  Simplex_K4=0.6630")


if __name__ == "__main__":
    main()
