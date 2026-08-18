#!/usr/bin/env python3
"""
19e_tlive_ensemble.py — прогноз крупного, ещё НЕ подтверждённого масштаба
(T_live), усреднённый с обычным T_big-прогнозом.

Идея пользователя: вместо фиксированного (глобального или откалиброванного)
T_super — для каждого origin искать T_live = наибольший порог T', при
котором ТЕКУЩИЙ (ещё не подтверждённый) экстремум причинного зигзага всё
ещё равен цене P текущего T_big-пивота — то есть зигзаг с порогом T' «не
сбросил» эту точку (более раннего, более экстремального движения в ту же
сторону в пределах окна T' не нашлось). T_live существует у КАЖДОГО origin
(не редкое событие) — величина превышения над T_big варьируется.

⚠️ Монотонность НЕ подтвердилась эмпирически (проверено дебагом): при росте
T' совпадение `ext(T')==P` идёт «островами», не монотонно — сразу над T_big
чаще всего НЕ совпадает (зигзаг ещё в другой фазе, не успел развернуться),
затем на каком-то T' начинает совпадать, потом снова перестаёт. Причина —
за сотни баров накапливается много пороговых развилок (где размер хода
почти точно равен порогу); сколь угодно малое изменение T' может
переключить одну из них далеко в истории, меняя всю последующую цепочку
подтверждённых пивотов (эффект бабочки). Поэтому вместо бинарного поиска —
полный скан T' от T_big вверх с фиксированным шагом; T_live = верхняя
граница ПОСЛЕДНЕГО (самого крупного по T') найденного острова совпадения.
Для origin без единого совпадения в диапазоне скана — вырожденный случай,
используется только T_big-прогноз.

Прогноз T_live-масштаба — той же уже откалиброванной методологией (m, θ,
T_ratio_frac=0.8987 из эксп.17f), просто применённой к T_live вместо
T_big: вектор запроса строится из последних m ПОДТВЕРЖДЁННЫХ T_live-плечей
С ДОБАВЛЕНИЕМ P как условно-подтверждённого последнего плеча (P — реально
текущий фронт этого масштаба, просто ещё не подтверждён разворотом).
Обучающий пул (own T_live legs + T_frac_live=0.8987×T_live, свой + кросс-
тикерные пиры D_allpeers) строится обычным образом, БЕЗ искусственного P.

Результат — pred_live (прогноз следующего разворота НА МАСШТАБЕ T_live,
от того же анкера P) усредняется (в лог-доходности) с pred_big (обычный
T_big-прогноз). Сравниваются три rMAE: baseline (T_big only, эксп.17f),
live-only (диагностика), averaged (T_big + T_live).

SBER, T_big=20%, D_allpeers, m=3, θ=25.697, T_ratio_frac=0.8987 — та же
точка, что и везде в этой ветке (19_quick_test/19c).
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

T_HI_CAP = 1.0          # верхняя граница скана T_live (100%)
SCAN_STEP = 0.001
TOL_PRICE = 1e-9

MIN_HIST = exp17.MIN_HIST


def zigzag_running_state(lh, ll, dates, threshold):
    """Финальное (незафиксированное) состояние причинного зигзага: (направление, текущий экстремум)."""
    cur = 0
    ext = (lh[0] + ll[0]) / 2.0
    for i in range(len(lh)):
        if cur == 0:
            if lh[i] - ext >= threshold:
                cur = 1; ext = lh[i]
            elif ext - ll[i] >= threshold:
                cur = -1; ext = ll[i]
        elif cur == 1:
            if lh[i] > ext:
                ext = lh[i]
            elif ext - ll[i] >= threshold:
                cur = -1; ext = ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= threshold:
                cur = 1; ext = lh[i]
    return cur, ext


def find_t_live(lh, ll, dates, t_big, P, q_dir):
    """Полный скан T' от t_big вверх (не монотонно — см. докстринг модуля).
    Возвращает верхнюю границу ПОСЛЕДНЕГО найденного острова совпадения."""
    last_match = None
    t = t_big + SCAN_STEP
    while t <= T_HI_CAP:
        cur, ext = zigzag_running_state(lh, ll, dates, t)
        if cur == q_dir and abs(ext - P) < TOL_PRICE:
            last_match = t
        t += SCAN_STEP
    if last_match is None:
        return t_big, False
    return last_match, True


def build_live_pool_and_query(target_data, peer_data, rankings, checkpoints, t_lh, t_ll, t_dt,
                               confirm_date, t_live, m, P, q_dir):
    own_live_lp, _, own_live_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_live)
    if len(own_live_lp) < m:
        return None

    extended = np.concatenate([own_live_lp, [P]])
    qvec_live = np.array([extended[-1 - lag] - extended[-2 - lag] for lag in range(m)])
    if not np.all(np.isfinite(qvec_live)):
        return None

    t_frac_live = T_RATIO_FRAC * t_live
    pool_feats, pool_tgts, pool_dirs = [], [], []

    def add(lp, dirs):
        f, tg, dd = exp17.build_pool_rows(lp, dirs, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

    add(own_live_lp, own_live_dir)

    own_frac_live_lp, _, own_frac_live_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac_live)
    add(own_frac_live_lp, own_frac_live_dir)
    for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
        p_dates = peer_data[peer]["dates"]
        p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
        if p_cutoff < m + 2:
            continue
        p_lp, _, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                             peer_data[peer]["ll"][:p_cutoff],
                                             p_dates[:p_cutoff], t_frac_live)
        add(p_lp, p_dir)

    feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts); dirs = np.concatenate(pool_dirs)
    mask = dirs == q_dir
    feats_d, tgts_d = feats[mask], tgts[mask]

    if len(feats_d):
        d = np.linalg.norm(feats_d - qvec_live, axis=1)
        dup_mask = d < exp17.DUP_EPS
        if dup_mask.any():
            feats_d, tgts_d = feats_d[~dup_mask], tgts_d[~dup_mask]

    return qvec_live, feats_d, tgts_d


def main():
    t0 = time.time()
    print("=== 19e_tlive_ensemble — прогноз T_live (плавающий, per-origin) + усреднение с T_big ===")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, full_dirs = exp17.build_zigzag(lh, ll, dates, T_BIG)
    n_big = len(full_lp)
    T_FRAC = T_RATIO_FRAC * T_BIG

    records = []
    for i in range(MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]

        own_big_lp, _, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, T_BIG)
        if len(own_big_lp) < M + 1 or own_big_lp[-1] != full_lp[i]:
            continue
        qvec_big = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(M)])
        if not np.all(np.isfinite(qvec_big)):
            continue
        q_dir = int(own_big_dir[-1])
        P = float(own_big_lp[-1])

        # ── обычный T_big-пул (как в эксп.17f) ──
        pool_feats, pool_tgts, pool_dirs = [], [], []
        big_f, big_tg, big_dd = exp17.build_pool_rows(own_big_lp, own_big_dir, M)
        pool_feats.append(big_f); pool_tgts.append(big_tg); pool_dirs.append(big_dd)
        own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, T_FRAC)
        f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, M)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)
        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < M + 2:
                continue
            p_lp, _, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                                 peer_data[peer]["ll"][:p_cutoff],
                                                 p_dates[:p_cutoff], T_FRAC)
            f, tg, dd = exp17.build_pool_rows(p_lp, p_dir, M)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)
        feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts); dirs = np.concatenate(pool_dirs)
        mask = dirs == q_dir
        feats_d, tgts_d = feats[mask], tgts[mask]
        if len(feats_d):
            dq = np.linalg.norm(feats_d - qvec_big, axis=1)
            dup = dq < exp17.DUP_EPS
            if dup.any():
                feats_d, tgts_d = feats_d[~dup], tgts_d[~dup]
        lr_big = exp17._smap(qvec_big, feats_d, tgts_d, M + 2, THETA)

        # ── T_live: бинарный поиск + прогноз ──
        t_live, found = find_t_live(t_lh, t_ll, t_dt, T_BIG, P, q_dir)
        lr_live = np.nan
        if found and t_live > T_BIG + 1e-4:
            res = build_live_pool_and_query(target_data, peer_data, rankings, checkpoints,
                                             t_lh, t_ll, t_dt, confirm_date, t_live, M, P, q_dir)
            if res is not None:
                qvec_live, feats_live, tgts_live = res
                lr_live = exp17._smap(qvec_live, feats_live, tgts_live, M + 2, THETA)

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)

        row = {"step": i, "t_live": t_live, "t_live_excess": t_live - T_BIG, "pers_err": pers_err}
        if np.isfinite(lr_big):
            row["abs_err_big"] = abs(float(np.exp(P + lr_big)) - actual_price)
        else:
            row["abs_err_big"] = np.nan
        if np.isfinite(lr_live):
            row["abs_err_live"] = abs(float(np.exp(P + lr_live)) - actual_price)
            lr_avg = (lr_big + lr_live) / 2.0
            row["abs_err_avg"] = abs(float(np.exp(P + lr_avg)) - actual_price)
        else:
            row["abs_err_live"] = np.nan
            row["abs_err_avg"] = row["abs_err_big"]  # нет T_live-компоненты — среднее = сам T_big
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(RESULTS / "tlive_ensemble.csv", index=False, float_format="%.6f")

    dz = df["pers_err"].mean()
    print(f"\nn_origins={len(df)}   mean t_live_excess={df['t_live_excess'].mean():.4f}  "
          f"(median={df['t_live_excess'].median():.4f}, max={df['t_live_excess'].max():.4f})")
    print(f"n с найденным T_live > T_big: {df['abs_err_live'].notna().sum()} / {len(df)}")

    for col, label in [("abs_err_big", "baseline (T_big only)"),
                        ("abs_err_live", "T_live only (диагностика, где найден)"),
                        ("abs_err_avg", "усреднённый (T_big + T_live)")]:
        valid = df[col].dropna()
        rmae = float(valid.mean() / dz) if len(valid) > 3 else np.nan
        print(f"  {label:<40s} rMAE={rmae:.4f}  (n={len(valid)})")

    print(f"\nСохранено: {RESULTS / 'tlive_ensemble.csv'}")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
