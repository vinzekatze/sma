#!/usr/bin/env python3
"""
25_adaptive_threshold_calibration.py — волатильность-нормированный
(адаптивный) порог зигзага: T[t] = k · vol[t], где vol[t] — каузальное
скользящее std дневных лог-доходностей (окно=60, тот же W, что vol60 в
эксп.19h). Идея пользователя: вместо ФИКСИРОВАННОГО T (гарантирует только
МИНИМУМ размера события, не максимум — размер плеча после подтверждения
может быть каким угодно) — плавающий порог, растягивающийся/сжимающийся
вместе с текущим режимом волатильности, чтобы подтверждённые события
были более ОДНОРОДНЫ по масштабу относительно момента, а не в абсолюте.

⚠️ Не то же самое, что уже проваленная идея "волатильность как ПРИЗНАК/ВЕС
готового прогноза" (эксп.19h, r≈0 с ошибкой) — здесь волатильность
определяет саму НАРЕЗКУ событий (сегментацию), до всякого прогноза,
принципиально другая роль той же величины.

k — калибруется вместо T_big: подбирается под сопоставимое (с эталоном
73 пивота) число событий, дальше обычная калибровка (m,θ,T_ratio) — тот
же T_ratio×k используется как T_frac-уровень (с той же волатильностью).
Пиры считают СВОЮ волатильность, тот же k/T_ratio.
"""
import importlib.util
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
EXP17_DIR = HERE.parents[0] / "17_large_scale_pooled"
EXP23_DIR = HERE.parents[0] / "23_asymmetric_zigzag"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

spec17 = importlib.util.spec_from_file_location("exp17", EXP17_DIR / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

spec23 = importlib.util.spec_from_file_location("exp23", EXP23_DIR / "23_asymmetric_zigzag_calibration.py")
exp23 = importlib.util.module_from_spec(spec23)
spec23.loader.exec_module(exp23)

TARGET = "SBER"
ARM = "D_allpeers"
MIN_HIST = exp17.MIN_HIST
N_TARGET_PIVOTS = 73
VOL_WINDOW = 60

K_SEARCH_GRID = np.round(np.arange(1.0, 12.01, 0.25), 3)

M_GRID = [2, 3, 4, 5]
THETA_LO, THETA_HI, THETA_TOL = 0.0, 32.0, 0.1
TRATIO_LO, TRATIO_HI, TRATIO_TOL = 0.50, 0.95, 0.01
MAX_OUTER = 5

eval_smap_pools = exp23.eval_smap_pools
golden = exp23.golden

_vol_cache = {}
_ticker_cache = {}


def load_ticker_cached(ticker):
    """exp17.load_ticker сам не кэширует — оборачиваем, иначе пиры
    перечитываются с диска на каждом origin (десятки раз за калибровку)."""
    if ticker not in _ticker_cache:
        _ticker_cache[ticker] = exp17.load_ticker(ticker)
    return _ticker_cache[ticker]


def rolling_vol_causal(lc, window):
    ret = np.diff(lc, prepend=lc[0])
    ret[0] = np.nan
    s = pd.Series(ret)
    vol = s.rolling(window, min_periods=max(5, window // 3)).std()
    return vol.to_numpy()


def get_vol(ticker):
    if ticker not in _vol_cache:
        data = load_ticker_cached(ticker)
        _vol_cache[ticker] = rolling_vol_causal(data["lc"], VOL_WINDOW)
    return _vol_cache[ticker]


def build_zigzag_adaptive(lh, ll, dates, vol, k):
    """Тот же алгоритм, что exp17.build_zigzag, но порог T[i]=k*vol[i]
    пересчитывается на каждом баре (каузальная волатильность). Бары без
    валидной vol (разогрев) не могут подтвердить пивот (T=inf)."""
    lp, conf, dirs = [], [], []
    cur = 0
    ext = (lh[0] + ll[0]) / 2.0
    n = len(lh)
    for i in range(n):
        v = vol[i]
        t = k * v if np.isfinite(v) and v > 0 else np.inf
        h, l = lh[i], ll[i]
        if cur == 0:
            if h - ext >= t:
                cur = 1; ext = h
            elif ext - l >= t:
                cur = -1; ext = l
        elif cur == 1:
            if h > ext:
                ext = h
            elif ext - l >= t:
                lp.append(ext); conf.append(dates[i]); dirs.append(1)
                cur = -1; ext = l
        else:
            if l < ext:
                ext = l
            elif h - ext >= t:
                lp.append(ext); conf.append(dates[i]); dirs.append(-1)
                cur = 1; ext = h
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


def pivot_count_adaptive(lh, ll, dates, vol, k):
    lp, _, _ = build_zigzag_adaptive(lh, ll, dates, vol, k)
    return len(lp)


def build_all_pools(rankings, checkpoints, k_big, t_ratio, m):
    data = load_ticker_cached(TARGET)
    lh, ll, dates = data["lh"], data["ll"], data["dates"]
    vol = get_vol(TARGET)
    k_frac = t_ratio * k_big

    full_lp, full_conf, full_dirs = build_zigzag_adaptive(lh, ll, dates, vol, k_big)
    n_big = len(full_lp)

    pools = []
    for i in range(MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        t_lh, t_ll, t_dt, t_vol = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx], vol[:cutoff_idx]

        own_big_lp, _, own_big_dir = build_zigzag_adaptive(t_lh, t_ll, t_dt, t_vol, k_big)
        if len(own_big_lp) < m + 1 or own_big_lp[-1] != full_lp[i]:
            continue
        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue
        q_dir = int(own_big_dir[-1])

        pool_feats, pool_tgts, pool_dirs = [], [], []
        f, tg, dd = exp17.build_pool_rows(own_big_lp, own_big_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        own_frac_lp, _, own_frac_dir = build_zigzag_adaptive(t_lh, t_ll, t_dt, t_vol, k_frac)
        f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            peer_data = load_ticker_cached(peer)
            p_dates = peer_data["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_vol = get_vol(peer)
            p_lp, _, p_dir = build_zigzag_adaptive(peer_data["lh"][:p_cutoff], peer_data["ll"][:p_cutoff],
                                                    p_dates[:p_cutoff], p_vol[:p_cutoff], k_frac)
            f, tg, dd = exp17.build_pool_rows(p_lp, p_dir, m)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts); dirs = np.concatenate(pool_dirs)
        mask = dirs == q_dir
        feats_d, tgts_d = feats[mask], tgts[mask]
        if len(feats_d):
            d = np.linalg.norm(feats_d - qvec, axis=1)
            dup = d < exp17.DUP_EPS
            if dup.any():
                feats_d, tgts_d = feats_d[~dup], tgts_d[~dup]

        cur_lp = float(own_big_lp[-1])
        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)
        pools.append((qvec, feats_d, tgts_d, cur_lp, actual_price, pers_err))
    return pools


def calibrate(rankings, checkpoints, k_big):
    m, theta, T = 3, 5.0, 0.85
    prev = None
    for outer in range(MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            pools = build_all_pools(rankings, checkpoints, k_big, T, mc)
            v, _ = eval_smap_pools(pools, theta, mc + 2)
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m
        pools = build_all_pools(rankings, checkpoints, k_big, T, m)
        theta = golden(lambda th: eval_smap_pools(pools, th, m + 2)[0], THETA_LO, THETA_HI, THETA_TOL)
        T = golden(lambda t: eval_smap_pools(build_all_pools(rankings, checkpoints, k_big, t, m), theta, m + 2)[0],
                   TRATIO_LO, TRATIO_HI, TRATIO_TOL)
        pools = build_all_pools(rankings, checkpoints, k_big, T, m)
        v, n = eval_smap_pools(pools, theta, m + 2)
        print(f"    iter {outer+1}: m={m} θ={theta:.3f} T_ratio={T:.4f} → rMAE={v:.4f} (n={n})")
        cur = (m, round(theta, 2), round(T, 3))
        if cur == prev:
            break
        prev = cur
    return m, theta, T, v, n


def main():
    t0 = time.time()
    print("=== 25_adaptive_threshold_calibration — волатильность-нормированный порог (SBER) ===\n")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != TARGET}
    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    vol = get_vol(TARGET)
    print(f"vol[{VOL_WINDOW}]: median={np.nanmedian(vol):.4f}  mean={np.nanmean(vol):.4f}")

    counts = {k: pivot_count_adaptive(lh, ll, dates, vol, k) for k in K_SEARCH_GRID}
    best_k = min(counts, key=lambda k: abs(counts[k] - N_TARGET_PIVOTS))
    print(f"Подбор k под {N_TARGET_PIVOTS} пивотов (грид {K_SEARCH_GRID[0]}..{K_SEARCH_GRID[-1]}): "
          f"k={best_k} → {counts[best_k]} пивотов\n")

    print(f"--- adaptive: k={best_k} ({counts[best_k]} пивотов) ---")
    t1 = time.time()
    m, theta, T, v, n = calibrate(rankings, checkpoints, best_k)
    el = time.time() - t1
    print(f"  → m={m} θ={theta:.3f} T_ratio={T:.4f} rMAE={v:.4f} (n={n})  ({el:.1f}s)\n")

    results = [
        {"condition": "raw+symmetric (эталон 17f)", "param": 0.20, "n_pivots": 73,
         "m": 3, "theta": 25.697, "T_ratio": 0.8987, "rMAE": 0.6049, "n_eval": 42},
        {"condition": "raw+asymmetric (эксп.23)", "param": 0.21, "n_pivots": 75,
         "m": 4, "theta": 26.340, "T_ratio": 0.8581, "rMAE": 0.5900, "n_eval": 44},
        {"condition": "adaptive (эксп.25)", "param": best_k, "n_pivots": counts[best_k],
         "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4), "rMAE": round(v, 4), "n_eval": n},
    ]
    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "adaptive_threshold.csv", index=False, float_format="%.4f")

    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
