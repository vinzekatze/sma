#!/usr/bin/env python3
"""
21_ssa_zigzag_calibration.py — сравнение предсказуемости "настоящих" (SSA-
очищенных) разворотов против "шумных" (сырых) — переформулировка эксперимента
из research/phase6_attractor/zigzag_event_time (там T фиксировали одинаковым
для no_ssa/k=2/k=3 → SSA-условия теряли 35% пивотов, числа rMAE к тому же
были до исправления формулы persistence — см. feedback_rmae_persistence.md).

Вопрос НЕ "ухудшает или улучшает SSA точность" (ухудшит однозначно относительно
обычных пивотов) — а: какие пивоты мы ЛУЧШЕ прогнозируем В ПРИНЦИПЕ, если
сравнивать КАЖДОЕ условие с его же persistence (не абсолютные числа друг с
другом)? Если SSA-очищенные пивоты дают БОЛЬШЕЕ относительное улучшение —
структура сильнее у "настоящих" разворотов. Если сырые не хуже — мелкие
движения тоже несут информацию (шума в чистом виде может не быть вообще).

Три условия (SBER):
  raw    — log((open+close)/2), без сглаживания
  ssa_k1 — то же + каузальная скользящая SSA (W=256, L=8, k=1)
  ssa_k2 — то же, k=2

Зигзаг строится на ОДНОЙ (не high/low, как в остальной фазе 7) серии —
midprice после усреднения O/C уже не имеет отдельного high/low. Для
честности raw и SSA-условия используют ОДИНАКОВУЮ (midprice) базу зигзага,
отличаются только наличием сглаживания — иначе сравнение спутало бы эффект
SSA с эффектом перехода от high/low к midprice.

midprice-зигзаг (даже без SSA) даёт заметно меньше пивотов, чем high/low-зигзаг
той же фазы 7 при том же T (нет доступа к внутрибарному диапазону) — поэтому
T_big подбирается ЗАНОВО для ВСЕХ ТРЁХ условий (включая raw) под общий
ориентир N_TARGET=73 пивота (масштаб, привычный по всей сессии эксп.17-20,
даёт разумное n после MIN_HIST=30). После подбора T_big — для КАЖДОГО
условия свой покоординатный спуск (m,θ,T_ratio), тот же алгоритм, что
17f_calibrator.py.

Кросс-тикерный пул (D_allpeers, 45 тикеров) строится тем же способом —
пиры проходят ТУ ЖЕ предобработку (midprice/SSA), что и целевой тикер, для
консистентности условия. Отбор пиров (кто доступен на дату) — по обычной
корреляции ЗАКРЫТИЯ (не зависит от предобработки зигзага, эксп.17,
compute_peer_rankings — переиспускается без изменений).
"""
import importlib.util
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
EXP17_DIR = HERE.parents[0] / "17_large_scale_pooled"
DATA = HERE.parents[2] / "data" / "candles"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

spec17 = importlib.util.spec_from_file_location("exp17", EXP17_DIR / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

TARGET = "SBER"
ARM = "D_allpeers"
SSA_W, SSA_L = 256, 8
SSA_KS = (1, 2)

M_GRID = [2, 3, 4, 5]
THETA_LO, THETA_HI, THETA_TOL = 0.0, 32.0, 0.1
T_LO, T_HI, T_TOL = 0.50, 0.95, 0.01
MAX_OUTER = 5
DEF_M, DEF_THETA, DEF_T_RATIO = 3, 5.0, 0.85
MIN_HIST = 30

TBIG_SEARCH_GRID = np.round(np.arange(0.06, 0.31, 0.01), 3)


# ── загрузка + предобработка (midprice → log → [SSA]) ───────────────────────

_raw_cache = {}


def load_midprice_log(ticker):
    if ticker not in _raw_cache:
        raw = json.load(open(DATA / ticker / "1d.json"))
        open_ = np.array([c["open"] for c in raw], dtype=np.float64)
        close = np.array([c["close"] for c in raw], dtype=np.float64)
        open_ = np.where(open_ <= 0, np.nan, open_)
        close = np.where(close <= 0, np.nan, close)
        mid = (open_ + close) / 2.0
        dates = np.array([c["begin"] for c in raw])
        log_mid = np.log(mid)
        # безсделочные дни (close/open<=0, напр. KZOS 2013-05-21) дают NaN — каузально
        # протягиваем последнее валидное значение вперёд, иначе NaN ломает SVD в SSA
        # (LinAlgError: SVD did not converge на матрице Ханкеля с NaN).
        log_mid = pd.Series(log_mid).ffill().bfill().to_numpy()
        _raw_cache[ticker] = (log_mid, dates)
    return _raw_cache[ticker]


_ssa_cache = {}


def ssa_smooth_multi_k(ticker, log_price, W, L, ks):
    """Каузальная скользящая SSA: на каждом t — SVD траекторной матрицы
    из log_price[t-W+1..t], реконструкция k компонентами, берётся только
    последний элемент (Xr[K_m-1, L-1]). Общий SVD на оба k за один проход.
    Для t < W-1 — без изменений (warmup)."""
    key = ticker
    if key in _ssa_cache:
        return _ssa_cache[key]
    N = len(log_price)
    K_m = W - L + 1
    out = {k: log_price.copy() for k in ks}
    if K_m < 2 or N < W:
        _ssa_cache[key] = out
        return out
    idx = np.arange(K_m)[:, None] + np.arange(L)[None, :]
    max_k = max(ks)
    for t in range(W - 1, N):
        w = log_price[t - W + 1: t + 1]
        X = w[idx]
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        for k in ks:
            nk = min(k, len(s))
            out[k][t] = float(((U[:, :nk] * s[:nk]) @ Vt[:nk, :])[K_m - 1, L - 1])
    _ssa_cache[key] = out
    return out


def get_condition_series(ticker, condition):
    log_price, dates = load_midprice_log(ticker)
    if condition == "raw":
        return log_price, dates
    k = {"ssa_k1": 1, "ssa_k2": 2}[condition]
    smoothed = ssa_smooth_multi_k(ticker, log_price, SSA_W, SSA_L, SSA_KS)
    return smoothed[k], dates


# ── зигзаг на ОДНОЙ серии (не high/low) ──────────────────────────────────────

def build_zigzag_single(log_price, dates, threshold):
    lp, conf, dirs = [], [], []
    cur = 0
    ext = log_price[0]
    for i in range(len(log_price)):
        v = log_price[i]
        if not np.isfinite(v):
            continue
        if cur == 0:
            if v - ext >= threshold:
                cur = 1; ext = v
            elif ext - v >= threshold:
                cur = -1; ext = v
        elif cur == 1:
            if v > ext:
                ext = v
            elif ext - v >= threshold:
                lp.append(ext); conf.append(dates[i]); dirs.append(1)
                cur = -1; ext = v
        else:
            if v < ext:
                ext = v
            elif v - ext >= threshold:
                lp.append(ext); conf.append(dates[i]); dirs.append(-1)
                cur = 1; ext = v
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


# ── подбор T_big под сопоставимое число пивотов ──────────────────────────────

def pivot_count(ticker, condition, t_big):
    log_price, dates = get_condition_series(ticker, condition)
    lp, conf, dirs = build_zigzag_single(log_price, dates, t_big)
    return len(lp)


def find_tbig_for_target_count(ticker, condition, target_n):
    counts = {t: pivot_count(ticker, condition, t) for t in TBIG_SEARCH_GRID}
    best_t = min(counts, key=lambda t: abs(counts[t] - target_n))
    return best_t, counts


# ── пул (own + кросс-тикер), покоординатный спуск (m,θ,T_ratio) ─────────────

def build_all_pools(target_ticker, peer_tickers, rankings, checkpoints, condition, t_big, m, t_frac):
    log_price, dates = get_condition_series(target_ticker, condition)
    full_lp, full_conf, full_dirs = build_zigzag_single(log_price, dates, t_big)
    n_big = len(full_lp)

    pools = []
    for i in range(MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        t_lp_c, t_dt_c = log_price[:cutoff_idx], dates[:cutoff_idx]

        own_big_lp, _, own_big_dir = build_zigzag_single(t_lp_c, t_dt_c, t_big)
        if len(own_big_lp) < m + 1 or own_big_lp[-1] != full_lp[i]:
            continue
        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue
        q_dir = int(own_big_dir[-1])

        pool_feats, pool_tgts, pool_dirs = [], [], []
        f, tg, dd = exp17.build_pool_rows(own_big_lp, own_big_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        own_frac_lp, _, own_frac_dir = build_zigzag_single(t_lp_c, t_dt_c, t_frac)
        f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            if peer not in peer_tickers:
                continue
            p_log_price, p_dates = get_condition_series(peer, condition)
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_lp, _, p_dir = build_zigzag_single(p_log_price[:p_cutoff], p_dates[:p_cutoff], t_frac)
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


def eval_smap_pools(pools, theta, min_pool):
    errs, dz = [], []
    for qvec, feats, tgts, cur_lp, actual_price, pers_err in pools:
        lr = exp17._smap(qvec, feats, tgts, min_pool, theta)
        if np.isfinite(lr):
            errs.append(abs(float(np.exp(cur_lp + lr)) - actual_price)); dz.append(pers_err)
    if len(errs) < 5:
        return float("inf"), 0
    return float(np.mean(errs) / np.mean(dz)), len(errs)


def golden(func, a, b, tol, max_iter=50):
    phi = (5 ** 0.5 - 1) / 2
    x1 = b - phi * (b - a); x2 = a + phi * (b - a)
    f1 = func(x1); f2 = func(x2)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a); f1 = func(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a); f2 = func(x2)
    return (a + b) / 2


def calibrate(target_ticker, peer_tickers, rankings, checkpoints, condition, t_big):
    m, theta, T = DEF_M, DEF_THETA, DEF_T_RATIO
    trace, prev = [], None
    for outer in range(MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            pools = build_all_pools(target_ticker, peer_tickers, rankings, checkpoints, condition, t_big, mc, T * t_big)
            v, _ = eval_smap_pools(pools, theta, mc + 2)
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m

        pools = build_all_pools(target_ticker, peer_tickers, rankings, checkpoints, condition, t_big, m, T * t_big)
        theta = golden(lambda th: eval_smap_pools(pools, th, m + 2)[0], THETA_LO, THETA_HI, THETA_TOL)

        T = golden(lambda t: eval_smap_pools(
            build_all_pools(target_ticker, peer_tickers, rankings, checkpoints, condition, t_big, m, t * t_big),
            theta, m + 2)[0], T_LO, T_HI, T_TOL)

        pools = build_all_pools(target_ticker, peer_tickers, rankings, checkpoints, condition, t_big, m, T * t_big)
        v, n = eval_smap_pools(pools, theta, m + 2)
        trace.append({"iter": outer + 1, "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4), "rMAE": round(v, 4)})
        print(f"      iter {outer+1}: m={m} θ={theta:.3f} T_ratio={T:.4f} → rMAE={v:.4f} (n={n})")
        cur = (m, round(theta, 2), round(T, 3))
        if cur == prev:
            break
        prev = cur
    return m, theta, T, v, n, trace


def main():
    t0 = time.time()
    print("=== 21_ssa_zigzag_calibration — raw vs SSA(k=1,2), SBER ===\n")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != TARGET}
    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)
    peer_tickers = set(exp17.PEERS)

    # N_TARGET=73 — привычный для этой сессии размер (high/low-зигзаг SBER@T_big=20%,
    # весь эксп.17-20 работал с этим порядком n). midprice-зигзаг на 0.20 даёт лишь 39 —
    # T_big для ВСЕХ трёх условий (включая raw) подбирается заново под этот ориентир,
    # чтобы после MIN_HIST=30 оставалось разумное число шагов оценки (не n=8).
    N_TARGET = 73
    conditions = {}
    for cond in ["raw", "ssa_k1", "ssa_k2"]:
        best_t, counts = find_tbig_for_target_count(TARGET, cond, N_TARGET)
        conditions[cond] = best_t
        print(f"{cond}: подобран T_big={best_t} → {counts[best_t]} пивотов "
              f"(цель {N_TARGET}, грид {TBIG_SEARCH_GRID[0]}..{TBIG_SEARCH_GRID[-1]})")

    print()
    results = []
    for cond, t_big in conditions.items():
        n_piv = pivot_count(TARGET, cond, t_big)
        print(f"--- {cond}: T_big={t_big} ({n_piv} пивотов) ---")
        t1 = time.time()
        m, theta, T, v, n, trace = calibrate(TARGET, peer_tickers, rankings, checkpoints, cond, t_big)
        elapsed = time.time() - t1
        print(f"  → m={m} θ={theta:.3f} T_ratio={T:.4f} rMAE={v:.4f} (n={n})  ({elapsed:.1f}s)\n")
        results.append({"condition": cond, "T_big": t_big, "n_pivots": n_piv,
                        "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4),
                        "rMAE": round(v, 4), "n_eval": n, "elapsed_s": round(elapsed, 1)})
        pd.DataFrame(results).to_csv(RESULTS / "ssa_comparison.csv", index=False, float_format="%.4f")

    df = pd.DataFrame(results)
    print(f"{'='*90}")
    print(df.to_string(index=False))
    print(f"{'='*90}")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
