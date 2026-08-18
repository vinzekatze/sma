#!/usr/bin/env python3
"""
17_large_scale_pooled.py — Stage 1: крупномасштабные зигзаг-события ("месячный"
масштаб) + кросс-тикерный фрактальный пул.

Гипотезы
────────
H1: на T_big ≈ 10-25% (1d) прогноз крупных пивотов улучшается от добавления
    фрактального пула мелких (T_frac) событий того же тикера — как и на малых T
    в фазе 7.
H2: добавление в пул T_frac-событий коррелирующих с целевым тикером пиров даёт
    улучшение сверх собственного фрактального пула, потому что на большом T
    у одного тикера физически мало пивотов.
H3 (контроль): улучшение от H2 объясняется именно корреляцией, а не просто
    увеличением объёма пула — проверяется сравнением top-N коррелирующих пиров
    против N случайных пиров того же размера.

Каузальный контракт
────────────────────
Целевой ряд (SBER) и КАЖДЫЙ пир обрезаются физическим срезом массивов
(`arr[:cutoff_idx]`) на каждом шаге walk-forward, ДО построения зигзага —
паттерн `trim_to_origin` из research/reference/lwr_ref.py, smap_ref.py.
Никаких масок/флагов по дате внутри цикла построения пула.

Единственное исключение — сам список шагов walk-forward и "правильный ответ"
для метрики: они берутся из ОДНОГО полного (не обрезанного) зигзага целевого
тикера, построенного один раз в начале. Это не утечка в модель: полный ряд
используется только (а) чтобы перечислить моменты, где T_big-пивот уже
подтверждён, и (б) чтобы после прогноза сравнить его с фактической ценой
следующего пивота — сам прогноз всегда считается на заново обрезанных данных.

Отбор коррелирующих пиров — тоже физический срез: на каждой годовой
контрольной точке возвраты пира и целевого тикера берутся ТОЛЬКО из среза
`returns[:idx_на_тот_момент]`, ранжирование фиксируется на год вперёд.

Данные: data/candles/{TICKER}/1d.json (MOEX ISS, закрытые бары).
"""
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parents[2] / "data" / "candles"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# ── Параметры ────────────────────────────────────────────────────────────────
TARGET = "SBER"
UNIVERSE = [
    "SBER", "LKOH", "CHMF", "NVTK", "MGNT", "VTBR", "NLMK", "MRKP", "GAZP", "PLZL",
    "ROSN", "TATN", "MTSS", "ALRS", "MOEX", "SNGS", "IRAO", "RUAL", "MAGN", "PHOR",
    "AFLT", "HYDR", "SIBN", "TRNFP", "RTKM",
    # второй эшелон, добавлено 2026-07-01
    "BANE", "BANEP", "MTLR", "MTLRP", "RASP", "VSMO", "KMAZ", "AKRN", "MSNG", "TGKA",
    "UPRO", "PIKK", "MVID", "LSRG", "CBOM", "GCHE", "SVAV", "FESH", "KZOS", "NKNC",
]
PEERS = [t for t in UNIVERSE if t != TARGET]

T_BIG_GRID = [0.10, 0.15, 0.20, 0.25]
RATIO_GRID = [0.15, 0.30, 0.50, 0.70, 0.85]
ARMS = ["A_baseline", "B_own_frac", "C_top8", "D_allpeers", "E_random8"]

M          = 3       # глубина вложения (число лаговых лог-диффов)
K_LWR      = 50       # число соседей LA0/LWR
THETA_SMAP = 1.0       # bandwidth S-map
MIN_HIST   = 30       # минимум собственных T_big пивотов до начала оценки

CORR_WINDOW_DAYS  = 756   # ~3 торговых года
MIN_CORR_HIST_DAYS = 250  # минимум дней до первой оценки корреляции
N_TOP     = 8
N_RANDOM  = 8
RANDOM_SEED = 42

MIN_POOL_SMAP = M + 2
DUP_EPS = 1e-9   # порог для удаления из пула событий, совпадающих с запросом


# ── 1. Загрузка ──────────────────────────────────────────────────────────────

def load_ticker(ticker):
    path = DATA / ticker / "1d.json"
    raw = json.load(open(path))
    high  = np.array([c["high"]  for c in raw], dtype=np.float64)
    low   = np.array([c["low"]   for c in raw], dtype=np.float64)
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    # MOEX ISS иногда отдаёт close=0 на безсделочных днях (напр. KZOS 2013-05-21) —
    # такой день не несёт ценовой информации, гасим в NaN вместо log(0)=-inf.
    # build_zigzag сравнивает через >=, NaN даёт False на обеих ветках — день
    # молча пропускается, ext не портится. compute_peer_rankings берёт diff(lc):
    # окна корреляции, задевающие NaN, корректно уходят в nan и уже отфильтровываются
    # проверкой np.isfinite(c).
    high  = np.where(high  <= 0, np.nan, high)
    low   = np.where(low   <= 0, np.nan, low)
    close = np.where(close <= 0, np.nan, close)
    lh = np.log(high)
    ll = np.log(low)
    lc = np.log(close)
    dt = np.array([c["begin"] for c in raw])
    return {"lh": lh, "ll": ll, "lc": lc, "dates": dt}


# ── 2. Зигзаг (каузальный по построению) ─────────────────────────────────────

def build_zigzag(lh, ll, dates, thr):
    lp, conf, dirs = [], [], []
    cur = 0
    ext = (lh[0] + ll[0]) / 2.0
    for i in range(len(lh)):
        if cur == 0:
            if lh[i] - ext >= thr:
                cur = 1; ext = lh[i]
            elif ext - ll[i] >= thr:
                cur = -1; ext = ll[i]
        elif cur == 1:
            if lh[i] > ext:
                ext = lh[i]
            elif ext - ll[i] >= thr:
                lp.append(ext); conf.append(dates[i]); dirs.append(1)
                cur = -1; ext = ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); conf.append(dates[i]); dirs.append(-1)
                cur = 1; ext = lh[i]
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


def build_pool_rows(lp, dirs, m):
    """feats[j], target[j] = следующая лог-доходность, dir[j] — направление пивота j."""
    n = len(lp)
    if n < m + 2:
        return np.empty((0, m)), np.empty(0), np.empty(0, dtype=np.int8)
    idx = np.arange(m, n - 1)
    feats = np.zeros((len(idx), m))
    tgts  = np.zeros(len(idx))
    ds    = np.zeros(len(idx), dtype=np.int8)
    for row, j in enumerate(idx):
        for lag in range(m):
            feats[row, lag] = lp[j - lag] - lp[j - lag - 1]
        tgts[row] = lp[j + 1] - lp[j]
        ds[row]   = dirs[j]
    finite = np.all(np.isfinite(feats), axis=1) & np.isfinite(tgts)
    return feats[finite], tgts[finite], ds[finite]


# ── 3. Отбор коррелирующих пиров (каузально, физическим срезом) ──────────────

def compute_peer_rankings(target_data, peer_data, checkpoints):
    t_dates, t_lc = target_data["dates"], target_data["lc"]
    rankings = {}
    for ci, cp in enumerate(checkpoints):
        cutoff = int(np.searchsorted(t_dates, cp, side="left"))
        if cutoff < MIN_CORR_HIST_DAYS + 1:
            rankings[cp] = {"top8": [], "all": [], "random8": []}
            continue
        t_dates_slice = t_dates[:cutoff][-CORR_WINDOW_DAYS:]
        t_lc_slice    = t_lc[:cutoff][-CORR_WINDOW_DAYS:]
        t_ret         = np.diff(t_lc_slice)
        t_ret_dates   = t_dates_slice[1:]

        corrs = {}
        for peer in PEERS:
            p_dates, p_lc = peer_data[peer]["dates"], peer_data[peer]["lc"]
            p_cutoff = int(np.searchsorted(p_dates, cp, side="left"))
            if p_cutoff < 2:
                continue
            p_dates_slice = p_dates[:p_cutoff]
            p_lc_slice    = p_lc[:p_cutoff]
            p_ret         = np.diff(p_lc_slice)
            p_ret_dates   = p_dates_slice[1:]

            common, ti, pi = np.intersect1d(t_ret_dates, p_ret_dates, return_indices=True)
            if len(common) < 100:
                continue
            c = np.corrcoef(t_ret[ti], p_ret[pi])[0, 1]
            if np.isfinite(c):
                corrs[peer] = c

        ranked = sorted(corrs.items(), key=lambda kv: -kv[1])
        all_valid = [p for p, _ in ranked]
        top8 = all_valid[:N_TOP]
        rng = np.random.RandomState(RANDOM_SEED + ci)
        random8 = (list(rng.choice(all_valid, size=min(N_RANDOM, len(all_valid)), replace=False))
                   if all_valid else [])
        rankings[cp] = {"top8": top8, "all": all_valid, "random8": random8}
    return rankings


def peers_for_date(rankings, checkpoints, date, arm):
    if arm == "A_baseline" or arm == "B_own_frac":
        return []
    idx = int(np.searchsorted(checkpoints, date, side="right")) - 1
    if idx < 0:
        return []
    key = {"C_top8": "top8", "D_allpeers": "all", "E_random8": "random8"}[arm]
    return rankings[checkpoints[idx]][key]


# ── 4. Методы прогноза ────────────────────────────────────────────────────────

def _la0(qvec, feats, tgts, K):
    d = np.linalg.norm(feats - qvec, axis=1)
    if len(d) < K:
        return np.nan
    knn = np.argpartition(d, K - 1)[:K]
    dd = d[knn]
    d_max = dd.max()
    if d_max < 1e-12:
        return float(tgts[knn].mean())
    w = np.exp(-0.5 * (dd / d_max) ** 2)
    w /= w.sum()
    return float((w * tgts[knn]).sum())


def _lwr(qvec, feats, tgts, K):
    d = np.linalg.norm(feats - qvec, axis=1)
    if len(d) < K:
        return np.nan
    knn = np.argpartition(d, K - 1)[:K]
    dd = d[knn]
    d_max = dd.max()
    if d_max < 1e-12:
        return float(tgts[knn].mean())
    w = np.exp(-0.5 * (dd / d_max) ** 2)
    sw = np.sqrt(w)
    A = np.column_stack([np.ones(K), feats[knn]]) * sw[:, None]
    b = tgts[knn] * sw
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(c[0] + c[1:] @ qvec)


def _smap(qvec, feats, tgts, min_pool, theta):
    if len(tgts) < min_pool:
        return np.nan
    d = np.linalg.norm(feats - qvec, axis=1)
    d_mean = d.mean()
    if d_mean < 1e-14:
        return float(tgts.mean())
    w = np.exp(-theta * d / d_mean)
    sw = np.sqrt(w)
    A = np.column_stack([np.ones(len(tgts)), feats]) * sw[:, None]
    b = tgts * sw
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(c[0] + c[1:] @ qvec)


METHODS = ["LA0", "LWR", "Smap"]


# ── 5. Walk-forward для одной комбинации (T_big, ratio, arm) ────────────────

def walk_forward(target_data, peer_data, rankings, checkpoints, t_big, t_frac, arm):
    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]

    # Полный (не обрезанный) T_big зигзаг — ТОЛЬКО для перечисления шагов
    # и получения фактического значения следующего пивота при скоринге.
    full_lp, full_conf, full_dirs = build_zigzag(lh, ll, dates, t_big)
    n_big = len(full_lp)

    records = []
    for i in range(MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))

        # ── физическая обрезка целевого ряда ──
        t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]
        own_big_lp, _, own_big_dir = build_zigzag(t_lh, t_ll, t_dt, t_big)
        if len(own_big_lp) < M + 1 or own_big_lp[-1] != full_lp[i]:
            continue  # защитная проверка согласованности обрезки

        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(M)])
        if not np.all(np.isfinite(qvec)):
            continue
        q_dir = int(own_big_dir[-1])

        own_big_feats, own_big_tgts, own_big_dirs = build_pool_rows(own_big_lp, own_big_dir, M)
        pool_feats = [own_big_feats]; pool_tgts = [own_big_tgts]; pool_dirs = [own_big_dirs]

        if arm != "A_baseline":
            own_frac_lp, _, own_frac_dir = build_zigzag(t_lh, t_ll, t_dt, t_frac)
            f, tg, dd = build_pool_rows(own_frac_lp, own_frac_dir, M)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

            for peer in peers_for_date(rankings, checkpoints, confirm_date, arm):
                p_dates = peer_data[peer]["dates"]
                p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
                if p_cutoff < M + 2:
                    continue
                p_lh = peer_data[peer]["lh"][:p_cutoff]
                p_ll = peer_data[peer]["ll"][:p_cutoff]
                p_dt = p_dates[:p_cutoff]
                p_lp, _, p_dir = build_zigzag(p_lh, p_ll, p_dt, t_frac)
                f, tg, dd = build_pool_rows(p_lp, p_dir, M)
                pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        feats = np.concatenate(pool_feats) if pool_feats else np.empty((0, M))
        tgts  = np.concatenate(pool_tgts)  if pool_tgts  else np.empty(0)
        dirs  = np.concatenate(pool_dirs)  if pool_dirs  else np.empty(0, dtype=np.int8)

        mask = dirs == q_dir
        feats_d, tgts_d = feats[mask], tgts[mask]

        # ── удаление событий пула, совпадающих с вектором запроса ──
        # Zigzag-артефакт: если между текущим и предыдущим T_big-пивотом не было
        # встречного движения ≥ T_frac, T_frac-зигзаг того же тикера (и его пул)
        # содержит ТЕ ЖЕ САМЫЕ пивоты — включая пивот-анкер, из которого строится
        # qvec. Такая "копия" запроса имеет d≈0 и не является содержательным
        # соседом (это не утечка из будущего, а тривиальное совпадение прошлого
        # события с самим собой). Чем ближе ratio к 1, тем чаще это происходит.
        n_dup = 0
        if len(feats_d):
            d_to_query = np.linalg.norm(feats_d - qvec, axis=1)
            dup_mask = d_to_query < DUP_EPS
            n_dup = int(dup_mask.sum())
            if n_dup:
                feats_d, tgts_d = feats_d[~dup_mask], tgts_d[~dup_mask]

        cur_lp = float(own_big_lp[-1])
        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)

        row = {"step": i, "n_pool": len(tgts_d), "n_dup": n_dup, "pers_err": pers_err}
        preds = {
            "LA0":  _la0(qvec, feats_d, tgts_d, K_LWR),
            "LWR":  _lwr(qvec, feats_d, tgts_d, K_LWR),
            "Smap": _smap(qvec, feats_d, tgts_d, MIN_POOL_SMAP, THETA_SMAP),
        }
        for name, lr in preds.items():
            if np.isfinite(lr):
                row[f"e_{name}"] = abs(float(np.exp(cur_lp + lr)) - actual_price)
            else:
                row[f"e_{name}"] = np.nan
        records.append(row)

    return pd.DataFrame(records)


def summarize(df):
    if df.empty:
        return {}
    dz = float(df["pers_err"].mean())
    out = {"n_steps": len(df), "pool_avg": float(df["n_pool"].mean()), "dup_avg": float(df["n_dup"].mean())}
    for name in METHODS:
        valid = df[f"e_{name}"].dropna()
        out[f"rMAE_{name}"] = float(valid.mean() / dz) if len(valid) > 5 and dz > 1e-12 else np.nan
        out[f"n_{name}"] = len(valid)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"=== 17_large_scale_pooled — Stage 1 ===")
    print(f"Target: {TARGET}   Universe: {len(UNIVERSE)} тикеров")
    print(f"T_BIG_GRID={T_BIG_GRID}  RATIO_GRID={RATIO_GRID}  ARMS={ARMS}")
    print(f"M={M}  K_LWR={K_LWR}  THETA_SMAP={THETA_SMAP}  MIN_HIST={MIN_HIST}")
    sys.stdout.flush()

    print("Загрузка данных...")
    target_data = load_ticker(TARGET)
    peer_data = {p: load_ticker(p) for p in PEERS}
    print(f"  {TARGET}: {len(target_data['dates'])} баров "
          f"({target_data['dates'][0][:10]} … {target_data['dates'][-1][:10]})")

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    print(f"Контрольные точки корреляции: {len(checkpoints)} (годовые)")
    rankings = compute_peer_rankings(target_data, peer_data, checkpoints)
    n_valid_last = len(rankings[checkpoints[-1]]["all"])
    print(f"  Пиров с достаточной историей на последней точке: {n_valid_last}/{len(PEERS)}")
    sys.stdout.flush()

    out_path = RESULTS / "stage1_sweep.csv"
    rows = []
    combos = [(tb, r, arm) for tb in T_BIG_GRID for r in RATIO_GRID for arm in ARMS]
    total = len(combos)

    for idx, (t_big, ratio, arm) in enumerate(combos, 1):
        t_frac = ratio * t_big
        df = walk_forward(target_data, peer_data, rankings, checkpoints, t_big, t_frac, arm)
        metrics = summarize(df)
        row = {"T_big": t_big, "ratio": ratio, "T_frac": t_frac, "arm": arm, **metrics}
        rows.append(row)

        pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.5f")

        elapsed = time.time() - t0
        eta = elapsed / idx * (total - idx)
        la0 = metrics.get("rMAE_LA0", float("nan"))
        lwr = metrics.get("rMAE_LWR", float("nan"))
        smap = metrics.get("rMAE_Smap", float("nan"))
        print(f"[{idx:3d}/{total}] T_big={t_big*100:.0f}% ratio={ratio:.2f} arm={arm:<11s} "
              f"n={metrics.get('n_steps', 0):3d} pool={metrics.get('pool_avg', 0):6.0f}  "
              f"LA0={la0:.4f} LWR={lwr:.4f} Smap={smap:.4f}  ETA {eta:.0f}s")
        sys.stdout.flush()

    print(f"\nСохранено: {out_path}")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
