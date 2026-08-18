#!/usr/bin/env python3
"""
27_pf_reversal.py — P&F-style (Point & Figure) сетка уровней с фильтром
разворота R>1, как расширение эксп.26 (level crossing, R=1).

Идея (эксп.26 §8, "дальнейшие направления"): в эксп.26 разворот был так же
лёгок, как продолжение (R=1) — это давало accuracy ~ 0.5 (нет сигнала) и
сильный дребезг (frac_reversal ≈ 0.4-0.5). Классический P&F/renko приём:
продолжение тренда подтверждается 1 уровнем, но РАЗВОРОТ требует пройти R
уровней ПРОТИВ текущего направления колонки. При подтверждённом развороте
"задним числом" эмитятся R событий подряд в новом направлении (backfill) —
амплитуда каждого события остаётся ровно Δ, прогноз остаётся классификацией
направления (как в эксп.26), меняется только правило подтверждения события.

Для R=1 эта конструкция ЭКВИВАЛЕНТНА эксп.26 (порог "продолжение" и порог
"разворот" совпадают — code path сливается в тот же результат), поэтому R=1
отдельно не гоняем — уже есть результаты эксп.26.

Каузальный контракт: тот же принцип, что эксп.17/21/23/24/25/26 — прогноз на
каждом origin строится на данных, обрезанных ФИЗИЧЕСКИМ СРЕЗОМ массивов
(`arr[:cutoff_idx]`) до момента origin. Каузальность самого детектора событий
(обрезка+rebuild = точно префикс полного прохода) проверяется ЭМПИРИЧЕСКИ на
нескольких срезах — по стандарту, установленному в эксп.25/26.

Без кросс-тикерного пула — один тикер SBER (то же решение, что эксп.26: не
менять два фактора конструкции сразу).

Метрика — как в эксп.26: rMAE относительно momentum-continuation baseline
("повторить последнее направление"), accuracy направления против (а) монетки
0.5, (б) этого же persist-baseline. ВАЖНО (критика, см. README §5): при R>1
persist-baseline механически смещён в сторону продолжения (продолжение легче
разворота по конструкции) — ожидается более высокая baseline-accuracy, чем в
эксп.26; критерий сигнала — обгон persist-baseline, не монетки.
"""
import importlib.util
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
EXP17_DIR = HERE.parents[0] / "17_large_scale_pooled"
EXP23_DIR = HERE.parents[0] / "23_asymmetric_zigzag"
DATA = HERE.parents[2] / "data" / "candles"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

spec17 = importlib.util.spec_from_file_location("exp17", EXP17_DIR / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

spec23 = importlib.util.spec_from_file_location("exp23", EXP23_DIR / "23_asymmetric_zigzag_calibration.py")
exp23 = importlib.util.module_from_spec(spec23)
spec23.loader.exec_module(exp23)

golden = exp23.golden
_smap = exp17._smap
build_pool_rows = exp17.build_pool_rows

TARGET = "SBER"

DELTA_GRID = [0.032, 0.05, 0.08, 0.13, 0.20]   # тот же рабочий грид, что эксп.26
R_GRID = [2, 3]                                  # R=1 = эксп.26, не повторяем

MIN_HIST = 30
H_LIST = [1, 3, 5, 10]
HMAX = max(H_LIST)

N_ORIGINS_QUICK = 300
N_ORIGINS_FULL = 800
N_TOP_COMBOS = 2

M_GRID = [2, 3, 4, 5]
THETA_LO, THETA_HI, THETA_TOL = 0.0, 32.0, 0.1
MAX_OUTER = 5
DEF_M, DEF_THETA = 3, 1.0


# ── 1. Загрузка ──────────────────────────────────────────────────────────────

def load_target():
    path = DATA / TARGET / "1d.json"
    raw = json.load(open(path))
    high = np.array([c["high"] for c in raw], dtype=np.float64)
    low = np.array([c["low"] for c in raw], dtype=np.float64)
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    open_ = np.array([c["open"] for c in raw], dtype=np.float64)
    high = np.where(high <= 0, np.nan, high)
    low = np.where(low <= 0, np.nan, low)
    close = np.where(close <= 0, np.nan, close)
    open_ = np.where(open_ <= 0, np.nan, open_)
    dates = np.array([c["begin"] for c in raw])
    return {"lh": np.log(high), "ll": np.log(low), "lo": np.log(open_),
            "lc": np.log(close), "dates": dates}


# ── 2. Детектор P&F-уровней (каузальный по построению) ───────────────────────

def _advance_toward(direction, cur_level, cur_val, target, sign, delta, R, i, dates, lp, conf, dirs):
    """Продвигает состояние (direction, cur_level, cur_val) максимально в
    сторону `target` (sign=+1 если target=hi, sign=-1 если target=low).
    direction==sign или 0 (не установлено) -> порог продолжения = 1 уровень.
    direction==-sign -> порог разворота = R уровней, событие эмитится
    backfill'ом (R событий подряд), затем состояние продолжает тем же циклом
    (дальнейшее движение в ту же сторону снова требует лишь 1 уровня)."""
    while True:
        gap = sign * (target - cur_val)
        if direction == sign or direction == 0:
            if gap >= delta:
                cur_level += sign
                cur_val = cur_level * delta
                lp.append(cur_val); conf.append(dates[i]); dirs.append(sign)
                direction = sign
                continue
            break
        else:
            if gap >= R * delta:
                for _ in range(R):
                    cur_level += sign
                    cur_val = cur_level * delta
                    lp.append(cur_val); conf.append(dates[i]); dirs.append(sign)
                direction = sign
                continue
            break
    return direction, cur_level, cur_val


def build_level_events_R(lh, ll, lo, lc, dates, delta, R):
    """P&F-сетка: продолжение = 1 уровень, разворот = R уровней (backfill).
    R=1 эквивалентен эксп.26 (build_level_events)."""
    lp, conf, dirs = [], [], []
    anchor = (lh[0] + ll[0]) / 2.0
    cur_level = round(anchor / delta)
    cur_val = cur_level * delta
    direction = 0
    n = len(lh)
    for i in range(n):
        hi, low_, cl, op = lh[i], ll[i], lc[i], lo[i]
        if not (np.isfinite(hi) and np.isfinite(low_) and np.isfinite(cl) and np.isfinite(op)):
            continue
        order = (1, -1) if cl >= op else (-1, 1)
        for sign in order:
            target = hi if sign == 1 else low_
            direction, cur_level, cur_val = _advance_toward(
                direction, cur_level, cur_val, target, sign, delta, R, i, dates, lp, conf, dirs)
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


def verify_causality(full):
    lh, ll, lo, lc, dates = full["lh"], full["ll"], full["lo"], full["lc"], full["dates"]
    delta, R = 0.05, 3
    full_lp, full_conf, full_dirs = build_level_events_R(lh, ll, lo, lc, dates, delta, R)
    cutoffs = [200, 800, 1800, 2800, 3800, 4500, len(lh)]
    bad = 0
    print(f"Проверка каузальности детектора (Δ={delta}, R={R}, срез -> rebuild -> сравнение с префиксом):")
    for c in cutoffs:
        sub_lp, sub_conf, sub_dirs = build_level_events_R(lh[:c], ll[:c], lo[:c], lc[:c], dates[:c], delta, R)
        n_sub = len(sub_lp)
        ok = (n_sub <= len(full_lp) and np.allclose(sub_lp, full_lp[:n_sub])
              and np.array_equal(sub_dirs, full_dirs[:n_sub])
              and np.array_equal(sub_conf, full_conf[:n_sub]))
        bad += (not ok)
        print(f"  cutoff bar={c:5d}: n_sub_events={n_sub:5d}  {'OK' if ok else 'MISMATCH'}")
    print(f"  -> {len(cutoffs) - bad}/{len(cutoffs)} срезов совпадают с префиксом полного прохода\n")
    return bad == 0


def scale_diagnostics(full):
    lh, ll, lo, lc, dates = full["lh"], full["ll"], full["lo"], full["lc"], full["dates"]
    rows = []
    for R in R_GRID:
        for d in DELTA_GRID:
            lp, conf, dirs = build_level_events_R(lh, ll, lo, lc, dates, d, R)
            n = len(lp)
            rev_frac = float(np.mean(dirs[1:] != dirs[:-1])) if n > 1 else float("nan")
            rows.append({"R": R, "delta": d, "n_events": n, "events_per_bar": n / len(lh),
                         "frac_reversal": rev_frac,
                         "zone": "artifact (events/bar>1)" if n / len(lh) > 1.0 else "usable"})
    df = pd.DataFrame(rows)
    print("Масштаб P&F-сетки (SBER, {} баров):".format(len(lh)))
    print(df.to_string(index=False))
    print()
    df.to_csv(RESULTS / "scale_diagnostics.csv", index=False, float_format="%.4f")
    return df


# ── 3. Пул + оценка (H=1, тот же каскад, что эксп.17/23/26) ─────────────────

def subsample(arr, n):
    if len(arr) <= n:
        return arr
    idx = np.unique(np.linspace(0, len(arr) - 1, n).round().astype(int))
    return arr[idx]


def build_all_pools_levels(full, delta, R, m, origin_indices):
    lh, ll, lo, lc, dates = full["lh"], full["ll"], full["lo"], full["lc"], full["dates"]
    full_lp, full_conf, full_dirs = build_level_events_R(lh, ll, lo, lc, dates, delta, R)
    pools = []
    for i in origin_indices:
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        own_lp, own_conf, own_dirs = build_level_events_R(
            lh[:cutoff_idx], ll[:cutoff_idx], lo[:cutoff_idx], lc[:cutoff_idx], dates[:cutoff_idx], delta, R)
        if len(own_lp) < m + 2 or abs(own_lp[-1] - full_lp[i]) > 1e-9 or own_dirs[-1] != full_dirs[i]:
            continue
        feats, tgts, ds = build_pool_rows(own_lp, own_dirs, m)
        if len(tgts) < m + 2:
            continue
        qvec = np.array([own_lp[-1 - lag] - own_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue
        cur_lp = float(own_lp[-1])
        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(own_lp[-1] + full_dirs[i] * delta))
        pers_err = abs(actual_price - pers_price)
        pools.append((qvec, feats, tgts, cur_lp, actual_price, pers_err))
    return pools


def eval_direction_and_rmae(pools, theta, min_pool):
    errs, pers_errs, correct = [], [], []
    for qvec, feats, tgts, cur_lp, actual_price, pers_err in pools:
        y = _smap(qvec, feats, tgts, min_pool, theta)
        if not np.isfinite(y):
            continue
        pred_price = float(np.exp(cur_lp + y))
        errs.append(abs(pred_price - actual_price))
        pers_errs.append(pers_err)
        actual_dir = 1 if actual_price >= np.exp(cur_lp) else -1
        pred_dir = 1 if y >= 0 else -1
        correct.append(int(pred_dir == actual_dir))
    if len(errs) < 5:
        return float("inf"), 0, float("nan")
    return float(np.mean(errs) / np.mean(pers_errs)), len(errs), float(np.mean(correct))


def calibrate(full, delta, R, origins):
    m, theta = DEF_M, DEF_THETA
    prev = None
    v, acc, n = float("inf"), float("nan"), 0
    for outer in range(MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            pools = build_all_pools_levels(full, delta, R, mc, origins)
            vv, _, _ = eval_direction_and_rmae(pools, theta, mc + 2)
            if vv < best_v:
                best_v, best_m = vv, mc
        m = best_m
        pools = build_all_pools_levels(full, delta, R, m, origins)
        theta = golden(lambda th: eval_direction_and_rmae(pools, th, m + 2)[0], THETA_LO, THETA_HI, THETA_TOL)
        v, n, acc = eval_direction_and_rmae(pools, theta, m + 2)
        print(f"    iter {outer + 1}: m={m} θ={theta:.3f} -> rMAE={v:.4f} acc={acc:.3f} (n={n})")
        cur = (m, round(theta, 2))
        if cur == prev:
            break
        prev = cur
    return m, theta, v, acc, n


# ── 4. Рекурсивный многошаговый прогноз (H событий вперёд) ───────────────────

def evaluate_multistep(full, delta, R, m, theta, origin_indices, h_list):
    lh, ll, lo, lc, dates = full["lh"], full["ll"], full["lo"], full["lc"], full["dates"]
    full_lp, full_conf, full_dirs = build_level_events_R(lh, ll, lo, lc, dates, delta, R)
    hmax = max(h_list)
    step_correct = {h: [] for h in range(1, hmax + 1)}
    persist_correct = {h: [] for h in range(1, hmax + 1)}
    price_err = {h: [] for h in h_list}
    pers_price_err = {h: [] for h in h_list}
    n_used = 0
    for i in origin_indices:
        if i + hmax >= len(full_lp):
            continue
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        own_lp, own_conf, own_dirs = build_level_events_R(
            lh[:cutoff_idx], ll[:cutoff_idx], lo[:cutoff_idx], lc[:cutoff_idx], dates[:cutoff_idx], delta, R)
        if len(own_lp) < m + 2 or abs(own_lp[-1] - full_lp[i]) > 1e-9 or own_dirs[-1] != full_dirs[i]:
            continue
        feats, tgts, ds = build_pool_rows(own_lp, own_dirs, m)
        if len(tgts) < m + 2:
            continue
        qvec = np.array([own_lp[-1 - lag] - own_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue

        cur_qvec = qvec.copy()
        preds = []
        for h in range(1, hmax + 1):
            y = _smap(cur_qvec, feats, tgts, m + 2, theta)
            if not np.isfinite(y):
                break
            d_hat = 1 if y >= 0 else -1
            preds.append(d_hat)
            cur_qvec = np.roll(cur_qvec, 1)
            cur_qvec[0] = d_hat * delta
        if len(preds) < hmax:
            continue

        n_used += 1
        for h in range(1, hmax + 1):
            step_correct[h].append(int(preds[h - 1] == full_dirs[i + h]))
            persist_correct[h].append(int(full_dirs[i] == full_dirs[i + h]))
        cum = 0.0
        for h in range(1, hmax + 1):
            cum += preds[h - 1] * delta
            if h in h_list:
                pred_price = float(np.exp(own_lp[-1] + cum))
                actual_price = float(np.exp(full_lp[i + h]))
                pers_price = float(np.exp(own_lp[-1] + full_dirs[i] * h * delta))
                price_err[h].append(abs(pred_price - actual_price))
                pers_price_err[h].append(abs(pers_price - actual_price))

    step_acc = {h: float(np.mean(v)) for h, v in step_correct.items() if v}
    persist_acc = {h: float(np.mean(v)) for h, v in persist_correct.items() if v}
    rmae_h = {h: float(np.mean(price_err[h]) / np.mean(pers_price_err[h])) for h in h_list if price_err[h]}
    return step_acc, persist_acc, rmae_h, n_used


# ── 5. main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=== 27_pf_reversal — P&F-сетка уровней (R>1), многошаговый прогноз (SBER) ===\n")

    full = load_target()
    ok = verify_causality(full)
    if not ok:
        print("!!! ПРИЧИННОСТЬ НЕ ПОДТВЕРЖДЕНА — прекращаю выполнение.")
        return
    scale_df = scale_diagnostics(full)

    lh, ll, lo, lc, dates = full["lh"], full["ll"], full["lo"], full["lc"], full["dates"]

    usable = scale_df[scale_df["zone"] == "usable"]
    combos = list(zip(usable["R"], usable["delta"]))
    print(f"--- Фаза 1: быстрая проверка (m=3, θ=1.0, без калибровки) на {len(combos)} комбинациях (Δ,R) ---")
    quick_rows = []
    for R, delta in combos:
        full_lp, full_conf, full_dirs = build_level_events_R(lh, ll, lo, lc, dates, delta, R)
        n_full = len(full_lp)
        all_origins = np.arange(MIN_HIST, max(MIN_HIST, n_full - HMAX - 1))
        origins = subsample(all_origins, N_ORIGINS_QUICK)
        pools = build_all_pools_levels(full, delta, R, DEF_M, origins)
        v, n, acc = eval_direction_and_rmae(pools, DEF_THETA, DEF_M + 2)
        rev_frac = float(np.mean(full_dirs[1:] != full_dirs[:-1])) if n_full > 1 else float("nan")
        persist_baseline = 1 - rev_frac
        print(f"  R={R} Δ={delta:.3f}  n_full_events={n_full:5d}  n_eval={n:4d}  "
              f"rMAE={v:.4f}  acc={acc:.3f}  (persist-baseline={persist_baseline:.3f}, coin=0.500)")
        quick_rows.append({"R": R, "delta": delta, "n_full_events": n_full, "n_eval": n,
                            "rMAE": v, "accuracy": acc, "persist_baseline_acc": persist_baseline})
    quick_df = pd.DataFrame(quick_rows)
    quick_df.to_csv(RESULTS / "phase1_quick.csv", index=False, float_format="%.4f")
    print()

    top_combos = quick_df.sort_values("rMAE").head(N_TOP_COMBOS)[["R", "delta"]].values.tolist()
    print(f"--- Фаза 2: полная калибровка на (R,Δ)={top_combos} (подвыборка {N_ORIGINS_FULL} origin'ов) ---")

    calib_rows = []
    multistep_rows = []
    for R, delta in top_combos:
        R = int(R)
        full_lp, full_conf, full_dirs = build_level_events_R(lh, ll, lo, lc, dates, delta, R)
        n_full = len(full_lp)
        all_origins = np.arange(MIN_HIST, max(MIN_HIST, n_full - HMAX - 1))
        origins = subsample(all_origins, N_ORIGINS_FULL)

        print(f"  R={R} Δ={delta}:")
        t1 = time.time()
        m, theta, v, acc, n = calibrate(full, delta, R, origins)
        el = time.time() - t1
        print(f"    -> m={m} θ={theta:.3f} rMAE={v:.4f} acc={acc:.3f} (n={n})  ({el:.1f}s)")
        calib_rows.append({"R": R, "delta": delta, "m": m, "theta": round(theta, 3),
                            "rMAE_H1": round(v, 4), "acc_H1": round(acc, 4), "n_eval": n})

        step_acc, persist_acc, rmae_h, n_used = evaluate_multistep(full, delta, R, m, theta, origins, H_LIST)
        rev_frac = float(np.mean(full_dirs[1:] != full_dirs[:-1])) if n_full > 1 else float("nan")
        print(f"    многошаговый (n_used={n_used}):")
        for h in range(1, HMAX + 1):
            marker = " *" if h in H_LIST else ""
            print(f"      h={h:2d}  smap_acc={step_acc.get(h, float('nan')):.3f}  "
                  f"persist_acc={persist_acc.get(h, float('nan')):.3f}  coin=0.500{marker}")
        for h in H_LIST:
            print(f"      H={h:2d}  price_rMAE(smap/persist)={rmae_h.get(h, float('nan')):.4f}")
            multistep_rows.append({"R": R, "delta": delta, "H": h, "smap_acc": step_acc.get(h, float("nan")),
                                    "persist_acc": persist_acc.get(h, float("nan")), "coin_acc": 0.5,
                                    "price_rMAE": rmae_h.get(h, float("nan")), "n_used": n_used})
        print()

    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(RESULTS / "phase2_calibration.csv", index=False, float_format="%.4f")
    multistep_df = pd.DataFrame(multistep_rows)
    multistep_df.to_csv(RESULTS / "phase3_multistep.csv", index=False, float_format="%.4f")

    print(f"{'=' * 100}")
    print("Фаза 1 (быстрая проверка):")
    print(quick_df.to_string(index=False))
    print(f"\nФаза 2 (калибровка, H=1):")
    print(calib_df.to_string(index=False))
    print(f"\nФаза 3 (многошаговый прогноз):")
    print(multistep_df.to_string(index=False))
    print(f"{'=' * 100}")
    print(f"Всего: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
