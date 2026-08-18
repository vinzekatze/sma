#!/usr/bin/env python3
"""
causality_check_smap_band.py — проверка каузальности smap_band_ref.py.

Два теста:
  1. Префиксная устойчивость build_zigzag: если обрезать сырые бары, ранее
     подтверждённые пивоты (и T_query, и T_pool) не должны измениться.
  2. Побитовое совпадение прогноза (точка + ВСЯ полоса, все 5 квантилей,
     несколько шагов) на "полном" и "урезанном" прогонах для origin'ов,
     достаточно ранних, чтобы обрезка их не задела.

Origin здесь — индекс пивота T_query (не смещение в барах, как --origin в
CLI smap_band_ref.py). Для каждого origin пул строится ТОЛЬКО из T_pool-
пивотов с датой подтверждения <= дате origin'а (searchsorted) — тот же
принцип единственной точки обрезки, что и в остальном пайплайне фазы 7.
Тест 4 — build_causal_pool (кросс-тикерный пул, эксп.17): для НЕСКОЛЬКИХ
пиров сырые бары каждого обрезаются НЕЗАВИСИМО по дате <= cutoff_date (маска
в самом начале обработки пира). Проверяем: пул, построенный из ПОЛНЫХ
массивов пиров, идентичен пулу, построенному из массивов, у которых сырые
бары УЖЕ обрезаны намного дальше вперёд (но всё ещё позже cutoff_date) —
если это так, дополнительные будущие бары пира не могли повлиять на пул.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from smap_band_ref import (
    load_log_candles, load_ticker_candles, build_zigzag, build_pool_vectors,
    build_query_vector, build_causal_pool, run_band_forecast, QUANTILE_LEVELS,
)

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "candles" / "SBER" / "10m.json"

T_QUERY   = 0.04
T_POOL    = 0.036
M         = 2
THETA     = 2.0
STEPS     = 3
MIN_HIST  = M + 3          # запас над m+1, чтобы избежать краевых эффектов build_query_vector
BOUNDARY_MARGIN = 3        # origin'ы ближе margin к концу cut-серии считаются небезопасными
CUT_RATIO = 0.75

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def run_predictions_over_origins(q_lp, q_dates, q_dirs, pool_lp, pool_dates, pool_dirs,
                                 m, theta, steps, min_hist):
    """Для каждого origin (индекс пивота T_query, min_hist..n_query-2) строит
    query_vector из ПРЕФИКСА q_lp[:origin+1] и пул из ВСЕХ pool-пивотов с
    датой подтверждения <= дате origin'а. Возвращает {origin: forecast_list}."""
    n_query = len(q_lp)
    min_pool_size = m + 2
    preds = {}

    for origin in range(min_hist, n_query - 1):
        q_lp_prefix = q_lp[: origin + 1]
        try:
            qv = build_query_vector(q_lp_prefix, m)
        except ValueError:
            continue
        q_dir       = int(q_dirs[origin])
        origin_date = q_dates[origin]
        last_lp     = float(q_lp_prefix[-1])

        pool_cutoff    = int(np.searchsorted(pool_dates, origin_date, side="right"))
        pool_lp_prefix  = pool_lp[:pool_cutoff]
        pool_dir_prefix = pool_dirs[:pool_cutoff]
        pfm, ptr, pdir = build_pool_vectors(pool_lp_prefix, pool_dir_prefix, m)

        results = run_band_forecast(qv, q_dir, last_lp, pfm, ptr, pdir,
                                    theta, min_pool_size, steps)
        preds[origin] = results

    return preds


def compare_forecasts(a: list[dict], b: list[dict]) -> list[str]:
    """Побитовое сравнение двух списков шагов прогноза (точка + полоса).
    Возвращает список текстовых расхождений (пусто, если всё совпало)."""
    diffs = []
    if len(a) != len(b):
        diffs.append(f"разное число шагов: {len(a)} vs {len(b)}")
        return diffs
    for step_a, step_b in zip(a, b):
        if step_a["ok"] != step_b["ok"]:
            diffs.append(f"шаг {step_a['step']}: ok={step_a['ok']} vs {step_b['ok']}")
            continue
        if not step_a["ok"]:
            continue
        if step_a["price"] != step_b["price"]:
            diffs.append(f"шаг {step_a['step']}: price {step_a['price']!r} vs {step_b['price']!r}")
        for q in QUANTILE_LEVELS:
            va, vb = step_a["band_prices"][q], step_b["band_prices"][q]
            if va != vb:
                diffs.append(f"шаг {step_a['step']}: band[{q}] {va!r} vs {vb!r}")
    return diffs


def main():
    print(f"Файл: {DATA_FILE}")
    log_highs, log_lows, dates = load_log_candles(str(DATA_FILE))
    n_bars = len(dates)
    cut_bar = int(n_bars * CUT_RATIO)
    print(f"Всего баров: {n_bars}   обрезаем до: {cut_bar}  (CUT_RATIO={CUT_RATIO})\n")

    # ── полный прогон ────────────────────────────────────────────────────────
    q_lp_f, q_dt_f, q_dir_f = build_zigzag(log_highs, log_lows, dates, T_QUERY)
    p_lp_f, p_dt_f, p_dir_f = build_zigzag(log_highs, log_lows, dates, T_POOL)
    print(f"Полный прогон:    T_query пивотов: {len(q_lp_f)}   T_pool пивотов: {len(p_lp_f)}")

    # ── урезанный прогон (сырые бары обрезаны ДО построения зигзага) ─────────
    log_highs_c = log_highs[:cut_bar]
    log_lows_c  = log_lows[:cut_bar]
    dates_c     = dates[:cut_bar]
    q_lp_c, q_dt_c, q_dir_c = build_zigzag(log_highs_c, log_lows_c, dates_c, T_QUERY)
    p_lp_c, p_dt_c, p_dir_c = build_zigzag(log_highs_c, log_lows_c, dates_c, T_POOL)
    print(f"Урезанный прогон: T_query пивотов: {len(q_lp_c)}   T_pool пивотов: {len(p_lp_c)}\n")

    # ── Тест 1: префиксная устойчивость build_zigzag (T_query и T_pool) ──────
    safe_q = len(q_lp_c) - 2
    q_match = (np.allclose(q_lp_f[:safe_q], q_lp_c[:safe_q])
              and np.all(q_dt_f[:safe_q] == q_dt_c[:safe_q])
              and np.all(q_dir_f[:safe_q] == q_dir_c[:safe_q]))
    print(f"[Тест 1a] T_query build_zigzag: первые {safe_q} пивотов совпадают "
          f"после обрезки баров? {PASS if q_match else FAIL}")
    if not q_match:
        diffs = np.where(~np.isclose(q_lp_f[:safe_q], q_lp_c[:safe_q]))[0]
        print(f"  Расхождения на индексах: {diffs[:5]}")

    safe_p = len(p_lp_c) - 2
    p_match = (np.allclose(p_lp_f[:safe_p], p_lp_c[:safe_p])
              and np.all(p_dt_f[:safe_p] == p_dt_c[:safe_p])
              and np.all(p_dir_f[:safe_p] == p_dir_c[:safe_p]))
    print(f"[Тест 1b] T_pool build_zigzag:  первые {safe_p} пивотов совпадают "
          f"после обрезки баров? {PASS if p_match else FAIL}")
    if not p_match:
        diffs = np.where(~np.isclose(p_lp_f[:safe_p], p_lp_c[:safe_p]))[0]
        print(f"  Расхождения на индексах: {diffs[:5]}")

    # ── Тест 2: побитовое совпадение прогноза (точка + полоса) ───────────────
    preds_full = run_predictions_over_origins(q_lp_f, q_dt_f, q_dir_f, p_lp_f, p_dt_f, p_dir_f,
                                              M, THETA, STEPS, MIN_HIST)
    preds_cut  = run_predictions_over_origins(q_lp_c, q_dt_c, q_dir_c, p_lp_c, p_dt_c, p_dir_c,
                                              M, THETA, STEPS, MIN_HIST)

    safe_origin_max = len(q_lp_c) - 1 - BOUNDARY_MARGIN
    safe_origins = [o for o in preds_full if o in preds_cut and o <= safe_origin_max]
    print(f"\n[Тест 2] прогноз (точка+полоса, {STEPS} шага): "
          f"{len(safe_origins)} origin'ов в безопасной зоне (<= {safe_origin_max})")

    mismatches = {}
    for o in safe_origins:
        diffs = compare_forecasts(preds_full[o], preds_cut[o])
        if diffs:
            mismatches[o] = diffs

    ok2 = len(mismatches) == 0
    print(f"  побитовое совпадение? {PASS if ok2 else FAIL}")
    if not ok2:
        print(f"  Расхождений: {len(mismatches)}/{len(safe_origins)}")
        for o, diffs in list(mismatches.items())[:5]:
            print(f"    origin={o}:")
            for d in diffs[:3]:
                print(f"      {d}")

    # ── Тест 3: граничные origin'ы (последние BOUNDARY_MARGIN перед обрезкой) ─
    boundary_origins = [o for o in range(max(MIN_HIST, len(q_lp_c) - BOUNDARY_MARGIN - 2),
                                         len(q_lp_c) - 1)
                        if o in preds_full and o in preds_cut]
    bnd_mismatches = {}
    for o in boundary_origins:
        diffs = compare_forecasts(preds_full[o], preds_cut[o])
        if diffs:
            bnd_mismatches[o] = diffs
    print(f"\n[Тест 3] граничные origin'ы {boundary_origins} "
          f"(ожидаемо МОГУТ расходиться — pending-пивот):")
    for o in boundary_origins:
        status = "расходится" if o in bnd_mismatches else "совпал"
        print(f"    origin={o}: {status}")

    # ── Тест 4: build_causal_pool (кросс-тикерный пул) ────────────────────────
    peer_tickers = ["SBER", "LKOH", "GAZP", "CHMF", "MGNT"]
    peer_full, peer_cut = {}, {}
    for tk in peer_tickers:
        loaded = load_ticker_candles(tk, "1d")
        if loaded is None:
            continue
        lh, ll, dt = loaded
        peer_full[tk] = (lh, ll, dt)
        cut_bar_peer = int(len(dt) * CUT_RATIO)
        peer_cut[tk] = (lh[:cut_bar_peer], ll[:cut_bar_peer], dt[:cut_bar_peer])

    # cutoff_date заведомо раньше 75%-обрезки любого пира — берём дату на
    # уровне 50% полной истории самого короткого пира
    min_len = min(len(dt) for _, _, dt in peer_full.values())
    cutoff_date = str(sorted(peer_full.values(), key=lambda x: len(x[2]))[0][2][min_len // 2])
    m_test = 5
    t_pool_test = 0.18

    feats_full, tars_full, dirs_full = build_causal_pool(cutoff_date, peer_full, t_pool_test, m_test)
    feats_cut,  tars_cut,  dirs_cut  = build_causal_pool(cutoff_date, peer_cut,  t_pool_test, m_test)

    pool_match = (feats_full.shape == feats_cut.shape
                 and np.allclose(feats_full, feats_cut)
                 and np.allclose(tars_full, tars_cut)
                 and np.array_equal(dirs_full, dirs_cut))
    print(f"\n[Тест 4] build_causal_pool ({len(peer_full)} пиров, cutoff_date={cutoff_date[:10]}): "
          f"пул из ПОЛНЫХ данных пиров == пул из данных, обрезанных на {CUT_RATIO:.0%}? "
          f"{PASS if pool_match else FAIL}")
    print(f"  событий в пуле: {len(tars_full)} (полные данные) vs {len(tars_cut)} (обрезанные)")
    if not pool_match:
        print(f"  shapes: {feats_full.shape} vs {feats_cut.shape}")

    print()
    if q_match and p_match and ok2 and pool_match:
        print("Итог: ВСЕ ОБЯЗАТЕЛЬНЫЕ ТЕСТЫ ПРОЙДЕНЫ — smap_band_ref.py каузален "
              "(включая кросс-тикерный пул).")
    else:
        print("Итог: ЕСТЬ НАРУШЕНИЯ КАУЗАЛЬНОСТИ.")


if __name__ == "__main__":
    main()
