#!/usr/bin/env python3
"""
causality_check.py

Проверка каузальности zigzag_forecast_ref.py:
  1. прогноз на полных данных (n_big пивотов)
  2. прогноз на урезанных данных (CUT_RATIO * n_big пивотов)
  3. для шагов в пересекающемся диапазоне предсказания должны совпасть побитово

Дополнительно проверяется каузальность самого find_pivots:
  — если обрезать сырые бары, первые N-1 подтверждённых пивотов не должны меняться.
"""
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from zigzag_forecast_ref import (
    load_tf, find_pivots, build_X, get_pool, ensemble_step,
    T_BIG, T_SMALL, H, MIN_HISTORY, P_LWR, P_SX,
)

DATA = Path(__file__).parent.parent.parent / "data" / "candles" / "SBER"

CUT_RATIO = 0.75  # обрезаем сырые бары на 75% от общего числа

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def run_predictions(p_big, dt_big, X_big_p3, X_big_p8,
                    p_small, dt_small, X_sm_p3, X_sm_p8,
                    n_big, n_small):
    """Возвращает dict: step -> pred (float или nan)."""
    preds = {}
    for step in range(MIN_HISTORY, n_big - H):
        if np.any(np.isnan(X_big_p3[step])) or np.any(np.isnan(X_big_p8[step])):
            preds[step] = np.nan
            continue
        ce    = int(np.searchsorted(dt_small, dt_big[step], side="left"))
        p_cur = float(p_big[step])
        res_lwr = get_pool(step, ce, X_big_p3, X_sm_p3, p_big, p_small,
                           n_big, n_small, P_LWR, use_fractal=True)
        res_sx  = get_pool(step, ce, X_big_p8, X_sm_p8, p_big, p_small,
                           n_big, n_small, P_SX, use_fractal=True)
        preds[step] = ensemble_step(res_lwr, res_sx, p_cur)
    return preds


def main():
    h10m, l10m, d10m = load_tf("10m")
    n_bars = len(h10m)
    cut_bar = int(n_bars * CUT_RATIO)

    print(f"Всего баров: {n_bars}   обрезаем до: {cut_bar}  (CUT_RATIO={CUT_RATIO})")
    print()

    # ── полный прогон ────────────────────────────────────────────────────────
    p_big_f,   dt_big_f   = find_pivots(h10m, l10m, d10m, T_BIG)
    p_small_f, dt_small_f = find_pivots(h10m, l10m, d10m, T_SMALL)
    n_big_f = len(p_big_f); n_small_f = len(p_small_f)
    X_big_p3_f  = build_X(p_big_f,   P_LWR)
    X_big_p8_f  = build_X(p_big_f,   P_SX)
    X_sm_p3_f   = build_X(p_small_f, P_LWR)
    X_sm_p8_f   = build_X(p_small_f, P_SX)
    preds_full = run_predictions(p_big_f, dt_big_f, X_big_p3_f, X_big_p8_f,
                                 p_small_f, dt_small_f, X_sm_p3_f, X_sm_p8_f,
                                 n_big_f, n_small_f)
    print(f"Полный прогон:   пивотов T=4%: {n_big_f}   T=0.6%: {n_small_f}")

    # ── урезанный прогон ─────────────────────────────────────────────────────
    p_big_c,   dt_big_c   = find_pivots(h10m[:cut_bar], l10m[:cut_bar], d10m[:cut_bar], T_BIG)
    p_small_c, dt_small_c = find_pivots(h10m[:cut_bar], l10m[:cut_bar], d10m[:cut_bar], T_SMALL)
    n_big_c = len(p_big_c); n_small_c = len(p_small_c)
    X_big_p3_c  = build_X(p_big_c,   P_LWR)
    X_big_p8_c  = build_X(p_big_c,   P_SX)
    X_sm_p3_c   = build_X(p_small_c, P_LWR)
    X_sm_p8_c   = build_X(p_small_c, P_SX)
    preds_cut  = run_predictions(p_big_c, dt_big_c, X_big_p3_c, X_big_p8_c,
                                 p_small_c, dt_small_c, X_sm_p3_c, X_sm_p8_c,
                                 n_big_c, n_small_c)
    print(f"Урезанный прогон: пивотов T=4%: {n_big_c}   T=0.6%: {n_small_c}")
    print()

    # ── Тест 1: каузальность find_pivots ─────────────────────────────────────
    # Первые n_big_c-2 пивотов должны совпасть с полными (последний pending)
    safe_n = n_big_c - 2
    piv_match = np.allclose(p_big_f[:safe_n], p_big_c[:safe_n]) and \
                np.all(dt_big_f[:safe_n] == dt_big_c[:safe_n])
    print(f"[Тест 1] find_pivots: первые {safe_n} пивотов совпадают "
          f"после обрезки баров? {PASS if piv_match else FAIL}")
    if not piv_match:
        diffs = np.where(p_big_f[:safe_n] != p_big_c[:safe_n])[0]
        print(f"  Расхождения на индексах: {diffs[:5]}")

    # ── Тест 2: каузальность walk-forward ────────────────────────────────────
    # Шаги в диапазоне [MIN_HISTORY, n_big_c - H - 2] — безопасная зона
    safe_steps = [s for s in preds_full if s in preds_cut and s < n_big_c - H - 2]
    n_safe = len(safe_steps)

    mismatches = []
    for s in safe_steps:
        a = preds_full[s]; b = preds_cut[s]
        if np.isnan(a) and np.isnan(b):
            continue
        if not np.isfinite(a) or not np.isfinite(b) or a != b:
            mismatches.append((s, a, b))

    ok2 = len(mismatches) == 0
    print(f"[Тест 2] walk-forward: {n_safe} шагов в безопасной зоне совпадают побитово? "
          f"{PASS if ok2 else FAIL}")
    if not ok2:
        print(f"  Расхождений: {len(mismatches)}")
        for s, a, b in mismatches[:5]:
            print(f"    step={s}: full={a:.6f}  cut={b:.6f}  diff={abs(a-b):.2e}")

    # ── Тест 3: предсказания на граничных шагах (n_big_c-4 .. n_big_c-H-1) ──
    # Убеждаемся, что даже у самой границы обрезки предсказания совпадают
    boundary_steps = [s for s in range(max(MIN_HISTORY, n_big_c - 4), n_big_c - H)
                      if s in preds_full and s in preds_cut]
    bnd_mismatches = [(s, preds_full[s], preds_cut[s])
                      for s in boundary_steps
                      if not (np.isnan(preds_full[s]) and np.isnan(preds_cut[s]))
                      and preds_full[s] != preds_cut[s]]
    ok3 = len(bnd_mismatches) == 0
    print(f"[Тест 3] граничные шаги {boundary_steps}: совпадают? "
          f"{PASS if ok3 else FAIL}")
    if not ok3:
        for s, a, b in bnd_mismatches:
            print(f"  step={s}: full={a:.6f}  cut={b:.6f}  |diff|={abs(a-b):.2e}")

    print()
    if piv_match and ok2 and ok3:
        print("Итог: ВСЕ ТЕСТЫ ПРОЙДЕНЫ — алгоритм каузален.")
    else:
        print("Итог: ЕСТЬ НАРУШЕНИЯ КАУЗАЛЬНОСТИ.")


if __name__ == "__main__":
    main()
