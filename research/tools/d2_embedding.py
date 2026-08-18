#!/usr/bin/env python3
"""
d2_embedding.py — Фрактальная размерность и оптимальное вложение для набора пивотов.

════════════════════════════════════════════════════════════════════════════════
КАУЗАЛЬНОСТЬ
════════════════════════════════════════════════════════════════════════════════

  Основная точка входа — measure_d2_causal():

      result = measure_d2_causal(highs, lows, origin=600, thresholds=[0.04])

  Функция:
    1. Обрезает ряды по origin: highs[:origin], lows[:origin].
    2. Строит пивоты зигзага из обрезанных данных для каждого T.
    3. Объединяет пивоты всех T в один ряд (если T несколько).
    4. Проводит измерения D₂.

  Обрезка происходит в первых строках функции — явно, с assert.
  После обрезки алгоритм не имеет доступа к исходным массивам.

  Низкоуровневая функция measure_d2(prices) принимает уже готовый
  массив пивотов и не знает ни о каком origin.

════════════════════════════════════════════════════════════════════════════════
ЧТО ИЗМЕРЯЕМ
════════════════════════════════════════════════════════════════════════════════

  Вложение: hybrid X[i] = [price_i, lr₁, ..., lr_{p-1}]
    lr_k = log(price_{i−k+1} / price_{i−k})
  Совпадает с вложением прогноза (zigzag_forecast_ref.py).

  Нормировка: per-field z-score по облаку пивотов.

  D₂ sweep по p = 1 .. p_max.

  Диапазон плато: p с R² ≥ r2_min и D₂ ≥ (1-plateau_tol)×D₂_max.
  Фрактальная D₂ = медиана D₂ на плато.

════════════════════════════════════════════════════════════════════════════════
Использование как модуль:
    from research.tools.d2_embedding import measure_d2_causal
    r = measure_d2_causal(highs, lows, origin=600, thresholds=[0.04])
    r = measure_d2_causal(highs, lows, origin=600, thresholds=[0.04, 0.004])
    print(r)

Использование как CLI:
    python research/tools/d2_embedding.py \\
        --data data/candles/SBER/10m.json --origin 600 --T 0.04
    python research/tools/d2_embedding.py \\
        --data data/candles/SBER/10m.json --origin 600 --T 0.04 0.004
    python research/tools/d2_embedding.py --demo
════════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from scipy.spatial.distance import pdist

__all__ = ["measure_d2_causal", "measure_d2", "D2Result"]

# ── параметры по умолчанию ────────────────────────────────────────────────────

P_MAX_DEFAULT  = 12     # максимальная размерность sweep
N_R_DEFAULT    = 35     # число r-точек для C(r)
PLATEAU_TOL    = 0.10   # плато: D₂ ≥ (1-tol) × D₂_max
R2_MIN_DEFAULT = 0.95   # минимальный R² линейного участка
MIN_POINTS     = 30     # минимальное число точек в облаке
MAX_SUBSAMPLE  = 3000   # максимум точек для расчёта расстояний
THEILER_W      = 1      # Theiler window: исключаем пары |i-j| ≤ W (автокорреляция)
R_MAX_PCT      = 60     # верхний перцентиль d для r_max (граничные эффекты начинаются раньше)
LIN_LO         = 0.15   # нижняя граница линейного участка (доля от диапазона log r)
LIN_HI         = 0.70   # верхняя граница линейного участка


# ════════════════════════════════════════════════════════════════════════════
# Структура результата
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class D2Result:
    """Результат измерения D₂ для точки прогноза."""

    d2_fractal: float
    """Фрактальная D₂ — медиана D₂ на плато."""

    p_range: tuple
    """(p_lo, p_hi) — диапазон p на плато (R² ≥ r2_min, D₂ ≥ 90% D₂_max)."""

    p_opt: int = 0
    """p с максимальной D₂ в надёжной зоне (R² ≥ r2_min)."""

    curve: dict = field(default_factory=dict)
    """D₂(p) для каждого p: {p: d2}."""

    r2_curve: dict = field(default_factory=dict)
    """R² линейного участка C(r): {p: r2}."""

    n_points: int = 0
    """Число пивотов в облаке."""

    n_pivots_per_T: dict = field(default_factory=dict)
    """Число пивотов от каждого T: {T: n}."""

    origin: int = 0
    """Точка отсчёта (индекс в исходном ряду)."""

    p_max: int = P_MAX_DEFAULT

    def __str__(self) -> str:
        lines = [
            f"D₂Result  origin={self.origin}  N={self.n_points}  p_max={self.p_max}",
        ]
        if self.n_pivots_per_T:
            ppt = "  ".join(f"T={t:.3f}:{n}" for t, n in self.n_pivots_per_T.items())
            lines.append(f"  Пивотов            : {ppt}")
        lines += [
            f"  Фрактальная D₂     : {self.d2_fractal:.3f}",
            f"  p_opt              : {self.p_opt}",
            f"  Диапазон p (плато) : {self.p_range[0]} .. {self.p_range[1]}",
            "",
            f"  {'p':>4}  {'D₂':>7}  {'R²':>7}",
            f"  {'─'*26}",
        ]
        for p in range(1, self.p_max + 1):
            d2 = self.curve.get(p, float("nan"))
            r2 = self.r2_curve.get(p, float("nan"))
            if p == self.p_opt:
                mark = " ★"
            elif self.p_range[0] <= p <= self.p_range[1]:
                mark = " ←"
            else:
                mark = ""
            if np.isnan(d2):
                lines.append(f"  {p:>4}  {'—':>7}  {'—':>7}{mark}")
            else:
                lines.append(f"  {p:>4}  {d2:>7.3f}  {r2:>7.3f}{mark}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Зигзаг
# ════════════════════════════════════════════════════════════════════════════

def _find_pivots(highs: np.ndarray, lows: np.ndarray, thr: float) -> np.ndarray:
    """
    Каузальный зигзаг по high/low.
    Возвращает массив цен пивотов (H и L вместе, в событийном порядке).
    """
    prices = []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1; ext_val = highs[i]
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val = lows[i]
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val = highs[i]
            elif ext_val - lows[i] >= thr * ext_val:
                prices.append(ext_val)
                direction = -1; ext_val = lows[i]
        else:
            if lows[i] < ext_val:
                ext_val = lows[i]
            elif highs[i] - ext_val >= thr * ext_val:
                prices.append(ext_val)
                direction = 1; ext_val = highs[i]
    return np.array(prices, dtype=np.float64)


def _find_pivots_with_idx(highs: np.ndarray, lows: np.ndarray,
                          thr: float) -> tuple:
    """
    Как _find_pivots, но возвращает (prices, bar_indices).
    bar_indices — индекс бара в обрезанном ряду, где зафиксирован пивот.
    Нужен для объединения нескольких T в правильном временно́м порядке.
    """
    prices, idxs = [], []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    ext_idx = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1; ext_val = highs[i]; ext_idx = i
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val = lows[i]; ext_idx = i
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val = highs[i]; ext_idx = i
            elif ext_val - lows[i] >= thr * ext_val:
                prices.append(ext_val); idxs.append(ext_idx)
                direction = -1; ext_val = lows[i]; ext_idx = i
        else:
            if lows[i] < ext_val:
                ext_val = lows[i]; ext_idx = i
            elif highs[i] - ext_val >= thr * ext_val:
                prices.append(ext_val); idxs.append(ext_idx)
                direction = 1; ext_val = highs[i]; ext_idx = i
    return (np.array(prices, dtype=np.float64),
            np.array(idxs,   dtype=np.int64))


# ════════════════════════════════════════════════════════════════════════════
# Вложение
# ════════════════════════════════════════════════════════════════════════════

def build_hybrid_embedding(prices: np.ndarray, p: int) -> np.ndarray:
    """
    X[i] = [price_i, lr₁, ..., lr_{p-1}],  lr_k = log(p_{i-k+1}/p_{i-k}).
    Первые p-1 строк — NaN.
    """
    n  = len(prices)
    lp = np.log(np.maximum(prices, 1e-12))
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


# ════════════════════════════════════════════════════════════════════════════
# D₂
# ════════════════════════════════════════════════════════════════════════════

def _d2_one(X_norm: np.ndarray, orig_indices: np.ndarray, n_r: int) -> tuple:
    """
    D₂ и R² для нормированного облака.

    Args:
        X_norm:       нормированные векторы вложения (уже субсэмплированные).
        orig_indices: исходные позиции каждой строки в массиве prices[].
                      Нужны для коррекции Тейлера: пары с |i-j| ≤ THEILER_W
                      исключаются из подсчёта (они автокоррелированы).
        n_r:          число точек сетки r для C(r).
    """
    N = len(X_norm)
    if N < MIN_POINTS:
        return np.nan, np.nan

    d_flat = pdist(X_norm).astype(np.float32)

    # Коррекция Тейлера: исключаем пары с |i-j| ≤ THEILER_W
    if THEILER_W > 0:
        ii, jj   = np.triu_indices(N, k=1)
        valid    = np.abs(orig_indices[ii] - orig_indices[jj]) > THEILER_W
        d_valid  = d_flat[valid]
    else:
        d_valid = d_flat

    d_pos = d_valid[d_valid > 1e-9]
    if len(d_pos) < 10:
        return np.nan, np.nan

    r_min = float(np.percentile(d_pos, 5))
    r_max = float(np.percentile(d_pos, R_MAX_PCT))
    if r_min >= r_max:
        return np.nan, np.nan

    n_pairs = len(d_valid)
    r_arr = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    C_arr = np.array([np.sum(d_valid < r) / n_pairs for r in r_arr],
                     dtype=np.float32)

    mask = C_arr > 0
    if mask.sum() < 4:
        return np.nan, np.nan

    lr = np.log10(r_arr[mask])
    lc = np.log10(C_arr[mask])
    lo = int(len(lr) * LIN_LO)
    hi = int(len(lr) * LIN_HI)
    if hi - lo < 3:
        lo, hi = 0, len(lr)

    seg_lr, seg_lc = lr[lo:hi], lc[lo:hi]
    coef = np.polyfit(seg_lr, seg_lc, 1)
    d2   = float(coef[0])

    ss_res = np.sum((seg_lc - np.polyval(coef, seg_lr)) ** 2)
    ss_tot = np.sum((seg_lc - seg_lc.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    return d2, r2


def _plateau_range(curve: dict, r2_curve: dict,
                   plateau_tol: float, r2_min: float) -> tuple:
    """
    Диапазон p плато: R² ≥ r2_min и D₂ ≥ (1-plateau_tol)×D₂_max.
    Возвращает (p_lo, p_hi, p_opt).
    """
    ps = sorted(curve.keys())
    reliable = [
        (p, curve[p]) for p in ps
        if not np.isnan(curve.get(p, np.nan))
        and r2_curve.get(p, 0.0) >= r2_min
    ]
    if not reliable:
        return ps[-1], ps[-1], ps[-1]

    d2_max = max(v for _, v in reliable)
    p_opt  = next(p for p, v in reliable if v == d2_max)
    if d2_max < 1e-6:
        return ps[-1], ps[-1], p_opt

    threshold = (1.0 - plateau_tol) * d2_max
    plateau   = [p for p, v in reliable if v >= threshold]
    if not plateau:
        return p_opt, p_opt, p_opt

    return min(plateau), max(plateau), p_opt


# ════════════════════════════════════════════════════════════════════════════
# Основные функции
# ════════════════════════════════════════════════════════════════════════════

def measure_d2(
    prices:      np.ndarray,
    p_max:       int   = P_MAX_DEFAULT,
    n_r:         int   = N_R_DEFAULT,
    plateau_tol: float = PLATEAU_TOL,
    r2_min:      float = R2_MIN_DEFAULT,
) -> D2Result:
    """
    Измеряет D₂ по готовому массиву цен пивотов.

    Низкоуровневая функция — не знает об origin и зигзаге.
    Используйте measure_d2_causal() для работы с сырыми барными данными.
    """
    if len(prices) < MIN_POINTS + p_max:
        return D2Result(d2_fractal=np.nan, p_range=(p_max, p_max),
                        n_points=len(prices), p_max=p_max)

    X_big = build_hybrid_embedding(prices, p_max)
    valid_mask   = ~np.any(np.isnan(X_big), axis=1)
    orig_indices = np.where(valid_mask)[0]   # позиция в prices[] для Тейлера
    X_big        = X_big[valid_mask]

    if len(X_big) < MIN_POINTS:
        return D2Result(d2_fractal=np.nan, p_range=(p_max, p_max),
                        n_points=len(X_big), p_max=p_max)

    # нормировка по полному облаку — одни μ/σ для всех p
    mu_full  = X_big.mean(axis=0)
    sig_full = X_big.std(axis=0)
    sig_full = np.where(sig_full < 1e-12, 1.0, sig_full)

    # субсэмплинг один раз — одни и те же точки для всех p
    if len(X_big) > MAX_SUBSAMPLE:
        rng      = np.random.default_rng(42)
        sub_idx  = np.sort(rng.choice(len(X_big), MAX_SUBSAMPLE, replace=False))
        X_sub    = X_big[sub_idx]
        orig_sub = orig_indices[sub_idx]
    else:
        X_sub    = X_big
        orig_sub = orig_indices

    curve:    dict = {}
    r2_curve: dict = {}

    for p in range(1, p_max + 1):
        Xp   = X_sub[:, :p]
        Xp_n = (Xp - mu_full[:p]) / sig_full[:p]
        curve[p], r2_curve[p] = _d2_one(Xp_n, orig_sub, n_r)

    p_lo, p_hi, p_opt = _plateau_range(curve, r2_curve, plateau_tol, r2_min)
    plateau_vals = [curve[p] for p in range(p_lo, p_hi + 1)
                    if not np.isnan(curve.get(p, np.nan))]
    d2_fractal   = float(np.median(plateau_vals)) if plateau_vals else np.nan

    return D2Result(
        d2_fractal = d2_fractal,
        p_range    = (p_lo, p_hi),
        p_opt      = p_opt,
        curve      = curve,
        r2_curve   = r2_curve,
        n_points   = len(X_sub),
        p_max      = p_max,
    )


def measure_d2_causal(
    highs:       np.ndarray,
    lows:        np.ndarray,
    origin:      int,
    thresholds:  List[float],
    p_max:       int   = P_MAX_DEFAULT,
    n_r:         int   = N_R_DEFAULT,
    plateau_tol: float = PLATEAU_TOL,
    r2_min:      float = R2_MIN_DEFAULT,
) -> D2Result:
    """
    Измеряет D₂ с гарантированной каузальностью.

    Шаги:
      1. Обрезка: highs[:origin], lows[:origin] — строго до точки прогноза.
      2. Построение пивотов зигзага из обрезанных данных для каждого T.
      3. Если thresholds содержит несколько T — пивоты объединяются
         и сортируются по временно́му индексу бара.
      4. Измерение D₂ по объединённому облаку.

    Args:
        highs:      массив максимумов баров.
        lows:       массив минимумов баров.
        origin:     индекс точки отсчёта (не включается: данные = [:origin]).
        thresholds: список T зигзага, например [0.04] или [0.04, 0.004].
        p_max:      максимальная размерность sweep.
        n_r:        число r-точек для C(r).
        plateau_tol:ширина плато (10% D₂_max по умолчанию).
        r2_min:     минимальный R² линейного участка.

    Returns:
        D2Result.
    """
    # ── КАУЗАЛЬНАЯ ОБРЕЗКА ────────────────────────────────────────────────────
    assert 0 < origin <= len(highs), (
        f"origin={origin} вне диапазона [1, {len(highs)}]"
    )
    highs_c = highs[:origin]   # данные строго до точки прогноза
    lows_c  = lows[:origin]    # дальше алгоритм не видит ничего после origin
    # ─────────────────────────────────────────────────────────────────────────

    if not thresholds:
        raise ValueError("thresholds не должен быть пустым")

    # строим пивоты для каждого T из обрезанных данных
    all_prices: List[float] = []
    all_idxs:   List[int]   = []
    n_pivots_per_T: dict    = {}

    for T in thresholds:
        prices_T, idxs_T = _find_pivots_with_idx(highs_c, lows_c, T)
        n_pivots_per_T[T] = len(prices_T)
        all_prices.extend(prices_T.tolist())
        all_idxs.extend(idxs_T.tolist())

    if not all_prices:
        return D2Result(d2_fractal=np.nan, p_range=(p_max, p_max),
                        n_points=0, n_pivots_per_T=n_pivots_per_T,
                        origin=origin, p_max=p_max)

    if len(thresholds) > 1:
        # сортируем по индексу бара, чтобы сохранить событийный порядок
        order  = np.argsort(all_idxs, kind="stable")
        prices = np.array(all_prices)[order]
    else:
        prices = np.array(all_prices)

    result = measure_d2(prices, p_max=p_max, n_r=n_r,
                        plateau_tol=plateau_tol, r2_min=r2_min)
    result.n_pivots_per_T = n_pivots_per_T
    result.origin         = origin
    return result


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _load_json(path: str) -> tuple:
    """Загружает JSON свечей MOEX ISS. Возвращает (highs, lows)."""
    with open(path) as f:
        raw = json.load(f)
    highs = np.array([d["high"] for d in raw], dtype=np.float64)
    lows  = np.array([d["low"]  for d in raw], dtype=np.float64)
    return highs, lows


def _demo():
    import os
    here = os.path.dirname(__file__)
    data = os.path.join(here, "../../data/candles/SBER/10m.json")

    print("ДЕМО: SBER 10m")
    print("─" * 50)
    highs, lows = _load_json(data)
    n_bars = len(highs)
    print(f"Всего баров 10m: {n_bars}")
    print("(origin — индекс бара в ряду, не индекс пивота)\n")

    # origin подобран так чтобы получить ~200 / ~400 / все пивоты T=4%
    for origin, T_list in [
        (64_000,  [0.04]),           # ~200 пивотов T=4%
        (128_000, [0.04]),           # ~400 пивотов T=4%
        (n_bars,  [0.04]),           # все пивоты T=4%
        (n_bars,  [0.04, 0.004]),    # T=4% + T=0.4%
    ]:
        label = f"origin={origin}  T={T_list}"
        print(f"{label}")
        r = measure_d2_causal(highs, lows, origin=origin, thresholds=T_list)
        print(r)
        print()


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--demo" in args or not args:
        _demo()
        sys.exit(0)

    # --data path --origin N --T t1 [t2 ...]
    def _get_one(flag):
        """Возвращает один аргумент после флага."""
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args):
                return args[i + 1]
        return None

    def _get_multi(flag):
        """Возвращает все аргументы после флага до следующего флага (--...)."""
        if flag not in args:
            return None
        i = args.index(flag) + 1
        result = []
        while i < len(args) and not args[i].startswith("--"):
            result.append(args[i])
            i += 1
        return result if result else None

    data_path_s = _get_one("--data")
    origin_s    = _get_one("--origin")
    t_args      = _get_multi("--T")

    if not data_path_s or not origin_s or not t_args:
        print("Использование:")
        print("  python d2_embedding.py --data path.json --origin N --T 0.04 [0.004 ...]")
        print("  python d2_embedding.py --demo")
        sys.exit(1)

    data_path  = data_path_s
    origin     = int(origin_s)
    thresholds = [float(t) for t in t_args]

    highs, lows = _load_json(data_path)
    result = measure_d2_causal(highs, lows, origin=origin,
                                thresholds=thresholds)
    print(result)
