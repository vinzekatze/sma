"""
99_pocket_indicators — сравнение индикаторов качества «кармана» локального аттрактора.

Гипотеза: области фазового пространства, где LWR даёт хороший прогноз att,
характеризуются измеримыми геометрическими свойствами. Проверяем 4 индикатора:

  rho_A    — спектр. радиус якобиана LWR (companion matrix; < 1 → локально сжимающийся поток)
  d_L      — локальная размерность TwoNN на K_IND=50 соседях att
  DET      — детерминизм локального RQA (K_IND×K_IND; высокий → детерминированная область)
  loo_mape — LOO-MAPE LWR на XI=35 acc_ang-соседях (прокси val_mape)

Целевая переменная: test_rMAE_att = MAE(forecast, actual) / MAE(naive_last, actual)
Анализ: ранговая корреляция Спирмена каждого индикатора с test_rMAE_att.

Causal LP:
  att вычисляется на скользящем окне LP_W=600 баров, заканчивающемся на origin.
  Данные после origin не используются.

Запуск:
  python 99_pocket_indicators.py          # полный прогон (8 тикеров, 100 origins)
  python 99_pocket_indicators.py --test   # 2 тикера, 10 origins
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

# ── Пути ──────────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).resolve().parent
ROOT    = _HERE.parents[2]   # sma/
DATADIR = ROOT / "data" / "candles"
RESDIR  = _HERE / "results"
FIGDIR  = _HERE / "figures"

# ── Параметры ─────────────────────────────────────────────────────────────────
TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
TICKERS_TEST = ["SBER", "CHMF"]
INTERVAL     = "1d"

N_ORIGINS = 100        # origins на тикер (последние доступные)
T_AHEAD   = 10         # горизонт прогноза и test_rMAE

P_FIT  = 9             # размерность вложения (delay embedding)
P_MAX  = 288           # максимальная размерность каскада (октава от P_FIT: 9→18→36→72→144→288)
XI     = 35            # соседей для LWR: 3*(P_FIT+1) + 5
K_IND  = 50            # соседей для TwoNN и DET
LAMBDA = 0.01          # вес acc_ang в комбинированном расстоянии

LP_W = 600             # ширина скользящего окна causal LP
LP_M, LP_D, LP_K, LP_N = 9, 3, 30, 3   # параметры LP-фильтра

DET_EPS_PCTILE = 20    # ε = 20-й перцентиль попарных расстояний среди соседей
DET_LMIN       = 2     # минимальная длина диагональной линии для DET

HIST_SIZE = 3000       # длина истории att_point для режима --longhist

# ── Выбор режима ──────────────────────────────────────────────────────────────
# --lp2000   : Path 1 — LP_W=2000 (широкое скользящее окно)
# --longhist : Path 2 — LP_W=600 + библиотека из 3000 накопленных att_point
TEST_MODE   = "--test"     in sys.argv
LP2000_MODE = "--lp2000"   in sys.argv
LONGHIST_MODE = "--longhist" in sys.argv

if LP2000_MODE:
    LP_W = 2000


# ══════════════════════════════════════════════════════════════════════════════
# Данные
# ══════════════════════════════════════════════════════════════════════════════

def load_close(ticker: str) -> np.ndarray:
    path = DATADIR / ticker / f"{INTERVAL}.json"
    data = json.loads(path.read_text())
    return np.array([c["close"] for c in data], dtype=np.float64)


def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n   = len(close)
    lc  = np.log(np.maximum(close, 1e-10))
    t   = np.arange(n, dtype=np.float64)
    cn  = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    den = cn * ct2 - ct ** 2
    b   = np.where(den > 0, (cn * cty - ct * cy) / den, 0.0)
    a   = (cy - b * ct) / cn
    tr  = np.exp(a + b * t)
    tr[:2] = close[:2]
    return tr


# ── LP-фильтр (быстрый, векторизованный, без утечки за пределы окна) ─────────

def lp_filter(x: np.ndarray, m: int, d: int, k: int, n_iter: int) -> np.ndarray:
    """
    Local Projective noise reduction на ФИКСИРОВАННОМ окне x.
    Возвращает np.diff(lp_ratio) — att длиной len(x)-1.

    Окно x заканчивается на t_orig+1 (ratio[t_orig+1] включён для вычисления diff).
    Данные за пределами окна не используются → нет утечки будущего.

    Внутри окна LP некаузален (соседи включают «ближайшее будущее» в пределах окна),
    но это стандартный компромисс для rolling-window подхода.
    """
    s = x.copy().astype(np.float64)
    N = len(s)
    for _ in range(n_iter):
        M     = N - m + 1
        X     = np.lib.stride_tricks.sliding_window_view(s, m).copy()  # (M, m)
        k_eff = min(k, M - 1)
        d_eff = min(d, k_eff - 1)

        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)
        nn_idx = np.argpartition(D, k_eff, axis=1)[:, :k_eff]  # (M, k_eff)
        del D

        # Батчевая PCA: covariance (M, m, m) → eigh → проекция
        nbrs    = X[nn_idx]                                   # (M, k_eff, m)
        centers = nbrs.mean(axis=1, keepdims=True)            # (M, 1, m)
        nbrs_c  = nbrs - centers                              # (M, k_eff, m)
        C       = np.einsum("bki,bkj->bij", nbrs_c, nbrs_c)  # (M, m, m)
        _, vecs = np.linalg.eigh(C)                           # (M, m), (M, m, m)
        Vd      = vecs[:, :, -d_eff:]                         # (M, m, d_eff) — top d
        xc      = X - centers[:, 0, :]                        # (M, m)
        coef    = np.einsum("bm,bmd->bd", xc, Vd)             # (M, d_eff)
        proj    = np.einsum("bd,bmd->bm", coef, Vd)           # (M, m)
        Xp      = centers[:, 0, :] + proj                     # (M, m)

        res = np.zeros(N); cnt = np.zeros(N, int)
        idx2d = np.arange(M)[:, None] + np.arange(m)[None, :]
        np.add.at(res, idx2d, Xp)
        np.add.at(cnt, idx2d, 1)
        s = res / np.maximum(cnt, 1)

    return np.diff(s)   # att, длина N-1


def compute_att_local(ratio: np.ndarray, t_orig: int) -> np.ndarray | None:
    """
    Causal att для origin t_orig.
    Использует ratio[t_orig-LP_W+2 : t_orig+2] — окно LP_W баров,
    заканчивающееся включительно ratio[t_orig+1].
    Возвращает att длиной LP_W-1; att[-1] = att в точке t_orig.
    """
    start = t_orig - LP_W + 2
    end   = t_orig + 2          # slicing: ratio[start:end] → LP_W баров
    if start < 0 or end > len(ratio):
        return None
    return lp_filter(ratio[start:end], LP_M, LP_D, LP_K, LP_N)


# ── Тестовые актуальные значения att ─────────────────────────────────────────

def precompute_att_points(ratio: np.ndarray, t_start: int, t_end: int) -> dict[int, float]:
    """
    Для каждого t в [t_start, t_end] вычисляет att_point[t] =
    последний элемент att из causal LP(ratio[t-LP_W+2 : t+2]).
    Используется как «истинное» att для test_rMAE.
    """
    result: dict[int, float] = {}
    for t in range(t_start, t_end + 1):
        att_loc = compute_att_local(ratio, t)
        if att_loc is not None:
            result[t] = float(att_loc[-1])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Библиотека (пространство задержек att)
# ══════════════════════════════════════════════════════════════════════════════

def build_library(att_local: np.ndarray):
    """
    att_local: att в окне LP_W-1 значений, заканчивающемся в t_orig.

    Возвращает две библиотеки:
      dense  (P_FIT dim, n_dense pts) — для TwoNN и DET (плотная, больше точек)
      cascade (P_MAX dim, n_casc pts) — для каскадного LWR

    Схема индексации (оба варианта):
      X[j]: вектор состояния; y[j]: следующее att (цель)
      query: текущее состояние att[-P_FIT:] (запрос для прогноза)
    """
    n = len(att_local)  # LP_W - 1 = 599
    n_casc  = n - P_MAX              # 599-288 = 311 точек
    n_dense = n - P_FIT              # 599-9   = 590 точек

    if n_casc < XI + 5 or n_dense < K_IND + 5:
        return None

    # Ускорение (общее)
    acc = np.zeros(n)
    acc[2:] = att_local[2:] - 2 * att_local[1:-1] + att_local[:-2]

    # ── Плотная библиотека (P_FIT-мерная, 590 точек) для TwoNN и DET ───────────
    idx_d  = np.arange(n_dense)[:, None] + np.arange(P_FIT)[None, :]
    X_dense = att_local[idx_d]                       # (590, P_FIT)
    y_dense = att_local[P_FIT : P_FIT + n_dense]     # (590,)

    # ── Каскадная библиотека (P_MAX-мерная, 311 точек) для LWR ─────────────────
    idx_c    = np.arange(n_casc)[:, None] + np.arange(P_MAX)[None, :]
    X_full   = att_local[idx_c]                      # (311, P_MAX)
    y_casc   = att_local[P_MAX : P_MAX + n_casc]     # (311,)
    X_acc_full = acc[idx_c]                          # (311, P_MAX)

    # ── Запросы ─────────────────────────────────────────────────────────────────
    query      = att_local[-P_FIT:].copy()           # (P_FIT,)  для плотного поиска
    query_full = att_local[-P_MAX:].copy()           # (P_MAX,)  для каскада
    q_acc_full = acc[-P_MAX:].copy()                 # (P_MAX,)

    return (X_dense, y_dense,
            X_full, y_casc, X_acc_full,
            query, query_full, q_acc_full,
            n_dense, n_casc)


def build_library_longhist(att_history: np.ndarray):
    """
    Path 2 — библиотека из HIST_SIZE накопленных att_point значений.

    att_history: 1D массив длиной HIST_SIZE+1 = att_point[t_orig-HIST_SIZE : t_orig+1]
    (все значения — последние элементы LP на разных rolling-окнах, каузальны).

    Возвращает те же поля, что build_library, но n_casc ≈ HIST_SIZE-P_MAX ~ 2712.
    """
    n = len(att_history)           # HIST_SIZE + 1
    n_casc  = n - P_MAX            # ~2712
    n_dense = n - P_FIT            # ~2991

    if n_casc < XI + 5 or n_dense < K_IND + 5:
        return None

    acc = np.zeros(n)
    acc[2:] = att_history[2:] - 2 * att_history[1:-1] + att_history[:-2]

    idx_d   = np.arange(n_dense)[:, None] + np.arange(P_FIT)[None, :]
    X_dense = att_history[idx_d]
    y_dense = att_history[P_FIT : P_FIT + n_dense]

    idx_c      = np.arange(n_casc)[:, None] + np.arange(P_MAX)[None, :]
    X_full     = att_history[idx_c]
    y_casc     = att_history[P_MAX : P_MAX + n_casc]
    X_acc_full = acc[idx_c]

    query      = att_history[-P_FIT:].copy()
    query_full = att_history[-P_MAX:].copy()
    q_acc_full = acc[-P_MAX:].copy()

    return (X_dense, y_dense,
            X_full, y_casc, X_acc_full,
            query, query_full, q_acc_full,
            n_dense, n_casc)


# ══════════════════════════════════════════════════════════════════════════════
# Октавный каскад
# ══════════════════════════════════════════════════════════════════════════════

def _cascade_levels() -> list[int]:
    """Октавные уровни от P_MAX до P_FIT (убывание вдвое)."""
    levs = [P_FIT]
    p = P_FIT
    while p * 2 <= P_MAX:
        p *= 2
        levs.append(p)
    return list(reversed(levs))   # [288, 144, 72, 36, 18, 9]


_LEVELS = _cascade_levels()  # вычисляем один раз


def cascade_neighbors(
    X_full: np.ndarray,
    X_acc_full: np.ndarray,
    y_casc: np.ndarray,
    query_full: np.ndarray,
    q_acc_full: np.ndarray,
    n_casc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Октавный каскад: поиск XI лучших соседей в n_casc-точечной библиотеке.

    Шаги:
      - Начинаем с ВСЕХ кандидатов
      - На каждом уровне p_lvl: сужаем до XI ближайших по p_lvl последним изм.
      - Расширяем временными соседями (radius = p_lvl - p_next)
      - На последнем уровне P_FIT: финальный отбор по d_pos + λ·d_acc

    Возвращает:
      X_nn  (XI, P_FIT)  — P_FIT-мерные векторы для LWR
      y_nn  (XI,)        — цели
      q_pfit (P_FIT,)    — P_FIT-мерный запрос (= query_full[-P_FIT:])
    """
    cands = np.arange(n_casc)
    levels = _LEVELS

    for k_lev, p_lvl in enumerate(levels):
        if k_lev == len(levels) - 1:
            break
        xi_lvl = min(XI, len(cands))
        if len(cands) > xi_lvl:
            # Сужение: последние p_lvl измерений (наиболее свежие)
            cols  = P_MAX - p_lvl   # slice [cols:]
            d_arr = np.linalg.norm(
                X_full[cands, cols:] - query_full[cols:], axis=1
            )
            top   = np.argpartition(d_arr, xi_lvl - 1)[:xi_lvl]
            cands = cands[top]
        # Расширение временными соседями
        radius = p_lvl - levels[k_lev + 1]
        if radius > 0:
            exp   = cands[:, None] - np.arange(radius + 1)[None, :]
            cands = np.unique(np.clip(exp, 0, n_casc - 1))

    # Финальный отбор: d_pos + λ·d_acc в P_FIT измерениях
    q_pfit  = query_full[-P_FIT:]
    qa_pfit = q_acc_full[-P_FIT:]
    cols_p  = P_MAX - P_FIT          # slice [cols_p:]
    if len(cands) > XI:
        d_pos = np.linalg.norm(X_full[cands, cols_p:] - q_pfit, axis=1)
        d_acc = _cosine_dist(X_acc_full[cands, cols_p:], qa_pfit)
        top   = np.argpartition(d_pos + LAMBDA * d_acc, XI - 1)[:XI]
        cands = cands[top]

    if len(cands) < P_FIT + 2:
        return None, None, None

    X_nn = X_full[cands, cols_p:]   # (<=XI, P_FIT)
    y_nn = y_casc[cands]
    return X_nn, y_nn, q_pfit


# ══════════════════════════════════════════════════════════════════════════════
# Поиск соседей
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_dist(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    nA = np.linalg.norm(A, axis=1)
    nb = float(np.linalg.norm(b))
    if nb < 1e-12:
        return np.ones(len(A))
    return 1.0 - np.clip((A @ b) / np.where(nA > 1e-12, nA, 1.0) / nb, -1.0, 1.0)


def find_neighbors_l2(X_lib: np.ndarray, query: np.ndarray, k: int):
    """k ближайших соседей по L2 в att-пространстве."""
    dists = np.linalg.norm(X_lib - query, axis=1)
    k_eff = min(k, len(X_lib))
    idx   = np.argpartition(dists, k_eff - 1)[:k_eff]
    return idx, dists[idx]


def find_neighbors_acc(
    X_lib: np.ndarray, query: np.ndarray,
    X_acc: np.ndarray, q_acc: np.ndarray, xi: int,
) -> np.ndarray:
    """xi соседей по комбинированной метрике d_pos + LAMBDA * d_acc."""
    d_pos = np.linalg.norm(X_lib - query, axis=1)
    d_acc = _cosine_dist(X_acc, q_acc)
    combined = d_pos + LAMBDA * d_acc
    xi_eff = min(xi, len(X_lib))
    return np.argpartition(combined, xi_eff - 1)[:xi_eff]


# ══════════════════════════════════════════════════════════════════════════════
# Индикаторы кармана
# ══════════════════════════════════════════════════════════════════════════════

def _lwr_weights(X_nn: np.ndarray, q: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(X_nn - q, axis=1)
    h = max(float(d.max()), 1e-10)
    return np.exp(-0.5 * (d / h) ** 2)


def _lwr_fit(X_nn: np.ndarray, y_nn: np.ndarray, q: np.ndarray):
    """Возвращает (prediction, coefficients c[0]=intercept, c[1:]=slope)."""
    w  = _lwr_weights(X_nn, q)
    sw = np.sqrt(np.maximum(w, 1e-30))
    A  = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + q @ c[1:]), c


def indicator_rho_A(X_nn: np.ndarray, y_nn: np.ndarray, q: np.ndarray) -> float:
    """
    Спектральный радиус companion-матрицы якобиана LWR.

    В delay-координатах переход x→x' = (att[t+1], att[t], ..., att[t-p+2]).
    Якобиан — companion matrix: первая строка = c[1:] (градиент LWR),
    остальные — единичный сдвиг.
    ρ(J) < 1 → локально сжимающийся поток → хороший карман.
    """
    if len(X_nn) < P_FIT + 2:
        return np.nan
    _, c = _lwr_fit(X_nn, y_nn, q)
    p    = P_FIT
    J    = np.zeros((p, p))
    J[0, :] = c[1:p + 1]
    J[1:, :-1] = np.eye(p - 1)
    eigvals = np.linalg.eigvals(J)
    return float(np.max(np.abs(eigvals)))


def indicator_twonn(X_nn: np.ndarray) -> float:
    """
    Локальная размерность через TwoNN (Facco et al. 2017).
    d_L = (k-1) / sum(log(r2/r1)) — MLE оценка Pareto.
    d_L << P_FIT → гладкий низкоразмерный карман.
    """
    k = len(X_nn)
    if k < 3:
        return np.nan
    D = np.sum((X_nn[:, None, :] - X_nn[None, :, :]) ** 2, axis=2)  # (k, k)
    np.fill_diagonal(D, np.inf)
    top2 = np.partition(D, 1, axis=1)[:, :2]
    r1   = np.sqrt(np.maximum(top2[:, 0], 1e-30))
    r2   = np.sqrt(np.maximum(top2[:, 1], 1e-30))
    mu   = r2 / r1
    mu   = mu[mu > 1.0]
    if len(mu) < 2:
        return np.nan
    return float((len(mu) - 1) / np.sum(np.log(mu)))


def indicator_det(X_nn: np.ndarray) -> float:
    """
    Детерминизм локального RQA.
    DET = доля рекуррентных точек на диагональных линиях длиной >= DET_LMIN.
    DET высокий → область детерминирована → хороший карман.
    """
    k = len(X_nn)
    if k < 4:
        return np.nan
    D = np.sum((X_nn[:, None, :] - X_nn[None, :, :]) ** 2, axis=2)
    mask = D < np.inf
    eps_sq = np.percentile(D[mask], DET_EPS_PCTILE)
    R      = (D <= eps_sq).astype(np.int8)
    np.fill_diagonal(R, 0)

    total_rec = int(R.sum())
    if total_rec == 0:
        return 0.0

    diag_pts = 0
    for lag in range(-(k - 1), k):
        if lag == 0:
            continue
        diag = np.diag(R, lag)
        if len(diag) < DET_LMIN:
            continue
        padded = np.concatenate([[0], diag, [0]])
        starts = np.where(np.diff(padded) == 1)[0]
        ends   = np.where(np.diff(padded) == -1)[0]
        lengths = ends - starts
        diag_pts += int(np.sum(lengths[lengths >= DET_LMIN]))

    return diag_pts / total_rec


def indicator_loo_mape(X_nn: np.ndarray, y_nn: np.ndarray, q: np.ndarray) -> float:
    """LOO-MAPE LWR: для каждого соседа i подгоняем LWR на остальных, предсказываем y_i."""
    n = len(X_nn)
    if n < P_FIT + 3:
        return np.nan
    errors = []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        X_loo = X_nn[mask]; y_loo = y_nn[mask]; q_loo = X_nn[i]
        if len(X_loo) < P_FIT + 2:
            continue
        pred, _ = _lwr_fit(X_loo, y_loo, q_loo)
        denom = abs(y_nn[i])
        if denom > 1e-10:
            errors.append(abs(pred - y_nn[i]) / denom)
    return float(np.mean(errors)) if errors else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Прогноз
# ══════════════════════════════════════════════════════════════════════════════

def forecast_lwr(
    att_local: np.ndarray,
    X_full: np.ndarray, y_casc: np.ndarray, X_acc_full: np.ndarray,
    n_casc: int,
) -> list[float]:
    """
    Рекурсивный LWR-прогноз на T_AHEAD шагов с октавным каскадом.
    context хранит P_MAX последних att-значений (история + прогноз).
    """
    context = list(att_local[-P_MAX:])   # начальный контекст длины P_MAX
    preds: list[float] = []

    for _ in range(T_AHEAD):
        q_full = np.array(context[-P_MAX:])
        # ускорение из контекста
        acc_ctx = np.zeros(P_MAX)
        if len(context) >= 3:
            ctx = np.array(context[-P_MAX:])
            acc_ctx[2:] = ctx[2:] - 2 * ctx[1:-1] + ctx[:-2]
        qa_full = acc_ctx

        X_nn, y_nn, q_pfit = cascade_neighbors(
            X_full, X_acc_full, y_casc, q_full, qa_full, n_casc
        )
        if X_nn is None or len(X_nn) < P_FIT + 2:
            preds.append(context[-1])
            context.append(context[-1])
            continue

        pred, _ = _lwr_fit(X_nn, y_nn, q_pfit)
        preds.append(pred)
        context.append(pred)

    return preds


# ══════════════════════════════════════════════════════════════════════════════
# Один origin
# ══════════════════════════════════════════════════════════════════════════════

def run_origin(
    ratio: np.ndarray, t_orig: int, att_points: dict[int, float],
    att_history: np.ndarray | None = None,   # для longhist-режима
) -> dict | None:
    """Вычисляет все индикаторы и test_rMAE для одного origin."""

    # 1. Causal att (нужен для att_local[-P_MAX:] как контекст прогноза)
    att_local = compute_att_local(ratio, t_orig)
    if att_local is None:
        return None

    # 2. Две библиотеки: longhist или rolling-window
    if LONGHIST_MODE and att_history is not None:
        lib = build_library_longhist(att_history)
    else:
        lib = build_library(att_local)
    if lib is None:
        return None
    (X_dense, y_dense,
     X_full, y_casc, X_acc_full,
     query, query_full, q_acc_full,
     n_dense, n_casc) = lib

    # 3a. Соседи для TwoNN и DET: K_IND ближайших по L2 в P_FIT пространстве
    idx_ind, _ = find_neighbors_l2(X_dense, query, K_IND)
    if len(idx_ind) < 4:
        return None
    X_ind = X_dense[idx_ind]

    # 3b. Каскадные соседи для LWR, ρ(A), LOO-MAPE
    X_nn, y_nn, q_pfit = cascade_neighbors(
        X_full, X_acc_full, y_casc, query_full, q_acc_full, n_casc
    )
    if X_nn is None or len(X_nn) < P_FIT + 2:
        return None

    # 4. Индикаторы
    rho_A    = indicator_rho_A(X_nn, y_nn, q_pfit)
    d_L      = indicator_twonn(X_ind)
    DET      = indicator_det(X_ind)
    loo_mape = indicator_loo_mape(X_nn, y_nn, q_pfit)

    # 5. Прогноз с каскадом
    preds = forecast_lwr(att_local, X_full, y_casc, X_acc_full, n_casc)

    # 6. Тестовые актуальные значения и test_rMAE
    actuals = [att_points.get(t_orig + h) for h in range(1, T_AHEAD + 1)]
    if any(v is None for v in actuals):
        return None
    actuals = np.array(actuals, dtype=float)
    preds_a = np.array(preds[:T_AHEAD], dtype=float)

    last_att  = float(att_local[-1])
    naive     = np.full(T_AHEAD, last_att)
    mae_pred  = float(np.mean(np.abs(preds_a - actuals)))
    mae_naive = float(np.mean(np.abs(naive - actuals)))
    test_rMAE = mae_pred / (mae_naive + 1e-10)

    return {
        "t_orig":    t_orig,
        "rho_A":     rho_A,
        "d_L":       d_L,
        "DET":       DET,
        "loo_mape":  loo_mape,
        "test_rMAE": test_rMAE,
        "mae_pred":  mae_pred,
        "mae_naive": mae_naive,
        "n_lib":     n_casc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Один тикер
# ══════════════════════════════════════════════════════════════════════════════

def run_ticker(ticker: str, n_origins: int) -> list[dict]:
    print(f"\n{'─'*50}", flush=True)
    print(f"  {ticker}", flush=True)
    t0 = time.time()

    close = load_close(ticker)
    lt    = logtrend_causal(close)
    ratio = close / np.maximum(lt, 1e-10)
    N     = len(ratio)

    # Диапазон origins: последние n_origins, отступив T_AHEAD баров от конца
    t_last  = N - 1 - T_AHEAD
    t_first = t_last - n_origins + 1
    if t_first < LP_W + P_FIT:
        print(f"  Недостаточно данных для {ticker}", flush=True)
        return []

    origins = list(range(t_first, t_last + 1))

    # Предвычисление att_points для test_rMAE (всегда нужно)
    tp_start = t_first + 1
    tp_end   = t_last + T_AHEAD
    print(f"  Предвычисляем att_points [{tp_start}:{tp_end}]...", flush=True)
    att_points = precompute_att_points(ratio, tp_start, tp_end)

    # Path 2: предвычисление полной att_point-истории для longhist-библиотеки
    att_point_all: dict[int, float] = {}
    if LONGHIST_MODE:
        h_start = max(0, t_first - HIST_SIZE)
        print(f"  [longhist] Предвычисляем att_history [{h_start}:{t_last}]...",
              flush=True)
        att_point_all = precompute_att_points(ratio, h_start, t_last)
        print(f"  [longhist] {len(att_point_all)} точек готово", flush=True)

    rows: list[dict] = []
    for i, t_orig in enumerate(origins):
        # Для longhist: срез истории att_point для текущего origin
        att_history = None
        if LONGHIST_MODE:
            h_range = range(t_orig - HIST_SIZE, t_orig + 1)
            arr = np.array([att_point_all.get(t, np.nan) for t in h_range])
            if not np.any(np.isnan(arr)):
                att_history = arr

        rec = run_origin(ratio, t_orig, att_points, att_history)
        if rec is not None:
            rec["ticker"] = ticker
            rows.append(rec)
        if (i + 1) % 20 == 0 or (i + 1) == len(origins):
            print(f"  {i+1}/{len(origins)} origins, "
                  f"elapsed {time.time()-t0:.1f}s", flush=True)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Анализ: Spearman-корреляции
# ══════════════════════════════════════════════════════════════════════════════

INDICATORS = ["rho_A", "d_L", "DET", "loo_mape"]
IND_LABELS = {
    "rho_A":    "ρ(A) якобиан",
    "d_L":      "d_L TwoNN",
    "DET":      "DET RQA",
    "loo_mape": "LOO-MAPE",
}


def analyze(rows: list[dict]) -> str:
    if not rows:
        return "Нет данных."

    tickers = sorted({r["ticker"] for r in rows})
    lines   = []
    lines.append(f"\n{'═'*60}")
    lines.append(f"  Spearman rank-corr vs test_rMAE (N={len(rows)})")
    lines.append(f"{'═'*60}")
    lines.append(f"  {'Индикатор':<18} {'Global':>8} " +
                 "  ".join(f"{t:>6}" for t in tickers))
    lines.append(f"  {'-'*58}")

    for ind in INDICATORS:
        vals_all = np.array([r[ind] for r in rows], dtype=float)
        tgt_all  = np.array([r["test_rMAE"] for r in rows], dtype=float)
        valid    = np.isfinite(vals_all) & np.isfinite(tgt_all)
        if valid.sum() < 5:
            rg = "n/a"
        else:
            rg = f"{spearmanr(vals_all[valid], tgt_all[valid]).statistic:+.3f}"

        per_ticker = []
        for t in tickers:
            mask = np.array([r["ticker"] == t for r in rows])
            v2   = vals_all[mask]; tg2 = tgt_all[mask]
            ok   = np.isfinite(v2) & np.isfinite(tg2)
            if ok.sum() < 5:
                per_ticker.append("  n/a")
            else:
                r2 = spearmanr(v2[ok], tg2[ok]).statistic
                per_ticker.append(f"{r2:+.3f}")

        lines.append(f"  {IND_LABELS[ind]:<18} {rg:>8}  " +
                     "  ".join(f"{v:>6}" for v in per_ticker))

    lines.append(f"{'═'*60}")

    # Базовые статистики test_rMAE
    tgt = np.array([r["test_rMAE"] for r in rows], dtype=float)
    lines.append(f"\n  test_rMAE: mean={tgt.mean():.3f}  "
                 f"median={np.median(tgt):.3f}  "
                 f"std={tgt.std():.3f}  "
                 f"q25={np.percentile(tgt,25):.3f}  "
                 f"q75={np.percentile(tgt,75):.3f}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Визуализация
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter(rows: list[dict]) -> None:
    vals_all = {ind: np.array([r[ind] for r in rows], dtype=float)
                for ind in INDICATORS}
    tgt = np.array([r["test_rMAE"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Индикаторы кармана vs test_rMAE_att", fontsize=13)

    for ax, ind in zip(axes.flat, INDICATORS):
        v = vals_all[ind]
        ok = np.isfinite(v) & np.isfinite(tgt)
        ax.scatter(v[ok], tgt[ok], s=8, alpha=0.35, color="steelblue")
        ax.set_xlabel(IND_LABELS[ind])
        ax.set_ylabel("test_rMAE")
        if ok.sum() >= 5:
            r = spearmanr(v[ok], tgt[ok]).statistic
            ax.set_title(f"Spearman r = {r:+.3f}  (N={ok.sum()})")
        else:
            ax.set_title(f"n/a (N={ok.sum()})")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGDIR / "99_scatter.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Scatter → {path}", flush=True)


def plot_quantile_rMAE(rows: list[dict]) -> None:
    """Средний test_rMAE по квартилям каждого индикатора."""
    tgt = np.array([r["test_rMAE"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, len(INDICATORS), figsize=(14, 4))
    fig.suptitle("test_rMAE по квартилям индикатора", fontsize=12)

    for ax, ind in zip(axes.flat, INDICATORS):
        v  = np.array([r[ind] for r in rows], dtype=float)
        ok = np.isfinite(v) & np.isfinite(tgt)
        if ok.sum() < 8:
            ax.set_title(f"{IND_LABELS[ind]}\nn/a")
            continue
        qs = np.percentile(v[ok], [0, 25, 50, 75, 100])
        labels, means = [], []
        for qi in range(4):
            mask = (v[ok] >= qs[qi]) & (v[ok] < qs[qi + 1])
            if qi == 3:
                mask = (v[ok] >= qs[qi]) & (v[ok] <= qs[qi + 1])
            labels.append(f"Q{qi+1}")
            means.append(tgt[ok][mask].mean() if mask.sum() > 0 else np.nan)
        colors = ["#2ca02c", "#98df8a", "#ffbb78", "#d62728"]
        ax.bar(labels, means, color=colors)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.set_ylabel("mean test_rMAE")
        ax.set_title(IND_LABELS[ind])
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = FIGDIR / "99_quantiles.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Quantiles → {path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def _run_tag() -> str:
    if LP2000_MODE:   return "lp2000"
    if LONGHIST_MODE: return "longhist"
    return "lp600"


def main():
    tickers   = TICKERS_TEST if TEST_MODE else TICKERS
    n_origins = 10          if TEST_MODE else N_ORIGINS
    tag       = _run_tag()

    run_label = ("TEST" if TEST_MODE else "FULL") + f"/{tag.upper()}"
    lib_desc  = (f"LP_W={LP_W}, n_casc≈{LP_W-1-P_MAX}"
                 if not LONGHIST_MODE
                 else f"LP_W=600 + longhist={HIST_SIZE}, n_casc≈{HIST_SIZE-P_MAX}")

    print(f"{'='*60}", flush=True)
    print(f"  99_pocket_indicators  v1.1  [{tag}]", flush=True)
    print(f"  mode={run_label}  tickers={tickers}  origins={n_origins}", flush=True)
    print(f"  P_FIT={P_FIT}  P_MAX={P_MAX}  XI={XI}  K_IND={K_IND}", flush=True)
    print(f"  {lib_desc}", flush=True)
    print(f"{'='*60}", flush=True)

    t_global = time.time()
    all_rows: list[dict] = []

    for ticker in tickers:
        rows = run_ticker(ticker, n_origins)
        all_rows.extend(rows)

    print(f"\nВсего строк: {len(all_rows)}", flush=True)

    # ── Сохранение CSV ────────────────────────────────────────────────────────
    suffix   = f"_{tag}" if tag != "lp600" else ""
    csv_path = RESDIR / f"indicators{suffix}.csv"
    fieldnames = ["ticker", "t_orig", "rho_A", "d_L", "DET",
                  "loo_mape", "test_rMAE", "mae_pred", "mae_naive", "n_lib"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"  CSV → {csv_path}", flush=True)

    # ── Анализ ────────────────────────────────────────────────────────────────
    summary = analyze(all_rows)
    print(summary, flush=True)

    summary_path = RESDIR / f"summary{suffix}.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Summary → {summary_path}", flush=True)

    # ── Рисунки ───────────────────────────────────────────────────────────────
    if all_rows:
        plot_scatter(all_rows)
        plot_quantile_rMAE(all_rows)

    print(f"\nГотово за {time.time()-t_global:.1f}с", flush=True)


if __name__ == "__main__":
    main()
