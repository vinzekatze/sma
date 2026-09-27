"""
regime_mixture_potential — regime-conditioned pairlag lagged-ensemble +
GaussianMixture-сценарии потенциала, с премоткой по origin.

Третий, независимый model_type в архитектуре прогнозов (band_lambda —
полоса на пивотах зигзага; simplex_ensemble — ансамбль траекторий).
Отвечает на другой вопрос: не "куда пойдёт цена" и не "какой диапазон", а
"какие СЦЕНАРИИ (дискретные, не размытая полоса) вероятны на каждом шаге
h" — терминальное распределение на каждом h раскладывается на 1-3
Gaussian-компоненты (BIC), вместо одной гладкой плотности.

Портировано из prototype/forcaster/ui/app33-regime-mixture-rewind.py
(преемник app32, эксп.49 — research/phase19_monte_carlo_foundations/
49_regime_mixture_potential/), ядро — verbatim app31 (pairlag-дистанция,
combo-лаги эксп.48, Gaussian-кернель, lagged-ensemble).

Regime-фильтр — часть метода, не настройка: пул соседей ограничен тем же
терцилем трейлинг-волатильности (окно=60 баров), что и origin (fallback на
полный пул при <30 кандидатов в режиме). ⚠️ Проверен только для потенциала
— эксп.49 показал разнонаправленный эффект на риск (pinball), не
переносить на band_lambda/risk_corridor по аналогии.

Премотка — n_forecasts НЕЗАВИСИМЫХ origin (origin, origin−rewind_step, …),
каждый — полный цикл (causal-logtrend → dratio → режим → lagged-ensemble
пул → ресэмплинг путей → GMM-фит) — самая дорогая часть модуля, поэтому
может выполняться в ProcessPoolExecutor (см. run_multi_snapshot), как и
simplex_ensemble.run_multi_origin.

Causality contract (см. память feedback-causality-enforcement): origin =
len(close) - 1 — обеспечивается ЕДИНСТВЕННО тем, что вызывающий код
(sma/api/task_manager.py) грузит close/times уже обрезанными до origin.
Внутри премотки origin_i = origin - i*rewind_step получает СВОЮ причинную
обрезку (close[:origin_i+1]) — никакой утечки между снапшотами.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.mixture import GaussianMixture

from sma.core.forecast.simplex_ensemble import _logtrend_causal

# ── дефолты (см. index.html regime_mixture_potential-panel / app33) ────────

DEFAULT_HORIZON = 30
DEFAULT_N_FORECASTS = 10        # общее число прогнозов премотки (включая origin), мин=1
DEFAULT_REWIND_STEP = 1
DEFAULT_THETA = 5.0
DEFAULT_WARMUP = 250
DEFAULT_THEILER_WINDOW = 20
DEFAULT_BARS = 3000
DEFAULT_N_LOOKBACK = 8
DEFAULT_LOOKBACK_STEP = 1
DEFAULT_N_SIM = 10000
DEFAULT_SEED = 42
DEFAULT_MIX_N_RESAMPLE = 1500
DEFAULT_BIN_HEIGHT_PCT = 0.2
DEFAULT_COVERAGE_PCT = 95.0

# ═══════════════════════════════════════════════════════════════════════════
# Filter bank + pair-lag вектор — verbatim app31/app32/app33 (combo-лаги,
# эксп.48)
# ═══════════════════════════════════════════════════════════════════════════

FB_FILTER_ORDER = 4
FB_CUTOFFS = [0.25, 0.125, 0.0625, 0.03125, 0.015625]


def make_filter_bank(series: np.ndarray) -> np.ndarray:
    components: list[np.ndarray] = []
    remaining = series.copy()
    for fc in FB_CUTOFFS:
        sos = butter(FB_FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        components.append(remaining - low)
        remaining = low
    components.append(remaining)
    return np.array(components)


def _lags_from_cutoffs(cutoffs: list[float], divisor: float) -> list[int]:
    periods = [2.0] + [2.0 / fc for fc in cutoffs]
    return [max(1, int(round(t / divisor))) for t in periods]


LAGS_T2 = _lags_from_cutoffs(FB_CUTOFFS, 2.0)
LAGS_T4 = _lags_from_cutoffs(FB_CUTOFFS, 4.0)
LAGS_PER_COMP_COMBO = [[l4, l2] for l4, l2 in zip(LAGS_T4, LAGS_T2)]   # эксп.48 дефолт
P_DIM = sum(1 + len(l) for l in LAGS_PER_COMP_COMBO)                   # 18


def _pair_matrix(comp: np.ndarray, idxs: np.ndarray, lags_per_comp) -> np.ndarray:
    cols = []
    for i, lags in enumerate(lags_per_comp):
        cols.append(comp[i, idxs])
        for lag in lags:
            cols.append(comp[i, idxs - lag])
    return np.column_stack(cols)


def _pair_vector(comp: np.ndarray, idx: int, lags_per_comp) -> np.ndarray:
    parts = []
    for i, lags in enumerate(lags_per_comp):
        parts.append(comp[i, idx])
        for lag in lags:
            parts.append(comp[i, idx - lag])
    return np.array(parts)


MAX_LAG = max(lag for lags in LAGS_PER_COMP_COMBO for lag in lags)


# ═══════════════════════════════════════════════════════════════════════════
# Режим — трейлинг-волатильность (эксп.49). Терцили из всей причинной
# истории, доступной на момент прогноза.
# ═══════════════════════════════════════════════════════════════════════════

VOL_WINDOW = 60


def rolling_vol(dratio: np.ndarray, window: int = VOL_WINDOW) -> np.ndarray:
    n = len(dratio)
    vol = np.full(n, np.nan)
    c1 = np.cumsum(np.insert(dratio, 0, 0.0))
    c2 = np.cumsum(np.insert(dratio ** 2, 0, 0.0))
    for i in range(window - 1, n):
        s1 = c1[i + 1] - c1[i + 1 - window]
        s2 = c2[i + 1] - c2[i + 1 - window]
        mean = s1 / window
        vol[i] = np.sqrt(max(s2 / window - mean ** 2, 0.0))
    return vol


def regime_labels(vol: np.ndarray):
    valid = vol[np.isfinite(vol)]
    if len(valid) > 10:
        cutoffs = np.nanquantile(valid, [1 / 3, 2 / 3])
    else:
        cutoffs = np.array([np.nan, np.nan])
    labels = np.searchsorted(cutoffs, vol)
    labels = np.where(np.isfinite(vol), labels, -1)
    return labels, cutoffs


MIN_REGIME_POOL = 30

# ═══════════════════════════════════════════════════════════════════════════
# Пул соседей — pairlag-дистанция + regime-фильтр (часть метода, эксп.49)
# ═══════════════════════════════════════════════════════════════════════════

def build_pool_pairlag(dratio: np.ndarray, O: int, warmup: int, theiler_window: int,
                        h_max: int, theta: float, regime_lab: np.ndarray | None):
    """pairlag-дистанция (combo лаги) + Gaussian-кернель (без top-k) +
    regime-фильтр ПОСЛЕ Theiler, ДО нормировки sigma: кандидат допускается,
    только если его режим трейлинг-волатильности == режим origin. Fallback
    на полный (после Theiler) пул, если после фильтра осталось
    < MIN_REGIME_POOL."""
    t0 = max(warmup, MAX_LAG)
    if O - h_max < t0:
        return None, None, False
    comp_full = make_filter_bank(dratio[: O + 1])

    idx = np.arange(t0, O - h_max + 1)
    if len(idx) < 15:
        return None, None, False
    X = _pair_matrix(comp_full, idx, LAGS_PER_COMP_COMBO)
    query = _pair_vector(comp_full, O, LAGS_PER_COMP_COMBO)

    if theiler_window > 0:
        valid = np.abs(O - idx) > theiler_window
        if valid.sum() >= 15:
            idx, X = idx[valid], X[valid]

    finite_rows = np.all(np.isfinite(X), axis=1)
    idx, X = idx[finite_rows], X[finite_rows]
    if len(idx) < 15 or not np.all(np.isfinite(query)):
        return None, None, False

    regime_used = False
    if regime_lab is not None and 0 <= O < len(regime_lab) and regime_lab[O] >= 0:
        q_reg = regime_lab[O]
        cand_reg = regime_lab[idx]
        reg_mask = cand_reg == q_reg
        if reg_mask.sum() >= MIN_REGIME_POOL:
            idx, X = idx[reg_mask], X[reg_mask]
            regime_used = True

    sigma = np.maximum(np.std(X, axis=0), 1e-12)
    Xn, qn = X / sigma, query / sigma
    dist = np.linalg.norm(Xn - qn, axis=1)

    mean_d = np.mean(dist) if np.mean(dist) > 0 else 1.0
    bw = mean_d / theta
    w = np.exp(-0.5 * (dist / bw) ** 2)
    wsum = w.sum()
    if wsum <= 0 or not np.isfinite(wsum):
        return None, None, regime_used
    anchors = idx + 1
    return anchors, w / wsum, regime_used


def method_whole_analog(dratio, ratio_o, anchors, weights, h_max):
    offsets = np.arange(h_max)[None, :]
    idx = anchors[:, None] + offsets
    ratio_paths = ratio_o + np.cumsum(dratio[idx], axis=1)
    return ratio_paths, weights.copy()


def build_lagged_ensemble_ratio_paths(dratio, ratio_hist, O, h_ui, theta, n_lookback,
                                       lookback_step, warmup, theiler_window, regime_lab):
    cols_list, w_list = [], []
    any_regime_used = False
    for k in range(0, n_lookback * lookback_step + 1, lookback_step):
        Ok = O - k
        h_max_k = h_ui + k
        anchors, weights, regime_used = build_pool_pairlag(
            dratio, Ok, warmup, theiler_window, h_max_k, theta, regime_lab)
        if anchors is None:
            continue
        any_regime_used = any_regime_used or regime_used
        ratio_o_k = ratio_hist[Ok + 1]
        ratio_paths_k, w_k = method_whole_analog(dratio, ratio_o_k, anchors, weights, h_max_k)
        sub = ratio_paths_k[:, k:k + h_ui]
        cols_list.append(sub)
        w_list.append(w_k / w_k.sum())
    if not cols_list:
        return None, None, False
    return np.vstack(cols_list), np.concatenate(w_list), any_regime_used


# ═══════════════════════════════════════════════════════════════════════════
# Отображение — гистограмма/границы verbatim app31/app32/app33
# ═══════════════════════════════════════════════════════════════════════════

def corridor_density_heatmap(paths_price: np.ndarray, bin_height: float):
    if paths_price.size == 0:
        return None
    lo, hi = float(paths_price.min()), float(paths_price.max())
    if lo == hi:
        return None
    n_bins = max(1, int(np.ceil((hi - lo) / bin_height)))
    edges = lo + bin_height * np.arange(n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    h_max = paths_price.shape[1]
    Z = np.zeros((n_bins, h_max))
    for j in range(h_max):
        counts, _ = np.histogram(paths_price[:, j], bins=edges)
        csum = counts.sum()
        Z[:, j] = counts / csum if csum > 0 else 0.0
    return centers, Z


def corridor_bounds(paths_price: np.ndarray, coverage_pct: float):
    half = (100 - coverage_pct) / 200.0
    lo = np.quantile(paths_price, half, axis=0)
    hi = np.quantile(paths_price, 1 - half, axis=0)
    return lo, hi


# ═══════════════════════════════════════════════════════════════════════════
# Mixture-модель сценариев (эксп.49) — GaussianMixture(k=1/2/3, BIC),
# ресэмплинг по весам ансамбля (sklearn GMM.fit не принимает sample_weight).
# ═══════════════════════════════════════════════════════════════════════════

def fit_mixture(price_h: np.ndarray, weights: np.ndarray, n_resample: int, seed: int):
    w_norm = weights / weights.sum()
    rng = np.random.default_rng(seed)
    draw = rng.choice(len(price_h), size=n_resample, p=w_norm)
    sample = price_h[draw]
    mu, sd = sample.mean(), sample.std()
    if sd < 1e-12:
        return [(float(mu), float(sd), 1.0)]
    best_bic, best_gm = np.inf, None
    z = sample.reshape(-1, 1)
    for k in (1, 2, 3):
        gm = GaussianMixture(n_components=k, random_state=seed, n_init=3).fit(z)
        bic = gm.bic(z)
        if bic < best_bic:
            best_bic, best_gm = bic, gm
    means = best_gm.means_.ravel()
    stds = np.sqrt(best_gm.covariances_.ravel())
    weights_gm = best_gm.weights_
    order = np.argsort(-weights_gm)
    return [(float(means[i]), float(stds[i]), float(weights_gm[i])) for i in order]


def fit_mixture_per_h(paths_price: np.ndarray, weights: np.ndarray, n_resample: int, seed: int):
    """GaussianMixture независимо на каждом h. Возвращает
    list[list[(mean, std, weight)]], длина = paths_price.shape[1]."""
    h_max = paths_price.shape[1]
    return [fit_mixture(paths_price[:, j], weights, n_resample, seed) for j in range(h_max)]


def scenario_density_heatmap_from_components(components_by_h, lo: float, hi: float, bin_height: float):
    """Heatmap «сценарии»: каждая компонента вносит ненормированный вклад
    `w·exp(-0.5·((x-m)/σ)²)` — пик кривой РОВНО на высоте w (0..1). Всегда
    настоящая статистическая σ — при близких сценариях холмы честно
    перекрываются (реальная неопределённость, не баг)."""
    if hi <= lo:
        return None
    n_bins = max(1, int(np.ceil((hi - lo) / bin_height)))
    edges = lo + bin_height * np.arange(n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    h_max = len(components_by_h)
    Z = np.zeros((len(centers), h_max))
    for j, components in enumerate(components_by_h):
        for m, s, w_comp in components:
            Z[:, j] += w_comp * np.exp(-0.5 * ((centers - m) / max(s, 1e-6)) ** 2)
    return centers, Z


# ═══════════════════════════════════════════════════════════════════════════
# Снапшот на один origin + премотка (joblib/loky-фан-аут, см. run_multi_snapshot)
# ═══════════════════════════════════════════════════════════════════════════

def _build_one_snapshot(
    close: np.ndarray, origin_i: int, i: int, bars: int, h: int, theta: float,
    n_lookback: int, lookback_step: int, warmup: int, theiler_window: int,
    n_sim: int, seed: int, mix_n_resample: int, bin_height_pct: float, coverage_pct: float,
) -> dict:
    """Top-level (picklable) — полный цикл для ОДНОГО origin_i, безопасен
    для ProcessPoolExecutor (см. run_multi_snapshot). Возвращает
    {"i", "origin_i", "reason": ...} при недостатке истории/соседей, либо
    полный снапшот (все numpy-массивы уже .tolist()/float — готов к
    json.dumps без дополнительной обработки)."""
    if origin_i < 10:
        return {"i": i, "origin_i": origin_i, "reason": "origin_i < 10 (мало истории)"}

    _, a_arr, b_arr = _logtrend_causal(close[: origin_i + 1])
    a_o, b_o = float(a_arr[-1]), float(b_arr[-1])
    trend_full = np.exp(a_o + b_o * np.arange(len(close)))
    ratio_full = close / trend_full

    ratio_hist = ratio_full[: origin_i + 1]
    if bars > 0:
        ratio_hist = ratio_hist[max(0, len(ratio_hist) - (bars + 1)):]
    dratio_hist = np.diff(ratio_hist)
    O = len(dratio_hist) - 1

    vol = rolling_vol(dratio_hist)
    regime_lab, _regime_cutoffs = regime_labels(vol)
    cur_regime = int(regime_lab[O]) if 0 <= O < len(regime_lab) else -1

    ratio_paths, ens_weights, regime_used = build_lagged_ensemble_ratio_paths(
        dratio_hist, ratio_hist, O, h, theta, n_lookback, lookback_step, warmup,
        theiler_window, regime_lab)
    if ratio_paths is None:
        return {"i": i, "origin_i": origin_i,
                "reason": "Недостаточно причинно допустимых соседей на этом origin"}

    future_idx = np.arange(origin_i + 1, origin_i + 1 + h, dtype=np.float64)
    trend_fwd = np.exp(a_o + b_o * future_idx)
    full_price_paths = ratio_paths * trend_fwd[None, :]

    rng = np.random.default_rng(int(seed))
    p = ens_weights / ens_weights.sum()
    draw_idx = rng.choice(len(ens_weights), size=n_sim, p=p)
    close_path = full_price_paths[draw_idx]

    components_by_h = fit_mixture_per_h(full_price_paths, ens_weights, mix_n_resample, int(seed))

    bin_height = float(close[origin_i]) * bin_height_pct / 100.0
    raw_density = corridor_density_heatmap(close_path, bin_height=bin_height)
    raw_histogram = (
        {"bin_centers": raw_density[0].tolist(), "z": raw_density[1].tolist()}
        if raw_density is not None else None
    )

    lo, hi = corridor_bounds(close_path, coverage_pct)

    return {
        "i": i, "origin_i": origin_i,
        "origin_price": float(close[origin_i]),
        "cur_regime": cur_regime, "regime_used": bool(regime_used),
        "components_by_h": components_by_h,
        "raw_histogram": raw_histogram,
        "bounds": {"lo": lo.tolist(), "hi": hi.tolist()},
    }


def run_multi_snapshot(
    close: np.ndarray, origin: int, n_forecasts: int, rewind_step: int, bars: int, h: int,
    theta: float, n_lookback: int, lookback_step: int, warmup: int, theiler_window: int,
    n_sim: int, seed: int, mix_n_resample: int, bin_height_pct: float, coverage_pct: float,
    progress_cb=None, max_workers: int = 1,
) -> dict:
    """Прогоняет _build_one_snapshot для origin, origin−rewind_step, …, до
    n_forecasts штук (меньше, если не хватает истории). Каждый origin_i
    независим (свои ratio_i/O_i).

    ⚠️ ПОЧЕМУ joblib.Parallel(backend="loky"), НЕ голый ProcessPoolExecutor
    (как у simplex_ensemble.run_multi_origin): здесь, в отличие от simplex,
    воркер использует sklearn.GaussianMixture. Голый ProcessPoolExecutor с
    дефолтным fork() РЕАЛЬНО ПОДВИСАЕТ (не просто медленнее — бесконечный
    hang), если родитель уже успел инициализировать BLAS/joblib-тредпул до
    форка — унаследованное состояние потоков ломает joblib/BLAS в ребёнке
    (замерено эмпирически при разработке этого модуля). `loky` — тот же
    backend, которым sklearn сам параллелит свои внутренности — сам
    избегает этой ловушки (spawn-семантика) И, что важнее для прод-сервера
    с долгим временем жизни, ДЕРЖИТ переиспользуемый пул воркеров МЕЖДУ
    вызовами (не только между снапшотами одного прогноза) — холодный импорт
    numpy/scipy/sklearn в каждом воркере платится один раз за жизнь
    процесса API, а не на каждый POST /forecasts."""
    origins = [origin - i * rewind_step for i in range(n_forecasts)]
    origins = [(i, o) for i, o in enumerate(origins) if o >= 0]
    if not origins:
        return {"error": "Точка отсчёта слишком близко к началу истории для премотки."}

    results_by_i: dict[int, dict] = {}
    total = len(origins)
    if max_workers > 1 and total > 1:
        from joblib import Parallel, delayed
        jobs = (
            delayed(_build_one_snapshot)(
                close, origin_i, i, bars, h, theta,
                n_lookback, lookback_step, warmup, theiler_window,
                n_sim, seed, mix_n_resample, bin_height_pct, coverage_pct,
            )
            for i, origin_i in origins
        )
        done = 0
        for r in Parallel(n_jobs=max_workers, backend="loky", return_as="generator_unordered")(jobs):
            results_by_i[r["i"]] = r
            done += 1
            if progress_cb:
                progress_cb(done, total)
    else:
        for done, (i, origin_i) in enumerate(origins, start=1):
            results_by_i[i] = _build_one_snapshot(
                close, origin_i, i, bars, h, theta,
                n_lookback, lookback_step, warmup, theiler_window,
                n_sim, seed, mix_n_resample, bin_height_pct, coverage_pct,
            )
            if progress_cb:
                progress_cb(done, total)

    snapshots, skipped = [], []
    for i, _origin_i in origins:
        r = results_by_i[i]
        if "reason" in r:
            skipped.append({"origin_i": r["origin_i"], "reason": r["reason"]})
        else:
            snapshots.append(r)

    if not snapshots:
        return {"error": "Ни один origin премотки не дал валидного прогноза.", "skipped": skipped}

    return {"snapshots": snapshots, "skipped": skipped}


# ═══════════════════════════════════════════════════════════════════════════
# Точка входа для task_manager: close/times уже причинно обрезаны до origin
# (последний элемент = origin) вызывающим кодом.
# ═══════════════════════════════════════════════════════════════════════════

def forecast_regime_mixture_potential(
    times: np.ndarray, close: np.ndarray,
    horizon: int = DEFAULT_HORIZON, n_forecasts: int = DEFAULT_N_FORECASTS,
    rewind_step: int = DEFAULT_REWIND_STEP, theta: float = DEFAULT_THETA,
    warmup: int = DEFAULT_WARMUP, theiler_window: int = DEFAULT_THEILER_WINDOW,
    bars: int = DEFAULT_BARS, n_lookback: int = DEFAULT_N_LOOKBACK,
    lookback_step: int = DEFAULT_LOOKBACK_STEP, n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_SEED, mix_n_resample: int = DEFAULT_MIX_N_RESAMPLE,
    bin_height_pct: float = DEFAULT_BIN_HEIGHT_PCT, coverage_pct: float = DEFAULT_COVERAGE_PCT,
    progress_cb=None, max_workers: int = 1,
) -> dict:
    """Полный пайплайн от close/times до result_json-контракта (см.
    докстринг модуля sma/api/task_manager.py::_run_forecast_regime_mixture_
    potential). origin = len(close)-1.

    Возвращает {"error": "..."} при неустранимой ошибке — вызывающий код
    (task_manager) должен проверить это и поднять ValueError с этим текстом."""
    n = len(close)
    origin = n - 1

    mo = run_multi_snapshot(
        close, origin, max(1, n_forecasts), max(1, rewind_step), bars, horizon, theta,
        n_lookback, lookback_step, warmup, theiler_window,
        n_sim, seed, mix_n_resample, bin_height_pct, coverage_pct,
        progress_cb=progress_cb, max_workers=max_workers,
    )
    if "error" in mo:
        return mo

    snapshots_out = []
    for r in mo["snapshots"]:
        origin_i = r["origin_i"]
        snapshots_out.append({
            "origin_date": str(times[origin_i]),
            "origin_price": r["origin_price"],
            "cur_regime": r["cur_regime"], "regime_used": r["regime_used"],
            "components_by_h": r["components_by_h"],
            "raw_histogram": r["raw_histogram"],
            "bounds": r["bounds"],
        })

    main = mo["snapshots"][0]   # i=0 => origin_i=origin — самый свежий, гарантированно первый по построению run_multi_snapshot
    origin_date = str(times[main["origin_i"]])
    origin_price = main["origin_price"]
    top_component = max(main["components_by_h"][-1], key=lambda c: c[2])   # доминантный сценарий на h=H
    origin_direction = 1 if top_component[0] >= origin_price else -1

    return {
        "origin_date": origin_date,
        "origin_extreme_date": origin_date,   # нет пивота — origin сам себе "экстремум"
        "origin_price": origin_price,
        "origin_direction": origin_direction,
        "horizon": horizon, "n_forecasts": len(mo["snapshots"]),
        "rewind_step": rewind_step, "coverage_pct": coverage_pct, "bin_height_pct": bin_height_pct,
        "snapshots": snapshots_out,
        "skipped": mo["skipped"],
    }
