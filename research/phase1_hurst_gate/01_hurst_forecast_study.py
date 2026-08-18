"""
Исследование гипотез: h_last / h_drop vs MAPE прогноза.

Гипотезы (из INRESERCH.md):
  1. Прогноз в точках с резким спадом показателя Хёрста точнее.
  2. Низкое значение Хёрста в точке прогнозирования → лучше прогноз.
  3. Существуют пороговые значения H, где прогноз достовернее.

Для каждого тикера:
  - загружаем все свечи через GET /candles
  - нормализуем, вычисляем dratio
  - по всей истории вычисляем Хёрст в rolling-окне (step=1)
  - стратифицируем origin-точки: green (H<0.45), border (0.45-0.55), red (H>0.55)
  - дополнительно включаем точки с разным знаком h_drop
  - для каждой origin-точки: POST /forecasts → ждём результата → считаем метрики
  - сравниваем mean_price (15 баров будущего) с реальными close

Вывод: results/01_results.jsonl (дополняется по мере работы)
       results/01_progress.txt   (текущее состояние)

Запуск:
    python research/01_hurst_forecast_study.py [--host http://10.0.2.3:8000]
                                               [--per-bucket 15]
                                               [--dry-run]
                                               [--tickers SBER GAZP LKOH GMKN]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from scipy.stats import linregress, pearsonr, spearmanr

# Чтобы работало из любой рабочей директории
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sma.core.forecast.normalize import normalize
from sma.core.forecast.hurst import hurst_rs

# ─── константы ───────────────────────────────────────────────────────────────

INTERVAL = "1d"

# Параметры вычисления Хёрста для исследования (фиксированы)
HURST_WINDOW = 100      # bars per H estimate
FINE_N       = 10       # consecutive H values in fine window

# Пороги зон
H_GREEN  = 0.45
H_RED    = 0.55

# Параметры прогноза
FORECAST_PARAMS = {
    "p_max":       70,
    "val_horizon": 5,
    "horizon":     15,
    "auto_ma":     True,
    "use_lwr":     True,
    "top_n":       5,
}

RESULTS_DIR = Path(__file__).parent / "results"

# ─── вспомогательные функции ─────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    progress_file = RESULTS_DIR / "01_progress.txt"
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


_NO_PROXY = {"http": None, "https": None}   # обходим системный прокси для локального хоста


def api_get(host: str, path: str, **params) -> dict | list:
    r = requests.get(f"{host}{path}", params=params, timeout=30, proxies=_NO_PROXY)
    r.raise_for_status()
    return r.json()


def api_post(host: str, path: str, body: dict) -> dict:
    r = requests.post(f"{host}{path}", json=body, timeout=30, proxies=_NO_PROXY)
    r.raise_for_status()
    return r.json()


def fetch_candles(host: str, ticker: str) -> list[dict]:
    return api_get(host, "/candles", ticker=ticker, data_source="moex", interval=INTERVAL)


def get_series_settings(host: str, ticker: str) -> dict:
    return api_get(host, "/series/settings", ticker=ticker, data_source="moex", interval=INTERVAL)


def submit_forecast(host: str, ticker: str, origin_ts: str) -> int:
    resp = api_post(host, "/forecasts", {
        "ticker":       ticker,
        "data_source":  "moex",
        "interval":     INTERVAL,
        "origin_ts":    origin_ts,
        "params":       FORECAST_PARAMS,
    })
    return int(resp["task_id"])


def poll_task(host: str, task_id: int, timeout: int = 600) -> int:
    """Ждём завершения задачи; возвращаем forecast_id."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = api_get(host, f"/tasks/{task_id}")
        status = task["status"]
        if status == "done":
            return int(task["forecast_id"])
        if status == "error":
            raise RuntimeError(f"task {task_id} error: {task.get('error_msg', '')}")
        time.sleep(3)
    raise TimeoutError(f"task {task_id} not done after {timeout}s")


def get_forecast_result(host: str, forecast_id: int) -> dict:
    raw = api_get(host, f"/forecasts/{forecast_id}")
    # result_json хранится как вложенный dict в ключе "result"
    result = raw.get("result", {})
    if isinstance(result, str):
        return json.loads(result)
    return result


# ─── вычисление признаков Хёрста ─────────────────────────────────────────────

def compute_h_features(dratio: np.ndarray, origin_k: int) -> dict:
    """
    Вычисляет h_last, h_drop, h_slope для origin_k.

    fine window: H значения при i in [origin_k - FINE_N + 1 .. origin_k]
      каждое H[i] = hurst_rs(dratio[i - HURST_WINDOW : i])
    h_drop  = H[first valid] - H[last]  (положительное = H падал)
    h_slope = наклон линейной регрессии H по позиции (отрицательный = тренд вниз)
    """
    h_vals = []
    for i in range(origin_k - FINE_N + 1, origin_k + 1):
        if i < HURST_WINDOW:
            h_vals.append(np.nan)
        else:
            h_vals.append(hurst_rs(dratio[i - HURST_WINDOW: i]))

    h_arr = np.array(h_vals, dtype=float)
    valid = ~np.isnan(h_arr)

    if valid.sum() < 2:
        return dict(h_last=np.nan, h_drop=np.nan, h_slope=np.nan,
                    h_mean_fine=np.nan, h_min_fine=np.nan, h_max_fine=np.nan)

    h_last      = float(h_arr[-1]) if not np.isnan(h_arr[-1]) else float(h_arr[valid][-1])
    h_drop      = float(h_arr[valid][0] - h_arr[valid][-1])   # >0 если H упал
    positions   = np.where(valid)[0].astype(float)
    slope, *_   = linregress(positions, h_arr[valid])

    return dict(
        h_last     = h_last,
        h_drop     = h_drop,
        h_slope    = float(slope),
        h_mean_fine= float(h_arr[valid].mean()),
        h_min_fine = float(h_arr[valid].min()),
        h_max_fine = float(h_arr[valid].max()),
    )


# ─── выбор origin-точек ──────────────────────────────────────────────────────

def build_hurst_profile(dratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling H с шагом 1 для всего dratio.
    Возвращает (indices, values) — H для каждого бара начиная с HURST_WINDOW.
    """
    n = len(dratio)
    idxs, vals = [], []
    for i in range(HURST_WINDOW, n + 1):
        idxs.append(i)
        vals.append(hurst_rs(dratio[i - HURST_WINDOW: i]))
    return np.array(idxs), np.array(vals)


def select_origins(
    valid_df,
    dratio: np.ndarray,
    h_indices: np.ndarray,
    h_values: np.ndarray,
    per_bucket: int,
    min_future: int = 15,
    min_history: int = 100,   # >= HURST_WINDOW + FINE_N
) -> list[dict]:
    """
    Стратифицированная выборка origin-точек.

    Для каждой зоны (green/border/red) отбираем per_bucket точек,
    причём ~половину с h_drop > 0, половину с h_drop ≤ 0.
    Точки равномерно распределены по временной оси.
    """
    n = len(valid_df)
    h_map = {int(idx): float(val) for idx, val in zip(h_indices, h_values)
             if not np.isnan(val)}

    # Признаки для всех потенциальных origin
    candidates = []
    for k in range(min_history + FINE_N, n - min_future):
        if k not in h_map:
            continue
        h_last = h_map[k]
        # h_drop требует fine_window: последние FINE_N H-значений
        # h_map[k - FINE_N + 1] .. h_map[k]
        fine = [h_map.get(i) for i in range(k - FINE_N + 1, k + 1)]
        if any(v is None for v in fine):
            continue
        fine_arr = np.array(fine, dtype=float)
        h_drop = float(fine_arr[0] - fine_arr[-1])

        if h_last < H_GREEN:
            zone = "green"
        elif h_last <= H_RED:
            zone = "border"
        else:
            zone = "red"

        candidates.append(dict(k=k, zone=zone, h_last=h_last, h_drop=h_drop))

    # Стратифицируем
    selected = []
    for zone in ("green", "border", "red"):
        pool = [c for c in candidates if c["zone"] == zone]
        if not pool:
            log(f"  WARN: zone '{zone}' — нет кандидатов")
            continue

        # Разбиваем на h_drop>0 и h_drop<=0
        drop_pos = [c for c in pool if c["h_drop"] >  0]
        drop_neg = [c for c in pool if c["h_drop"] <= 0]

        half = per_bucket // 2
        def even_sample(lst, n):
            if not lst:
                return []
            lst_sorted = sorted(lst, key=lambda x: x["k"])
            if len(lst_sorted) <= n:
                return lst_sorted
            step = len(lst_sorted) / n
            return [lst_sorted[int(i * step)] for i in range(n)]

        chosen  = even_sample(drop_pos, half) + even_sample(drop_neg, per_bucket - half)
        # убираем дубликаты по k
        seen = set()
        for c in chosen:
            if c["k"] not in seen:
                seen.add(c["k"])
                selected.append(c)

    return selected


# ─── вычисление метрик на будущем ────────────────────────────────────────────

def compute_forecast_metrics(
    mean_price: list[float],
    actual_closes: list[float],
    close_at_origin: float,
) -> dict:
    """
    mape_f5, mape_f15: MAPE mean_price vs actual будущих цен.
    dir_acc5: точность направления на 5 баров.
    """
    pred = np.array(mean_price[:15], dtype=float)
    act  = np.array(actual_closes[:15], dtype=float)
    n    = min(len(pred), len(act))

    if n == 0:
        return {}

    pred, act = pred[:n], act[:n]
    mape = np.abs((pred - act) / (np.abs(act) + 1e-12))

    out = {"n_future_bars": n}
    if n >= 5:
        out["mape_f5"] = float(mape[:5].mean())
    if n >= 15:
        out["mape_f15"] = float(mape.mean())

    # Направление: sign(bar - close_at_origin)
    k5 = min(n, 5)
    if k5 > 0:
        act_dir  = np.sign(act[:k5]  - close_at_origin)
        pred_dir = np.sign(pred[:k5] - close_at_origin)
        out["dir_acc5"] = float((act_dir == pred_dir).mean())

    return out


# ─── главный цикл ────────────────────────────────────────────────────────────

def already_done(results_file: Path, ticker: str, origin_ts: str) -> bool:
    """Проверяет, был ли этот origin уже обработан (для resume)."""
    if not results_file.exists():
        return False
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("ticker") == ticker and rec.get("origin_ts")[:10] == origin_ts[:10]:
                    return True
            except Exception:
                pass
    return False


def run(
    host: str,
    tickers: list[str],
    per_bucket: int,
    dry_run: bool,
    results_file: Path,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    total_done = 0
    total_err  = 0

    for ticker in tickers:
        log(f"=== {ticker} ===")

        # 1. Загружаем свечи
        log(f"  Загружаем свечи {ticker}...")
        try:
            candles = fetch_candles(host, ticker)
        except Exception as e:
            log(f"  ОШИБКА загрузки свечей: {e}")
            continue
        log(f"  Получено {len(candles)} свечей")

        # 2. Получаем оптимальную MA
        try:
            settings = get_series_settings(host, ticker)
            ma_window = settings.get("optimal_ma") or 200
        except Exception:
            ma_window = 200
        log(f"  Оптимальная MA = {ma_window}")

        # 3. Нормализуем
        norm  = normalize(candles, window=ma_window)
        valid = norm.dropna(subset=["ma"]).reset_index(drop=True)
        dratio = np.diff(valid["ratio"].values)
        closes  = valid["close"].values
        begins  = valid["begin"].values
        log(f"  Валидных баров: {len(valid)}, dratio: {len(dratio)}")

        # 4. Строим профиль Хёрста (шаг=1)
        log(f"  Вычисляем профиль Хёрста (window={HURST_WINDOW}, step=1)...")
        h_indices, h_values = build_hurst_profile(dratio)
        log(f"  Готово. H-значений: {(~np.isnan(h_values)).sum()}")

        # 5. Выбираем origin-точки
        origins = select_origins(valid, dratio, h_indices, h_values, per_bucket)
        log(f"  Выбрано {len(origins)} origin-точек")

        by_zone = {}
        for o in origins:
            by_zone.setdefault(o["zone"], 0)
            by_zone[o["zone"]] += 1
        for z, cnt in sorted(by_zone.items()):
            log(f"    {z}: {cnt}")

        if dry_run:
            log(f"  [dry-run] пропускаем прогнозы для {ticker}")
            continue

        # 6. Прогнозируем
        for i, origin_info in enumerate(origins):
            k        = origin_info["k"]
            zone     = origin_info["zone"]
            origin_ts = str(begins[k])[:19]

            # resume: пропускаем уже сделанные
            if already_done(results_file, ticker, origin_ts):
                log(f"  [{i+1}/{len(origins)}] {ticker} {origin_ts} — уже готово, пропускаем")
                total_done += 1
                continue

            log(f"  [{i+1}/{len(origins)}] {ticker} {origin_ts} (zone={zone}, "
                f"h_last={origin_info['h_last']:.3f}, h_drop={origin_info['h_drop']:+.3f})")

            # Вычисляем полный набор признаков Хёрста для этой точки
            h_feat = compute_h_features(dratio, k)

            try:
                # Запускаем прогноз
                task_id = submit_forecast(host, ticker, origin_ts)
                log(f"    task_id={task_id} — ожидаем...")
                forecast_id = poll_task(host, task_id, timeout=600)
                log(f"    forecast_id={forecast_id}")

                # Получаем результат
                fc = get_forecast_result(host, forecast_id)

                # Фактические будущие цены
                actual_future = closes[k + 1: k + 16].tolist()

                mean_price      = fc.get("mean_price", [])
                close_at_origin = float(fc.get("close_at_origin", closes[k]))
                hurst_server    = float(fc.get("hurst_at_origin", np.nan))
                ma_used         = int(fc.get("ma_window", ma_window))

                # val_mape: MAPE лучшего кандидата на валидационном окне
                candidates  = fc.get("candidates", [])
                val_mape    = float(candidates[0]["mape"]) if candidates else np.nan

                metrics = compute_forecast_metrics(mean_price, actual_future, close_at_origin)

                record = dict(
                    ticker          = ticker,
                    interval        = INTERVAL,
                    origin_ts       = origin_ts,
                    origin_k        = k,
                    zone            = zone,
                    ma_window       = ma_used,
                    forecast_id     = forecast_id,
                    # Хёрст-признаки (наш расчёт)
                    **h_feat,
                    # Хёрст от сервера (для кросс-проверки)
                    hurst_server    = hurst_server,
                    # Метрика сервера (val = прогноз на известных барах)
                    val_mape        = val_mape,
                    # Метрики на будущем (наш расчёт)
                    **metrics,
                )

                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

                mape_str = f"mape_f5={metrics.get('mape_f5', float('nan')):.4f}" if 'mape_f5' in metrics else ""
                log(f"    OK  h_last={h_feat['h_last']:.3f}  h_drop={h_feat['h_drop']:+.3f}  "
                    f"val_mape={val_mape:.4f}  {mape_str}")

                total_done += 1

            except Exception as e:
                log(f"    ОШИБКА: {e}")
                total_err += 1
                # Не прерываем цикл, продолжаем
                time.sleep(5)

    log(f"\n=== Итого: выполнено={total_done}, ошибок={total_err} ===")
    log(f"Результаты: {results_file}")


# ─── анализ результатов ───────────────────────────────────────────────────────

def analyze(results_file: Path) -> None:
    if not results_file.exists():
        print("Нет данных для анализа")
        return

    records = []
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    if not records:
        print("Файл пуст")
        return

    print(f"\n{'='*60}")
    print(f"АНАЛИЗ: {len(records)} прогнозов")
    print(f"{'='*60}")

    import pandas as pd
    df = pd.DataFrame(records)

    # Фильтруем строки с полными данными
    cols_needed = ["h_last", "h_drop", "mape_f5", "mape_f15", "dir_acc5", "val_mape"]
    df_clean = df.dropna(subset=[c for c in cols_needed if c in df.columns])
    print(f"Строк с полными данными: {len(df_clean)}")

    # ── 1. Корреляции ──────────────────────────────────────────────────────
    print("\n── Корреляции Пирсона (h_* ↔ метрики) ──")
    features  = ["h_last", "h_drop", "h_slope", "h_mean_fine", "val_mape"]
    targets   = ["mape_f5", "mape_f15", "dir_acc5"]

    for feat in features:
        if feat not in df_clean.columns:
            continue
        row_parts = [f"  {feat:<14}"]
        for tgt in targets:
            if tgt not in df_clean.columns:
                continue
            x = df_clean[feat].values
            y = df_clean[tgt].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                row_parts.append(f"  {tgt}: —")
                continue
            r, p = pearsonr(x[mask], y[mask])
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row_parts.append(f"  {tgt}: r={r:+.3f}{stars:<3}(p={p:.3f})")
        print("".join(row_parts))

    # ── 2. Пороговые группы ────────────────────────────────────────────────
    print("\n── Медианный MAPE_F5 по зонам ──")
    if "zone" in df_clean.columns and "mape_f5" in df_clean.columns:
        for zone in ("green", "border", "red"):
            grp = df_clean[df_clean["zone"] == zone]["mape_f5"]
            if len(grp):
                print(f"  {zone:<8}: n={len(grp):3d}  median={grp.median():.4f}  "
                      f"mean={grp.mean():.4f}  std={grp.std():.4f}")

    # ── 3. Гипотеза h_drop > 0.07 ─────────────────────────────────────────
    if "h_drop" in df_clean.columns and "mape_f5" in df_clean.columns:
        print("\n── Порог h_drop > 0.07 ──")
        above = df_clean[df_clean["h_drop"] >  0.07]["mape_f5"]
        below = df_clean[df_clean["h_drop"] <= 0.07]["mape_f5"]
        if len(above) and len(below):
            print(f"  h_drop>0.07  : n={len(above):3d}  median={above.median():.4f}")
            print(f"  h_drop<=0.07 : n={len(below):3d}  median={below.median():.4f}")

    # ── 4. Комбо h_last < 0.5 AND h_drop > 0.05 ──────────────────────────
    if "h_last" in df_clean.columns and "h_drop" in df_clean.columns and "mape_f5" in df_clean.columns:
        print("\n── Комбо: h_last < 0.5 AND h_drop > 0.05 ──")
        combo = df_clean[(df_clean["h_last"] < 0.5) & (df_clean["h_drop"] > 0.05)]["mape_f5"]
        rest  = df_clean[~((df_clean["h_last"] < 0.5) & (df_clean["h_drop"] > 0.05))]["mape_f5"]
        if len(combo):
            print(f"  условие выполнено : n={len(combo):3d}  median={combo.median():.4f}")
        if len(rest):
            print(f"  иначе             : n={len(rest):3d}  median={rest.median():.4f}")

    # ── 5. По тикерам ─────────────────────────────────────────────────────
    if "ticker" in df_clean.columns and "mape_f5" in df_clean.columns:
        print("\n── По тикерам ──")
        for t, grp in df_clean.groupby("ticker"):
            m5  = grp["mape_f5"].median()
            m15 = grp["mape_f15"].median() if "mape_f15" in grp.columns else float("nan")
            dr  = grp["dir_acc5"].mean()  if "dir_acc5" in grp.columns else float("nan")
            print(f"  {t:<6}: n={len(grp):3d}  mape_f5={m5:.4f}  "
                  f"mape_f15={m15:.4f}  dir_acc5={dr:.3f}")

    print(f"\nРезультаты сохранены в: {results_file}")


# ─── точка входа ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",       default="http://10.0.2.3:8000")
    parser.add_argument("--tickers",    nargs="+", default=["SBER", "GAZP", "LKOH", "GMKN"])
    parser.add_argument("--per-bucket", type=int, default=15,
                        help="origin-точек из каждой зоны (green/border/red)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="только выводит выбранные origin-точки, не запускает прогнозы")
    parser.add_argument("--analyze-only", action="store_true",
                        help="только анализирует уже собранные результаты")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = RESULTS_DIR / "01_results.jsonl"

    if args.analyze_only:
        analyze(results_file)
        return

    run(
        host        = args.host,
        tickers     = args.tickers,
        per_bucket  = args.per_bucket,
        dry_run     = args.dry_run,
        results_file= results_file,
    )

    analyze(results_file)


if __name__ == "__main__":
    main()
