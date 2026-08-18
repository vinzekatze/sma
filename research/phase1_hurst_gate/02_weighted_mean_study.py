"""
Исследование 02: взвешенное среднее по val_mape кандидатов vs простое среднее.

Гипотеза: если взвешивать forecast_price кандидатов обратно пропорционально их
val_mape (хуже валидация → меньший вес), итоговый прогноз будет точнее.

Схемы взвешивания (для сравнения):
  1. uniform     — простое среднее (текущая реализация)
  2. inv_mape    — weight = 1/mape_i, нормировано
  3. inv_mape2   — weight = 1/mape_i², более агрессивное
  4. softmax     — weight = exp(-mape_i / T), T = median(mape_i)
  5. best_only   — только candidates[0] (наилучший по val_mape)

Данные: 175 forecast_id из research/results/01_results.jsonl.
Кандидаты загружаются через GET /forecasts/{id}.
Реальные будущие цены берутся из GET /candles.

Вывод: research/results/02_results.jsonl, research/02_report.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sma.core.forecast.normalize import normalize

HOST     = "http://10.0.2.3:8000"
PROXIES  = {"http": None, "https": None}
RESULTS  = Path(__file__).parent / "results"
IN_FILE  = RESULTS / "01_results.jsonl"
OUT_FILE = RESULTS / "02_results.jsonl"


# ─── API ──────────────────────────────────────────────────────────────────────

def api_get(path: str, **params):
    r = requests.get(f"{HOST}{path}", params=params, proxies=PROXIES, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_candidates(forecast_id: int) -> list[dict]:
    """Возвращает список кандидатов [{mape, forecast_price, val_price, p, pca_k}]."""
    raw = api_get(f"/forecasts/{forecast_id}")
    result = raw.get("result", {})
    if isinstance(result, str):
        result = json.loads(result)
    return result.get("candidates", [])


def fetch_candles(ticker: str) -> list[dict]:
    return api_get("/candles", ticker=ticker, data_source="moex", interval="1d")


# ─── схемы взвешивания ────────────────────────────────────────────────────────

def weighted_mean(candidates: list[dict], scheme: str) -> np.ndarray:
    """Возвращает взвешенное среднее forecast_price для заданной схемы."""
    prices  = np.array([c["forecast_price"] for c in candidates], dtype=float)  # (N, horizon)
    mapes   = np.array([c["mape"]           for c in candidates], dtype=float)  # (N,)

    if scheme == "uniform":
        return prices.mean(axis=0)

    if scheme == "best_only":
        return prices[0]  # кандидаты уже отсортированы по mape asc

    # защита от нулевых mape
    mapes_safe = np.where(mapes < 1e-9, 1e-9, mapes)

    if scheme == "inv_mape":
        w = 1.0 / mapes_safe

    elif scheme == "inv_mape2":
        w = 1.0 / (mapes_safe ** 2)

    elif scheme == "softmax":
        T = float(np.median(mapes_safe))
        if T < 1e-9:
            T = 1e-9
        w = np.exp(-mapes_safe / T)

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    w = w / w.sum()
    return (prices * w[:, None]).sum(axis=0)


# ─── метрики ─────────────────────────────────────────────────────────────────

def mape_k(predicted: np.ndarray, actual: np.ndarray, k: int) -> float | None:
    n = min(k, len(predicted), len(actual))
    if n == 0:
        return None
    p, a = predicted[:n], actual[:n]
    return float(np.mean(np.abs((p - a) / (np.abs(a) + 1e-12))))


def dir_acc(predicted: np.ndarray, actual: np.ndarray, close_at_origin: float, k: int = 5) -> float | None:
    n = min(k, len(predicted), len(actual))
    if n == 0:
        return None
    return float(np.mean(np.sign(actual[:n] - close_at_origin) == np.sign(predicted[:n] - close_at_origin)))


# ─── главный цикл ─────────────────────────────────────────────────────────────

SCHEMES = ["uniform", "inv_mape", "inv_mape2", "softmax", "best_only"]


def run() -> None:
    RESULTS.mkdir(exist_ok=True)

    # Загружаем результаты исследования 01
    records_01 = []
    with open(IN_FILE) as f:
        for line in f:
            records_01.append(json.loads(line))
    print(f"Загружено {len(records_01)} записей из 01_results.jsonl")

    # Кэшируем свечи по тикеру
    candles_cache: dict[str, list] = {}
    valid_cache:   dict[str, object] = {}

    def get_closes_by_ts(ticker: str, origin_ts: str, n_future: int = 16):
        """Возвращает (close_at_origin, actual_future) по timestamp, не по числовому индексу."""
        if ticker not in candles_cache:
            print(f"  Загружаем свечи {ticker}...")
            candles_cache[ticker] = fetch_candles(ticker)
        candles = candles_cache[ticker]
        origin_date = origin_ts[:10]
        # Ищем свечу по дате
        idx = next((i for i, c in enumerate(candles) if c["begin"][:10] == origin_date), None)
        if idx is None:
            return None, None
        close_at_origin = float(candles[idx]["close"])
        actual_future   = [float(candles[j]["close"]) for j in range(idx + 1, min(idx + 1 + n_future, len(candles)))]
        return close_at_origin, actual_future

    # Проверяем уже готовые записи
    done_ids: set[int] = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["forecast_id"])
                except Exception:
                    pass
    print(f"Уже готово: {len(done_ids)} записей (resume)")

    errors = 0
    for i, rec in enumerate(records_01):
        fid      = int(rec["forecast_id"])
        ticker   = rec["ticker"]
        origin_k = int(rec["origin_k"])
        ma_window= int(rec["ma_window"])

        if fid in done_ids:
            continue

        print(f"  [{i+1}/{len(records_01)}] {ticker} {rec['origin_ts'][:10]} forecast_id={fid}", end="", flush=True)

        try:
            candidates = fetch_candidates(fid)
            if not candidates:
                print(" — нет кандидатов, пропуск")
                errors += 1
                continue

            close_at_origin, actual_future = get_closes_by_ts(ticker, rec["origin_ts"])
            if close_at_origin is None or not actual_future:
                print(" — origin-дата не найдена или нет будущих баров, пропуск")
                errors += 1
                continue
            actual_future = np.array(actual_future)

            result = {
                "forecast_id": fid,
                "ticker":      ticker,
                "origin_ts":   rec["origin_ts"],
                "zone":        rec["zone"],
                "h_last":      rec["h_last"],
                "h_drop":      rec["h_drop"],
                "val_mape":    rec["val_mape"],
                "n_candidates": len(candidates),
                "candidate_mapes": [c["mape"] for c in candidates],
            }

            for scheme in SCHEMES:
                try:
                    pred = weighted_mean(candidates, scheme)
                    result[f"mape_f5_{scheme}"]  = mape_k(pred, actual_future, 5)
                    result[f"mape_f15_{scheme}"] = mape_k(pred, actual_future, 15)
                    result[f"dir5_{scheme}"]     = dir_acc(pred, actual_future, close_at_origin, 5)
                except Exception as e:
                    result[f"mape_f5_{scheme}"]  = None
                    result[f"mape_f15_{scheme}"] = None
                    result[f"dir5_{scheme}"]     = None

            with open(OUT_FILE, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

            # краткий вывод: uniform vs best схема
            u  = result.get("mape_f5_uniform")
            bw = result.get("mape_f5_inv_mape")
            print(f"  uniform={u:.4f}  inv_mape={bw:.4f}" if u and bw else " OK")

        except Exception as e:
            print(f"  ОШИБКА: {e}")
            errors += 1
            time.sleep(2)

    print(f"\nГотово. Ошибок: {errors}")
    analyze()


# ─── анализ ───────────────────────────────────────────────────────────────────

def analyze() -> None:
    records = []
    with open(OUT_FILE) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    if not records:
        print("Нет данных для анализа")
        return

    import pandas as pd
    from scipy.stats import wilcoxon, ttest_rel

    df = pd.DataFrame(records)
    print(f"\n{'='*65}")
    print(f"АНАЛИЗ ВЗВЕШИВАНИЯ: {len(df)} прогнозов")
    print(f"{'='*65}")

    # Фильтр выбросов
    df_clean = df[df["mape_f5_uniform"] <= 0.20].copy()
    print(f"Без выбросов (mape_f5_uniform <= 0.20): n={len(df_clean)}")

    def wilcoxon_vs_uniform(df_c, col_test, col_ref):
        """Парный Wilcoxon col_test vs col_ref, возвращает (p, stars, pct_better)."""
        if col_test == col_ref:
            return 1.0, "", 50.0
        tmp = df_c[[col_test, col_ref]].rename(columns={col_test: "test", col_ref: "ref"}).dropna()
        if len(tmp) < 6 or tmp["test"].std() == 0:
            return 1.0, "", float((tmp["test"] < tmp["ref"]).mean() * 100)
        stat, p = wilcoxon(tmp["test"].values, tmp["ref"].values)
        stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        better = float((tmp["test"] < tmp["ref"]).mean() * 100)
        return float(p), stars, better

    print("\n── Медианный MAPE_F5 по схемам ──")
    for sc in SCHEMES:
        col = f"mape_f5_{sc}"
        if col not in df_clean.columns:
            continue
        d = df_clean[col].dropna()
        p, stars, better = wilcoxon_vs_uniform(df_clean, col, "mape_f5_uniform")
        prefix = "  [baseline]" if sc == "uniform" else ""
        print(f"  {sc:<12}: median={d.median():.4f}  mean={d.mean():.4f}  "
              f"лучше uniform={better:.1f}%  Wilcoxon p={p:.3f}{stars}{prefix}")

    print("\n── Медианный MAPE_F15 по схемам ──")
    for sc in SCHEMES:
        col = f"mape_f15_{sc}"
        if col not in df_clean.columns:
            continue
        d = df_clean[col].dropna()
        p, stars, better = wilcoxon_vs_uniform(df_clean, col, "mape_f15_uniform")
        print(f"  {sc:<12}: median={d.median():.4f}  mean={d.mean():.4f}  "
              f"лучше uniform={better:.1f}%  Wilcoxon p={p:.3f}{stars}")

    print("\n── Точность направления DIR_ACC5 ──")
    for sc in SCHEMES:
        col = f"dir5_{sc}"
        if col not in df_clean.columns:
            continue
        d = df_clean[col].dropna()
        print(f"  {sc:<12}: mean={d.mean():.3f}")

    # Зависимость выигрыша от разброса mape среди кандидатов
    print("\n── Когда inv_mape выигрывает у uniform? ──")
    both = df_clean[["mape_f5_uniform", "mape_f5_inv_mape", "candidate_mapes"]].dropna()
    both = both.copy()
    both["mape_spread"] = both["candidate_mapes"].apply(
        lambda x: float(np.std(x)) if isinstance(x, list) else np.nan
    )
    both["inv_better"] = both["mape_f5_inv_mape"] < both["mape_f5_uniform"]
    q33, q66 = both["mape_spread"].quantile([0.33, 0.66])
    for label, mask in [
        (f"spread < {q33:.4f} (кандидаты похожи)",   both["mape_spread"] < q33),
        (f"spread {q33:.4f}–{q66:.4f} (средний)",    (both["mape_spread"] >= q33) & (both["mape_spread"] < q66)),
        (f"spread > {q66:.4f} (кандидаты разные)", both["mape_spread"] >= q66),
    ]:
        grp = both[mask]
        if len(grp) == 0:
            continue
        pct = grp["inv_better"].mean() * 100
        delta_med = (grp["mape_f5_uniform"] - grp["mape_f5_inv_mape"]).median()
        print(f"  {label}: n={len(grp):3d}  inv_mape лучше в {pct:.1f}% случаев  "
              f"delta_median={delta_med:+.4f}")

    print("\n── По зонам H ──")
    for zone in ("green", "border", "red"):
        grp = df_clean[df_clean["zone"] == zone]
        if len(grp) == 0:
            continue
        u  = grp["mape_f5_uniform"].median()
        iw = grp["mape_f5_inv_mape"].median()
        bo = grp["mape_f5_best_only"].median()
        print(f"  {zone:<8}: uniform={u:.4f}  inv_mape={iw:.4f}  best_only={bo:.4f}")

    # Запись итогового отчёта
    report_lines = [
        "# Исследование 02: взвешенное среднее по val_mape\n",
        f"**Дата:** 2026-06-07  n={len(df_clean)} (без выбросов)\n\n",
        "## Медианный MAPE_F5\n",
        "| Схема | median | mean | % лучше uniform | Wilcoxon p |\n",
        "|---|---|---|---|---|\n",
    ]
    for sc in SCHEMES:
        col = f"mape_f5_{sc}"
        d = df_clean[col].dropna()
        p, stars, better = wilcoxon_vs_uniform(df_clean, col, "mape_f5_uniform")
        report_lines.append(
            f"| {sc} | {d.median():.4f} | {d.mean():.4f} | {better:.1f}% | {p:.3f}{stars} |\n"
        )
    report_path = Path(__file__).parent / "02_report.md"
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"\nОтчёт: {report_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if args.analyze_only:
        analyze()
    else:
        run()
