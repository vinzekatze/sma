"""
Визуализация 07: сравнение трёх подходов к интервалам на одном графике.

Для каждого из 3 прогнозов (хороший/средний/плохой) три слоя:
  1. Дисперсия кандидатов: min/max из 5 прогнозов (светлая заливка)
  2. Эмпирический интервал: base_band × (vm/0.0043)^0.58 (штриховая граница)
  3. LWC Conformal 80% CI (плотная заливка)
"""

from __future__ import annotations
import json
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

HOST    = "http://10.0.2.3:8000"
PROXIES = {"http": None, "https": None}

# ─── Калибровочные данные для LWC ─────────────────────────────────────────────

records = [json.loads(l) for l in open(RESULTS / "04_results.jsonl")]
cal     = [r for r in records if r["mape_f5"] <= 0.20]
MAX_K   = 10

cal_scores = np.full((len(cal), MAX_K), np.nan)
cal_vm     = np.array([r["val_mape"] for r in cal])
for i, r in enumerate(cal):
    for k in range(min(MAX_K, len(r["rel_errors"]))):
        cal_scores[i, k] = abs(r["rel_errors"][k])

def lwc_quantile(vm_new: float, alpha: float = 0.20, h: float = 1.0) -> np.ndarray:
    log_r = np.log(cal_vm / vm_new)
    w     = np.exp(-log_r**2 / (2 * h**2))
    qs = []
    for k in range(MAX_K):
        sk = cal_scores[:, k]; ok = ~np.isnan(sk)
        sk_k = sk[ok]; w_k = w[ok]
        w_aug = np.append(w_k, w_k.mean())
        s_aug = np.append(sk_k, np.inf)
        idx   = np.argsort(s_aug)
        vs    = s_aug[idx]; ws = w_aug[idx] / w_aug.sum()
        cumw  = np.cumsum(ws)
        j     = np.searchsorted(cumw, 1 - alpha)
        qs.append(float(vs[min(j, len(vs)-1)]))
    return np.array(qs)

# ─── Эмпирическая база (из 04_report.md, надёжные прогнозы) ──────────────────
# Асимметричные подписанные перцентили rel_error[k] для надёжных (vm < 0.0043)
# Строим прямо из данных

good_cal = [r for r in cal if r["val_mape"] < 0.0043]
base_p10 = np.zeros(MAX_K); base_p90 = np.zeros(MAX_K)
for k in range(MAX_K):
    errs = [r["rel_errors"][k] for r in good_cal if k < len(r["rel_errors"])]
    if errs:
        base_p10[k] = np.percentile(errs, 10)
        base_p90[k] = np.percentile(errs, 90)

REF_VM = 0.0043; BETA = 0.58

def empirical_band(vm_new: float, n_k: int):
    scale  = (vm_new / REF_VM) ** BETA
    lo = base_p10[:n_k] * scale
    hi = base_p90[:n_k] * scale
    return lo, hi

# ─── API ──────────────────────────────────────────────────────────────────────

candles_cache: dict[str, list] = {}

def get_candles(ticker):
    if ticker not in candles_cache:
        r = requests.get(f"{HOST}/candles",
                         params={"ticker": ticker, "data_source": "moex", "interval": "1d"},
                         proxies=PROXIES, timeout=30)
        candles_cache[ticker] = r.json()
    return candles_cache[ticker]

def fetch_forecast_full(fid: int):
    """Возвращает (mean_price, min_price, max_price) по барам."""
    r    = requests.get(f"{HOST}/forecasts/{fid}", proxies=PROXIES, timeout=30)
    raw  = r.json()
    res  = raw.get("result", {})
    if isinstance(res, str): res = json.loads(res)
    cands = res.get("candidates", [])
    if not cands: return None, None, None
    prices = np.array([c["forecast_price"] for c in cands], dtype=float)
    return prices.mean(axis=0), prices.min(axis=0), prices.max(axis=0)

def get_context(ticker, origin_date, n_before=20, n_after=MAX_K):
    candles = get_candles(ticker)
    idx     = next((i for i, c in enumerate(candles) if c["begin"][:10] == origin_date), None)
    if idx is None: return None, None
    before = [float(candles[j]["close"]) for j in range(max(0, idx - n_before), idx + 1)]
    after  = [float(candles[j]["close"]) for j in range(idx + 1, min(idx + 1 + n_after, len(candles)))]
    return before, after

# ─── Отбор прогнозов ──────────────────────────────────────────────────────────

import pandas as pd
df = pd.DataFrame(cal)

good = df[(df["val_mape"] < 0.002)  & (df["mape_f5"] < 0.012)].sort_values("mape_f5").iloc[0]
med  = df[(df["val_mape"].between(0.003, 0.005)) & (df["mape_f5"].between(0.018, 0.028))].sort_values("val_mape").iloc[0]
bad  = df[(df["val_mape"] > 0.012)  & (df["mape_f5"] > 0.06)].sort_values("mape_f5", ascending=False).iloc[0]

picks = [
    ("Хороший",  good, "#27ae60"),
    ("Средний",  med,  "#e67e22"),
    ("Плохой",   bad,  "#c0392b"),
]

# ─── Построение ───────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle("Интервалы неопределённости: три подхода", fontsize=13, fontweight="bold")

for ax, (label, row, color) in zip(axes, picks):
    fid    = int(row["forecast_id"])
    ticker = row["ticker"]
    origin = str(row["origin_ts"])[:10]
    vm     = float(row["val_mape"])

    print(f"Загружаем {label}: {ticker} {origin} fid={fid}")
    mean_fc, min_fc, max_fc = fetch_forecast_full(fid)
    prices_before, prices_after = get_context(ticker, origin)

    if mean_fc is None or prices_before is None:
        continue

    n_k          = min(MAX_K, len(mean_fc), len(prices_after))
    origin_close = prices_before[-1]
    x_hist       = list(range(-len(prices_before) + 1, 1))
    x_fc         = list(range(1, n_k + 1))
    fc           = mean_fc[:n_k]
    fc_min       = min_fc[:n_k]
    fc_max       = max_fc[:n_k]
    actual       = np.array(prices_after[:n_k])

    # --- Слой 1: дисперсия кандидатов (min/max) ---
    ax.fill_between(x_fc, fc_min, fc_max,
                    color=color, alpha=0.12, label="Разброс кандидатов (min–max)")

    # --- Слой 2: эмпирический интервал (80%) ---
    emp_lo, emp_hi = empirical_band(vm, n_k)
    emp_lower = fc + emp_lo * origin_close
    emp_upper = fc + emp_hi * origin_close
    ax.plot(x_fc, emp_lower, color=color, lw=1.2, ls="--", alpha=0.7)
    ax.plot(x_fc, emp_upper, color=color, lw=1.2, ls="--", alpha=0.7,
            label="Empirical 80% CI")

    # --- Слой 3: LWC Conformal 80% CI ---
    qs        = lwc_quantile(vm, alpha=0.20)[:n_k]
    lwc_lower = fc - qs * origin_close
    lwc_upper = fc + qs * origin_close
    ax.fill_between(x_fc, lwc_lower, lwc_upper,
                    color=color, alpha=0.25, label="LWC Conformal 80% CI")

    # --- История и origin ---
    ax.plot(x_hist, prices_before, color="black", lw=1.5)
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)

    # --- Прогноз ---
    ax.plot(x_fc, fc, color=color, lw=2.0, label="Прогноз (mean)", zorder=5)

    # --- Факт ---
    ax.plot(x_fc, actual, color="black", lw=1.5, ls="--",
            marker="o", ms=3.5, label="Фактические цены", zorder=6)

    # --- Аннотация ---
    gate = "✓ Gate pass" if vm < REF_VM else "✗ Gate fail"
    ax.set_title(
        f"{label}: {ticker}  {origin}\n"
        f"val_mape={vm*100:.2f}%   mape_f5={row['mape_f5']*100:.2f}%   {gate}",
        fontsize=9.5
    )
    ax.set_xlabel("Бар (0 = origin)", fontsize=9)
    ax.set_ylabel("Цена, руб.", fontsize=9)
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(True, alpha=0.25)

plt.tight_layout()
out = FIGURES / "07_compare_intervals.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nСохранено: {out}")
plt.show()
