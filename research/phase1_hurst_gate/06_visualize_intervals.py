"""
Визуализация conformal prediction intervals для 3 категорий прогнозов:
  - хороший  (низкий val_mape, низкий mape_f5)
  - средний  (средний val_mape)
  - плохой   (высокий val_mape или высокий mape_f5)

Для каждого прогноза: линия прогноза + LWC-полоса + фактические цены.
"""

from __future__ import annotations
import json
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT    = Path(__file__).parent.parent
RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

HOST    = "http://10.0.2.3:8000"
PROXIES = {"http": None, "https": None}

# ─── Загрузка калибровочных данных ────────────────────────────────────────────

records = [json.loads(l) for l in open(RESULTS / "04_results.jsonl")]
cal = [r for r in records if r["mape_f5"] <= 0.20]
print(f"Калибровочная выборка: n={len(cal)}")

MAX_K = 10
cal_scores = np.full((len(cal), MAX_K), np.nan)
cal_vm     = np.array([r["val_mape"] for r in cal])

for i, r in enumerate(cal):
    for k in range(min(MAX_K, len(r["rel_errors"]))):
        cal_scores[i, k] = abs(r["rel_errors"][k])

# ─── LWC: взвешенный квантиль ─────────────────────────────────────────────────

def lwc_quantile(vm_new: float, alpha: float = 0.20, h: float = 1.0) -> np.ndarray:
    """Возвращает LWC-квантиль уровня 1-alpha для каждого бара k."""
    log_r = np.log(cal_vm / vm_new)
    w     = np.exp(-log_r**2 / (2 * h**2))
    qs = []
    for k in range(MAX_K):
        sk = cal_scores[:, k]
        ok = ~np.isnan(sk)
        sk_k = sk[ok]; w_k = w[ok]
        # conformal augmentation
        w_aug = np.append(w_k, w_k.mean())
        s_aug = np.append(sk_k, np.inf)
        # взвешенный квантиль
        idx   = np.argsort(s_aug)
        vs    = s_aug[idx]; ws = (w_aug[idx] / w_aug.sum())
        cumw  = np.cumsum(ws)
        j     = np.searchsorted(cumw, 1 - alpha)
        qs.append(float(vs[min(j, len(vs)-1)]))
    return np.array(qs)

# ─── API ──────────────────────────────────────────────────────────────────────

candles_cache: dict[str, list] = {}

def get_candles(ticker: str) -> list:
    if ticker not in candles_cache:
        r = requests.get(f"{HOST}/candles",
                         params={"ticker": ticker, "data_source": "moex", "interval": "1d"},
                         proxies=PROXIES, timeout=30)
        candles_cache[ticker] = r.json()
    return candles_cache[ticker]

def fetch_mean_forecast(fid: int) -> np.ndarray | None:
    r   = requests.get(f"{HOST}/forecasts/{fid}", proxies=PROXIES, timeout=30)
    raw = r.json()
    res = raw.get("result", {})
    if isinstance(res, str): res = json.loads(res)
    cands = res.get("candidates", [])
    if not cands: return None
    return np.mean([c["forecast_price"] for c in cands], axis=0)

def get_prices(ticker: str, origin_date: str, n_before: int = 20, n_after: int = MAX_K):
    candles = get_candles(ticker)
    idx = next((i for i, c in enumerate(candles) if c["begin"][:10] == origin_date), None)
    if idx is None: return None, None, None
    before = [float(candles[j]["close"]) for j in range(max(0, idx-n_before), idx+1)]
    after  = [float(candles[j]["close"]) for j in range(idx+1, min(idx+1+n_after, len(candles)))]
    ts_before = [candles[j]["begin"][:10] for j in range(max(0, idx-n_before), idx+1)]
    ts_after  = [candles[j]["begin"][:10] for j in range(idx+1, min(idx+1+n_after, len(candles)))]
    return before, after, ts_before + ts_after

# ─── Отбор прогнозов ──────────────────────────────────────────────────────────

import pandas as pd
df = pd.DataFrame(cal)

# Хороший: малый val_mape И малый mape_f5
good_cands = df[(df["val_mape"] < 0.002) & (df["mape_f5"] < 0.012)].sort_values("mape_f5")
# Средний: val_mape около медианы, mape_f5 близко к медиане
med_vm = df["val_mape"].median()
mid_cands = df[(df["val_mape"].between(med_vm*0.8, med_vm*1.2)) &
               (df["mape_f5"].between(0.018, 0.028))].sort_values("val_mape")
# Плохой: высокий val_mape, высокий mape_f5 (прошёл в выборку — без катастроф)
bad_cands = df[(df["val_mape"] > 0.012) & (df["mape_f5"] > 0.06)].sort_values("mape_f5", ascending=False)

picks = []
for label, pool, color in [
    ("Хороший",  good_cands, "#2ecc71"),
    ("Средний",  mid_cands,  "#f39c12"),
    ("Плохой",   bad_cands,  "#e74c3c"),
]:
    if len(pool) == 0:
        print(f"  {label}: нет кандидатов!")
        continue
    row = pool.iloc[0]
    picks.append((label, row, color))
    print(f"  {label}: {row['ticker']} {str(row['origin_ts'])[:10]}  "
          f"val_mape={row['val_mape']:.4f}  mape_f5={row['mape_f5']:.4f}")

# ─── Построение графика ───────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Conformal Prediction Intervals (LWC, 80% coverage)", fontsize=13, fontweight="bold")

ALPHA_LEVELS = [0.10, 0.20, 0.40]   # 90%, 80%, 60% CI
FILL_ALPHAS  = [0.15, 0.20, 0.25]
CI_LABELS    = ["90% CI", "80% CI", "60% CI"]

for ax, (label, row, color) in zip(axes, picks):
    fid    = int(row["forecast_id"])
    ticker = row["ticker"]
    origin = str(row["origin_ts"])[:10]
    vm     = float(row["val_mape"])

    print(f"\n  Загружаем {label} ({ticker} {origin} fid={fid})...")
    mean_fc = fetch_mean_forecast(fid)
    prices_before, prices_after, ts_all = get_prices(ticker, origin)

    if mean_fc is None or prices_before is None:
        print("    ошибка загрузки")
        continue

    n_before = len(prices_before)
    n_after  = min(len(prices_after), MAX_K, len(mean_fc))
    origin_close = prices_before[-1]

    # Индексы по горизонтальной оси
    x_hist = list(range(-n_before + 1, 1))        # ..., -1, 0
    x_fc   = list(range(1, n_after + 1))           # 1, 2, ..., n_after

    # Исторические цены
    ax.plot(x_hist, prices_before, color="black", lw=1.5, label="История")
    ax.axvline(0, color="gray", lw=0.8, ls="--")

    # Прогноз
    fc_vals = mean_fc[:n_after]
    ax.plot(x_fc, fc_vals, color=color, lw=2, label="Прогноз", zorder=5)

    # LWC полосы для разных уровней покрытия
    for alph, fill_a, ci_label in zip(ALPHA_LEVELS, FILL_ALPHAS, CI_LABELS):
        qs = lwc_quantile(vm, alpha=alph)[:n_after]
        lower = fc_vals - qs * origin_close
        upper = fc_vals + qs * origin_close
        ax.fill_between(x_fc, lower, upper, color=color, alpha=fill_a, label=ci_label)

    # Фактические цены
    ax.plot(x_fc[:len(prices_after)], prices_after[:n_after],
            color="black", lw=1.5, ls="--", marker="o", ms=3, label="Факт")

    # Оформление
    mape_pct = row["mape_f5"] * 100
    vm_pct   = vm * 100
    ax.set_title(f"{label}: {ticker} {origin}\n"
                 f"val_mape={vm_pct:.2f}%   mape_f5={mape_pct:.2f}%",
                 fontsize=10)
    ax.set_xlabel("Бар (0 = origin)")
    ax.set_ylabel("Цена, руб.")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    ax.axvspan(0.5, n_after + 0.5, color="lightyellow", alpha=0.3, zorder=0)

plt.tight_layout()
out = FIGURES / "06_conformal_intervals.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nСохранено: {out}")
plt.show()
