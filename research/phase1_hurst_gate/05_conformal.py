"""
Исследование 05: Conformal Prediction для интервалов прогноза.

Сравниваем три подхода:
  1. Split Conformal — глобальный квантиль, гарантия 1-alpha
  2. Locally Weighted Conformal — взвешен по сходству val_mape
  3. Наш empirical power-law — (vm/0.0043)^0.58

Данные: research/results/04_results.jsonl (168 прогнозов, rel_errors по барам).
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

records = [json.loads(l) for l in open(RESULTS / "04_results.jsonl")]
df = pd.DataFrame(records)
df = df[df["mape_f5"] <= 0.20].copy().reset_index(drop=True)
n = len(df)
print(f"Калибровочная выборка: n={n}")

MAX_K = 10
scores = np.full((n, MAX_K), np.nan)
for i, row in df.iterrows():
    for k in range(min(MAX_K, len(row["rel_errors"]))):
        scores[i, k] = abs(row["rel_errors"][k])

vm = df["val_mape"].values
alpha = 0.20  # целевое покрытие 80%

# ─── 1. Split Conformal (LOO) ──────────────────────────────────────────────────

def loo_split_conformal(scores_k, alpha):
    """LOO coverage для глобального split conformal."""
    covered, widths = 0, []
    sorted_idx = np.argsort(scores_k)
    for i in range(len(scores_k)):
        cal = np.concatenate([scores_k[:i], scores_k[i+1:]])
        cal_sorted = np.sort(cal)
        q_idx = min(int(np.ceil((len(cal) + 1) * (1 - alpha))) - 1, len(cal) - 1)
        q = cal_sorted[q_idx]
        widths.append(q)
        if scores_k[i] <= q:
            covered += 1
    return covered / len(scores_k), float(np.median(widths))


print()
print("=" * 65)
print("1. SPLIT CONFORMAL (глобальный, LOO)")
print(f"   Целевое покрытие: {(1-alpha)*100:.0f}%")
print("=" * 65)
print(f"  bar   coverage  median_q")

sc_coverages, sc_widths = [], []
for k in range(MAX_K):
    sk = scores[:, k]
    sk = sk[~np.isnan(sk)]
    cov, mq = loo_split_conformal(sk, alpha)
    sc_coverages.append(cov)
    sc_widths.append(mq)
    print(f"  {k+1:>3}   {cov*100:>7.1f}%  {mq*100:>7.2f}%")


# ─── 2. Locally Weighted Conformal (LOO) ──────────────────────────────────────

def weighted_quantile(values, weights, alpha):
    """Взвешенный квантиль уровня 1-alpha."""
    idx = np.argsort(values)
    vals_s = values[idx]
    w_s = weights[idx]
    w_s = w_s / w_s.sum()
    cumw = np.cumsum(w_s)
    j = np.searchsorted(cumw, 1 - alpha)
    return float(vals_s[min(j, len(vals_s) - 1)])


def loo_lwc(scores_k, vm_all, alpha, h):
    """LOO locally weighted conformal."""
    covered, widths = 0, []
    for i in range(len(scores_k)):
        cal_s  = np.concatenate([scores_k[:i], scores_k[i+1:]])
        cal_vm = np.concatenate([vm_all[:i], vm_all[i+1:]])
        vm_new = vm_all[i]
        log_ratio = np.log(cal_vm / vm_new)
        w = np.exp(-log_ratio**2 / (2 * h**2))
        # conformal augmentation: добавляем точку с весом среднего и score=inf
        w_aug = np.append(w, np.mean(w))
        s_aug = np.append(cal_s, np.inf)
        q = weighted_quantile(s_aug, w_aug, alpha)
        widths.append(q if np.isfinite(q) else np.nan)
        if scores_k[i] <= q:
            covered += 1
    return covered / len(scores_k), float(np.nanmedian(widths))


print()
print("=" * 65)
print("2. LOCALLY WEIGHTED CONFORMAL (LOO)")
print(f"   Целевое покрытие: {(1-alpha)*100:.0f}%  |  ядро Гаусса в лог-пространстве")
print("=" * 65)

bandwidths = [0.5, 1.0, 2.0]
print(f"  bar  " + "  ".join(f"h={h:.1f}: cov   mq  " for h in bandwidths))

lwc_results = {h: {"coverages": [], "widths": []} for h in bandwidths}

for k in range(MAX_K):
    sk  = scores[:, k]
    msk = ~np.isnan(sk)
    sk_c  = sk[msk]
    vm_c  = vm[msk]
    row_str = f"  {k+1:>3}  "
    for h in bandwidths:
        cov, mq = loo_lwc(sk_c, vm_c, alpha, h)
        lwc_results[h]["coverages"].append(cov)
        lwc_results[h]["widths"].append(mq)
        row_str += f"  h={h:.1f}: {cov*100:>5.1f}% {mq*100:>5.2f}%"
    print(row_str)


# ─── 3. Адаптивность: ширина PI при разных val_mape ──────────────────────────

print()
print("=" * 65)
print("3. АДАПТИВНОСТЬ: ширина PI по квинтилям val_mape (бар 5, h=1.0)")
print("   (сравниваем LWC vs empirical power-law vs split conformal)")
print("=" * 65)

k = 4  # бар 5
sk  = scores[:, k][~np.isnan(scores[:, k])]
vm_c = vm[~np.isnan(scores[:, k])]

REF = 0.0043
BETA = 0.58
BASE_P80 = np.quantile(sk[vm_c < REF], 0.80) if (vm_c < REF).sum() > 5 else 0.036

vm_quantiles = np.quantile(vm_c, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
vm_centers   = [(vm_quantiles[i] + vm_quantiles[i+1]) / 2 for i in range(5)]

print(f"  vm_center  LWC(h=1.0)  power-law   split_cf   actual_p80")

for vm_test in vm_centers:
    # LWC: взвешенный квантиль по всей выборке (не LOO, для анализа)
    log_ratio = np.log(vm_c / vm_test)
    w = np.exp(-log_ratio**2 / (2 * 1.0**2))
    w_aug = np.append(w, np.mean(w))
    s_aug = np.append(sk, np.inf)
    q_lwc = weighted_quantile(s_aug, w_aug, alpha)

    # Power-law empirical
    scale = (vm_test / REF) ** BETA
    q_pl = BASE_P80 * scale

    # Split conformal (глобальный, не зависит от vm)
    q_sc = sc_widths[k]

    # Фактический p80 в окне ±30% от vm_test (приблизительно)
    mask = np.abs(np.log(vm_c / vm_test)) < 0.5
    q_act = float(np.quantile(sk[mask], 0.80)) if mask.sum() >= 5 else np.nan

    print(f"  {vm_test:.5f}   {q_lwc*100:>7.2f}%    {q_pl*100:>7.2f}%   {q_sc*100:>7.2f}%  "
          f"  {q_act*100:>7.2f}%" if not np.isnan(q_act) else
          f"  {vm_test:.5f}   {q_lwc*100:>7.2f}%    {q_pl*100:>7.2f}%   {q_sc*100:>7.2f}%      (мало данных)")


# ─── 4. Итог: покрытие по методам ─────────────────────────────────────────────

print()
print("=" * 65)
print("4. ИТОГ: СРЕДНЕЕ ФАКТИЧЕСКОЕ ПОКРЫТИЕ ПО БАРАМ 1-10")
print(f"   Цель: {(1-alpha)*100:.0f}%")
print("=" * 65)

print(f"  Split Conformal:          "
      f"{np.mean(sc_coverages)*100:.1f}%  (min={min(sc_coverages)*100:.1f}%  max={max(sc_coverages)*100:.1f}%)")
for h in bandwidths:
    covs = lwc_results[h]["coverages"]
    print(f"  LWC h={h:.1f}:               "
          f"{np.mean(covs)*100:.1f}%  (min={min(covs)*100:.1f}%  max={max(covs)*100:.1f}%)")

print()
print("Ключевой вопрос: LWC дает меньшую ширину PI при малом vm?")
for h in bandwidths:
    w_lwc = lwc_results[h]["widths"]
    ratio = np.mean(w_lwc) / np.mean(sc_widths)
    print(f"  LWC h={h:.1f} vs split_cf: средняя ширина {ratio:.2f}x")
