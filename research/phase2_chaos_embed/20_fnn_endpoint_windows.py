"""
FNN на 10 последовательных сегментах по 200 баров от конца ряда.

Берём последние 2000 баров Δratio, разбиваем на 10 непересекающихся
блоков по 200 баров (последний блок = самые свежие данные),
и запускаем FNN на каждом.

Вопрос: насколько однороден «режим» ряда по последним 2000 барам?
Есть ли блоки, где FNN заметно ниже глобального baseline?
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from sma.core.forecast.normalize import normalize

# ── параметры ────────────────────────────────────────────────────────────────
MA_WINDOW   = 1000
TICKER      = "SBER"
INTERVAL    = "1d"

N_BLOCKS    = 10     # количество сегментов
BLOCK_SIZE  = 200    # баров в каждом сегменте
M_MAX       = 8
TAU         = 1
R_TOL       = 10.0
A_TOL       = 2.0


# ── FNN ───────────────────────────────────────────────────────────────────────
def fnn(series: np.ndarray, m_max: int = M_MAX, tau: int = TAU) -> dict:
    from sklearn.neighbors import KDTree
    sigma = np.std(series)
    if sigma < 1e-12:
        return {}
    res = {}
    for m in range(1, m_max + 1):
        n_m1 = len(series) - m * tau - 1
        n_m  = len(series) - (m - 1) * tau - 1
        if n_m1 < 10:
            break
        X_m = np.column_stack([series[k * tau: k * tau + n_m] for k in range(m)])
        dists, idxs = KDTree(X_m[:n_m1]).query(X_m[:n_m1], k=2)
        R_m   = dists[:, 1]
        j     = idxs[:, 1]
        ei    = series[m * tau: m * tau + n_m1]
        ej    = series[m * tau: m * tau + n_m1][j]
        R_m1  = np.sqrt(R_m**2 + (ei - ej)**2)
        is_false = (np.abs(ei - ej) / (R_m + 1e-12) > R_TOL) | (R_m1 / sigma > A_TOL)
        valid = R_m > 1e-12
        res[m] = float(np.sum(is_false[valid]) / np.sum(valid)) if np.any(valid) else np.nan
    return res


# ── данные ────────────────────────────────────────────────────────────────────
path = DATA_DIR / TICKER / f"{INTERVAL}.json"
with open(path) as f:
    candles = json.load(f)

df     = normalize(candles, window=MA_WINDOW).dropna(subset=["ma"]).reset_index(drop=True)
dratio = np.diff(df["ratio"].values)

total_needed = N_BLOCKS * BLOCK_SIZE
print(f"{TICKER} {INTERVAL}: {len(dratio)} баров Δratio")
print(f"Берём последние {total_needed} баров → {N_BLOCKS} блоков × {BLOCK_SIZE} баров")

tail = dratio[-total_needed:]

# baseline — весь ряд
fnn_all = fnn(dratio)
base2 = fnn_all.get(2, np.nan)
base3 = fnn_all.get(3, np.nan)
print(f"\nBaseline (весь ряд {len(dratio)} баров): FNN m=2={base2:.1%}  m=3={base3:.1%}")

# ── FNN по блокам ─────────────────────────────────────────────────────────────
print(f"\n{'Блок':>6}  {'Позиция':>14}  {'FNN m=2':>9}  {'FNN m=3':>9}  {'vs base m=2':>12}")
print("─" * 62)

block_results = []
for i in range(N_BLOCKS):
    start = i * BLOCK_SIZE
    end   = start + BLOCK_SIZE
    seg   = tail[start:end]

    # абсолютные позиции в исходном ряду
    abs_start = len(dratio) - total_needed + start
    abs_end   = abs_start + BLOCK_SIZE - 1

    f  = fnn(seg)
    m2 = f.get(2, np.nan)
    m3 = f.get(3, np.nan)

    diff = (m2 - base2) * 100 if not np.isnan(m2) else np.nan
    flag = "▼" if diff < -5 else ("▲" if diff > 5 else " ")
    label = f"#{i+1} (крайний)" if i == N_BLOCKS - 1 else f"#{i+1}"

    print(f"{label:>10}  [{abs_start:>4}…{abs_end:>4}]  {m2:>8.1%}  {m3:>8.1%}  {diff:>+10.1f}%{flag}")
    block_results.append({"block": i + 1, "start": abs_start, "end": abs_end,
                           "fnn": f, "seg": seg})


# ── графики ───────────────────────────────────────────────────────────────────
colors = cm.RdYlGn(np.linspace(0.1, 0.9, N_BLOCKS))

# рис. 1: FNN-кривые всех блоков + baseline
fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle(f"FNN по {N_BLOCKS} блокам × {BLOCK_SIZE} баров от конца ряда  |  {TICKER} {INTERVAL}",
             fontsize=12)

for br, col in zip(block_results, colors):
    f  = br["fnn"]
    ms = sorted(f)
    if not ms:
        continue
    vals = [f[m] * 100 for m in ms]
    lw   = 2.5 if br["block"] == N_BLOCKS else 1.3
    alpha = 1.0 if br["block"] == N_BLOCKS else 0.7
    label = f"блок #{br['block']} [{br['start']}–{br['end']}]"
    if br["block"] == N_BLOCKS:
        label += "  ← крайний (свежий)"
    ax.plot(ms, vals, "o-", color=col, linewidth=lw, markersize=5,
            alpha=alpha, label=label)

# baseline
ms_b = sorted(fnn_all)
ax.plot(ms_b, [fnn_all[m] * 100 for m in ms_b], "s--", color="black",
        linewidth=2.2, markersize=7, alpha=0.7, label=f"baseline (весь ряд, {len(dratio)} баров)")
ax.axhline(5, color="red", linestyle=":", linewidth=1.2, label="5%")

ax.set_xlabel("m (размерность вложения)")
ax.set_ylabel("FNN, %")
ax.set_xticks(range(1, M_MAX + 1))
ax.set_ylim(0, 105)
ax.legend(fontsize=7.5, ncol=2, loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()

out1 = OUT_DIR / "20_fnn_endpoint_blocks.png"
plt.savefig(out1, dpi=140, bbox_inches="tight")
print(f"\nГрафик (кривые): {out1}")


# рис. 2: FNN(m=2) и FNN(m=3) по номеру блока (хронологически)
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
fig2.suptitle(f"FNN по хронологии блоков  |  {TICKER} {INTERVAL}  (← прошлое  |  настоящее →)",
              fontsize=11)

block_nums = [r["block"] for r in block_results]
for ax2, m_t, title, col in [
    (ax2a, 2, "FNN m=2", "steelblue"),
    (ax2b, 3, "FNN m=3", "darkorange"),
]:
    vals = [r["fnn"].get(m_t, np.nan) * 100 for r in block_results]
    base = fnn_all.get(m_t, np.nan) * 100

    bar_cols = []
    for v in vals:
        if np.isnan(v):
            bar_cols.append("lightgray")
        elif v < base - 5:
            bar_cols.append("seagreen")
        elif v > base + 5:
            bar_cols.append("tomato")
        else:
            bar_cols.append("steelblue" if m_t == 2 else "darkorange")

    ax2.bar(block_nums, vals, color=bar_cols, alpha=0.8, edgecolor="white")
    ax2.axhline(base, color="black", linestyle="--", linewidth=1.5,
                label=f"baseline = {base:.1f}%")
    ax2.axhline(5, color="red", linestyle=":", linewidth=1.2, label="5%")
    ax2.set_xlabel("Номер блока (1=старый, 10=свежий)")
    ax2.set_ylabel("FNN, %")
    ax2.set_title(title)
    ax2.set_xticks(block_nums)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out2 = OUT_DIR / "20_fnn_endpoint_barchart.png"
plt.savefig(out2, dpi=140, bbox_inches="tight")
print(f"График (бары):   {out2}")


# ── итог ──────────────────────────────────────────────────────────────────────
print("\n── Итог ──")
vals2 = [r["fnn"].get(2, np.nan) for r in block_results]
valid_vals = [v for v in vals2 if not np.isnan(v)]
if valid_vals:
    print(f"FNN m=2:  min={min(valid_vals):.1%}  max={max(valid_vals):.1%}  "
          f"median={np.median(valid_vals):.1%}  baseline={base2:.1%}")
    best = block_results[np.argmin(valid_vals)]
    worst = block_results[np.argmax(valid_vals)]
    print(f"Лучший блок:   #{best['block']} [{best['start']}–{best['end']}] → "
          f"FNN m=2={best['fnn'].get(2, np.nan):.1%}")
    print(f"Худший блок:   #{worst['block']} [{worst['start']}–{worst['end']}] → "
          f"FNN m=2={worst['fnn'].get(2, np.nan):.1%}")
    last_fnn = block_results[-1]["fnn"].get(2, np.nan)
    print(f"\nКрайний блок (самый свежий): FNN m=2 = {last_fnn:.1%}  "
          f"({'ниже' if last_fnn < base2 else 'выше'} baseline на "
          f"{abs(last_fnn - base2) * 100:.1f}%)")

plt.show()
