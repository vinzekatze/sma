"""
74 — Rolling SSA на ratio: казуальное сглаживание + сравнение с ценой.

Подход: применяем rolling SSA напрямую к ratio (а не к dratio),
чтобы избежать drift-проблемы при cumsum.

  ratio_ssa[t]  = rolling_ssa(ratio[:t+1], W, L, k)[-1]   (causal)
  price_ssa[t]  = ratio_ssa[t] * trend[t]

LP-causal для сравнения применяется к dratio + cumsum (как обычно).
Diff(ratio_ssa) — эффективный SSA dratio — показываем на панели сигналов.

Графики:
  A — цена: close / logtrend / LP-cumsum / SSA-на-ratio (несколько k)
  B — зум последних ZOOM баров
  C — dratio: raw / LP / diff(ratio_ssa) — сравнение сигналов
  D — фазовый портрет: raw / LP / diff(ratio_ssa)
"""

import json, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

ROOT    = Path(__file__).resolve().parent.parent.parent
FIGDIR  = ROOT / "research" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATADIR = ROOT / "data" / "candles"
sys.path.insert(0, str(ROOT))

TICKER   = "SBER"
INTERVAL = "1d"
ZOOM     = 300

SOS_LP = butter(8, 0.125, btype="low", output="sos")

# ── helpers ────────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n  = len(close); lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    b  = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a  = (cy - b * ct) / cn
    trend = np.exp(a + b * t); trend[:2] = close[:2]
    return trend


def rolling_ssa(series: np.ndarray, W: int, L: int, k: int) -> np.ndarray:
    """
    Causal rolling SSA applied to `series`.
    At each t >= W-1: SSA of series[t-W+1..t], returns last recon. value.
    For t < W-1: returns series[t] unchanged.
    """
    N    = len(series)
    out  = series.copy().astype(np.float64)
    K_m  = W - L + 1
    if K_m < 2:
        return out
    rows = np.arange(K_m)[:, None] + np.arange(L)[None, :]

    for t in range(W - 1, N):
        w  = series[t - W + 1:t + 1]
        X  = w[rows]
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        nk = min(k, len(s))
        Xr = (U[:, :nk] * s[:nk]) @ Vt[:nk, :]
        recon = np.zeros(W); cnt = np.zeros(W, dtype=np.int32)
        for i in range(K_m):
            recon[i:i + L] += Xr[i]; cnt[i:i + L] += 1
        out[t] = (recon / np.maximum(cnt, 1))[-1]
    return out

# ── загрузка данных ────────────────────────────────────────────────────────────

raw   = json.loads((DATADIR / TICKER / f"{INTERVAL}.json").read_text())
cands = raw["candles"] if isinstance(raw, dict) else raw
close = np.array([c["close"] for c in cands], dtype=np.float64)
trend = logtrend_causal(close)
ratio = close / trend
dratio = np.diff(ratio)
N      = len(close)
print(f"{TICKER} {INTERVAL}: {N} баров")

# ── LP-causal на dratio → cumsum → price ──────────────────────────────────────
lp_dr    = sosfilt(SOS_LP, dratio)
lp_ratio = ratio[0] + np.concatenate([[0.0], np.cumsum(lp_dr)])
lp_price = lp_ratio * trend

# ── Rolling SSA на ratio → price ──────────────────────────────────────────────
# W=128, L=32 — окно достаточно большое для разделения медленных мод
# Тестируем k=1,2,3 — от "только тренд" до нескольких мод
configs = [
    dict(W=128, L=32, k=1, color="#1a9850", ls="-",  lw=1.6, label="SSA W=128 L=32 k=1"),
    dict(W=128, L=32, k=2, color="#31a354", ls="-",  lw=1.3, label="SSA W=128 L=32 k=2"),
    dict(W=128, L=32, k=3, color="#74c476", ls="--", lw=1.3, label="SSA W=128 L=32 k=3"),
    dict(W=256, L=64, k=2, color="#006d2c", ls="-.", lw=1.3, label="SSA W=256 L=64 k=2"),
]

t0 = time.time()
for cfg in configs:
    print(f"  {cfg['label']} ...", flush=True)
    ratio_ssa     = rolling_ssa(ratio, cfg["W"], cfg["L"], cfg["k"])
    cfg["ratio"]  = ratio_ssa
    cfg["price"]  = ratio_ssa * trend
    cfg["dratio"] = np.diff(ratio_ssa)   # эффективный SSA dratio
print(f"Rolling SSA готов за {time.time()-t0:.1f}с")

# ── График A: полная история ───────────────────────────────────────────────────
fig_a, ax = plt.subplots(figsize=(14, 5))
ax.set_title(f"74: Реконструкция цены — {TICKER} (полная история)", fontsize=12)
ax.plot(close,    color="black",   lw=0.7, label="close (факт)")
ax.plot(trend,    color="grey",    lw=1.1, ls="--", label="logtrend", alpha=0.7)
ax.plot(lp_price, color="#fb8c00", lw=1.0, label="LP-causal (dratio→cumsum)", alpha=0.85)
for cfg in configs:
    ax.plot(cfg["price"], color=cfg["color"], lw=cfg["lw"],
            ls=cfg["ls"], label=cfg["label"], alpha=0.85)
ax.legend(fontsize=8, loc="upper left"); ax.set_xlabel("бар"); ax.set_ylabel("цена")
fig_a.tight_layout()
fig_a.savefig(FIGDIR / "74_price_full.png", dpi=120)
plt.close(fig_a)
print("Рис. A сохранён")

# ── График B: зум ─────────────────────────────────────────────────────────────
sl = slice(N - ZOOM, N)
fig_b, ax = plt.subplots(figsize=(14, 5))
ax.set_title(f"74: Зум последних {ZOOM} баров — {TICKER}", fontsize=12)
ax.plot(close[sl],    color="black",   lw=1.0, label="close (факт)")
ax.plot(trend[sl],    color="grey",    lw=1.1, ls="--", label="logtrend", alpha=0.7)
ax.plot(lp_price[sl], color="#fb8c00", lw=1.2, label="LP-causal", alpha=0.9)
for cfg in configs:
    ax.plot(cfg["price"][sl], color=cfg["color"], lw=cfg["lw"],
            ls=cfg["ls"], label=cfg["label"], alpha=0.9)
ax.legend(fontsize=8); ax.set_xlabel("бар (отн. окна)"); ax.set_ylabel("цена")
fig_b.tight_layout()
fig_b.savefig(FIGDIR / "74_price_zoom.png", dpi=120)
plt.close(fig_b)
print("Рис. B сохранён")

# ── График C: dratio сигналы ───────────────────────────────────────────────────
sl_dr = slice(N - 1 - ZOOM, N - 1)
n_panels = 2 + len(configs)
fig_c, axes = plt.subplots(n_panels, 1, figsize=(14, 2.2 * n_panels), sharex=True)
fig_c.suptitle(f"74: dratio сигналы — последние {ZOOM} баров", fontsize=11)

axes[0].plot(dratio[sl_dr], color="steelblue", lw=0.7)
axes[0].set_title("raw dratio"); axes[0].axhline(0, color="k", lw=0.4, ls="--")

axes[1].plot(lp_dr[sl_dr], color="#fb8c00", lw=1.0)
axes[1].set_title("LP-causal dratio"); axes[1].axhline(0, color="k", lw=0.4, ls="--")

for i, cfg in enumerate(configs):
    axes[2 + i].plot(cfg["dratio"][sl_dr], color=cfg["color"], lw=1.0)
    axes[2 + i].set_title(f"diff(ratio_ssa) — {cfg['label']}")
    axes[2 + i].axhline(0, color="k", lw=0.4, ls="--")

axes[-1].set_xlabel("бар")
fig_c.tight_layout()
fig_c.savefig(FIGDIR / "74_dratio_signals.png", dpi=120)
plt.close(fig_c)
print("Рис. C сохранён")

# ── График D: фазовый портрет ──────────────────────────────────────────────────
# Показываем raw / LP / лучший SSA (k=2)
best = configs[1]   # W=128 L=32 k=2
fig_d, axes = plt.subplots(1, 3, figsize=(14, 5))
fig_d.suptitle(f"74: Фазовый портрет dratio[t] vs dratio[t-1] — {TICKER}", fontsize=11)

def phase_plot(ax, sig, title, color):
    ax.scatter(sig[:-1], sig[1:], s=1.5, alpha=0.3, color=color)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("dratio[t-1]"); ax.set_ylabel("dratio[t]")
    ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)

phase_plot(axes[0], dratio,         "raw dratio",             "steelblue")
phase_plot(axes[1], lp_dr,          "LP-causal",              "#fb8c00")
phase_plot(axes[2], best["dratio"], f"diff(SSA) {best['label']}", best["color"])

fig_d.tight_layout()
fig_d.savefig(FIGDIR / "74_phase_portrait.png", dpi=120)
plt.close(fig_d)
print("Рис. D сохранён")

print("\nГотово.")
for ltr, fn in zip("ABCD", ["74_price_full.png", "74_price_zoom.png",
                             "74_dratio_signals.png", "74_phase_portrait.png"]):
    print(f"  {ltr}: {FIGDIR/fn}")
