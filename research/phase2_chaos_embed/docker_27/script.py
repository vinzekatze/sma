"""
27 — GMM hard mask: multi-ticker тест.

Скрипт 26 показал: k_pca=8, k_gmm=5, hard → −9.1% на SBER 1d (N=200).
Цель: подтвердить на 8 тикерах (1600 точек) для статистической значимости.

Тестируемые конфиги:
  baseline          — LWR без кластеризации
  gmm_8_5_hard      — k_pca=8, k_gmm=5, hard mask       (лучший по медиане)
  gmm_10_4_soft     — k_pca=10, k_gmm=4, soft posterior  (значимый в скр. 26)
  gmm_5_4_hard      — k_pca=5, k_gmm=4, hard             (стабильный второй)

Параметры: p=20, ξ=63, val_h=10, N=200 origins per ticker.
Вывод: /output/27_results.json, /output/27_report.txt
"""

import json, time, sys
from pathlib import Path
from itertools import product as iproduct

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
except ImportError:
    print("sklearn not found"); sys.exit(1)

try:
    from scipy.stats import wilcoxon as _wlcx
    _HAS_SCI = True
except ImportError:
    _HAS_SCI = False

sys.path.insert(0, "/app")
from sma.core.forecast.normalize import normalize
from sma.core.forecast.embedding import build_delay_matrix, last_vector

# ── параметры ─────────────────────────────────────────────────────────────────
TICKERS  = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL = "1d"
MA_WIN   = 1000
DATA_DIR = Path("/app/data/candles")
OUT_DIR  = Path("/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

P       = 20
XI      = 3 * (P + 1)   # 63
VAL_H   = 10
N_EVAL  = 200
SEED    = 42

# (k_pca, k_gmm, mode)
CONFIGS = {
    "baseline":      None,
    "gmm_8_5_hard":  (8, 5, "hard"),
    "gmm_10_4_soft": (10, 4, "soft"),
    "gmm_5_4_hard":  (5, 4, "hard"),
}


# ── LWR ──────────────────────────────────────────────────────────────────────
def _lwr_step(Xm, ym, vec, xi, prob=None):
    dists  = np.linalg.norm(Xm - vec, axis=1)
    nn_idx = np.argpartition(dists, xi)[:xi]
    h_bw   = max(float(dists[nn_idx].max()), 1e-10)
    w      = np.exp(-0.5 * (dists[nn_idx] / h_bw) ** 2)
    if prob is not None:
        w *= prob[nn_idx]
        w  = np.clip(w, 1e-30, None)
    A  = np.hstack([np.ones((xi, 1)), Xm[nn_idx]])
    sw = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * ym[nn_idx], rcond=None)
    return float(c[0] + vec @ c[1:])


def _eval(X, y, vec, ratio, vo, xi, mask, prob):
    if mask is not None and mask.sum() >= xi:
        Xu, yu = X[mask], y[mask]
        pu     = prob[mask] if prob is not None else None
    else:
        Xu, yu, pu = X, y, prob

    v   = vec.copy()
    hat = np.empty(VAL_H)
    for h in range(VAL_H):
        hat[h] = _lwr_step(Xu, yu, v, xi, pu)
        v = np.roll(v, -1); v[-1] = hat[h]

    r0     = float(ratio[vo])
    r_hat  = r0 + np.cumsum(hat)
    actual = ratio[vo + 1: vo + 1 + VAL_H]
    n_     = min(len(r_hat), len(actual))
    if n_ == 0:
        return np.nan
    return float(np.mean(
        np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)
    ))


# ── основной цикл ─────────────────────────────────────────────────────────────
# results[ticker][config] = list of mape
results = {t: {c: [] for c in CONFIGS} for t in TICKERS}

t_total = time.time()

for ticker in TICKERS:
    path = DATA_DIR / ticker / f"{INTERVAL}.json"
    with open(path) as f:
        candles = json.load(f)
    df     = normalize(candles, window=MA_WIN).dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values
    dratio = np.diff(ratio)

    X_all, y_all = build_delay_matrix(dratio, P)

    # обучить PCA один раз на всём пуле для каждого k_pca
    pca_cache: dict[int, tuple] = {}
    needed_kpca = set()
    for cfg in CONFIGS.values():
        if cfg is not None:
            needed_kpca.add(cfg[0])
    for kp in needed_kpca:
        pca = PCA(n_components=kp).fit(X_all)
        pca_cache[kp] = (pca, pca.transform(X_all))

    min_orig = P + XI + 10
    max_orig = len(dratio) - VAL_H
    origins  = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)

    t1 = time.time()
    for vo in origins:
        pool = vo - P
        if pool < XI + 2:
            continue
        X_p  = X_all[:pool]
        y_p  = y_all[:pool]
        vec  = last_vector(dratio[:vo], P).copy()

        # baseline
        results[ticker]["baseline"].append(
            _eval(X_p, y_p, vec, ratio, vo, XI, None, None)
        )

        # GMM конфиги
        gmm_cache: dict[tuple, tuple] = {}   # (kp, kg) → (labels, probs)
        for cname, cfg in CONFIGS.items():
            if cfg is None:
                continue
            kp, kg, mode = cfg
            if (kp, kg) not in gmm_cache:
                pca, X_pca_a = pca_cache[kp]
                X_pca_p      = X_pca_a[:pool]
                vpca         = pca.transform(vec.reshape(1, -1))[0]
                gmm          = GaussianMixture(
                    n_components=kg, random_state=SEED,
                    max_iter=100, n_init=1,
                ).fit(X_pca_p)
                labels        = gmm.predict(X_pca_p)
                qlbl          = int(gmm.predict(vpca.reshape(1, -1))[0])
                probs_q       = gmm.predict_proba(X_pca_p)[:, qlbl]
                mask_q        = (labels == qlbl)
                gmm_cache[(kp, kg)] = (mask_q, probs_q)

            mask_q, probs_q = gmm_cache[(kp, kg)]
            mask = mask_q  if mode == "hard" else None
            prob = probs_q if mode == "soft" else None
            results[ticker][cname].append(
                _eval(X_p, y_p, vec, ratio, vo, XI, mask, prob)
            )

    # краткий статус на stdout
    bm   = float(np.nanmedian(results[ticker]["baseline"]))
    best = float(np.nanmedian(results[ticker]["gmm_8_5_hard"]))
    print(f"{ticker:5s}  N={len(origins):3d}  "
          f"base={bm:.5f}  gmm_8_5_hard={best:.5f}  "
          f"Δ={(best-bm)/bm*100:+.2f}%  t={time.time()-t1:.1f}s",
          flush=True)

print(f"\nTotal: {time.time()-t_total:.1f}s", flush=True)

# ── конвертация в numpy ───────────────────────────────────────────────────────
for t in TICKERS:
    for c in CONFIGS:
        results[t][c] = np.array(results[t][c])

# ── отчёт ─────────────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 80)
lines.append(f"27 — GMM hard mask: multi-ticker  |  p={P}, ξ={XI}, val_h={VAL_H}, N={N_EVAL}")
lines.append(f"Тикеры: {', '.join(TICKERS)}")
lines.append("=" * 80)

# per-ticker table
header = f"{'Тикер':>6}  {'N':>4}  {'baseline':>9}"
for cname in CONFIGS:
    if cname == "baseline": continue
    header += f"  {cname:>15}  {'Δ%':>7}"
lines.append(header)
lines.append("─" * 80)

agg = {c: [] for c in CONFIGS}

for ticker in TICKERS:
    bm = float(np.nanmedian(results[ticker]["baseline"]))
    n  = int(np.sum(~np.isnan(results[ticker]["baseline"])))
    row = f"{ticker:>6}  {n:>4}  {bm:>9.5f}"
    for cname in CONFIGS:
        if cname == "baseline": continue
        med = float(np.nanmedian(results[ticker][cname]))
        dp  = (med - bm) / bm * 100
        row += f"  {med:>15.5f}  {dp:>+6.2f}%"
    lines.append(row)
    for c in CONFIGS:
        agg[c].append(results[ticker][c])

lines.append("─" * 80)

# aggregated row
all_base = np.concatenate(agg["baseline"])
bm_all   = float(np.nanmedian(all_base))
n_all    = int(np.sum(~np.isnan(all_base)))
agg_row  = f"{'ИТОГО':>6}  {n_all:>4}  {bm_all:>9.5f}"
for cname in CONFIGS:
    if cname == "baseline": continue
    arr = np.concatenate(agg[cname])
    med = float(np.nanmedian(arr))
    dp  = (med - bm_all) / bm_all * 100
    agg_row += f"  {med:>15.5f}  {dp:>+6.2f}%"
lines.append(agg_row)
lines.append("=" * 80)

# Wilcoxon
if _HAS_SCI:
    lines.append("\n── Wilcoxon (aggregated) ──")
    for cname in CONFIGS:
        if cname == "baseline": continue
        arr  = np.concatenate(agg[cname])
        ok   = ~(np.isnan(all_base) | np.isnan(arr))
        diff = arr[ok] - all_base[ok]
        if ok.sum() < 10 or np.all(diff == 0):
            continue
        _, pv = _wlcx(diff)
        med   = float(np.nanmedian(arr))
        dp    = (med - bm_all) / bm_all * 100
        sig   = "✓ значимо" if pv < 0.05 else "— незначимо"
        lines.append(f"  {cname:<20}  ΔMAPE={dp:+.2f}%  p={pv:.4f}  {sig}")

report_text = "\n".join(lines)
print("\n" + report_text, flush=True)

# ── сохранение ────────────────────────────────────────────────────────────────
# JSON (все сырые mape)
json_out = {}
for t in TICKERS:
    json_out[t] = {}
    for c in CONFIGS:
        arr = results[t][c]
        json_out[t][c] = [x if not np.isnan(x) else None for x in arr.tolist()]

with open(OUT_DIR / "27_results.json", "w") as f:
    json.dump(json_out, f)

with open(OUT_DIR / "27_report.txt", "w") as f:
    f.write(report_text + "\n")

print(f"\nСохранено: {OUT_DIR}/27_results.json, 27_report.txt", flush=True)
