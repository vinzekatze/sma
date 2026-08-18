"""
Сравнение нормализаций для зигзаг-прогноза.

Пять вариантов нормализации ряда z[i] на пивотах:
  ratio       — close / logtrend  (текущий стандарт)
  raw         — close  (сырая цена, нет нормализации)
  swing_pct   — (close[i] − close[i−1]) / close[i−1]  (относительный свинг)
  log_swing   — log(close[i] / close[i−1])  (лог-свинг)
  ratio_zscore— (ratio[i] − mean(ratio[-W:])) / std(ratio[-W:])  (локальный z-score)

Все прогнозы конвертируются в ratio-пространство для сравнения.
Метрика: rMAE = mean|pred_ratio − actual_ratio| / mean|Δactual_ratio|.
Дополнительно: H3-диагностика (bias по направлению up/down).

Базовые параметры: theta=8, p=2, T=2%, TRAIN_WIN=200, h=1.
SBER: пул 1d+1h+10m. LKOH: пул 1d.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ── Конфигурация ─────────────────────────────────────────────────────────────

THRESH     = 0.02
TRAIN_WIN  = 200
THETA      = 8.0
P          = 2
H          = 1
ZSCORE_W   = 100   # окно rolling-zscore (пивоты)

BASE = Path(__file__).parent
DATA = BASE.parent.parent.parent / "data" / "candles"
OUT  = BASE / "results"
OUT.mkdir(parents=True, exist_ok=True)

TICKER_CFG = {
    "SBER": {
        "1d":  DATA / "SBER" / "1d.json",
        "1h":  DATA / "SBER" / "1h.json",
        "10m": DATA / "SBER" / "10m.json",
    },
    "LKOH": {
        "1d":  DATA / "LKOH" / "1d.json",
    },
}

VARIANTS = ["ratio", "raw", "swing_pct", "log_swing", "ratio_zscore"]

# ── Базовые утилиты ──────────────────────────────────────────────────────────

def load_candles(path):
    with open(path) as f:
        data = json.load(f)
    close = np.array([d["close"] for d in data], dtype=np.float64)
    dates = np.array([d["begin"] for d in data])
    return close, dates


def logtrend_causal(close):
    n  = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    N  = np.arange(1, n + 1, dtype=np.float64)
    St, Sp   = np.cumsum(t),      np.cumsum(lc)
    St2, Stp = np.cumsum(t ** 2), np.cumsum(t * lc)
    den  = N * St2 - St ** 2
    b    = np.where(den > 1e-12, (N * Stp - St * Sp) / den, 0.0)
    a    = (Sp - b * St) / N
    trd  = np.exp(a + b * t)
    trd[:2] = close[:2]
    return trd, close / trd


def find_pivots(ratio, thr):
    pivots    = [0]
    direction = 0
    ext_val, ext_idx = ratio[0], 0
    for i in range(1, len(ratio)):
        v = ratio[i]
        if direction == 0:
            if abs(v - ext_val) >= thr * ext_val:
                direction = 1 if v > ext_val else -1
                ext_val, ext_idx = v, i
        elif direction == 1:
            if v > ext_val:
                ext_val, ext_idx = v, i
            elif (ext_val - v) >= thr * ext_val:
                pivots.append(ext_idx)
                direction = -1
                ext_val, ext_idx = v, i
        else:
            if v < ext_val:
                ext_val, ext_idx = v, i
            elif (v - ext_val) >= thr * ext_val:
                pivots.append(ext_idx)
                direction = 1
                ext_val, ext_idx = v, i
    return np.array(pivots, dtype=int)


# ── Построение z-рядов для каждой нормализации ───────────────────────────────

class SeriesData:
    """Всё необходимое для одного временного ряда (тикер + интервал)."""

    def __init__(self, path, thr=THRESH, zscore_w=ZSCORE_W):
        close, dates = load_candles(path)
        trend, ratio = logtrend_causal(close)
        pivots       = find_pivots(ratio, thr)
        n_piv        = len(pivots)

        # ratio на пивотах
        ratio_piv  = ratio[pivots]
        close_piv  = close[pivots]
        trend_piv  = trend[pivots]
        dates_piv  = dates[pivots]

        # ── z-ряды ───────────────────────────────────────────────────────────

        # ratio: уровень ratio на каждом пивоте
        z_ratio = ratio_piv.copy()

        # raw: цена закрытия на каждом пивоте
        z_raw = close_piv.copy()

        # swing_pct: (close[i] - close[i-1]) / close[i-1]
        z_swing_pct       = np.full(n_piv, np.nan)
        z_swing_pct[1:]   = (close_piv[1:] - close_piv[:-1]) / close_piv[:-1]

        # log_swing: log(close[i] / close[i-1])
        z_log_swing       = np.full(n_piv, np.nan)
        z_log_swing[1:]   = np.log(close_piv[1:] / close_piv[:-1])

        # ratio_zscore: (ratio[i] - mean(ratio[-W:])) / std(ratio[-W:])
        # Храним также mean и std для восстановления
        z_zscore     = np.full(n_piv, np.nan)
        zs_mean      = np.full(n_piv, np.nan)
        zs_std       = np.full(n_piv, np.nan)
        for i in range(zscore_w, n_piv):
            window     = ratio_piv[i - zscore_w : i]
            mu, sigma  = window.mean(), window.std()
            zs_mean[i] = mu
            zs_std[i]  = max(sigma, 1e-10)
            z_zscore[i] = (ratio_piv[i] - mu) / zs_std[i]

        self.close     = close
        self.ratio     = ratio
        self.trend     = trend
        self.pivots    = pivots
        self.dates_piv = dates_piv
        self.ratio_piv = ratio_piv
        self.close_piv = close_piv
        self.trend_piv = trend_piv
        self.zs_mean   = zs_mean
        self.zs_std    = zs_std
        self.z = {
            "ratio":        z_ratio,
            "raw":          z_raw,
            "swing_pct":    z_swing_pct,
            "log_swing":    z_log_swing,
            "ratio_zscore": z_zscore,
        }

    def make_X(self, variant):
        """Матрица признаков для варианта нормализации."""
        z   = self.z[variant]
        n   = len(z)
        X   = np.full((n, P), np.nan)
        if variant in ("swing_pct", "log_swing"):
            # swing-based: [z[i], z[i-1]]
            for i in range(P, n):
                X[i, 0] = z[i]
                X[i, 1] = z[i - 1]
        else:
            # level-based: [z[i], z[i]-z[i-1]]
            for i in range(P, n):
                X[i, 0] = z[i]
                X[i, 1] = z[i] - z[i - 1]
        return X

    def pred_to_ratio(self, variant, pred_z, step):
        """Конвертирует pred_z в предсказанный ratio."""
        if variant == "ratio":
            return pred_z
        elif variant == "raw":
            return pred_z / self.trend_piv[step]
        elif variant == "swing_pct":
            pred_close = self.close_piv[step] * (1.0 + pred_z)
            return pred_close / self.trend_piv[step]
        elif variant == "log_swing":
            pred_close = self.close_piv[step] * np.exp(pred_z)
            return pred_close / self.trend_piv[step]
        elif variant == "ratio_zscore":
            mu    = self.zs_mean[step]
            sigma = self.zs_std[step]
            if np.isnan(mu):
                return np.nan
            return pred_z * sigma + mu
        return np.nan

    def m0_ratio(self, step):
        """M0-прогноз (random walk) в ratio-пространстве."""
        return self.ratio_piv[step]


# ── S-map ────────────────────────────────────────────────────────────────────

def smap_pred(X_pool, y_pool, x_q, theta=THETA):
    if len(X_pool) < 3:
        return np.nan
    mu    = X_pool.mean(0)
    sigma = X_pool.std(0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    Xn    = (X_pool - mu) / sigma
    xn    = (x_q    - mu) / sigma
    dists = np.sqrt(((Xn - xn) ** 2).sum(1))
    d_bar = dists.mean()
    w = np.ones(len(y_pool)) if (theta == 0.0 or d_bar < 1e-12) \
        else np.exp(-theta * dists / d_bar)
    ws   = np.sqrt(w)
    A    = np.column_stack([np.ones(len(y_pool)), Xn]) * ws[:, None]
    b    = y_pool * ws
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0] + coef[1:] @ xn)


def build_pool(step, ts, X_prim, z_prim, dates_prim, aux_list):
    """Обучающий пул: 1d-окно + aux до текущей даты. h=1."""
    Xr, yr = [], []
    for j in range(ts + P, step):          # j+1 <= step-1  →  j+H < step ✓
        if np.any(np.isnan(X_prim[j])):
            continue
        if j + H >= len(z_prim) or np.isnan(z_prim[j + H]):
            continue
        Xr.append(X_prim[j])
        yr.append(z_prim[j + H])

    cur_date = dates_prim[step]
    for z_a, X_a, d_a in aux_list:
        ce = int(np.searchsorted(d_a, cur_date, side="left"))
        for k in range(P, ce - H):
            if np.any(np.isnan(X_a[k])):
                continue
            if k + H >= len(z_a) or np.isnan(z_a[k + H]):
                continue
            Xr.append(X_a[k])
            yr.append(z_a[k + H])

    if not Xr:
        return None, None
    return np.array(Xr), np.array(yr)


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(ticker, cfg):
    print(f"\n{'-'*60}")
    print(f"  Тикер: {ticker}")

    sd1d = SeriesData(cfg["1d"])

    # Aux-ряды: для каждого варианта нормализации отдельный список aux
    aux = {v: [] for v in VARIANTS}
    for key in ("1h", "10m"):
        if key not in cfg:
            continue
        sd_a = SeriesData(cfg[key])
        for v in VARIANTS:
            X_a = sd_a.make_X(v)
            aux[v].append((sd_a.z[v], X_a, sd_a.dates_piv))

    # Матрицы признаков для 1d
    Xmat = {v: sd1d.make_X(v) for v in VARIANTS}

    n1d        = len(sd1d.pivots)
    test_start = TRAIN_WIN + P
    test_end   = n1d - H

    print(f"  пивотов 1d: {n1d}  тест: [{test_start}-{test_end}]  n={test_end-test_start}")

    records = []

    for step in range(test_start, test_end):
        ts         = max(0, step - TRAIN_WIN)
        actual_rat = sd1d.ratio_piv[step + H]
        m0_rat     = sd1d.m0_ratio(step)
        direction  = "up" if sd1d.ratio_piv[step] < sd1d.ratio_piv[step - 1] else "down"

        rec = {
            "step":       step,
            "actual_rat": actual_rat,
            "m0_rat":     m0_rat,
            "direction":  direction,
        }

        for v in VARIANTS:
            x_q = Xmat[v][step]
            if np.any(np.isnan(x_q)):
                rec[f"pred_{v}"] = np.nan
                continue

            Xp, yp = build_pool(step, ts, Xmat[v], sd1d.z[v], sd1d.dates_piv, aux[v])
            if Xp is None or len(Xp) < 5:
                rec[f"pred_{v}"] = np.nan
                continue

            pred_z         = smap_pred(Xp, yp, x_q)
            pred_rat       = sd1d.pred_to_ratio(v, pred_z, step)
            rec[f"pred_{v}"] = pred_rat

        records.append(rec)

        if step % 100 == 0:
            print(f"  step {step}/{test_end}", flush=True)

    df = pd.DataFrame(records)
    for v in VARIANTS:
        df[f"serr_{v}"] = df[f"pred_{v}"] - df["actual_rat"]
        df[f"aerr_{v}"] = df[f"serr_{v}"].abs()
    df["m0_aerr"] = (df["m0_rat"] - df["actual_rat"]).abs()
    return df


# ── Аналитика ─────────────────────────────────────────────────────────────────

def analyze(ticker, df):
    print(f"\n{'='*60}")
    print(f"  Анализ: {ticker}  (n={len(df)})")
    print(f"{'='*60}")

    mean_dz = df["actual_rat"].diff().abs().mean()
    if mean_dz < 1e-12:
        mean_dz = 1.0

    m0_rmae = df["m0_aerr"].mean() / mean_dz
    print(f"\n  M0 rMAE: {m0_rmae:.4f}  (mean_dz={mean_dz:.5f})\n")

    # rMAE и bias
    print(f"  {'Вариант':14}  {'bias':>10}  {'rMAE':>7}  {'vs M0':>8}  {'n':>5}")
    for v in VARIANTS:
        s = df[f"serr_{v}"].dropna()
        a = df[f"aerr_{v}"].dropna()
        if len(s) == 0:
            print(f"  {v:14}  нет данных")
            continue
        bias = s.mean()
        rmae = a.mean() / mean_dz
        print(f"  {v:14}  {bias:>+10.5f}  {rmae:>7.4f}  {(rmae/m0_rmae-1)*100:>+7.1f}%  {len(s):>5}")

    # H3: асимметрия по направлению
    print(f"\n  -- H3: bias по направлению (BASE ratio vs swing_pct vs log_swing) --")
    print(f"  {'Вариант':14}  {'bias_up':>9}  {'bias_down':>10}  {'разница':>10}")
    for v in ["ratio", "swing_pct", "log_swing"]:
        up = df.loc[df["direction"] == "up",   f"serr_{v}"].mean()
        dn = df.loc[df["direction"] == "down", f"serr_{v}"].mean()
        print(f"  {v:14}  {up:>+9.5f}  {dn:>+10.5f}  {up-dn:>+10.5f}")

    # Корреляции с уровнем ratio
    print(f"\n  -- H1: corr(signed_err, actual_ratio_current) --")
    for v in VARIANTS:
        sub = df[[f"serr_{v}", "m0_rat"]].dropna()
        if len(sub) < 10:
            continue
        r, p = scipy_stats.pearsonr(sub[f"serr_{v}"], sub["m0_rat"])
        sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {v:14}  r={r:+.3f}  p={p:.3f} {sig}")

    return m0_rmae


# ── График ────────────────────────────────────────────────────────────────────

COLORS = {
    "ratio":        "steelblue",
    "raw":          "gray",
    "swing_pct":    "darkorange",
    "log_swing":    "green",
    "ratio_zscore": "purple",
}


def plot_results(ticker, df, m0_rmae):
    mean_dz = df["actual_rat"].diff().abs().mean()

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(
        f"{ticker}: сравнение нормализаций  "
        f"(theta={THETA}, p={P}, T={THRESH*100:.0f}%, W={TRAIN_WIN})",
        fontsize=11)

    # 1. rMAE bar
    ax = axes[0, 0]
    labels_bar = ["M0"] + VARIANTS
    rmaes = [m0_rmae] + [
        df[f"aerr_{v}"].dropna().mean() / mean_dz for v in VARIANTS
    ]
    colors_bar = ["#aaaaaa"] + [COLORS[v] for v in VARIANTS]
    bars = ax.bar(labels_bar, rmaes, color=colors_bar, alpha=0.85)
    for bar, val in zip(bars, rmaes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_title("rMAE по вариантам нормализации")
    ax.set_ylabel("rMAE (ratio-space)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.2)

    # 2. signed_err по времени (ratio, swing_pct, log_swing)
    ax = axes[0, 1]
    ax.axhline(0, color="black", lw=0.8)
    for v in ["ratio", "swing_pct", "log_swing"]:
        roll = df[f"serr_{v}"].rolling(20, min_periods=5).mean()
        ax.plot(df["step"], roll, color=COLORS[v], lw=1.8, label=v, alpha=0.85)
    ax.set_title("Rolling mean signed_err (w=20)")
    ax.set_xlabel("step")
    ax.set_ylabel("mean(pred − actual) ratio-space")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # 3. H3 bar: bias up vs down
    ax = axes[0, 2]
    h3_variants = ["ratio", "swing_pct", "log_swing"]
    x = np.arange(len(h3_variants))
    w = 0.35
    bias_up = [df.loc[df["direction"]=="up",   f"serr_{v}"].mean() for v in h3_variants]
    bias_dn = [df.loc[df["direction"]=="down",  f"serr_{v}"].mean() for v in h3_variants]
    ax.bar(x - w/2, bias_up, w, label="up (к хаю)", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, bias_dn, w, label="down (к лою)", color="salmon", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(h3_variants, fontsize=8)
    ax.set_title("H3: bias по направлению")
    ax.set_ylabel("mean(pred − actual) ratio-space")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # 4. Распределение signed_err: ratio vs swing
    ax = axes[1, 0]
    for v in ["ratio", "swing_pct", "log_swing", "ratio_zscore"]:
        vals = df[f"serr_{v}"].dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=40, alpha=0.4, color=COLORS[v], label=v, density=True)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Распределение signed_err (в ratio-space)")
    ax.set_xlabel("pred − actual")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # 5. raw vs ratio: scatter signed_err vs price level
    ax = axes[1, 1]
    for v in ["ratio", "raw"]:
        sub = df[["m0_rat", f"serr_{v}"]].dropna()
        if len(sub) < 5:
            continue
        ax.scatter(sub["m0_rat"], sub[f"serr_{v}"],
                   alpha=0.15, s=8, color=COLORS[v])
        zf = np.polyfit(sub["m0_rat"], sub[f"serr_{v}"], 1)
        xs = np.linspace(sub["m0_rat"].min(), sub["m0_rat"].max(), 50)
        ax.plot(xs, np.polyval(zf, xs), color=COLORS[v], lw=2, label=v)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("signed_err ~ уровень ratio (ratio vs raw)")
    ax.set_xlabel("ratio_current")
    ax.set_ylabel("pred − actual (ratio-space)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # 6. ratio_zscore: rMAE тренд по времени
    ax = axes[1, 2]
    w20 = 20
    for v in VARIANTS:
        roll_abs = df[f"aerr_{v}"].rolling(w20, min_periods=5).mean() / mean_dz
        ax.plot(df["step"], roll_abs, color=COLORS[v], lw=1.5, label=v, alpha=0.8)
    ax.axhline(m0_rmae, color="black", lw=0.8, ls="--", label="M0")
    ax.set_title(f"Rolling rMAE (w={w20})")
    ax.set_xlabel("step")
    ax.set_ylabel("rMAE")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = OUT / f"{ticker}_norm_study.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  График: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    summary_rows = []

    for ticker, cfg in TICKER_CFG.items():
        df      = walk_forward(ticker, cfg)
        m0_rmae = analyze(ticker, df)
        plot_results(ticker, df, m0_rmae)

        df["ticker"] = ticker
        csv_path = OUT / f"{ticker}_norm_study.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}")

        mean_dz = df["actual_rat"].diff().abs().mean()
        for v in VARIANTS:
            s = df[f"serr_{v}"].dropna()
            a = df[f"aerr_{v}"].dropna()
            up = df.loc[df["direction"]=="up",   f"serr_{v}"].mean()
            dn = df.loc[df["direction"]=="down",  f"serr_{v}"].mean()
            summary_rows.append({
                "ticker":    ticker,
                "variant":   v,
                "bias":      s.mean() if len(s) else np.nan,
                "rMAE":      a.mean() / mean_dz if len(a) else np.nan,
                "m0_rMAE":   m0_rmae,
                "bias_up":   up,
                "bias_down": dn,
                "n":         len(s),
            })

    print(f"\n{'='*65}")
    print("  Кросс-тикерная сводка")
    print(f"{'='*65}")
    print(f"  {'Тикер':6}  {'Вариант':14}  {'bias':>10}  {'rMAE':>7}  {'vs M0':>8}")
    for r in summary_rows:
        vs = (r["rMAE"] / r["m0_rMAE"] - 1) * 100 if r["m0_rMAE"] else np.nan
        print(f"  {r['ticker']:6}  {r['variant']:14}  "
              f"{r['bias']:>+10.5f}  {r['rMAE']:>7.4f}  {vs:>+7.1f}%")

    pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)
    print(f"\n  Сводка: {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
