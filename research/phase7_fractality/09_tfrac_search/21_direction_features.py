#!/usr/bin/env python3
"""
21_direction_features.py — Поиск предикторов oracle T_frac по направлениям.

Берём oracle из скр.20 (сетка 0.1%, 1.5-4.0%) и ищем признаки,
коррелирующие с oracle_tf раздельно для up (LOW pivot) и down (HIGH pivot).

Признаки (каузальные — только из данных до текущего step):
  A) T_BIG зигзаг:
     amp_curr      — амплитуда текущего плеча |lp[step] - lp[step-1]|
     amp_prev      — амплитуда предыдущего плеча
     amp_ma5/10    — rolling mean амплитуды за 5/10 событий
     amp_z         — z-score amp_curr относительно amp_ma10
     amp_ratio     — amp_curr / amp_ma5
     dur_curr      — длина текущего плеча в барах (подтверждение − старт)
     dur_ma5       — rolling mean длины за 5 событий
     dur_ratio     — dur_curr / dur_ma5

  B) Сырой лог-ряд (log(high/low)), на баре подтверждения:
     vol_10/30/100 — std log(high/low) за N баров
     vol_r10_30    — vol_10 / vol_30
     atr_10/30     — mean log(high/low) за N баров
     rng_10/30     — max(lh)-min(ll) за N баров

  C) Контекст:
     hour          — час подтверждения (0-23)
     dow           — день недели (0=пн)
     price_pos100  — позиция цены в диапазоне последних 100 баров (0-1)

Метрики: Pearson r + Spearman ρ vs oracle_tf, раздельно up/down.
Графики: heatmap корреляций, scatter top-признаков.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from datetime import datetime
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

ORACLE_CSV = RESULTS / "stability_oracle_10m.csv"
T_BIG      = 0.04


# ── загрузка ──────────────────────────────────────────────────────────────────

def load_candles(path):
    with open(path) as f:
        raw = json.load(f)
    lh  = np.log(np.array([c["high"]  for c in raw], dtype=np.float64))
    ll  = np.log(np.array([c["low"]   for c in raw], dtype=np.float64))
    dt  = np.array([c["begin"] for c in raw])
    bar = np.arange(len(raw))
    return lh, ll, dt, bar


def build_zigzag(lh, ll, dt, thr):
    """Возвращает (log_prices, confirm_dates, directions, confirm_bar_indices)."""
    lp, cd, dirs, cb = [], [], [], []
    cur_dir = 0
    ext     = (lh[0] + ll[0]) / 2.0
    ext_bar = 0
    for i in range(len(lh)):
        if cur_dir == 0:
            if lh[i] - ext >= thr:
                cur_dir, ext, ext_bar = 1, lh[i], i
            elif ext - ll[i] >= thr:
                cur_dir, ext, ext_bar = -1, ll[i], i
        elif cur_dir == 1:
            if lh[i] > ext:
                ext, ext_bar = lh[i], i
            elif ext - ll[i] >= thr:
                lp.append(ext); cd.append(dt[i]); dirs.append(+1); cb.append(i)
                cur_dir, ext, ext_bar = -1, ll[i], i
        else:
            if ll[i] < ext:
                ext, ext_bar = ll[i], i
            elif lh[i] - ext >= thr:
                lp.append(ext); cd.append(dt[i]); dirs.append(-1); cb.append(i)
                cur_dir, ext, ext_bar = 1, lh[i], i
    return (np.array(lp), np.array(cd),
            np.array(dirs, dtype=np.int8), np.array(cb))


# ── признаки ─────────────────────────────────────────────────────────────────

def compute_features(lp_big, cd_big, dir_big, cb_big, lh, ll, dt):
    """
    Для каждого пивота step вычисляем каузальные признаки.
    Возвращает DataFrame с колонками признаков, индексированный по step.
    """
    n = len(lp_big)
    records = []

    for step in range(10, n):
        # A) Амплитуды и длительности T_BIG зигзага
        amps = np.abs(np.diff(lp_big[:step + 1]))   # длины всех плечей до step
        if len(amps) < 5:
            continue
        amp_curr  = amps[-1]
        amp_prev  = amps[-2] if len(amps) >= 2 else np.nan
        amp_ma5   = amps[-5:].mean()
        amp_ma10  = amps[-10:].mean() if len(amps) >= 10 else amps.mean()
        amp_std10 = amps[-10:].std()  if len(amps) >= 10 else amps.std()
        amp_z     = (amp_curr - amp_ma10) / amp_std10 if amp_std10 > 1e-12 else 0.0
        amp_ratio = amp_curr / amp_ma5 if amp_ma5 > 1e-12 else 1.0

        # Длительность в барах: подтверждение[step] − подтверждение[step-1]
        dur_curr = int(cb_big[step] - cb_big[step - 1]) if step >= 1 else 0
        durs = np.array([cb_big[i] - cb_big[i - 1]
                         for i in range(max(1, step - 4), step + 1)], dtype=float)
        dur_ma5   = durs.mean() if len(durs) > 0 else dur_curr
        dur_ratio = dur_curr / dur_ma5 if dur_ma5 > 1e-12 else 1.0

        # B) Волатильность на баре подтверждения
        bar_now = cb_big[step]
        for win, sfx in [(10, "10"), (30, "30"), (100, "100")]:
            pass  # вычислим ниже

        def win_stat(win):
            sl = slice(max(0, bar_now - win + 1), bar_now + 1)
            h_ = lh[sl]; l_ = ll[sl]
            if len(h_) < 2:
                return np.nan, np.nan, np.nan
            hl = h_ - l_
            return hl.std(), hl.mean(), (h_.max() - l_.min())

        vol10, atr10, rng10   = win_stat(10)
        vol30, atr30, rng30   = win_stat(30)
        vol100, _,    _       = win_stat(100)
        vol_r10_30 = vol10 / vol30 if (vol30 is not None and vol30 > 1e-12) else np.nan

        # C) Контекст
        try:
            dt_parsed = datetime.strptime(cd_big[step][:16], "%Y-%m-%d %H:%M")
            hour = dt_parsed.hour
            dow  = dt_parsed.weekday()
        except Exception:
            hour, dow = np.nan, np.nan

        # Позиция цены в последних 100 барах
        sl100 = slice(max(0, bar_now - 99), bar_now + 1)
        h100  = lh[sl100]; l100 = ll[sl100]
        price_rng = h100.max() - l100.min()
        price_pos100 = (lp_big[step] - l100.min()) / price_rng if price_rng > 1e-12 else 0.5

        records.append({
            "step":         step,
            "amp_curr":     amp_curr,
            "amp_prev":     amp_prev,
            "amp_ma5":      amp_ma5,
            "amp_ma10":     amp_ma10,
            "amp_z":        amp_z,
            "amp_ratio":    amp_ratio,
            "dur_curr":     dur_curr,
            "dur_ma5":      dur_ma5,
            "dur_ratio":    dur_ratio,
            "vol_10":       vol10,
            "vol_30":       vol30,
            "vol_100":      vol100,
            "vol_r10_30":   vol_r10_30,
            "atr_10":       atr10,
            "atr_30":       atr30,
            "rng_10":       rng10,
            "rng_30":       rng30,
            "hour":         hour,
            "dow":          dow,
            "price_pos100": price_pos100,
        })

    return pd.DataFrame(records)


# ── корреляции ────────────────────────────────────────────────────────────────

FEAT_COLS = [
    "amp_curr", "amp_prev", "amp_ma5", "amp_ma10",
    "amp_z", "amp_ratio",
    "dur_curr", "dur_ma5", "dur_ratio",
    "vol_10", "vol_30", "vol_100", "vol_r10_30",
    "atr_10", "atr_30", "rng_10", "rng_30",
    "hour", "dow", "price_pos100",
]


def correlations(df, label):
    """Pearson r + Spearman ρ для каждого признака vs oracle_tf."""
    rows = []
    for col in FEAT_COLS:
        sub = df[["oracle_tf", col]].dropna()
        if len(sub) < 20:
            rows.append({"feature": col, "pearson_r": np.nan, "spearman_r": np.nan,
                         "pearson_p": np.nan, "n": len(sub)})
            continue
        pr, pp = stats.pearsonr(sub[col], sub["oracle_tf"])
        sr, sp = stats.spearmanr(sub[col], sub["oracle_tf"])
        rows.append({"feature": col,
                     "pearson_r":  float(pr), "pearson_p":  float(pp),
                     "spearman_r": float(sr), "spearman_p": float(sp),
                     "n": len(sub)})
    out = pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)
    print(f"\n  [{label}]  n={df['oracle_tf'].notna().sum()}")
    print(out[["feature", "pearson_r", "spearman_r", "n"]].to_string(index=False))
    return out


# ── графики ───────────────────────────────────────────────────────────────────

def plot_heatmap(corr_up, corr_dn):
    feats = FEAT_COLS
    pr_up = corr_up.set_index("feature").reindex(feats)["pearson_r"].values
    sr_up = corr_up.set_index("feature").reindex(feats)["spearman_r"].values
    pr_dn = corr_dn.set_index("feature").reindex(feats)["pearson_r"].values
    sr_dn = corr_dn.set_index("feature").reindex(feats)["spearman_r"].values

    data = np.array([pr_up, sr_up, pr_dn, sr_dn])  # (4, n_feats)
    ylabels = ["Pearson r  (up)", "Spearman ρ (up)",
               "Pearson r  (down)", "Spearman ρ (down)"]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-0.35, vmax=0.35)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(ylabels, fontsize=9)

    for i in range(4):
        for j in range(len(feats)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < 0.20 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Корреляции признаков с oracle T_frac  —  up vs down  (SBER 10m, T_big=4%)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "direction_features_heatmap_10m.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → direction_features_heatmap_10m.png")


def plot_top_scatter(merged, corr_up, corr_dn, top_n=3):
    """Scatter top-N признаков (по |r|) для up и down."""
    top_up = corr_up.dropna(subset=["pearson_r"]) \
                    .reindex(corr_up["pearson_r"].abs().sort_values(ascending=False).index) \
                    .head(top_n)["feature"].tolist()
    top_dn = corr_dn.dropna(subset=["pearson_r"]) \
                    .reindex(corr_dn["pearson_r"].abs().sort_values(ascending=False).index) \
                    .head(top_n)["feature"].tolist()
    top_feats = list(dict.fromkeys(top_up + top_dn))[:top_n * 2]

    up_df   = merged[merged["direction"] == -1]
    down_df = merged[merged["direction"] == +1]

    ncols = len(top_feats)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7), sharey=False)

    for col_i, feat in enumerate(top_feats):
        for row_i, (sub, label, color) in enumerate([
            (up_df,   "up",   "steelblue"),
            (down_df, "down", "tomato"),
        ]):
            ax  = axes[row_i, col_i]
            sub = sub[["oracle_tf", feat]].dropna()
            r_u = corr_up[corr_up["feature"] == feat]["pearson_r"].values
            r_d = corr_dn[corr_dn["feature"] == feat]["pearson_r"].values
            r   = r_u[0] if row_i == 0 and len(r_u) else (r_d[0] if len(r_d) else np.nan)
            ax.scatter(sub[feat], sub["oracle_tf"] * 100,
                       alpha=0.3, s=10, color=color)
            ax.set_xlabel(feat, fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f"Oracle T_frac (%)  [{label}]", fontsize=9)
            ax.set_title(f"r = {r:.3f}" if not np.isnan(r) else "", fontsize=9)
            ax.grid(alpha=0.25)

    fig.suptitle("Top признаки vs oracle T_frac  (up / down)  —  SBER 10m", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "direction_features_scatter_10m.png", dpi=150)
    plt.close(fig)
    print("  → direction_features_scatter_10m.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    oracle_df = pd.read_csv(ORACLE_CSV)
    print(f"Oracle загружен: {len(oracle_df)} строк")

    lh, ll, dt, _ = load_candles(DATA / "10m.json")
    lp_big, cd_big, dir_big, cb_big = build_zigzag(lh, ll, dt, T_BIG)
    print(f"T_BIG={T_BIG*100:.0f}%  пивотов: {len(lp_big)}\n")

    print("Вычисляем признаки...")
    feat_df = compute_features(lp_big, cd_big, dir_big, cb_big, lh, ll, dt)

    # Объединяем с oracle
    merged = oracle_df.merge(feat_df, on="step", how="inner")
    merged.to_csv(RESULTS / "direction_features_10m.csv", index=False)
    print(f"Объединено: {len(merged)} строк\n")

    up_df   = merged[merged["direction"] == -1]
    down_df = merged[merged["direction"] == +1]

    print("Корреляции:")
    corr_up = correlations(up_df,   "UP   (LOW pivot → прогноз роста)")
    corr_dn = correlations(down_df, "DOWN (HIGH pivot → прогноз падения)")

    corr_up.to_csv(RESULTS / "direction_corr_up_10m.csv",   index=False)
    corr_dn.to_csv(RESULTS / "direction_corr_down_10m.csv", index=False)

    print("\nГрафики...")
    plot_heatmap(corr_up, corr_dn)
    plot_top_scatter(merged, corr_up, corr_dn, top_n=4)
    print("Готово.")


if __name__ == "__main__":
    main()
