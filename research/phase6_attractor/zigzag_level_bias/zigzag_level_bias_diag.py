"""
Диагностика систематической переоценки уровня в зигзаг-прогнозе.

Три гипотезы:
  H1 — bias коррелирует с текущим уровнем z_current (logtrend/level drift)
  H2 — bias коррелирует с локальной волатильностью зигзага (режим волатильности)
  H3 — bias асимметричен по направлению прогноза (вверх vs вниз)

Три фикса:
  Fix-A — предсказывать Dz = z[j+h] - z[j], затем pred = z[step] + pred_D
  Fix-B — убрать z[i] из признаков (только дельты), предсказывать уровень
  Fix-C — предсказывать z[j+h]/z[j], затем pred = z[step] x pred_ratio

Базовые параметры: theta=8, p=2, T=2%, TRAIN_WIN=200, h=1.
SBER: пул 1d+1h+10m (как лучший результат). LKOH: пул 1d.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

THRESH     = 0.02
TRAIN_WIN  = 200
THETA      = 8.0
P          = 2
H          = 1
RECENT_K   = 20

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


def logtrend_causal(close):
    n  = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    N  = np.arange(1, n + 1, dtype=np.float64)
    St, Sp   = np.cumsum(t), np.cumsum(lc)
    St2, Stp = np.cumsum(t ** 2), np.cumsum(t * lc)
    den = N * St2 - St ** 2
    b   = np.where(den > 1e-12, (N * Stp - St * Sp) / den, 0.0)
    a   = (Sp - b * St) / N
    ratio = close / np.exp(a + b * t)
    ratio[:2] = 1.0
    return ratio


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
    return pivots


def load_zigzag(path, thr=THRESH):
    with open(path) as f:
        data = json.load(f)
    close = np.array([d["close"] for d in data], dtype=np.float64)
    dates = np.array([d["begin"] for d in data])
    ratio = logtrend_causal(close)
    piv   = find_pivots(ratio, thr)
    return ratio[np.array(piv)], dates[np.array(piv)]


def make_X_level(z):
    """Признаки с уровнем: [z[i], z[i]-z[i-1]]  (dim=P=2)."""
    n = len(z)
    X = np.full((n, P), np.nan)
    for i in range(P, n):
        X[i, 0] = z[i]
        for k in range(P - 1):
            X[i, k + 1] = z[i - k] - z[i - k - 1]
    return X


def make_X_delta(z):
    """Признаки только дельты: [z[i]-z[i-1]]  (dim=1)."""
    n   = len(z)
    dim = max(1, P - 1)
    X   = np.full((n, dim), np.nan)
    for i in range(P, n):
        for k in range(dim):
            X[i, k] = z[i - k] - z[i - k - 1]
    return X


def compute_y(z, j, h, target_type):
    if j + h >= len(z):
        return np.nan
    yf, yc = z[j + h], z[j]
    if target_type == "level":
        return yf
    if target_type == "delta":
        return yf - yc
    if target_type == "ratio_y":
        return yf / yc if abs(yc) > 1e-12 else np.nan
    return np.nan


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


def build_pool(step, h, ts, X_prim, z_prim, dates_prim, aux_list, target_type):
    """1d-окно [ts+P .. step-h] + aux до текущей даты."""
    X_rows, y_rows = [], []
    for j in range(ts + P, step - h + 1):
        if np.any(np.isnan(X_prim[j])):
            continue
        yj = compute_y(z_prim, j, h, target_type)
        if not np.isnan(yj):
            X_rows.append(X_prim[j])
            y_rows.append(yj)
    cur_date = dates_prim[step]
    for z_a, X_a, d_a in aux_list:
        ce = int(np.searchsorted(d_a, cur_date, side="left"))
        for k in range(P, ce - h):
            if np.any(np.isnan(X_a[k])):
                continue
            yk = compute_y(z_a, k, h, target_type)
            if not np.isnan(yk):
                X_rows.append(X_a[k])
                y_rows.append(yk)
    if not X_rows:
        return None, None
    return np.array(X_rows), np.array(y_rows)


def walk_forward(ticker, cfg):
    print(f"\n{'-'*60}")
    print(f"  Тикер: {ticker}")

    z1d, d1d = load_zigzag(cfg["1d"])

    aux_lv, aux_dv = [], []
    for key in ("1h", "10m"):
        if key in cfg:
            za, da = load_zigzag(cfg[key])
            aux_lv.append((za, make_X_level(za), da))
            aux_dv.append((za, make_X_delta(za), da))

    X1d_lv = make_X_level(z1d)
    X1d_dv = make_X_delta(z1d)

    n1d        = len(z1d)
    test_start = TRAIN_WIN + P
    test_end   = n1d - H

    aug_label = "+".join(k for k in cfg if k != "1d") or "1d only"
    print(f"  пивотов 1d: {n1d}  тест: [{test_start}-{test_end}]  "
          f"n={test_end - test_start}  aug={aug_label}")

    records = []

    for step in range(test_start, test_end):
        ts     = max(0, step - TRAIN_WIN)
        z_cur  = z1d[step]
        z_prev = z1d[step - 1]
        actual = z1d[step + H]

        k0         = max(1, step - RECENT_K)
        recent_vol = np.mean(np.abs(np.diff(z1d[k0 : step + 1])))

        # direction: если z_cur < z_prev — текущий пивот лоу, следующий хай → "up"
        direction = "up" if z_cur < z_prev else "down"

        x_q_lv = X1d_lv[step]
        x_q_dv = X1d_dv[step]
        if np.any(np.isnan(x_q_lv)):
            continue

        # BASE: level features + level target
        Xp, yp = build_pool(step, H, ts, X1d_lv, z1d, d1d, aux_lv, "level")
        if Xp is None or len(Xp) < 5:
            continue
        pred_base = smap_pred(Xp, yp, x_q_lv)

        # Fix-A: level features + delta target → pred = z_cur + pred_delta
        Xpa, ypa = build_pool(step, H, ts, X1d_lv, z1d, d1d, aux_lv, "delta")
        pred_a = np.nan
        if Xpa is not None and len(Xpa) >= 5:
            pdelta = smap_pred(Xpa, ypa, x_q_lv)
            if not np.isnan(pdelta):
                pred_a = z_cur + pdelta

        # Fix-B: delta features only + level target
        Xpb, ypb = build_pool(step, H, ts, X1d_dv, z1d, d1d, aux_dv, "level")
        pred_b = np.nan
        if (not np.any(np.isnan(x_q_dv))) and Xpb is not None and len(Xpb) >= 5:
            pred_b = smap_pred(Xpb, ypb, x_q_dv)

        # Fix-C: level features + ratio target → pred = z_cur * pred_ratio
        Xpc, ypc = build_pool(step, H, ts, X1d_lv, z1d, d1d, aux_lv, "ratio_y")
        pred_c = np.nan
        if Xpc is not None and len(Xpc) >= 5:
            pry = smap_pred(Xpc, ypc, x_q_lv)
            if not np.isnan(pry):
                pred_c = z_cur * pry

        records.append({
            "step": step, "z_cur": z_cur, "actual": actual,
            "recent_vol": recent_vol, "direction": direction,
            "m0_err": z_cur - actual,
            "pred_base": pred_base, "pred_a": pred_a,
            "pred_b": pred_b,       "pred_c": pred_c,
        })

        if step % 100 == 0:
            print(f"  step {step}/{test_end}", flush=True)

    df = pd.DataFrame(records)
    for col in ["base", "a", "b", "c"]:
        df[f"serr_{col}"] = df[f"pred_{col}"] - df["actual"]
        df[f"aerr_{col}"] = df[f"serr_{col}"].abs()
    return df


def analyze(ticker, df):
    print(f"\n{'='*60}")
    print(f"  Анализ: {ticker}  (n={len(df)})")
    print(f"{'='*60}")

    mean_dz = df["actual"].diff().abs().mean()
    if mean_dz < 1e-12:
        mean_dz = 1.0

    m0_rmae = df["m0_err"].abs().mean() / mean_dz

    variants = [
        ("BASE",  "serr_base", "aerr_base"),
        ("Fix-A", "serr_a",    "aerr_a"),
        ("Fix-B", "serr_b",    "aerr_b"),
        ("Fix-C", "serr_c",    "aerr_c"),
    ]

    print(f"\n  M0 rMAE: {m0_rmae:.4f}\n")
    print(f"  {'Вариант':8}  {'bias':>10}  {'rMAE':>7}  {'vs M0':>8}")
    for label, sc, ac in variants:
        s = df[sc].dropna()
        a = df[ac].dropna()
        if len(s) == 0:
            continue
        bias = s.mean()
        rmae = a.mean() / mean_dz
        print(f"  {label:8}  {bias:>+10.5f}  {rmae:>7.4f}  {(rmae/m0_rmae-1)*100:>+7.1f}%")

    print(f"\n  -- H1: corr(signed_err, z_current) --")
    for label, sc, _ in variants:
        sub = df[[sc, "z_cur"]].dropna()
        if len(sub) < 10:
            continue
        r, p = stats.pearsonr(sub[sc], sub["z_cur"])
        sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {label:8}  r={r:+.3f}  p={p:.3f} {sig}")

    print(f"\n  -- H2: corr(signed_err, recent_vol) --")
    for label, sc, _ in variants:
        sub = df[[sc, "recent_vol"]].dropna()
        if len(sub) < 10:
            continue
        r, p = stats.pearsonr(sub[sc], sub["recent_vol"])
        sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {label:8}  r={r:+.3f}  p={p:.3f} {sig}")

    print(f"\n  -- H3: mean signed_err по направлению --")
    print(f"  {'Вариант':8}  {'bias_up':>10}  {'bias_down':>10}  {'разница':>10}")
    for label, sc, _ in variants:
        up = df.loc[df["direction"] == "up",   sc].mean()
        dn = df.loc[df["direction"] == "down", sc].mean()
        print(f"  {label:8}  {up:>+10.5f}  {dn:>+10.5f}  {up-dn:>+10.5f}")

    return m0_rmae


def plot_results(ticker, df, m0_rmae):
    mean_dz = df["actual"].diff().abs().mean()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"{ticker}: диагностика переоценки уровня  "
        f"(theta={THETA}, p={P}, T={THRESH*100:.0f}%, W={TRAIN_WIN})",
        fontsize=11)

    palette  = {"BASE": "steelblue", "Fix-A": "darkorange",
                "Fix-B": "green",    "Fix-C": "red"}
    variants = [("BASE", "serr_base"), ("Fix-A", "serr_a"),
                ("Fix-B", "serr_b"),   ("Fix-C", "serr_c")]

    # 1. signed_err BASE по времени
    ax = axes[0, 0]
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(df["step"], df["serr_base"], alpha=0.3, lw=0.5, color="steelblue")
    ax.plot(df["step"],
            df["serr_base"].rolling(20, min_periods=5).mean(),
            color="steelblue", lw=2, label="rolling mean (w=20)")
    ax.set_title("BASE: signed_err по времени")
    ax.set_xlabel("step (пивот)")
    ax.set_ylabel("pred − actual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # 2. H1: signed_err vs z_cur
    ax = axes[0, 1]
    for label, sc in variants:
        sub = df[["z_cur", sc]].dropna()
        if len(sub) < 5:
            continue
        ax.scatter(sub["z_cur"], sub[sc], alpha=0.12, s=6, color=palette[label])
        zf = np.polyfit(sub["z_cur"], sub[sc], 1)
        xs = np.linspace(sub["z_cur"].min(), sub["z_cur"].max(), 50)
        ax.plot(xs, np.polyval(zf, xs), color=palette[label], lw=2, label=label)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("H1: signed_err ~ z_current")
    ax.set_xlabel("z_current (ratio)")
    ax.set_ylabel("pred − actual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # 3. H2: signed_err vs recent_vol
    ax = axes[0, 2]
    for label, sc in variants:
        sub = df[["recent_vol", sc]].dropna()
        if len(sub) < 5:
            continue
        ax.scatter(sub["recent_vol"], sub[sc], alpha=0.12, s=6, color=palette[label])
        zf = np.polyfit(sub["recent_vol"], sub[sc], 1)
        xs = np.linspace(sub["recent_vol"].min(), sub["recent_vol"].max(), 50)
        ax.plot(xs, np.polyval(zf, xs), color=palette[label], lw=2, label=label)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("H2: signed_err ~ recent_vol")
    ax.set_xlabel("recent_vol (mean |Dz|, K=20)")
    ax.set_ylabel("pred − actual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # 4. H3: box по направлению (BASE)
    ax = axes[1, 0]
    data_h3 = [
        df.loc[df["direction"] == "up",   "serr_base"].dropna().values,
        df.loc[df["direction"] == "down", "serr_base"].dropna().values,
    ]
    bp = ax.boxplot(data_h3,
                    labels=["up\n(к хаю)", "down\n(к лоу)"],
                    patch_artist=True)
    for patch, color in zip(bp["boxes"], ["lightblue", "lightsalmon"]):
        patch.set_facecolor(color)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("H3: BASE signed_err по направлению")
    ax.set_ylabel("pred − actual")
    ax.grid(alpha=0.2)

    # 5. rMAE по вариантам
    ax = axes[1, 1]
    labels_bar = ["M0", "BASE", "Fix-A", "Fix-B", "Fix-C"]
    cols_bar   = [None, "aerr_base", "aerr_a", "aerr_b", "aerr_c"]
    colors_bar = ["gray", "steelblue", "darkorange", "green", "red"]
    rmaes = [m0_rmae] + [df[c].dropna().mean() / mean_dz for c in cols_bar[1:]]
    bars = ax.bar(labels_bar, rmaes, color=colors_bar, alpha=0.8)
    for bar, val in zip(bars, rmaes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("rMAE по вариантам")
    ax.set_ylabel("rMAE")
    ax.grid(axis="y", alpha=0.2)

    # 6. bias по вариантам
    ax = axes[1, 2]
    bias_labels = ["BASE", "Fix-A", "Fix-B", "Fix-C"]
    bias_cols   = ["serr_base", "serr_a", "serr_b", "serr_c"]
    biases      = [df[c].dropna().mean() for c in bias_cols]
    colors_bias = ["steelblue", "darkorange", "green", "red"]
    ax.bar(bias_labels, biases, color=colors_bias, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    for i, v in enumerate(biases):
        ax.text(i, v + np.sign(v) * 0.0005, f"{v:+.5f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_title("Среднее смещение (bias = mean signed_err)")
    ax.set_ylabel("mean(pred − actual)")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    out = OUT / f"{ticker}_level_bias.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  График: {out}")


def main():
    all_rows = []

    for ticker, cfg in TICKER_CFG.items():
        df      = walk_forward(ticker, cfg)
        m0_rmae = analyze(ticker, df)
        plot_results(ticker, df, m0_rmae)

        df["ticker"] = ticker
        csv_path = OUT / f"{ticker}_level_bias.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}")

        mean_dz = df["actual"].diff().abs().mean()
        for label, sc, ac in [
            ("BASE",  "serr_base", "aerr_base"),
            ("Fix-A", "serr_a",    "aerr_a"),
            ("Fix-B", "serr_b",    "aerr_b"),
            ("Fix-C", "serr_c",    "aerr_c"),
        ]:
            all_rows.append({
                "ticker":  ticker,
                "variant": label,
                "bias":    df[sc].dropna().mean(),
                "rMAE":    df[ac].dropna().mean() / mean_dz,
                "m0_rMAE": m0_rmae,
            })

    print(f"\n{'='*60}")
    print("  Кросс-тикерная сводка")
    print(f"{'='*60}")
    print(f"  {'Тикер':6}  {'Вариант':8}  {'bias':>10}  {'rMAE':>7}  {'vs M0':>8}")
    for r in all_rows:
        print(f"  {r['ticker']:6}  {r['variant']:8}  "
              f"{r['bias']:>+10.5f}  {r['rMAE']:>7.4f}  "
              f"{(r['rMAE']/r['m0_rMAE']-1)*100:>+7.1f}%")

    pd.DataFrame(all_rows).to_csv(OUT / "summary.csv", index=False)
    print(f"\n  Сводка: {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
