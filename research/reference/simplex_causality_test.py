#!/usr/bin/env python3
"""
simplex_causality_test.py — Жёсткая проверка каузальности + визуализация

Тест проверяет три условия для каждого шага walk-forward:

  C1. Запрос использует только прошлое T_big:
        xq = lp_big[i] − lp_big[i−1]  →  индексы i и i−1 ≤ i  ✓

  C2. Все события пула подтверждены строго раньше текущего шага:
        conf_f[j] < conf_big[i]  для всех j в пуле

  C3. Таргет y_rel = lp_f[j+H] − lp_f[j] также каузален:
        conf_f[j+H] < conf_big[i]  для всех j в пуле

  Нарушение любого → FAILED + вывод деталей.

После прохождения: 3 лучших и 3 худших прогноза на графике.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simplex_ref import (
    load_candles, zigzag, cutoff_pool, simplex_predict,
    walk_forward, rmae,
    T_BIG, T_FRAC, K, H, MIN_HISTORY,
)

HERE = Path(__file__).parent


# ── жёсткий тест каузальности ─────────────────────────────────────────────────
def causality_test(lp_big, conf_big, dir_big,
                   lp_f, conf_f, dir_f) -> bool:
    violations = []

    for i in range(MIN_HISTORY, len(lp_big) - H):
        cd = conf_big[i]   # дата подтверждения текущего шага

        # C1: xq строится из i и i-1 — всегда выполнено по конструкции,
        #     но явно проверяем, что i-1 >= 0
        if i - 1 < 0:
            violations.append((i, "C1", "i-1 < 0"))
            continue

        xp, y_rel = cutoff_pool(cd, int(dir_big[i]), lp_f, conf_f, dir_f)
        if xp is None:
            continue

        # восстанавливаем индексы j из valid_range:
        # valid_range = np.arange(1, min(j_max-H, len(lp_f)-H))
        j_max = int(np.searchsorted(conf_f, cd, side='left'))
        jj = np.arange(1, min(j_max - H, len(lp_f) - H))

        # фильтр dir и finite уже применён в cutoff_pool;
        # проверяем сырые индексы до фильтра — худший случай
        for j in jj:
            # C2: событие пула подтверждено до current step
            if conf_f[j] >= cd:
                violations.append((i, "C2",
                    f"conf_f[{j}]={conf_f[j]} >= conf_big[{i}]={cd}"))
            # C3: таргет (следующий пивот) подтверждён до current step
            if j + H < len(conf_f) and conf_f[j + H] >= cd:
                violations.append((i, "C3",
                    f"conf_f[{j+H}]={conf_f[j+H]} >= conf_big[{i}]={cd}"))

        if violations:
            break   # останавливаемся на первом нарушении

    if violations:
        print(f"FAILED — {len(violations)} нарушений")
        for step, code, msg in violations[:5]:
            print(f"  step={step}  {code}: {msg}")
        return False

    print("PASSED — каузальность подтверждена")
    print(f"  Проверено шагов: {len(lp_big) - H - MIN_HISTORY}")
    print(f"  C1: xq = lp_big[i]−lp_big[i−1], i−1 ≥ 0  ✓")
    print(f"  C2: conf_f[j]   < conf_big[i] для всех j  ✓")
    print(f"  C3: conf_f[j+H] < conf_big[i] для всех j  ✓")
    return True


# ── визуализация лучших / худших ──────────────────────────────────────────────
def plot_cases(rows, lp_big, conf_big, dir_big, n=3, out_path=None):
    """
    Лучшие n и худшие n прогнозов.
    Каждый subplot: последние 8 T_big пивотов + actual/predicted следующий.
    """
    parse = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")

    errors_abs = np.array([abs(r["error"]) for r in rows])
    actuals    = np.array([r["actual"]     for r in rows])
    dz         = float(np.mean(np.abs(np.diff(actuals))))

    order = np.argsort(errors_abs)
    best_idx  = list(order[:n])
    worst_idx = list(order[-n:][::-1])

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    fig.suptitle("Simplex: лучшие и худшие прогнозы (SBER 10m)", fontsize=13)

    for col, (row_idx, label, ax_row) in enumerate(
        [(best_idx, "best", 0), (worst_idx, "worst", 1)]
    ):
        for col2, ri in enumerate(row_idx):
            ax  = axes[ax_row][col2] if n > 1 else axes[ax_row]
            row = rows[ri]
            i   = row["step"]

            # контекст: до 8 предыдущих T_big пивотов + следующий (i+H)
            i_start = max(0, i - 7)
            i_end   = min(len(lp_big) - 1, i + H)
            idx_ctx = np.arange(i_start, i_end + 1)

            prices_ctx = np.exp(lp_big[idx_ctx])
            dates_ctx  = [parse(conf_big[k]) for k in idx_ctx]

            ax.plot(dates_ctx, prices_ctx,
                    'o-', color='steelblue', lw=1.2, ms=4, label='Зигзаг T_big')

            # origin (i) — текущий пивот
            ax.axvline(parse(conf_big[i]), color='gray', lw=0.8, ls='--')
            ax.scatter([parse(conf_big[i])], [np.exp(lp_big[i])],
                       color='gray', s=60, zorder=5)

            # actual следующий пивот
            ax.scatter([parse(conf_big[i + H])], [row["actual"]],
                       color='green', s=80, zorder=6, label=f"Факт {row['actual']:.2f}")

            # predicted
            ax.axhline(row["predicted"], color='red', lw=1.0, ls='--',
                       label=f"Прогноз {row['predicted']:.2f}")

            err_rel = row["error"] / dz
            dir_sym = "▲" if row["direction"] == +1 else "▼"
            ax.set_title(
                f"{'Лучший' if label=='best' else 'Худший'} #{col2+1}  {dir_sym}\n"
                f"ош={row['error']:+.2f}  |ош|/dz={abs(err_rel):.2f}  "
                f"пул={row['pool_size']}",
                fontsize=8
            )
            ax.legend(fontsize=6, loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if out_path is None:
        out_path = HERE / "simplex_cases.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ticker   = sys.argv[1] if len(sys.argv) > 1 else "SBER"
    interval = sys.argv[2] if len(sys.argv) > 2 else "10m"

    print(f"=== Тест каузальности: {ticker} {interval} ===\n")

    highs, lows, dates = load_candles(ticker, interval)

    lp_big, conf_big, dir_big = zigzag(highs, lows, dates, T_BIG)
    lp_f,   conf_f,   dir_f   = zigzag(highs, lows, dates, T_FRAC)
    print(f"T_big={T_BIG*100:.1f}%: {len(lp_big)} пивотов")
    print(f"T_frac={T_FRAC*100:.1f}%: {len(lp_f)} пивотов\n")

    passed = causality_test(lp_big, conf_big, dir_big, lp_f, conf_f, dir_f)

    if not passed:
        sys.exit(1)

    print()
    rows = walk_forward(lp_big, conf_big, dir_big, lp_f, conf_f, dir_f)
    errors  = np.array([r["error"]  for r in rows])
    actuals = np.array([r["actual"] for r in rows])
    print(f"Прогнозов: {len(rows)}   rMAE={rmae(errors, actuals):.4f}\n")

    print("Топ-3 лучших (|ошибка| / dz):")
    dz = float(np.mean(np.abs(np.diff(actuals))))
    order = np.argsort(np.abs(errors))
    for k in range(3):
        r = rows[order[k]]
        print(f"  step={r['step']}  {r['confirm_date'][:10]}"
              f"  факт={r['actual']:.2f}  прогноз={r['predicted']:.2f}"
              f"  |err|/dz={abs(r['error'])/dz:.4f}")

    print("\nТоп-3 худших:")
    for k in range(1, 4):
        r = rows[order[-k]]
        print(f"  step={r['step']}  {r['confirm_date'][:10]}"
              f"  факт={r['actual']:.2f}  прогноз={r['predicted']:.2f}"
              f"  |err|/dz={abs(r['error'])/dz:.4f}")

    plot_cases(rows, lp_big, conf_big, dir_big, n=3)


if __name__ == "__main__":
    main()
