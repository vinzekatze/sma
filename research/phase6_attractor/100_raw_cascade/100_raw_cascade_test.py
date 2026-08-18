"""
100_raw_cascade_test.py — тестовый прогноз: gridy dratio каскад + LP на пуле.

Алгоритм:
  Уровень 1 (D_1D, gridy dratio, Theiler=D_1):
    lv1_search=2 ближайших кандидата по сырому dratio

  LP на 2 кандидатах + запросе (3 LP-применения):
    att_cand = lp_smooth(dratio[i:i+D_1])   # сглаженный dratio, длина D_1
    att_query = lp_smooth(dratio[origin-D_1:origin])

  Каскад в att-пространстве [D_1//2 → ... → p_fit]:
    Пул: (k, j0) — скользящие окна по att_cand[k]
    На каждом уровне p_lvl: суффикс att_cand[k][j0 + (D_1//2 - p_lvl) : j0 + D_1//2]
    Запрос: att_query[-p_lvl:]
    acc_ang на финальном уровне

  LWR (итеративный: каскад + LWR пересчёт на каждом шаге) + LP-коррекция (n=1)
  Реконструкция: ratio[origin] + cumsum(att_hat) → price

Запуск:
  cd /home/kali/workspace/apps/sma
  source /home/kali/.venvs/sma/bin/activate
  python research/phase6_attractor/100_raw_cascade/100_raw_cascade_test.py
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

ROOT   = Path(__file__).resolve().parents[3]
DATA   = ROOT / "data" / "candles"
FIGDIR = Path(__file__).resolve().parent / "figures"

# ── Параметры ──────────────────────────────────────────────────────────────────
TICKER   = "SBER"
INTERVAL = "1d"

P_FIT      = 16
D_1        = P_FIT * 8             # 128 — первый уровень (gridy dratio)
LV1_SEARCH = 2                      # кандидатов на уровне 1
XI_LWR     = 3 * (P_FIT + 1) + 5   # = 56
THEILER    = D_1                    # исключаем соседей ближе D_1 баров

LP_M, LP_D, LP_K, LP_N = 9, 3, 30, 3   # LP параметры (шаги 1→att и коррекция)
LP_CORR_N  = 1                           # итерации LP-коррекции
ACC_LAMBDA = 0.01

HORIZON  = 20


# ── Данные ────────────────────────────────────────────────────────────────────

def load_close(ticker: str, interval: str) -> np.ndarray:
    p = DATA / ticker / f"{interval}.json"
    return np.array([c["close"] for c in json.loads(p.read_text())], dtype=float)


def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=float)
    cn = np.arange(1, n + 1, dtype=float)
    ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    den = cn * ct2 - ct ** 2
    b = np.where(den > 0, (cn * cty - ct * cy) / den, 0.)
    a = (cy - b * ct) / cn
    tr = np.exp(a + b * t); tr[:2] = close[:2]
    return tr


# ── LP-сглаживатель ───────────────────────────────────────────────────────────

def lp_smooth(x: np.ndarray, m: int, d: int, k: int, n_iter: int) -> np.ndarray:
    """
    Local Projective noise reduction.
    Применяем к dratio → возвращаем сглаженный dratio (att).
    Без np.diff: вход и выход одной длины.
    """
    s = x.copy().astype(float)
    N = len(s)
    for _ in range(n_iter):
        M = N - m + 1
        if M < 2:
            break
        X     = np.lib.stride_tricks.sliding_window_view(s, m).copy()
        k_eff = min(k, M - 1)
        d_eff = min(d, k_eff - 1)
        D     = cdist(X, X)
        np.fill_diagonal(D, np.inf)
        nn_idx = np.argpartition(D, k_eff, axis=1)[:, :k_eff]
        del D
        nbrs    = X[nn_idx]
        centers = nbrs.mean(axis=1, keepdims=True)
        nbrs_c  = nbrs - centers
        C       = np.einsum("bki,bkj->bij", nbrs_c, nbrs_c)
        _, vecs = np.linalg.eigh(C)
        Vd      = vecs[:, :, -d_eff:]
        xc      = X - centers[:, 0, :]
        coef    = np.einsum("bm,bmd->bd", xc, Vd)
        proj    = np.einsum("bd,bmd->bm", coef, Vd)
        Xp      = centers[:, 0, :] + proj
        res = np.zeros(N); cnt = np.zeros(N, int)
        idx2d = np.arange(M)[:, None] + np.arange(m)[None, :]
        np.add.at(res, idx2d, Xp)
        np.add.at(cnt, idx2d, 1)
        s = res / np.maximum(cnt, 1)
    return s


# ── LP-коррекция точки прогноза ──────────────────────────────────────────────

def lp_corr_point(v: np.ndarray, X_lib: np.ndarray,
                  k: int, d: int, n_iter: int) -> float:
    """Проецирует m-мерный вектор v на локальную d-мерную плоскость аттрактора."""
    v = v.copy().astype(float)
    for _ in range(n_iter):
        dists = np.linalg.norm(X_lib - v, axis=1)
        k_eff = min(k, len(X_lib) - 1)
        idx   = np.argpartition(dists, k_eff)[:k_eff]
        X_nn  = X_lib[idx]
        cen   = X_nn.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_nn - cen, full_matrices=False)
        Vd = Vt[:min(d, len(Vt))].T
        v  = cen + Vd @ (Vd.T @ (v - cen))
    return float(v[-1])


# ── Вспомогательные ──────────────────────────────────────────────────────────

def cosine_dist(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    nA = np.linalg.norm(A, axis=1); nb = float(np.linalg.norm(b))
    if nb < 1e-12:
        return np.ones(len(A))
    cos = np.where(nA > 1e-12, (A @ b) / (nA * nb), 0.)
    return 1. - np.clip(cos, -1., 1.)


def cascade_levels(d1: int, p_fit: int) -> list[int]:
    """Уровни att-каскада: [D_1//2, ..., p_fit] по ×2 убывание."""
    levs = [p_fit]; p = p_fit
    while p * 2 <= d1 // 2:
        p *= 2; levs.append(p)
    return list(reversed(levs))


# ── Основной прогноз ─────────────────────────────────────────────────────────

def forecast(
    dratio: np.ndarray,
    ratio:  np.ndarray,
    logtrend: np.ndarray,
    origin: int,
    p_fit:      int   = P_FIT,
    d1:         int   = D_1,
    lv1_search: int   = LV1_SEARCH,
    xi_lwr:     int   = XI_LWR,
    theiler:    int   = THEILER,
    lp_m:  int = LP_M, lp_d: int = LP_D,
    lp_k:  int = LP_K, lp_n: int = LP_N,
    lp_corr_n:  int   = LP_CORR_N,
    acc_lambda: float = ACC_LAMBDA,
    horizon:    int   = HORIZON,
) -> np.ndarray:

    # ── 1. Поиск кандидатов в gridy dratio ───────────────────────────────────
    n_raw = origin - d1          # число возможных стартовых позиций
    if n_raw < lv1_search:
        return np.zeros(horizon)

    # Матрица d1-мерных задержанных векторов dratio
    idx_raw = np.arange(n_raw)[:, None] + np.arange(d1)[None, :]
    X_raw   = dratio[idx_raw]                        # (n_raw, d1)
    q_raw   = dratio[origin - d1 : origin].copy()   # (d1,) — запрос

    # Theiler: исключаем кандидатов ближе THEILER баров к origin
    last  = n_raw - 1
    valid = np.where(np.abs(np.arange(n_raw) - last) >= theiler)[0]
    if len(valid) < lv1_search:
        return np.zeros(horizon)

    d_raw = np.linalg.norm(X_raw[valid] - q_raw, axis=1)
    top   = np.argpartition(d_raw, lv1_search - 1)[:lv1_search]
    cand_starts = valid[top]   # стартовые индексы в dratio

    # ── 2. LP на кандидатах и запросе ────────────────────────────────────────
    att_cands = [lp_smooth(dratio[c : c + d1], lp_m, lp_d, lp_k, lp_n)
                 for c in cand_starts]              # lv1_search × (d1,)
    att_q = lp_smooth(dratio[origin - d1 : origin],
                      lp_m, lp_d, lp_k, lp_n)      # (d1,)

    # ── 3. Каскад в att-пространстве ─────────────────────────────────────────
    levels    = cascade_levels(d1, p_fit)   # e.g. [64, 32, 16]
    first_lvl = levels[0]                   # = d1 // 2

    # Начальный пул: (k, j0) — кандидат k, старт скользящего окна j0
    # j0 ∈ [0, d1 - first_lvl - 1]: att_cand[k][j0 : j0+first_lvl] + y = att_cand[k][j0+first_lvl]
    pool = [(k, j0)
            for k in range(len(att_cands))
            for j0 in range(d1 - first_lvl)
            if j0 + first_lvl < d1]
    # Пул: lv1_search × (d1 - first_lvl) = 2 × 64 = 128 кандидатов при d1=128

    if len(pool) < p_fit + 2:
        return np.zeros(horizon)

    active = list(range(len(pool)))

    for lvl_idx, p_lvl in enumerate(levels):
        # Суффикс -p_lvl от first_lvl-окна: att_cand[k][j0 + (first_lvl - p_lvl) : j0 + first_lvl]
        offset  = first_lvl - p_lvl
        is_last = (lvl_idx == len(levels) - 1)

        V   = np.array([att_cands[pool[a][0]][pool[a][1] + offset :
                                               pool[a][1] + offset + p_lvl]
                         for a in active])     # (|active|, p_lvl)
        q_p = att_q[-p_lvl:]                  # запрос: последние p_lvl att_q

        xi_here = min(xi_lwr, len(active))
        if len(active) <= xi_here:
            continue   # пул уже достаточно мал

        if is_last and acc_lambda > 0.:
            # acc: второй diff att-вектора
            acc_c = np.zeros_like(V)
            if p_lvl >= 3:
                acc_c[:, 2:] = V[:, 2:] - 2 * V[:, 1:-1] + V[:, :-2]
            q_acc = np.zeros(p_lvl)
            if p_lvl >= 3:
                q_acc[2:] = q_p[2:] - 2 * q_p[1:-1] + q_p[:-2]
            score = (np.linalg.norm(V - q_p, axis=1)
                     + acc_lambda * cosine_dist(acc_c, q_acc))
        else:
            score = np.linalg.norm(V - q_p, axis=1)

        top    = np.argpartition(score, xi_here - 1)[:xi_here]
        active = [active[t] for t in top]

    # ── 4. LP-коррекция: библиотека lp_m-мерных окон (строится один раз) ─────
    offset_fit = first_lvl - p_fit
    X_lib_lp = np.array([att_cands[k][j : j + lp_m]
                          for k in range(len(att_cands))
                          for j in range(d1 - lp_m + 1)])  # (n_lib, lp_m)

    # ── 5. Итеративный LWR (каскад + LWR пересчёт на каждом шаге) ──────────
    # att_cands фиксированы; каскад и LWR пересчитываются по обновлённому q_cur

    buf = np.empty(d1 + horizon)
    buf[:d1] = att_q
    out = np.empty(horizon)

    for h in range(horizon):
        t     = d1 + h - 1
        q_cur = buf[t - d1 + 1 : t + 1]   # (d1,) — текущий att-контекст

        # Каскад по q_cur
        active_h = list(range(len(pool)))
        for lvl_idx, p_lvl in enumerate(levels):
            offset_lvl = first_lvl - p_lvl
            is_last    = (lvl_idx == len(levels) - 1)
            V   = np.array([att_cands[pool[ai][0]][pool[ai][1] + offset_lvl :
                                                    pool[ai][1] + offset_lvl + p_lvl]
                             for ai in active_h])
            q_p     = q_cur[-p_lvl:]
            xi_here = min(xi_lwr, len(active_h))
            if len(active_h) <= xi_here:
                continue
            if is_last and acc_lambda > 0.:
                acc_c = np.zeros_like(V)
                if p_lvl >= 3:
                    acc_c[:, 2:] = V[:, 2:] - 2 * V[:, 1:-1] + V[:, :-2]
                q_acc = np.zeros(p_lvl)
                if p_lvl >= 3:
                    q_acc[2:] = q_p[2:] - 2 * q_p[1:-1] + q_p[:-2]
                score = (np.linalg.norm(V - q_p, axis=1)
                         + acc_lambda * cosine_dist(acc_c, q_acc))
            else:
                score = np.linalg.norm(V - q_p, axis=1)
            top      = np.argpartition(score, xi_here - 1)[:xi_here]
            active_h = [active_h[ai] for ai in top]

        # X_nn, y_nn из active_h (p_fit-мерные суффиксы → следующий att)
        rows = []
        for a in active_h:
            k, j0 = pool[a]
            y_idx  = j0 + first_lvl
            if y_idx < d1:
                x_vec = att_cands[k][j0 + offset_fit : j0 + offset_fit + p_fit]
                rows.append((x_vec, att_cands[k][y_idx]))

        if len(rows) < p_fit + 2:
            val = 0.0
        else:
            X_nn_h  = np.array([r[0] for r in rows])
            y_nn_h  = np.array([r[1] for r in rows])
            q_fit   = q_cur[-p_fit:]
            d_fit   = np.linalg.norm(X_nn_h - q_fit, axis=1)
            h_bw    = max(float(d_fit.max()), 1e-10)
            w       = np.exp(-0.5 * (d_fit / h_bw) ** 2)
            sw      = np.sqrt(w)
            A_lwr   = np.hstack([np.ones((len(X_nn_h), 1)), X_nn_h])
            c_h, *_ = np.linalg.lstsq(sw[:, None] * A_lwr, sw * y_nn_h, rcond=None)
            val = float(c_h[0] + q_fit @ c_h[1:])

        # LP-коррекция
        if lp_corr_n > 0 and len(X_lib_lp) > 0:
            ctx  = buf[max(0, t - lp_m + 2) : t + 1]
            if len(ctx) < lp_m - 1:
                ctx = np.pad(ctx, (lp_m - 1 - len(ctx), 0))
            v_lp = np.concatenate([ctx[-(lp_m - 1):], [val]])
            val  = lp_corr_point(v_lp, X_lib_lp, k=lp_k, d=lp_d, n_iter=lp_corr_n)

        out[h]     = val
        buf[t + 1] = val

    # ── 6. Реконструкция цены ────────────────────────────────────────────────
    ratio_hat = ratio[origin] + np.cumsum(out)
    price_hat = ratio_hat * logtrend[origin]
    return price_hat


# ── Старый алгоритм: глобальный LP ───────────────────────────────────────────

def forecast_global_lp(
    ratio:    np.ndarray,
    logtrend: np.ndarray,
    origin:   int,
    p_fit:    int   = P_FIT,
    p_max:    int   = D_1,          # = D_1, для честного сравнения уровней
    xi_lwr:   int   = XI_LWR,
    lp_m:  int = LP_M, lp_d: int = LP_D,
    lp_k:  int = LP_K, lp_n: int = LP_N,
    lp_corr_n:  int   = LP_CORR_N,
    acc_lambda: float = ACC_LAMBDA,
    horizon:    int   = HORIZON,
) -> np.ndarray:
    """Старый алгоритм: LP на ratio[:origin+1] → att=diff → p-aligned каскад → LWR (фиксированный)."""
    # 1. Глобальный LP на ratio
    s   = lp_smooth(ratio[:origin + 1], lp_m, lp_d, lp_k, lp_n)
    att = np.diff(s)           # att[i] = s[i+1]-s[i], длина = origin

    # 2. P-aligned levels
    levs = [p_fit]; p = p_fit
    while p * 2 <= p_max:
        p *= 2; levs.append(p)
    levels    = list(reversed(levs))   # e.g. [128, 64, 32, 16]
    first_lvl = levels[0]

    n_att = len(att)
    if n_att < first_lvl + 1:
        return np.zeros(horizon)

    # 3. Каскадная матрица: X_full[i] = att[i:i+first_lvl], y = att[i+first_lvl]
    n_casc  = n_att - first_lvl
    X_full  = np.lib.stride_tricks.sliding_window_view(att, first_lvl).copy()[:n_casc]
    q_full  = att[-first_lvl:]       # запрос — последние first_lvl att

    # 4. Каскад
    active = list(range(n_casc))
    for lvl_idx, p_lvl in enumerate(levels):
        offset  = first_lvl - p_lvl
        is_last = (lvl_idx == len(levels) - 1)
        V   = X_full[active][:, offset : offset + p_lvl]
        q_p = q_full[-p_lvl:]
        xi_here = min(xi_lwr, len(active))
        if len(active) <= xi_here:
            continue
        if is_last and acc_lambda > 0.:
            acc_c = np.zeros_like(V)
            if p_lvl >= 3:
                acc_c[:, 2:] = V[:, 2:] - 2 * V[:, 1:-1] + V[:, :-2]
            q_acc = np.zeros(p_lvl)
            if p_lvl >= 3:
                q_acc[2:] = q_p[2:] - 2 * q_p[1:-1] + q_p[:-2]
            score = (np.linalg.norm(V - q_p, axis=1)
                     + acc_lambda * cosine_dist(acc_c, q_acc))
        else:
            score = np.linalg.norm(V - q_p, axis=1)
        top    = np.argpartition(score, xi_here - 1)[:xi_here]
        active = [active[ai] for ai in top]

    # 5. LWR (фиксированный, как в app3)
    offset_fit = first_lvl - p_fit
    X_nn = X_full[active][:, offset_fit : offset_fit + p_fit]
    y_nn = np.array([att[a + first_lvl] for a in active])
    q_fit = q_full[-p_fit:]
    d_fit = np.linalg.norm(X_nn - q_fit, axis=1)
    h_bw  = max(float(d_fit.max()), 1e-10)
    w     = np.exp(-0.5 * (d_fit / h_bw) ** 2)
    sw    = np.sqrt(w)
    A_lwr = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    c0, *_ = np.linalg.lstsq(sw[:, None] * A_lwr, sw * y_nn, rcond=None)

    # LP-коррекция: m-мерные окна из att
    X_lib_lp = np.lib.stride_tricks.sliding_window_view(att, lp_m).copy()

    # 6. Итеративный прогноз
    buf = np.empty(first_lvl + horizon)
    buf[:first_lvl] = q_full
    out = np.empty(horizon)

    for h in range(horizon):
        t     = first_lvl + h - 1
        q_cur = buf[t - p_fit + 1 : t + 1]
        val   = float(c0[0] + q_cur @ c0[1:])
        if lp_corr_n > 0 and len(X_lib_lp) > 0:
            ctx  = buf[max(0, t - lp_m + 2) : t + 1]
            if len(ctx) < lp_m - 1:
                ctx = np.pad(ctx, (lp_m - 1 - len(ctx), 0))
            v_lp = np.concatenate([ctx[-(lp_m - 1):], [val]])
            val  = lp_corr_point(v_lp, X_lib_lp, k=lp_k, d=lp_d, n_iter=lp_corr_n)
        out[h]     = val
        buf[t + 1] = val

    # 7. Реконструкция
    ratio_hat = ratio[origin] + np.cumsum(out)
    price_hat = ratio_hat * logtrend[origin]
    return price_hat


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGDIR.mkdir(exist_ok=True)

    close    = load_close(TICKER, INTERVAL)
    lt       = logtrend_causal(close)
    ratio    = close / np.maximum(lt, 1e-10)
    dratio   = np.diff(ratio)
    # dratio[i] = ratio[i+1] - ratio[i]; длина N-1
    # origin в терминах ratio → в dratio используем dratio[:origin]

    origin = len(close) - 1 - HORIZON   # резервируем горизонт для сравнения

    print(f"Тикер:     {TICKER} {INTERVAL}")
    print(f"Origin:    {origin}  (из {len(close)} свечей)")
    print(f"D_1={D_1}, P_FIT={P_FIT}, lv1_search={LV1_SEARCH}, xi_lwr={XI_LWR}")
    print(f"Theiler={THEILER}, LP=({LP_M},{LP_D},{LP_K},{LP_N}), corr_n={LP_CORR_N}")
    print(f"Уровни каскада: {cascade_levels(D_1, P_FIT)}")
    print(f"Нач. пул: {LV1_SEARCH} × {D_1 - D_1//2} = {LV1_SEARCH*(D_1-D_1//2)} кандидатов")
    print()

    actual = close[origin + 1 : origin + 1 + HORIZON]

    print("─── Новый алгоритм (gridy dratio + LP на 3 окнах, итеративный LWR) ───")
    price_new = forecast(dratio, ratio, lt, origin)
    mape_new  = float(np.mean(np.abs(price_new - actual) / actual)) * 100
    print(f"MAPE: {mape_new:.2f}%")
    print(f"Прогноз: {price_new[:5].round(2)}")
    print(f"Факт:    {actual[:5].round(2)}")

    print()
    print("─── Старый алгоритм (глобальный LP на ratio, фиксированный LWR) ───────")
    price_old = forecast_global_lp(ratio, lt, origin)
    mape_old  = float(np.mean(np.abs(price_old - actual) / actual)) * 100
    print(f"MAPE: {mape_old:.2f}%")
    print(f"Прогноз: {price_old[:5].round(2)}")
    print(f"Факт:    {actual[:5].round(2)}")

    # График
    n_hist = 80
    fig, ax = plt.subplots(figsize=(13, 5))
    x_hist = range(-n_hist, 1)
    x_fc   = range(1, HORIZON + 1)
    ax.plot(x_hist, close[origin - n_hist : origin + 1], 'k-', lw=1.2, label='история')
    ax.plot(x_fc,   actual,    'b-',  lw=1.5, label='факт')
    ax.plot(x_fc,   price_new, 'r--', lw=1.5, label=f'новый MAPE={mape_new:.1f}%')
    ax.plot(x_fc,   price_old, 'g--', lw=1.5, label=f'старый MAPE={mape_old:.1f}%')
    ax.axvline(0, color='gray', ls=':', lw=1)
    ax.set_title(f"{TICKER} — сравнение  |  p_fit={P_FIT}  xi={XI_LWR}  LP=({LP_M},{LP_D},{LP_K},{LP_N})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()

    out_fig = FIGDIR / f"{TICKER}_raw_cascade_test.png"
    plt.savefig(out_fig, dpi=120)
    print(f"\nГрафик: {out_fig}")
