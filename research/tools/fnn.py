#!/usr/bin/env python3
"""
fnn.py — False Nearest Neighbors (FNN) utility.

Определяет минимальную размерность вложения p_min: наименьшее p,
при котором все ближайшие соседи в p-мерном пространстве остаются
соседями в (p+1)-мерном — то есть аттрактор полностью развёрнут.

═══════════════════════════════════════════════════════════════════
АЛГОРИТМ (Kennel 1992, адаптация)
═══════════════════════════════════════════════════════════════════

Вложение:
    X[i] = [z[i], log(z[i]/z[i-1]), log(z[i-1]/z[i-2]), ..., log(z[i-p+2]/z[i-p+1])]
    Столбец 0 — абсолютное значение. Столбцы 1..(p-1) — последовательные лог-доходности.
    Строки 0..p-2 заполнены NaN (нет достаточной истории).

Для каждого p = 1..p_max:
    1. Построить X_p (p-мерное) и X_{p+1} ((p+1)-мерное) вложение.
    2. Для каждой точки i найти ближайшего соседа j(i) в p-мерном пространстве.
    3. Вычислить:
         d_p  = ||X_p[i]  - X_p[j(i)]||   — расстояние в p-мерном пространстве
         d_p1 = ||X_{p+1}[i] - X_{p+1}[j(i)]||  — расстояние в (p+1)-мерном
    4. Точка i — ложный сосед (False NN), если:
         d_p1 / d_p  > R_thr   (сосед "убегает" при добавлении измерения)
         ИЛИ
         d_p1 / std_z > A_thr  (сосед изначально далеко в шумовых единицах)
    5. FNN fraction = доля точек, для которых выполнено условие.
    p_min = наименьшее p, при котором FNN fraction = 0.

Параметры по умолчанию (Kennel 1992):
    R_thr = 15.0
    A_thr = 2.0

Отличия от оригинала Kennel 1992:
    - d_p1 использует ПОЛНУЮ (p+1)-мерную L2 норму (не только новую координату).
      Это вариант «Hegger–Kantz», более устойчивый для смешанных вложений.
    - std_z = std ценового ряда (не std новой координаты).
    - Вложение гибридное: абсолютная цена + лог-доходности.

Использование как модуль:
    from research.tools.fnn import fnn_curve, p_min_fnn
    curve = fnn_curve(prices, p_max=10)
    print(p_min_fnn(curve))  # → 3 (или '>10')

Использование как CLI:
    python research/tools/fnn.py prices.npy
    python research/tools/fnn.py --demo        # тест на ряде Лоренца

═══════════════════════════════════════════════════════════════════
"""
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors

__all__ = ["build_embedding", "fnn_fraction", "fnn_curve", "p_min_fnn", "print_table"]

R_THR_DEFAULT = 15.0
A_THR_DEFAULT = 2.0


# ─── Ядро ──────────────────────────────────────────────────────────────────────

def build_embedding(z: np.ndarray, p: int) -> np.ndarray:
    """
    Строит p-мерное вложение для ценового ряда z.

    X[i] = [z[i], log(z[i]/z[i-1]), ..., log(z[i-p+2]/z[i-p+1])]

    Первые p-1 строк содержат NaN.

    Args:
        z: одномерный массив цен (все > 0).
        p: размерность вложения.

    Returns:
        X: shape (len(z), p), dtype float64.
    """
    n  = len(z)
    lz = np.log(z)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = z[i]
        for lag in range(1, p):
            X[i, lag] = lz[i - lag + 1] - lz[i - lag]
    return X


def fnn_fraction(z: np.ndarray, p: int,
                 r_thr: float = R_THR_DEFAULT,
                 a_thr: float = A_THR_DEFAULT) -> float:
    """
    Доля False Nearest Neighbors для размерности вложения p.

    Returns:
        Число в [0, 1] или np.nan при нехватке данных.
    """
    if len(z) < p + 2:
        return np.nan
    X_p  = build_embedding(z, p)
    X_p1 = build_embedding(z, p + 1)
    valid = np.where(~np.any(np.isnan(X_p1), axis=1))[0]
    if len(valid) < 5:
        return np.nan
    Xv,  Xv1 = X_p[valid], X_p1[valid]
    std_z = float(np.std(z))
    if std_z < 1e-12:
        return np.nan
    nn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(Xv)
    dists, idxs = nn.kneighbors(Xv)
    nn_idx = idxs[:, 1]
    d_p  = dists[:, 1]
    d_p1 = np.linalg.norm(Xv1 - Xv1[nn_idx], axis=1)   # полная (p+1)-мерная норма
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d_p > 1e-12, d_p1 / d_p, np.inf)
    return float(((ratio > r_thr) | (d_p1 / std_z > a_thr)).mean())


def fnn_curve(z: np.ndarray, p_max: int = 10,
              r_thr: float = R_THR_DEFAULT,
              a_thr: float = A_THR_DEFAULT) -> dict[int, float]:
    """
    FNN fraction для p = 1..p_max.

    Returns:
        {p: fraction}  (fraction = np.nan если данных недостаточно)
    """
    return {p: fnn_fraction(z, p, r_thr, a_thr) for p in range(1, p_max + 1)}


def p_min_fnn(curve: dict[int, float]) -> int | str:
    """
    Первое p, при котором FNN fraction == 0.
    Если не найдено — возвращает строку '>p_max'.
    """
    for p, v in sorted(curve.items()):
        if v is not None and not np.isnan(v) and v == 0.0:
            return p
    pmax = max(curve.keys())
    return f">{pmax}"


# ─── Вывод ─────────────────────────────────────────────────────────────────────

def print_table(curves: dict[str, dict[int, float]],
                p_max: int = 10) -> None:
    """
    Печатает таблицу FNN для нескольких рядов.

    Args:
        curves: {label: {p: fraction}}
    """
    labels = list(curves.keys())
    col_w  = max(12, max(len(l) for l in labels) + 2)
    header = f"{'p':>4}  " + "  ".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("─" * len(header))
    for p in range(1, p_max + 1):
        row = f"  {p:>2}  "
        for label in labels:
            v = curves[label].get(p, np.nan)
            if v is None or np.isnan(v):
                cell = "—"
            else:
                mark = " ←" if v == 0.0 else ""
                cell = f"{v:.3f}{mark}"
            row += f"  {cell:>{col_w}}"
        print(row)
    print()
    for label in labels:
        pm = p_min_fnn(curves[label])
        print(f"  p_min [{label}] = {pm}")


# ─── CLI / демо ────────────────────────────────────────────────────────────────

def _lorenz_prices(n: int = 5000, dt: float = 0.01,
                   sigma: float = 10.0, rho: float = 28.0,
                   beta: float = 8.0 / 3.0) -> np.ndarray:
    """Числовое интегрирование системы Лоренца; X-компонента как «цена»."""
    x, y, z = 0.1, 0.0, 0.0
    xs = []
    for _ in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dt * dx; y += dt * dy; z += dt * dz
        xs.append(x)
    xs = np.array(xs)
    # Сдвигаем в область > 0 для совместимости с вложением
    xs -= xs.min() - 1.0
    return xs


def _run_demo() -> None:
    """
    Демо: FNN на X-компоненте системы Лоренца.

    Классический результат (стандартное delay-вложение [z, z-τ, z-2τ]): p_min = 3.
    Наш результат: p_min = 2, т.к. вложение гибридное [z[i], lr₁, lr₂, ...] —
    абсолютное значение + направление движения уже дают полный 2D-срез.
    Главная проверка: FNN fraction падает до 0 (не стохастика).
    """
    print("Демо: система Лоренца (n=5000, σ=10, ρ=28, β=8/3)")
    print("FNN должен упасть до 0 (аттрактор). p_min=2 при нашем гибридном вложении.\n")
    z = _lorenz_prices()
    curve = fnn_curve(z, p_max=10)
    print_table({"Lorenz X": curve})


def _run_file(path: str) -> None:
    """Загружает массив из .npy файла и вычисляет FNN."""
    z = np.load(path).flatten().astype(np.float64)
    if np.any(z <= 0):
        # Если есть нули/отрицательные — применяем shift
        z = z - z.min() + 1.0
    print(f"Загружен: {path}  (n={len(z)})")
    curve = fnn_curve(z, p_max=10)
    print_table({path: curve})


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = sys.argv[1:]
    if not args or "--demo" in args:
        _run_demo()
        # График для демо
        z = _lorenz_prices()
        curve = fnn_curve(z, p_max=10)
        ps = sorted(curve.keys())
        vals = [curve[p] for p in ps]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ps, vals, "o-", color="steelblue", lw=2, ms=7)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("p (размерность вложения)")
        ax.set_ylabel("Доля FNN")
        ax.set_title("FNN — аттрактор Лоренца (p_min=2 при гибридном вложении)")
        ax.set_xticks(ps)
        ax.grid(alpha=0.3)
        out = "/tmp/fnn_demo.png"
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"\nГрафик → {out}")
    else:
        _run_file(args[0])
