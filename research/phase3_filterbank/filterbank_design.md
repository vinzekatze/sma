# Filter Bank + LA: теоретическая база и план реализации

Документ фиксирует результаты исследований скриптов 30–34 и содержит
всё необходимое для реализации метода в прототипе и production.

---

## 1. Проблема: почему raw dratio плохо предсказывается LA

`dratio[i] = ratio[i+1] − ratio[i]` — это приращение нормализованной цены.
Энергетический анализ (скр. 30) показывает:

- **62% энергии** сосредоточено в компонентах с периодом ≤ 4 баров (высокочастотный шум)
- **38% энергии** — в медленных компонентах с периодом > 52 баров

При поиске соседей LA работает в полном пространстве задержек.
Быстрый шум доминирует в евклидовом расстоянии между векторами:
два вектора, совпадающих по медленной динамике, оказываются «далёкими»
из-за расхождения в быстром шуме → отобранные соседи случайны → прогноз
медленной компоненты не работает.

**Решение**: разложить dratio на частотные полосы, выполнить LA на медленных
компонентах (где шум убран), просуммировать прогнозы.

---

## 2. Структура фильтр-банка

### 2.1 Каскадные каузальные фильтры Баттерворта

```python
from scipy.signal import butter, sosfilt

FILTER_ORDER = 4
CUTOFFS = [0.25, 0.125, 0.0625, 0.03125, 0.015625]  # нормированные частоты (Найквист = 1)

def make_filter_bank(series: np.ndarray) -> np.ndarray:
    """
    Разбивает series на 6 компонент (C0..C5).
    Возвращает array shape (6, N).
    Каузален: sosfilt — однонаправленный, без look-ahead.
    """
    components = []
    remaining = series.copy()
    for fc in CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        components.append(remaining - low)   # полоса пропускания
        remaining = low
    components.append(remaining)             # последний остаток (C5)
    return np.array(components)              # (6, N)
```

**Почему sosfilt, а не sosfiltfilt:**
`sosfiltfilt` — нулевая фаза (двухпроходный), требует знания будущих значений.
`sosfilt` — однопроходный (каузальный), полностью time-correct.
Применительно к walk-forward: для вычисления `COMP[:, :vo]` нужны только
бары до `vo` включительно — соблюдается строго.

### 2.2 Карта компонент

| Компонента | Полоса (нормир.) | Период (1d-баров) | Описание |
|---|---|---|---|
| C0 | [0.25, 0.5] | 2–4 | Быстрый шум, тиковый уровень |
| C1 | [0.125, 0.25] | 4–8 | Недельные движения |
| C2 | [0.0625, 0.125] | 8–16 | Двухнедельные паттерны |
| **C3** | **[0.03125, 0.0625]** | **16–32 → ~52** | **Месячный цикл** |
| **C4** | **[0.015625, 0.03125]** | **32–64 → ~103** | **Квартальный цикл** |
| **C5** | **[0, 0.015625]** | **64+ → ~206+** | **Долгосрочный тренд dratio** |

*Период в барах вычислен как 1/cutoff. Уточнённые значения учитывают
переходную зону фильтра 4-го порядка (−3 dB на cutoff частоте).*

**C0+C1+C2**: ~62% энергии, LA на них не работает (шум).
**C3+C4+C5**: ~38% энергии, медленный предсказуемый сигнал — **наша цель**.

---

## 3. Правило выбора p для медленных компонент

**Эмпирическое правило (подтверждено скр. 32–34):**

```
p_min ≥ T_min / 4
```

где `T_min` — минимальный период самой быстрой из используемых компонент.

Для C3+C4+C5 при интервале 1d: `T_min ≈ 52 бара` (C3).

```
p_min ≥ 52 / 4 ≈ 13
```

Поэтому **p=20** — оптимальное практическое значение:
- p=5 < T_min/4 ≈ 13 → LA не различает фазы медленной волны → сбой на LKOH, VTBR, MRKP
- p=20 (≈40% от T_min) → delay-вектор фиксирует форму волны → работает на всех 8 тикерах
- p=40 не даёт улучшения vs p=20 в абсолютных значениях (0.00918 vs 0.00917)

**Для других интервалов** T_min масштабируется:

| Интервал | T_min C3 (баров) | p_min | Рекомендуемый p |
|---|---|---|---|
| 1d | ~52 | ~13 | **20** |
| 1h | ~52 | ~13 | 20 (требует проверки) |
| 10m | ~52 | ~13 | 20 (требует проверки) |

*Примечание: cutoffs фиксированы в нормированных частотах — T в барах
не меняется при смене интервала. При 1h те же компоненты описывают
~52 часа (≈2 торговых дня). Это требует отдельного исследования (скр. 35).*

---

## 4. Алгоритм прогноза (LA/LWR + filter bank)

### 4.1 Схема шагов

```
1. normalize(candles, window=MA_WINDOW) → ratio, MA
2. dratio = diff(ratio)
3. COMP = make_filter_bank(dratio)         # (6, N), каузально
4. Для каждого origin vo:
   a. C3_hist = COMP[3, :vo]
      C4_hist = COMP[4, :vo]
      C5_hist = COMP[5, :vo]
   b. Для ci в [3, 4, 5]:
        X_ci, y_ci = build_delay_matrix(COMP[ci, :vo], p=20)
        vec_ci     = last_vector(COMP[ci, :vo], p=20)
        hat_ci     = LWR(X_ci, y_ci, vec_ci, xi=63)   # shape (VAL_H,)
   c. hat_fb = hat_C3 + hat_C4 + hat_C5
   d. ratio_hat = ratio[vo] + cumsum(hat_fb)
   e. price_hat = ratio_hat * MA[vo]
```

### 4.2 Ключевые параметры (production-validated)

| Параметр | Значение | Источник |
|---|---|---|
| `FILTER_ORDER` | 4 | Butterworth order |
| `CUTOFFS` | `[0.25, 0.125, 0.0625, 0.03125, 0.015625]` | Октавные интервалы |
| Компоненты | C3, C4, C5 (индексы 3, 4, 5) | |
| `p` | **20** | Правило p ≥ T_min/4 |
| `ξ` (n_neighbors) | **63** = 3×(20+1) | Правило лекции Ξ ≥ 3(p+1) |
| `MA_WINDOW` | 1000 (для 1d) | Из текущего production |
| `VAL_H` | 10 | Горизонт валидации |

### 4.3 Почему sep (раздельно), а не lp (суммарный сигнал)

```
sep: LA(C3) + LA(C4) + LA(C5)   → каждая компонента в своём пространстве задержек
lp:  LA(C3+C4+C5)               → один LA на суммарном сигнале
```

| p | sep abs MAPE | lp abs MAPE |
|---|---|---|
| 5 | 0.00937 | 0.01002 |
| 20 | **0.00917** | 0.00995 |

sep стабильно лучше lp. Причина: C3, C4, C5 имеют разные характерные периоды
(52, 103, 206+ баров). В раздельном пространстве каждая компонента находит
«структурно похожие» соседей в своём масштабе. При суммировании сигналов
медленная компонента маскирует быструю.

---

## 5. Результаты (multi-ticker, 8 тикеров, 1d, N=200×8=1600)

### 5.1 Итоговая таблица p=20

| Тикер | baseline p=20 | fb p=20 | Δ% |
|---|---|---|---|
| CHMF | 0.02904 | 0.01880 | **−35.3%** |
| LKOH | 0.03297 | 0.03058 | **−7.3%** |
| MGNT | 0.03089 | 0.01937 | **−37.3%** |
| MRKP | 0.03314 | 0.02338 | **−29.5%** |
| NLMK | 0.02801 | 0.02205 | **−21.3%** |
| NVTK | 0.02984 | 0.02788 | **−6.6%** |
| SBER | 0.01469 | 0.00917 | **−37.6%** |
| VTBR | 0.03262 | 0.02462 | **−24.5%** |
| **AGG** | **0.02694** | **0.02101** | **−22.0%** |

**Wilcoxon (N=1600): p < 0.0001. Все 8 тикеров улучшились.**

Vs производственный baseline p=5: 0.02229 → 0.02101 = **−5.7%** в абсолюте.

---

## 6. Реализация в прототипе

### 6.1 Новый модуль: `prototype/forcaster/forecast/filterbank.py`

```python
"""
Filter bank decomposition + LWR forecast for slow components.

Usage:
    from forcaster.forecast.filterbank import (
        make_filter_bank,
        forecast_lwr_fb,
        C3_IDX,
        FB_P,
        FB_XI,
    )
"""
import numpy as np
from scipy.signal import butter, sosfilt
from .embedding import build_delay_matrix, last_vector

FILTER_ORDER = 4
CUTOFFS      = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
C3_IDX       = [3, 4, 5]   # медленные компоненты (периоды >52 баров)
FB_P         = 20
FB_XI        = 3 * (FB_P + 1)   # 63


def make_filter_bank(series: np.ndarray) -> np.ndarray:
    """
    Каузальный фильтр-банк. Возвращает (6, N) — компоненты C0..C5.
    Использует sosfilt (однопроходный), строго time-correct.
    """
    components = []
    remaining  = series.copy()
    for fc in CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        components.append(remaining - low)
        remaining = low
    components.append(remaining)
    return np.array(components)   # (6, N)


def forecast_lwr_fb(
    dratio: np.ndarray,
    origin_k: int,
    horizon: int,
    p: int = FB_P,
    n_neighbors: int = FB_XI,
    component_idx: list[int] = C3_IDX,
) -> np.ndarray:
    """
    Filter bank LWR forecast на медленных компонентах dratio.

    Args:
        dratio:        полный Δratio ряд (не обрезать до origin_k — функция сама)
        origin_k:      индекс в ratio-ряде (прогноз с origin_k+1)
        horizon:       шагов вперёд
        p:             размерность вложения (рекомендуется 20 для C3+)
        n_neighbors:   Ξ соседей (рекомендуется 3*(p+1) = 63)
        component_idx: индексы медленных компонент (по умолчанию [3,4,5])

    Returns:
        dratio_hat: shape (horizon,) — суммарный прогноз по медленным компонентам
    """
    # Вычисляем разложение каузально (только история до origin_k)
    history = dratio[:origin_k]
    COMP    = make_filter_bank(history)   # (6, origin_k)

    hats = []
    for ci in component_idx:
        comp_hist = COMP[ci]              # (origin_k,)
        X, y = build_delay_matrix(comp_hist, p)
        if len(X) < n_neighbors:
            hats.append(np.zeros(horizon))
            continue
        vec = last_vector(comp_hist, p=p).copy()

        out     = np.empty(horizon)
        current = vec.copy()
        for h in range(horizon):
            dists  = np.linalg.norm(X - current, axis=1)
            nn_idx = np.argpartition(dists, n_neighbors)[:n_neighbors]
            h_bw   = max(float(dists[nn_idx].max()), 1e-10)
            w      = np.exp(-0.5 * (dists[nn_idx] / h_bw) ** 2)
            A      = np.hstack([np.ones((n_neighbors, 1)), X[nn_idx]])
            sw     = np.sqrt(w)
            coeffs, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx],
                                               rcond=None)
            val            = float(coeffs[0] + current @ coeffs[1:])
            out[h]         = val
            current        = np.roll(current, -1)
            current[-1]    = val
        hats.append(out)

    return np.sum(hats, axis=0)   # (horizon,)
```

### 6.2 Интеграция в `app.py` — одиночная модель (LA1/LWR)

Добавить cached-обёртку рядом с `_forecast_lwr`:

```python
from forcaster.forecast.filterbank import forecast_lwr_fb, FB_P, FB_XI

@st.cache_data(show_spinner=False)
def _forecast_fb(dratio_bytes: bytes, origin_k: int,
                 horizon: int, p: int = FB_P, xi: int = FB_XI) -> np.ndarray:
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    return forecast_lwr_fb(dratio, origin_k, horizon, p=p, n_neighbors=xi)
```

В разделе `model = st.radio(...)` добавить вариант:

```python
model = st.radio(
    "Модель",
    ["LA1 — одиночный", "LA1 — авто p", "LWR — авто p",
     "LWR (Гауссовы веса)", "LWR + filter bank", "LA1 — ансамбль"],
    horizontal=True,
)
```

В блоке `if run_btn:` добавить ветку:

```python
elif model == "LWR + filter bank":
    dhat = _forecast_fb(dratio_key, origin_k, horizon)
    forecast_price = reconstruct_price(dhat, ratio0, _ma_fwd)
    forecast_indivs = None
```

### 6.3 Интеграция в `app.py` — авто-p (`_auto_p_forecast`)

В функцию `_auto_p_forecast` добавить параметр `use_filterbank: bool = False`
и отдельную ветку sweep:

```python
# В начале функции, если use_filterbank=True:
if use_filterbank:
    # Фиксированный p=FB_P, sweep не нужен (абсолютное MAPE стабильно)
    # Запускаем единственный конфиг: p=20, без PCA
    xi = FB_XI
    dhat = forecast_lwr_fb(dratio, val_origin + 1, total_h,
                           p=FB_P, n_neighbors=xi)
    # ... вычисляем MAPE на валидационном окне, формируем top_cands
    return [{"p": FB_P, "pca_k": 0, "mape": mape, "dhat": dhat.tolist()}], {}
```

Или более гибко — включить filter bank как один из кандидатов в общем sweep:

```python
# Добавить fb-кандидата к списку результатов:
if use_filterbank:
    fb_dhat = forecast_lwr_fb(dratio, origin_arg, total_h)
    fb_r_hat = ratio0_val + np.cumsum(fb_dhat[:val_horizon])
    fb_mape  = mean_absolute_percentage_error(fb_r_hat, actual_ratio)
    all_results.append((fb_mape, FB_P, -1, fb_dhat))  # pca_k=-1 = fb-маркер
```

Это позволит сравнить filter bank с лучшим auto-p в одном UI.

### 6.4 Sidebar: новые контролы (в секции «LA-прогноз»)

```python
st.divider()
st.subheader("Filter bank (медленные компоненты)")

use_fb = st.checkbox(
    "LWR + Filter bank C3+C4+C5",
    help="Разбивает Δratio на частотные полосы Butterworth order=4. "
         "LA применяется к медленным компонентам (периоды >52 баров). "
         "Используется фиксированный p=20, Ξ=63.",
)
if use_fb:
    fb_p  = st.slider("p для filter bank", 10, 40, FB_P, 5, key="fb_p",
                      help="Рекомендуется 20 (≈40% от периода C3≈52 бара). "
                           "При p<13 — сбои на LKOH/VTBR/MRKP.")
    fb_xi = 3 * (fb_p + 1)
    st.caption(f"ξ = {fb_xi}  |  C3+C4+C5  |  order=4  |  sep-режим")
```

---

## 7. Особенности при разных интервалах

При интервале 1h компоненты C3+C4+C5 описывают другие временные масштабы:

| Компонента | 1d (часы торг.) | 1h (часы) |
|---|---|---|
| C3 период | ~52 дня | ~52 часа ≈ 7 торг. дней |
| C4 период | ~103 дня | ~103 часа ≈ 13 торг. дней |
| C5 период | ~206+ дней | ~206+ часов ≈ 26+ торг. дней |

Для 1h тест не проводился (задача скрипта 35).
Рекомендация: тот же p=20 как стартовая точка (T_min для 1h ≈ 52h → p_min≈13).

---

## 8. Чего НЕ делать

1. **Не применять filt bank к ratio** — ratio содержит тренд MA; фильтр разложит
   тренд на компоненты некорректно. Метод разработан для dratio.

2. **Не использовать sosfiltfilt** — нулевая фаза требует будущих данных →
   look-ahead bias (как в EMD, скр. 28–29).

3. **Не вычислять COMP на всём ряду** во внешнем цикле (соблазн оптимизации):

   ```python
   # НЕПРАВИЛЬНО — look-ahead:
   COMP = make_filter_bank(dratio)   # использует данные после vo!
   for vo in origins:
       hat = _lwr_forecast(COMP[3, :vo], ...)

   # ПРАВИЛЬНО — каузально:
   for vo in origins:
       COMP = make_filter_bank(dratio[:vo])
       hat = _lwr_forecast(COMP[3], ...)
   ```

   **Исключение**: при однократном прогнозе из UI (не walk-forward) оба способа
   дают одинаковый результат, т.к. пользователь прогнозирует только от последнего бара.
   Во walk-forward тесте — строго каузально.

4. **Не делать sweep p** для filter bank — абсолютное MAPE стабильно при
   p ∈ [5..40] (0.00917–0.00937). Фиксировать p=20.

---

## 9. Связанные файлы исследований

| Скрипт | Что проверяет | Результат |
|---|---|---|
| `research/30_filterbank_la.py` | Первый тест FB на SBER, p=5 | −28.3%, p<0.0001 |
| `research/31_ssa_la.py` | SSA как альтернатива FB | −27.2% (конвергенция) |
| `research/32_fb_multiticker.py` | Multi-ticker, p=5 | −5.4% агрегат, LKOH +36% |
| `research/33_fb_psweep.py` | Sweep p∈{5,10,20,40}, SBER | p=20: abs MAPE 0.00917 |
| `research/34_fb_multiticker_p20.py` | Multi-ticker, p=20 | **−22.0%, все 8 тикеров** |
