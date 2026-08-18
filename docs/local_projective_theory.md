# Local Projective Noise Reduction: теория и параметры

Исследование проведено 2026-06-19. Источники — академическая литература и TISEAN документация.

---

## 1. Метод и условия применимости

**LP (GHKSS, Grassberger-Hegger-Kantz-Schaffrath-Schreiber, 1993)** предполагает, что данные лежат вблизи низкоразмерного многообразия в пространстве вложения. Каждая точка m-мерного фазового пространства проецируется на локальное d-мерное касательное подпространство (PCA по k ближайшим соседям), затем диагональное усреднение восстанавливает скалярный ряд.

**Критическое предупреждение TISEAN:** на чистом шуме алгоритм «picks statistical fluctuations and spuriously interprets them as structure» — конвергенция к многообразию и схлопывание облака точек внешне неразличимы. LP применим только там, где данные реально лежат вблизи низкоразмерного многообразия.

**Применение к финансовым рядам:** литература противоречива. Medленная составляющая (trend-нормированная цена `ratio = close / logtrend`) может иметь детерминированную структуру; быстрая (`dratio` = returns) — нет. Поэтому методически правильно применять LP к `ratio`, а не к `dratio`.

---

## 2. Корреляционный интеграл при шуме: два масштаба

На графике `ln C(m,r)` от `ln r` при наличии шума видны три зоны:

```
ln C(m,r)
    │   шум-режим    │  аттрактор-режим  │  насыщение
    │   slope ≈ m    │  slope ≈ D₂       │  slope ≈ 0
    └────────────────────────────────────────── ln r
      малые r         средние r            крупные r
      r < σ_noise     σ_noise < r < L_att
```

- **Малые r < σ_noise**: шум доминирует, наклон ≈ m (размерность вложения). Аттрактор недоступен.
- **Средние r**: детерминированный аттрактор, наклон стабилизируется на D₂ независимо от m. Кривые для разных m становятся параллельны. Это рабочая зона.
- **Крупные r**: насыщение (конечный размер аттрактора).

**Следствие для выбора m:** m считают достаточным, когда кривые `ln C(m,r)` при разных m становятся параллельны в средней зоне. Это устойчивее FNN: проверяет геометрию, а не соседей.

**Следствие для LP:** окрестность k должна охватывать масштаб r > σ_noise — только тогда локальная PCA видит структуру аттрактора, а не шум.

---

## 3. Параметр m: over-embedding

Из TISEAN (прямая цитата): *«The embedding parameters are usually chosen quite differently from other applications since considerable over-embedding may lead to better noise averaging»*.

**Смысл over-embedding:** при m >> q алгоритм имеет много «шумовых направлений» для нахождения, SVD точнее отделяет сигнальное подпространство от шумового.

```
m = 9, q = 3:  6 шумовых измерений, 3 сигнальных → SVD стабильна
m = 4, q = 3:  1 шумовое измерение → SVD нестабильна, мало что отбрасывать
```

**Нижняя граница:** теорема Такенса: m ≥ 2q + 1. На практике для зашумлённых рядов: m = 3q...5q.

**При высоком шуме:** базовая LP (без higher-order refinements) работает лучше. Higher-order фильтры выигрывают только при низком шуме и длинных рядах (Chu et al., 2015, PubMed 26117108).

---

## 4. Параметр q (= d): manifold dimension

Самый «политический» параметр. TISEAN: *«The answer partly depends on the purpose of the experiment»*.

| Выбор q | Эффект | Риск |
|---|---|---|
| Малое q (агрессивная проекция) | Минимальный rms-остаток | Систематические искажения структуры аттрактора |
| Большее q (консервативная) | Сохранение геометрии | Меньше шума убирается |

**Важная оговорка TISEAN:** *«Points are only moved towards the local linear subspace and too low a value of q does not do as much harm as may be thought»* — алгоритм перемещает точки только ортогонально многообразию, поэтому занижение q менее опасно, чем кажется.

**Как выбирать:** из FNN или корреляционного интеграла на исходном ряду: q ≈ D₂. В нашем случае `ratio` имеет аттрактор при p≈4 (FNN, фаза 2), поэтому q=3 — разумный выбор с небольшим запасом ниже D₂.

---

## 5. Параметр k: число соседей

Правило: k >> m для стабильности SVD. TISEAN default: k=50. Из документации `ghkss`: пример m=7, k=20 (минимально). В нашей реализации k=30 при m=9 → k ≈ 3.3m.

**Связь с шумом:** размер окрестности (в пространственных единицах) должен быть «по меньшей мере сопоставим с предполагаемым уровнем шума» (TISEAN). Это значит: если σ_noise оценена, минимальный радиус окрестности `r_min` должен быть ≥ σ_noise. Параметр `r` в `ghkss` (default 1/1000) задаёт именно этот минимальный радиус.

---

## 6. Embedding window и задержка τ

**Embedding window:** τ_w = (m-1)·τ — суммарный временной охват вектора вложения. Предложен как более фундаментальный параметр, чем m и τ по отдельности.

При τ=1 (наш случай): τ_w = m-1. Оптимальный τ_w — когда AMI или redundancy достигает первого «плоскогорья».

**AMI vs Redundancy при сильном шуме:**
- AMI (Average Mutual Information) — первый минимум I(x(t), x(t+τ)). При высоком шуме минимум смещается к нулю или исчезает.
- **Redundancy** (Paluš, 1995) — оценивает совместное распределение всего m-мерного вектора, а не попарную корреляцию. Устойчивее к шуму, потому что AMI — частный случай (m=2).
- Для выбора m: redundancy(m) перестаёт убывать — «достаточная» размерность вложения найдена.

---

## 7. FNN как критерий выбора параметров LP

**FNN не работает для выбора LP-параметров.** Причины:
1. FNN чувствителен к шуму: при большом SNR даже малый шум даёт некорректный p_opt (OSTI 503675).
2. LP-фильтрация создаёт автокорреляцию на ~m баров, что искажает FNN-статистику.
3. Наш скр.97 подтвердил: p_opt = 3–4 при любых разумных (d, m) — FNN не различает конфиги.

**Честный критерий — только rMAE walk-forward.**

---

## 8. Оценка уровня шума

Schreiber разработал метод оценки σ_noise через влияние шума на корреляционную размерность (coarse-grained entropy, arxiv cond-mat/0301326). Алгоритм:

1. Вычислить C(m, r) на нескольких m.
2. В зоне малых r наклон ≈ m (шум-режим). Граница перехода r* ≈ σ_noise.
3. σ_noise оценивается как точка перегиба между двумя режимами.

Практически: нужно знать σ_noise ДО подбора k (чтобы задать минимальный радиус окрестности ≥ σ_noise).

---

## 9. Применимость к нашим данным (MOEX 1d)

| Аспект | Вывод |
|---|---|
| Есть ли аттрактор? | В `ratio` — да (FNN, фаза 2, p≈4). В `dratio` — нет (стохастик) |
| LP к `ratio`? | Методически корректно — LP убирает шум из медленного компонента |
| LP к `dratio`? | Некорректно — нет аттрактора, LP схлопывает облако в статфлуктуации |
| m=9, q=3 | Соответствует over-embedding (3q), согласуется с p_opt≈4 из FNN |
| k=30 | Приемлемо (3.3m), но ниже TISEAN default 50 |
| Оценка σ_noise | Не делали — открытое направление |

---

## Источники

1. [Locally projective nonlinear noise reduction — TISEAN 2.1](https://www.pks.mpg.de/tisean//TISEAN_2.1/docs/chaospaper/node24.html)
2. [TISEAN: Practical implementation — Hegger, Kantz, Schreiber (chao-dyn/9810005)](https://arxiv.org/pdf/chao-dyn/9810005)
3. [ghkss function reference — Octave Forge / TISEAN](https://octave.sourceforge.io/tisean/function/ghkss.html)
4. [Improvements to LP noise reduction: higher order refinements — Chu et al. (PubMed 26117108)](https://pubmed.ncbi.nlm.nih.gov/26117108/)
5. [Selecting embedding delays: overview + persistent homology — Chaos 2023](https://pubs.aip.org/aip/cha/article/33/3/032101/2881154/Selecting-embedding-delays-An-overview-of)
6. [Selecting embedding delays (arxiv 2302.03447)](https://arxiv.org/pdf/2302.03447)
7. [How to Determine the Redundancy of Noisy Chaotic Time Series — Paluš (ResearchGate)](https://www.researchgate.net/publication/2776513_How_to_Determine_the_Redundancy_of_Noisy_Chaotic_Time_Series)
8. [Generalized redundancies for time series analysis (comp-gas/9405006)](https://arxiv.org/pdf/comp-gas/9405006)
9. [False-nearest-neighbors algorithm and noise-corrupted time series (OSTI 503675)](https://www.osti.gov/biblio/503675)
10. [Noise level estimation via coarse-grained entropy — Schreiber (cond-mat/0301326)](https://arxiv.org/pdf/cond-mat/0301326)
11. [Nonlinear Forecasting of Noisy Financial Data (Springer)](https://link.springer.com/chapter/10.1007/978-1-4615-0931-8_22)
12. [Chaoticity vs stochasticity in financial markets — S&P 500 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1007570421004858)
13. [The correlation dimension of returns with stochastic volatility (ResearchGate)](https://www.researchgate.net/publication/24128051_The_correlation_dimension_of_returns_with_stochastic_volatility)
14. [Nonlinear Time Series Analysis — Kantz & Schreiber (Cambridge UP, 2nd ed.)](https://www.cambridge.org/core/books/nonlinear-time-series-analysis/519783E4E8A2C3DCD4641E42765309C7)
15. [Nonlinear dynamics, delay times, and embedding windows (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0167278998002401)
