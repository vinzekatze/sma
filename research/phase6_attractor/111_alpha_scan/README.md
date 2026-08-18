# 111 — скрининг мультипликатора alpha в m = round(alpha · d)

## 1. Гипотезы

Два эксперимента дали опорные точки для качества LP-фильтра:
- **Эксп.101 (m=3d):** rMAE=0.327 при n_iter=2, d=11
- **Эксп.109 (m=2d+1):** rMAE=0.393 при n_iter=3, d=3

Непрерывный параметр `alpha` в `m = max(3, round(alpha · d))` позволяет прощупать
пространство между этими точками.

- **H1.** Существует промежуточное значение alpha ∈ (2.0, 3.0), при котором
  rMAE лучше обеих опорных точек — то есть m=3d не является оптимальным
  мультипликатором.
- **H2.** Форма rMAE(alpha) при фиксированном d не монотонна — есть локальный
  минимум вблизи одного из промежуточных значений (2.25, 2.50, 2.75).
- **H3.** Оптимальное alpha сдвигается с ростом d — при малых d (d=3)
  предпочтительно меньшее alpha, при больших d (d=11, 15) предпочтительно
  большее alpha (≈3.0), что согласуется с выводом эксп.109.

## 2. Концепции, терминология, обозначения

- **alpha** — мультипликатор окна LP-фильтра. `m = max(3, round(alpha · d))`.
  Управляет соотношением «пространство шума / пространство сигнала»:
  noise_dim = m − d; при alpha=3 → noise_dim=2d (в 2× шире сигнала);
  при alpha=2 → noise_dim=d (равно сигналу, фильтр слабый).
- **m** — фактический размер окна после округления. При малых d соседние alpha
  могут давать одинаковый m (артефакт дискретизации); эти дубликаты выявляются
  через колонку `m` в выходном CSV.
- **d** — размерность локального касательного подпространства (сигнал) в LP-фильтре.
  Одновременно определяет k=10d, p_fit=m, p_max=8m, xi_lwr=3(m+1)+5.
- **n_iter** — число проходов LP-фильтра (2 или 3).
- **att** — сигнал аттрактора: diff(LP_filter(ratio)), ratio = close / logtrend.
- **rMAE** — mean(|error|) / std(true_att), где true_att — ground truth att из
  close[:origin+2] (раскрытие одного бара вперёд без утечки).
- **Якоря сравнения:** эксп.101 (alpha=3.0, d=11, n_iter=2, m=33, rMAE=0.327) и
  эксп.109 (alpha≈2.33 at d=3, n_iter=3, m=7, rMAE=0.393).

## 3. Методология и протокол

Идентична эксп.101/109, за исключением отсутствия LB-диагностики:

1. Для каждого тикера, origin, (alpha, d, n_iter):
   a. `att` из `close[:origin+1]` (без утечки).
   b. Каскад 4 уровней `[8m, 4m, 2m, m]`: отбор ξ_lwr ближайших кандидатов
      (acc_ang/global_blend выключены).
   c. LWR → `pred_att`.
   d. `true_att = att_ext[-1]` из `close[:origin+2]` (ground truth).
   e. Запись (alpha, d, n_iter, m, abs_error, ok) в forecast_accuracy.csv.

2. Анализ (`112_analyze.py`): rMAE(alpha, d, n_iter) = mean(abs_error)/std(true),
   четыре рисунка (кривые по alpha, комбинированные, ось p_fit, heatmap),
   сравнение с якорями эксп.101/109.

## 4. Параметры

| Параметр | Значение |
|---|---|
| Тикеры (скрининг) | SBER, MRKP, CHMF, NVTK |
| Тикеры (FULL_MODE=1) | все 8 |
| Origins | 10 (скрининг) / 40 (полный) на тикер, шаг 5 |
| alpha | {2.00, 2.25, 2.50, 2.75, 3.00} |
| d | {3, 7, 11, 15} |
| n_iter | {2, 3} |
| m | max(3, round(alpha · d)) |
| k | 10 · d |
| p_fit | m |
| p_max | 8 · m (N_LEVELS=4, ×2 octave) |
| ξ_lwr | 3·(m+1)+5 |
| LB-диагностика | выключена |
| acc_ang / global_blend / LP-коррекция | выключены |
| Всего задач (скрининг) | 5×4×2 = 40 комбо × 4 тикера × 10 origins = 1 600 |
| Всего задач (полный) | 40 × 8 × 40 = 12 800 |

Дубликаты (alpha,d) → одинаковый m: при d=3 alpha=2.50 и alpha=2.75 оба дают m=8.
Эти задачи запускаются, дубликат виден через колонку m в CSV; анализ группирует
по alpha (не по m), дубликаты усредняются дополнительным data point.

## 5. Критика постановки

- **Совместный эффект d.** Как и в эксп.101/109, d одновременно меняет m, k,
  p_fit, p_max, xi_lwr. Кривая rMAE(alpha) при фиксированном d отражает совместный
  эффект всего вектора параметров, а не только m/alpha.
- **Малый размер скрининга.** 40 origins на (alpha,d,n_iter)-ячейку: SD оценки
  rMAE высока. Вывод о точном оптимуме alpha следует перепроверить на полном прогоне
  (8 тикеров × 40 origins = 320 obs на ячейку).
- **Без LB-диагностики.** Сигнал d≥LB (34.7% в эксп.101, 23.3% в эксп.109) не
  измеряется. Если alpha оптимум найден, стоит повторить с LB для проверки гейта.
- **Утечка не вводится.** att строится из close[:origin+1], true_val из att_ext[-1]
  (close[:origin+2]) — идентично проверенной схеме эксп.101/109.
- **Дискретизация alpha при малом d.** При d=3 шаг m между alpha-точками = 1, и
  только 4 из 5 alpha дают различный m (alpha=2.50 и alpha=2.75 → m=8). Для d=7,11,15
  все 5 alpha дают уникальные m.
- **Время на kali:** ~25–30 мин с 4 воркерами (d=15 доминирует: m≤45, k=150,
  LP-SVD дорог). Без Docker — скрининговый формат.

## 6. Результаты

_Заполняется после прогона._

## 7. Анализ и выводы

_Заполняется после прогона._

## 8. Дальнейшие направления

_Заполняется после прогона._

---

## Запуск

### Локально (тест, 2 тикера, 3 origins, ~30 сек)

```bash
source /home/kali/.venvs/sma/bin/activate
TEST_MODE=1 WORKERS=4 python research/phase6_attractor/111_alpha_scan/111_alpha_scan.py
```

### Скрининг на kali (4 тикера, 10 origins, ~25–30 мин)

```bash
source /home/kali/.venvs/sma/bin/activate
WORKERS=4 python research/phase6_attractor/111_alpha_scan/111_alpha_scan.py
```

### Полный прогон на хосте (8 тикеров, 40 origins, ~3–4 ч)

```powershell
# PowerShell (Windows)
cd C:\path\to\sma
python research/phase6_attractor/111_alpha_scan/111_alpha_scan.py `
    --  # FULL_MODE=1 WORKERS=8 перед командой:
```

```powershell
$env:FULL_MODE="1"; $env:WORKERS="8"
python research/phase6_attractor/111_alpha_scan/111_alpha_scan.py
```

### Анализ (после прогона)

```bash
python research/phase6_attractor/111_alpha_scan/112_analyze.py
# тест:
python research/phase6_attractor/111_alpha_scan/112_analyze.py --test
```

## Файлы

```
111_alpha_scan.py    — сбор данных (каскад + LWR), alpha = переменный
112_analyze.py       — rMAE(alpha,d,n_iter), 4 рисунка, сравнение с exp.101/109
results/
  forecast_accuracy.csv    — 1 строка = (тикер,alpha,d,n_iter,origin)
  run_meta.json
  analysis_summary.json
figures/
  rmae_vs_alpha.png          — кривые по (d,n_iter)
  rmae_vs_alpha_combined.png — совмещение по d (раздельно n_iter)
  rmae_vs_pfit.png           — ось p_fit=m (сравнение с exp.101/109)
  rmae_heatmap.png           — тепловая карта rMAE(alpha,d)
```
