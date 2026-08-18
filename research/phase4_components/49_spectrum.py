"""
49 — Спектральный анализ: dratio, компоненты C0–C5, АЧХ фильтров.

Графики:
  1. PSD dratio + PSD каждой компоненты на одних осях (лог/лог)
  2. АЧХ (frequency response) каскадных Butterworth-фильтров — что реально вырезается
  3. Энергия по полосам (bar chart) — сравнение модели и Parseval
  4. Спектрограмма dratio (STFT) — нестационарность спектра во времени
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, freqz, welch, spectrogram
from scipy.signal import sosfiltfilt   # только для АЧХ-демонстрации (zero-phase)

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL     = "1d"
EPS          = 1e-10
STD_CUTOFFS  = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
FILTER_ORDER = 4
NPERSEG      = 256    # Welch окно

COMP_COLORS = ["#e41a1c", "#ff7f00", "#a6a600",
               "#4daf4a", "#377eb8", "#984ea3"]
COMP_NAMES  = ["C0  2–4 bar", "C1  4–8 bar", "C2  8–16 bar",
               "C3  16–52 bar", "C4  52–103 bar", "C5  103+ bar"]

# Границы полос (нижняя, верхняя) в единицах периода
BAND_PERIODS = [
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, 16.0),
    (16.0, 52.0),
    (52.0, 103.0),
    (103.0, np.inf),
]


# ── нормализация + данные ─────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    log_c = np.log(close + EPS); n = len(log_c)
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n+1, dtype=np.float64)
    ct=np.cumsum(t); ct2=np.cumsum(t**2); cy=np.cumsum(log_c); cty=np.cumsum(t*log_c)
    denom = cn*ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom>0, (cn*cty-ct*cy)/denom, 0.0)
    a=(cy-b*ct)/cn; trend=a+b*t; trend[:2]=log_c[:2]
    return np.exp(trend)


def make_fb(series: np.ndarray, cutoffs: list, order: int = 4) -> np.ndarray:
    comps=[]; rem=series.copy()
    for fc in cutoffs:
        sos=butter(order,fc,btype="low",output="sos"); low=sosfilt(sos,rem)
        comps.append(rem-low); rem=low
    comps.append(rem); return np.array(comps)


# ── загрузка ──────────────────────────────────────────────────────────────────

print("Загрузка данных...")
all_dratio = []
all_comps  = []

for ticker in TICKERS:
    with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f: c=json.load(f)
    close = np.array([x["close"] for x in c], dtype=np.float64)
    ratio = close / logtrend_causal(close)
    dratio = np.diff(ratio)
    comps  = make_fb(dratio, STD_CUTOFFS, FILTER_ORDER)
    all_dratio.append(dratio)
    all_comps.append(comps)


# ── Welch PSD ─────────────────────────────────────────────────────────────────

def psd_median(signals: list, nperseg: int = NPERSEG):
    """Медианный PSD по списку сигналов."""
    psds = []
    for s in signals:
        f, p = welch(s, fs=1.0, nperseg=min(nperseg, len(s)//4))
        psds.append(p)
    # Все одинаковой длины (берём мин)
    min_len = min(len(p) for p in psds)
    return f[:min_len], np.median([p[:min_len] for p in psds], axis=0)


f_raw, p_raw = psd_median(all_dratio)

comp_psds = []
for ci in range(6):
    signals = [all_comps[ti][ci] for ti in range(len(TICKERS))]
    f_c, p_c = psd_median(signals)
    comp_psds.append((f_c, p_c))


# ══════════════════════════════════════════════════════════════════════════════
#  РИС 1: PSD dratio + компоненты (log-log)
# ══════════════════════════════════════════════════════════════════════════════

fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle(
    "Спектр мощности (Welch PSD): dratio и компоненты C0–C5\n"
    "Медиана по 8 тикерам, 1d, logtrend",
    fontsize=12, fontweight="bold"
)

# Левая: частота (линейная) с вертикальными линиями на границах полос
ax_lin = axes1[0]
ax_lin.semilogy(f_raw, p_raw, color="black", lw=2.5, label="dratio (исходный)", zorder=6)
for ci in range(6):
    f_c, p_c = comp_psds[ci]
    ax_lin.semilogy(f_c, p_c, color=COMP_COLORS[ci], lw=1.8,
                    label=COMP_NAMES[ci], alpha=0.85)

# Вертикали — границы фильтров
for fc in STD_CUTOFFS:
    ax_lin.axvline(fc, color="gray", lw=1.0, ls="--", alpha=0.5)
    period = 1.0 / fc
    ax_lin.text(fc + 0.002, ax_lin.get_ylim()[0] if ax_lin.get_ylim()[0] > 0 else 1e-12,
                f"{period:.0f}b", fontsize=7, color="gray", va="bottom", rotation=90)

ax_lin.set_xlabel("Частота (цикл/бар)", fontsize=10)
ax_lin.set_ylabel("PSD (log)", fontsize=10)
ax_lin.set_title("Частота — линейная шкала", fontsize=10)
ax_lin.legend(fontsize=8, loc="lower left")
ax_lin.grid(alpha=0.3, which="both")
ax_lin.set_xlim(0, 0.5)

# Правая: период (log) — наглядно
ax_per = axes1[1]
# Конвертируем: f → period = 1/f, избегаем f=0
mask = f_raw > 0
periods_raw = 1.0 / f_raw[mask]
ax_per.loglog(periods_raw, p_raw[mask], color="black", lw=2.5, label="dratio", zorder=6)

for ci in range(6):
    f_c, p_c = comp_psds[ci]
    mask_c = f_c > 0
    ax_per.loglog(1.0 / f_c[mask_c], p_c[mask_c], color=COMP_COLORS[ci],
                  lw=1.8, label=COMP_NAMES[ci], alpha=0.85)

# Вертикали на границах полос (периоды)
for fc in STD_CUTOFFS:
    ax_per.axvline(1.0/fc, color="gray", lw=1.0, ls="--", alpha=0.5)
    ax_per.text(1.0/fc * 1.05, ax_per.get_ylim()[0] if ax_per.get_ylim()[0] > 0 else 1e-12,
                f"{1/fc:.0f} bar", fontsize=7, color="gray", va="bottom", rotation=90)

# Заливки полос
poly_alpha = 0.07
ylim_dummy = (1e-15, 1e-3)
for ci, (t_lo, t_hi) in enumerate(BAND_PERIODS):
    t_hi_plot = min(t_hi, 1.0 / f_raw[1])
    if t_lo < t_hi_plot:
        ax_per.axvspan(t_lo, t_hi_plot, color=COMP_COLORS[ci], alpha=poly_alpha)

ax_per.set_xlabel("Период (баров, лог. шкала)", fontsize=10)
ax_per.set_ylabel("PSD (log)", fontsize=10)
ax_per.set_title("Период — логарифмическая шкала", fontsize=10)
ax_per.legend(fontsize=8, loc="upper left")
ax_per.grid(alpha=0.3, which="both")
ax_per.set_xlim(2, 600)

plt.tight_layout()
out1 = OUT_DIR / "49_psd_components.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"  Рис 1: {out1}")
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
#  РИС 2: АЧХ каскадных Butterworth-фильтров
# ══════════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig2.suptitle(
    f"АЧХ каскадных Butterworth LP-фильтров (order={FILTER_ORDER}, causal sosfilt)\n"
    "Полосовые фильтры = разность двух LP",
    fontsize=12, fontweight="bold"
)

N_FREQ = 2048
f_hz = np.linspace(0, 0.5, N_FREQ)
w = 2 * np.pi * f_hz   # нормированная угловая частота

# LP АЧХ для каждой ступени
ax_lp = axes2[0]
ax_bp = axes2[1]

lp_responses = {}
prev_mag = np.ones(N_FREQ)   # «полный» сигнал при входе

for i, fc in enumerate(STD_CUTOFFS):
    sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
    # freqz работает с ba, но sos можно конвертировать
    # Используем каскадное произведение секций
    H = np.ones(N_FREQ, dtype=complex)
    for section in sos:
        b = section[:3]; a = section[3:]
        _, h_sec = freqz(b, a, worN=N_FREQ, fs=1.0)
        H *= h_sec
    lp_mag = np.abs(H)
    lp_responses[fc] = lp_mag

    ax_lp.plot(f_hz, 20 * np.log10(lp_mag + EPS),
               label=f"LP fc={fc}  (T={1/fc:.0f} bar)", lw=1.8)

ax_lp.set_ylabel("АЧХ LP (дБ)", fontsize=10)
ax_lp.set_title("LP-фильтры (Butterworth, causal)", fontsize=10)
ax_lp.legend(fontsize=8, ncol=2)
ax_lp.grid(alpha=0.3)
ax_lp.axhline(-3, color="gray", ls="--", lw=1, alpha=0.6)
ax_lp.set_ylim(-80, 5)

# Полосовые = разность LP ступеней
prev_lp = np.ones(N_FREQ)
bp_responses = []
for i, fc in enumerate(STD_CUTOFFS):
    cur_lp = lp_responses[fc]
    bp_mag = np.abs(prev_lp - cur_lp)
    bp_responses.append(bp_mag)
    ax_bp.plot(f_hz, 20 * np.log10(bp_mag + EPS),
               color=COMP_COLORS[i], lw=2.0, label=COMP_NAMES[i])
    prev_lp = cur_lp

# Последняя LP — это C5
ax_bp.plot(f_hz, 20 * np.log10(lp_responses[STD_CUTOFFS[-1]] + EPS),
           color=COMP_COLORS[5], lw=2.0, ls="--", label=COMP_NAMES[5])

# Вертикали на fc
for fc in STD_CUTOFFS:
    ax_bp.axvline(fc, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax_bp.text(fc, -5, f"{1/fc:.0f}b", fontsize=7, color="gray",
               ha="center", va="bottom")

ax_bp.set_xlabel("Частота (цикл/бар)", fontsize=10)
ax_bp.set_ylabel("АЧХ полосового фильтра (дБ)", fontsize=10)
ax_bp.set_title("Полосовые характеристики компонент (разность LP)", fontsize=10)
ax_bp.legend(fontsize=8, ncol=3)
ax_bp.grid(alpha=0.3)
ax_bp.set_ylim(-80, 5)
ax_bp.set_xlim(0, 0.5)

plt.tight_layout()
out2 = OUT_DIR / "49_filter_response.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"  Рис 2: {out2}")
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
#  РИС 3: Энергия по полосам — Parseval-проверка
# ══════════════════════════════════════════════════════════════════════════════

# Медианная энергия компонент по тикерам
energies_med = []
total_energy_med = []
for ti in range(len(TICKERS)):
    comp = all_comps[ti]
    total_e = float(np.sum(all_dratio[ti] ** 2))
    comp_e  = [float(np.sum(comp[ci] ** 2)) for ci in range(6)]
    energies_med.append(comp_e)
    total_energy_med.append(total_e)

energies_arr = np.array(energies_med)   # (8, 6)
total_arr    = np.array(total_energy_med)

# Нормированная доля
frac_med = np.median(energies_arr / total_arr[:, None], axis=0)

# Parseval: сумма компонент / исходный
parseval_sum = np.median(energies_arr.sum(axis=1) / total_arr)

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle(
    "Распределение энергии dratio по компонентам  |  Parseval-проверка",
    fontsize=12, fontweight="bold"
)

ax_bar = axes3[0]
bars = ax_bar.bar(range(6), frac_med * 100, color=COMP_COLORS, alpha=0.85,
                  edgecolor="k", lw=0.7)
for bar, frac in zip(bars, frac_med):
    ax_bar.text(bar.get_x() + bar.get_width()/2, frac * 100 + 0.3,
                f"{frac*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_bar.set_xticks(range(6))
ax_bar.set_xticklabels(COMP_NAMES, rotation=20, ha="right", fontsize=8)
ax_bar.set_ylabel("Доля энергии dratio (%)", fontsize=10)
ax_bar.set_title(f"Медиана по 8 тикерам\nΣ компонент / исходный = {parseval_sum:.4f} (Parseval)", fontsize=10)
ax_bar.grid(alpha=0.3, axis="y")

# Правая: per-ticker (scatter + box)
ax_box = axes3[1]
data_for_box = [(energies_arr[:, ci] / total_arr * 100).tolist() for ci in range(6)]
bp = ax_box.boxplot(data_for_box, patch_artist=True, medianprops={"color": "black", "lw": 2})
for patch, color in zip(bp["boxes"], COMP_COLORS):
    patch.set_facecolor(color); patch.set_alpha(0.7)
# Scatter per-ticker
for ci in range(6):
    y = energies_arr[:, ci] / total_arr * 100
    x = np.full(len(y), ci + 1) + np.random.uniform(-0.15, 0.15, len(y))
    ax_box.scatter(x, y, color=COMP_COLORS[ci], s=30, zorder=5, alpha=0.8)

ax_box.set_xticklabels([f"C{i}" for i in range(6)], fontsize=9)
ax_box.set_ylabel("Доля энергии (%)", fontsize=10)
ax_box.set_title("Разброс по тикерам (box + scatter)", fontsize=10)
ax_box.grid(alpha=0.3, axis="y")

plt.tight_layout()
out3 = OUT_DIR / "49_energy_bands.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"  Рис 3: {out3}")
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
#  РИС 4: Спектрограмма dratio — нестационарность (SBER)
# ══════════════════════════════════════════════════════════════════════════════

ticker_spec = "SBER"
ti_spec = TICKERS.index(ticker_spec)
dratio_sber = all_dratio[ti_spec]
n_sber = len(dratio_sber)

fig4, axes4 = plt.subplots(3, 1, figsize=(15, 11), sharex=False,
                            gridspec_kw={"height_ratios": [1, 2, 1]})
fig4.suptitle(
    f"Спектрограмма dratio — {ticker_spec} 1d  |  Нестационарность спектра",
    fontsize=12, fontweight="bold"
)

# Верхняя: dratio
ax_dr = axes4[0]
t_arr = np.arange(n_sber)
ax_dr.plot(t_arr, dratio_sber, color="steelblue", lw=0.6, alpha=0.7)
ax_dr.set_ylabel("dratio", fontsize=9)
ax_dr.set_title("dratio (исходный)", fontsize=9)
ax_dr.grid(alpha=0.3)

# Средняя: спектрограмма
ax_sg = axes4[1]
f_sg, t_sg, Sxx = spectrogram(dratio_sber, fs=1.0,
                               nperseg=min(128, n_sber // 8),
                               noverlap=min(96, n_sber // 12))
# Конвертируем f → period (ось Y)
f_sg_nonzero = f_sg[1:]   # убираем DC
t_sg_plot    = t_sg
Sxx_plot     = Sxx[1:, :]   # убираем DC строку

im = ax_sg.pcolormesh(t_sg_plot, f_sg_nonzero, 10 * np.log10(Sxx_plot + EPS),
                      cmap="inferno", shading="gouraud")
plt.colorbar(im, ax=ax_sg, label="дБ", shrink=0.8)
for fc in STD_CUTOFFS:
    ax_sg.axhline(fc, color="white", lw=1.0, ls="--", alpha=0.6)
    ax_sg.text(t_sg_plot[-1] * 1.01, fc, f"{1/fc:.0f}b", fontsize=7,
               color="white", va="center")

ax_sg.set_ylabel("Частота (цикл/бар)", fontsize=9)
ax_sg.set_title("Спектрограмма (STFT, лог шкала)", fontsize=9)
ax_sg.set_ylim(0, 0.5)

# Нижняя: суммарная мощность в медленной зоне (f < 0.0625) vs быстрой (f > 0.0625)
ax_pow = axes4[2]
fc_split = 0.0625
idx_slow = f_sg_nonzero < fc_split
idx_fast = f_sg_nonzero >= fc_split
power_slow = Sxx_plot[idx_slow, :].sum(axis=0)
power_fast = Sxx_plot[idx_fast, :].sum(axis=0)
ratio_slow = power_slow / (power_slow + power_fast + EPS)

ax_pow.fill_between(t_sg_plot, ratio_slow * 100, alpha=0.5, color="tab:blue",
                    label="Медленные (< 1/16 Hz)")
ax_pow.fill_between(t_sg_plot, ratio_slow * 100, 100, alpha=0.4, color="tab:red",
                    label="Быстрые (> 1/16 Hz)")
ax_pow.axhline(np.mean(ratio_slow) * 100, color="navy", lw=1.5, ls="--",
               label=f"Среднее {np.mean(ratio_slow)*100:.1f}%")
ax_pow.set_ylim(0, 100); ax_pow.set_xlim(0, t_sg_plot[-1])
ax_pow.set_ylabel("% мощности", fontsize=9)
ax_pow.set_xlabel("Бар", fontsize=9)
ax_pow.set_title("Доля мощности: медленные vs быстрые компоненты", fontsize=9)
ax_pow.legend(fontsize=8); ax_pow.grid(alpha=0.3)

plt.tight_layout()
out4 = OUT_DIR / "49_spectrogram.png"
fig4.savefig(out4, dpi=150, bbox_inches="tight")
print(f"  Рис 4: {out4}")
plt.close(fig4)


# ══════════════════════════════════════════════════════════════════════════════
#  РИС 5: PSD компонент — нормированные (каждая на свою дисперсию)
# ══════════════════════════════════════════════════════════════════════════════

fig5, axes5 = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
fig5.suptitle(
    "PSD каждой компоненты (нормирован на σ²) — ожидаемая и реальная форма\n"
    "Серый: PSD белого шума, синяя кривая: медиана по 8 тикерам",
    fontsize=11, fontweight="bold"
)

for ci, ax in zip(range(6), axes5.flatten()):
    signals_ci = [all_comps[ti][ci] for ti in range(len(TICKERS))]
    f_c, _ = welch(signals_ci[0], fs=1.0, nperseg=min(NPERSEG, len(signals_ci[0])//4))

    psds_norm = []
    for s in signals_ci:
        f_s, p_s = welch(s, fs=1.0, nperseg=min(NPERSEG, len(s)//4))
        var_s = float(np.var(s)) + EPS
        psds_norm.append(p_s[:len(f_c)] / var_s)

    psd_med_n = np.median(psds_norm, axis=0)
    psd_per_n = np.array(psds_norm)

    # Ожидаемый PSD белого шума = 2/N (плоский)
    white_level = 2.0 / NPERSEG

    # Ожидаемый PSD идеального bandpass: прямоугольник
    t_lo, t_hi = BAND_PERIODS[ci]
    f_lo_band = 0.0 if t_hi == np.inf else 1.0 / t_hi
    f_hi_band = 0.5 if t_lo <= 2 else 1.0 / t_lo

    # Per-ticker (светлые)
    for p_n in psd_per_n:
        ax.semilogy(f_c, p_n, color=COMP_COLORS[ci], lw=0.7, alpha=0.3)
    # Медиана
    ax.semilogy(f_c, psd_med_n, color=COMP_COLORS[ci], lw=2.5, label="Медиана", zorder=5)
    # Белый шум
    ax.axhline(white_level, color="gray", lw=1.0, ls=":", alpha=0.8, label=f"Белый шум")
    # Границы полосы
    ax.axvline(f_lo_band, color="black", lw=1.2, ls="--", alpha=0.5)
    if f_hi_band < 0.5:
        ax.axvline(f_hi_band, color="black", lw=1.2, ls="--", alpha=0.5)

    ax.set_title(f"{COMP_NAMES[ci]}", fontsize=10, color=COMP_COLORS[ci], fontweight="bold")
    ax.set_xlabel("Частота", fontsize=8)
    ax.set_ylabel("PSD / σ²", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(0, 0.5)

plt.tight_layout()
out5 = OUT_DIR / "49_psd_normalized.png"
fig5.savefig(out5, dpi=150, bbox_inches="tight")
print(f"  Рис 5: {out5}")
plt.close(fig5)


# ── Вывод Parseval ─────────────────────────────────────────────────────────────

print("\n  Parseval-проверка (Σ энергий компонент / исходная):")
for ti, ticker in enumerate(TICKERS):
    s = parseval_sum  # медиана
    comp_e = energies_arr[ti]
    total_e = total_arr[ti]
    ratio_p = comp_e.sum() / total_e
    print(f"  {ticker}: Σ/total = {ratio_p:.6f}  ({'+' if ratio_p>=1 else '-'}{abs(ratio_p-1)*100:.4f}%)")

print(f"\n  Медиана: {parseval_sum:.6f}")
print(f"\n  Доля энергии по компонентам (медиана):")
for ci in range(6):
    print(f"    C{ci}: {frac_med[ci]*100:.2f}%  ({COMP_NAMES[ci]})")

print(f"\nФайлы:")
for out in [out1, out2, out3, out4, out5]:
    print(f"  {out.name}")
