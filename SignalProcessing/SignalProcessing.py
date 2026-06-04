import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft

# Параметри (варіант 5)
n = 500
Fs = 1000
F_max = 11
F_filter = 18
line_width = 1
font_size = 14

os.makedirs("./figures", exist_ok=True)


# ЛАБОРАТОРНА 2

# Генерація випадкового сигналу
np.random.seed(42)
random_signal = np.random.normal(0, 10, n)

# Вісь часу
time_axis = np.arange(n) / Fs

# Фільтрація ФНЧ
w = F_max / (Fs / 2)
sos_filter = signal.butter(3, w, 'low', output='sos')
filtered_signal = signal.sosfiltfilt(sos_filter, random_signal)

# Спектр
spectrum = fft.fft(filtered_signal)
spectrum_shifted = np.abs(fft.fftshift(spectrum))
freq_axis = fft.fftfreq(n, 1 / n)
freq_axis_shifted = fft.fftshift(freq_axis)


def plot_and_save(x_data, y_data, x_label, y_label, title_text, filename):
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(x_data, y_data, linewidth=line_width)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    plt.title(title_text, fontsize=font_size)
    ax.grid(True)
    fig.savefig(f"./figures/{filename}.png", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Збережено: {filename}.png")


def create_2x2_plot(y_data, x_data, title_text, filename, x_label, y_label):
    fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
    s = 0
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i][j].plot(x_data, y_data[s], linewidth=line_width)
            ax[i][j].grid(True)
            s += 1
    fig.supxlabel(x_label, fontsize=font_size)
    fig.supylabel(y_label, fontsize=font_size)
    fig.suptitle(title_text, fontsize=font_size)
    fig.tight_layout()
    fig.savefig(f"./figures/{filename}.png", dpi=600)
    plt.close(fig)
    print(f"Збережено: {filename}.png")


# Рисунок 2.1
plot_and_save(
    time_axis, filtered_signal,
    "Час (секунди)", "Амплітуда сигналу",
    f"Сигнал з максимальною частотою F_max = {F_max} Гц",
    f"Рисунок 2.1 – Сигнал з максимальною частотою F_max = {F_max} Гц"
)

# Рисунок 2.2
plot_and_save(
    freq_axis_shifted, spectrum_shifted,
    "Частота (Гц)", "Амплітуда спектру",
    f"Спектр сигналу з максимальною частотою F_max = {F_max} Гц",
    f"Рисунок 2.2 – Спектр сигналу з максимальною частотою F_max = {F_max} Гц"
)


# ЛАБОРАТОРНА 3

discrete_signals = []
discrete_spectrums = []
reconstructed_signals = []
variance_errors = []
snr_values = []
steps = [2, 4, 8, 16]

for Dt in steps:
    # Дискретизація
    discrete_signal = np.zeros(n)
    for i in range(0, round(n / Dt)):
        discrete_signal[i * Dt] = filtered_signal[i * Dt]
    discrete_signals += [list(discrete_signal)]

    # Спектр дискретизованого сигналу
    sig_fft = fft.fft(discrete_signal)
    sig_fft_shifted = np.abs(fft.fftshift(sig_fft))
    discrete_spectrums += [list(sig_fft_shifted)]

    # Відновлення через ФНЧ
    w_filter = F_filter / (Fs / 2)
    sos_reconstruct = signal.butter(3, w_filter, 'low', output='sos')
    reconstructed = signal.sosfiltfilt(sos_reconstruct, discrete_signal)
    reconstructed_signals += [list(reconstructed)]

    # Дисперсія та SNR
    E1 = reconstructed - filtered_signal
    var_signal = np.var(filtered_signal)
    var_error = np.var(E1)
    variance_errors.append(var_error)
    snr_values.append(var_signal / var_error)


# Рисунок 3.1
create_2x2_plot(
    discrete_signals, time_axis,
    "Сигнал з кроком дискретизації Dt = (2, 4, 8, 16)",
    "Рисунок 3.1 – Сигнал з кроком дискретизації Dt = (2, 4, 8, 16)",
    "Час (секунди)", "Амплітуда сигналу"
)

# Рисунок 3.2
create_2x2_plot(
    discrete_spectrums, freq_axis_shifted,
    "Спектри сигналів з кроком дискретизації Dt = (2, 4, 8, 16)",
    "Рисунок 3.2 – Спектри сигналів з кроком дискретизації Dt = (2, 4, 8, 16)",
    "Частота (Гц)", "Амплітуда спектру"
)

# Рисунок 3.3
create_2x2_plot(
    reconstructed_signals, time_axis,
    "Відновлені аналогові сигнали з кроком дискретизації Dt = (2, 4, 8, 16)",
    "Рисунок 3.3 – Відновлені аналогові сигнали з кроком дискретизації Dt = (2, 4, 8, 16)",
    "Час (секунди)", "Амплітуда сигналу"
)

# Рисунок 3.4
plot_and_save(
    steps, variance_errors,
    "Крок дискретизації", "Дисперсія",
    "Залежність дисперсії від кроку дискретизації",
    "Рисунок 3.4 – Залежність дисперсії від кроку дискретизації"
)

# Рисунок 3.5
plot_and_save(
    steps, snr_values,
    "Крок дискретизації", "ССШ",
    "Залежність співвідношення сигнал-шум від кроку дискретизації",
    "Рисунок 3.5 – Залежність співвідношення сигнал-шум від кроку дискретизації"
)

print("Готово! 7 графіків збережено у ./figures/")
