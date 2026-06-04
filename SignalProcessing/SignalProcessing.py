import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft

# Параметри (варіант 5)
n = 500
Fs = 1000
F_max = 11
line_width = 1
font_size = 14

os.makedirs("./figures", exist_ok=True)


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


# Функція побудови графіку
def plot_and_save(x_data, y_data, x_label, y_label, title_text):
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(x_data, y_data, linewidth=line_width)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    plt.title(title_text, fontsize=font_size)
    ax.grid(True)
    fig.savefig(f"./figures/{title_text}.png", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Збережено: {title_text}.png")


# Графік сигналу
plot_and_save(
    time_axis, filtered_signal,
    "Час (секунди)", "Амплітуда сигналу",
    f"Сигнал з максимальною частотою F_max = {F_max} Гц"
)

# Графік спектру
plot_and_save(
    freq_axis_shifted, spectrum_shifted,
    "Частота (Гц)", "Амплітуда спектру",
    f"Спектр сигналу з максимальною частотою F_max = {F_max} Гц"
)

