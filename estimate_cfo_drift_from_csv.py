import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules import ops_file

# --- PARAMETRY ---
#filename = "logs/rx_samples.csv"  # <-- ZMIEŃ NA SWÓJ PLIK
filename = "logs/rx_samples_32768.csv"  # <-- ZMIEŃ NA SWÓJ PLIK
f_s = 3_000_000              # częstotliwość próbkowania w Hz
sps = 4                     # samples per symbol

# --- WCZYTANIE ---
df = pd.read_csv(filename)
#samples = df['real'].values + 1j * df['imag'].values
samples = ops_file.open_csv_and_load_np_complex128 ( filename )
N = len(samples)

# --- DRYFT FAZY ---
# Obliczamy różnicę fazową między kolejnymi próbkami
phases = np.angle(samples[1:] * np.conj(samples[:-1]))  # Δφ
time_axis = np.arange(1, N) / f_s

# Estymacja dryftu częstotliwości
delta_phi_unwrapped = np.unwrap(phases)
instantaneous_freq = delta_phi_unwrapped * f_s / (2 * np.pi)  # w Hz

# --- WYNIKI ---
print(f"Samples no.: {N}")
print(f"Average drift CFO: {np.mean(instantaneous_freq):.2f} Hz")
print(f"Rozrzut: min {np.min(instantaneous_freq):.2f} Hz, max {np.max(instantaneous_freq):.2f} Hz")

# --- WYKRES ---
plt.figure(figsize=(10, 4))
plt.plot(time_axis, instantaneous_freq)
plt.title("Estimated Carrier Frequency Offset (CFO) over time")
plt.xlabel("time [s]")
plt.ylabel("Frequency [Hz]")
plt.grid(True)
plt.tight_layout()
plt.show()
