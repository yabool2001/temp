import numpy as np
import matplotlib.pyplot as plt

# Parametry
N = 1000  # liczba symboli
f_offset = 0.01  # "częstotliwość obracania się fazy" (np. z powodu offsetu LO)

# Generuj losowe symbole QPSK
symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=N)

# Symuluj obrót fazy w czasie
n = np.arange(N)
phase_drift = np.exp(1j * 2 * np.pi * f_offset * n)  # zmieniająca się faza
received = symbols * phase_drift  # „obrócony” sygnał

# Pokaż konstelację przed korekcją
plt.figure()
plt.plot(received.real, received.imag, '.', alpha=0.5)
plt.title("Obrócony sygnał (konstelacja przed korekcją)")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis('equal')
plt.show()
