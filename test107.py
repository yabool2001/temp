import numpy as np
import pandas as pd
import os

from modules import plot

filename = "logs/rx_samples_test103_raw_data.csv"

script_filename = os.path.basename ( __file__ )

# =============================================
# Parametry symulacji
# =============================================
N = 4000                    # liczba symboli
EbN0_dB = 12                # stosunek Eb/N0 w dB (im wyższy, tym mniej szumu)
bits_per_symbol = 2
EsN0_dB = EbN0_dB + 10*np.log10(bits_per_symbol)  # Es/N0

# =============================================
# Idealne symbole QPSK (energia symbolu = 1)
# =============================================
constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

# Losowe bity → symbole
bits = np.random.randint(0, 4, N)
symbols = constellation[bits]

# =============================================
# Dodanie szumu AWGN
# =============================================
snr_linear = 10**(EsN0_dB/10)
sigma = np.sqrt(1/(2*snr_linear))       # odchylenie standardowe szumu (dla I i Q osobno)

noise = sigma * (np.random.randn(N) + 1j*np.random.randn(N))
received = (symbols + noise).astype ( np.complex128 , copy = False )
plot.complex_symbols ( received , script_filename + " Received QPSK symbols with AWGN" )
