# Idealne ustawienie Attenuation TX1 (dB) = 1.2 w bloku "Pluto SDR Sink"
# W Rx ACG = Fast Attack

# Do zrobienia:
# 1. Sprawdzić jak działa bez llfilter

import adi
import csv
import keyboard
import numpy as np
from scipy.signal import lfilter
import sys
import time
import pandas as pd
import plotly.express as px

# Inicjalizacja pliku CSV
#csv_filename_waveform = "complex_tx_waveform.csv"
csv_filename_waveform = "logs/rx_samples.csv"

# Wczytanie danych i wyświetlenie wykresu w Plotly
print ( "Rysuję wykres..." )
df = pd.read_csv(csv_filename_waveform)
# Zbuduj sygnał zespolony (opcjonalnie, jeśli chcesz jako 1D)
signal = df["real"].values + 1j * df["imag"].values
# Przygotuj dane do wykresu
df["index"] = df.index
# Wykres Plotly Express – wersja liniowa z filtrowanym sygnałem
fig = px.line(df, x="index", y="real", title="Sygnał BPSK raw: I i Q. Size " + f"{signal.size}")
fig.add_scatter(x=df["index"], y=df["imag"], mode="lines", name="Q (imag filtrowane)", line=dict(dash="dash"))
fig.update_layout(
    xaxis_title="Numer próbki",
    yaxis_title="Amplituda",
    xaxis=dict(rangeslider_visible=True),
    legend=dict(x=0.01, y=0.99),
    height=500
)
fig.show()
'''
# Wczytanie danych i wyświetlenie wykresu w Plotly
print ( "Rysuję wykres..." )
df = pd.read_csv(csv_filename_waveform)
# Zbuduj sygnał zespolony (opcjonalnie, jeśli chcesz jako 1D)
signal = df["real"].values + 1j * df["imag"].values
# Przygotuj dane do wykresu
df["index"] = df.index
# Wykres Plotly Express – wersja liniowa z filtrowanym sygnałem
fig = px.line(df, x="timestamp", y="real", title="Sygnał BPSK raw: I i Q")
fig.add_scatter(x=df["timestamp"], y=df["imag"], mode="lines", name="Q (imag filtrowane)", line=dict(dash="dash"))
fig.update_layout(
    xaxis_title="ts próbki",
    yaxis_title="Amplituda",
    xaxis=dict(rangeslider_visible=True),
    legend=dict(x=0.01, y=0.99),
    height=500
)
fig.show()
'''