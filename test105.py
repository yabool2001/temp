import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules import ops_file

tx_samples_barker13_filename = "logs/tx_samples_barker13_clipped_74.csv"
rx_samples_filename = "logs/rx_samples_32768.csv"
f_s = 3000000              # częstotliwość próbkowania w Hz
sps = 4                     # samples per symbol

tx_samples_barker13 = ops_file.open_csv_and_load_np_complex128 ( tx_samples_barker13_filename )
rx_samples = ops_file.open_csv_and_load_np_complex128 ( rx_samples_filename )

import numpy as np
from scipy.signal import correlate, find_peaks
import plotly.express as px

def analyze_cfo(rx_samples, tx_samples_barker13, sps=4, fs=1e6, corr_threshold_factor=0.5, min_peak_distance_factor=0.5):
    """
    Analizuje sygnał RX w celu estymacji przesunięcia częstotliwości nośnej (CFO) dla każdej ramki.
    
    Parametry:
    - rx_samples: numpy array complex128 - odebrany sygnał.
    - tx_samples_barker13: numpy array complex128 - znany preambuła Barker13 (upsampled z SPS=4).
    - sps: int - samples per symbol (domyślnie 4).
    - fs: float - sample rate w Hz (np. 1e6 dla 1 MSPS; dostosuj do swojego setupu ADALM-PLUTO).
    - corr_threshold_factor: float - czynnik progu dla detekcji peaków korelacji (względem max).
    - min_peak_distance_factor: float - minimalna odległość między peakami względem długości preambuły.
    
    Zwraca:
    - cfo_estimates: lista estymowanych CFO w Hz dla każdej ramki.
    - frame_times: lista czasów startu ramek w sekundach.
    
    Rysuje wykres zmiany CFO w czasie za pomocą Plotly Express.
    """
    # Normalizacja preambuły i RX dla stabilności numerycznej
    tx_preamble = tx_samples_barker13 / np.linalg.norm(tx_samples_barker13)
    rx_norm = rx_samples / np.max(np.abs(rx_samples))  # Lekka normalizacja RX
    
    # Korelacja krzyżowa do znalezienia startów ramek (synchronizacja czasowa)
    corr = correlate(rx_norm, tx_preamble, mode='valid')
    corr_mag = np.abs(corr)
    
    # Znajdź peak'i korelacji - starty ramek
    min_distance = int(len(tx_preamble) * min_peak_distance_factor)
    peaks, properties = find_peaks(corr_mag, height=corr_threshold_factor * np.max(corr_mag), distance=min_distance)
    
    frame_starts = peaks
    cfo_estimates = []
    frame_times = []  # Czasy startu ramek w sekundach
    
    for start in frame_starts:
        end = start + len(tx_preamble)
        if end > len(rx_samples):
            continue  # Pomijanie niepełnych ramek na końcu
        
        rx_pream = rx_samples[start:end]
        
        # Usunięcie modulacji: mnożenie przez conj(preambuły)
        demod = rx_pream * np.conj(tx_samples_barker13)
        
        # Wyodrębnienie fazy i unwrap (odwijanie fazy, aby obsłużyć skoki > pi)
        phase = np.unwrap(np.angle(demod))
        
        # Wektor czasu w próbkach (dla normalized frequency)
        t_samples = np.arange(len(phase))
        
        # Liniowe dopasowanie: phase = 2 * pi * f_norm * t + phi
        if len(phase) < 2:
            continue  # Za krótki segment
        slope, _ = np.polyfit(t_samples, phase, 1)
        
        # Obliczenie normalized CFO (cycles per sample), potem w Hz
        f_norm = slope / (2 * np.pi)
        f_hz = f_norm * fs
        cfo_estimates.append(f_hz)
        
        # Czas startu ramki w sekundach
        frame_times.append(start / fs)
    
    # Rysowanie wykresu zmiany CFO w czasie za pomocą Plotly Express
    if cfo_estimates:
        fig = px.line(x=frame_times, y=cfo_estimates, markers=True,
                      labels={'x': 'Czas (s)', 'y': 'Przesunięcie CFO (Hz)'},
                      title='Zmiana przesunięcia częstotliwości nośnej w czasie')
        fig.update_traces(line=dict(width=2), marker=dict(size=8))
        fig.show()
    else:
        print("Nie wykryto żadnych ramek - sprawdź próg korelacji lub sygnał.")
    
    return cfo_estimates, frame_times

# Przykład użycia:
# Zakładając, że masz rx_samples i tx_samples_barker13 załadowane
# fs = 1000000  # Dostosuj do swojego sample rate w Hz (np. z ADALM-PLUTO)
# cfo_list, times = analyze_cfo(rx_samples, tx_samples_barker13, sps=4, fs=fs)