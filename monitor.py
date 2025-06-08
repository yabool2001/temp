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

from modules.rrc import rrc_filter
from modules.clock_sync import polyphase_clock_sync

# Import lokalnego modułu RRC
sys.path.append ( "modules" )

# App settings
#verbose = True
verbose = False

# Parametry SDR
RX_GAIN = 0.1

# Parametry RF 
F_C = 2_900_000_000
#F_S = 521_100
F_S = 3_000_000
BW  = 1_000_000
NUM_SAMPLES = 32768
NUM_POINTS = 16384
SPS = 4
GAIN_CONTROL = "fast_attack"
#GAIN_CONTROL = "manual"

# Parametry filtru RRC
RRC_BETA = 0.35 # Excess_bw
RRC_SPS = SPS   # Samples per symbol
RRC_SPAN = 11

NFILTS = 32

# Inicjalizacja Pluto SDR
sdr = adi.Pluto ( uri = "ip:192.168.2.1" )
sdr.rx_lo = int ( F_C )
sdr.sample_rate = int ( F_S )
sdr.rx_rf_bandwidth = int ( BW )
sdr.rx_buffer_size = int ( NUM_SAMPLES )
sdr.gain_control_mode_chan0 = GAIN_CONTROL
sdr.rx_hardwaregain_chan0 = float ( RX_GAIN )
sdr.rx_output_type = "SI"
if verbose : help ( adi.Pluto.rx_output_type ) ; help ( adi.Pluto.gain_control_mode_chan0 ) ; help ( adi.Pluto.tx_lo ) ; help ( adi.Pluto.tx  )

# Inicjalizacja pliku CSV
csv_filename_raw = "complex_rx_raw.csv"
csv_filename_filtered = "complex_rx_filtered.csv"
#csv_file_raw = open ( csv_filename_raw , mode = "w" , newline = '' )
csv_file_filtered = open ( csv_filename_filtered , mode = "w" , newline = '' )
#csv_writer_raw = csv.writer ( csv_file_raw )
csv_writer_filtered = csv.writer ( csv_file_filtered )
#csv_writer_raw.writerow ( [ "timestamp" , "real" , "imag" ] )
csv_writer_filtered.writerow ( [ "timestamp" , "real" , "imag" ] )

# Inicjalizacja filtry RRC
rrc_taps = rrc_filter ( RRC_BETA , RRC_SPS , RRC_SPAN )

print ( "Rozpoczynam zbieranie danych... (wciśnij Esc, aby zakończyć)" )
t0 = time.time ()
try :
    while True:
        if keyboard.is_pressed ('esc') :
            print ( "Naciśnięto Esc. Kończę zbieranie." )
            break
        raw_samples = sdr.rx ()
        filtered_samples = lfilter ( rrc_taps , 1.0 , raw_samples )
        ts = time.time () - t0
        synced_samples = polyphase_clock_sync ( filtered_samples , sps = SPS, nfilts = NFILTS , excess_bw = RRC_BETA )
        if verbose : acg_vaule = sdr._get_iio_attr ( 'voltage0' , 'hardwaregain' , False ) ; print ( f"{acg_vaule=}" )
        for sample in synced_samples :
            csv_writer_filtered.writerow ( [ ts , sample.real , sample.imag ] )
        csv_file_filtered.flush ()
        if verbose : print ( f"{type ( filtered_samples )=}, {filtered_samples.dtype=}" ) ; print ( f"{filtered_samples=}" )

except KeyboardInterrupt :
    print ( "Zakończono ręcznie (Ctrl+C)" )

finally:
    csv_file_filtered.close ()

# Wczytanie danych i wyświetlenie wykresu w Plotly
print("Rysuję wykres...")
df = pd.read_csv(csv_filename_filtered)
# Zbuduj sygnał zespolony (opcjonalnie, jeśli chcesz jako 1D)
signal = df["real"].values + 1j * df["imag"].values
# Przygotuj dane do wykresu
df["index"] = df.index
# Wykres Plotly Express – wersja liniowa z filtrowanym sygnałem
fig = px.line(df, x="index", y="real", title="Sygnał BPSK po filtracji RRC: I i Q")
fig.add_scatter(x=df["index"], y=df["imag"], mode="lines", name="Q (imag filtrowane)", line=dict(dash="dash"))
fig.update_layout(
    xaxis_title="Numer próbki",
    yaxis_title="Amplituda",
    xaxis=dict(rangeslider_visible=True),
    legend=dict(x=0.01, y=0.99),
    height=500
)
fig.show()

# Wczytanie danych i wyświetlenie wykresu w Plotly
print ( "Rysuję wykres..." )

# Wykres Plotly Express – wersja liniowa z filtrem
fig = px.line(df, x="timestamp", y="real", title="Sygnał BPSK po filtracji RRC I i Q na osi czasu")
fig.add_scatter(x=df["timestamp"], y=df["imag"], mode="markers", name="Q (imag filtrowane)", line=dict(dash="dash"))

fig.update_layout(
    xaxis_title="Czas względny [s]",
    yaxis_title="Amplituda",
    xaxis=dict(
        rangeslider_visible=True
    ),
    yaxis=dict(
        autorange=True
    ),
    legend=dict(x=0.01, y=0.99)
)


fig.show ()