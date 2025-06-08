import adi
import csv
import numpy as np
from scipy.signal import lfilter
import time
import zlib
from scipy.signal import upfirdn
import pandas as pd
import plotly.express as px

from modules.rrc import rrc_filter
from modules.clock_sync import polyphase_clock_sync

# App settings
#verbose = True
verbose = False

# Inicjalizacja pliku CSV
csv_filename_waveform = "complex_tx_waveform.csv"
csv_file_waveform = open ( csv_filename_waveform , mode = "w" , newline = '' )
csv_writer_waveform = csv.writer ( csv_file_waveform )
csv_writer_waveform.writerow ( [ "timestamp" , "real" , "imag" ] )


# ------------------------ PARAMETRY KONFIGURACJI ------------------------
F_C = 2_900_000_000     # częstotliwość nośna [Hz]
F_S = 521_100           # częstotliwość próbkowania [Hz]
F_S = 3_000_000         # częstotliwość próbkowania [Hz]
BW  = 1_000_000         # szerokość pasma [Hz]
SPS = 4                 # próbek na symbol
TX_ATTENUATION = 10.0
# NUM_SAMPLES = 32768
NUM_SAMPLES = 1000
GAIN_CONTROL = "fast_attack"
#GAIN_CONTROL = "manual"

RRC_BETA = 0.35         # roll-off factor
RRC_SPAN = 11           # długość filtru RRC w symbolach
CYCLE_MS = 10           # opóźnienie między pakietami [ms]; <0 = liczba powtórzeń

# Inicjalizacja Pluto SDR
sdr = adi.Pluto ( uri = "usb:" )
sdr.tx_destroy_buffer ()
sdr.tx_cyclic_buffer = False
sdr.rx_lo = int ( F_C )
sdr.sample_rate = int ( F_S )
sdr.rx_rf_bandwidth = int ( BW )
sdr.rx_buffer_size = int ( NUM_SAMPLES )
sdr.gain_control_mode_chan0 = GAIN_CONTROL
sdr.tx_hardwaregain_chan0 = float ( -TX_ATTENUATION )
sdr.rx_output_type = "SI"
if verbose : help ( adi.Pluto.rx_output_type ) ; help ( adi.Pluto.gain_control_mode_chan0 ) ; help ( adi.Pluto.tx_lo ) ; help ( adi.Pluto.tx  )



# ------------------------ DANE DO MODULACJI ------------------------
header = [ 0xAA , 0xAA , 0xAA , 0xAA ]
payload = [ 0x0F , 0x0F , 0x0F , 0x0F ]  # można zmieniać dynamicznie

def create_packet(header, payload):
    length_byte = [len(payload) - 1]
    crc32 = zlib.crc32(bytes(payload))
    crc_bytes = list(crc32.to_bytes(4, 'big'))
    return header + length_byte + payload + crc_bytes

def bits_to_bpsk(bits):
    return np.array([1.0 if bit else -1.0 for bit in bits], dtype=np.float32)

def rrc_filter(beta, sps, span):
    N = sps * span
    t = np.arange(-N//2, N//2 + 1, dtype=np.float32) / sps
    taps = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2)
    taps[np.isnan(taps)] = 0
    taps /= np.sqrt(np.sum(taps**2))
    return taps

def modulate_packet(packet, sps, beta, span):
    bits = np.unpackbits(np.array(packet, dtype=np.uint8))
    symbols = bits_to_bpsk(bits)
    rrc = rrc_filter(beta, sps, span)
    shaped = upfirdn(rrc, symbols, up=sps)
    wf = (shaped + 0j).astype(np.complex64)
    return wf

def shape ( shaped ):
    return (shaped + 0j).astype(np.complex64)

def write_waveform ( waveform , file, writer , t0 ) :
    for sample in waveform :
        writer.writerow ( [ time.time () - t0 , sample.real , sample.imag ] )
    file.flush ()

def transmit ( waveform , sdr ) :
    try :
        while True :
            sdr.tx ( waveform )
            time.sleep( 10 / 1000.0)
    except KeyboardInterrupt :
        print ( "Zakończono ręcznie (Ctrl+C)" )
    finally:
        sdr.tx_destroy_buffer()
        sdr.tx_cyclic_buffer = False
        print ( f"{sdr.tx_cyclic_buffer=}" )

def plot_tx_waveform ( filename ) :
    # Wczytanie danych i wyświetlenie wykresu w Plotly
    print ( "Rysuję wykres..." )
    df = pd.read_csv ( filename )
    # Zbuduj sygnał zespolony (opcjonalnie, jeśli chcesz jako 1D)
    signal = df["real"].values + 1j * df["imag"].values
    # Przygotuj dane do wykresu
    df["index"] = df.index
    # Wykres Plotly Express – wersja liniowa z filtrowanym sygnałem
    fig = px.line(df, x="index", y="real", title="Sygnał BPSK raw: I i Q")
    fig.add_scatter(x=df["index"], y=df["imag"], mode="lines", name="Q (imag filtrowane)", line=dict(dash="dash"))
    fig.update_layout(
        xaxis_title="Numer próbki",
        yaxis_title="Amplituda",
        xaxis=dict(rangeslider_visible=True),
        legend=dict(x=0.01, y=0.99),
        height=500
    )
    fig.show()

# ------------------------ KONFIGURACJA SDR ------------------------
def main():
    packet = create_packet ( header , payload )
    print (packet )
    waveform = modulate_packet ( packet , SPS , RRC_BETA , RRC_SPAN )
    t0 = time.time ()
    write_waveform ( waveform , csv_file_waveform , csv_writer_waveform , t0 )
    transmit ( waveform , sdr )
    plot_tx_waveform ( csv_filename_waveform )

if __name__ == "__main__":
    main ()
