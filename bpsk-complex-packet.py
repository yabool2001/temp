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
TX_GAIN = -10.0
NUM_SAMPLES = 32768
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
sdr.tx_hardwaregain_chan0 = float ( TX_GAIN )
sdr.gain_control_mode_chan0 = GAIN_CONTROL
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

def bits_to_bpsk ( bits ) :
    return np.array ( [ 1.0 if bit else -1.0 for bit in bits ] , dtype = np.int64 )

def rrc_filter(beta, sps, span):
    N = sps * span
    t = np.arange(-N//2, N//2 + 1, dtype=np.float64) / sps
    taps = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2)
    taps[np.isnan(taps)] = 0
    taps /= np.sqrt(np.sum(taps**2))
    return taps

def create_bpsk_symbols ( packet ) :
    bits = np.unpackbits ( np.array ( packet , dtype = np.uint8 ) )
    return bits_to_bpsk ( bits )

def modulate_packet ( symbols , sps , beta , span ) :
    rrc = rrc_filter(beta, sps, span)
    shaped = upfirdn(rrc, symbols, up=sps)
    wf = (shaped + 0j).astype(np.complex128)
    return wf

def bpsk_modulation ( bpsk_symbols ) :
    zeros = np.zeros_like ( bpsk_symbols )
    zeros[bpsk_symbols == -1] = 180
    x_radians = zeros*np.pi/180.0 # sin() and cos() takes in radians
    samples = np.cos(x_radians) + 1j*0 # this produces our QPSK complex symbols
    #samples = np.repeat(symbols, 4) # 4 samples per symbol (rectangular pulses) ale to robi rrc
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    plot_tx_waveform ( samples )
    pass

def shape ( shaped ):
    return (shaped + 0j).astype(np.complex128)

def write_waveform ( waveform , file, writer , t0 ) :
    for sample in waveform :
        writer.writerow ( [ time.time () - t0 , sample.real , sample.imag ] )
    file.flush ()

def transmit_2_pluto ( waveform , sdr ) :
    waveform *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    try :
        while True :
            sdr.tx ( waveform )
            time.sleep( CYCLE_MS / 1000.0)
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
    fig = px.line(df, x="index", y="real", title="Sygnał Tx BPSK raw: I i Q")
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
    print ( packet )
    bpsk_symbols = create_bpsk_symbols ( packet )
    #waveform = modulate_packet ( bpsk_symbols , SPS , RRC_BETA , RRC_SPAN )
    waveform = bpsk_modulation ( bpsk_symbols )
    t0 = time.time ()
    write_waveform ( waveform , csv_file_waveform , csv_writer_waveform , t0 )
    plot_tx_waveform ( csv_filename_waveform )
    transmit_2_pluto ( waveform , sdr )

if __name__ == "__main__":
    main ()
