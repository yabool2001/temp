import adi
#import csv
import numpy as np
#from scipy.signal import lfilter
#import time
import zlib
#from scipy.signal import upfirdn
import pandas as pd
import plotly.express as px
from scipy.signal import lfilter, correlate
from scipy.signal.windows import hamming

from modules import sdr , ops_packet , ops_file , modulation

# Inicjalizacja plików CSV
csv_filename_tx_waveform = "complex_tx_waveform.csv"
csv_filename_rx_waveform = "complex_rx_waveform.csv"


# ------------------------ PARAMETRY KONFIGURACJI ------------------------
F_C = 2900e6     # częstotliwość nośna [Hz]
#F_S = 2e6     # częstotliwość próbkowania [Hz] >= 521e3 && <
F_S = 521100     # częstotliwość próbkowania [Hz] >= 521e3 && <
BW  = 1_000_000         # szerokość pasma [Hz]
SPS = 4                 # próbek na symbol
TX_GAIN = -10.0
RX_GAIN = 70.0
GAIN_CONTROL = "slow_attack"
URI = "ip:192.168.2.1"

sdr = adi.Pluto ( URI )
sdr.rx_lo = int ( F_C )
sdr.sample_rate = int ( F_S )
sdr.rx_rf_bandwidth = int ( BW )
sdr.tx_hardwaregain_chan0 = float ( TX_GAIN )
sdr.gain_control_mode_chan0 = GAIN_CONTROL
sdr.rx_hardwaregain_chan0 = float ( RX_GAIN )
sdr.rx_output_type = "SI"
sdr.tx_destroy_buffer ()
sdr.tx_cyclic_buffer = False

RRC_BETA = 0.35         # roll-off factor
RRC_SPAN = 11           # długość filtru RRC w symbolach
CYCLE_MS = 10           # opóźnienie między pakietami [ms]; <0 = liczba powtórzeń

BARKER13_BITS = np.array ( [ 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 ] , dtype = np.float64 )
PADDING_BITS = np.array ( [ 0 , 0 , 0 ] , dtype = np.float64 )
PAYLOAD_BITS = np.array ( [ 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 ] , dtype = np.float64 )


def main():
    tx_bits = np.concatenate ( [ BARKER13_BITS , PADDING_BITS , PAYLOAD_BITS ] )
    print ( f"{tx_bits=}")
    tx_symbols = 1 - 2 * tx_bits
    print ( f"{tx_symbols=}")
    tx_symbols = np.repeat ( tx_symbols , SPS )
    sdr.tx_cyclic_buffer = True
    sdr.tx ( tx_symbols )
    
    # Clear buffer just to be safe
    for i in range ( 0 , 10 ) :
        raw_data = sdr.rx ()
    
    # Receive samples
    acg_vaule = sdr._get_iio_attr ( 'voltage0' , 'hardwaregain' , False )
    rx_samples = sdr.rx ()
    # Stop transmitting
    sdr.tx_destroy_buffer ()

    csv_file_tx , csv_writer_tx = ops_file.open_and_write_samples_2_csv ( csv_filename_tx_waveform , tx_symbols )
    csv_file_rx , csv_writer_rx = ops_file.open_and_write_samples_2_csv ( csv_filename_rx_waveform , rx_samples )
    ops_file.flush_samples_and_close_csv ( csv_file_tx )
    ops_file.flush_samples_and_close_csv ( csv_file_rx )
    ops_file.plot_samples ( csv_filename_tx_waveform )
    ops_file.plot_samples ( csv_filename_rx_waveform )
    print ( f"{acg_vaule=}" )

if __name__ == "__main__":
    main ()
