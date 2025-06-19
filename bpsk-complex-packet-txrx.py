import adi
#import csv
import numpy as np
#from scipy.signal import lfilter
#import time
#import zlib
#from scipy.signal import upfirdn
import pandas as pd
import plotly.express as px
import matplotlib as plt

from modules import filters , sdr , ops_packet , ops_file , modulation , corrections
#from modules.rrc import rrc_filter
#from modules.clock_sync import polyphase_clock_sync
 


# App settings
verbose = True
verbose = False

# Inicjalizacja plików CSV
csv_filename_tx_waveform = "complex_tx_waveform.csv"
csv_filename_rx_waveform = "complex_rx_waveform.csv"


# ------------------------ PARAMETRY KONFIGURACJI ------------------------
F_C = 2900e6     # częstotliwość nośna [Hz]
#F_S = 2e6     # częstotliwość próbkowania [Hz] >= 521e3 && <
BW  = 1_000_000         # szerokość pasma [Hz]
#F_S = 521100     # częstotliwość próbkowania [Hz] >= 521e3 && <
F_S = BW * 3 if ( BW * 3 ) >= 521100 and ( BW * 3 ) <= 61440000 else 521100
print (f"{F_S=}")
SPS = 4                 # próbek na symbol
TX_GAIN = -10.0
URI = "ip:192.168.2.1"
#URI = "usb:"

RRC_BETA = 0.35         # roll-off factor
RRC_SPAN = 11           # długość filtru RRC w symbolach
CYCLE_MS = 10           # opóźnienie między pakietami [ms]; <0 = liczba powtórzeń

PAYLOAD = [ 0x0F , 0x0F , 0x0F , 0x0F ]  # można zmieniać dynamicznie

# ------------------------ KONFIGURACJA SDR ------------------------
def main():
    packet_bits = ops_packet.create_packet_bits ( PAYLOAD )
    tx_bpsk_symbols = modulation.create_bpsk_symbols ( packet_bits )
    tx_samples = filters.apply_tx_rrc_filter ( tx_bpsk_symbols , SPS , RRC_BETA , RRC_SPAN , True )
    pluto = sdr.init_pluto ( URI , F_C , F_S , BW )
    if verbose : help ( adi.Pluto.rx_output_type ) ; help ( adi.Pluto.gain_control_mode_chan0 ) ; help ( adi.Pluto.tx_lo ) ; help ( adi.Pluto.tx  )
    sdr.tx_cyclic ( tx_samples , pluto )
    
    # Clear buffer just to be safe
    for i in range ( 0 , 10 ) :
        raw_data = sdr.rx_samples ( pluto )
    # Receive samples
    rx_samples = sdr.rx_samples ( pluto  )
    rx_samples_filtered = filters.apply_rrc_rx_filter ( rx_samples , SPS , RRC_BETA , RRC_SPAN , False ) # W przyszłości rozważyć implementację tego filtrowania sampli rx
    #offset, theta = sdr.correlate_and_estimate_phase ( rx_samples )
    rx_samples_phase_corrected = corrections.phase_shift_corr ( rx_samples )
    bpsk_symbols = corrections.samples_2_bpsk_symbols ( rx_samples_phase_corrected , SPS )

    acg_vaule = pluto._get_iio_attr ( 'voltage0' , 'hardwaregain' , False )
    # Stop transmitting
    sdr.stop_tx_cyclic ( pluto )

    csv_file_tx , csv_writer_tx = ops_file.open_and_write_samples_2_csv ( csv_filename_tx_waveform , tx_samples )
    csv_file_rx , csv_writer_rx = ops_file.open_and_write_samples_2_csv ( csv_filename_rx_waveform , rx_samples_phase_corrected )
    ops_file.flush_samples_and_close_csv ( csv_file_tx )
    ops_file.flush_samples_and_close_csv ( csv_file_rx )
    ops_file.plot_samples ( csv_filename_tx_waveform )
    ops_file.plot_samples ( csv_filename_rx_waveform )
    print ( f"{acg_vaule=}" )

if __name__ == "__main__":
    main ()
