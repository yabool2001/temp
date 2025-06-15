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
URI = "ip:192.168.2.1"

sdr = adi.Pluto ( URI )
sdr.rx_lo = int ( F_C )
sdr.sample_rate = int ( F_S )
sdr.rx_rf_bandwidth = int ( BW )
sdr.rx_buffer_size = int ( NUM_SAMPLES )
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

def bpsk_modulation ( bpsk_symbols ) :
    zeros = np.zeros_like ( bpsk_symbols )
    zeros[bpsk_symbols == -1] = 180
    x_radians = zeros*np.pi/180.0 # sin() and cos() takes in radians
    samples = np.cos(x_radians) + 1j*0 # this produces our QPSK complex symbols
    #samples = np.repeat(symbols, 4) # 4 samples per symbol (rectangular pulses) ale to robi rrc
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    pass

def rrc_filter(sps=4, beta=0.35, span=11):
    N = sps * span
    t = np.arange(-N//2, N//2 + 1, dtype=np.float64) / sps
    taps = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2)
    taps[np.isnan(taps)] = 0
    taps /= np.sqrt(np.sum(taps**2))
    return taps

def create_bpsk_symbols ( bits ) :
    return np.array ( [ 1.0 if bit else -1.0 for bit in bits ] , dtype = np.int64 )

def upfirdn(h, x, up=1, down=1):
    """Simple implementation of upfirdn (upsample, FIR filter, downsample)."""
    x_up = np.zeros(len(x) * up)
    x_up[::up] = x
    y = lfilter(h, 1.0, x_up)
    return y[::down]

def detect_barker13(rx_samples, sps=4, beta=0.35, span=11, threshold=0.8):
    """
    Detects Barker13 preamble in received samples from ADALM-Pluto.
    
    Args:
        rx_samples (np.array): Complex received samples (complex128).
        sps (int): Samples per symbol.
        beta (float): RRC filter roll-off factor.
        span (int): RRC filter span in symbols.
        threshold (float): Detection threshold (0.0 to 1.0).
    
    Returns:
        tuple: (correlation_peak, detected_position) or (None, None) if not found.
    """
    # 1. Apply RRC matched filtering
    filtered_samples = apply_rrc_filter(rx_samples, beta, sps, span)
    
    # 2. Generate the matched filter (Barker13 + RRC shaping)
    barker_symbols = BARKER13_BPSK
    rrc_taps = rrc_filter_v2(sps, beta, span)
    matched_filter = upfirdn(rrc_taps, barker_symbols, up=sps)
    matched_filter = matched_filter / np.linalg.norm(matched_filter)  # Normalize
    
    # 3. Compute cross-correlation (sliding matched filter)
    correlation = np.abs(correlate(filtered_samples, matched_filter, mode='valid'))
    correlation /= np.max(correlation)  # Normalize to [0, 1]
    
    # 4. Find the peak above threshold
    peak_pos = np.argmax(correlation)
    peak_value = correlation[peak_pos]
    
    if peak_value >= threshold:
        return peak_value, peak_pos
    else:
        return None, None

# ------------------------ KONFIGURACJA SDR ------------------------
def main():
    tx_bits = np.concatenate ( [ BARKER13_BITS , PADDING_BITS , PAYLOAD_BITS ] )
    print ( f"{tx_bits=}")
    tx_symbols = 1 - 2 * tx_bits
    print ( f"{tx_symbols=}")
    rrc_taps = rrc_filter ( SPS , RRC_BETA , RRC_SPAN )
    matched_filter = upfirdn ( rrc_taps , tx_symbols , up = SPS )
    tx_samples = matched_filter / np.linalg.norm ( matched_filter )  # Normalize
    pluto = sdr.init_pluto ( URI , F_C , F_S , BW )
    if verbose : help ( adi.Pluto.rx_output_type ) ; help ( adi.Pluto.gain_control_mode_chan0 ) ; help ( adi.Pluto.tx_lo ) ; help ( adi.Pluto.tx  )
    sdr.tx_cyclic ( tx_samples , pluto )
    
    # Clear buffer just to be safe
    for i in range ( 0 , 10 ) :
        raw_data = sdr.rx_samples ( pluto )
    
    # Receive samples
    peak_val, peak_pos = detect_barker13 ( sdr.rx_samples ( pluto ) )
    if peak_val is not None:
        print(f"BARKER13 detected at position {peak_pos} with correlation {peak_val:.2f}")
    else:
        print("BARKER13 not detected.")
    
    
    rx_samples = sdr.rx_samples_filtered ( pluto , SPS , RRC_BETA , RRC_SPAN )
    offset, theta = sdr.correlate_and_estimate_phase (rx_samples )
    rx_samples2 = rx_samples[offset:] * np.exp ( -1j * theta )
    acg_vaule = pluto._get_iio_attr ( 'voltage0' , 'hardwaregain' , False )
    # Stop transmitting
    sdr.stop_tx_cyclic ( pluto )

    csv_file_tx , csv_writer_tx = ops_file.open_and_write_samples_2_csv ( csv_filename_tx_waveform , tx_samples )
    csv_file_rx , csv_writer_rx = ops_file.open_and_write_samples_2_csv ( csv_filename_rx_waveform , rx_samples2 )
    ops_file.flush_samples_and_close_csv ( csv_file_tx )
    ops_file.flush_samples_and_close_csv ( csv_file_rx )
    ops_file.plot_samples ( csv_filename_tx_waveform )
    ops_file.plot_samples ( csv_filename_rx_waveform )
    print ( f"{acg_vaule=}" )

if __name__ == "__main__":
    main ()
