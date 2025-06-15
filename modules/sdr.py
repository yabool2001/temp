import adi
from modules import filters
import numpy as np

TX_GAIN = -10
RX_GAIN = 70
GAIN_CONTROL = "slow_attack"
#GAIN_CONTROL = "fast_attack"
#GAIN_CONTROL = "manual"
NUM_SAMPLES = 32768


def init_pluto ( uri , f_c , f_s , bw ) :
    sdr = adi.Pluto ( uri )
    sdr.rx_lo = int ( f_c )
    sdr.sample_rate = int ( f_s )
    sdr.rx_rf_bandwidth = int ( bw )
    sdr.rx_buffer_size = int ( NUM_SAMPLES )
    sdr.tx_hardwaregain_chan0 = float ( TX_GAIN )
    sdr.gain_control_mode_chan0 = GAIN_CONTROL
    sdr.rx_hardwaregain_chan0 = float ( RX_GAIN )
    sdr.rx_output_type = "SI"
    sdr.tx_destroy_buffer ()
    sdr.tx_cyclic_buffer = False
    return sdr


def tx_cyclic ( samples , sdr ) :
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    sdr.tx_cyclic_buffer = True
    sdr.tx ( samples )

def stop_tx_cyclic ( sdr ) :
    sdr.tx_destroy_buffer ()
    sdr.tx_cyclic_buffer = False
    print ( f"{sdr.tx_cyclic_buffer=}" )

def rx_samples ( sdr ) :
    return sdr.rx ()

def rx_samples_filtered ( sdr , sps = 8 , beta = 0.35 , span = 11 ) :
    return filters.apply_rrc_filter ( rx_samples ( sdr ) , sps , beta , span )

def correlate_and_estimate_phase ( rx_samples ) :
    BARKER13 = [ 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 ]
    barker13_symbols = [1 if b == 0 else -1 for b in BARKER13]
    bpsk_waveform = np.array(barker13_symbols, dtype=np.complex128)
    # Korelacja w oknie
    corr = np.correlate ( rx_samples , bpsk_waveform , mode = 'valid' )
    max_index = np.argmax ( np.abs ( corr ) )
    peak_phase = np.angle ( corr[max_index] )
    return max_index , peak_phase