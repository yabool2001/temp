import numpy as np
from modules import filters

from scipy.signal import upfirdn

def bpsk_modulation ( bpsk_symbols ) :
    zeros = np.zeros_like ( bpsk_symbols )
    zeros[bpsk_symbols == -1] = 180
    x_radians = zeros*np.pi/180.0 # sin() and cos() takes in radians
    samples = np.cos(x_radians) + 1j*0 # this produces our QPSK complex symbols
    #samples = np.repeat(symbols, 4) # 4 samples per symbol (rectangular pulses) ale to robi rrc
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    # plot_tx_waveform ( samples )
    pass

def create_bpsk_symbols ( bits ) :
    return np.array ( [ 1.0 if bit else -1.0 for bit in bits ] , dtype = np.int64 )

def modulate_bpsk ( bits , sps = 4 , beta = 0.35 , span = 11 ) :
    symbols = create_bpsk_symbols ( bits )
    rrc = filters.rrc_filter_v4 ( sps , beta , span )
    shaped = upfirdn ( rrc , symbols , up = sps )
    return ( shaped + 0j ).astype ( np.complex128 )