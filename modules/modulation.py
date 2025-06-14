import numpy as np
from modules import filters

from scipy.signal import upfirdn

def create_bpsk_symbols ( bits ) :
    return np.array ( [ 1.0 if bit else -1.0 for bit in bits ] , dtype = np.int64 )

def modulate_bpsk ( packet , sps , beta , span ) :
    symbols = create_bpsk_symbols ( packet )
    rrc = filters.rrc_filter_v2 ( sps , beta , span )
    shaped = upfirdn ( rrc , symbols , up = sps )
    return ( shaped + 0j ).astype ( np.complex128 )