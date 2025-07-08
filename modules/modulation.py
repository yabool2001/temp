import numpy as np
from modules import ops_packet , filters , plot

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

def upsample_symbols ( symbols: np.ndarray , sps: int ) -> np.ndarray :
    """
    Zwraca ciąg zinterpolowany przez zero-stuffing (impulse upsampling).
    """
    upsampled = np.zeros ( len ( symbols ) * sps , dtype = symbols.dtype )
    upsampled[ ::sps ] = symbols
    return upsampled

def bpsk_symbols_2_bits ( symbols ) :
    return ( symbols.real > 0 ).astype ( int )

def signal_correlation(samples, lag=1):
    """
    Detekcja obecności sygnału na podstawie korelacji między sąsiednimi próbkami.

    samples  : kompleksowe próbki z RX (np. z PyADI)
    lag      : opóźnienie w próbkach (1 = sąsiednie)

    Zwraca: (float) wartość korelacji (0–1 przy szumie, wyżej przy sygnale)
    """
    if len(samples) <= lag:
        return 0.0

    x = samples[:-lag]
    y = samples[lag:]

    corr = np.vdot(x, y)
    norm = np.sqrt(np.vdot(x, x).real * np.vdot(y, y).real)

    corr_norm = np.abs(corr) / (norm + 1e-12)
    return corr_norm

def get_barker13_bpsk_samples ( sps , rrc_beta , rrc_span , clipped = False ) :
    symbols = create_bpsk_symbols ( ops_packet.BARKER13_BITS )
    samples = filters.apply_tx_rrc_filter ( symbols , sps , rrc_beta , rrc_span , True )
    if clipped :
        samples = samples[ :72 ]
        #samples = samples[ 18:72 ]
    #plot.plot_complex_waveform ( samples , "  barker13 samples")
    return samples

def zero_quadrature ( samples ) :
    """
    Zeruje składową Q (urojoną) sygnału zespolonego, pozostawiając tylko składową I (rzeczywistą).
    """
    return np.real ( samples ) + 0j

def normalized_cross_correlation ( signal , template ) :
    template = (template - np.mean(template)) / np.std(template)
    n = len(template)
    corr = []

    for i in range(len(signal) - n + 1):
        window = signal[i:i+n]
        if np.std(window) == 0:  # unikanie dzielenia przez 0
            corr.append(0)
            continue
        window_norm = (window - np.mean(window)) / np.std(window)
        corr.append(np.sum(window_norm * template))

    return np.array(corr)

