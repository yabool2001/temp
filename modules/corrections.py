from modules import modulation , ops_packet 
import numpy as np
from scipy.signal import correlate

def correlate_and_estimate_phase ( rx_samples ) :
    barker13_symbols = [ 1 if b == 0 else -1 for b in ops_packet.BARKER13 ]
    bpsk_waveform = np.array ( barker13_symbols , dtype = np.complex128 )
    # Korelacja w oknie
    corr = np.correlate ( rx_samples , bpsk_waveform , mode = 'valid' )
    max_index = np.argmax ( np.abs ( corr ) )
    peak_phase = np.angle ( corr[max_index] )
    return max_index , peak_phase

def phase_shift_corr ( samples ) :
    # Żeby I i Q się nie zmieniały
    offset , theta = correlate_and_estimate_phase ( samples )
    return samples[offset:] * np.exp ( -1j * theta )

def find_frame_start ( samples , sps ) :
    # Żeby samplować w odpowiednim miejscu
    barker = modulation.create_bpsk_symbols ( ops_packet.BARKER13 )
    barker_upsampled = np.zeros ( len ( barker ) * sps )
    barker_upsampled[::sps] = barker
    
    corr = np.abs ( correlate ( samples , barker_upsampled , mode = 'full' ) )
    peak_pos = np.argmax ( corr ) - len ( barker_upsampled ) + 1
    frame_start = peak_pos + ( len ( barker_upsampled ) // 2 )  # Środek preambuły
    return frame_start

def samples_2_bpsk_symbols ( samples , sps ) :
    frame_start = find_frame_start ( samples , sps )
    # Pobierz symbole (1 na symbol)
    symbols = samples[ frame_start::sps ]
    # Demodulacja BPSK (proste dekodowanie fazy)
    bpsk_symbols = np.sign ( np.real ( symbols ) )  # BPSK: 1 dla Re>0, -1 dla Re<0
    
    return bpsk_symbols