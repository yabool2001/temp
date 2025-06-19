from modules import modulation , ops_packet , plot
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

def samples_2_bpsk_symbols ( samples , sps , beta , span ) :
    # Żeby zacząć samplować w odpowiednim miejscu
    barker_waveform = modulation.modulate_bpsk ( ops_packet.BARKER13 , sps , beta , span )
    plot.plot_complex_waveform ( barker_waveform )
    corr = np.correlate ( samples , barker_waveform , mode = 'valid' )
    peak_index = np.argmax ( np.abs ( corr ) )
    theta = np.angle ( corr[ peak_index ] )
    # 3. Korekta fazy i przycięcie
    aligned_rx_samples = samples[ peak_index:] * np.exp( -1j * theta)
    # 4. Próbkowanie co sps, po preambule
    start = len(barker_waveform)
    symbol_samples = aligned_rx_samples[start::sps]
    # 5. Decyzja BPSK
    return symbol_samples