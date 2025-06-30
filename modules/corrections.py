import matplotlib.pyplot as plt
from modules import modulation , ops_packet , plot
import numpy as np
from scipy.signal import correlate , butter, lfilter

def pll ( rx_samples , fs , freq_offset_initial ) :
    phase_estimate = 0.0
    freq_estimate = freq_offset_initial
    loop_bw = 2 * np.pi * 100 / fs  # szerokość pasma pętli (np. 50 Hz)
    alpha = loop_bw
    #alpha = 0.132 # wstawiłem z costas loop z pysdr ale okazało się klapą
    beta = alpha**2 / 4
    #beta = 0.00932 # wstawiłem z costas loop z pysdr ale okazało się klapą
    corrected_samples = np.zeros_like ( rx_samples , dtype = complex )

    for n, sample in enumerate ( rx_samples ) :
        # Korekcja aktualną estymacją
        corrected_samples[n] = sample * np.exp ( -1j * phase_estimate )

        # Błąd fazy z demodulowanego symbolu BPSK
        error = np.sign ( np.real ( corrected_samples[n] ) ) * np.imag ( corrected_samples[n] )

        # Aktualizacja estymacji częstotliwości i fazy
        freq_estimate += beta * error
        phase_estimate += freq_estimate + alpha * error

    return corrected_samples

# Korekcja offsetu fazowego:
def correct_phase_offset ( rx_samples , preamble_samples ) :

    correlation = np.correlate(rx_samples, preamble_samples , mode='valid')
    phase_offset = np.angle(correlation[np.argmax(np.abs(correlation))])
    return rx_samples * np.exp(-1j * phase_offset)

def correct_phase_offset_v2 ( rx_samples , preamble_samples ) :
    correlation = np.correlate(rx_samples, preamble_samples , mode='valid')
    max_corr = correlation[np.argmax(np.abs(correlation))]
    phase_offset = np.angle(max_corr)

    # Sprawdzenie odbicia fazy o 180°
    if np.real(max_corr) < 0:
        phase_offset += np.pi  # obrót o dodatkowe 180°

    return rx_samples * np.exp(-1j * phase_offset)

def correct_phase_offset_v3 ( samples , preamble_samples ) :
    # Korelacja bez sprzężenia (dla detekcji offsetu fazowego)
    correlation = np.correlate( samples, preamble_samples, mode='valid')
    max_corr = correlation[np.argmax(np.abs(correlation))]
    phase_offset = np.angle(max_corr)

    # Korekcja rotacji fazowej
    rx_corrected = samples * np.exp(-1j * phase_offset)

    # Detekcja lustrzanego odbicia (poprzez porównanie energii korelacji dla oryginalnej i sprzężonej preambuły)
    corr_normal = np.max(np.abs(np.correlate(rx_corrected, preamble_samples, mode='valid')))
    corr_conj = np.max(np.abs(np.correlate(rx_corrected, np.conj(preamble_samples), mode='valid')))

    if corr_conj > corr_normal:
        rx_corrected = np.conj(rx_corrected)

    return rx_corrected

def iq_balance ( samples ) :
    I = np.real ( samples )
    Q = np.imag ( samples )
    Q -= np.mean ( Q )
    scale = np.std ( I ) / np.std ( Q )
    Q *= scale
    return I + 1j * Q

# Pipeline kompensacji CFO, fazy, IQ
def full_compensation ( samples, fs , preamble_samples ) :
    
    # 1. CFO compensation (PLL-based)
    rx_pll_corrected = pll ( samples, fs, freq_offset_initial=0.0)

    # 2. Korekcja offsetu fazowego (na podstawie korelacji z wzorcem)
    rx_phase_corrected = correct_phase_offset_v3 ( rx_pll_corrected , preamble_samples )

    # 3. IQ imbalance (opcjonalne, ale zalecane)
    rx_final_corrected = iq_balance(rx_phase_corrected)

    return rx_final_corrected

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

def simple_correlation ( rx_samples , preamble_samples ) :
    # Korelacja w oknie
    corr = np.correlate ( rx_samples , preamble_samples , mode = 'valid' )
    max_index = np.argmax ( np.abs ( corr ) )
    peak_phase = np.angle ( corr[max_index] )
    return rx_samples[max_index:] * np.exp ( -1j * peak_phase )

def costas_loop ( samples , f_s ) :
    N = len ( samples )
    phase = 0
    freq = 0
    # These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
    alpha = 0.132
    beta = 0.00932
    out = np.zeros ( N , dtype = np.complex64 )
    freq_log = []
    for i in range ( N ) :
        out[i] = samples[i] * np.exp ( -1j * phase ) # adjust the input sample by the inverse of the estimated phase offset
        error = np.real ( out[i] ) * np.imag ( out[i] ) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)

        # Advance the loop (recalc phase and freq offset)
        freq += ( beta * error )
        freq_log.append ( freq * f_s /  ( 2 * np.pi ) ) # convert from angular velocity to Hz for logging
        phase += freq + (alpha * error)

        # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
        while phase >= 2 * np.pi :
            phase -= 2 * np.pi
        while phase < 0 :
            phase += 2 * np.pi

    # Plot freq over time to see how long it takes to hit the right offset
    plt.plot ( freq_log , '.-' )
    plt.show ()
    return out