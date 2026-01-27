import matplotlib.pyplot as plt
from modules import modulation , ops_packet , plot , sdr
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.signal import correlate , butter, lfilter
from numba import jit
import time as t
import tomllib

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

def estimate_cfo_drit ( samples , f_s ) :
    N = len(samples)
    # --- DRYFT FAZY ---
    phases = np.angle(samples[1:] * np.conj(samples[:-1]))  # Δφ
    time_axis = np.arange(1, N) / f_s
    delta_phi_unwrapped = np.unwrap(phases)
    instantaneous_freq = delta_phi_unwrapped * f_s / (2 * np.pi)  # Hz

    # --- WYNIKI ---
    print(f"Samples no.: {N}")
    print(f"Average drift CFO: {np.mean(instantaneous_freq):.2f} Hz")
    print(f"Rozrzut: min {np.min(instantaneous_freq):.2f} Hz, max {np.max(instantaneous_freq):.2f} Hz")
    '''
    # --- PRZYGOTOWANIE DANYCH DO WYKRESU ---
    plot_df = pd.DataFrame({
        "time_s": time_axis,
        "instantaneous_freq_Hz": instantaneous_freq
    })

    # --- WYKRES W PLOTLY ---
    fig = px.line(plot_df,
                x="time_s",
                y="instantaneous_freq_Hz",
                title="Estimated Carrier Frequency Offset (CFO) over time",
                labels={"time_s": "Czas [s]", "instantaneous_freq_Hz": "Częstotliwość [Hz]"}
                )

    fig.update_layout(template="simple_white")
    fig.show()
    '''
    # Okno Hann (redukuje przecieki widmowe)
    window = np.hanning(N)
    windowed_signal = samples * window

    # FFT i przesunięcie zerowej częstotliwości do środka
    spectrum = np.fft.fftshift(np.fft.fft(windowed_signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/f_s))

    # Obliczanie mocy widma
    power_spectrum = np.abs(spectrum) ** 2

    # Wyszukanie częstotliwości o maksymalnej mocy (największy pik)
    peak_index = np.argmax(power_spectrum)
    estimated_cfo = freqs[peak_index]
    print ( f"{estimated_cfo=}" )

@jit(nopython=True)
def pll_kernel_numba(rx_samples, freq_estimate, alpha, beta):
    phase_estimate = 0.0
    corrected_samples = np.zeros_like(rx_samples)
    
    for n in range(len(rx_samples)):
        sample = rx_samples[n]
        # Korekcja aktualną estymacją
        val = sample * np.exp(-1j * phase_estimate)
        corrected_samples[n] = val

        # Błąd fazy z demodulowanego symbolu BPSK
        # np.real zwraca float, więc np.sign działa poprawnie w Numbie
        error = np.sign(np.real(val)) * np.imag(val)

        # Aktualizacja estymacji częstotliwości i fazy
        freq_estimate += beta * error
        phase_estimate += freq_estimate + alpha * error
        
    return corrected_samples

def pll_v0_1_3 ( rx_samples , freq_offset_initial ) :

    loop_bw = 2 * np.pi * 100 / sdr.F_S  # szerokość pasma pętli (np. 50 Hz)
    alpha = loop_bw
    beta = alpha**2 / 4
    
    # Wywołanie skompilowanego kernela Numby
    return pll_kernel_numba(rx_samples, freq_offset_initial, alpha, beta)

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
    correlation = np.correlate ( samples , preamble_samples , mode='valid' )
    max_corr = correlation[ np.argmax ( np.abs ( correlation ) ) ]
    phase_offset = np.angle ( max_corr )

    # Korekcja rotacji fazowej
    rx_corrected = samples * np.exp ( -1j * phase_offset )

    # Detekcja lustrzanego odbicia (poprzez porównanie energii korelacji dla oryginalnej i sprzężonej preambuły)
    corr_normal = np.max ( np.abs ( np.correlate ( rx_corrected , preamble_samples , mode = 'valid' ) ) )
    corr_conj = np.max ( np.abs ( np.correlate ( rx_corrected , np.conj ( preamble_samples ) , mode = 'valid' ) ) )

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
def full_compensation_v0_1_3 ( samples , preamble_samples ) :
    
    # 1. CFO compensation (PLL-based)
    rx_pll_corrected = pll_v0_1_3 ( samples, freq_offset_initial = 0.0 )

    # 2. Korekcja offsetu fazowego (na podstawie korelacji z wzorcem)
    rx_phase_corrected = correct_phase_offset_v3 ( rx_pll_corrected , preamble_samples )

    # 3. IQ imbalance (opcjonalne, ale zalecane)
    rx_final_corrected = iq_balance ( rx_phase_corrected )

    return rx_final_corrected

# Początek nowej wersji v0_1_5 full compensation z estymacją CFO z preambuły
# Funkcja powstała na bazie rozzdziału 10.5 CFO Estimation książki SDR4Engineers

def estimate_cfo_from_preamble_v0_1_5 ( rx , preamble , fs , sps ) :
    """
    Simple coarse CFO estimator using a known preamble.
    Uses products of samples separated by `sps` (M) and returns frequency offset in Hz.
    """
    if rx is None or preamble is None:
        return 0.0
    corr = np.correlate ( rx , preamble , mode = 'valid' )
    if corr.size == 0:
        return 0.0
    peak = np.argmax ( np.abs ( corr ) )
    seg_len = len ( preamble )
    # take segment aligned to preamble (clip if necessary)
    if peak + seg_len <= len ( rx ):
        seg = rx[ peak : peak + seg_len ]
    else:
        seg = rx[ peak : ]
    if len ( seg ) <= sps :
        return 0.0
    prods = seg[ sps : ] * np.conj ( seg[ : -sps ] )
    # average product to reduce noise
    avg = np.mean ( prods )
    delta = np.angle ( avg )
    f_offset = delta * fs / (2.0 * np.pi * sps)
    return float ( f_offset )

def estimate_cfo_from_preamble_early ( rx , preamble , fs , sps ):
    # korelacja do zlokalizowania preambuły
    corr = np.correlate ( rx , preamble , mode = 'valid' )
    if corr.size == 0:
        return 0.0
    peak = np.argmax ( np.abs ( corr ) )
    # wyciągnij segment sygnału odpowiadający preambule (upewnij się, że jest wystarczająco długi)
    seg_len = len ( preamble )
    seg = rx[ peak : peak + seg_len ]
    if len ( seg ) <= sps :
        return 0.0
    # produkty próbek oddalonych o sps
    prods = seg[ sps : ] * np.conj ( seg[ : -sps ] )
    # średnia faza (bardziej odporna na szum niż pojedynczy pomiar)
    delta = np.angle ( np.mean ( prods ) )
    # przelicz na Hz: delta to faza na M próbek (M = sps)
    f_offset = delta * fs / (2.0 * np.pi * sps)
    return f_offset

def full_compensation_v0_1_5 ( samples , preamble_samples ) :
    """
    Improved full compensation pipeline:
      1) NEW FEATURE: coarse CFO estimate from preamble (apply frequency rotation) -> multiplicative correction
      2) PLL (fine tracking)
      3) phase offset correction via correlation with preamble
      4) IQ balance

    Returns corrected samples (complex numpy array).
    """
    fs = sdr.F_S
    sps = modulation.SPS

    # 1) Coarse CFO estimate and correction
    ts = t.perf_counter_ns ()
    coarse_f = estimate_cfo_from_preamble_v0_1_5 ( samples , preamble_samples , fs, sps )
    if settings["log"]["verbose_1"] : print(f"estimate_cfo_from_preamble_v0_1_5 w czasie [ms]: {( t.perf_counter_ns () - ts ) / 1e6:.1f} ")
    # Apply coarse correction
    if coarse_f != 0.0 :
    # Alternative apply coarse correction: if abs(coarse_f) > 1e-12:
        n = np.arange ( len ( samples ) )
        samples = samples * np.exp ( -1j * 2.0 * np.pi * coarse_f * n / fs )

    # 2) PLL-based fine tracking
    ts = t.perf_counter_ns ()
    pl_corrected = pll_v0_1_3 ( samples , freq_offset_initial = 0.0 )
    if settings["log"]["verbose_1"] : print(f"pll_v0_1_3 w czasie [ms]: {( t.perf_counter_ns () - ts ) / 1e6:.1f} ")

    # 3) Phase offset correction using preamble
    ts = t.perf_counter_ns ()
    rx_phase_corrected = correct_phase_offset_v3 ( pl_corrected , preamble_samples )
    if settings["log"]["verbose_1"] : print(f"correct_phase_offset_v3 w czasie [ms]: {( t.perf_counter_ns () - ts ) / 1e6:.1f} ")

    # 4) IQ imbalance compensation
    rx_final_corrected = iq_balance ( rx_phase_corrected )

    return rx_final_corrected
# Koniec nowej wersji v0_1_5 full compensation z estymacją CFO z preambuły

# Pipeline kompensacji CFO, fazy, IQ
def full_compensation ( samples, fs , preamble_samples ) :
    
    # 1. CFO compensation (PLL-based)
    rx_pll_corrected = pll_v0_1_3 ( samples, freq_offset_initial=0.0)

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

def generate_noisy_samples () :
    '''
    Zakłócenia są bardzo podobne do tych w pliku rx_samples_32768.csv i obejmują:
    stopniowy dryft fazy (frequency offset),
    zmienność amplitudy,
    dodatkowy szum.

    Pomiędzy powtórzeniami jest tylko 10 próbek bardzo słabego szumu, więc korelacja powinna wykryć wielokrotne szczyty, ale każdy z nich będzie nieco zniekształcony
    – idealne do testowania robustności detekcji synchronizacji w Twoim modemie BPSK na Pluto.
    Poziom zakłóceń można zmianiać (więcej/mniej offsetu, szumu), dostosować parametry phase_drift_rate, initial_phase, noise_std.
    '''

    # Definicja sekwencji synchronizacyjnej sync_sequence_2 (jako próbki I/Q - tutaj czysto rzeczywiste)
    sync_seq_raw = np.array([0, 100, 0, -100, 0, 200, 0, -200, 0, 1000, 0, -200, 0, 200, 0, -100, 0, 100, 0], dtype=np.int32)

    # Na podstawie analizy Twojego rzeczywistego odebranego sygnału rx_samples_32768.csv
    # sekwencja jest przesunięta w fazie (clock offset / frequency offset) oraz ma zmienną amplitudę
    # Symulujemy podobne zakłócenie: rotację fazy + skalowanie amplitudy + dodatkowy szum

    np.random.seed(42)  # dla powtarzalności

    sync_length = len(sync_seq_raw)

    # Parametry zakłócenia (dostosowane, żeby było podobnie trudne jak w Twoim pliku)
    phase_drift_rate = 0.012   # stopniowe dryft fazy (symuluje frequency offset)
    initial_phase = np.deg2rad(15)  # początkowa faza
    amp_variation = 0.85 + 0.15 * np.sin(np.linspace(0, np.pi, sync_length))  # zmienność amplitudy ~15%
    noise_std = 80  # dodatkowy szum gaussowski (mniejszy niż w noisy_1, ale wystarczający)

    # Jedna zakłócona wersja sekwencji
    t = np.arange(sync_length)
    phase = initial_phase + phase_drift_rate * t
    rotation = np.exp(1j * phase)
    distorted_sync = (sync_seq_raw * amp_variation) * np.real(rotation) + 1j * (sync_seq_raw * amp_variation) * np.imag(rotation)
    distorted_sync = distorted_sync + noise_std * (np.random.randn(sync_length) + 1j * np.random.randn(sync_length))

    distorted_sync_int = np.round(np.real(distorted_sync)).astype(np.int32) + 1j * np.round(np.imag(distorted_sync)).astype(np.int32)

    # Teraz budujemy samples_2_noisy_4_fo: 10 powtórzeń tej samej zakłóconej sekwencji
    # oddzielonych 10 próbkami bardzo słabego szumu (std=15)
    repeats = 10
    separator_len = 10
    separator = 15 * (np.random.randn(separator_len) + 1j * np.random.randn(separator_len))
    separator = np.round(np.real(separator)).astype(np.int32) + 1j * np.round(np.imag(separator)).astype(np.int32)

    # Lista części
    parts = [distorted_sync_int]
    for _ in range(repeats - 1):
        parts.append(separator)
        parts.append(distorted_sync_int)

    samples_2_noisy_4_fo_complex = np.concatenate(parts)

    # Konwersja na oddzielne real i imag jako np.int32 (jeśli potrzebujesz dwóch tablic)
    samples_2_noisy_4_fo_real = np.real(samples_2_noisy_4_fo_complex).astype(np.int32)
    samples_2_noisy_4_fo_imag = np.imag(samples_2_noisy_4_fo_complex).astype(np.int32)

    # Lub jako jedna tablica interleaved I, Q (często używana w Pluto)
    samples_2_noisy_4_fo_interleaved = np.empty(2 * len(samples_2_noisy_4_fo_complex), dtype=np.int32)
    samples_2_noisy_4_fo_interleaved[0::2] = samples_2_noisy_4_fo_real
    samples_2_noisy_4_fo_interleaved[1::2] = samples_2_noisy_4_fo_imag

    #print("Długość samples_2_noisy_4_fo_interleaved:", len(samples_2_noisy_4_fo_interleaved))
    #print("Przykład pierwszych 60 wartości (I,Q):")
    #print(samples_2_noisy_4_fo_interleaved[:60])
    return samples_2_noisy_4_fo_real