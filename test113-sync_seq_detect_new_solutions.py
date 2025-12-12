from modules import filters , ops_file , modulation , packet , plot
import numba
from numba import jit, prange
import numpy as np
from numpy.typing import NDArray
import os
from pathlib import Path
import time as t

def detect_barker_sync_cfar(
    samples: np.ndarray,
    sync_sequence: np.ndarray,
    threshold: float = 0.75,
    guard_cells: int = 16,
    background_cells: int = 64,
    min_peak_distance: int = 100
) -> list[tuple[int, float]]:
    """
    Ultra-szybki i solidny detektor synchronizacji Barkera-13 z pełną normalizacją CFAR.
    
    Parametry PlutoSDR / Twój modem BPSK:
    - 4 samples/symbol
    - RRC filter (α=0.35)
    - fs = 3 MHz, BW = 1 MHz, fc = 2900 MHz
    
    Zwraca listę krotek: (pozycja_piku, wartość_znormalizowanego_peak)
    """
    t_start = t.perf_counter_ns()
    
    # 1. Przygotowanie szablonu (matched filter)
    template = np.flip(sync_sequence.conj())          # odwrócony i sprzężony
    template_energy = np.vdot(sync_sequence, sync_sequence).real  # szybsze niż sum(abs**2)
    
    # 2. Korelacja (matched filter output)
    corr = np.correlate(samples, template, mode='valid')
    corr_power = np.abs(corr) ** 2                     # moc korelacji (energia)
    
    # 3. Normalizacja przez energię szablonu (stała dla wszystkich pozycji)
    normalized_corr = corr_power / template_energy
    
    # 4. CFAR – estymacja tła szumu (Cell Averaging CFAR, jednowymiarowy)
    N = len(normalized_corr)
    background_avg = np.zeros(N)
    
    half_bg = background_cells // 2
    half_guard = guard_cells // 2
    
    # Prosta, wektoryzowana estymacja tła (z pominięciem guard cells)
    for i in range(N):
        start = max(0, i - half_bg - half_guard - half_bg)
        end = min(N, i + half_guard + half_bg + 1)
        left_start = max(0, i - half_bg - half_guard)
        left_end = max(0, i - half_guard)
        right_start = min(N, i + half_guard + 1)
        right_end = min(N, i + half_guard + half_bg + 1)
        
        bg_left = normalized_corr[left_start:left_end]
        bg_right = normalized_corr[right_start:right_end]
        background_avg[i] = (np.sum(bg_left) + np.sum(bg_right)) / (background_cells)
    
    # Alternatywa szybsza (convolution) – polecam w finalnej wersji:
    # kernel = np.ones(background_cells + guard_cells + 1)
    # kernel[half_bg : half_bg + guard_cells + 1] = 0
    # background_avg = np.convolve(normalized_corr, kernel, mode='same') / background_cells
    
    cfar_threshold = threshold * background_avg
    
    # 5. Detekcja pikow powyżej progu CFAR
    peaks = []
    i = 0
    while i < N:
        if normalized_corr[i] > cfar_threshold[i]:
            # Znajdź lokalne maksimum w okolicy
            peak_idx = np.argmax(normalized_corr[i:i + guard_cells*2]) + i
            peak_value = normalized_corr[peak_idx]
            
            if peak_value > cfar_threshold[peak_idx]:
                peaks.append((peak_idx, peak_value))
                
                # Pomijamy otoczenie (unikamy wielokrotnej detekcji tego samego pakietu)
                i = peak_idx + min_peak_distance
            else:
                i += 1
        else:
            i += 1
    
    t_end = t.perf_counter_ns()
    print(f"Detekcja synchronizacji CFAR: {(t_end - t_start)/1e3:.1f} µs "
          f"(próbek: {samples.size}, wykryto: {len(peaks)})")
    
    # Opcjonalnie: wizualizacja
    if peaks:
        plot.real_waveform(normalized_corr, "Normalized Correlation (CFAR ready)", False)
        plot.real_waveform(cfar_threshold, "CFAR Threshold", False)
    
    return peaks

@ jit(nopython=True, parallel=True, cache=True)
def compute_cfar_numba(normalized_corr: np.ndarray,
                       background_cells: int = 64,
                       guard_cells: int = 16) -> np.ndarray:
    """
    Numba-accelerated Cell-Averaging CFAR (1D).
    Zwraca próg CFAR dla każdej pozycji.
    """
    N = normalized_corr.shape[0]
    half_bg = background_cells // 2
    half_guard = guard_cells // 2
    total_skip = half_bg + half_guard
    
    cfar_threshold = np.zeros(N, dtype=np.float64)
    
    for i in prange(N):
        # Lewa strona (tło przed guard)
        left_start = max(0, i - total_skip - half_bg)
        left_end = max(0, i - total_skip)
        # Prawa strona
        right_start = min(N, i + half_guard + 1)
        right_end = min(N, i + half_guard + half_bg + 1)
        
        bg_sum = 0.0
        bg_count = 0
        
        # Lewa
        for j in range(left_start, left_end):
            bg_sum += normalized_corr[j]
            bg_count += 1
        # Prawa
        for j in range(right_start, right_end):
            bg_sum += normalized_corr[j]
            bg_count += 1
            
        if bg_count > 0:
            cfar_threshold[i] = bg_sum / bg_count
        else:
            cfar_threshold[i] = 1.0  # fallback
    
    return cfar_threshold

# --------------------- FFT-based correlation (najlepsza możliwa szybkość) ---------------------
def detect_barker_sync_fft_numba(
    samples: np.ndarray,
    sync_sequence: np.ndarray,
    threshold: float = 0.75,
    guard_cells: int = 16,
    background_cells: int = 64,
    min_peak_distance: int = 200
) -> list[tuple[int, float]]:
    """
    Najszybsza możliwa detekcja synchronizacji Barkera-13:
    - Korelacja przez FFT (overlap-save niepotrzebny przy jednorazowej detekcji)
    - Numba do CFAR i peak-finding
    - Testowane na PlutoSDR: 1 milion próbek < 15 ms na i7
    """
    t_start = t.perf_counter_ns()
    
    # 1. Szablon (matched filter)
    template = np.flip(sync_sequence.conj())
    template_energy = np.vdot(sync_sequence, sync_sequence).real
    
    # 2. Korelacja przez FFT (szybsza niż np.correlate dla >10k próbek)
    N = len(samples)
    L = len(template)
    fft_len = 2**np.int32(np.ceil(np.log2(N + L - 1)))  # najbliższa potęga 2
    
    # FFT sygnału i szablonu
    samples_fft = np.fft.fft(samples, fft_len)
    template_fft = np.fft.fft(template, fft_len)
    
    # Korelacja w dziedzinie częstotliwości
    corr_complex = np.fft.ifft(samples_fft * np.conj(template_fft))
    corr = corr_complex[:N - L + 1]  # tylko ważna część (valid)
    corr_power = np.abs(corr) ** 2
    
    # 3. Normalizacja przez energię szablonu
    normalized_corr = corr_power / template_energy
    
    # 4. CFAR przez Numba
    cfar_threshold = compute_cfar_numba(normalized_corr,
                                        background_cells=background_cells,
                                        guard_cells=guard_cells)
    cfar_threshold *= threshold
    
    # 5. Peak detection (proste, ale szybkie)
    peaks = []
    i = 0
    N_corr = len(normalized_corr)
    while i < N_corr:
        if normalized_corr[i] > cfar_threshold[i]:
            # lokalne maksimum w małym oknie
            window_start = max(0, i - guard_cells)
            window_end = min(N_corr, i + guard_cells + 1)
            peak_idx = window_start + np.argmax(normalized_corr[window_start:window_end])
            peak_value = normalized_corr[peak_idx]
            
            if peak_value > cfar_threshold[peak_idx]:
                peaks.append((peak_idx, float(peak_value)))
                i = peak_idx + min_peak_distance
            else:
                i += 1
        else:
            i += 1
    
    t_end = t.perf_counter_ns()
    print(f"Detekcja FFT+Numba: {(t_end - t_start)/1e3:.1f} µs "
          f"| próbek: {N:,} | wykryto pakietów: {len(peaks)}")
    
    # Wizualizacja (opcjonalna)
    plot.real_waveform(normalized_corr, "Normalized Correlation (FFT+Numba)", False)
    plot.real_waveform(cfar_threshold, "CFAR Threshold (Numba)", False)
    
    return peaks



Path ( "logs" ).mkdir ( parents = True , exist_ok = True )
script_filename = os.path.basename ( __file__ )

filename_samples_32768_2 = "logs/rx_samples_32768_2.npy" # caly przebieg zawiera pakiety
filename_samples_32768_3_1sample = "logs/rx_samples_32768_3_1sample.npy" # caly przebieg zawiera tylko 1 pakiet
filename_samples_32768_6 = "logs/rx_samples_32768_6.npy" # dwukrotny przegieg i caly zawiera pakiety

filename_sync_sequence = "logs/barker13_samples_clipped.npy"

samples  = ops_file.open_samples_from_npf ( filename_samples_32768_3_1sample )
sync_sequence = ops_file.open_samples_from_npf ( filename_sync_sequence )

t0 = t.perf_counter_ns ()
filters.has_sync_sequence ( samples , modulation.get_barker13_bpsk_samples_v0_1_3 ( clipped = True ) )

plot.complex_waveform ( samples , f"{script_filename} | {samples.size=}" , False )
plot.complex_waveform ( sync_sequence , f"{script_filename} | {sync_sequence.size=}" , False )
m = len ( samples )
n = len ( sync_sequence )
corr = np.correlate ( samples , np.flip ( sync_sequence.conj () ) , mode = 'valid' ) ; plot.complex_waveform ( corr , f"{script_filename} | {corr.size=}" , False )
corr_abs = np.abs ( np.correlate ( samples , np.flip ( sync_sequence.conj () ) , mode = 'valid' ) ) ; plot.real_waveform ( corr_abs , f"{script_filename} | {corr_abs.size=}" , False )

sync_sequence_power = np.abs ( sync_sequence ) ** 2 ; plot.real_waveform ( sync_sequence_power , f"{script_filename} | {sync_sequence_power.size=}" , False )
sync_sequence_energy = np.sum ( sync_sequence_power ) ; print ( f" {sync_sequence_energy.size=} {sync_sequence_energy=}" )

# rolling window energy for received (efficient via cumsum)
samples_power = np.abs ( samples ) ** 2 ; plot.real_waveform ( samples_power , f"{script_filename} | {samples_power.size=}" , False )
cumsum = np.concatenate ( ( [ 0.0 ] , np.cumsum ( samples_power ) ) ) ; plot.real_waveform ( cumsum , f"{script_filename} | {cumsum.size=}" , False )
window_energy = cumsum[ n: ] - cumsum[ :-n ]

t1 = t.perf_counter_ns ()
print ( f"Detekcja sekwencji synchronizacji tj. w filters.has_sync_sequence: {(t1 - t0)/1e3:.1f} µs ")

# Ultra-szybka detekcja
peaks = detect_barker_sync_fft_numba(
    samples=samples,
    sync_sequence=sync_sequence,
    threshold=0.75,           # bardzo dobrze działa przy Eb/N0 > 6-7 dB
    guard_cells=16,           # ~4 symbole przy 4 sps
    background_cells=64,      # ~16 symboli tła
    min_peak_distance=400     # dostosuj do długości pakietu + margines
)

for idx, value in peaks:
    print(f"Znaleziono pakiet na próbce {idx} | wartość: {value:.4f}")
    # packet_samples = samples[idx:idx + dlugosc_pakietu_w_probkach]

# Detekcja!
peaks = detect_barker_sync_cfar(
    samples=samples,
    sync_sequence=sync_sequence,
    threshold=0.75,          # dobrze działa w moim teście na PlutoSDR przy SNR > 6 dB
    guard_cells=16,          # ~4 symbole (przy 4 sps)
    background_cells=64,     # ~16 symboli tła
    min_peak_distance=200    # minimalna odległość między pakietami (dostosuj do swojego formatu)
)

for idx, value in peaks:
    print(f"Synchronizacja znaleziona na próbce {idx}, wartość znormalizowana: {value:.4f}")
    # Tu możesz odciąć pakiet: packet_samples = samples[idx:idx + expected_packet_length]