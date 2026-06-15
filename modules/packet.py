import numpy as np , os , torch , time as t , tomllib , zlib

from fileinput import filename
from adi import Pluto
from dataclasses import dataclass , field
from modules import corrections , filters , modulation, ml , ops_file, plot , sdr
from numpy.typing import NDArray
from pathlib import Path
from scipy.signal import find_peaks
from typing import Any

np.set_printoptions ( threshold = np.inf , linewidth = np.inf ) # Ensures all array elements are displayed without truncation and prevents line wrapping for long output lines.

script_filename = os.path.basename ( __file__ )

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

log_packet : str = ""

def add2log_packet ( entry : str ) -> None :
    global log_packet
    log_packet += entry + "\n"

def bits_2_byte_list ( bits : NDArray[ np.uint8 ] ) -> list [ int ] :
    """
    na bazie def bits_2_byte_list stwórz nową funkcję def bits_2_byte_list_v0_1_7 w której ostatnie brakujące do 8, bity będą uzupełniane zerami. Dzięki temu 

    Zamienia tablicę bitów (0/1) na listę bajtów (List[NDArray[ np.uint8 ]]),
    traktując każdy zestaw 8 bitów jako jeden bajt (big-endian w bajcie).

    Parametry:
    -----------
    bits : NDArray[ np.uint8 ]
        Tablica bitów typu np.uint8, długość podzielna przez 8.

    Zwraca:
    --------
    List[NDArray[ np.uint8 ]]
        Lista bajtów jako tablice typu np.uint8 z przedziału 0–255.
    """
    if not isinstance(bits, np.ndarray):
        raise TypeError("Argument musi być typu numpy.ndarray.")
    if len(bits) % 8 != 0:
        raise ValueError("Długość bitów musi być wielokrotnością 8.")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Tablica może zawierać tylko 0 i 1.")

    byte_list = []
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i+8]:
            byte = (byte << 1) | int(bit)
        byte_list.append(np.array([byte], dtype=np.uint8))

    return byte_list

def bits_2_int ( bits : NDArray[ np.uint8 ] ) -> int:
    """
    Zamienia tablicę bitów (najstarszy bit pierwszy) na wartość dziesiętną,
    używając operacji bitowych.

    Parametry:
    -----------
    bits : NDArray[ np.uint8 ]
        Tablica bitów (0/1) typu np.uint8, maks. 16 bitów.

    Zwraca:
    --------
    int
        Wartość dziesiętna odpowiadająca zakodowanym bitom.
    """
    if not isinstance(bits, np.ndarray):
        raise TypeError("Argument musi być typu numpy.ndarray.")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Tablica może zawierać tylko wartości 0 i 1.")
    result = 0
    for bit in bits:
        result = (result << 1) | int ( bit )
    return result

BITS_IN_BYTE = np.uint32 ( 8 )
CONCATENATE_THS : int = 10 # Ograniczenie na liczbę próbek dołączanych do siebie w celu uniknięcia zbyt długich tablic w pamięci.

CORR_SEQ_BITS : NDArray[ np.uint8 ] = np.array ( settings[ "BARKER13_BITS" ] , dtype = np.uint8 )
CORR_SEQ_SAMPLES_LEN : np.uint32 = np.uint32 ( len ( CORR_SEQ_BITS ) * modulation.SPS )
PADDING_BITS : NDArray[ np.uint8 ] = np.array ( settings[ "PADDING_BITS" ] , dtype = np.uint8 )
PADDING_SAMPLES_LEN : np.uint32 = np.uint32 ( len ( PADDING_BITS ) * modulation.SPS )
SYNC_SEQ_BITS : NDArray[ np.uint8 ] = np.array ( settings[ "SYNC_SEQ_BITS" ] , dtype = np.uint8 )
SYNC_SEQ_SAMPLES_LEN : np.uint32 = np.uint32 ( len ( SYNC_SEQ_BITS ) * modulation.SPS )
RADIO_PREAMBLE_BITS : NDArray[ np.uint8 ] = np.concatenate ( [ CORR_SEQ_BITS , PADDING_BITS , SYNC_SEQ_BITS ] , dtype = np.uint8 )
RADIO_PREAMBLE_BITS_LEN : np.uint32 = np.uint32 ( len ( RADIO_PREAMBLE_BITS ) )
RADIO_PREAMBLE_SAMPLES_LEN : np.uint32 = np.uint32 ( RADIO_PREAMBLE_BITS_LEN * modulation.SPS )

RESERVE_BITS : NDArray[ np.uint8 ] = np.array ( settings[ "RESERVE_BITS" ] , dtype = np.uint8 )
RESERVE_BITS_LEN : np.uint32 = np.uint32 ( len ( RESERVE_BITS ) )
RESERVE_SAMPLES_LEN : np.uint32 = np.uint32 ( RESERVE_BITS_LEN * modulation.SPS )

PACKET_LEN_BITS_LEN : np.uint32 = np.uint32 ( 11 )
PACKET_LEN_SAMPLES_LEN : np.uint32 = np.uint32 ( PACKET_LEN_BITS_LEN * modulation.SPS )
CRC32_BYTES_LEN : np.uint32 = np.uint32 ( 4 )
CRC32_BITS_LEN : np.uint32 = CRC32_BYTES_LEN * BITS_IN_BYTE
CRC32_SAMPLES_LEN : np.uint32 = np.uint32 ( CRC32_BITS_LEN * modulation.SPS )
FRAME_HEADER_BITS_LEN : np.uint32 = RESERVE_BITS_LEN + PACKET_LEN_BITS_LEN + CRC32_BITS_LEN
FRAME_HEADER_SAMPLES_LEN : np.uint32 = FRAME_HEADER_BITS_LEN * modulation.SPS
MIN_FRAME_LEN_BITS : np.uint32 = RESERVE_BITS_LEN + PACKET_LEN_BITS_LEN + CRC32_BITS_LEN

PAYLOAD_BYTES_LEN_THS : np.uint32 = np.uint32 ( 1500 ) # MTU dla IP over ETHERNET
PAYLOAD_SAMPLES_LEN_THS : np.uint32 = np.uint32 ( PAYLOAD_BYTES_LEN_THS * BITS_IN_BYTE * modulation.SPS )
PACKET_SAMPLES_LEN_THS : np.uint32 = np.uint32 ( PAYLOAD_BYTES_LEN_THS * BITS_IN_BYTE * modulation.SPS + CRC32_BITS_LEN * modulation.SPS )

def detect_sync_sequence_peaks ( samples: NDArray[ np.complex64 ] , sync_sequence : NDArray[ np.complex64 ] , deep : bool = False ) -> NDArray[ np.uint32 ] :
    
    plt = False
    if settings["log"]["verbose_1"] : ts = t.perf_counter_ns ()
    min_peak_height_ratio = 0.8
    
    if deep :
        peaks_real = np.array ( [] ).astype ( np.uint32 )
        peaks_neg_real = np.array ( [] ).astype ( np.uint32 )
        peaks_imag = np.array ( [] ).astype ( np.uint32 )
        peaks_neg_imag = np.array ( [] ).astype ( np.uint32 )
    peaks_all = np.array ( [] ).astype ( np.uint32 )
    peaks = np.array ( [] ).astype ( np.uint32 )

    if deep :
        corr_real = np.abs ( np.correlate ( samples.real , sync_sequence.real , mode = "valid" ) )
        corr_neg_real = np.abs ( np.correlate ( -samples.real , sync_sequence.real , mode = "valid" ) )
        corr_imag = np.abs ( np.correlate ( samples.imag , sync_sequence.real , mode = "valid" ) )
        corr_neg_imag = np.abs ( np.correlate ( -samples.imag , sync_sequence.real , mode = "valid" ) )
    corr = np.abs ( np.correlate ( samples , np.conj ( sync_sequence ) , mode = "valid" ) )

    ones = np.ones ( len ( sync_sequence ) )
    sync_seq_norm = np.linalg.norm ( sync_sequence )
    
    if deep :
        local_energy_real = np.correlate ( samples.real**2 , ones , mode = "valid" )
        local_energy_neg_real = np.correlate ( ( -samples.real )**2 , ones , mode = "valid" )
        local_energy_imag = np.correlate ( samples.imag**2 , ones , mode = "valid" )
        local_energy_neg_imag = np.correlate ( ( -samples.imag )**2 , ones , mode = "valid" )
    local_energy_abs = np.correlate ( np.abs ( samples )**2 , ones , mode = "valid" )
    
    if deep :
        local_signal_real_norm = np.sqrt ( np.maximum ( local_energy_real , 1e-10 ) )
        local_signal_neg_real_norm = np.sqrt ( np.maximum ( local_energy_neg_real , 1e-10 ) )
        local_signal_imag_norm = np.sqrt ( np.maximum ( local_energy_imag , 1e-10 ) )
        local_signal_neg_imag_norm = np.sqrt ( np.maximum ( local_energy_neg_imag , 1e-10 ) )
    local_signal_norm = np.sqrt ( np.maximum ( local_energy_abs , 1e-10 ) )
    
    # Wynik znormalizowany (wartości teoretycznie od -1.0 do 1.0)
    if deep :
        corr_real_norm = corr_real / ( local_signal_real_norm * sync_seq_norm )
        corr_neg_real_norm = corr_neg_real / ( local_signal_neg_real_norm * sync_seq_norm )
        corr_imag_norm = corr_imag / ( local_signal_imag_norm * sync_seq_norm )
        corr_neg_imag_norm = corr_neg_imag / ( local_signal_neg_imag_norm * sync_seq_norm )
    corr_norm = corr / ( local_signal_norm * sync_seq_norm )

    if deep :
        max_peak_real_val = np.max ( corr_real_norm )
        max_peak_neg_real_val = np.max ( corr_neg_real_norm )
        max_peak_imag_val = np.max ( corr_imag_norm )
        max_peak_neg_imag_val = np.max ( corr_neg_imag_norm )
    max_peak_val = np.max ( corr_norm )

    min_correlation_threshold = 0.8

    if deep :
        final_threshold_real = max ( min_correlation_threshold , max_peak_real_val * min_peak_height_ratio )
        final_threshold_neg_real = max ( min_correlation_threshold , max_peak_neg_real_val * min_peak_height_ratio )
        final_threshold_imag = max ( min_correlation_threshold , max_peak_imag_val * min_peak_height_ratio )
        final_threshold_neg_imag = max ( min_correlation_threshold , max_peak_neg_imag_val * min_peak_height_ratio )

        peaks_real , _ = find_peaks ( corr_real_norm , height = final_threshold_real , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_neg_real , _ = find_peaks ( corr_neg_real_norm , height = final_threshold_neg_real , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_imag , _ = find_peaks ( corr_imag_norm , height = final_threshold_imag , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_neg_imag , _ = find_peaks ( corr_neg_imag_norm , height = final_threshold_neg_imag , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_all = np.unique ( np.concatenate ( ( peaks_real , peaks_neg_real , peaks_imag , peaks_neg_imag ) ).astype ( np.uint32 ) )
    final_threshold = max ( min_correlation_threshold , max_peak_val * min_peak_height_ratio )
    peaks , _ = find_peaks ( corr_norm , height = final_threshold )

    if plt:
        if deep :
            if peaks_real.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_real_norm , f"corr_real_norm {corr_real_norm.size=} {peaks_real.size=}" , False , peaks_real )
            if peaks_neg_real.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_neg_real_norm , f"corr_neg_real_norm {corr_neg_real_norm.size=} {peaks_neg_real.size=}" , False , peaks_neg_real )
            if peaks_imag.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_imag_norm , f"corr_imag_norm {corr_imag_norm.size=} {peaks_imag.size=}" , False , peaks_imag )
            if peaks_neg_imag.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_neg_imag_norm , f"corr_neg_imag_norm {corr_neg_imag_norm.size=} {peaks_neg_imag.size=}" , False , peaks_neg_imag )
            if peaks_all.size > 0 :
                plot.complex_waveform_v0_1_6 ( samples , f"samples all {samples.size=} {peaks_all.size=}" , False , peaks_all )
        if peaks.size > 0 :
            plot.real_waveform_v0_1_6 ( corr_norm , f"corr_norm {corr_norm.size=} {peaks.size=}" , False , peaks )
            plot.complex_waveform_v0_1_6 ( samples , f"samples {samples.size=} {peaks.size=}" , False , peaks )

    peaks_all = np.unique ( np.concatenate ( ( peaks_all , peaks ) ).astype ( np.uint32 ) ) # Nie łączyłem tego wcześniej, bo chciałem zobaczyć co dają różne metody korelacji bez abs i jak to się ma w porównaniu do abs.
    if settings["log"]["verbose_1"] : print(f"detect_sync_sequence_peaks {peaks_all.size=} w czasie [ms]: {( t.perf_counter_ns () - ts ) / 1e6:.1f} ")
    return peaks_all

def detect_sync_sequence_peaks_v0_1_16 ( samples: NDArray[ np.complex128 ] , sync_sequence : NDArray[ np.complex128 ] , deep : bool = False ) -> NDArray[ np.uint32 ] :

    # Wzięte z Gemini Ultra Deep Thinking

    # 1. Transformacja różnicowa (Matematyczna, bezwzględna obrona przed CFO)
    rx_diff = samples[1:] * np.conj(samples[:-1])
    sync_diff = sync_sequence[1:] * np.conj(sync_sequence[:-1])

    # 2. Korelacja różnicowa (surowa)
    corr_diff = np.abs(np.correlate(rx_diff, sync_diff, mode="valid"))

    # --- LOKALNA NORMALIZACJA (Magia stabilności pożyczona z Twojego Wariantu 2) ---
    sync_diff_norm = np.linalg.norm(sync_diff)

    # Obliczamy lokalną energię sygnału różnicowego pod oknem korelacji
    ones = np.ones(len(sync_diff))
    # rx_diff podniesione do kwadratu modułu to moc sygnału w danym punkcie
    local_energy_diff = np.correlate(np.abs(rx_diff)**2, ones, mode="valid")
    local_signal_diff_norm = np.sqrt(np.maximum(local_energy_diff, 1e-10))

    # 3. Wynik znormalizowany (wartości twardo zamknięte w przedziale od 0.0 do 1.0)
    corr_norm = corr_diff / (local_signal_diff_norm * sync_diff_norm)

    # 4. Stabilny, sztywny próg bezwzględny!
    # Ze względu na podbijanie szumu przez operację różniczkową (squaring loss), 
    # próg detekcji ustawia się zazwyczaj ciut niżej niż klasyczne 0.8.
    threshold = 0.65 

    peaks, _ = find_peaks ( corr_norm , height = threshold , distance = len ( sync_sequence ) * modulation.SPS )

    return peaks

def detect_sync_sequence_peaks_v0_1_15 ( samples: NDArray[ np.complex128 ] , sync_sequence : NDArray[ np.complex128 ] , deep : bool = False ) -> NDArray[ np.uint32 ] :
    
    plt = False
    if settings["log"]["verbose_1"] : ts = t.perf_counter_ns ()
    min_peak_height_ratio = 0.8
    
    if deep :
        peaks_real = np.array ( [] ).astype ( np.uint32 )
        peaks_neg_real = np.array ( [] ).astype ( np.uint32 )
        peaks_imag = np.array ( [] ).astype ( np.uint32 )
        peaks_neg_imag = np.array ( [] ).astype ( np.uint32 )
    peaks_all = np.array ( [] ).astype ( np.uint32 )
    peaks = np.array ( [] ).astype ( np.uint32 )

    if deep :
        corr_real = np.abs ( np.correlate ( samples.real , sync_sequence.real , mode = "valid" ) )
        corr_neg_real = np.abs ( np.correlate ( -samples.real , sync_sequence.real , mode = "valid" ) )
        corr_imag = np.abs ( np.correlate ( samples.imag , sync_sequence.real , mode = "valid" ) )
        corr_neg_imag = np.abs ( np.correlate ( -samples.imag , sync_sequence.real , mode = "valid" ) )
    corr = np.abs ( np.correlate ( samples , np.conj ( sync_sequence ) , mode = "valid" ) )

    ones = np.ones ( len ( sync_sequence ) )
    sync_seq_norm = np.linalg.norm ( sync_sequence )
    
    if deep :
        local_energy_real = np.correlate ( samples.real**2 , ones , mode = "valid" )
        local_energy_neg_real = np.correlate ( ( -samples.real )**2 , ones , mode = "valid" )
        local_energy_imag = np.correlate ( samples.imag**2 , ones , mode = "valid" )
        local_energy_neg_imag = np.correlate ( ( -samples.imag )**2 , ones , mode = "valid" )
    local_energy_abs = np.correlate ( np.abs ( samples )**2 , ones , mode = "valid" )
    
    if deep :
        local_signal_real_norm = np.sqrt ( np.maximum ( local_energy_real , 1e-10 ) )
        local_signal_neg_real_norm = np.sqrt ( np.maximum ( local_energy_neg_real , 1e-10 ) )
        local_signal_imag_norm = np.sqrt ( np.maximum ( local_energy_imag , 1e-10 ) )
        local_signal_neg_imag_norm = np.sqrt ( np.maximum ( local_energy_neg_imag , 1e-10 ) )
    local_signal_norm = np.sqrt ( np.maximum ( local_energy_abs , 1e-10 ) )
    
    # Wynik znormalizowany (wartości teoretycznie od -1.0 do 1.0)
    if deep :
        corr_real_norm = corr_real / ( local_signal_real_norm * sync_seq_norm )
        corr_neg_real_norm = corr_neg_real / ( local_signal_neg_real_norm * sync_seq_norm )
        corr_imag_norm = corr_imag / ( local_signal_imag_norm * sync_seq_norm )
        corr_neg_imag_norm = corr_neg_imag / ( local_signal_neg_imag_norm * sync_seq_norm )
    corr_norm = corr / ( local_signal_norm * sync_seq_norm )

    if deep :
        max_peak_real_val = np.max ( corr_real_norm )
        max_peak_neg_real_val = np.max ( corr_neg_real_norm )
        max_peak_imag_val = np.max ( corr_imag_norm )
        max_peak_neg_imag_val = np.max ( corr_neg_imag_norm )
    max_peak_val = np.max ( corr_norm )

    min_correlation_threshold = 0.8

    if deep :
        final_threshold_real = max ( min_correlation_threshold , max_peak_real_val * min_peak_height_ratio )
        final_threshold_neg_real = max ( min_correlation_threshold , max_peak_neg_real_val * min_peak_height_ratio )
        final_threshold_imag = max ( min_correlation_threshold , max_peak_imag_val * min_peak_height_ratio )
        final_threshold_neg_imag = max ( min_correlation_threshold , max_peak_neg_imag_val * min_peak_height_ratio )

        peaks_real , _ = find_peaks ( corr_real_norm , height = final_threshold_real , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_neg_real , _ = find_peaks ( corr_neg_real_norm , height = final_threshold_neg_real , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_imag , _ = find_peaks ( corr_imag_norm , height = final_threshold_imag , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_neg_imag , _ = find_peaks ( corr_neg_imag_norm , height = final_threshold_neg_imag , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_all = np.unique ( np.concatenate ( ( peaks_real , peaks_neg_real , peaks_imag , peaks_neg_imag ) ).astype ( np.uint32 ) )
    final_threshold = max ( min_correlation_threshold , max_peak_val * min_peak_height_ratio )
    peaks , _ = find_peaks ( corr_norm , height = final_threshold )

    if plt:
        if deep :
            if peaks_real.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_real_norm , f"corr_real_norm {corr_real_norm.size=} {peaks_real.size=}" , False , peaks_real )
            if peaks_neg_real.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_neg_real_norm , f"corr_neg_real_norm {corr_neg_real_norm.size=} {peaks_neg_real.size=}" , False , peaks_neg_real )
            if peaks_imag.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_imag_norm , f"corr_imag_norm {corr_imag_norm.size=} {peaks_imag.size=}" , False , peaks_imag )
            if peaks_neg_imag.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_neg_imag_norm , f"corr_neg_imag_norm {corr_neg_imag_norm.size=} {peaks_neg_imag.size=}" , False , peaks_neg_imag )
            if peaks_all.size > 0 :
                plot.complex_waveform_v0_1_6 ( samples , f"samples all {samples.size=} {peaks_all.size=}" , False , peaks_all )
        if peaks.size > 0 :
            plot.real_waveform_v0_1_6 ( corr_norm , f"corr_norm {corr_norm.size=} {peaks.size=}" , False , peaks )
            plot.complex_waveform_v0_1_6 ( samples , f"samples {samples.size=} {peaks.size=}" , False , peaks )

    peaks_all = np.unique ( np.concatenate ( ( peaks_all , peaks ) ).astype ( np.uint32 ) ) # Nie łączyłem tego wcześniej, bo chciałem zobaczyć co dają różne metody korelacji bez abs i jak to się ma w porównaniu do abs.
    if settings["log"]["verbose_1"] : print(f"detect_sync_sequence_peaks_v0_1_15 {peaks_all.size=} w czasie [ms]: {( t.perf_counter_ns () - ts ) / 1e6:.1f} ")
    return peaks_all

def detect_sync_sequence_peaks_v0_1_15_no_deep ( samples: NDArray[ np.complex128 ] , sync_sequence : NDArray[ np.complex128 ] , deep : bool = False ) -> NDArray[ np.uint32 ] :

    plt = False
    if settings["log"]["verbose_1"] : ts = t.perf_counter_ns ()
    
    min_peak_height_ratio = 0.8

    peaks = np.array ( [] ).astype ( np.uint32 )

    # W BPSK Q=0 teoretycznie, ale jeśli plik 'sync_sequence' jest typu complex,
    # musimy użyć sprzężenia (conj), inaczej korelacja będzie błędna. To bezpieczne.
    # Różnica w czasie jest pomijalna a więc zostawię conjugate
    corr = np.abs ( np.correlate ( samples , np.conj ( sync_sequence ) , mode = "valid" ) )
    #corr = np.correlate ( samples , np.conj(sync_sequence) , mode = "valid" )
    #corr = np.abs ( np.correlate ( samples , sync_sequence , mode = "valid" ) )

    ones = np.ones ( len ( sync_sequence ) )
    # Fix: Use abs(samples)**2 for calculating energy of complex signal
    local_energy = np.correlate ( np.abs ( samples )**2 , ones , mode = "valid" )
    sync_seq_norm = np.linalg.norm ( sync_sequence )
    local_signal_norm = np.sqrt ( np.maximum ( local_energy , 1e-10 ) )
    corr_norm = corr / ( local_signal_norm * sync_seq_norm )

    # Dodajemy próg bezwzględny dla znormalizowanej korelacji (np. 0.6).
    # W samym szumie max korelacja jest niska (np. 0.3), więc adaptive threshold (0.8 * max)
    # ustawiałby się na 0.24 i wykrywał szum. Wymuszenie min. 0.6 eliminuje te piki.
    # EDIT: Dla filtrowanego szumu 0.6 to za mało (True Positives in Noise). Podnoszę do 0.8.
    min_correlation_threshold_abs = 0.8
    
    max_peak_val_normalized = np.max ( corr_norm )
    
    # Próg to maksimum z (bezwzględnego minimum, relatywnego progu od piku)
    final_threshold = max ( min_correlation_threshold_abs , max_peak_val_normalized * min_peak_height_ratio )

    peaks , _ = find_peaks ( corr_norm , height = final_threshold )

    if settings["log"]["verbose_1"] : print(f"detect_sync_sequence_peaks_v0_1_15 {peaks.size=} w czasie [ms]: {( t.perf_counter_ns () - ts ) / 1e6:.1f} ")

    if plt :
        plot.real_waveform_v0_1_6 ( corr_norm , f"corr normalized {peaks.size=} {corr_norm.size=}" , False , peaks )
        plot.complex_waveform_v0_1_6 ( samples , f"samples normalized {peaks.size=} {samples.size=}" , False , peaks )

    return peaks

def detect_sync_sequence_peaks_v0_1_15_old ( samples: NDArray[ np.complex128 ] , sync_sequence : NDArray[ np.complex128 ] , fast : bool = False ) -> NDArray[ np.uint32 ] :
    
    ts = t.perf_counter_ns ()
    plt = False
    min_peak_height_ratio = 0.8
    
    if not fast :
        peaks_real = np.array ( [] ).astype ( np.uint32 )
        peaks_neg_real = np.array ( [] ).astype ( np.uint32 )
        peaks_imag = np.array ( [] ).astype ( np.uint32 )
        peaks_neg_imag = np.array ( [] ).astype ( np.uint32 )
    peaks_all = np.array ( [] ).astype ( np.uint32 )
    peaks_abs = np.array ( [] ).astype ( np.uint32 )

    if not fast :
        corr_real = np.correlate ( samples.real , sync_sequence.real , mode = "valid" )
        corr_neg_real = np.correlate ( -samples.real , sync_sequence.real , mode = "valid" )
        corr_imag = np.correlate ( samples.imag , sync_sequence.real , mode = "valid" )
        corr_neg_imag = np.correlate ( -samples.imag , sync_sequence.real , mode = "valid" )
    corr_abs = np.abs ( np.correlate ( samples , sync_sequence , mode = "valid" ) )

    ones = np.ones ( len ( sync_sequence ) )
    sync_seq_norm = np.linalg.norm ( sync_sequence )
    
    if not fast :
        local_energy_real = np.correlate ( samples.real**2 , ones , mode = "valid" )
        local_energy_neg_real = np.correlate ( ( -samples.real )**2 , ones , mode = "valid" )
        local_energy_imag = np.correlate ( samples.imag**2 , ones , mode = "valid" )
        local_energy_neg_imag = np.correlate ( ( -samples.imag )**2 , ones , mode = "valid" )
    local_energy_abs = np.correlate ( np.abs ( samples )**2 , ones , mode = "valid" )
    
    if not fast :
        local_signal_real_norm = np.sqrt ( np.maximum ( local_energy_real , 1e-10 ) )
        local_signal_neg_real_norm = np.sqrt ( np.maximum ( local_energy_neg_real , 1e-10 ) )
        local_signal_imag_norm = np.sqrt ( np.maximum ( local_energy_imag , 1e-10 ) )
        local_signal_neg_imag_norm = np.sqrt ( np.maximum ( local_energy_neg_imag , 1e-10 ) )
    local_signal_abs_norm = np.sqrt ( np.maximum ( local_energy_abs , 1e-10 ) )
    
    # Wynik znormalizowany (wartości teoretycznie od -1.0 do 1.0)
    if not fast :
        corr_real_norm = corr_real / ( local_signal_real_norm * sync_seq_norm )
        corr_neg_real_norm = corr_neg_real / ( local_signal_neg_real_norm * sync_seq_norm )
        corr_imag_norm = corr_imag / ( local_signal_imag_norm * sync_seq_norm )
        corr_neg_imag_norm = corr_neg_imag / ( local_signal_neg_imag_norm * sync_seq_norm )
    corr_abs_norm = corr_abs / ( local_signal_abs_norm * sync_seq_norm )

    if not fast :
        max_peak_real_val = np.max ( corr_real_norm )
        max_peak_neg_real_val = np.max ( corr_neg_real_norm )
        max_peak_imag_val = np.max ( corr_imag_norm )
        max_peak_neg_imag_val = np.max ( corr_neg_imag_norm )
    max_peak_abs_val = np.max ( corr_abs_norm )

    if not fast :
        peaks_real , _ = find_peaks ( corr_real_norm , height = max_peak_real_val * min_peak_height_ratio , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_neg_real , _ = find_peaks ( corr_neg_real_norm , height = max_peak_neg_real_val * min_peak_height_ratio , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_imag , _ = find_peaks ( corr_imag_norm , height = max_peak_imag_val * min_peak_height_ratio , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_neg_imag , _ = find_peaks ( corr_neg_imag_norm , height = max_peak_neg_imag_val * min_peak_height_ratio , distance = len ( sync_sequence ) * modulation.SPS )
        peaks_all = np.unique ( np.concatenate ( ( peaks_real , peaks_neg_real , peaks_imag , peaks_neg_imag ) ).astype ( np.uint32 ) )
    peaks_abs , _ = find_peaks ( corr_abs_norm , height = max_peak_abs_val * min_peak_height_ratio , distance = len ( sync_sequence ) * modulation.SPS )

    if plt and peaks_all.size > 0 :
        if not fast :
            if peaks_real.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_real_norm , f"corr_real_norm {corr_real_norm.size=} {peaks_real.size=}" , False , peaks_real )
            if peaks_neg_real.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_neg_real_norm , f"corr_neg_real_norm {corr_neg_real_norm.size=} {peaks_neg_real.size=}" , False , peaks_neg_real )
            if peaks_imag.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_imag_norm , f"corr_imag_norm {corr_imag_norm.size=} {peaks_imag.size=}" , False , peaks_imag )
            if peaks_neg_imag.size > 0 :
                plot.real_waveform_v0_1_6 ( corr_neg_imag_norm , f"corr_neg_imag_norm {corr_neg_imag_norm.size=} {peaks_neg_imag.size=}" , False , peaks_neg_imag )
            if peaks_all.size > 0 :
                plot.complex_waveform_v0_1_6 ( samples , f"samples all {samples.size=} {peaks_all.size=}" , False , peaks_all )
        if peaks_abs.size > 0 :
            plot.real_waveform_v0_1_6 ( corr_abs_norm , f"corr_abs_norm {corr_abs_norm.size=} {peaks_abs.size=}" , False , peaks_abs )
            plot.complex_waveform_v0_1_6 ( samples , f"samples abs {samples.size=} {peaks_abs.size=}" , False , peaks_abs )
    '''
    if wrt and sync:
        filename = base_path.parent / f"V7_{samples.size=}_{base_path.name}"
        with open ( filename , 'w' , newline='' ) as csvfile :
            fieldnames = ['corr', 'peak_idx', 'peak_val']
            writer = csv.DictWriter ( csvfile , fieldnames = fieldnames )
            writer.writeheader ()
            for idx in peaks_abs :
                writer.writerow ( { 'corr': 'abs' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_abs[ idx ] ) } )
            for idx in peaks_real :
                writer.writerow ( { 'corr' : 'real' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_real[ idx ] ) } )
            for idx in peaks_imag :
                writer.writerow ( { 'corr' : 'imag' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_imag[ idx ] ) } )
            for idx in peaks :
                writer.writerow ( { 'corr' : 'all' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_abs[ idx ] ) } )
    '''
    peaks_all = np.unique ( np.concatenate ( ( peaks_all , peaks_abs ) ).astype ( np.uint32 ) ) # Nie łączyłem tego wcześniej, bo chciałem zobaczyć co dają różne metody korelacji bez abs i jak to się ma w porównaniu do abs.
    if settings["log"]["verbose_1"] : print(f"Detekcja {peaks_all.size=} w czasie [ms]: {( t.perf_counter_ns () - ts ) / 1e6:.1f} ")
    return peaks_all

def gen_bits ( bytes ) :
    return np.unpackbits ( np.array ( bytes , dtype = np.uint8 ) )

def bytes2bits ( bytes : NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
    return np.unpackbits ( np.array ( bytes , dtype = np.uint8 ) ).astype ( np.uint8 ) # zawsze MSB first

def dec2bits ( dec : np.uint8 | np.uint16 | np.uint32 | np.uint64 , num_bits : int ) -> NDArray[ np.uint8 ] :
    """
    Zamienia liczbę dziesiętną na tablicę bitów (MSB first), biorąc ostatnie num_bits bitów z prawej strony.

    Parametry:
    -----------
    dec : np.uint8 | np.uint16 | np.uint32 | np.uint64
        Liczba dziesiętna do konwersji.
    num_bits : int
        Liczba bitów do wyciągnięcia (ostatnie num_bits bitów). Musi być <= liczbie bitów w typie dec.

    Zwraca:
    --------
    NDArray[np.uint8]
        Tablica bitów typu uint8, długość num_bits.
    """
    if isinstance ( dec , np.uint8 ) :
        max_bits = 8
    elif isinstance ( dec , np.uint16 ) :
        max_bits = 16
    elif isinstance ( dec , np.uint32 ) :
        max_bits = 32
    elif isinstance ( dec , np.uint64 ) :
        max_bits = 64
    else :
        raise TypeError ( "dec musi być typu np.uint8, np.uint16, np.uint32 lub np.uint64" )
    
    if num_bits > max_bits :
        raise ValueError ( f"num_bits ({num_bits}) nie może być większa niż liczba bitów w typie dec ({max_bits})" )
    
    return np.array ( [ ( dec >> i ) & 1 for i in range ( num_bits - 1 , -1 , -1 ) ] , dtype = np.uint8 )

def pad_bits2bytes ( bits : NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
    # dopełnij do pełnych bajtów (z prawej zerami) i spakuj
    pad = ( -len ( bits ) ) % 8
    if pad :
        bits = np.concatenate ( [ bits , np.zeros ( pad , dtype = np.uint8 ) ] )
    return np.packbits ( bits )

def create_crc32_bytes ( bytes : NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
    crc32 = zlib.crc32 ( bytes )
    return np.frombuffer ( crc32.to_bytes ( 4 , 'big' ) , dtype = np.uint8 )

@dataclass ( slots = True , eq = False )
class RxPacket_v0_1_18 :

    samples : NDArray[ np.complex128 ]
    has_packet : bool = False
    samples_corrected : NDArray[ np.complex128 ] = field ( init = False )
    bits : NDArray[ np.uint8 ] = field ( init = False )
    bytes : NDArray[ np.uint8 ] = field ( init = False )
    payload_bytes : NDArray[ np.uint8 ] = field ( init = False )
    bpsk_symbols : NDArray[ np.complex128 ] = field ( init = False )
    
    # Pola uzupełnianie w __post_init__

    def __post_init__ ( self ) -> None :
        self.process_packet ()
    
    def process_packet ( self ) -> None :
        sps = modulation.SPS
        payload_end_idx = self.samples.size - CRC32_SAMPLES_LEN
        samples_components = [ ( self.samples.real , "packet real" ) , ( self.samples.imag , "packet imag" ) , ( -self.samples.real , "packet -real" ) , ( -self.samples.imag , "packet -imag" ) ]
        for samples_component , samples_name in samples_components :
            payload_symbols = samples_component [ : payload_end_idx : sps ]
            crc32_symbols = samples_component [ payload_end_idx : : sps ]
            payload_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( payload_symbols )
            payload_bytes = pad_bits2bytes ( payload_bits )
            crc32_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( crc32_symbols )
            crc32_bytes_read = pad_bits2bytes ( crc32_bits )
            crc32_bytes_calculated = create_crc32_bytes ( payload_bytes )
            if ( crc32_bytes_read == crc32_bytes_calculated ).all () :
                self.has_packet = True
                self.bits = np.concatenate ( [ payload_bits , crc32_bits ] )
                self.bpsk_symbols = modulation.bits_2_bpsk_symbols_v0_1_18 ( self.bits )
                self.bytes = np.concatenate ( [ payload_bytes , crc32_bytes_read ] )
                self.payload_bytes = payload_bytes
                if settings["log"]["verbose_2"] : print ( samples_name )
                return
        # Przed dużymi zminami 
        # To poniższe cfo nie może zadziałać dobrze bo w samplach nie przekazuję barker13 do korekcji cfo, przecież przekazuję tylko wyciątą część sampli pakietu bez sync sequence, rozważyć przekazywanie całej ramki.
        if settings["log"]["verbose_2"] : self.analyze ( samples = self.samples )
        self.correct_cfo ()
        if settings["log"]["verbose_2"] : self.plot_complex_samples_filtered_and_corrected ( title = f"RxPacket_v0_1_13 after CFO" , marker = False )
        if settings["log"]["verbose_2"] : self.analyze ( samples = self.samples_corrected )
        payload_symbols = self.samples_corrected [ : payload_end_idx : sps ]
        crc32_symbols = self.samples_corrected [ payload_end_idx : : sps ]
        payload_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( payload_symbols )
        payload_bytes = pad_bits2bytes ( payload_bits )
        crc32_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( crc32_symbols )
        crc32_bytes_read = pad_bits2bytes ( crc32_bits )
        crc32_bytes_calculated = create_crc32_bytes ( payload_bytes )
        if ( crc32_bytes_read == crc32_bytes_calculated ).all () :
            self.has_packet = True
            self.bits = np.concatenate ( [ payload_bits , crc32_bits ] )
            self.bpsk_symbols = modulation.bits_2_bpsk_symbols_v0_1_18 ( self.bits )
            self.bytes = np.concatenate ( [ payload_bytes , crc32_bytes_read ] )
            self.payload_bytes = payload_bytes
            return

    def correct_cfo ( self ) -> None :
        self.samples_corrected = modulation.zero_quadrature ( corrections.full_compensation_v0_1_5 ( self.samples , modulation.generate_barker13_bpsk_samples_v0_1_7 ( clip_tail = True ) ) )

    def plot_complex_samples_filtered_and_corrected ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples , f"{title} {self.samples.size=}" , marker_squares = marker , marker_peaks = peaks )

    def analyze ( self , samples ) -> None :
        sdr.analyze_rx_signal ( samples )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.samples.size= }, { self.has_packet= }, { self.payload_bytes.size if self.has_packet else None= }"
        )

@dataclass ( slots = True , eq = False )
class RxFrame :
    
    samples : NDArray[ np.complex128 ]
    first_symbol_abs_idx : np.uint32

    # Pola uzupełnianie w __post_init__
    SPS = modulation.SPS
    SPAN = filters.SPAN
    header_bpsk_symbols : NDArray[ np.complex128 ] = field ( init = False )
    header_bits : NDArray[ np.uint8 ] = field ( init = False )
    frame_end_abs_idx : np.uint32 = field ( init = False )
    packet_len : np.uint16 = field ( init = False )
    packet_first_symbol_abs_idx : NDArray[ np.uint32 ] = field ( init = False )
    leftovers_start_abs_idx : np.uint32 = field ( init = False )
    has_header : bool = False # przydate jeśli nie wykryje pakietu/payload
    has_frame : bool = False # ustawiany dopiero po walidacji pakietu, wcześniej używamy tylko lokalnego has_header
    has_leftovers : bool = False
    # do zapamiętania jako tip przed skasowaniem payload_bytes : NDArray[ np.uint8 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint8 ) , init = False )
    packet : RxPacket_v0_1_18 = field ( init = False )
    
    def __post_init__ ( self ) -> None :
        if not self.frame_len_validation () :
            return self.first_symbol_abs_idx
        self.process_frame ()
    
    def process_frame ( self ) -> None :
        # Przetestować działanie sync_sequence_start_idx = filter.FIRST_TO_MIDDLE_SYMBOL_OFFSET, czyli 0, bo to jest offset od początku ramki do środka symbolu, a więc jeśli znajdę dopasowanie ramki, to jest ono w jednym z 4 punktów oddalonych o 0, 1/4 SPS, 1/2 SPS, 3/4 SPS od początku ramki. To jest ważne, bo jeśli znajdę dopasowanie ramki, to jest ono w jednym z tych 4 punktów i wtedy muszę zacząć czytać header i pakiet od tego punktu, a nie od początku ramki. Jeśli znajdę dopasowanie ramki i zacznę czytać header i pakiet od tego punktu, to będzie dobrze, bo ten punkt jest początkiem symbolu sync sequence i wtedy czytając co SPS próbkę będę czytał kolejne symbole sync sequence, a potem header i pakiet. Jeśli znajdę dopasowanie ramki i zacznę czytać header i pakiet od początku ramki, to będzie źle, bo ten punkt może być w połowie symbolu sync sequence i wtedy czytając co SPS próbkę będę czytał połowy symboli sync sequence i połowy symboli header i pakietu, co będzie błędne. Dlatego ważne jest, żeby sync_sequence_start_idx był równy filter.FIRST_TO_MIDDLE_SYMBOL_OFFSET, czyli 0.
        reserve_start_idx : np.uint32 = filters.FIRST_TO_MIDDLE_SYMBOL_OFFSET
        reserve_end_idx : np.uint32 = reserve_start_idx + RESERVE_SAMPLES_LEN
        packet_len_start_idx : np.uint32 = reserve_end_idx
        packet_len_end_idx : np.uint32 = packet_len_start_idx + PACKET_LEN_SAMPLES_LEN
        crc32_start_idx : np.uint32 = packet_len_end_idx
        crc32_end_idx : np.uint32 = crc32_start_idx + CRC32_SAMPLES_LEN

        samples_components = [ ( self.samples.real , "sync sequence real" ) , ( self.samples.imag , "sync sequence imag" ) , ( -self.samples.real , "sync sequence -real" ) , ( -self.samples.imag , "sync sequence -imag" ) ]
        for samples_component , samples_name in samples_components :
            reserve_symbols = samples_component [ reserve_start_idx : reserve_end_idx : self.SPS ]
            reserve_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( reserve_symbols )
            if np.array_equal ( reserve_bits , RESERVE_BITS ) : add2log_packet ( f"{t.time()},has_reserve_bits,{(self.first_symbol_abs_idx + reserve_start_idx)=}")
            packet_len_symbols = samples_component [ packet_len_start_idx : packet_len_end_idx : self.SPS ]
            packet_len_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( packet_len_symbols )
            packet_len_uint16 = self.packet_len = self.bits2uint16 ( packet_len_bits )
            check_components = [ ( self.samples.real , " frame real" ) , ( self.samples.imag , " frame imag" ) , ( -self.samples.real , " frame -real" ) , ( -self.samples.imag , " frame -imag" ) ]
            for samples_comp , frame_name in check_components :
                crc32_symbols = samples_comp [ crc32_start_idx : crc32_end_idx : self.SPS ]
                crc32_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( crc32_symbols )
                crc32_bytes_read = pad_bits2bytes ( crc32_bits )
                crc32_bytes_calculated = create_crc32_bytes ( pad_bits2bytes ( np.concatenate ( [ reserve_bits, packet_len_bits ] ) ) )
                if ( crc32_bytes_read == crc32_bytes_calculated ).all () :
                    packet_end_idx = crc32_end_idx + ( packet_len_uint16 * BITS_IN_BYTE * self.SPS )
                    self.has_header = True
                    self.packet_first_symbol_abs_idx = self.first_symbol_abs_idx + crc32_end_idx - filters.FIRST_TO_MIDDLE_SYMBOL_OFFSET # używać tylko jeśli self.has_packet, inaczej może być poza zakresem sampli
                    self.frame_end_abs_idx = self.first_symbol_abs_idx + packet_end_idx - filters.FIRST_TO_MIDDLE_SYMBOL_OFFSET # używać tylko jeśli self.has_packet, inaczej może być poza zakresem sampli
                    self.header_bits = np.concatenate ( [ reserve_bits , packet_len_bits , crc32_bits ] )
                    self.header_bpsk_symbols = modulation.bits_2_bpsk_symbols_v0_1_18 ( self.header_bits )
                    add2log_packet ( f"{t.time()},{self.has_header=},{self.first_symbol_abs_idx=}" )
                    if not self.packet_len_validation ( packet_end_idx ) :
                        add2log_packet ( f"{t.time()},{self.has_header=},{reserve_start_idx=},{self.first_symbol_abs_idx=}" )
                        if settings["log"]["verbose_2"] : print ( f"{self.first_symbol_abs_idx=} {samples_name} {frame_name=} {self.has_header=}" )
                        return
                    packet = RxPacket_v0_1_18 ( samples = self.samples [ crc32_end_idx : packet_end_idx ] )
                    if packet.has_packet :
                        self.has_frame = True # has_frame jeśli ma header i pakiet, inaczej nie ma całej ramki
                        self.packet = packet
                        add2log_packet(f"{t.time()},{packet.has_packet=},{crc32_end_idx=}")
                        if settings["log"]["verbose_2"] : print ( f"{reserve_start_idx=},{self.first_symbol_abs_idx=} {self.has_frame=},{packet.has_packet=}" )
                        return
        if settings["log"]["verbose_2"] : print ( f"{self.first_symbol_abs_idx=},{self.has_frame=}" )
        return

    def samples2bits ( self , samples : NDArray[ np.complex128 ] ) -> NDArray[ np.uint8 ] :
        return modulation.bpsk_symbols_2_bits_v0_1_7 ( samples [ : : self.sps ] )

    def bits2uint16 ( self , bits : NDArray[ np.uint8 ] ) -> np.uint16 :
        return np.uint16 ( bits_2_int ( bits ) )

    def samples2bytes ( self , samples : NDArray[ np.complex128 ] ) -> NDArray[ np.uint8 ] :
        bits = self.samples2bits ( samples )
        return pad_bits2bytes ( bits )
    
    def set_leftovers_idx_for_incomplete_frame ( self ) -> None :
        if settings["log"]["verbose_2"] : print ( f"Samples at index { self.first_symbol_abs_idx } is too close to the end of samples to contain a complete frame. Skipping." )
        self.leftovers_start_abs_idx = self.first_symbol_abs_idx - filters.FIRST_SYMBOL_OFFSET # Bez cofniecia się do początku filtra RRC nie ma wykrycia ramnki i pakietu w następnym wywołaniu
        self.has_leftovers = True

    def frame_len_validation ( self ) -> bool :
        if np.uint32 ( self.samples.size ) <= np.uint32 ( FRAME_HEADER_SAMPLES_LEN ) :
            self.set_leftovers_idx_for_incomplete_frame ()
            return False
        return True

    def packet_len_validation ( self , packet_end_idx : np.uint32 ) -> bool :
        if packet_end_idx > np.uint32 ( self.samples.size ) :
            self.set_leftovers_idx_for_incomplete_frame ()
            return False
        return True

    def plot_complex_samples_filtered ( self , title = "" , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples , f"{title} {self.samples.size=}" , marker_peaks = peaks )

    def __repr__ ( self ) -> str :
        return ( f"{self.packet=}, {self.has_frame=}, {self.has_leftovers=}" )

@dataclass ( slots = True , eq = False )
class RxSamples :

    # Pola uzupełnianie w __post_init__
    samples_raw : NDArray[ np.complex64 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex64 ) , init = False )
    #samples_filtered : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    X_train_samples : NDArray[ np.complex64 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex64 ) , init = False )
    y_train_tensor : torch.Tensor = field ( default_factory = lambda : torch.tensor ( [] , dtype = torch.complex64 ) , init = False )
    sync_sequence_peaks : NDArray[ np.uint32 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint32 ) , init = False )
    first_symbol_idx : np.uint32 = None # Pierwszy symbol pierwszej ramki.
    concatenates : int = 0
    frames : list[ RxFrame ] = field ( init = False , default_factory = list )
    idxs : NDArray[ np.uint32 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint32 ) , init = False )
    SPS = modulation.SPS
    SPAN = filters.SPAN
    CONCATENATE_THS : int = 10

    def __post_init__ ( self ) -> None :
        pass

    def rx ( self , sdr_ctx : Pluto  | None = None , filename_and_dirname : str | None = None , concatenate : bool = False ) -> NDArray[ np.complex64 ] :

        if sdr_ctx is not None :
            samples = sdr_ctx.rx ()
        elif filename_and_dirname is not None :
            if filename_and_dirname.endswith('.npy'):
                samples = self.open_and_load_npf ( filename_and_dirname = filename_and_dirname )
            elif filename_and_dirname.endswith('.csv'):
                samples = ops_file.open_csv_and_load_np_complex ( filename = filename_and_dirname )
            else:
                raise ValueError(f"Error: unsupported file format for {filename_and_dirname}! Supported formats: .npy, .csv")
        else :
            raise ValueError ( "Either sdr_ctx or filename_and_dirname must be provided." )
        if concatenate :
            if self.concatenates < self.CONCATENATE_THS :
                self.concatenates += 1
                self.samples_raw = np.append ( self.samples_raw , samples.astype ( np.complex64 ) ) # to samo co powyżej, ale append jest szybszy dla 1 elementu
            else :
                raise MemoryError ( f"{self.concatenates=}. To prevent modem performance issues, further concatenation is blocked." )
        else :
            self.samples_raw = samples
        print ( f"{samples.dtype=}" )

    def detect_frames ( self , deep : bool = False , samples_filtered : bool = True , correct_samples : bool = False , add_peak_at_0 : bool = False ) -> None :
        
        if correct_samples and not samples_filtered :
            raise ValueError ( "Cannot apply correction without filtering. You must set filter=True to apply correction!" )
        samples = filters.apply_rrc_rx_convolve_v0_1_18 ( self.samples_raw ).copy () if samples_filtered else self.samples_raw.copy ()
        if correct_samples :
            samples = modulation.zero_quadrature ( corrections.full_compensation_v0_1_5 ( samples , self.create_corr_seq_samples ( clip_tail = True ) ) )
        self.sync_sequence_peaks = detect_sync_sequence_peaks ( samples , self.create_corr_seq_samples ( clip_tail = True ) , deep = deep )
        if add_peak_at_0 : self.sync_sequence_peaks = np.insert ( self.sync_sequence_peaks , 0 , 0 )
        previous_processed_idx : np.uint32 = 0
        for idx in self.sync_sequence_peaks :
            if idx > previous_processed_idx or idx == 0 : # idx == 0 jest wtedy kiedy chcemy dodać szczyt na 0, mimo że nie jest on wykryty w detekcji pików, ale chcemy żeby funkcja detect_frames() działała poprawnie nawet wtedy kiedy detekcja pików nie wykryje żadnego piku, a mamy leftoversy z poprzedniego wywołania, które zaczynają się od początku sampli.
                previous_processed_idx = idx
                idx = self.leap_radio_preamble ( samples , idx )
                if idx is not None :
                    frame = RxFrame ( samples = samples [ idx : ] , first_symbol_abs_idx = idx )
                    if frame.has_header :
                        self.frames.append ( frame )
                        previous_processed_idx = frame.frame_end_abs_idx
                        # Dodaj kolejne frame, które zostały wysłane w tym samym strumieniu danych.
                        # Szansa, że jest kolejna ramka jest tylko wtedy jeśli poprzednia była cała.
                        # Chociaż to podejście nie uwzględnia sytuacji braku całej ramki z powodu chwilowego zaszumienia i możliwości wykrycia kolejnej.
                        while ( frame.has_frame ) :
                            frame = RxFrame ( samples = samples [ frame.frame_end_abs_idx : ] , first_symbol_abs_idx = frame.frame_end_abs_idx )
                            if frame.has_header :
                                self.frames.append ( frame )
                                previous_processed_idx = frame.frame_end_abs_idx
        self.create_idxs ()

    def leap_radio_preamble ( self , samples : NDArray [ np.complex64 ] , idx : np.uint32 ) -> np.uint32 :
        # Wycinanie sekwencji korelacyjnej, sampli dopełniających i sekwencji synchronizacyjnej z sampli, aby sample zaczynały się od początku pakietu.
        # Sekwencja korelacyjnej jest potrzebne tylko do detekcji ramki, a nie do dalszego przetwarzania pakietu.
        # Dzięki temu funkcja detect_frames() może zacząć od razu demodulowanie ramki oraz z funkcji detect_frames() korzystać inne funkcje,
        # które np. przekazują próbki tx_active_samples bez preambuły ani sekwencji synchronizacyjnej.
        samples_components = [ samples.real , samples.imag , -samples.real , -samples.imag ]
        for samples_component in samples_components :
            radio_preamble_symbols = samples_component [ idx + filters.MIDDLE_SYMBOL_OFFSET : idx + filters.MIDDLE_SYMBOL_OFFSET + RADIO_PREAMBLE_SAMPLES_LEN : self.SPS ]
            radio_preamble_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( radio_preamble_symbols )
            if np.array_equal ( radio_preamble_bits , RADIO_PREAMBLE_BITS ) :
                return idx + filters.FIRST_SYMBOL_OFFSET + RADIO_PREAMBLE_SAMPLES_LEN
        return None

    def create_idxs ( self ) -> None :

        if self.frames is None or len ( self.frames ) == 0 :
            raise ValueError ( "ERROR!: No frames available." )
        self.first_symbol_idx = self.frames[ 0 ].first_symbol_abs_idx - SYNC_SEQ_SAMPLES_LEN - PADDING_SAMPLES_LEN - CORR_SEQ_SAMPLES_LEN
        idxs : NDArray[ np.uint32 ] = np.array ( [] , dtype = np.uint32 )
        radio_preamble_idxs : NDArray[ np.uint32 ] = np.array ( [ self.first_symbol_idx , # Pierwszy sample SYNC_SEQ obliczony powyżej
                                                                self.frames[0].first_symbol_abs_idx - SYNC_SEQ_SAMPLES_LEN - PADDING_SAMPLES_LEN , # Pierwszy sample PADDING_SAMPLES
                                                                self.frames[0].first_symbol_abs_idx - SYNC_SEQ_SAMPLES_LEN , # Pierwszy sample SYNC_SEQ
                                                                ] , dtype = np.uint32 )
        idxs = np.concatenate ( [ idxs , radio_preamble_idxs ] )
        for frame in self.frames :
            frame_idxs : NDArray[ np.uint32 ] = np.array ( [ frame.first_symbol_abs_idx , # Pierwszy sample RESERVE ramki
                                                            frame.first_symbol_abs_idx + RESERVE_SAMPLES_LEN , # Pierwszy sample PACKET_LEN ramki
                                                            frame.first_symbol_abs_idx + RESERVE_SAMPLES_LEN + PACKET_LEN_SAMPLES_LEN , # Pierwszy sample CRC32 ramki
                                                            ] , dtype = np.uint32 )
            if frame.has_frame :
                payload_samples_len : np.uint32 = np.uint32 ( ( frame.packet_len - CRC32_BYTES_LEN ) * BITS_IN_BYTE * self.SPS )
                frame_idxs = np.concatenate ( [ frame_idxs , np.array ( [ frame.first_symbol_abs_idx + RESERVE_SAMPLES_LEN + PACKET_LEN_SAMPLES_LEN + CRC32_SAMPLES_LEN , # Pierwszy sample pakietu, czyli pierwszy sample payload ramki
                                                                        frame.first_symbol_abs_idx + RESERVE_SAMPLES_LEN + PACKET_LEN_SAMPLES_LEN + CRC32_SAMPLES_LEN + payload_samples_len ,
                                                                        frame.first_symbol_abs_idx + RESERVE_SAMPLES_LEN + PACKET_LEN_SAMPLES_LEN + CRC32_SAMPLES_LEN + payload_samples_len + CRC32_SAMPLES_LEN - 1 # Ostatni sample pakietu, czyli ostatni sample payload ramki. Muszę odjąc 1 żeby nie nakładał się na pierwszy pakiet następnej ramki, który jest równy frame.first_symbol_abs_idx + RESERVE_SAMPLES_LEN + PACKET_LEN_SAMPLES_LEN + CRC32_SAMPLES_LEN + payload_samples_len. Ten ostatni sample pakietu jest potrzebny do wykrycia kolejnej ramki, bo jeśli następna ramka zaczyna się dokładnie po tym sample'u, to będzie ona wykryta, a jeśli ten ostatni sample pakietu nie będzie dodany do idxs, to następna ramka nie będzie wykryta, bo będzie zaczynała się dokładnie po ostatnim sample'u pakietu, który nie będzie w idxs i wtedy funkcja detect_frames() nie będzie wiedziała, że ma zacząć szukać ramki od tego sample'u. Dodanie tego ostatniego sample'u pakietu do idxs pozwala na wykrycie kolejnej ramki, nawet jeśli zaczyna się ona dokładnie po ostatnim sample'u pakietu poprzedniej ramki.
                                                                        ] , dtype = np.uint32 ) ] )
            idxs = np.concatenate ( [ idxs , frame_idxs ] )
        self.idxs = idxs

    def create_X_train_samples_and_y_train_tensor ( self , src_dir : Path , timestamp_group : str , X_train_samples_filtered : bool = False , symbols_src : str = None ) -> np.uint32 :

        if src_dir is None or timestamp_group is None or symbols_src is None :
            raise ValueError ( "ERROR: src_dir, timestamp_group, and symbols_src must be provided." )
        if self.frames is None or len ( self.frames ) == 0 :
            raise ValueError ( "ERROR!: No frames available." )
        
        first_symbol_idx = None
        tx_symbols = self.open_and_load_npf ( filename_and_dirname = f"{src_dir.name}/{timestamp_group}_tx_{symbols_src}.npy" )
        tx_samples = type ( self ) ()
        tx_samples.rx ( filename_and_dirname = str ( f"{src_dir.name}/{timestamp_group}_tx_samples.npy" ) )
        tx_samples.detect_frames ( deep = False , samples_filtered = False , correct_samples = False , add_peak_at_0 = True )
        for rx_frame in self.frames :
            for tx_frame in tx_samples.frames :
                if settings["log"]["verbose_2"] : print ( f"rx: {rx_frame.packet_len}	{pad_bits2bytes ( rx_frame.header_bits )}	{rx_frame.first_symbol_abs_idx}" )
                if settings["log"]["verbose_2"] : print ( f"tx: {tx_frame.packet_len}	{pad_bits2bytes ( tx_frame.header_bits )}	{tx_frame.first_symbol_abs_idx}" )
                if np.array_equal ( rx_frame.header_bits , tx_frame.header_bits ) :
                    first_symbol_idx = rx_frame.first_symbol_abs_idx - tx_frame.first_symbol_abs_idx + filters.FIRST_SYMBOL_OFFSET
                    if settings["log"]["verbose_1"] : print ( f"\r\nRamka {timestamp_group=} dopasowana w: {first_symbol_idx=}" )
                    break
            if first_symbol_idx is not None :
                break
        if first_symbol_idx is not None :
            self.X_train_samples = self.samples_filtered[ first_symbol_idx : first_symbol_idx + tx_symbols.size ].copy () if X_train_samples_filtered else self.samples_raw[ first_symbol_idx : first_symbol_idx + tx_symbols.size ].copy ()
            self.y_train_tensor = torch.from_numpy ( tx_symbols ).to ( dtype = torch.complex64 )
            return first_symbol_idx
        else :
            print ( f"ERROR: No matching frame found for timestamp_group {timestamp_group} in both rx and tx samples." )
            return None

    def create_corr_seq_samples ( self , clip_tail : bool = False ) -> NDArray[ np.complex64 ] :

        corr_seq_bpsk_symbols : NDArray [ np.complex64 ] = modulation.bits_2_bpsk_symbols ( CORR_SEQ_BITS )
        corr_seq_samples = filters.apply_tx_rrc_filter_v0_1_3 ( corr_seq_bpsk_symbols , upsample = True )
        return corr_seq_samples[ : - filters.LAST_SYMBOL_OFFSET ] if clip_tail else corr_seq_samples

    def open_and_load_npf ( self , filename_and_dirname : str ) -> NDArray[ np.complex64 ] :

        return ops_file.open_samples_from_npf ( filename_and_dirname )
    
    def save_samples_2_npf ( self , file_name : str , dir_name : str , add_timestamp : bool = False ) -> None :

        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.samples_raw )

    def save_X_and_y ( self , timestamp_group : str , dir_name : str , add_timestamp : bool = False ) -> None :

        filename = ops_file.add_timestamp_2_filename ( f"{timestamp_group}_X_train_samples.npy" ) if add_timestamp else f"{timestamp_group}_X_train_samples.npy"
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        dirname_and_filename = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( dirname_and_filename = dirname_and_filename , samples = self.X_train_samples )
        filename = ops_file.add_timestamp_2_filename ( f"{timestamp_group}_y_train_tensor.pt" ) if add_timestamp else f"{timestamp_group}_y_train_tensor.pt"
        dirname_and_filename = f"{dir_name}/{filename}"
        torch.save ( obj = self.y_train_tensor , f = dirname_and_filename )

    def plot_samples ( self , title : str = "" , samples_filtered : bool = False , mark_samples : bool = True ) -> None :
        samples = filters.apply_rrc_rx_convolve_v0_1_18 ( self.samples_raw ) if samples_filtered else self.samples_raw
        plot.complex_waveform_v0_1_6 ( samples , f"{title} {samples.size=}" , marker_peaks = self.idxs )

    def plot_X_and_y ( self , title : str = "" , mark_samples : bool = True ) -> None :
        plot.complex_waveform_v0_1_6 ( self.X_train_samples , title = f"{title} {self.X_train_samples.size=}" , marker_peaks = self.idxs - self.first_symbol_idx if mark_samples else None )
        plot.flat_tensor_v0_1_18 ( self.y_train_tensor , title = f"{title} {self.y_train_tensor.shape=}" , marker_peaks = self.idxs - self.first_symbol_idx if mark_samples else None )

    def __repr__ ( self ) -> str :

        return ( f"{self.samples_raw.size=}, {self.samples_raw.dtype=}")

@dataclass ( slots = True , eq = False )
class RxPluto_v0_1_17 :

    sn : str | None = None
    gain_control_mode_chan0 : str = field ( default = sdr.GAIN_CONTROL )
    rx_gain_chan0_int : int = field ( default = sdr.RX_GAIN )
    
    # Pola uzupełnianie w __post_init__
    pluto_rx_ctx : Pluto | None = None
    #samples : RxSamples_v0_1_16 = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.init_pluto_rx ()

    def init_pluto_rx ( self ) -> None :
        if self.sn is not None :
            self.pluto_rx_ctx = sdr.init_pluto_v0_1_17 ( sn = self.sn , gain_control_mode_chan0 = self.gain_control_mode_chan0 , rx_gain_chan0_int = self.rx_gain_chan0_int )

    def __repr__ ( self ) -> str :
        return ( f"{ self.pluto_rx_ctx= }" if self.sn is not None else f"There's no ADALM-Pluto connected!" )

@dataclass ( slots = True , eq = False )
class TxPacket :
    
    payload_bytes : NDArray[ np.uint8 ]
    
    # Pola uzupełnianie w __post_init__
    crc32_bytes : NDArray[ np.uint8 ] = field ( init = False )
    len : np.uint32 = field ( init = False )

    def __post_init__ ( self ) -> None :

        self.check_payload_bytes ()
        self.crc32_bytes = create_crc32_bytes ( self.payload_bytes )
        self.len = np.uint32 ( self.payload_bytes.size + self.crc32_bytes.size )  # payload + crc32
        pass

    def check_payload_bytes ( self ) -> None :

        if self.payload_bytes is None :
            raise ValueError ( "Error: payload_bytes must be provided." )
        payload_bytes = np.asarray ( self.payload_bytes ).ravel ()
        if payload_bytes.size == 0 :
            raise ValueError ( "Error: payload_bytes must not be empty." )
        if payload_bytes.size > PAYLOAD_BYTES_LEN_THS :
            raise ValueError ( f"ERROR: {payload_bytes.size=} exceeds {PAYLOAD_BYTES_LEN_THS=}!" )
        if not np.issubdtype ( payload_bytes.dtype , np.integer ) :
            if not np.all ( np.equal ( payload_bytes , np.floor ( payload_bytes ) ) ) :
                raise ValueError ( "Error: payload_bytes must contain only integer values in range 0..255." )
        if np.any ( payload_bytes < 0 ) or np.any ( payload_bytes > 255 ) :
            raise ValueError ( "Error: payload_bytes values must be in range 0..255." )
        self.payload_bytes = payload_bytes.astype ( np.uint8 , copy = False )

    def __repr__ ( self ) -> str :
        return ( f"{self.payload_bytes.size=}, {self.payload_bytes=}, {self.crc32_bytes=}, {self.len=}" )

@dataclass ( slots = True , eq = False )
class TxFrame :

    tx_packet : TxPacket
        
    # Pola uzupełnianie w __post_init__
    header_bytes : NDArray[ np.uint8 ] = field ( init = False )

    def __post_init__ ( self ) -> None :

        packet_len_bits : NDArray[ np.uint8 ] = dec2bits ( dec = self.tx_packet.len , num_bits = PACKET_LEN_BITS_LEN )
        reserve_and_packet_len_bytes : NDArray[ np.uint8 ] = pad_bits2bytes ( np.concatenate ( [ RESERVE_BITS , packet_len_bits ] ) )
        crc32_bytes = create_crc32_bytes ( bytes = reserve_and_packet_len_bytes )
        self.header_bytes = np.concatenate ( [ reserve_and_packet_len_bytes , crc32_bytes ] )

    def __repr__ ( self ) -> str :
        return (
            f"{self.header_bytes.size=}, {self.header_bytes=}, {self.tx_packet=}" )

@dataclass ( slots = True , eq = False )
class TxSamples :

    payload_bytes : list | tuple | np.ndarray[ np.uint8 ] | None = None

    samples : NDArray[ np.complex64 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex64 ) , init = False )
    first_symbol_idx : np.uint32 = filters.FIRST_SYMBOL_OFFSET
    # symbols_from_samples to symbole wzięte z próbkowania samples w miejscach gdzie powinny być aktywne symbole, ale nie z symboli ramek.
    # Dlatego te symbole mogą się różnić od tych z ramek, bo są wzięte z próbkowania.
    symbols_from_samples : NDArray[ np.complex64 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex64 ) , init = False )
    frames : list[ TxFrame ] = field ( init = False , default_factory = list )
    idxs : NDArray[ np.uint32 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint32 ) , init = False )
    SPS = modulation.SPS

    def __post_init__ ( self ) -> None :

        if self.payload_bytes is not None :
            self.add_frame ( payload_bytes = self.payload_bytes )

    def add_frame ( self , payload_bytes : list | tuple | NDArray[ np.uint8 ] = None ) -> None :

        tx_packet = TxPacket ( payload_bytes = payload_bytes )
        tx_frame = TxFrame ( tx_packet = tx_packet )
        self.frames.append ( tx_frame )
        self.create_samples_and_symbols_from_samples ()

    def create_samples_and_symbols_from_samples ( self ) -> None :

        frames_bytes : NDArray[ np.uint8 ] = np.concatenate ( [ np.concatenate ( [ frame.header_bytes , frame.tx_packet.payload_bytes , frame.tx_packet.crc32_bytes ] ) for frame in self.frames ] ).astype ( np.uint8 , copy = False )
        frames_bits : NDArray[ np.uint8 ] = bytes2bits ( frames_bytes )
        bpsk_symbols : NDArray[ np.complex64 ] = modulation.bits_2_bpsk_symbols ( np.concatenate ( [ RADIO_PREAMBLE_BITS, frames_bits ] ) )
        self.samples = np.ravel ( filters.apply_tx_rrc_filter_v0_1_6 ( bpsk_symbols ) ).astype ( np.complex64 , copy = False )
        
        tail_first_sample_idx = self.first_symbol_idx + bpsk_symbols.size * self.SPS
        active_samples = self.samples[ self.first_symbol_idx : tail_first_sample_idx ]
        self.symbols_from_samples = np.where ( active_samples < 0.0 , np.complex64 ( -1.0 + 0j ) , np.complex64 ( 1.0 + 0j ) )
        self.create_idxs ()

    def create_idxs ( self ) -> None :
        idxs = [ self.first_symbol_idx , # Pierwszy sample CORR_SEQ
                 self.first_symbol_idx + CORR_SEQ_SAMPLES_LEN , # Pierwszy sample PADDING
                 self.first_symbol_idx + CORR_SEQ_SAMPLES_LEN + PADDING_SAMPLES_LEN ] # Pierwszy sample SYNC_SEQ
        frame_first_symbol_idx : np.uint32 = self.first_symbol_idx + CORR_SEQ_SAMPLES_LEN + PADDING_SAMPLES_LEN + SYNC_SEQ_SAMPLES_LEN # Pierwszy sample RESERVE pierwszej ramki
        for frame in self.frames :
            payload_samples_len : np.uint32 = np.uint32 ( frame.tx_packet.payload_bytes.size * BITS_IN_BYTE * self.SPS )
            payload_first_symbol_idx : np.uint32 = frame_first_symbol_idx + FRAME_HEADER_SAMPLES_LEN
            idxs.extend ( [ frame_first_symbol_idx , # Pierwszy sample RESERVE
                            frame_first_symbol_idx + RESERVE_SAMPLES_LEN , # Pierwszy sample PACKET_LEN
                            frame_first_symbol_idx + RESERVE_SAMPLES_LEN + PACKET_LEN_SAMPLES_LEN , # Pierwszy sample CRC32 nagłówka ramki
                            payload_first_symbol_idx , # Pierwszy sample payload pakietu
                            payload_first_symbol_idx + payload_samples_len ] ) # Pierwszy sample CRC32 pakietu
            frame_first_symbol_idx += FRAME_HEADER_SAMPLES_LEN + payload_samples_len + CRC32_SAMPLES_LEN # Pierwszy sample RESERVE następnej ramki
        idxs.extend ( [ frame_first_symbol_idx ] )
        self.idxs = np.array ( idxs , dtype = np.uint32 )

    def offsets_accuracy_test ( self ) -> None :
        '''
        Zostawić tę funkcję do testowania dokładności offsetów samplowania, żeby mieć pewność że są one poprawne.
        Funkcja ta tworzy samples z ramek, a następnie sprawdza czy symbole aktywne w samples są takie same jak symbole aktywne w ramkach,
        biorąc pod uwagę offsety wynikające z filtracji i próbkowania.
        Skrypt test126-compare_offsets.py jest napisany specjalnie do testowania tej funkcji.
        '''
        frames_bpsk_symbols : NDArray [ np.complex64 ] = np.concatenate ( [ frame.bpsk_symbols for frame in self.frames ] ).astype ( np.complex64 , copy = False )
        if frames_bpsk_symbols.size > 0 :
            samples = np.ravel ( filters.apply_tx_rrc_filter_v0_1_6 ( frames_bpsk_symbols ) ).astype ( np.complex64 , copy = False )
            active_symbols = np.repeat ( frames_bpsk_symbols , self.SPS ).astype ( np.complex64 , copy = False )
            plot.complex_waveform_v0_1_6 ( samples , f"{script_filename} offset_accuracy_test {samples.size=}" )
            plot.complex_waveform_v0_1_6 ( active_symbols , f"{script_filename} offset_accuracy_test {active_symbols.size=}" )
            for i in range ( 4 ) :
                first_active_symbol_idx = np.uint32 ( self.first_symbol_idx + i -1 )
                last_frame_end_idx = first_active_symbol_idx + frames_bpsk_symbols.size * self.SPS
                active_samples = samples.real[ first_active_symbol_idx : last_frame_end_idx ]
                active_samples = np.where ( active_samples < 0.0 , np.complex64 ( -1.0 + 0j ) , np.complex64 ( 1.0 + 0j ) )
                idx = np.array ( [ first_active_symbol_idx ] , dtype = np.uint32 )
                plot.complex_waveform_v0_1_6 ( samples , f"{script_filename} offset_accuracy_test {first_active_symbol_idx=} {samples.size=}" , marker_peaks = idx )
                plot.complex_waveform_v0_1_6 ( active_samples , f"{script_filename} offset_accuracy_test {first_active_symbol_idx=} {active_samples.size=}" )
                # Porównaj active_samples z active_symbols i sprawdź czy są takie same
                if np.array_equal ( active_samples , active_symbols ) :
                    print ( f"!!!  Offsets accuracy 100% for {first_active_symbol_idx=}!" )
                else :
                    # Jak nie ma pełnej zgodności active_samples z active_symbols, to policz ile jest niezgodnosci.
                    num_different = np.sum ( active_samples != active_symbols )
                    print ( f"{num_different=} for {first_active_symbol_idx=}" )

    def tx ( self , sdr_ctx : Pluto , repeat : np.uint32 = 1 ) -> None :
        
        sdr_ctx.tx_destroy_buffer ()
        sdr_ctx.tx_cyclic_buffer = False
        samples_4_pluto : NDArray [ np.complex128 ] = sdr.scale_to_pluto_dac_v0_1_11 ( samples = self.samples , scale = 1.0 )
        if repeat < 1 or repeat > 4294967295 :
            raise ValueError ( "Error: reapt value is out of the range! Allowed range is 1 to 4294967295." )
        while repeat :
            sdr_ctx.tx ( samples_4_pluto )
            repeat -= 1

    def save_samples_2_npf ( self , file_name : str , dir_name : str , add_timestamp : bool = False ) -> None :
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.samples )

    def save_active_samples_2_npf ( self , file_name : str , dir_name : str , add_timestamp : bool = False ) -> None :
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.samples[ self.first_symbol_idx : self.first_symbol_idx + self.symbols_from_samples.size ] )

    def save_symbols_from_samples_2_npf ( self , file_name : str , dir_name : str , add_timestamp : bool = False ) -> None :
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.symbols_from_samples )

    def plot_samples ( self , title :str = "" , markers : bool = True ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples , f"{title} {self.samples.size=}" , marker_peaks = self.idxs )

    def plot_active_samples ( self , title :str = "" , markers : bool = True ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples[ self.first_symbol_idx : self.first_symbol_idx + self.symbols_from_samples.size ] , f"{title} {self.symbols_from_samples.size=}" )

    def plot_symbols_from_samples ( self , title : str = "" ) -> None :
        plot.complex_waveform_v0_1_6 ( self.symbols_from_samples , f"{title} {self.symbols_from_samples.size=}" )

    def plot_samples_4_pluto_spectrum ( self , title : str = "" ) -> None :
        samples_4_pluto : NDArray [ np.complex128 ] = sdr.scale_to_pluto_dac_v0_1_11 ( samples = self.samples , scale = 1.0 )
        plot.spectrum_occupancy ( samples = samples_4_pluto , nperseg = 1024 , title = f"{title} {samples_4_pluto.size=}" )

    def __repr__ ( self ) -> str :
        return ( f"{self.samples.size=}, {self.symbols_from_samples.size=}, {len(self.frames)=}, {self.frames=}" )

@dataclass ( slots = True , eq = False )
class TxPluto_v0_1_17 :
    
    sn : str
    tx_gain_float : float = field ( default = sdr.TX_GAIN )

    # Pola uzupełnianie w __post_init__
    pluto_tx_ctx : Pluto  = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.init_pluto_tx ()

    def init_pluto_tx ( self ) -> None :
        self.pluto_tx_ctx = sdr.init_pluto_v0_1_17 ( sn = self.sn , tx_gain_float = self.tx_gain_float )

    def __repr__ ( self ) -> str :
        return ( f"{self.pluto_tx_ctx=}" )