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

def bits_2_byte_list ( bits : np.ndarray ) :
    """
    na bazie def bits_2_byte_list stwórz nową funkcję def bits_2_byte_list_v0_1_7 w której ostatnie brakujące do 8, bity będą uzupełniane zerami. Dzięki temu 

    Zamienia tablicę bitów (0/1) na listę bajtów (List[int]),
    traktując każdy zestaw 8 bitów jako jeden bajt (big-endian w bajcie).

    Parametry:
    -----------
    bits : np.ndarray
        Tablica bitów typu np.int64 lub podobnego, długość podzielna przez 8.

    Zwraca:
    --------
    List[int]
        Lista bajtów jako liczby całkowite z przedziału 0–255.
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
        byte_list.append(byte)

    return byte_list

def bits_2_int ( bits : np.ndarray ) -> int:
    """
    Zamienia tablicę bitów (najstarszy bit pierwszy) na wartość dziesiętną,
    używając operacji bitowych.

    Parametry:
    -----------
    bits : np.ndarray
        Tablica bitów (0/1) typu np.int64 lub podobnego, maks. 16 bitów.

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

BARKER13_BITS = np.array ( settings[ "BARKER13_BITS" ] , dtype = np.uint8 )
SYNC_SEQUENCE_LEN_BITS = len ( BARKER13_BITS )
SYNC_SEQUENCE_LEN_SAMPLES = SYNC_SEQUENCE_LEN_BITS * modulation.SPS
PACKET_LEN_LEN_BITS = 11
CRC32_LEN_BITS = 32
MAX_ALLOWED_PAYLOAD_LEN_BYTES_LEN = np.uint16 ( 1500 ) # MTU dla IP over ETHERNET
PACKET_BYTE_LEN_BITS = 8
FRAME_LEN_BITS = SYNC_SEQUENCE_LEN_BITS + PACKET_LEN_LEN_BITS + CRC32_LEN_BITS
FRAME_LEN_SAMPLES = FRAME_LEN_BITS * modulation.SPS

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

    samples_filtered : NDArray[ np.complex128 ]
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
        payload_end_idx = len ( self.samples_filtered ) - ( CRC32_LEN_BITS * sps )
        samples_components = [ ( self.samples_filtered.real , "packet real" ) , ( self.samples_filtered.imag , "packet imag" ) , ( -self.samples_filtered.real , "packet -real" ) , ( -self.samples_filtered.imag , "packet -imag" ) ]
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
        if settings["log"]["verbose_2"] : self.analyze ( samples = self.samples_filtered )
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
        self.samples_corrected = modulation.zero_quadrature ( corrections.full_compensation_v0_1_5 ( self.samples_filtered , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) ) )

    def plot_complex_samples_filtered_and_corrected ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = peaks )
        plot.complex_waveform_v0_1_6 ( self.samples_corrected , f"{title} {self.samples_corrected.size=}" , marker_squares = marker , marker_peaks = peaks )

    def analyze ( self , samples ) -> None :
        sdr.analyze_rx_signal ( samples )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.samples_filtered.size= }, { self.has_packet= }, { self.payload_bytes.size if self.has_packet else None= }"
        )

@dataclass ( slots = True , eq = False )
class RxFrame_v0_1_18 :
    
    samples_filtered : NDArray[ np.complex128 ]
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
        sync_sequence_start_idx = filters.FIRST_TO_MIDDLE_SYMBOL_OFFSET # Można wróć do sprzed zmiany w git: 81d3093def3219f03c37e046d4f5141864f28c2c
        sync_sequence_end_idx = sync_sequence_start_idx + ( SYNC_SEQUENCE_LEN_BITS * self.SPS )
        packet_len_start_idx = sync_sequence_end_idx
        packet_len_end_idx = packet_len_start_idx + ( PACKET_LEN_LEN_BITS * self.SPS )
        crc32_start_idx = packet_len_end_idx
        crc32_end_idx : np.uint32 = np.uint32 ( crc32_start_idx + ( CRC32_LEN_BITS * self.SPS ) )

        samples_components = [ ( self.samples_filtered.real , "sync sequence real" ) , ( self.samples_filtered.imag , "sync sequence imag" ) , ( -self.samples_filtered.real , "sync sequence -real" ) , ( -self.samples_filtered.imag , "sync sequence -imag" ) ]
        for samples_component , samples_name in samples_components :
            sync_sequence_symbols = samples_component [ sync_sequence_start_idx : sync_sequence_end_idx : self.SPS ]
            sync_sequence_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( sync_sequence_symbols )
            if np.array_equal ( sync_sequence_bits , BARKER13_BITS ) :
                has_sync_sequence = True
                add2log_packet ( f"{t.time()},{has_sync_sequence=},{sync_sequence_start_idx}")
                packet_len_symbols = samples_component [ packet_len_start_idx : packet_len_end_idx : self.SPS ]
                packet_len_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( packet_len_symbols )
                packet_len_uint16 = self.packet_len = self.bits2uint16 ( packet_len_bits )
                check_components = [ ( self.samples_filtered.real , " frame real" ) , ( self.samples_filtered.imag , " frame imag" ) , ( -self.samples_filtered.real , " frame -real" ) , ( -self.samples_filtered.imag , " frame -imag" ) ]
                for samples_comp , frame_name in check_components :
                    crc32_symbols = samples_comp [ crc32_start_idx : crc32_end_idx : self.SPS ]
                    crc32_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( crc32_symbols )
                    crc32_bytes_read = pad_bits2bytes ( crc32_bits )
                    crc32_bytes_calculated = create_crc32_bytes ( pad_bits2bytes ( np.concatenate ( [ sync_sequence_bits, packet_len_bits ] ) ) )
                    if ( crc32_bytes_read == crc32_bytes_calculated ).all () :
                        packet_end_idx = crc32_end_idx + ( packet_len_uint16 * PACKET_BYTE_LEN_BITS * self.SPS )
                        self.has_header = True
                        self.packet_first_symbol_abs_idx = self.first_symbol_abs_idx + crc32_end_idx - filters.FIRST_TO_MIDDLE_SYMBOL_OFFSET # używać tylko jeśli self.has_packet, inaczej może być poza zakresem sampli
                        self.frame_end_abs_idx = self.first_symbol_abs_idx + packet_end_idx - filters.FIRST_TO_MIDDLE_SYMBOL_OFFSET # używać tylko jeśli self.has_packet, inaczej może być poza zakresem sampli
                        self.header_bits = np.concatenate ( [ sync_sequence_bits , packet_len_bits , crc32_bits ] )
                        self.header_bpsk_symbols = modulation.bits_2_bpsk_symbols_v0_1_18 ( self.header_bits )
                        add2log_packet ( f"{t.time()},{sync_sequence_start_idx=},{self.has_header=},{self.first_symbol_abs_idx=}" )
                        if not self.packet_len_validation ( packet_end_idx ) :
                            add2log_packet ( f"{t.time()},{self.has_header=},{sync_sequence_start_idx=},{self.first_symbol_abs_idx=}" )
                            if settings["log"]["verbose_2"] : print ( f"{self.first_symbol_abs_idx=} {samples_name} {frame_name=} {has_sync_sequence=}, {self.has_header=}" )
                            return
                        packet = RxPacket_v0_1_18 ( samples_filtered = self.samples_filtered [ crc32_end_idx : packet_end_idx ] )
                        if packet.has_packet :
                            self.has_frame = True # has_frame jeśli ma header i pakiet, inaczej nie ma całej ramki
                            #self.bpsk_symbols = np.concatenate ( [ sync_sequence_symbols , packet_len_symbols , crc32_symbols , packet.packet_symbols ] )
                            #self.header_bpsk_symbols = modulation.bits_2_bpsk_symbols_v0_1_18 ( np.concatenate ( [ sync_sequence_bits , packet_len_bits , crc32_bits ] ) , sps = self.SPS )
                            self.packet = packet
                            add2log_packet(f"{t.time()},{packet.has_packet=},{crc32_end_idx=}")
                            if settings["log"]["verbose_2"] : print ( f"{sync_sequence_start_idx=} {has_sync_sequence=}, {self.first_symbol_abs_idx=} {self.has_frame=}, {packet.has_packet=}" )
                            return
        if settings["log"]["verbose_2"] : print ( f"{self.first_symbol_abs_idx=} {has_sync_sequence=}, {self.has_frame=}" )
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
        if np.uint32 ( self.samples_filtered.size ) <= np.uint32 ( FRAME_LEN_SAMPLES ) :
            self.set_leftovers_idx_for_incomplete_frame ()
            return False
        return True

    def packet_len_validation ( self , packet_end_idx : np.uint32 ) -> bool :
        if packet_end_idx > np.uint32 ( self.samples_filtered.size ) :
            self.set_leftovers_idx_for_incomplete_frame ()
            return False
        return True

    def plot_complex_samples_filtered ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = peaks )

    def __repr__ ( self ) -> str :
        return ( f"{self.packet=}, {self.has_frame=}, {self.has_leftovers=}" )

@dataclass ( slots = True , eq = False )
class RxSamples :

    # Pola uzupełnianie w __post_init__
    samples_raw : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    samples_filtered : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    X_train_samples : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    tx_symbols : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    y_train_tensor : torch.Tensor = field ( default_factory = lambda : torch.tensor ( [] , dtype = torch.complex64 ) , init = False )
    sync_sequence_peaks : NDArray[ np.uint32 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint32 ) , init = False )
    first_symbol_idx : np.uint32 = None # Pierwszy symbol pierwszej ramki.
    concatenates : int = 0
    frames : list[ RxFrame_v0_1_18 ] = field ( init = False , default_factory = list )
    SPS = modulation.SPS
    SPAN = filters.SPAN
    CONCATENATE_THS : int = 10

    def __post_init__ ( self ) -> None :
        pass

    def rx ( self , sdr_ctx : Pluto  | None = None , file_name : str | None = None , concatenate : bool = False ) -> NDArray[ np.complex128 ] :

        if sdr_ctx is not None :
            samples = sdr_ctx.rx ()
        elif file_name is not None :
            if file_name.endswith('.npy'):
                samples = ops_file.open_samples_from_npf ( file_name )
            elif file_name.endswith('.csv'):
                samples = ops_file.open_csv_and_load_np_complex128 ( file_name )
            else:
                raise ValueError(f"Error: unsupported file format for {file_name}! Supported formats: .npy, .csv")
        else :
            raise ValueError ( "Either sdr_ctx or file_name must be provided." )
        if concatenate :
            if self.concatenates < self.CONCATENATE_THS :
                self.concatenates += 1
                #samples_raw = np.concatenate ( [ self.samples_raw , samples ] )
                self.samples_raw = np.append ( self.samples_raw , samples ) # to samo co powyżej, ale append jest szybszy dla 1 elementu
                self.samples_filtered = filters.apply_rrc_rx_convolve_v0_1_18 ( self.samples_raw )
            else :
                raise MemoryError ( f"{self.concatenates=}. To prevent modem performance issues, further concatenation is blocked." )
        else :
            self.samples_raw = samples
            self.samples_filtered = filters.apply_rrc_rx_convolve_v0_1_18 ( self.samples_raw )
        if self.samples_raw.size == self.samples_filtered.size :
            return samples
        else :
            print ( ( f"{self.samples_raw.size=} {self.samples_filtered.size=}"))
            raise ValueError ( f"ERROR! samples_raw.size != samples_filtered.size." )

    def detect_frames ( self , deep : bool = False , samples_filtered : bool = True , correct_samples : bool = False , add_peak_at_0 : bool = False ) -> None :
        
        if correct_samples and not samples_filtered :
            raise ValueError ( "Cannot apply correction without filtering. You must set filter=True to apply correction!" )
        samples = self.samples_filtered.copy () if samples_filtered else self.samples_raw.copy ()
        if correct_samples :
            samples = modulation.zero_quadrature ( corrections.full_compensation_v0_1_5 ( samples , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) ) )

        self.sync_sequence_peaks = detect_sync_sequence_peaks_v0_1_15 ( samples , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) , deep = deep )
        if add_peak_at_0 : self.sync_sequence_peaks = np.insert ( self.sync_sequence_peaks , 0 , 0 )
        previous_processed_idx : np.uint32 = 0
        for idx in self.sync_sequence_peaks :
            if idx > previous_processed_idx or idx == 0 : # idx == 0 jest wtedy kiedy chcemy dodać szczyt na 0, mimo że nie jest on wykryty w detekcji pików, ale chcemy żeby funkcja detect_frames() działała poprawnie nawet wtedy kiedy detekcja pików nie wykryje żadnego piku, a mamy leftoversy z poprzedniego wywołania, które zaczynają się od początku sampli.
                frame = RxFrame_v0_1_18 ( samples_filtered = samples [ idx + filters.FIRST_SYMBOL_OFFSET : ] , first_symbol_abs_idx = idx + filters.FIRST_SYMBOL_OFFSET )
                if frame.has_header :
                    self.frames.append ( frame )
                    previous_processed_idx = frame.frame_end_abs_idx
                    # Dodaj kolejne frame, które zostały wysłane w tym samym strumieniu danych.
                    # Szansa, że jest kolejna ramka jest tylko wtedy jeśli poprzednia była cała.
                    while ( frame.has_frame ) :
                        frame = RxFrame_v0_1_18 ( samples_filtered = samples [ frame.frame_end_abs_idx : ] , first_symbol_abs_idx = frame.frame_end_abs_idx )
                        if frame.has_header :
                            self.frames.append ( frame )
                            previous_processed_idx = frame.frame_end_abs_idx
                else :
                    previous_processed_idx = idx

    def aggregate_frame_and_packet_idxs ( self ) -> NDArray [ np.uint32 ] :

        frame_first_idxs : NDArray [ np.uint32 ] = np.array ( [ frame.first_symbol_abs_idx for frame in self.frames ] , dtype = np.uint32 )
        packet_first_idxs : NDArray [ np.uint32 ] = np.array ( [ frame.packet_first_symbol_abs_idx for frame in self.frames ] , dtype = np.uint32 )
        frame_last_idxs : NDArray [ np.uint32 ] = np.array ( [ ( frame.frame_end_abs_idx - 1 ) for frame in self.frames ] , dtype = np.uint32 )
        return np.concatenate ( [ frame_first_idxs , packet_first_idxs , frame_last_idxs ] )

    def create_X_train_samples_and_y_train_tensor ( self , src_dir : Path , timestamp_group : str , X_train_samples_filtered : bool = False , symbols_src : str = None ) -> np.uint32 :

        if src_dir is None or timestamp_group is None or symbols_src is None :
            raise ValueError ( "ERROR: src_dir, timestamp_group, and symbols_src must be provided." )
        
        self.first_symbol_idx = None
        if self.frames is not None and len ( self.frames ) > 0 :
            self.tx_symbols = self.open_and_load_npf ( filename_and_dirname = f"{src_dir.name}/{timestamp_group}_tx_{symbols_src}.npy" )
            tx_samples = type ( self ) ()
            tx_samples.rx ( file_name = str ( f"{src_dir.name}/{timestamp_group}_tx_samples.npy" ) )
            tx_samples.detect_frames ( deep = False , samples_filtered = False , correct_samples = False , add_peak_at_0 = True )
            for rx_frame in self.frames :
                for tx_frame in tx_samples.frames :
                    if settings["log"]["verbose_2"] : print ( f"rx: {rx_frame.packet_len}	{pad_bits2bytes ( rx_frame.header_bits )}	{rx_frame.first_symbol_abs_idx}" )
                    if settings["log"]["verbose_2"] : print ( f"tx: {tx_frame.packet_len}	{pad_bits2bytes ( tx_frame.header_bits )}	{tx_frame.first_symbol_abs_idx}" )
                    if np.array_equal ( rx_frame.header_bits , tx_frame.header_bits ) :
                        self.first_symbol_idx = rx_frame.first_symbol_abs_idx - tx_frame.first_symbol_abs_idx + filters.FIRST_SYMBOL_OFFSET
                        if settings["log"]["verbose_1"] : print ( f"\r\nRamka {timestamp_group=} dopasowana w: {self.first_symbol_idx=}" )
                        break
                if self.first_symbol_idx is not None :
                    break
        if self.first_symbol_idx is not None :
            self.X_train_samples = self.samples_filtered.copy () if X_train_samples_filtered else self.samples_raw.copy ()
            self.y_train_tensor = torch.zeros ( self.samples_filtered.size if X_train_samples_filtered else self.samples_raw.size , dtype = torch.complex64 )
            self.y_train_tensor[ self.first_symbol_idx : self.first_symbol_idx + self.tx_symbols.size ] = torch.tensor ( self.tx_symbols , dtype = torch.complex64 )
            return self.first_symbol_idx
        else :
            if settings["log"]["verbose_1"] : print ( f"ERROR: No matching frame found for timestamp_group {timestamp_group} in both rx and tx samples." )
            return None

    def clip_X_train_samples_and_y_train_tensor ( self , clipping_mode : str = 'balanced' ) -> None :
        '''
        clipping_mode = 'balanced' : Przycinanie ramki aby stosunek symboli BPSK do 0+j0 był ok. 80 do 20, co pomaga w treningu modelu.
        Nie powinno to być nigdy idealny 80/20, bo w rzeczywistych danych zawsze będzie pewna losowość, ale powinno być blisko tego.
        Poza tym należy dbać o to aby liczba sampli po przycięciu była wielokrotnością SPS i ml.CHUNK_SAMPLES_LEN.
        clipping_mode = 'symbols_only' : Przycinanie ramki tak aby były tylko aktywne symbole BPSK, bez żadnych 0+j0,
        co jest trybem treningowym bardzo trudnym, ale może być przydatny do eksperymentów i porównania z balanced.
        '''
        if self.first_symbol_idx is None or self.first_symbol_idx >= self.samples_raw.size :
            raise ValueError ( f"ERROR: There's a problem with packet.RxSamples.first_symbol_idx." )
        last_sample_idx = self.first_symbol_idx + self.tx_symbols.size
        match clipping_mode :
            case 'balanced' :
                i = ml.CHUNK_SAMPLES_LEN * 10 # mnożnik ma na celu niedopuszczenie do zbyt wysokiego ratio, stosunku symboli BPSK do 0+j0
                clip1 = ( ( self.first_symbol_idx - 1 ) // i ) * i
                clip2 = ( last_sample_idx // i + 1 ) * i
                # Clamping (zapewnienie że nie wyskoczymy poza zakres indeksowania arrayu)
                clip1 = np.maximum ( 0 , clip1 )
                clip2 = np.minimum ( np.uint32 ( self.X_train_samples.size ) , clip2 )
            case 'symbols_only' :
                clip1 = self.first_symbol_idx
                clip2 = last_sample_idx
            case _ :
                raise ValueError ( f"Invalid {clipping_mode=}. Valid options are 'balanced', 'symbols_only'." )
        # NumPy slice jest widokiem, więc .copy() odcina X_train_samples od dużego bufora źródłowego.
        self.X_train_samples = self.X_train_samples [ clip1 : clip2 ].copy ()
        self.y_train_tensor = self.y_train_tensor[ clip1 : clip2 ].clone ()

    def open_and_load_npf ( self , filename_and_dirname : str ) -> NDArray[ np.complex128 ] :

        return ops_file.open_samples_from_npf ( filename_and_dirname )
    
    def save_samples_2_npf ( self , file_name : str , dir_name : str , add_timestamp : bool = False ) -> None :

        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.samples_raw )

    def save_train_data ( self , timestamp_group : str , dir_name : str , add_timestamp : bool = False ) -> None :

        filename = ops_file.add_timestamp_2_filename ( f"{timestamp_group}_X_train_samples.npy" ) if add_timestamp else f"{timestamp_group}_X_train_samples.npy"
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.X_train_samples )
        filename = ops_file.add_timestamp_2_filename ( f"{timestamp_group}_y_train_tensor.pt" ) if add_timestamp else f"{timestamp_group}_y_train_tensor.pt"
        filename_and_dirname = f"{dir_name}/{filename}"
        torch.save ( self.y_train_tensor , filename_and_dirname )

    def plot_tx_symbols ( self , title : str = "" ) -> None :
        plot.complex_waveform_v0_1_6 ( self.tx_symbols , f"{title} {self.tx_symbols.size=}" )

    def plot_samples ( self , title : str = "" , samples_filtered : bool = False , mark_samples : bool = True ) -> None :
        samples = self.samples_filtered if samples_filtered else self.samples_raw
        if mark_samples :
            plot.complex_waveform_v0_1_6 ( samples , f"{title} {samples.size=}" , marker_peaks = self.aggregate_frame_and_packet_idxs () )
        else :
            plot.complex_waveform_v0_1_6 ( samples , f"{title} {samples.size=}" )

    def plot_X_and_y ( self , title : str = "" , mark_samples : bool = True ) -> None :
        marker_idxs = self.aggregate_frame_and_packet_idxs () if mark_samples else None
        plot.complex_waveform_v0_1_6 ( self.X_train_samples , title = f"{title} {self.X_train_samples.size=}" , marker_peaks = marker_idxs )
        plot.flat_tensor_v0_1_18 ( self.y_train_tensor , title = f"{title} {self.y_train_tensor.shape=}" , marker_idx = marker_idxs )

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
class TxPacket_v0_1_18 :
    
    payload_bytes : NDArray[ np.uint8 ]
    
    # Pola uzupełnianie w __post_init__
    bpsk_symbols : NDArray[ np.complex128 ] = field ( init = False )
    bytes : NDArray[ np.uint8 ] = field ( init = False )
    bits : NDArray[ np.uint8 ] = field ( init = False )
    len : np.uint16 = field ( init = False )

    def __post_init__ ( self ) -> None :
        crc32_bytes = create_crc32_bytes ( self.payload_bytes )
        self.bytes = np.concatenate ( [ self.payload_bytes , crc32_bytes ] )
        self.len = np.uint16 ( len ( self.payload_bytes ) + len ( crc32_bytes ) )  # payload + crc32
        self.bits = bytes2bits ( self.bytes )
        self.bpsk_symbols = modulation.create_bpsk_symbols_v0_1_6_fastest_short ( self.bits )

    def __repr__ ( self ) -> str :
        return ( f"{ self.payload_bytes= }, { self.bytes= }, { self.len= }" )

@dataclass ( slots = True , eq = False )
class TxFrame :

    tx_packet : TxPacket_v0_1_18
        
    # Pola uzupełnianie w __post_init__
    frame_start_abs_idx : np.uint32 = field ( init = False , default = np.uint32 ( 0 ) )
    frame_start_abs_first_idx : np.uint32 = field ( init = False , default = np.uint32 ( 0 ) )
    header_bytes : NDArray[ np.uint8 ] = field ( init = False )
    bytes : NDArray[ np.uint8 ] = field ( init = False )
    bits : NDArray[ np.uint8 ] = field ( init = False )
    header_bpsk_symbols : NDArray[ np.complex128 ] = field ( init = False )
    bpsk_symbols : NDArray[ np.complex128 ] = field ( init = False )
    bpsk_symbols_flat_tensor : NDArray[ np.complex64 ] = field ( init = False )
    samples4pluto : NDArray[ np.complex128 ] = field ( init = False )

    def __post_init__ ( self ) -> None :
        sync_sequence_bits : NDArray[ np.uint8 ] = self.create_sync_sequence_bits ()
        packet_len_bits : NDArray[ np.uint8 ] = self.create_packet_len_bits ()
        self.header_bytes = pad_bits2bytes ( np.concatenate ( [ sync_sequence_bits , packet_len_bits ] ) )
        crc32_bytes = self.create_crc32_bytes ( self.header_bytes )
        self.bytes = np.concatenate ( [ self.header_bytes , crc32_bytes , self.tx_packet.bytes ] )
        self.bits = self.create_frame_bits ()
        self.bpsk_symbols = self.create_frame_bpsk_symbols ()
        self.bpsk_symbols_flat_tensor = self.bpsk_symbols_2_flat_tensor ()
        self.samples4pluto = self.create_samples4pluto ()

    def create_sync_sequence_bits ( self ) -> NDArray[ np.uint8 ] :
        return BARKER13_BITS

    def create_packet_len_bits ( self ) -> NDArray[ np.uint8 ] :
        return dec2bits ( self.tx_packet.len , PACKET_LEN_LEN_BITS )

    def create_crc32_bytes ( self , frame_main_bytes ) -> NDArray[ np.uint8 ] :
        return create_crc32_bytes ( frame_main_bytes )
    
    def create_frame_bits ( self ) -> NDArray[ np.uint8 ] :
        return bytes2bits ( self.bytes )

    def create_frame_bpsk_symbols ( self ) -> NDArray[ np.complex128 ] :
        return modulation.create_bpsk_symbols_v0_1_6_fastest_short ( self.bits )

    def bpsk_symbols_2_flat_tensor ( self ) -> NDArray[ np.complex64 ] :
        '''Stworzenie flat_tensor reprezentujacego cały przebieg bpsk_symbols.
        Wstawienie -1+0j, 1+0j dla każdego symbolu reprezentujacego symbol w ramce
        '''
        bpsk_symbols_flat_tensor = np.repeat ( self.bpsk_symbols , modulation.SPS ).astype ( np.complex64 , copy = False )
        return bpsk_symbols_flat_tensor

    def create_samples4pluto ( self ) -> NDArray[ np.complex128 ] :
        samples_filtered = np.ravel ( filters.apply_tx_rrc_filter_v0_1_6 ( self.bpsk_symbols ) ).astype ( np.complex128 , copy = False )
        return sdr.scale_to_pluto_dac_v0_1_11 ( samples = samples_filtered , scale = 1.0 )

    def __repr__ ( self ) -> str :
        return (
            f"{self.bytes.size=}, {self.bpsk_symbols.size=}, {self.bpsk_symbols_flat_tensor.size=}, {self.samples4pluto.size=}" )

@dataclass ( slots = True , eq = False )
class TxSamples :

    payload_bytes : list | tuple | np.ndarray[ np.uint8 ] | None = None

    radio_preamble_bytes : NDArray[ np.uint8 ] = field ( default_factory = lambda : np.array ( settings[ "RADIO_PREAMBLE_BYTES" ] , dtype = np.uint8 ) , init = False )
    samples : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    samples_4_pluto : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    # symbols to symbole wzięte z frames
    symbols : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    # active_samples to symbole wzięte z próbkowania samples w miejscach gdzie powinny być aktywne symbole, ale nie z symboli ramek.
    # Dlatego te symbole mogą się różnić od tych z ramek, bo są wzięte z próbkowania.
    active_samples : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    frames : list[ TxFrame ] = field ( init = False , default_factory = list )
    SPS = modulation.SPS

    def __post_init__ ( self ) -> None :

        if self.payload_bytes is not None and len ( self.payload_bytes ) > 0 :
            self.add_frame ( payload_bytes = self.payload_bytes )

    def add_frame ( self , payload_bytes : list | tuple | np.ndarray[ np.uint8 ] = None ) -> None :

        if payload_bytes is not None and len ( payload_bytes ) > 0 :
            payload_bytes_np_arr = np.asarray ( payload_bytes , dtype = np.uint8 ).ravel ()
            if len ( payload_bytes_np_arr ) > MAX_ALLOWED_PAYLOAD_LEN_BYTES_LEN :
                raise ValueError ( "Error: Payload exceeds maximum allowed length!" )
        else :
            raise ValueError ( "Error: payload_bytes must be provided." )
        tx_frame = self.create_tx_frame ( payload_bytes_np_arr = payload_bytes_np_arr )
        self.frames.append ( tx_frame )
        self.create_samples4pluto_active_symbols_and_active_samples ()

    def create_tx_frame ( self , payload_bytes_np_arr : NDArray[ np.uint8 ] ) -> TxFrame :

        tx_frame_payload = TxPacket_v0_1_18 ( payload_bytes = payload_bytes_np_arr )
        tx_frame = TxFrame ( tx_packet = tx_frame_payload )
        return tx_frame

    def offsets_accuracy_test ( self ) -> None :
        '''
        Zostawić tę funkcję do testowania dokładności offsetów samplowania, żeby mieć pewność że są one poprawne.
        Funkcja ta tworzy samples z ramek, a następnie sprawdza czy symbole aktywne w samples są takie same jak symbole aktywne w ramkach,
        biorąc pod uwagę offsety wynikające z filtracji i próbkowania.
        Skrypt test126-compare_offsets.py jest napisany specjalnie do testowania tej funkcji.
        '''
        frames_bpsk_symbols : NDArray [ np.complex128 ] = np.concatenate ( [ frame.bpsk_symbols for frame in self.frames ] ).astype ( np.complex128 , copy = False )
        if frames_bpsk_symbols.size > 0 :
            samples = np.ravel ( filters.apply_tx_rrc_filter_v0_1_6 ( frames_bpsk_symbols ) ).astype ( np.complex128 , copy = False )
            active_symbols = np.repeat ( frames_bpsk_symbols , self.SPS ).astype ( np.complex128 , copy = False )
            plot.complex_waveform_v0_1_6 ( samples , f"{script_filename} offset_accuracy_test {samples.size=}" )
            plot.complex_waveform_v0_1_6 ( active_symbols , f"{script_filename} offset_accuracy_test {active_symbols.size=}" )
            for i in range ( 4 ) :
                first_active_symbol_idx = np.uint32 ( filters.FIRST_SYMBOL_OFFSET + i -1 )
                last_frame_end_idx = first_active_symbol_idx + frames_bpsk_symbols.size * self.SPS
                active_samples = samples.real[ first_active_symbol_idx : last_frame_end_idx ]
                active_samples = np.where ( active_samples < 0.0 , np.complex128 ( -1.0 + 0j ) , np.complex128 ( 1.0 + 0j ) )
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

    def create_samples4pluto_active_symbols_and_active_samples ( self ) -> None :

        frames_bpsk_symbols : NDArray [ np.complex128 ] = np.concatenate ( [ frame.bpsk_symbols for frame in self.frames ] ).astype ( np.complex128 , copy = False )
        radio_preamble_bits : NDArray [ np.complex128 ] = bytes2bits ( self.radio_preamble_bytes )
        radio_preamble_bpsk_symbols : NDArray [ np.complex128 ] = modulation.create_bpsk_symbols_v0_1_6_fastest_short ( radio_preamble_bits )
        if frames_bpsk_symbols.size > 0 :
            self.samples = np.ravel ( filters.apply_tx_rrc_filter_v0_1_6 ( frames_bpsk_symbols ) ).astype ( np.complex128 , copy = False )
            # Specjalnie tylko do samples_4_pluto używam preambuły radiowej, żeby nigdzie indziej się nie wliczała - na razie.
            bpsk_symbols_with_preamble = np.concatenate ( [ radio_preamble_bpsk_symbols , frames_bpsk_symbols ] ).astype ( np.complex128 )
            samples_with_preamble = np.ravel ( filters.apply_tx_rrc_filter_v0_1_6 ( bpsk_symbols_with_preamble ) ).astype ( np.complex128 , copy = False )
            self.samples_4_pluto = sdr.scale_to_pluto_dac_v0_1_11 ( samples = samples_with_preamble , scale = 1.0 )
            
            self.symbols = np.repeat ( frames_bpsk_symbols , self.SPS ).astype ( np.complex128 , copy = False )
            first_active_symbol_idx = np.uint32 ( filters.FIRST_SYMBOL_OFFSET )
            last_frame_end_idx = first_active_symbol_idx + frames_bpsk_symbols.size * self.SPS
            active_samples = self.samples.real[ first_active_symbol_idx : last_frame_end_idx ]
            self.active_samples = np.where ( active_samples < 0.0 , np.complex128 ( -1.0 + 0j ) , np.complex128 ( 1.0 + 0j ) )

    def tx ( self , sdr_ctx : Pluto , repeat : np.uint32 = 1 ) -> None :
        sdr_ctx.tx_destroy_buffer ()
        sdr_ctx.tx_cyclic_buffer = False
        if repeat < 1 or repeat > 4294967295 :
            raise ValueError ( "Error: reapt value is out of the range! Allowed range is 1 to 4294967295." )
        while repeat :
            sdr_ctx.tx ( self.samples_4_pluto ) # Uwaga w innych nie wprowadziłem tej zmiany, tj. wo_mute
            repeat -= 1

    def save_samples_4_pluto_2_npf ( self , file_name : str , dir_name : str , add_timestamp : bool = False ) -> None :
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.samples_4_pluto )

    def save_samples_2_npf ( self , file_name : str , dir_name : str , add_timestamp : bool = False ) -> None :
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.samples )

    def save_active_samples_2_npf ( self , file_name : str , dir_name : str , add_timestamp : bool = False ) -> None :
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.active_samples )

    def save_symbols_2_npf ( self , file_name : str , dir_name : str , add_timestamp : bool = False ) -> None :
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.symbols )

    def plot_samples ( self , title :str = "" , markers : bool = True ) -> None :
        if markers :
            idx = np.array ( [ filters.FIRST_SYMBOL_OFFSET ] , dtype = np.uint32 )
            plot.complex_waveform_v0_1_6 ( self.samples , f"{title} {self.samples.size=}" , marker_peaks = idx )
        else :
            plot.complex_waveform_v0_1_6 ( self.samples , f"{title} {self.samples.size=}" )

    def plot_samples_4_pluto ( self , title : str = "" , markers : bool = True ) -> None :
        if markers :
            idx = np.array ( [ filters.FIRST_SYMBOL_OFFSET ] , dtype = np.uint32 )
            plot.complex_waveform_v0_1_6 ( self.samples_4_pluto , f"{title} {self.samples_4_pluto.size=}" , marker_peaks = idx )
        else :
            plot.complex_waveform_v0_1_6 ( self.samples_4_pluto , f"{title} {self.samples_4_pluto.size=}" )

    def plot_symbols ( self , title : str = "" ) -> None :
        #plot.plot_bpsk_symbols_v2 ( symbols = self.symbols , title = f"{title} tx {self.symbols.size=}" )
        plot.complex_waveform_v0_1_6 ( self.symbols , f"{title} {self.symbols.size=}" )

    def plot_active_samples ( self , title : str = "" ) -> None :
        plot.complex_waveform_v0_1_6 ( self.active_samples , f"{title} {self.active_samples.size=}" )

    def plot_samples_4_pluto_spectrum ( self , title : str = "" ) -> None :
        plot.spectrum_occupancy ( self.samples_4_pluto , 1024 , title )

    def __repr__ ( self ) -> str :
        return ( f"{ self.bpsk_symbols.size= }, { self.samples_4_pluto.size= }" )

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
        return ( f"{ self.pluto_tx_ctx= }" )