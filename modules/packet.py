import csv
from fileinput import filename
import numpy as np
import os
import time as t
import tomllib
import torch
import zlib

from adi import Pluto
from dataclasses import dataclass , field
from modules import corrections , filters , modulation, ml , ops_file, plot , sdr
from numpy.typing import NDArray

from pathlib import Path
from scipy.signal import find_peaks
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    sync_sequence_peak_abs_idx : np.uint32

    # Pola uzupełnianie w __post_init__
    SPS = modulation.SPS
    SPAN = filters.SPAN
    header_bpsk_symbols : NDArray[ np.complex128 ] = field ( init = False )
    header_bits : NDArray[ np.uint8 ] = field ( init = False )
    frame_start_abs_idx : np.uint32 = field ( init = False )
    frame_start_abs_first_sample_idx : np.uint32 = field ( init = False )
    frame_end_abs_idx : np.uint32 = field ( init = False )
    packet_len : np.uint16 = field ( init = False )
    packet_start_abs_idx : NDArray[ np.uint32 ] = field ( init = False )
    leftovers_start_abs_idx : np.uint32 = field ( init = False )
    has_header : bool = False # przydate jeśli nie wykryje pakietu/payload
    has_frame : bool = False # ustawiany dopiero po walidacji pakietu, wcześniej używamy tylko lokalnego has_header
    has_leftovers : bool = False
    # do zapamiętania jako tip przed skasowaniem payload_bytes : NDArray[ np.uint8 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint8 ) , init = False )
    packet : RxPacket_v0_1_18 = field ( init = False )
    
    def __post_init__ ( self ) -> None :
        if not self.frame_len_validation () :
            return self.sync_sequence_peak_abs_idx
        self.process_packet ()
    
    def process_packet ( self ) -> None :
        sync_sequence_start_idx = 0 #self.SPAN * self.SPS // 2 # Można wróć do sprzed zmiany w git: 81d3093def3219f03c37e046d4f5141864f28c2c
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
                        self.frame_start_abs_idx = self.sync_sequence_peak_abs_idx + sync_sequence_start_idx
                        #self.frame_start_abs_first_sample_idx = self.frame_start_abs_idx - ( self.frame_start_abs_idx % self.SPS ) # szukam pierwszego sample, który jest początkiem ramki, czyli tego, który jest podzielny przez SPS, bo jeśli znalazłem dopasowanie ramki, to jest ono w jednym z 4 punktów odd
                        self.frame_start_abs_first_sample_idx = self.frame_start_abs_idx - self.SPS // 2
                        self.packet_start_abs_idx = self.sync_sequence_peak_abs_idx + crc32_end_idx # używać tylko jeśli self.has_packet, inaczej może być poza zakresem sampli
                        self.frame_end_abs_idx = self.sync_sequence_peak_abs_idx + packet_end_idx # używać tylko jeśli self.has_packet, inaczej może być poza zakresem sampli
                        self.header_bits = np.concatenate ( [ sync_sequence_bits , packet_len_bits , crc32_bits ] )
                        self.header_bpsk_symbols = modulation.bits_2_bpsk_symbols_v0_1_18 ( self.header_bits )
                        add2log_packet ( f"{t.time()},{sync_sequence_start_idx=},{self.has_header=},{self.frame_start_abs_idx=}" )
                        if not self.packet_len_validation ( packet_end_idx ) :
                            add2log_packet ( f"{t.time()},{self.has_header=},{sync_sequence_start_idx=},{self.frame_start_abs_idx=}" )
                            if settings["log"]["verbose_2"] : print ( f"{self.sync_sequence_peak_abs_idx=} {samples_name} {frame_name=} {has_sync_sequence=}, {self.has_header=}" )
                            return
                        packet = RxPacket_v0_1_18 ( samples_filtered = self.samples_filtered [ crc32_end_idx : packet_end_idx ] )
                        if packet.has_packet :
                            self.has_frame = True # has_frame jeśli ma header i pakiet, inaczej nie ma całej ramki
                            #self.bpsk_symbols = np.concatenate ( [ sync_sequence_symbols , packet_len_symbols , crc32_symbols , packet.packet_symbols ] )
                            #self.header_bpsk_symbols = modulation.bits_2_bpsk_symbols_v0_1_18 ( np.concatenate ( [ sync_sequence_bits , packet_len_bits , crc32_bits ] ) , sps = self.SPS )
                            self.packet = packet
                            add2log_packet(f"{t.time()},{packet.has_packet=},{crc32_end_idx=}")
                            if settings["log"]["verbose_2"] : print ( f"{sync_sequence_start_idx=} {has_sync_sequence=}, {self.frame_start_abs_idx=} {self.has_frame=}, {packet.has_packet=}" )
                            return
        if settings["log"]["verbose_2"] : print ( f"{self.frame_sync_sequence_peak_abs_idx=} {has_sync_sequence=}, {self.has_frame=}" )
        return

    def samples2bits ( self , samples : NDArray[ np.complex128 ] ) -> NDArray[ np.uint8 ] :
        return modulation.bpsk_symbols_2_bits_v0_1_7 ( samples [ : : self.sps ] )

    def bits2uint16 ( self , bits : NDArray[ np.uint8 ] ) -> np.uint16 :
        return np.uint16 ( bits_2_int ( bits ) )

    def samples2bytes ( self , samples : NDArray[ np.complex128 ] ) -> NDArray[ np.uint8 ] :
        bits = self.samples2bits ( samples )
        return pad_bits2bytes ( bits )
    
    def set_leftovers_idx_for_incomplete_frame ( self ) -> None :
        if settings["log"]["verbose_2"] : print ( f"Samples at index { self.sync_sequence_peak_abs_idx } is too close to the end of samples to contain a complete frame. Skipping." )
        self.leftovers_start_abs_idx = self.sync_sequence_peak_abs_idx - self.SPAN * self.SPS // 2 # Bez cofniecia się do początku filtra RRC nie ma wykrycia ramnki i pakietu w następnym wywołaniu
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
class RxSamples_v0_1_18 :

    # Pola uzupełnianie w __post_init__
    samples : NDArray[ np.complex128 ] = field ( init = False )
    y_train_np_array : NDArray[ np.complex128 ] = field ( init = False )
    y_train_tensor : torch.Tensor = field ( init = False )
    tx_active_symbols : NDArray[ np.complex128 ] = field ( init = False )
    flat_tensor : torch.Tensor = field ( init = False )
    #samples : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    samples_filtered : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    samples_corrected : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    has_amp_greater_than_ths : bool = False
    SPS = modulation.SPS
    SPAN = filters.SPAN
    ths : float = 1000.0
    frames : list[ RxFrame_v0_1_18 ] = field ( init = False , default_factory = list )
    leftovers : NDArray[ np.complex128 ] | None = field ( default = None )

    samples_corrected_len : np.uint32 = field ( init = False )
    sync_sequence_peaks : NDArray[ np.uint32 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint32 ) , init = False )
    has_leftovers : bool = False
    leftovers_start_idx : np.uint32 = field ( init = False )

    def __post_init__ ( self ) -> None :
            self.samples = np.array ( [] , dtype = np.complex128 )
            #self.samples_filtered = np.array ( [] , dtype = np.complex128 )
            #self.samples_corrected = np.array ( [] , dtype = np.complex128 )

    def rx ( self , sdr_ctx : Pluto  | None = None , previous_samples_leftovers : NDArray[ np.complex128 ] | None = None , samples_filename : str | None = None , concatenate : bool = False ) -> None :
        '''
        concatenated: powoduje nawarstwienie nowych sampli na stare.
        UWAGA! Nieostrożne stosowanie może spowodować zawieszenie komputera z powodu braku pamięci RAM,
        jeśli próbujemy nawarstwić zbyt dużo sampli.
        Używaj z rozwagą i monitoruj zużycie pamięci.
        '''
        if sdr_ctx is not None :
            samples = sdr_ctx.rx ()
        elif samples_filename is not None :
            if samples_filename.endswith('.npy'):
                samples = ops_file.open_samples_from_npf ( samples_filename )
            elif samples_filename.endswith('.csv'):
                samples = ops_file.open_csv_and_load_np_complex128 ( samples_filename )
            else:
                raise ValueError(f"Error: unsupported file format for {samples_filename}! Supported formats: .npy, .csv")
        else :
            raise ValueError ( "Either sdr_ctx or samples_filename must be provided." )
        if concatenate :
            self.samples = np.concatenate ( [ self.samples , samples ] )
        else :
            self.samples = samples
        if previous_samples_leftovers is not None :
            self.samples = np.concatenate ( [ previous_samples_leftovers , self.samples ] )
        self.sample_initial_assesment ()

    def sample_initial_assesment (self) -> None :
        self.has_amp_greater_than_ths = np.any ( np.abs ( self.samples ) > self.ths )

    def detect_frames ( self , deep : bool = False , filter : bool = False , correct : bool = False , add_peak_at_0 : bool = False ) -> None :
        if filter :
            self.filter_samples ()
        else :
            self.samples_filtered = self.samples
        if correct :
            self.correct_samples ()
        else :
            self.samples_corrected = self.samples_filtered
        self.has_leftovers = False
        self.samples_corrected_len = np.uint32 ( len ( self.samples_corrected ) )
        self.sync_sequence_peaks = detect_sync_sequence_peaks_v0_1_15 ( self.samples_corrected , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) , deep = deep )
        if add_peak_at_0 : self.sync_sequence_peaks = np.insert ( self.sync_sequence_peaks , 0 , 0 )
        previous_processed_idx : np.uint32 = 0
        for idx in self.sync_sequence_peaks :
            if idx > previous_processed_idx or idx == 0 : # idx == 0 jest wtedy kiedy chcemy dodać szczyt na 0, mimo że nie jest on wykryty w detekcji pików, ale chcemy żeby funkcja detect_frames() działała poprawnie nawet wtedy kiedy detekcja pików nie wykryje żadnego piku, a mamy leftoversy z poprzedniego wywołania, które zaczynają się od początku sampli.
                frame = RxFrame_v0_1_18 ( samples_filtered = self.samples_corrected [ idx + filters.PEAK_TO_ACTIVE_SAMPLE_OFFSET : ] , sync_sequence_peak_abs_idx = idx + filters.PEAK_TO_ACTIVE_SAMPLE_OFFSET )
                if frame.has_header :
                    self.frames.append ( frame )
                    previous_processed_idx = frame.frame_end_abs_idx
                else :
                    previous_processed_idx = idx
                if frame.has_leftovers :
                    self.has_leftovers = True
                    self.leftovers_start_idx = frame.leftovers_start_abs_idx
                    break
        if not self.has_leftovers :
            self.leftovers_start_idx = self.samples_corrected_len - SYNC_SEQUENCE_LEN_SAMPLES - self.SPAN * self.SPS // 2
        self.clip_samples_leftovers ()
        #self.y_train_tensor = self.y_train_tensor_from_frames () # To musi zostać zmienione i zastępowane plikiem z tx

    def clip_samples_for_training ( self ) -> None :
        '''Przycinanie ramki aby stosunek symboli BPSK do 0+j0 był ok. 80 do 20, co pomaga w treningu modelu.
        Nie powinno to być nigdy idealny 80/20, bo w rzeczywistych danych zawsze będzie pewna losowość, ale powinno być blisko tego.
        Poza tym należy dabć o to aby liczba sampli po przycięciu była wielokrotnością SPS i ml.CHUNK_SAMPLES_LEN.'''
        i = ml.CHUNK_SAMPLES_LEN * 10 # mnożnik ma na celu niedopuszczenie do zbyt wysokiego ratio, stosunku symboli BPSK do 0+j0
        total_bpsk_symbols = 0
        first_bpsk_symbol_idx = self.frames[ 0 ].frame_start_abs_idx
        last_bpsk_symbol_idx = self.frames[ -1 ].frame_end_abs_idx
        leftovers_start_idx = self.leftovers_start_idx
        total_bpsk_symbols = sum ( frame.header_bpsk_symbols.size + frame.packet.bpsk_symbols.size for frame in self.frames )
        ratio : float = total_bpsk_symbols / self.samples_corrected_len
        clip1 = ( ( first_bpsk_symbol_idx - 1 ) // i ) * i
        clip2 = ( last_bpsk_symbol_idx // i + 1) * i
        # Clamping (zapewnienie że nie wyskoczymy poza zakres indeksowania arrayu)
        clip1 = np.maximum ( 0 , clip1 )
        clip2 = np.minimum ( self.samples_corrected_len , clip2 )
        ratio_clipped = total_bpsk_symbols / ( clip2 - clip1 )
        print ( f"{clip1=} , {clip2=} , {ratio=:.2f} , {ratio_clipped=:.2f}" )
        clipped_samples = self.clip_samples_corrected ( self.samples_corrected , clip1 , clip2 )
        self.reset_frame_detection ()
        self.samples = clipped_samples
        self.sample_initial_assesment ()
        # Poniższa konfiguracja argumentów ma zapewnić, że funkcja detect_frames () będzie działać poprawnie na przyciętych, filtrowanych i po korekcji samplach,
        # bez ponownego filtrowania i korygowania.
        self.detect_frames ( deep = False , filter = False , correct = False )

    def clip_Xy_samples_wo_mute_for_training ( self , frames_first_sample_idx : np.uint32 = None , timestamp_group : str = None , dir_name : str = None ) -> None :
        # Przcięcie self.samples, self.samples_filtered, y_train_np_array i y_train_tensor do zakresu między frames_first_sample_idx a frames_last_sample_idx.
        if frames_first_sample_idx is None or timestamp_group is None or dir_name is None :
            raise ValueError ( f"All arguments must be provided." )
        if frames_first_sample_idx >= self.samples.size :
            raise ValueError ( f"{frames_first_sample_idx=} must be less than  {self.samples.size}." )
        '''Przycinanie ramki aby stosunek symboli BPSK do 0+j0 był ok. 80 do 20, co pomaga w treningu modelu.
        Nie powinno to być nigdy idealny 80/20, bo w rzeczywistych danych zawsze będzie pewna losowość, ale powinno być blisko tego.
        Poza tym należy dabć o to aby liczba sampli po przycięciu była wielokrotnością SPS i ml.CHUNK_SAMPLES_LEN.'''
        frames_last_sample_idx = frames_first_sample_idx + self.tx_active_symbols.size
        i = ml.CHUNK_SAMPLES_LEN * 10 # mnożnik ma na celu niedopuszczenie do zbyt wysokiego ratio, stosunku symboli BPSK do 0+j0
        ratio : float = self.tx_active_symbols.size / self.samples.size
        clip1 = ( ( frames_first_sample_idx - 1 ) // i ) * i
        clip2 = ( frames_last_sample_idx // i + 1) * i
        # Clamping (zapewnienie że nie wyskoczymy poza zakres indeksowania arrayu)
        clip1 = np.maximum ( 0 , clip1 )
        clip2 = np.minimum ( np.uint32 ( self.samples.size ) , clip2 )
        ratio_clipped = self.tx_active_symbols.size / ( clip2 - clip1 )
        print ( f"{clip1=} , {clip2=} , {ratio=:.2f} , {ratio_clipped=:.2f}" )
        self.samples = self.clip_samples_corrected ( self.samples , clip1 , clip2 )
        # Poniższe 2 komendy mają sens jeśli rzeczywiście filtrowałeś i korygowałeś sygnał.
        self.samples_filtered = self.clip_samples_corrected ( self.samples_filtered , clip1 , clip2 )
        self.samples_corrected = self.clip_samples_corrected ( self.samples_corrected , clip1 , clip2 )
        self.y_train_np_array = self.y_train_np_array[ clip1 : clip2 ]
        # Zapisanie self.y_train_np_array do self.y_train_tensor w typie torch.complex64 i odpowiednim formacie i shape do odczytu przez pętlę test129-training.py
        self.y_train_tensor = torch.from_numpy ( self.y_train_np_array.astype ( np.complex64 ) )
        self.save_tensor_2_pt ( tensor = self.y_train_tensor , file_name = f"{timestamp_group}_y_train" , dir_name = dir_name )
        self.save_complex_samples_2_npf_v0_1_20 ( samples = self.samples , file_name = f"{timestamp_group}_rx_samples" , dir_name = dir_name , add_timestamp = False )
        self.save_complex_samples_2_npf_v0_1_20 ( samples = self.samples_filtered , file_name = f"{timestamp_group}_rx_samples_filtered" , dir_name = dir_name , add_timestamp = False )
        self.save_complex_samples_2_npf_v0_1_20 ( samples = self.samples_corrected , file_name = f"{timestamp_group}_rx_samples_corrected" , dir_name = dir_name , add_timestamp = False )

    def reset_frame_detection ( self ) -> None :
        self.samples = self.samples_filtered = self.samples_corrected = np.array ( [] , dtype = np.complex128 )
        self.frames = []
        self.leftovers = None
        self.has_leftovers = False
        self.leftovers_start_idx = np.uint32 ( 0 )
        self.y_train_tensor = torch.tensor ( [] ,dtype = torch.complex64 )
        self.has_amp_greater_than_ths = False
        self.samples_corrected_len = np.uint32 ( 0 )
        self.sync_sequence_peaks = np.array ( [] , dtype = np.uint32 )
        self.has_leftovers = False
        self.leftovers_start_idx = np.uint32 ( 0 )

    def filter_samples ( self ) -> None :
        self.samples_filtered = filters.apply_rrc_rx_convolve_v0_1_18 ( self.samples )

    def correct_samples ( self ) -> None :
        self.samples_corrected = modulation.zero_quadrature ( corrections.full_compensation_v0_1_5 ( self.samples_filtered , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) ) )

    def plot_complex_samples ( self , title = "" , marker_all_samples : bool = False , markers_first_active_samples : bool = True ) -> None :
        if markers_first_active_samples :
            frames_first_sample_idx = np.array ( [ frame.frame_start_abs_first_sample_idx for frame in self.frames ] , dtype = np.uint32 )
            plot.complex_waveform_v0_1_6 ( self.samples , f"{title} {self.samples.size=}, {frames_first_sample_idx.size=}" , marker_squares = marker_all_samples , marker_peaks = frames_first_sample_idx )
        else :
            plot.complex_waveform_v0_1_6 ( self.samples , f"{title} {self.samples.size=}" )
        
    def plot_complex_samples_filtered ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"RxSamples filtered {title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = peaks )

    def plot_tensor ( self , title : str = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None , frame_idx : int | None = None ) -> None :
        plot.tensor_waveform_v0_1_16 ( self.y_train_tensor , title = f"y_train_tensor {title}" , marker_squares = marker , marker_peaks = peaks , frame_idx = frame_idx )

    def plot_flat_tensor ( self , title : str = "" ) -> None :
        frames_start_idx = np.array ( [ frame.frame_start_abs_first_sample_idx for frame in self.frames ] , dtype = np.uint32 )
        flat_tensor = self.flat_tensor_from_y_train ()
        plot.tensor_waveform_v0_1_16 ( flat_tensor , title = f"RxSamples y_train_tensor {title}" , marker_peaks = frames_start_idx )

    def plot_complex_samples_corrected_v0_1_20 ( self , title = "" , markers_first_active_samples : bool = False ) -> None :
        if markers_first_active_samples :
            frames_first_sample_idx = np.array ( [ frame.frame_start_abs_first_sample_idx for frame in self.frames ] , dtype = np.uint32 )
            plot.complex_waveform_v0_1_6 ( self.samples_corrected , f"{title} {self.samples_corrected.size=}, {frames_first_sample_idx.size=}" , marker_squares = markers_first_active_samples , marker_peaks = frames_first_sample_idx )
        else :
            plot.complex_waveform_v0_1_6 ( self.samples_corrected , f"{title} {self.samples_corrected.size=}" )

    def plot_complex_samples_corrected ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_corrected , f"RxSamples corrected {title} {self.samples_corrected.size=}" , marker_squares = marker , marker_peaks = peaks )

    def save_complex_samples2npf_v0_1_18 ( self , file_name : str , dir_name : str , add_timestamp : bool = True ) -> None :
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.samples )

    def save_complex_samples_2_npf_v0_1_20 ( self , samples : NDArray[ np.complex128 ] = None , file_name : str = None, dir_name : str = None , add_timestamp : bool = False ) -> None :
        if file_name is None or not isinstance ( samples , np.ndarray ) or samples.dtype != np.complex128 :
            raise TypeError ( "file_name must be provided and samples must be numpy.ndarray with dtype=np.complex128" )
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        if dir_name is not None :
            Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
            filename_and_dirname = f"{dir_name}/{filename}"
        else :
            filename_and_dirname = filename
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , samples )

    def save_complex_samples_2_npf ( self , file_name : str , dir_name : str ) -> None :
        filename_with_timestamp = ops_file.add_timestamp_2_filename ( file_name )
        filename_with_timestamp_and_dir = f"{dir_name}/{filename_with_timestamp}"
        ops_file.save_complex_samples_2_npf ( filename_with_timestamp_and_dir , self.samples )

    def save_complex_samples_2_csv ( self , file_name : str ) -> None :
        filename_with_timestamp = ops_file.add_timestamp_2_filename ( file_name )
        ops_file.save_complex_samples_2_csv ( filename_with_timestamp , self.samples )

    def analyze ( self ) -> None :
        sdr.analyze_rx_signal ( self.samples )

    def clip_samples ( self , start : np.uint32 , end : np.uint32 ) -> None :
        if start < 0 or end > ( self.samples.size - 1 ) :
            raise ValueError ( "Start must be >= 0 & end <= samples length" )
        if start >= end :
            raise ValueError ( "Start must be < end" )
        #self.samples_filtered = self.samples_filtered [ start : end + 1 ]
        self.samples = self.samples [ start : end ]

    def clip_samples_filtered ( self , start : np.uint32 , end : np.uint32 ) -> None :
        if start < 0 or end > ( self.samples_filtered.size - 1 ) :
            raise ValueError ( "Start must be >= 0 & end <= samples_filtered length" )
        if start >= end :
            raise ValueError ( "Start must be < end" )
        #self.samples_filtered = self.samples_filtered [ start : end + 1 ]
        self.samples_filtered = self.samples_filtered [ start : end ]

    def clip_samples_corrected ( self , samples : NDArray[ np.complex128 ] , start : np.uint32 , end : np.uint32 ) -> NDArray[ np.complex128 ] :
        if start < 0 or end > ( samples.size - 1 ) :
            raise ValueError ( "Start must be >= 0 & end <= samples length" )
        if start >= end :
            raise ValueError ( "Start must be < end" )
        return samples [ start : end ]

    def clip_samples_leftovers ( self ) -> None :
        self.leftovers = self.samples [ self.leftovers_start_idx : ]

    def symbols_2_flat_tensor ( self ) -> torch.Tensor :
        # Złożenie wszystkich bpsk_symbols z wszystkich ramek w jeden strumień w kolejności wysyłania we flat tensor w celu porównania z tensorami zapisany podczas tx
        # Tutaj są tylko tensory wszystkich ramek bez ich pozycji i bez tensorów 0+j0 dla sampli bez ramek.
        frames_with_packet = [ frame for frame in self.frames if frame.has_frame ]
        if frames_with_packet :
            all_symbols = np.concatenate (
                [
                    symbols
                    for frame in frames_with_packet
                    for symbols in ( frame.header_bpsk_symbols , frame.packet.bpsk_symbols )
                ]
            ).astype ( np.complex64 , copy = False )
        else :
            all_symbols = np.array ( [] , dtype = np.complex64 )
        return torch.from_numpy ( all_symbols )

    def samples_2_flat_tensor ( self ) -> None :
        '''Stworzenie flat_tensor w self.flat_tensor reprezentujacych cały przebieg. Wstawienie par -1+0j, 1+0j dla każdego symbolu reprezentujacego symbol w ramce
        oraz wstawienie 0+0j wszędzie tam gdzie w przebiegu samples nie ma symboli wynikajacych z ramek.'''
        self.flat_tensor = np.zeros ( self.samples.size , dtype = np.complex64 )
        for frame in self.frames :
            frame_start_idx = np.uint32 ( frame.frame_start_abs_idx )
            frame_end_idx = frame_start_idx + frame.bpsk_symbols.size * self.SPS
            if frame_start_idx >= self.flat_tensor.size or frame.bpsk_symbols.size == 0 :
                continue
            if frame_end_idx > self.flat_tensor.size :
                frame_end_idx = self.flat_tensor.size
            symbols_repeated = np.repeat ( frame.bpsk_symbols , self.SPS )[:frame_end_idx - frame_start_idx]
            self.flat_tensor [ frame_start_idx : frame_end_idx ] = symbols_repeated.astype ( np.complex64 , copy = False )

    def y_train_tensor_from_frames ( self ) -> torch.Tensor :
        y_train_symbols = np.zeros ( self.samples.size , dtype = np.complex64 )
        for frame in self.frames :
            if frame.has_frame :
                frame_symbols = np.concatenate ( [ frame.header_bpsk_symbols , frame.packet.bpsk_symbols ] ).astype ( np.complex64 , copy = False )
                frame_start_idx = int ( frame.frame_start_abs_idx )
                if frame_start_idx >= y_train_symbols.size or frame_symbols.size == 0 :
                    continue
                frame_end_idx = min ( frame_start_idx + frame_symbols.size , y_train_symbols.size )
                y_train_symbols[ frame_start_idx : frame_end_idx ] = frame_symbols[ : frame_end_idx - frame_start_idx ]
        return torch.from_numpy ( y_train_symbols )

    def active_symbols_2_y_train_tensor ( self , file_name : str , dir_name : str , rx_frames_first_sample_idx : np.uint32 = None ) -> torch.Tensor :
        if rx_frames_first_sample_idx is None :
            print ( "ERROR: rx_frames_first_sample_idx must be provided." )
            return None
        self.y_train_tensor = torch.zeros ( self.samples.size , dtype = torch.complex64 )
        last_sample_idx = self.samples.size
        self.y_train_tensor [ rx_frames_first_sample_idx : last_sample_idx ] = 0 + 0j


    def flat_tensor_from_y_train ( self ) -> torch.Tensor :
        if not isinstance ( self.y_train_tensor , torch.Tensor ) or not torch.is_complex ( self.y_train_tensor ) :
            raise TypeError ( "y_train_tensor must be a complex torch.Tensor" )
        return torch.stack ( [ self.y_train_tensor.real , self.y_train_tensor.imag ] )

    def save_frames2y_train_tensor ( self , file_name : str , dir_name : str ) -> None :
        if not isinstance ( self.y_train_tensor , torch.Tensor ) or self.y_train_tensor.dtype != torch.complex64 :
            raise TypeError ( "y_train_tensor must be torch.Tensor with dtype=torch.complex64" )
        tensor_filename = f"{dir_name}/{file_name}.pt"
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        torch.save ( self.y_train_tensor , tensor_filename )

    def save_tensor_2_pt ( self , tensor : torch.Tensor = None , file_name : str = None , dir_name : str = None ) -> None :
        if tensor is None or file_name is None or dir_name is None or not isinstance ( tensor , torch.Tensor ) or tensor.dtype != torch.complex64 :
            raise TypeError ( "All argument must be provided" )
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        tensor_filename = f"{dir_name}/{file_name}.pt"
        torch.save ( tensor , tensor_filename )

    def __repr__ ( self ) -> str :
        for frame in self.frames :
            if frame.has_frame :
                frame_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( np.concatenate ( [ frame.header_bpsk_symbols[ : : self.SPS ] , frame.packet.bpsk_symbols[ : : self.SPS ] ] ) )
                #print ( f"{ frame_bits.size=}, {frame.frame_start_abs_idx=}, {frame_bits[ : 10 ]}" )
        return ( f"{self.samples.size=}, {self.samples.dtype=}")

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
class TxFrame_v0_1_18 :

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
class TxSamples_v0_1_18 :
    '''Są 2 tryby tworzenia tx_samples. 
    1. samples_w_muting:
        Concatenując sample utworzone w tx_frames. Wtedy mamy ładne przerwy między ramkami,
        bo każda ramka jest filtrowana osobno i pomiędzy ramkami są przerwy bez sygnału z symbolami.
    2. samples_wo_muting:
        Tworząc ciągłe samples w tx_samples na podstawie symboli bpsk z tx_frames.
        Wtedy nie ma przerw między ramkami, bo tworzymy ciągły strumień symboli bpsk z ramek i filtrujemy go jako całość.'''
    payload_bytes : list | tuple | np.ndarray[ np.uint8 ] | None = None
    payload_bits : list | tuple | np.ndarray[ np.uint8 ] | None = None

    # Pola uzupełnianie w __post_init__
    bytes : np.ndarray[ np.uint8 ] = field ( init = False )
    bpsk_symbols : NDArray[ np.complex128 ] = field ( init = False )
    samples : NDArray[ np.complex128 ] = field ( init = False )
    samples4pluto : NDArray[ np.complex128 ] = field ( init = False )
    flat_tensor : NDArray[ np.complex128 ] = field ( init = False )
    frames : list[ TxFrame_v0_1_18 ] = field ( init = False , default_factory = list )

    samples4pluto_wo_mute : NDArray[ np.complex128 ] = field ( init = False )
    samples_flat_tensor_wo_mute : NDArray[ np.complex64 ] = field ( init = False )
    symbols_flat_tensor_wo_mute : torch.Tensor = field ( init = False )
    active_symbols_wo_mute : NDArray[ np.complex128 ] = field ( init = False )

    SPS = modulation.SPS

    def __post_init__ ( self ) -> None :
        self.create_empty_complex_samples ()
        if self.payload_bytes is not None and len ( self.payload_bytes ) > 0 :
            self.add_frame ( payload_bytes = self.payload_bytes )
        elif self.payload_bits is not None and len ( self.payload_bits ) > 0 :
            self.add_frame ( payload_bits = self.payload_bits )

    def create_empty_complex_samples ( self ) -> None :
        self.bytes = np.array ( [] , dtype = np.uint8 )
        self.bpsk_symbols = np.array ( [] , dtype = np.complex128 )
        self.flat_tensor = np.array ( [] , dtype = np.complex128 )
        self.samples = np.array ( [] , dtype = np.complex128 )
        self.samples4pluto = np.array ( [] , dtype = np.complex128 )
        self.samples4pluto_wo_mute = np.array ( [] , dtype = np.complex128 )
        self.samples_flat_tensor_wo_mute = np.array ( [] , dtype = np.complex64 )
        self.symbols_flat_tensor_wo_mute = torch.empty ( 0 , dtype = torch.complex64 )

    def create_tx_frame ( self , payload_bytes : np.ndarray[ np.uint8 ] ) -> TxFrame_v0_1_18 :
        tx_frame_payload = TxPacket_v0_1_18 ( payload_bytes = payload_bytes )
        tx_frame = TxFrame_v0_1_18 ( tx_packet = tx_frame_payload )
        return tx_frame

    def add_frame ( self , payload_bytes : list | tuple | np.ndarray[ np.uint8 ] = None , payload_bits : list | tuple | np.ndarray[ np.uint8 ] = None ) -> None :
        if payload_bytes is not None and len ( payload_bytes ) > 0 :
            payload_bytes_arr = np.asarray ( payload_bytes , dtype = np.uint8 ).ravel ()
            if payload_bytes_arr.max () > 255 :
                raise ValueError ( "Error: Payload has not all values in 0 - 255!" )
            if len ( payload_bytes_arr ) > MAX_ALLOWED_PAYLOAD_LEN_BYTES_LEN :
                raise ValueError ( "Error: Payload exceeds maximum allowed length!" )
        elif payload_bits is not None and len ( payload_bits ) > 0 :
            payload_bits_arr = np.asarray ( payload_bits , dtype = np.uint8 ).ravel ()
            if payload_bits_arr.max () > 1 :
                raise ValueError ( "Error: Payload has not all values only: zeros or ones!" )
            if len ( payload_bits_arr ) > MAX_ALLOWED_PAYLOAD_LEN_BYTES_LEN * 8 :
                raise ValueError ( "Error: Payload exceeds maximum allowed length!" )
            payload_bytes_arr = pad_bits2bytes ( payload_bits_arr )
        else :
            raise ValueError ( "Either payload_bytes or payload_bits must be provided." )
        tx_frame = self.create_tx_frame ( payload_bytes = payload_bytes_arr )
        tx_frame.frame_start_abs_idx = np.uint32 ( self.samples4pluto.size + ( filters.SPAN * self.SPS // 2 ) )
        tx_frame.frame_start_abs_first_idx = np.uint32 ( self.samples4pluto.size + ( ( filters.SPAN * self.SPS // 2 ) - ( modulation.SPS//2 ) ) )
        self.frames.append ( tx_frame )
        self.add_bytes ( tx_frame.bytes ) # Nie powinno być payload_bytes_arr bo wtedy w bytes będzie tylko payload bez sync sequence i packet len.
        self.add_symbols ( tx_frame.bpsk_symbols )
        self.add_samples4pluto ( tx_frame.samples4pluto )
        self.create_samples4pluto_wo_muting_and_flat_tensor ()

    def add_bytes ( self , payload_bytes_arr : NDArray[ np.uint8 ] ) -> None :
        if payload_bytes_arr.size < 1 :
            raise ValueError ( "Error: There are no bytes to add!" )
        self.bytes = np.concatenate ( [ self.bytes , payload_bytes_arr ] )

    def add_symbols ( self , bpsk_symbols : NDArray[ np.complex128 ] ) -> None :
        if bpsk_symbols.size < 1 :
            raise ValueError ( "Error: There are no symbols to add!" )
        self.bpsk_symbols = np.concatenate ( [ self.bpsk_symbols , bpsk_symbols ] )

    def add_samples4pluto ( self , samples4pluto : NDArray[ np.complex128 ] ) -> None :
        if samples4pluto.size < 1 :
            raise ValueError ( "Error: There are no samples to add!" )
        self.samples4pluto = np.concatenate ( [ self.samples4pluto , samples4pluto ] )

    def create_samples4pluto ( self , payload_bytes : list | tuple | np.ndarray[ np.uint8 ] = None , payload_bits : list | tuple | np.ndarray[ np.uint8 ] = None ) -> None :
        self.frames.clear ()
        self.create_empty_complex_samples ()
        self.add_frame ( payload_bytes = payload_bytes , payload_bits = payload_bits )

    def create_samples4pluto_wo_muting_and_flat_tensor ( self ) -> None :
        self.samples4pluto_wo_mute = np.array ( [] , dtype = np.complex128 )
        self.samples_flat_tensor_wo_mute = np.array ( [] , dtype = np.complex64 ) # razem z rozbiegówką i wygaszeniem filtra, czyli z 0+j0 na początku i na końcu, ale bez przerw między ramkami, bo tworzymy ciągły strumień symboli bpsk z ramek i filtrujemy go jako całość.
        self.symbols_flat_tensor_wo_mute = torch.empty ( 0 , dtype = torch.complex64 ) # to co w self.samples_flat_tensor_wo_mute tylko bez 0+j0
        frames_bpsk_symbols : NDArray [ np.complex128 ] = np.concatenate ( [ frame.bpsk_symbols for frame in self.frames ] ).astype ( np.complex128 , copy = False )
        if frames_bpsk_symbols.size > 0 :

            samples_filtered = np.ravel ( filters.apply_tx_rrc_filter_v0_1_6 ( frames_bpsk_symbols ) ).astype ( np.complex128 , copy = False )
            self.samples4pluto_wo_mute = sdr.scale_to_pluto_dac_v0_1_11 ( samples = samples_filtered , scale = 1.0 )

            self.samples_flat_tensor_wo_mute = np.zeros ( self.samples4pluto_wo_mute.size , dtype = np.complex64 )
            first_frame_start_idx = np.uint32 ( filters.PEAK_TO_FIRST_SAMPLE_OFFSET )
            last_frame_end_idx = first_frame_start_idx + frames_bpsk_symbols.size * modulation.SPS
            active_samples = self.samples4pluto_wo_mute.real[ first_frame_start_idx : last_frame_end_idx ]
            #active_symbols = np.where ( active_samples < 0.0 , -1.0 + 0.0j , 1.0 + 0.0j ).astype ( np.complex64 , copy = False )
            '''Ponieważ język Python domyślnie traktuje zadeklarowane z klawiatury wartości -1.0+0.0j jako 128-bitowe, funkcja np.where pod spodem uformuje tymczasową tablicę
            complex128. Kiedy następnie nałożysz .astype(np.complex64), biblioteka NumPy, by uniknąć uszkodzeń pamięci, i tak stworzy nową, pomniejszoną w bajtach tablicę w tle,
            całkowicie ignorując dopisek copy = False (zgodnie zresztą ze swoją oficjalną dokumentacją dla zmiany rozmiaru bloku pamięci).Aby zoptymalizować to do absolutnego,
            purystycznego maksimum (czyli dosłownie zera zbędnych alokacji i konwersji pamięciowych na procesorze), możesz przekazać mniejszy typ rzutowany bezpośrednio
            do wnętrza funkcji np.where. Wtedy dodawanie metody .astype(...) na końcu nie będzie w ogóle potrzebne.'''
            active_symbols = np.where ( active_samples < 0.0 , np.complex64 ( -1.0 + 0j ) , np.complex64 ( 1.0 + 0j ) )
            self.samples_flat_tensor_wo_mute[ first_frame_start_idx : last_frame_end_idx ] = active_symbols
            # Poniżej Problemem było to, że active_symbols miało już długość symbols * SPS, a potem było jeszcze raz błędnie krojone zakresem liczonym względem samples4pluto_wo_mute,
            # co ucinało dane.
            #self.symbols_wo_mute = active_symbols[ first_frame_start_idx : last_frame_end_idx ].astype ( np.complex128 , copy = False )
            self.active_symbols_wo_mute = active_symbols.astype ( np.complex128 , copy = False )
            self.symbols_flat_tensor_wo_mute =  torch.from_numpy ( active_symbols.astype ( np.complex64 , copy = False ) )

    def tx ( self , sdr_ctx : Pluto , repeat : np.uint32 = 1 ) -> None :
        sdr_ctx.tx_destroy_buffer ()
        sdr_ctx.tx_cyclic_buffer = False
        if repeat < 1 or repeat > 4294967295 :
            raise ValueError ( "Error: reapt value is out of the range! Allowed range is 1 to 4294967295." )
        while repeat :
            sdr_ctx.tx ( self.samples4pluto_wo_mute ) # Uwaga w innych nie wprowadziłem tej zmiany, tj. wo_mute
            repeat -= 1

    def tx_cyclic ( self , sdr_ctx : Pluto ) -> None :
        sdr_ctx.tx_destroy_buffer ()
        sdr_ctx.tx_cyclic_buffer = True
        sdr_ctx.tx ( self.samples4pluto )

    def stop_tx_cyclic ( self , sdr_ctx : Pluto ) -> None :
        sdr_ctx.tx_destroy_buffer ()
        sdr_ctx.tx_cyclic_buffer = False

    def tx_incremeant_payload_and_repeat ( self , sdr_ctx : Pluto , n_o_bytes : np.uint16 = 1 , n_o_repeats : np.uint32 = 1 ) -> None :
        sdr_ctx.tx_destroy_buffer ()
        sdr_ctx.tx_cyclic_buffer = False
        bytes = np.zeros ( n_o_bytes , dtype = np.uint8 )
        while n_o_repeats :
            self.create_samples4pluto ( payload_bytes = bytes )
            sdr_ctx.tx ( self.samples4pluto)
            print ( f"\n\r  { n_o_repeats }: Transmitted payload bytes: { bytes }" )
            for i in range ( n_o_bytes - 1 , -1 , -1 ) :
                bytes [ i ] = np.uint8( ( int(bytes [ i ]) + 1 ) % 256 )
                if bytes [ i ] != 0 :
                    break
            n_o_repeats -= 1

    def tx_random_payload ( self , sdr_ctx : Pluto , repeats : np.uint32 = 100 ) -> None :
        sdr_ctx.tx_destroy_buffer ()
        sdr_ctx.tx_cyclic_buffer = False
        rng = np.random.default_rng ()
        print ( f"\n\r Start random transmition {repeats} times." )
        while repeats :
            sdr_ctx.tx_destroy_buffer ()
            sdr_ctx.tx_cyclic_buffer = False
            payload_len = rng.integers ( 1 , 1501 )
            payload_bytes = rng.integers ( 0 , 256 , size = payload_len , dtype = np.uint8 )
            self.create_samples4pluto ( payload_bytes = payload_bytes )
            sdr_ctx.tx ( self.samples4pluto)
            repeats -= 1
        print ( f"\n\r Stop random transmition" )

    def plot_symbols ( self , title = "" , constellation : bool = False ) -> None :
        plot.plot_symbols ( self.bpsk_symbols , f"{title} {self.bpsk_symbols.size=}" )
        if constellation :
            plot.complex_symbols_v0_1_6 ( self.bpsk_symbols , f"{title}" )

    def plot_complex_samples4pluto ( self , title = "" , marker_peaks : bool = False ) -> None :
        if marker_peaks :
            frames_start_idx : NDArray [ np.uint32 ] = np.array ( [ frame.frame_start_abs_idx for frame in self.frames ] , dtype = np.uint32 )
        else :
            frames_start_idx : NDArray [ np.uint32 ] = np.array ( [] , dtype = np.uint32 )
        plot.complex_waveform_v0_1_6 ( self.samples4pluto , f"{title} {self.samples4pluto.size=}" , marker_squares = False , marker_peaks = frames_start_idx )

    def plot_complex_samples4pluto_wo_mute ( self , title = "" , mark_frames_first_sample : bool = False ) -> None :
        if mark_frames_first_sample :
            idx : NDArray [ np.uint32 ] = np.array ( [ frame.frame_start_abs_first_idx for frame in self.frames ] , dtype = np.uint32 )
            plot.complex_waveform_v0_1_6 ( self.samples4pluto_wo_mute , f"{title} {self.samples4pluto_wo_mute.size=}" , marker_squares = False , marker_peaks = idx )
        else :
            plot.complex_waveform_v0_1_6 ( self.samples4pluto_wo_mute , f"{title} {self.samples4pluto_wo_mute.size=}" )

    def plot_samples_spectrum ( self , title = "" ) -> None :
        plot.spectrum_occupancy ( self.samples4pluto , 1024 , title )

    def plot_flat_tensor ( self , title : str = "tx flat tensor" , marker_idx : bool = False ) -> None :
        if marker_idx :
            frames_start_abs_first_idx : NDArray [ np.uint32 ] = np.array ( [ frame.frame_start_abs_first_idx for frame in self.frames ] , dtype = np.uint32 )
            plot.flat_tensor_v0_1_18 ( flat_tensor = self.flat_tensor , title = title , marker_idx = frames_start_abs_first_idx )
        else :
            plot.flat_tensor_v0_1_18 ( flat_tensor = self.flat_tensor , title = title )

    def plot_symbols_flat_tensor_wo_mute ( self , title : str = "tx symbols flat tensor wo. mute" , marker_idx : bool = False ) -> None :
        if marker_idx :
            frames_start_abs_first_idx : NDArray [ np.uint32 ] = np.array ( [ frame.frame_start_abs_first_idx for frame in self.frames ] , dtype = np.uint32 )
            plot.flat_tensor_v0_1_18 ( flat_tensor = self.symbols_flat_tensor_wo_mute , title = f"{title} tx {self.symbols_flat_tensor_wo_mute.shape=}" , marker_idx = frames_start_abs_first_idx )
        else :
            plot.flat_tensor_v0_1_18 ( flat_tensor = self.symbols_flat_tensor_wo_mute , title = f"{title} tx {self.symbols_flat_tensor_wo_mute.shape=}" )

    def plot_samples_flat_tensor_wo_mute ( self , title : str = "" , mark_frames_first_sample : bool = False ) -> None :
        if mark_frames_first_sample :
            idx : NDArray [ np.uint32 ] = np.array ( [ frame.frame_start_abs_first_idx for frame in self.frames ] , dtype = np.uint32 )
            plot.flat_tensor_v0_1_18 ( flat_tensor = self.samples_flat_tensor_wo_mute , title = f"{title} tx {self.samples_flat_tensor_wo_mute.shape=}" , marker_idx = idx )
        else :
            plot.flat_tensor_v0_1_18 ( flat_tensor = self.samples_flat_tensor_wo_mute , title = f"{title} tx {self.samples_flat_tensor_wo_mute.shape=}" )

    def plot_active_symbols_wo_mute ( self , title : str = "" , mark_frames_first_sample : bool = False ) -> None :
        if mark_frames_first_sample :
            idx : NDArray [ np.uint32 ] = np.array ( [ ( frame.frame_start_abs_first_idx - ( filters.SPAN * self.SPS // 2 ) + ( modulation.SPS//2 ) ) for frame in self.frames ] , dtype = np.uint32 )
            plot.plot_bpsk_symbols_v2 ( symbols = self.active_symbols_wo_mute , title = f"{title} tx {self.active_symbols_wo_mute.size=}" )
        else :
            plot.plot_bpsk_symbols_v2 ( symbols = self.active_symbols_wo_mute , title = f"{title} tx {self.active_symbols_wo_mute.size=}" )

    def samples4pluto_2_flat_tensor ( self ) -> None :
        '''Stworzenie flat_tensor w self.flat_tensor reprezentujacych cały przebieg samples4pluto. Wstawienie par -1+0j, 1+0j dla każdego symbolu reprezentujacego symbol w ramce
        oraz wstawienie 0+0j wszędzie tam gdzie w przebiegu samples4pluto nie ma symboli wynikajacych z ramek. Dodatkowo wstawiam 0+j0 wszędzie tam gdzie jest przejście z -1+j0 na 1+j0 lub z 1+j0 na -1+j0, czyli tam gdzie jest zmiana symbolu.
        '''
        self.flat_tensor = np.zeros ( self.samples4pluto.size , dtype = np.complex64 )
        for frame in self.frames :
            frame_start_idx = np.uint32 ( frame.frame_start_abs_first_idx )
            frame_end_idx = frame_start_idx + frame.bpsk_symbols.size * self.SPS
            if frame_start_idx >= self.flat_tensor.size or frame.bpsk_symbols.size == 0 :
                continue
            if frame_end_idx > self.flat_tensor.size :
                frame_end_idx = self.flat_tensor.size
            symbols_repeated = np.repeat ( frame.bpsk_symbols , self.SPS )[:frame_end_idx - frame_start_idx]
            self.flat_tensor [ frame_start_idx : frame_end_idx ] = symbols_repeated.astype ( np.complex64 , copy = False )

    def save_frames2flat_tensor ( self , filename : str , dir_name : str ) -> None :
        # It's not the actual y_train, but a reference truth for validating rx data (after receiving the frame).
        # Złożenie wszystkich bpsk_symbols z wszystkich ramek w jeden strumień w kolejności wysyłania
        # oraz proste powielenie każdego symbolu self.SPS razy.
        tensor_filename = f"{dir_name}/{filename}.pt"
        if self.frames :
            all_symbols = np.concatenate ( [ np.repeat ( frame.bpsk_symbols , self.SPS ) for frame in self.frames ] ).astype ( np.complex64 , copy = False )
        else :
            all_symbols = np.array ( [] , dtype = np.complex64 )
        frames_bpsk_symbols = torch.from_numpy ( all_symbols )
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        torch.save ( frames_bpsk_symbols , tensor_filename )

    def save_samples_2_flat_tensor ( self , filename : str , dir_name : str ) -> None :
        '''Stworzenie symboli bpsk reprezentujacych cały przebieg samples4pluto. Wstawienie par -1+0j, 1+0j dla każdego symbolu reprezentujacego symbol w ramce
        oraz wstawienie 0+0j wszędzie tam gdzie w przebiegu samples4pluto nie ma symboli wynikajacych z ramek.'''
        tensor_filename = f"{dir_name}/{filename}.pt"
        frame_symbols = np.zeros ( self.samples4pluto.size , dtype = np.complex64 )
        for frame in self.frames :
            frame_start_idx = np.uint32 ( frame.frame_start_abs_idx )
            frame_end_idx = frame_start_idx + frame.bpsk_symbols.size * self.SPS
            if frame_start_idx >= frame_symbols.size or frame.bpsk_symbols.size == 0 :
                continue
            if frame_end_idx > frame_symbols.size :
                frame_end_idx = frame_symbols.size
            symbols_repeated = np.repeat ( frame.bpsk_symbols , self.SPS )[:frame_end_idx - frame_start_idx]
            frame_symbols [ frame_start_idx : frame_end_idx ] = symbols_repeated.astype ( np.complex64 , copy = False )
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        torch.save ( torch.from_numpy ( frame_symbols ) , tensor_filename )

    def save_symbols_flat_tensor_wo_mute_2_pt ( self , file_name : str , dir_name : str ) -> None :
        tensor_filename = f"{dir_name}/{file_name}.pt"
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        torch.save ( self.symbols_flat_tensor_wo_mute , tensor_filename )

    def save_complex_samples4pluto_2_npf ( self , file_name : str , dir_name : str , add_timestamp : bool = True ) -> None :
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.samples4pluto )

    def save_active_symbols_wo_mute_2_npf ( self , file_name : str , dir_name : str , add_timestamp : bool = True ) -> None :
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.active_symbols_wo_mute )

    def __repr__ ( self ) -> str :
        return ( f"{ self.bpsk_symbols.size= }, { self.samples4pluto.size= }" )

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