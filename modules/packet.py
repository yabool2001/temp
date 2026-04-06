import csv
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

def add_timestamp_2_filename ( filename : str ) -> str :
    timestamp = int ( t.time () * 1000 )
    name, ext = os.path.splitext ( filename )
    return f"{name}_{timestamp}{ext}"

class PureComplexLSTMCell ( nn.Module ) :
    def __init__ ( self , input_size , hidden_size ) :
        super ().__init__ ()
        self.hidden_size = hidden_size
        
        # Warstwa Linear w przeciwieństwie do wbudowanego LSTM działa 
        # w 100% na liczbach zespolonych! To tu sieć robi obroty Eulera i chwyta CFO.
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size, dtype=torch.complex64)

    def forward (self , x , hx = None ) :
        if hx is None:
            # Pusta pamięć jako natywne zera zespolone (complex64)
            h = torch.zeros ( x.size ( 0 ) , self.hidden_size , dtype = torch.complex64 , device = x.device )
            c = torch.zeros ( x.size ( 0 ) , self.hidden_size , dtype = torch.complex64 , device = x.device )
        else:
            h, c = hx

        combined = torch.cat ( [ x , h ] , dim = 1 )
        
        # Prawdziwe Zespolone Mnożenie Macierzy (RTX łapie wiatr w żagle)
        gates = self.W(combined)
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)

        # SPLIT-ACTIVATION: Ratujemy sieć przed wybuchami NaN.
        # Faza zostaje utrzymana w ryzach matematycznych!
        i = torch.complex(torch.sigmoid(i_gate.real), torch.sigmoid(i_gate.imag))
        f = torch.complex(torch.sigmoid(f_gate.real), torch.sigmoid(f_gate.imag))
        o = torch.complex(torch.sigmoid(o_gate.real), torch.sigmoid(o_gate.imag))
        c_tilde = torch.complex(torch.tanh(c_gate.real), torch.tanh(c_gate.imag))

        # Zespolona aktualizacja pamięci komórki
        c_next = f * c + i * c_tilde
        h_next = o * torch.complex(torch.tanh(c_next.real), torch.tanh(c_next.imag))

        return h_next, c_next

class PureComplexLSTM ( nn.Module ) :
    def __init__ ( self , input_size , hidden_size ) :
        super ().__init__ ()
        self.cell = PureComplexLSTMCell ( input_size , hidden_size )

    def forward(self, x):
        # x wchodzi jako: [Batch, Czas, Cechy_Zespolone]
        outputs = []
        hx = None
        
        # Musimy fizycznie przepętlić sygnał po czasie, bo ominęliśmy wbudowany silnik C++
        for t in range(x.size(1)):
            h, c = self.cell(x[:, t, :], hx)
            hx = (h, c)
            outputs.append(h)
        
        return torch.stack(outputs, dim=1), hx

class HardcoreComplexEqualizer ( nn.Module ) :
    def __init__ ( self ) :
        super ().__init__ ()
        
        self.conv1 = nn.Conv1d ( in_channels = 1 , out_channels = 16 , kernel_size = 8 , stride = 4 , padding = 2 , dtype = torch.complex64)
        
        # WCHODZI NASZE PRAWDZIWE ZESPOLONE LSTM (żadnego dzielenia!)
        self.lstm = PureComplexLSTM ( input_size = 16 , hidden_size = 64 )
        
        # Klasyfikator wypluwa urojone koordynaty
        self.fc = nn.Linear ( 64 , 1 , dtype = torch.complex64 )

    def forward(self, x):
        # Wejście: [Batch, 1, 4096] na twardym complex64
        x = self.conv1 ( x ) # -> [Batch, 16, 1024] 
        x = x.transpose ( 1 , 2 ) # -> [Batch, 1024, 16]
        
        # Pełna faza wchodzi, pełna faza wychodzi
        x_lstm, _ = self.lstm(x) # -> [Batch, 1024, 64] (complex64)
        out = self.fc(x_lstm) # -> [Batch, 1024, 1] (complex64)
        
        # BPSK leży tylko na osi Real (-1, 1). Na samym końcu wyciągamy
        # komponent rzeczywisty by obliczyć błąd (MSE Loss). Autograd PyTorcha
        # prawidłowo przetransferuje tzw. gradienty Wirtingera przez całą sieć!
        return out.squeeze(-1).real

@dataclass ( slots = True , eq = False )
class Rx_y_Train_v0_0_0 :
    # na razie klasa do niczego nei potrzeba!
    symbols : NDArray[ np.int8 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.int8 ) , init = False )
    
    def __post_init__ ( self ) -> None :
        #self.symbols = np.array ( [] , dtype = np.complex128 )
        pass
    
    def __repr__ ( self ) -> str :
        return ( f"{self.symbols.size=}, {self.symbols.dtype=} {self.symbols=}")

@dataclass ( slots = True , eq = False )
class RxPacket_v0_1_18 :

    samples_filtered : NDArray[ np.complex128 ]
    packet_start_idx : np.uint32
    has_packet : bool = False
    samples_corrected : NDArray[ np.complex128 ] = field ( init = False )
    payload_bytes : NDArray[ np.uint8 ] = field ( init = False )
    packet_symbols : NDArray[ np.complex128 ] = field ( init = False )
    
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
                self.packet_symbols = modulation.bits_2_bpsk_symbols_v0_1_18 ( np.concatenate ( [ payload_bits, crc32_bits ] ) , sps = sps )
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
            self.packet_symbols = modulation.bits_2_bpsk_symbols_v0_1_18 ( np.concatenate ( [ payload_bits, crc32_bits ] ) , sps = sps )
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
class RxFrames_v0_1_13 :
    
    samples_filtered : NDArray[ np.complex128 ]
    signal_start_idx : np.uint32
    deep : bool = False

    # Pola uzupełnianie w __post_init__
    sps = modulation.SPS
    #sync_sequence_peaks : NDArray[ np.uint32 ] = field ( init = False )
    sync_sequence_peaks : NDArray[ np.uint32 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint32 ) , init = False )
    packets_idx : NDArray[ np.uint32 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint32 ) , init = False )
    samples_filtered_len : np.uint32 = field ( init = False )
    last_processed_idx : np.uint32 = 0
    samples_leftovers_start_idx : np.uint32 = field ( init = False )
    has_leftovers : bool = False
    samples_payloads_bytes : NDArray[ np.uint8 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint8 ) , init = False )
    
    def __post_init__ ( self ) -> None :
        has_frame = False
        sync_sequence_start_idx = self.signal_start_idx + filters.SPAN * self.sps // 2
        sync_sequence_end_idx = sync_sequence_start_idx + ( SYNC_SEQUENCE_LEN_BITS * self.sps )
        packet_len_start_idx = sync_sequence_end_idx
        packet_len_end_idx = packet_len_start_idx + ( PACKET_LEN_LEN_BITS * self.sps )
        crc32_start_idx = packet_len_end_idx
        crc32_end_idx : np.uint32 = ( crc32_start_idx + ( CRC32_LEN_BITS * self.sps ) ).astype ( np.uint32 )

        samples_components = [ ( self.samples_filtered.real , "sync sequence real" ) , ( self.samples_filtered.imag , "sync sequence imag" ) , ( -self.samples_filtered.real , "sync sequence -real" ) , ( -self.samples_filtered.imag , "sync sequence -imag" ) ]
        for samples_component , samples_name in samples_components :
            sync_sequence_bits = self.samples2bits ( samples_component [ sync_sequence_start_idx : sync_sequence_end_idx ] )
            if np.array_equal ( sync_sequence_bits , BARKER13_BITS ) :
                has_sync_sequence = True
                add2log_packet ( f"{t.time()},{has_sync_sequence=},{sync_sequence_start_idx=}")
                packet_len_uint16 = self.samples2uint16 ( samples_component [ packet_len_start_idx : packet_len_end_idx ] )
                check_components = [ ( self.samples_filtered.real , " frame real" ) , ( self.samples_filtered.imag , " frame imag" ) , ( -self.samples_filtered.real , " frame -real" ) , ( -self.samples_filtered.imag , " frame -imag" ) ]
                for samples_comp , frame_name in check_components :
                    crc32_bytes_read = self.samples2bytes ( samples_comp [ crc32_start_idx : crc32_end_idx ] )
                    crc32_bytes_calculated = create_crc32_bytes ( pad_bits2bytes ( self.samples2bits ( samples_comp [ sync_sequence_start_idx : packet_len_end_idx ] ) ) )
                    if ( crc32_bytes_read == crc32_bytes_calculated ).all () :
                        packet_end_idx = crc32_end_idx + ( packet_len_uint16 * PACKET_BYTE_LEN_BITS * self.sps )
                        has_frame = True
                        add2log_packet ( f"{t.time()},{has_frame=},{sync_sequence_start_idx=}")
                        if not self.packet_len_validation ( self.signal_start_idx , packet_end_idx ) :
                            add2log_packet ( f"{t.time()},{has_frame=},{sync_sequence_start_idx=}")
                            if settings["log"]["verbose_2"] : print ( f"{ sync_sequence_start_idx= } { samples_name } { frame_name= } { has_sync_sequence= }, { has_frame= }" )
                            return self.signal_start_idx
                        packet = RxPacket_v0_1_18 ( samples_filtered = self.samples_filtered [ self.signal_start_idx : packet_end_idx ] , packet_start_idx = crc32_end_idx - self.signal_start_idx )
                        if packet.has_packet :
                            self.packets_idx = np.append ( self.packets_idx , self.signal_start_idx )
                            self.samples_payloads_bytes = np.concatenate ( [ self.samples_payloads_bytes , packet.payload_bytes ] )
                            add2log_packet(f"{t.time()},{packet.has_packet=},{crc32_end_idx}")
                            if settings["log"]["verbose_2"] : print ( f"{self.signal_start_idx=} {has_sync_sequence=}, {has_frame=}, {packet.has_packet= }" )
                            return packet_end_idx
        if settings["log"]["verbose_2"] : print ( f"{ self.signal_start_idx= } { has_sync_sequence= }, { has_frame= }" )
        return self.signal_start_idx
    
    def samples2bits ( self , samples : NDArray[ np.complex128 ] ) -> NDArray[ np.uint8 ] :
        sync_sequence_symbols = samples [ : : self.sps ]
        return modulation.bpsk_symbols_2_bits_v0_1_7 ( sync_sequence_symbols )

    def samples2uint16 ( self , samples : NDArray[ np.complex128 ] ) -> np.uint16 :
        bits = self.samples2bits ( samples )
        return np.uint16 ( bits_2_int ( bits ) )

    def samples2bytes ( self , samples : NDArray[ np.complex128 ] ) -> NDArray[ np.uint8 ] :
        bits = self.samples2bits ( samples )
        return pad_bits2bytes ( bits )
    
    def complete_process_frame ( self , idx : np.uint32 ) -> None :
        if settings["log"]["verbose_2"] : print ( f"Samples at index { idx } is too close to the end of samples to contain a full frame. Skipping." )
        self.samples_leftovers_start_idx = idx - filters.SPAN * self.sps // 2 # Bez cofniecia się do początku filtra RRC nie ma wykrycia ramnki i pakietu w następnym wywołaniu
        self.has_leftovers = True

    def frame_len_validation ( self, idx : np.uint32 ) -> bool :
        remainings_len = self.samples_filtered_len - idx
        if remainings_len <= FRAME_LEN_SAMPLES :
            self.complete_process_frame ( idx )
            return False
        return True

    def packet_len_validation ( self , idx : np.uint32 , packet_end_idx : np.uint32 ) -> bool :
        if packet_end_idx > self.samples_filtered_len :
            self.complete_process_frame ( idx )
            return False
        return True

    def plot_complex_samples_filtered ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = peaks )

    def __repr__ ( self ) -> str :
        return ( f"{ self.frames.size= } , dtype = { self.frames.dtype= }")

@dataclass ( slots = True , eq = False )
class RxFrame_v0_1_18 :
    
    samples_filtered : NDArray[ np.complex128 ]
    sync_sequence_peak_idx : np.uint32

    # Pola uzupełnianie w __post_init__
    SPS = modulation.SPS
    SPAN = filters.SPAN
    frame_start_idx : np.uint32 = field ( init = False )
    frame_end_idx : np.uint32 = field ( init = False )
    packet_start_idx : NDArray[ np.uint32 ] = field ( init = False )
    leftovers_start_idx : np.uint32 = field ( init = False )
    has_frame : bool = False # ustawiany dopiero po walidacji pakietu, wcześniej używamy tylko lokalnego has_frame_header
    has_leftovers : bool = False
    payload_bytes : NDArray[ np.uint8 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint8 ) , init = False )
    
    def __post_init__ ( self ) -> None :
        if not self.frame_len_validation () :
            return self.sync_sequence_peak_idx
        self.process_packet ()
    
    def process_packet ( self ) -> None :
        sync_sequence_start_idx = self.SPAN * self.SPS // 2
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
                packet_len_uint16 = self.bits2uint16 ( packet_len_bits )
                check_components = [ ( self.samples_filtered.real , " frame real" ) , ( self.samples_filtered.imag , " frame imag" ) , ( -self.samples_filtered.real , " frame -real" ) , ( -self.samples_filtered.imag , " frame -imag" ) ]
                for samples_comp , frame_name in check_components :
                    crc32_symbols = samples_comp [ crc32_start_idx : crc32_end_idx : self.SPS ]
                    crc32_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( crc32_symbols )
                    crc32_bytes_read = pad_bits2bytes ( crc32_bits )
                    crc32_bytes_calculated = create_crc32_bytes ( np.concatenate ( [ sync_sequence_bits, packet_len_bits ] ) )
                    if ( crc32_bytes_read == crc32_bytes_calculated ).all () :
                        packet_end_idx = crc32_end_idx + ( packet_len_uint16 * PACKET_BYTE_LEN_BITS * self.SPS )
                        has_frame_header = True
                        self.frame_start_idx = sync_sequence_start_idx
                        add2log_packet ( f"{t.time()},{sync_sequence_start_idx=},{has_frame_header=},{self.frame_start_idx=}")
                        if not self.packet_len_validation ( self.sync_sequence_peak_idx , packet_end_idx ) :
                            add2log_packet ( f"{t.time()},{has_frame_header=},{sync_sequence_start_idx=}")
                            if settings["log"]["verbose_2"] : print ( f"{self.sync_sequence_peak_idx=} {samples_name} {frame_name=} {has_sync_sequence=}, {has_frame_header=}" )
                            return
                        packet = RxPacket_v0_1_18 ( samples_filtered = self.samples_filtered [ crc32_end_idx : packet_end_idx ] , packet_start_idx = sync_sequence_start_idx + crc32_end_idx )
                        if packet.has_packet :
                            self.has_frame = True
                            self.frame_start_idx = sync_sequence_start_idx
                            self.frame_end_idx = sync_sequence_start_idx + packet_end_idx # to może być tylko wtedy kiedy mamy poprawny pakiet, bo inaczej nie wiemy, czy i gdzie się kończy ramka, a bez tego nie możemy poprawnie ustawić leftoversów
                            self.payload_bytes = np.concatenate ( [ self.payload_bytes , packet.payload_bytes ] )
                            add2log_packet(f"{t.time()},{packet.has_packet=},{crc32_end_idx=}")
                            if settings["log"]["verbose_2"] : print ( f"{sync_sequence_start_idx=} {has_sync_sequence=}, {self.frame_start_idx=} {self.has_frame=}, {packet.has_packet=}" )
                            return
        if settings["log"]["verbose_2"] : print ( f"{self.frame_sync_sequence_peak_idx=} {has_sync_sequence=}, {self.has_frame=}" )
        return
    
    def samples2bits ( self , samples : NDArray[ np.complex128 ] ) -> NDArray[ np.uint8 ] :
        return modulation.bpsk_symbols_2_bits_v0_1_7 ( samples [ : : self.sps ] )

    def bits2uint16 ( self , bits : NDArray[ np.uint8 ] ) -> np.uint16 :
        return np.uint16 ( bits_2_int ( bits ) )

    def samples2bytes ( self , samples : NDArray[ np.complex128 ] ) -> NDArray[ np.uint8 ] :
        bits = self.samples2bits ( samples )
        return pad_bits2bytes ( bits )
    
    def complete_process_frame ( self , idx : np.uint32 ) -> None :
        if settings["log"]["verbose_2"] : print ( f"Samples at index { idx } is too close to the end of samples to contain a full frame. Skipping." )
        self.leftovers_start_idx = self.sync_sequence_peak_idx + idx - self.SPAN * self.SPS // 2 # Bez cofniecia się do początku filtra RRC nie ma wykrycia ramnki i pakietu w następnym wywołaniu
        self.has_leftovers = True

    def frame_len_validation ( self ) -> bool :
        if np.uint32 ( self.samples_filtered.size ) <= np.uint32 ( FRAME_LEN_SAMPLES ) :
            self.complete_process_frame ( self.sync_sequence_peak_idx )
            return False
        return True

    def packet_len_validation ( self , idx : np.uint32 , packet_end_idx : np.uint32 ) -> bool :
        if packet_end_idx > np.uint32 ( self.samples_filtered.size ) :
            self.complete_process_frame ( idx )
            return False
        return True

    def plot_complex_samples_filtered ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = peaks )

    def __repr__ ( self ) -> str :
        return ( f"{ self.frames.size= } , dtype = { self.frames.dtype= }")

@dataclass ( slots = True , eq = False )
class RxSamples_v0_1_18 :

    # Pola uzupełnianie w __post_init__
    #samples : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    samples : NDArray[ np.complex128 ] = field ( init = False )
    tensor : torch.Tensor = field ( init = False )
    samples_filtered : NDArray[ np.complex128 ] = field ( init = False )
    has_amp_greater_than_ths : bool = False
    SPS = modulation.SPS
    SPAN = filters.SPAN
    ths : float = 1000.0
    frames_old : RxFrames_v0_1_13 = field ( init = False )
    frames : list[ RxFrame_v0_1_18 ] = field ( init = False , default_factory = list )
    samples_leftovers : NDArray[ np.complex128 ] | None = field ( default = None )

    samples_filtered_len : np.uint32 = field ( init = False )
    sync_sequence_peaks : NDArray[ np.uint32 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint32 ) , init = False )
    has_leftovers : bool = False
    leftovers_start_idx : np.uint32 = field ( init = False )

    def __post_init__ ( self ) -> None :
            self.samples = np.array ( [] , dtype = np.complex128 )
            self.samples_filtered = np.array ( [] , dtype = np.complex128 )
            #self.tensor = torch.tensor ( [] , dtype = torch.float32 )

    def rx ( self , sdr_ctx : Pluto  | None = None , previous_samples_leftovers : NDArray[ np.complex128 ] | None = None , samples_filename : str | None = None , concatenate : bool = False ) -> None :
        '''
        concatenated: powoduje nawarstwienie nowych sampli na stare. UWAGA! Nieostrożne stosowanie może spowodować zawieszenie komputera z powodu braku pamięci RAM, jeśli próbujemy nawarstwić zbyt dużo sampli. Używaj z rozwagą i monitoruj zużycie pamięci.
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

    def filter_samples ( self ) -> None :
        self.samples_filtered = filters.apply_rrc_rx_filter_v0_1_6 ( self.samples )

    def create_tensor ( self ) -> None :
        self.tensor = ml.iq_to_tensor_v2 ( self.samples )

    def detect_frames ( self , deep : bool = False ) -> None :
        self.filter_samples ()
        self.has_leftovers = False
        self.samples_filtered_len = np.uint32 ( len ( self.samples_filtered ) )
        self.sync_sequence_peaks = detect_sync_sequence_peaks_v0_1_15 ( self.samples_filtered , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) , deep = deep )
        previous_processed_idx : np.uint32 = 0
        for idx in self.sync_sequence_peaks :
            if idx > previous_processed_idx :
                frame = RxFrame_v0_1_18 ( samples_filtered = self.samples_filtered [ idx : ] , sync_sequence_peak_idx = idx )
                if frame.has_frame :
                    self.frames.append ( frame )
                    previous_processed_idx = frame.frame_end_idx
                else :
                    previous_processed_idx = idx
                if frame.has_leftovers :
                    self.has_leftovers = True
                    self.leftovers_start_idx = frame.leftovers_start_idx
                    break
        if not self.has_leftovers :
            self.leftovers_start_idx = self.samples_filtered_len - SYNC_SEQUENCE_LEN_SAMPLES - self.SPAN * self.SPS // 2
        self.clip_samples_leftovers ()

    def sample_initial_assesment (self) -> None :
        self.has_amp_greater_than_ths = np.any ( np.abs ( self.samples ) > self.ths )

    def plot_complex_samples ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples , f"RxSamples {title} {self.samples.size=}" , marker_squares = marker , marker_peaks = peaks )

    def plot_complex_samples_filtered ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"RxSamples filtered {title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = peaks )

    def plot_tensor ( self , title : str = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None , frame_idx : int | None = None ) -> None :
        plot.tensor_waveform_v0_1_16 ( self.tensor , title = f"RxSamples tensor {title}" , marker_squares = marker , marker_peaks = peaks , frame_idx = frame_idx )

    def plot_complex_samples_corrected ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_corrected , f"RxSamples corrected {title} {self.samples_corrected.size=}" , marker_squares = marker , marker_peaks = peaks )

    def save_complex_samples_2_npf ( self , filename : str ) -> None :
        filename_with_timestamp = add_timestamp_2_filename ( filename )
        ops_file.save_complex_samples_2_npf ( filename_with_timestamp , self.samples )

    def save_complex_samples_2_csv ( self , filename : str ) -> None :
        filename_with_timestamp = add_timestamp_2_filename ( filename )
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

    def clip_samples_leftovers ( self ) -> None :
        self.samples_leftovers = self.samples [ self.leftovers_start_idx : ]

    def __repr__ ( self ) -> str :
        return ( f"{ self.samples.size= }, { self.samples.dtype= }")

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
class TxPacket_v0_1_11 :
    
    payload_bytes : NDArray[ np.uint8 ]
    
    # Pola uzupełnianie w __post_init__
    crc32_bytes : NDArray[ np.uint8 ] = field ( init = False )
    packet_bytes : NDArray[ np.uint8 ] = field ( init = False )
    packet_len : np.uint16 = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.create_crc32_bytes ()
        self.create_packet_bytes ()
        self.packet_len = np.uint16 ( len ( self.payload_bytes ) + len ( self.crc32_bytes ) )  # payload + crc32

    def create_crc32_bytes ( self ) -> None :
        self.crc32_bytes = create_crc32_bytes ( self.payload_bytes )

    def create_packet_bytes ( self ) -> None:
        self.packet_bytes = np.concatenate ( [ self.payload_bytes , self.crc32_bytes ] )

    def __repr__ ( self ) -> str :
        return ( f"{ self.payload_bytes= }, { self.crc32_bytes= }, { self.packet_len= }" )

@dataclass ( slots = True , eq = False )
class TxFrame_v0_1_12 :

    tx_packet : TxPacket_v0_1_11
        
    # Pola uzupełnianie w __post_init__
    bytes : NDArray[ np.uint8 ] = field ( init = False )
    bits : NDArray[ np.uint8 ] = field ( init = False )
    bpsk_symbols : NDArray[ np.complex128 ] = field ( init = False )
    samples4pluto : NDArray[ np.complex128 ] = field ( init = False )

    def __post_init__ ( self ) -> None :
        sync_sequence_bits : NDArray[ np.uint8 ] = self.create_sync_sequence_bits ()
        packet_len_bits : NDArray[ np.uint8 ] = self.create_packet_len_bits ()
        frame_main_bytes = pad_bits2bytes ( np.concatenate ( [ sync_sequence_bits , packet_len_bits ] ) )
        crc32_bytes = self.create_crc32_bytes ( frame_main_bytes )
        self.bytes = np.concatenate ( [ frame_main_bytes , crc32_bytes , self.tx_packet.packet_bytes ] )
        self.bits = self.create_frame_bits ()
        self.bpsk_symbols = self.create_frame_bpsk_symbols ()

    def create_sync_sequence_bits ( self ) -> NDArray[ np.uint8 ] :
        return BARKER13_BITS

    def create_packet_len_bits ( self ) -> NDArray[ np.uint8 ] :
        return dec2bits ( self.tx_packet.packet_len , PACKET_LEN_LEN_BITS )

    def create_crc32_bytes ( self , frame_main_bytes ) -> NDArray[ np.uint8 ] :
        return create_crc32_bytes ( frame_main_bytes )
    
    def create_frame_bits ( self ) -> NDArray[ np.uint8 ] :
        return bytes2bits ( self.bytes )

    def create_frame_bpsk_symbols ( self ) -> NDArray[ np.complex128 ] :
        return modulation.create_bpsk_symbols_v0_1_6_fastest_short ( self.bits )

    def create_samples4pluto ( self ) -> None :
        samples_filtered = np.ravel ( filters.apply_tx_rrc_filter_v0_1_6 ( self.bpsk_symbols ) ).astype ( np.complex128 , copy = False )
        self.samples4pluto = sdr.scale_to_pluto_dac_v0_1_11 ( samples = samples_filtered , scale = 1.0 )

    def __repr__ ( self ) -> str :
        return (
            f"{self.bytes=}, {self.bytes.size=}, {self.bpsk_symbols.size=}" )

@dataclass ( slots = True , eq = False )
class TxSamples_v0_1_17 :
    '''Są 2 tryby tworzenia tx_samples. 
    1. Concatenując sample utworzone w tx_frames. Wtedy mamy ładne przerwy między ramkami,
        bo każda ramka jest filtrowana osobno i pomiędzy ramkami są przerwy bez sygnału z symbolami.
    2. Tworząc ciągłe samples w tx_samples na podstawie symboli bpsk z tx_frames.
        Wtedy nie ma przerw między ramkami, bo tworzymy ciągły strumień symboli bpsk z ramek i filtrujemy go jako całość.'''
    payload_bytes : list | tuple | np.ndarray[ np.uint8 ] | None = None
    payload_bits : list | tuple | np.ndarray[ np.uint8 ] | None = None

    # Pola uzupełnianie w __post_init__
    bytes : np.ndarray[ np.uint8 ] = field ( init = False )
    bpsk_symbols : NDArray[ np.complex128 ] = field ( init = False )
    samples : NDArray[ np.complex128 ] = field ( init = False )
    samples4pluto : NDArray[ np.complex128 ] = field ( init = False )
    frames : list[ TxFrame_v0_1_12 ] = field ( init = False , default_factory = list )

    def __post_init__ ( self ) -> None :
        self.create_empty_complex_samples ()
        if self.payload_bytes is not None and len ( self.payload_bytes ) > 0 :
            self.add_frame ( payload_bytes = self.payload_bytes )
        elif self.payload_bits is not None and len ( self.payload_bits ) > 0 :
            self.add_frame ( payload_bits = self.payload_bits )

    def create_empty_complex_samples ( self ) -> None :
        self.bytes = np.array ( [] , dtype = np.uint8 )
        self.bpsk_symbols = np.array ( [] , dtype = np.complex128 )
        self.samples = np.array ( [] , dtype = np.complex128 )
        self.samples4pluto = np.array ( [] , dtype = np.complex128 )

    def create_tx_frame ( self , payload_bytes : np.ndarray[ np.uint8 ] ) -> TxFrame_v0_1_12 :
        tx_packet = TxPacket_v0_1_11 ( payload_bytes = payload_bytes )
        tx_frame = TxFrame_v0_1_12 ( tx_packet = tx_packet )
        tx_frame.create_samples4pluto ()
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
        self.frames.append ( self.create_tx_frame ( payload_bytes = payload_bytes_arr ) )
        self.add_bytes ( self.frames[-1].bytes ) # Nie powinno być payload_bytes_arr bo wtedy w bytes będzie tylko payload bez sync sequence i packet len.
        self.add_symbols ( self.frames[-1].bpsk_symbols )
        self.add_samples4pluto ( self.frames[-1].samples4pluto )

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

    def tx ( self , sdr_ctx : Pluto , repeat : np.uint32 = 1 ) -> None :
        sdr_ctx.tx_destroy_buffer ()
        sdr_ctx.tx_cyclic_buffer = False
        if repeat < 1 or repeat > 4294967295 :
            raise ValueError ( "Error: reapt value is out of the range! Allowed range is 1 to 4294967295." )
        while repeat :
            sdr_ctx.tx ( self.samples4pluto )
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

    def plot_complex_samples4pluto ( self , title = "" ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples4pluto , f"{title} {self.samples4pluto.size=}" , marker_squares = False )

    def plot_samples_spectrum ( self , title = "" ) -> None :
        plot.spectrum_occupancy ( self.samples4pluto , 1024 , title )

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