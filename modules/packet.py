import csv
import logging
import numpy as np
import os
import time as t
import tomllib
import zlib

from adi import Pluto
from dataclasses import dataclass , field
from modules import corrections , filters , modulation, ops_file, plot , sdr
from numpy.typing import NDArray

from pathlib import Path
from scipy.signal import find_peaks
from typing import Any

np.set_printoptions ( threshold = np.inf , linewidth = np.inf )
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

def detect_sync_sequence_peaks_v0_1_11  ( samples: NDArray[ np.complex128 ] , sync_sequence : NDArray[ np.complex128 ] ) -> NDArray[ np.uint32 ] :

    plt = False
    wrt = False
    sync = False
    base_path = Path ( "logs/correlation_results.csv" )
    corr_2_amp_min_ratio = 12.0
    #min_peak_height_ratio = 0.6  # Required minimal height of peaks in find_peaks
    #min_peak_height_ratio = 0.5  # Działa ale za dużo plot wyświetla ale niby print liczy dobrze liczbę bajtów payload
    min_peak_height_ratio = 0.4  # Ten cudowanie opkazuje liczbę sampli na plot i chyba też dobrą w print liczbę bajtów!!!
    
    peaks = np.array ( [] ).astype ( np.uint32 )
    peaks_real = np.array ( [] ).astype ( np.uint32 )
    peaks_neg_real = np.array ( [] ).astype ( np.uint32 )
    peaks_imag = np.array ( [] ).astype ( np.uint32 )
    peaks_neg_imag = np.array ( [] ).astype ( np.uint32 )
    peaks_abs = np.array ( [] ).astype ( np.uint32 )
    max_amplitude_real = np.max ( np.abs ( samples.real ) )
    max_amplitude_imag = np.max ( np.abs ( samples.imag ) )
    max_amplitude_abs = np.max ( np.abs ( samples ) )
    
    #print ( f"{max_amplitude_real=} at {max_amplitude_real_idx=}, {max_amplitude_imag=} at {max_amplitude_imag_idx=}, {max_amplitude_abs=} at {max_amplitude_abs_idx=}" )
    #avg_amplitude = np.mean(np.abs(scenario['sample']))
    #percentile_95 = np.percentile(np.abs(scenario['sample']), 95)
    #rms_amplitude = np.sqrt(np.mean(np.abs(scenario['sample'])**2))

    corr_real = np.correlate ( samples.real , sync_sequence.real , mode = "valid" )
    corr_neg_real = np.correlate ( -samples.real , sync_sequence.real , mode = "valid" )
    corr_imag = np.correlate ( samples.imag , sync_sequence.real , mode = "valid" )
    corr_neg_imag = np.correlate ( -samples.imag , sync_sequence.real , mode = "valid" )
    corr_abs = np.abs ( np.correlate ( samples , sync_sequence , mode = "valid" ) )

    max_peak_real_val = np.max ( corr_real )
    max_peak_neg_real_val = np.max ( corr_neg_real )
    max_peak_imag_val = np.max ( corr_imag )
    max_peak_neg_imag_val = np.max ( corr_neg_imag )
    max_peak_abs_val = np.max ( corr_abs )

    corr_2_amp_real = max_peak_real_val / max_amplitude_real
    corr_2_amp_neg_real = max_peak_neg_real_val / max_amplitude_real
    corr_2_amp_imag = max_peak_imag_val / max_amplitude_imag
    corr_2_amp_neg_imag = max_peak_neg_imag_val / max_amplitude_imag
    corr_2_amp_abs = max_peak_abs_val / max_amplitude_abs

    if corr_2_amp_real > corr_2_amp_min_ratio :
        sync = True
        peaks_real , _ = find_peaks ( corr_real , height = max_peak_real_val * min_peak_height_ratio , distance = len ( sync_sequence ) * modulation.SPS , prominence = 0.1 )
        peaks = np.concatenate ( ( peaks , peaks_real ) ).astype ( np.uint32 )
    if corr_2_amp_neg_real > corr_2_amp_min_ratio :
        sync = True
        peaks_neg_real , _ = find_peaks ( corr_neg_real , height = max_peak_neg_real_val * min_peak_height_ratio , distance = len ( sync_sequence ) * modulation.SPS , prominence = 0.1 )
        peaks = np.concatenate ( ( peaks , peaks_neg_real ) ).astype ( np.uint32 )
    if corr_2_amp_imag > corr_2_amp_min_ratio :
        sync = True
        peaks_imag , _ = find_peaks ( corr_imag , height = max_peak_imag_val * min_peak_height_ratio , distance = len ( sync_sequence ) * modulation.SPS , prominence = 0.1 )
        peaks = np.unique ( np.concatenate ( ( peaks , peaks_imag ) ) ).astype ( np.uint32 )
    if corr_2_amp_neg_imag > corr_2_amp_min_ratio :
        sync = True
        peaks_neg_imag , _ = find_peaks ( corr_neg_imag , height = max_peak_neg_imag_val * min_peak_height_ratio , distance = len ( sync_sequence ) * modulation.SPS , prominence = 0.1 )
        peaks = np.unique ( np.concatenate ( ( peaks , peaks_neg_imag ) ) ).astype ( np.uint32 )
    if corr_2_amp_abs > corr_2_amp_min_ratio :
        sync = True
        peaks_abs , _ = find_peaks ( corr_abs , height = max_peak_abs_val * min_peak_height_ratio , distance = len ( sync_sequence ) * modulation.SPS , prominence = 0.1 )
        peaks = np.unique ( np.concatenate ( ( peaks , peaks_abs ) ) ).astype ( np.uint32 )

    if plt and sync :
        if peaks_real.size > 0 :
            plot.real_waveform_v0_1_6 ( corr_real , f"v0_1_7_u2 corr real {samples.size=}" , False , peaks_real )
            plot.real_waveform_v0_1_6 ( samples.real , f"v0_1_7_u2 samples real {samples.size=}" , False , peaks_real )
        if peaks_neg_real.size > 0 :
            plot.real_waveform_v0_1_6 ( corr_neg_real , f"v0_1_7_u2 corr neg real {samples.size=}" , False , peaks_neg_real )
            plot.real_waveform_v0_1_6 ( -samples.real , f"v0_1_7_u2 samples neg real {samples.size=}" , False , peaks_neg_real )
        if peaks_imag.size > 0 :
            plot.real_waveform_v0_1_6 ( corr_imag , f"v0_1_7_u2 corr imag {samples.size=}" , False , peaks_imag )
            plot.real_waveform_v0_1_6 ( samples.imag , f"v0_1_7_u2 samples imag {samples.size=}" , False , peaks_imag )
        if peaks_neg_imag.size > 0 :
            plot.real_waveform_v0_1_6 ( corr_neg_imag , f"v0_1_7_u2 corr neg imag {samples.size=}" , False , peaks_neg_imag )
            plot.real_waveform_v0_1_6 ( -samples.imag , f"v0_1_7_u2 samples neg imag {samples.size=}" , False , peaks_neg_imag )
        if peaks_abs.size > 0 :
            plot.real_waveform_v0_1_6 ( corr_abs , f"v0_1_7_u2 corr abs {samples.size=}" , False , peaks_abs )
            plot.complex_waveform_v0_1_6 ( samples , f"v0_1_7_u2 samples abs {samples.size=}" , False , peaks_abs )

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

    return peaks

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

@dataclass ( slots = True , eq = False )
class RxPacket_v0_1_12 :

    samples_filtered : NDArray[ np.complex128 ]
    has_packet : bool = False
    samples_corrected : NDArray[ np.complex128 ] = field ( init = False )
    payload_bytes : NDArray[ np.uint8 ] = field ( init = False )
    
    # Pola uzupełnianie w __post_init__

    def __post_init__ ( self ) -> None :
        self.samples_corrected = self.correct_cfo ()
        self.process_packet ()
    
    def process_packet ( self ) -> None :
        sps = modulation.SPS
        samples_components = [ ( self.samples_filtered.real , "packet real" ) , ( self.samples_filtered.imag , "packet imag" ) , ( -self.samples_filtered.real , "packet -real" ) , ( -self.samples_filtered.imag , "packet -imag" ) ]
        for samples_component , samples_name in samples_components :
            payload_end_idx = len ( self.samples_filtered ) - ( CRC32_LEN_BITS * sps )
            payload_symbols = samples_component [ : payload_end_idx : sps ]
            crc32_symbols = samples_component [ payload_end_idx : : sps ]
            payload_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( payload_symbols )
            payload_bytes = pad_bits2bytes ( payload_bits )
            crc32_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( crc32_symbols )
            crc32_bytes_read = pad_bits2bytes ( crc32_bits )
            crc32_bytes_calculated = create_crc32_bytes ( payload_bytes )
            if ( crc32_bytes_read == crc32_bytes_calculated ).all () :
                self.has_packet = True
                self.payload_bytes = payload_bytes
                if settings["log"]["debugging"] : print ( samples_name )
                return

    def correct_cfo ( self ) -> None :
        self.samples_corrected = modulation.zero_quadrature ( corrections.full_compensation_v0_1_5 ( self.samples_filtered , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) ) )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.samples_filtered.size= }, { self.has_packet= }, { self.payload_bytes.size if self.has_packet else None= }"
        )

@dataclass ( slots = True , eq = False )
class RxFrames_v0_1_12 :
    
    samples_filtered : NDArray[ np.complex128 ]

    # Pola uzupełnianie w __post_init__
    sps = modulation.SPS
    sync_sequence_peaks : NDArray[ np.uint32 ] = field ( init = False )
    samples_filtered_len : np.uint32 = field ( init = False )
    last_processed_idx : np.uint32 = 0
    samples_leftovers_start_idx : np.uint32 = field ( init = False )
    has_leftovers : bool = False
    samples_payloads_bytes : NDArray[ np.uint8 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint8 ) , init = False )
    
    def __post_init__ ( self ) -> None :
        self.samples_filtered_len = np.uint32 ( len ( self.samples_filtered ) )
        self.sync_sequence_peaks = detect_sync_sequence_peaks_v0_1_11 ( self.samples_filtered , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) )
        if self.sync_sequence_peaks.size > 0 and settings["log"]["debugging"] : self.plot_complex_samples_filtered ( title = f"RxFrames_v0_1_9 __post_init__" , marker = False , peaks = self.sync_sequence_peaks )
        for idx in self.sync_sequence_peaks :
            if idx > self.last_processed_idx :
                self.last_processed_idx = self.process_frame ( idx = idx )
                if self.has_leftovers :
                    break
        if not self.has_leftovers :
            self.samples_leftovers_start_idx = self.samples_filtered_len - SYNC_SEQUENCE_LEN_SAMPLES - filters.SPAN * self.sps // 2
            self.has_leftovers = True
    
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
        if settings["log"]["debugging"] : print ( f"Samples at index { idx } is too close to the end of samples to contain a full frame. Skipping." )
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

    def process_frame ( self , idx : np.uint32 ) -> np.uint32 :
        if not self.frame_len_validation ( idx ) :
            return idx
        has_frame = has_sync_sequence = False
        sync_sequence_start_idx = idx + filters.SPAN * self.sps // 2
        sync_sequence_end_idx = sync_sequence_start_idx + ( SYNC_SEQUENCE_LEN_BITS * self.sps )
        packet_len_start_idx = sync_sequence_end_idx
        packet_len_end_idx = packet_len_start_idx + ( PACKET_LEN_LEN_BITS * self.sps )
        crc32_start_idx = packet_len_end_idx
        crc32_end_idx = crc32_start_idx + ( CRC32_LEN_BITS * self.sps )

        samples_components = [ ( self.samples_filtered.real , "sync sequence real" ) , ( self.samples_filtered.imag , "sync sequence imag" ) , ( -self.samples_filtered.real , "sync sequence -real" ) , ( -self.samples_filtered.imag , "sync sequence -imag" ) ]
        for samples_component , samples_name in samples_components :
            sync_sequence_bits = self.samples2bits ( samples_component [ sync_sequence_start_idx : sync_sequence_end_idx ] )
            if np.array_equal ( sync_sequence_bits , BARKER13_BITS ) :
                has_sync_sequence = True
                packet_len_uint16 = self.samples2uint16 ( samples_component [ packet_len_start_idx : packet_len_end_idx ] )
                check_components = [ ( self.samples_filtered.real , " frame real" ) , ( self.samples_filtered.imag , " frame imag" ) , ( -self.samples_filtered.real , " frame -real" ) , ( -self.samples_filtered.imag , " frame -imag" ) ]
                for samples_comp , frame_name in check_components :
                    crc32_bytes_read = self.samples2bytes ( samples_comp [ crc32_start_idx : crc32_end_idx ] )
                    crc32_bytes_calculated = create_crc32_bytes ( pad_bits2bytes ( self.samples2bits ( samples_comp [ sync_sequence_start_idx : packet_len_end_idx ] ) ) )
                    if ( crc32_bytes_read == crc32_bytes_calculated ).all () :
                        packet_end_idx = crc32_end_idx + ( packet_len_uint16 * PACKET_BYTE_LEN_BITS * self.sps )
                        has_frame = True
                        if not self.packet_len_validation ( idx , packet_end_idx ) :
                            add2log_packet ( f"{t.time()},{idx},{has_sync_sequence},{has_frame}")
                            if settings["log"]["debugging"] : print ( f"{ idx= } { samples_name } { frame_name= } { has_sync_sequence= }, { has_frame= }" )
                            return idx
                        packet = RxPacket_v0_1_12 ( samples_filtered = self.samples_filtered [ crc32_end_idx : packet_end_idx ] )
                        if packet.has_packet :
                            self.samples_payloads_bytes = np.concatenate ( [ self.samples_payloads_bytes , packet.payload_bytes ] )
                            add2log_packet(f"{t.time()},{idx},{has_sync_sequence},{has_frame},{packet.has_packet}")
                            if settings["log"]["debugging"] : print ( f"{ idx= } { has_sync_sequence= }, { has_frame= }, { packet.has_packet= }" )
                            return packet_end_idx
        add2log_packet(f"{t.time()},{idx},{has_sync_sequence},{has_frame}")
        if settings["log"]["debugging"] : print ( f"{ idx= } { has_sync_sequence= }, { has_frame= }" )
        return idx

    def plot_complex_samples_filtered ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = peaks )

    def __repr__ ( self ) -> str :
        return ( f"{ self.frames.size= } , dtype = { self.frames.dtype= }")

@dataclass ( slots = True , eq = False )
class RxSamples_v0_1_12 :
    
    pluto_rx_ctx : Pluto | None = None
    #samples_filename : str | None = None

    # Pola uzupełnianie w __post_init__
    #samples : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    samples : NDArray[ np.complex128 ] = field ( init = False )
    samples_filtered : NDArray[ np.complex128 ] = field ( init = False )
    has_amp_greater_than_ths : bool = False
    ths : float = 1000.0
    sync_sequence_peaks : NDArray[ np.uint32 ] = field ( init = False )
    frames : RxFrames_v0_1_12 = field ( init = False )
    samples_leftovers : NDArray[ np.complex128 ] | None = field ( default = None )

    def __post_init__ ( self ) -> None :
            self.samples = np.array ( [] , dtype = np.complex128 )
            self.samples_filtered = np.array ( [] , dtype = np.complex128 )

    def rx ( self , previous_samples_leftovers : NDArray[ np.complex128 ] | None = None , samples_filename : str | None = None ) -> None :
        if self.pluto_rx_ctx is not None :
            if previous_samples_leftovers is None :
                self.samples = self.pluto_rx_ctx.rx ()
            else :
                self.samples = np.concatenate ( [ previous_samples_leftovers , self.pluto_rx_ctx.rx () ] )
            self.sample_initial_assesment ()
        elif samples_filename is not None :
            if samples_filename.endswith('.npy'):
                self.samples = ops_file.open_samples_from_npf ( samples_filename )
            elif samples_filename.endswith('.csv'):
                self.samples = ops_file.open_csv_and_load_np_complex128 ( samples_filename )
            else:
                raise ValueError(f"Error: unsupported file format for {samples_filename}! Supported formats: .npy, .csv")
            if previous_samples_leftovers is not None :
                self.samples = np.concatenate ( [ previous_samples_leftovers , self.samples ] )
        else :
            raise ValueError ( "Either pluto_rx_ctx or samples_filename must be provided." )

    def filter_samples ( self ) -> None :
        self.samples_filtered = filters.apply_rrc_rx_filter_v0_1_6 ( self.samples )

    def detect_frames ( self ) -> None :
        self.filter_samples ()
        self.frames = RxFrames_v0_1_12 ( samples_filtered = self.samples_filtered )
        if self.frames.has_leftovers :
            self.clip_samples_leftovers ()

    def sample_initial_assesment (self) -> None :
        self.has_amp_greater_than_ths = np.any ( np.abs ( self.samples ) > self.ths )

    def plot_complex_samples ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples , f"RxSamples {title} {self.samples.size=}" , marker_squares = marker , marker_peaks = peaks )

    def plot_complex_samples_filtered ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"RxSamples filtered {title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = peaks )

    def plot_complex_samples_corrected ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_corrected , f"RxSamples corrected {title} {self.samples_corrected.size=}" , marker_squares = marker , marker_peaks = peaks )

    def save_complex_samples_2_npf ( self , filename : str ) -> None :
        filename_with_timestamp = add_timestamp_2_filename ( filename )
        ops_file.save_complex_samples_2_npf ( filename_with_timestamp , self.samples )

    def save_complex_samples_2_csv ( self , filename : str ) -> None :
        filename_with_timestamp = add_timestamp_2_filename ( filename )
        ops_file.save_complex_samples_2_csv ( filename_with_timestamp , self.samples )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.samples.size= }, dtype = { self.samples.dtype= } { self.pluto_rx_ctx= }" if self.pluto_rx_ctx is not None else f"{ self.samples_filename= }"
        )

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
        self.samples_leftovers = self.samples [ self.frames.samples_leftovers_start_idx : ]

@dataclass ( slots = True , eq = False )
class RxPluto_v0_1_12 :

    sn : str | None = None
    
    # Pola uzupełnianie w __post_init__
    pluto_rx_ctx : Pluto | None = None
    samples : RxSamples_v0_1_12 = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.init_pluot_rx ()

    def init_pluot_rx ( self ) -> None :
        if self.sn is not None :
            self.pluto_rx_ctx = sdr.init_pluto_v0_1_9 ( sn = self.sn )
            self.samples = RxSamples_v0_1_12 ( pluto_rx_ctx = self.pluto_rx_ctx )
        else :
            self.samples = RxSamples_v0_1_12 ()

    def __repr__ ( self ) -> str :
        return (
            f"{self.samples.samples.size=}, { self.pluto_rx_ctx= }" if self.sn is not None else f" no ADALM-Pluto connected,"
        )

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
class TxFrame_v0_1_11 :

    tx_packet : TxPacket_v0_1_11
        
    # Pola uzupełnianie w __post_init__
    frame_bytes : NDArray[ np.uint8 ] = field ( init = False )
    frame_bits : NDArray[ np.uint8 ] = field ( init = False )

    def __post_init__ ( self ) -> None :
        sync_sequence_bits : NDArray[ np.uint8 ] = self.create_sync_sequence_bits ()
        packet_len_bits : NDArray[ np.uint8 ] = self.create_packet_len_bits ()
        frame_main_bytes = pad_bits2bytes ( np.concatenate ( [ sync_sequence_bits , packet_len_bits ] ) )
        crc32_bytes = self.create_crc32_bytes ( frame_main_bytes )
        self.frame_bytes = np.concatenate ( [ frame_main_bytes , crc32_bytes , self.tx_packet.packet_bytes ] )
        self.create_frame_bits ()

    def create_sync_sequence_bits ( self ) -> NDArray[ np.uint8 ] :
        return BARKER13_BITS

    def create_packet_len_bits ( self ) -> NDArray[ np.uint8 ] :
        return dec2bits ( self.tx_packet.packet_len , PACKET_LEN_LEN_BITS )

    def create_crc32_bytes ( self , frame_main_bytes ) -> NDArray[ np.uint8 ] :
        return create_crc32_bytes ( frame_main_bytes )
    
    def create_frame_bits ( self ) -> None :
        self.frame_bits = bytes2bits ( self.frame_bytes )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.frame_bytes= }, { self.frame_bytes.size= }" )

@dataclass ( slots = True , eq = False )
class TxSamples_v0_1_12 :

    pluto_tx_ctx : Pluto | None = None

    # Pola uzupełnianie w __post_init__
    payload_bytes : list | tuple | np.ndarray = field ( init = False )
    samples_bpsk_symbols : NDArray[ np.uint8 ] = field ( init = False )
    samples_filtered : NDArray[ np.complex128 ] = field ( init = False )
    samples4pluto : NDArray[ np.complex128 ] = field ( init = False )
    frame : TxFrame_v0_1_11 = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.create_empty_complex_samples ()

    def create_empty_complex_samples ( self ) -> None :
        self.samples_filtered = np.array ( [] , dtype = np.complex128 )
        self.samples4pluto = np.array ( [] , dtype = np.complex128 )

    def create_samples4pluto ( self , payload_bytes : list | tuple | NDArray[ np.uint8 ] = None , payload_bits : list | tuple | NDArray[ np.uint8 ] = None ) -> None :
        if payload_bytes is not None and len ( payload_bytes ) > 0 :
            payload_arr = np.asarray ( payload_bytes , dtype = np.uint8 ).ravel ()
            if payload_arr.max () > 255 :
                raise ValueError ( "Error: Payload has not all values in 0 - 255!")
            if len ( payload_arr ) > MAX_ALLOWED_PAYLOAD_LEN_BYTES_LEN :
                raise ValueError ( "Error: Payload exceeds maximum allowed length!" )
            self.payload_bytes = payload_arr
        elif payload_bits is not None and len ( payload_bits ) > 0 :
            payload_arr = np.asarray ( self.payload_bits , dtype = np.uint8 ).ravel ()
            if payload_arr.max () > 1 :
                raise ValueError ( "Error: Payload has not all values only: zeros or ones!" )
            if len ( payload_arr ) > MAX_ALLOWED_PAYLOAD_LEN_BYTES_LEN * 8 :
                raise ValueError ( "Error: Payload exceeds maximum allowed length!" )
            self.payload_bytes = pad_bits2bytes ( payload_arr )
        else :
            raise ValueError ( "Either payload_bytes or payload_bits must be provided." )
        self.create_tx_frame ()
        self.create_samples_bpsk_symbols ()
        self.create_samples_filtered ()
        self.create_samples_4pluto ()

    def create_tx_frame ( self ) -> None :
        tx_packet = TxPacket_v0_1_11 ( payload_bytes = self.payload_bytes )
        self.frame = TxFrame_v0_1_11 ( tx_packet = tx_packet )

    def create_samples_bpsk_symbols ( self ) -> None :
        self.samples_bpsk_symbols = modulation.create_bpsk_symbols_v0_1_6_fastest_short ( self.frame.frame_bits )

    def create_samples_filtered ( self ) -> None :
        self.samples_filtered = np.ravel ( filters.apply_tx_rrc_filter_v0_1_6 ( self.samples_bpsk_symbols ) ).astype ( np.complex128 , copy = False )

    def create_samples_4pluto ( self ) -> None :
        self.samples4pluto = sdr.scale_to_pluto_dac_v0_1_11 ( samples = self.samples_filtered , scale = 1.0 )

    def tx ( self , repeat : np.uint32 = 1 ) -> None :
        self.pluto_tx_ctx.tx_destroy_buffer ()
        if repeat < 1 or repeat > 4294967295 :
            raise ValueError ( "Error: reapt value is out of the range! Allowed range is 1 to 4294967295." )
        self.pluto_tx_ctx.tx_cyclic_buffer = False
        while repeat :
            self.pluto_tx_ctx.tx ( self.samples4pluto )
            repeat -= 1

    def tx_cyclic ( self ) -> None :
        self.pluto_tx_ctx.tx_destroy_buffer ()
        self.pluto_tx_ctx.tx_cyclic_buffer = True
        self.pluto_tx_ctx.tx ( self.samples4pluto )

    def stop_tx_cyclic ( self ) :
        self.pluto_tx_ctx.tx_destroy_buffer ()
        self.pluto_tx_ctx.tx_cyclic_buffer = False

    def tx_incremeant_payload_and_repeat ( self , nob : np.uint16 = 1 , repeat : np.uint32 = 1 ) -> None :
        self.pluto_tx_ctx.tx_destroy_buffer ()
        self.pluto_tx_ctx.tx_cyclic_buffer = False
        bytes = np.zeros ( nob , dtype = np.uint8 )
        while repeat :
            self.create_samples4pluto ( payload_bytes = bytes )
            self.pluto_tx_ctx.tx ( self.samples4pluto)
            print ( f"\n\r{repeat}: Transmitted payload bytes: { bytes }" )
            for i in range ( nob - 1 , -1 , -1 ) :
                bytes [ i ] = np.uint8( ( int(bytes [ i ]) + 1 ) % 256 )
                if bytes [ i ] != 0 :
                    break
            repeat -= 1
        

    def plot_symbols ( self , title = "" , constellation : bool = False ) -> None :
        plot.plot_symbols ( self.samples_bpsk_symbols , f"{title}" )
        if constellation :
            plot.complex_symbols_v0_1_6 ( self.samples_bpsk_symbols , f"{title}" )

    def plot_complex_samples_filtered ( self , title = "" ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title} {self.samples_filtered.size=}" , marker_squares = False )

    def plot_complex_samples4pluto ( self , title = "" ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples4pluto , f"{title} {self.samples4pluto.size=}" , marker_squares = False )

    def plot_samples_spectrum ( self , title = "" ) -> None :
        plot.spectrum_occupancy ( self.samples4pluto , 1024 , title )

    def __repr__ ( self ) -> str :
        return ( f"{ self.frame.frame_bytes= }, { self.samples.size= }" )

@dataclass ( slots = True , eq = False )
class TxPluto_v0_1_12 :
    
    sn : str
    
    # Pola uzupełnianie w __post_init__
    pluto_tx_ctx : Pluto  = field ( init = False )
    samples : TxSamples_v0_1_12 = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.init_pluto_tx ()

    def init_pluto_tx ( self ) -> None :
        self.pluto_tx_ctx = sdr.init_pluto_v0_1_9 ( sn = self.sn )
        self.samples = TxSamples_v0_1_12 ( pluto_tx_ctx = self.pluto_tx_ctx )

    def __repr__ ( self ) -> str :
        return ( f"{ self.pluto_tx_ctx= }, { self.samples.samples4pluto.size= }" )