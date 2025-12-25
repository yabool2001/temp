import csv
import zlib
import numpy as np
import os
import tomllib

from dataclasses import dataclass , field
from modules import filters , modulation, plot , sdr
from numpy.typing import NDArray

from pathlib import Path
from scipy.signal import find_peaks

script_filename = os.path.basename ( __file__ )

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

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
PADDING_BITS = np.array ( settings[ "PADDING_BITS" ] , dtype = np.uint8 )
#BARKER13_W_PADDING_BITS = np.array ( BARKER13_BITS + PADDING_BITS , dtype = np.uint8 )
BARKER13_W_PADDING_BITS = np.concatenate ( ( BARKER13_BITS , PADDING_BITS ) ).astype ( np.uint8 )
BARKER13_W_PADDING_BYTES = bits_2_byte_list ( BARKER13_W_PADDING_BITS )
BARKER13_W_PADDING_INT = bits_2_int ( BARKER13_W_PADDING_BITS )
#BARKER13_W_PADDING = [ 6 , 80 ] # Jak będzie błąd to zamienić na BARKER13_W_PADDING_BYTES
#BARKER13_W_PADDING_INT = 1616 # Jak będzie błąd to zamienić na BARKER13_W_PADDING_BYTES
#PREAMBLE_BITS_LEN = 16
PREAMBLE_BITS_LEN = len ( BARKER13_W_PADDING_BITS )
PAYLOAD_LENGTH_BITS_LEN = 8
CRC32_BITS_LEN = 32

SYNC_SEQUENCE_LEN_BITS = len ( BARKER13_BITS )
PACKET_LEN_BITS = 11
CRC32_LEN_BITS = 32


def detect_sync_sequence_peaks_v0_1_7  ( samples: NDArray[ np.complex128 ] , sync_sequence : NDArray[ np.complex128 ] ) -> NDArray[ np.uint32 ] :

    plt = False
    wrt = True
    sync = False
    base_path = Path ( "logs/correlation_results.csv" )
    corr_2_amp_min_ratio = 12.0

    peaks = np.array ( [] ).astype ( np.uint32 )
    sync = False
    max_amplitude = np.max ( np.abs ( samples ) )
    #avg_amplitude = np.mean(np.abs(scenario['sample']))
    #percentile_95 = np.percentile(np.abs(scenario['sample']), 95)
    #rms_amplitude = np.sqrt(np.mean(np.abs(scenario['sample'])**2))

    corr_bpsk = np.correlate ( samples , sync_sequence , mode = "valid" )
    corr_real = np.abs ( corr_bpsk.real )
    corr_imag = np.abs ( corr_bpsk.imag )
    corr_abs = np.abs ( corr_bpsk )

    max_peak_real_val = np.max ( corr_real )
    max_peak_imag_val = np.max ( corr_imag )
    max_peak_abs_val = np.max ( corr_abs )

    corr_2_amp = np.max ( [ max_peak_real_val , max_peak_imag_val , max_peak_abs_val ] ) / max_amplitude

    if corr_2_amp > corr_2_amp_min_ratio :
        sync = True
        # Znajdź peaks powyżej threshold i z prominence dla real, imag, abs
        #peaks_real, _ = find_peaks ( corr_real , height = max_peak_real_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS , prominence = 0.5 )
        peaks_real , _ = find_peaks ( corr_real , height = max_peak_real_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
        peaks_imag , _ = find_peaks ( corr_imag , height = max_peak_imag_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
        peaks_abs , _ = find_peaks ( corr_abs , height = max_peak_abs_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
        peaks = np.unique ( np.concatenate ( ( peaks_real , peaks_imag , peaks_abs ) ) ).astype ( np.uint32 )

    
    if plt and sync :
        plot.real_waveform_v0_1_6 ( corr_abs , f"V7 corr abs {samples.size=}" , False , peaks_abs )
        plot.complex_waveform_v0_1_6 ( samples , f"V7 samples abs {samples.size=}" , False , peaks_abs )
        plot.real_waveform_v0_1_6 ( corr_real , f"V7 corr real {samples.size=}" , False , peaks_real )
        plot.real_waveform_v0_1_6 ( samples.real , f"V7 samples real {samples.size=}" , False , peaks_real )
        plot.real_waveform_v0_1_6 ( corr_imag , f"V7 corr imag {samples.size=}" , False , peaks_imag )
        plot.real_waveform_v0_1_6 ( samples.imag , f"V7 samples imag {samples.size=}" , False , peaks_imag )
        plot.complex_waveform_v0_1_6 ( corr_bpsk , f"V7 corr all {samples.size=}" , False , peaks )
        plot.complex_waveform_v0_1_6 ( samples , f"V7 samples all {samples.size=}" , False , peaks )

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

def create_packet_bits ( payload ) :
    length_byte = [ len ( payload ) - 1 ]
    crc32 = zlib.crc32 ( bytes ( payload ) )
    crc32_bytes = list ( crc32.to_bytes ( 4 , 'big' ) )
    print ( f"{BARKER13_W_PADDING_BITS=}, {length_byte=}, {payload=}, {crc32_bytes=}")
    preamble_bits = gen_bits ( BARKER13_W_PADDING_BYTES )
    header_bits = gen_bits ( length_byte )
    payload_bits = gen_bits ( payload )
    crc32_bits = gen_bits ( crc32_bytes )
    return np.concatenate ( [ preamble_bits , header_bits , payload_bits , crc32_bits ] )

def create_doubled_payload_packet_bits ( payload ) :
    print ( f"{payload=}")
    payload_bits = gen_bits ( payload )
    return np.concatenate ( [ payload_bits , payload_bits ] )

def is_preamble_v0_1_3 ( samples : np.ndarray ) -> bool :
    preamble_start = filters.SPAN * modulation.SPS // 2
    preamble_end = preamble_start + ( PREAMBLE_BITS_LEN * modulation.SPS )
    symbols = samples [ preamble_start : preamble_end : modulation.SPS ]
    bits = ( symbols.real > 0 ).astype ( int )
    preamble_int = bits_2_int ( bits )
    if preamble_int == BARKER13_W_PADDING_INT :
        return True
    return False

def is_preamble ( samples , rrc_span , sps ) :
    preamble_start = rrc_span * sps // 2
    preamble_end = preamble_start + ( PREAMBLE_BITS_LEN * sps )
    symbols = samples [ preamble_start : preamble_end : sps ]
    bits = ( symbols.real > 0 ).astype ( int )
    preamble_int = bits_2_int ( bits )
    if preamble_int == BARKER13_W_PADDING_INT :
        return True
    return False

def is_sync_seq ( samples , sync_seq ) :
    """
    Detect presence of sync sequence using normalized cross-correlation and a power gate.

    This replaces the previous fragile amplitude-only test which used a hard-coded
    threshold (mean_amplitude > 100) and produced false positives when the
    received samples had different scaling. The new method computes a normalized
    correlation (0..1) and also checks the mean power (dB) in the best-matching
    window to avoid detecting low-energy noise.

    Returns:
        bool
    """
    x = np.asarray(samples)
    tpl = np.asarray(sync_seq)
    n = len(tpl)
    m = len(x)
    if m < n or n == 0:
        return False

    # valid cross-correlation (tpl reversed) -> length m-n+1
    # use complex conjugate on template for proper correlation with complex samples
    corr = np.abs(np.correlate(x, tpl.conj()[::-1], mode='valid'))
    if corr.size == 0:
        return False

    # template energy
    tpl_energy = np.sum(np.abs(tpl) ** 2)

    # rolling window energy for received samples (efficient via cumsum)
    x_sq = np.abs(x) ** 2
    cumsum = np.concatenate(([0.0], np.cumsum(x_sq)))
    window_energy = cumsum[n:] - cumsum[:-n]

    # normalized correlation: corr / (sqrt(E_window * E_template))
    norm_corr = corr / (np.sqrt(window_energy * tpl_energy) + 1e-12)

    peak_idx = int(np.argmax(norm_corr))
    peak_val = float(norm_corr[peak_idx])

    # compute mean power in dB for the best window (helps to reject tiny noise peaks)
    mean_power = window_energy[peak_idx] / float(n)
    mean_power_db = 10 * np.log10(mean_power + 1e-12)

    # Diagnostic print (keeps previous behaviour of printing a diagnostic)
    print(f"is_sync_seq: peak_corr={peak_val:.4f}, mean_power_db={mean_power_db:.2f} dB")

    # Decision thresholds (tunable): require reasonably high normalized correlation
    # and a minimum power level. These defaults are conservative; adjust to taste.
    MIN_CORR = 0.55      # normalized correlation (0..1)
    MIN_POWER_DB = -40.0 # minimum mean power (dB) to accept as signal

    return (peak_val >= MIN_CORR) and (mean_power_db >= MIN_POWER_DB)

def fast_energy_gate ( samples , power_threshold_dB = -30 ) :
    # użyj tylko energii w pasmach, bez korelacji:
    # Zlicz średnią moc w całym buforze,
    # Jeśli energia przekracza ustalony próg → prawdopodobnie transmisja.
    avg_power = np.mean ( np.abs ( samples ) **2 )
    power_db = 10 * np.log10 ( avg_power + 1e-12 )
    print ( f"{power_db=}")
    return power_db > power_threshold_dB

def get_payload_bytes_v0_1_3 ( samples : np.ndarray ) :
    sps = modulation.SPS
    payload_length_start_index = ( filters.SPAN * modulation.SPS // 2 ) + ( PREAMBLE_BITS_LEN * sps )
    payload_length_end_index = payload_length_start_index + ( PAYLOAD_LENGTH_BITS_LEN * sps )
    symbols = samples [ payload_length_start_index : payload_length_end_index : modulation.SPS ]
    bits = ( symbols.real > 0 ).astype ( int )
    payload_length = bits_2_int ( bits ) + 1
    payload_start_index = payload_length_start_index + ( PAYLOAD_LENGTH_BITS_LEN * sps )
    payload_end_index = payload_start_index + ( payload_length * 8 * sps )
    # Zauważyłem, że oblicza bity i próbuje liczyć dla niej crc32 nawet jak ramka jest ucięta i brakuje ostatniego bajtu i nie ma 4 bajtów, np. dla symulacji w pliku "logs/rx_samples_987-no_crc32.csv". Oblicza bity payload, rzeczywista długość payload 3 bajty nie zgadza sie z wartością długości payload 3+1=4 w nagłówku.
    symbols = samples [ payload_start_index : payload_end_index : sps ]
    payload_bits = ( symbols.real > 0 ).astype ( int )
    try :
        crc32_calculated = zlib.crc32 ( bytes ( bits_2_byte_list ( payload_bits ) ) )
    except :
        print ( f"Brak całej ramki." )
        return None
    # Tu możesz też zapisać błąd do loga lub podjąć inne działanie
    crc32_start_index = payload_start_index + ( payload_length * 8 * sps )
    crc32_end_index = crc32_start_index + ( CRC32_BITS_LEN * sps )
    # Zauważyłem, że oblicza jakieś crc nawet jak ramka jest delikatnie ucięta i brakuje ostatniego symbolu, np. dla symulacji w pliku "logs/rx_samples_987-no_crc32.csv". Nie widać tego problemu, bo oblicza crc32, które nie zgadza sie z prawidłowym crc32 dla całego payload.
    symbols = samples [ crc32_start_index : crc32_end_index : sps ]
    bits = ( symbols.real > 0 ).astype ( int )
    crc32_received = bits_2_int ( bits )
    if crc32_calculated == crc32_received :
        return payload_bits
    else :
        print ( f"Brak całej ramki." )
        return None

def get_payload_bytes ( samples , rrc_span , sps ) :
    payload_length_start_index = ( rrc_span * sps // 2 ) + ( PREAMBLE_BITS_LEN * sps )
    payload_length_end_index = payload_length_start_index + ( PAYLOAD_LENGTH_BITS_LEN * sps )
    symbols = samples [ payload_length_start_index : payload_length_end_index : sps ]
    bits = ( symbols.real > 0 ).astype ( int )
    payload_length = bits_2_int ( bits ) + 1
    payload_start_index = payload_length_start_index + ( PAYLOAD_LENGTH_BITS_LEN * sps )
    payload_end_index = payload_start_index + ( payload_length * 8 * sps )
    # Zauważyłem, że oblicza bity i próbuje liczyć dla niej crc32 nawet jak ramka jest ucięta i brakuje ostatniego bajtu i nie ma 4 bajtów, np. dla symulacji w pliku "logs/rx_samples_987-no_crc32.csv". Oblicza bity payload, rzeczywista długość payload 3 bajty nie zgadza sie z wartością długości payload 3+1=4 w nagłówku.
    symbols = samples [ payload_start_index : payload_end_index : sps ]
    payload_bits = ( symbols.real > 0 ).astype ( int )
    try :
        crc32_calculated = zlib.crc32 ( bytes ( bits_2_byte_list ( payload_bits ) ) )
    except :
        print ( f"Brak całej ramki." )
        return None
    # Tu możesz też zapisać błąd do loga lub podjąć inne działanie
    crc32_start_index = payload_start_index + ( payload_length * 8 * sps )
    crc32_end_index = crc32_start_index + ( CRC32_BITS_LEN * sps )
    # Zauważyłem, że oblicza jakieś crc nawet jak ramka jest delikatnie ucięta i brakuje ostatniego symbolu, np. dla symulacji w pliku "logs/rx_samples_987-no_crc32.csv". Nie widać tego problemu, bo oblicza crc32, które nie zgadza sie z prawidłowym crc32 dla całego payload.
    symbols = samples [ crc32_start_index : crc32_end_index : sps ]
    bits = ( symbols.real > 0 ).astype ( int )
    crc32_received = bits_2_int ( bits )
    if crc32_calculated == crc32_received :
        return payload_bits
    else :
        print ( f"Brak całej ramki." )
        return None

@dataclass ( slots = True , eq = False )
class RxSamples_v0_1_7 :
    
    samples : NDArray[ np.complex128 ]

    # Pola uzupełnianie w __post_init__
    samples_filtered : NDArray[ np.complex128 ] = field ( init = False )
    sync_seguence_peaks : NDArray[ np.uint32 ] | None = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.samples_filtered = self.filter_samples ()
        self.sync_seguence_peaks = detect_sync_sequence_peaks_v0_1_7 ( self.samples_filtered , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) )
    
    def filter_samples ( self ) -> NDArray[ np.complex128 ] :
        return filters.apply_rrc_rx_filter_v0_1_6 ( self.samples )

    def plot_complex_waveform ( self , title = "" , marker : bool = False , peaks : bool = False ) -> None :
        if peaks and self.sync_seguence_peaks is not None :
            plot.complex_waveform_v0_1_6 ( self.samples , f"{title} {self.samples.size=}" , marker_squares = marker , marker_peaks = self.sync_seguence_peaks )
            plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = self.sync_seguence_peaks )
        else :
            plot.complex_waveform_v0_1_6 ( self.samples , f"{title}" , marker_squares = marker )
            plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title} {self.samples_filtered.size=}" , marker_squares = marker )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.samples.shape= } , dtype = { self.samples.dtype= }"
        )

    def clip_samples_filtered ( self , start : np.uint32 , end : np.uint32 ) -> None :
        if start < 0 or end > ( self.samples_filtered.size - 1 ) :
            raise ValueError ( "Start must be >= 0 & end cannot exceed samples length" )
        if start >= end :
            raise ValueError ( "Start must be < end" )
        #self.samples_filtered = self.samples_filtered [ start : end + 1 ]
        self.samples_filtered = self.samples_filtered [ start : end ]

@dataclass ( slots = True , eq = False )
class RxFrame_v0_1_7 :
    
    samples_filtered : NDArray[ np.complex128 ]

    # Pola uzupełnianie w __post_init__
    sync_sequence_start_idx : np.uint32 | None = field ( init = False )
    sync_sequence_end_idx : np.uint32 | None = field ( init = False )
    packet_len_start_idx : np.uint32 | None = field ( init = False )
    packet_len_end_idx : np.uint32 | None = field ( init = False )
    packet_len_dec : np.uint32 | None = field ( init = False )
    has_sync_sequence : bool = False

    def __post_init__ ( self ) -> None :
        self.plot_waveform ( f"RxFrame_v0_1_7 samples_filtered { self.samples_filtered.size= }" , marker = True )
        self.process_frame ()
    
    def process_frame ( self ) -> None :
        sps = modulation.SPS
        self.sync_sequence_start_idx = filters.SPAN * sps // 2
        self.sync_sequence_end_idx = self.sync_sequence_start_idx + ( SYNC_SEQUENCE_LEN_BITS * sps )
        self.packet_len_start_idx = self.sync_sequence_end_idx
        self.packet_len_end_idx = self.packet_len_start_idx + ( PACKET_LEN_BITS * sps )
        sync_sequence_symbols = self.samples_filtered [ self.sync_sequence_start_idx : self.sync_sequence_end_idx : sps ]
        sync_sequence_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( sync_sequence_symbols.real )
        if np.array_equal ( sync_sequence_bits , BARKER13_BITS ) :
            self.has_sync_sequence = True
            packet_len_symbols = self.samples_filtered [ self.packet_len_start_idx : self.packet_len_end_idx : sps ]
            packet_len_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( packet_len_symbols.real )
            self.packet_len_dec = bits_2_int ( packet_len_bits ) + 1
        else :
            sync_sequence_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( sync_sequence_symbols.imag )
            if np.array_equal ( sync_sequence_bits , BARKER13_BITS ) :
                self.has_sync_sequence = True
                packet_len_symbols = self.samples_filtered [ self.packet_len_start_idx : self.packet_len_end_idx : sps ]
                packet_len_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( packet_len_symbols.imag )
                self.packet_len_dec = bits_2_int ( packet_len_bits ) + 1
            else :
                sync_sequence_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( -sync_sequence_symbols.real )
                if np.array_equal ( sync_sequence_bits , BARKER13_BITS ) :
                    self.has_sync_sequence = True
                    packet_len_symbols = self.samples_filtered [ self.packet_len_start_idx : self.packet_len_end_idx : sps ]
                    packet_len_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( packet_len_symbols.imag )
                    self.packet_len_dec = bits_2_int ( packet_len_bits ) + 1
                else :
                    sync_sequence_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( -sync_sequence_symbols.imag )
                    if np.array_equal ( sync_sequence_bits , BARKER13_BITS ) :
                        self.has_sync_sequence = True
                        packet_len_symbols = self.samples_filtered [ self.packet_len_start_idx : self.packet_len_end_idx : sps ]
                        packet_len_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( packet_len_symbols.imag )
                        self.packet_len_dec = bits_2_int ( packet_len_bits ) + 1
    
    def get_bits_at_peak ( self , peak_idx : int ) -> NDArray[ np.uint8 ] | None :
        payload_bits = get_payload_bytes_v0_1_3 ( self.samples_filtered[ peak_idx : ] )
        return payload_bits
    
    def plot_waveform ( self , title = "" , marker : bool = False ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title}" , marker_squares = marker )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.samples_filtered.shape= } , dtype = { self.samples_filtered.dtype= }"
        )

@dataclass ( slots = True , eq = False )
class RxPackets :
    
    samples : NDArray[ np.complex128 ]

    # Pola uzupełnianie w __post_init__
    samples_filtered : NDArray[ np.complex128 ] = field ( init = False )
    sync_seguence_peak_idxs : NDArray[ np.uint32 ] | None = field ( init = False )
    sync_power_db : float | None = field ( init = False )
    max_amplitude : float | None = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.samples_filtered = self.filter_samples ()
        self.sync_seguence_peak_idxs = detect_sync_sequence_peaks_v0_1_7 ( self.samples_filtered , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) )
    
    def filter_samples ( self ) -> NDArray[ np.complex128 ] :
        return filters.apply_rrc_rx_filter_v0_1_6 ( self.samples )

    def get_bits_at_peak ( self , peak_idx : int ) -> NDArray[ np.uint8 ] | None :
        payload_bits = get_payload_bytes_v0_1_3 ( self.samples_filtered[ peak_idx : ] )
        return payload_bits
    
    def plot_waveform ( self , title = "" , marker : bool = False , peaks : bool = False ) -> None :
        if peaks and self.sync_seguence_peak_idxs is not None :
            plot.complex_waveform_v0_1_6 ( self.samples , f"{title}" , marker_squares = marker , marker_peaks = self.sync_seguence_peak_idxs )
        else :
            plot.complex_waveform_v0_1_6 ( self.samples , f"{title}" , marker_squares = marker )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.samples.shape= } , dtype = { self.samples.dtype= }"
        )

    def clip_samples ( self , start : int , end : int ) -> None :
        """Trim internal samples to the inclusive [ start , end ] range."""
        if start < 0 or end > ( self.samples.size - 1 ) :
            raise ValueError ( "start must be >= 0 & end cannot exceed samples length" )
        self.samples = self.samples [ start : end + 1 ]

@dataclass ( slots = True , eq = False )
class TxPacket :
    
    payload: list | tuple | np.ndarray = field ( default_factory = list )
    is_bits : bool = False
    
    # Pola uzupełnianie w __post_init__
    payload_bits : NDArray[ np.uint8 ] = field ( init = False )
    payload_bytes : NDArray[ np.uint8 ] = field ( init = False )
    payload_symbols : NDArray[ np.complex128 ] = field ( init = False )
    payload_samples : NDArray[ np.complex128 ] = field ( init = False )
    packet_bits : NDArray[ np.uint8 ] = field ( init = False )
    packet_symbols : NDArray[ np.uint8 ] = field ( init = False )
    packet_samples : NDArray[ np.uint8 ] = field ( init = False )

    def __post_init__ ( self ) -> None :
        
        self.create_payload_bits_and_bytes ()
        self.create_packet_bits ()
        self.payload_symbols = self.create_symbols ( self.payload_bits )
        self.packet_symbols = self.create_symbols ( self.packet_bits )
        self.payload_samples = self.create_samples_4pluto ( self.payload_symbols )
        self.packet_samples = self.create_samples_4pluto ( self.packet_symbols )

    def create_payload_bits_and_bytes ( self ) -> None :
        if not self.payload:
            raise ValueError ( "Error: Payload is empty!" )
        payload_arr = np.asarray ( self.payload , dtype = np.uint8 ).ravel ()
        if self.is_bits :         
            if payload_arr.max () > 1 :
                raise ValueError ( "Error: Payload has not all values only: zeros or ones!" )
            self.payload_bits = payload_arr.copy ()
            # dopełnij do pełnych bajtów (z prawej zerami) i spakuj
            pad = ( -len ( payload_arr ) ) % 8
            if pad :
                payload_arr = np.concatenate ( [ payload_arr , np.zeros ( pad , dtype = np.uint8 ) ] )
            self.payload_bytes = np.packbits ( payload_arr )
        else :
            # ------------------ payload podany jako bajty -----------------
            if payload_arr.max () > 255 :
                raise ValueError ( "Error: Payload has not all values in 0 - 255!")
            self.payload_bytes = payload_arr.copy ()
            self.payload_bits = np.unpackbits ( self.payload_bytes )   # zawsze MSB first

    def create_packet_bits ( self ) -> None:
        length_bytes = [ len ( self.payload_bytes ) - 1 ]
        crc32 = zlib.crc32 ( self.payload_bytes )
        crc32_bytes = list ( crc32.to_bytes ( 4 , 'big' ) )
        print ( f"{ BARKER13_W_PADDING_BITS= }, { length_bytes= }, { self.payload_bytes= }, { crc32_bytes= }")
        preamble_bits = gen_bits ( BARKER13_W_PADDING_BYTES )
        
        header_bits = gen_bits ( length_bytes )
        crc32_bits = gen_bits ( crc32_bytes )
        self.packet_bits = np.concatenate ( [ preamble_bits , header_bits , self.payload_bits , crc32_bits ] )

    def create_symbols ( self , bits : NDArray[ np.uint8 ] ) -> NDArray[ np.complex128 ] :
        return modulation.create_bpsk_symbols_v0_1_6_fastest_short ( bits )

    def create_samples_4pluto ( self , symbols : NDArray[ np.complex128 ] ) -> None :
        samples = np.ravel ( filters.apply_tx_rrc_filter_v0_1_6 ( symbols ) ).astype ( np.complex128 , copy = False )
        return sdr.scale_to_pluto_dac ( samples )

    def create_payload_samples_4pluto ( self ) -> None :
        self.payload_samples = np.ravel (
            filters.apply_tx_rrc_filter_v0_1_6 ( self.payload_symbols )
        ).astype ( np.complex128 , copy = False )
        self.payload_samples = sdr.scale_to_pluto_dac ( self.payload_samples )

    def plot_symbols ( self , symbols : NDArray[ np.complex128 ] , title = "" ) -> None :
        plot.plot_symbols ( symbols , f"{title}" )
        plot.complex_symbols_v0_1_6 ( symbols , f"{title}" )

    def plot_waveform ( self , samples : NDArray[ np.complex128 ] , title = "" , marker : bool = False ) -> None :
        plot.complex_waveform ( samples , f"{title}" , marker_squares = marker )

    def plot_spectrum ( self , samples : NDArray[ np.complex128 ] , title = "" ) -> None :
        plot.spectrum_occupancy ( samples , 1024 , title )

    def bytes2bits ( bytes : NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
        np.unpackbits ( np.array ( bytes , dtype = np.uint8 ) )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.bits.shape= } , dtype={ self.bits.dtype= }"
            f"{ self.bytes.shape= } , dtype={ self.bytes.dtype= }"
            f"{ self.symbols.shape= } , dtype={ self.symbols.dtype= }"
            f"{ self.samples.shape= } , dtype={ self.samples.dtype= }"
        )
    