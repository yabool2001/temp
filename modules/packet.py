import csv
import zlib
import numpy as np
import os
import tomllib

from adi import Pluto
from dataclasses import dataclass , field
from modules import filters , modulation, ops_file, plot , sdr
from numpy.typing import NDArray

from pathlib import Path
from scipy.signal import find_peaks
from typing import Any

np.set_printoptions ( threshold = np.inf , linewidth = np.inf )
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
PACKET_LEN_LEN_BITS = 11
CRC32_LEN_BITS = 32
MAX_ALLOWED_PAYLOAD_LEN_BYTES_LEN = np.uint16 ( 2 ** PACKET_LEN_LEN_BITS - 1 )
MAX_RECOMMENDED_PAYLOAD_LEN_BYTES_LEN = 1500 # MTU dla IP over ETHERNET
PACKET_BYTE_LEN_BITS = 8
FRAME_LEN_BITS = SYNC_SEQUENCE_LEN_BITS + PACKET_LEN_LEN_BITS + CRC32_LEN_BITS
FRAME_LEN_SAMPLES = FRAME_LEN_BITS * modulation.SPS

def detect_sync_sequence_peaks_v0_1_7  ( samples: NDArray[ np.complex128 ] , sync_sequence : NDArray[ np.complex128 ] ) -> NDArray[ np.uint32 ] :

    plt = True
    wrt = False
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
        '''
        plot.real_waveform_v0_1_6 ( corr_abs , f"V7 corr abs {samples.size=}" , False , peaks_abs )
        plot.complex_waveform_v0_1_6 ( samples , f"V7 samples abs {samples.size=}" , False , peaks_abs )
        plot.real_waveform_v0_1_6 ( corr_real , f"V7 corr real {samples.size=}" , False , peaks_real )
        plot.real_waveform_v0_1_6 ( samples.real , f"V7 samples real {samples.size=}" , False , peaks_real )
        plot.real_waveform_v0_1_6 ( corr_imag , f"V7 corr imag {samples.size=}" , False , peaks_imag )
        plot.real_waveform_v0_1_6 ( samples.imag , f"V7 samples imag {samples.size=}" , False , peaks_imag )
        plot.complex_waveform_v0_1_6 ( corr_bpsk , f"V7 corr all {samples.size=}" , False , peaks )'''
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

def count_bytes ( payload , has_bits : bool = False ) -> np.uint64 :
    payload_arr = np.asarray ( payload , dtype = np.uint8 ).ravel ()
    if has_bits :
        payload_bytes_len = len ( payload_arr ) // 8
    else :
        payload_bytes_len = len ( payload_arr )
    return np.uint64 ( payload_bytes_len )

def create_crc32_bytes ( bytes : NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
    crc32 = zlib.crc32 ( bytes )
    return np.frombuffer ( crc32.to_bytes ( 4 , 'big' ) , dtype = np.uint8 )

@dataclass ( slots = True , eq = False )
class RxPacket_v0_1_8 :
    
    samples_filtered : NDArray[ np.complex128 ]
    has_packet : bool = False
    payload_bytes : NDArray[ np.uint8 ] = field ( init = False )
    # Pola uzupełnianie w __post_init__

    def __post_init__ ( self ) -> None :
        self.process_packet ( self.samples_filtered )
    
    def process_packet ( self , samples_filtered : NDArray[ np.complex128 ] ) -> None :
        sps = modulation.SPS

        samples_components = [ ( self.samples_filtered.real , "packet real" ) , ( self.samples_filtered.imag , "packet imag" ) , ( -self.samples_filtered.real , "packet -real" ) , ( -self.samples_filtered.imag , "packet -imag" ) ]
        for samples_component , samples_name in samples_components :
        
            payload_end_idx = len ( samples_filtered ) - ( CRC32_LEN_BITS * sps )
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
                print ( samples_name )
                return

    def __repr__ ( self ) -> str :
        return (
            f"{ self.samples_filtered.size= }, { self.has_packet= }, { self.payload_bytes.size if self.has_packet else None= }"
        )

@dataclass ( slots = True , eq = False )
class RxFrames_v0_1_9 :
    
    samples_filtered : NDArray[ np.complex128 ]

    # Pola uzupełnianie w __post_init__
    sync_seguence_peaks : NDArray[ np.uint32 ] = field ( init = False )
    samples_filtered_len : np.uint32 = field ( init = False )
    samples_leftovers_start_idx : np.uint32 = field ( init = False )
    #samples_payloads_bytes : list[ RxPacket_v0_1_8 ] = field ( default_factory = list )
    sps = modulation.SPS
    samples_payloads_bytes : NDArray[ np.uint8 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint8 ) , init = False )
    has_leftovers : bool = False
    
    def __post_init__ ( self ) -> None :
        self.samples_filtered_len = np.uint32 ( len ( self.samples_filtered ) )
        self.sync_seguence_peaks = detect_sync_sequence_peaks_v0_1_7 ( self.samples_filtered , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) )
        for idx in self.sync_seguence_peaks :
            self.process_frame ( idx = idx )
    
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
        print ( f"Samples at index { idx } is too close to the end of samples to contain a full frame. Skipping." )
        self.samples_leftovers_start_idx = idx
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

    def process_frame ( self , idx : np.uint32 ) -> None :
        # znajdz na drive plik Zrzut ekranu z 2025-12-30 09-28-42.png i obacz, który if by zadziałał. Roważ sprawdzenie -real - imag?!
        if not self.frame_len_validation ( idx ) :
            return
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
                print ( samples_name )
                has_sync_sequence = True
                packet_len_uint16 = self.samples2uint16 ( samples_component [ packet_len_start_idx : packet_len_end_idx ] )
                check_components = [ ( self.samples_filtered.real , " frame real" ) , ( self.samples_filtered.imag , " frame imag" ) , ( -self.samples_filtered.real , " frame -real" ) , ( -self.samples_filtered.imag , " frame -imag" ) ]
                for samples_comp , frame_name in check_components :
                    crc32_bytes_read = self.samples2bytes ( samples_comp [ crc32_start_idx : crc32_end_idx ] )
                    crc32_bytes_calculated = create_crc32_bytes ( pad_bits2bytes ( self.samples2bits ( samples_comp [ sync_sequence_start_idx : packet_len_end_idx ] ) ) )
                    if ( crc32_bytes_read == crc32_bytes_calculated ).all () :
                        print ( frame_name )
                        packet_end_idx = crc32_end_idx + ( packet_len_uint16 * PACKET_BYTE_LEN_BITS * self.sps )
                        if not self.packet_len_validation ( idx , packet_end_idx ) :
                            return
                        has_frame = True
                        packet = RxPacket_v0_1_8 ( samples_filtered = self.samples_filtered [ crc32_end_idx : packet_end_idx ] )
                        if packet.has_packet :
                            self.samples_payloads_bytes = np.concatenate ( [ self.samples_payloads_bytes , packet.payload_bytes ] )
                            return
                        #break # UWAGA! To chyba jest bez sensu
            
        print ( f"{ idx= } { has_sync_sequence= }, { has_frame= }" )

    def __repr__ ( self ) -> str :
        return ( f"{ self.frames.size= } , dtype = { self.frames.dtype= }")

@dataclass ( slots = True , eq = False )
class RxSamples_v0_1_9 :
    
    pluto_rx_ctx : Pluto | None = None
    #samples_filename : str | None = None

    # Pola uzupełnianie w __post_init__
    #samples : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    samples : NDArray[ np.complex128 ] = field ( init = False )
    samples_filtered : NDArray[ np.complex128 ] = field ( init = False )
    has_sync_sequence : bool = False
    has_amp_greater_than_ths : bool = False
    ths : float = 1000.0
    sync_seguence_peaks : NDArray[ np.uint32 ] = field ( init = False )
    frames : RxFrames_v0_1_9 = field ( init = False )
    samples_leftovers : NDArray[ np.complex128 ] | None = field ( default = None )

    def __post_init__ ( self ) -> None :
            self.samples = np.array ( [] , dtype = np.complex128 )

    def rx ( self , previous_samples_leftovers : NDArray[ np.complex128 ] , samples_filename : str | None = None ) -> None :
        if self.pluto_rx_ctx is not None :
            if previous_samples_leftovers is None :
                self.samples = self.pluto_rx_ctx.rx ()
            else :
                self.samples = np.concatenate ( [ previous_samples_leftovers , self.pluto_rx_ctx.rx () ] )
        elif samples_filename is not None :
            self.samples = ops_file.open_samples_from_npf ( samples_filename )
            if previous_samples_leftovers is not None :
                self.samples = np.concatenate ( [ previous_samples_leftovers , self.samples ] )
        else :
            raise ValueError ( "Either pluto_rx_ctx or samples_filename must be provided." )

    def filter_samples ( self ) -> None :
        self.samples_filtered = filters.apply_rrc_rx_filter_v0_1_6 ( self.samples )

    def detect_frames ( self ) -> None :
        self.filter_samples ()
        self.frames = RxFrames_v0_1_9 ( samples_filtered = self.samples_filtered )
        self.clip_samples_leftovers ()

    def sample_initial_assesment (self) -> None :
        self.has_amp_greater_than_ths = np.any ( np.abs ( self.samples ) > self.ths )

    def plot_complex_waveform ( self , title = "" , marker : bool = False , peaks : bool = False ) -> None :
        if peaks and self.sync_seguence_peaks is not None :
            plot.complex_waveform_v0_1_6 ( self.samples , f"{title} {self.samples.size=}" , marker_squares = marker , marker_peaks = self.sync_seguence_peaks )
            plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = self.sync_seguence_peaks )
        else :
            plot.complex_waveform_v0_1_6 ( self.samples , f"{title}" , marker_squares = marker )
            plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title} {self.samples_filtered.size=}" , marker_squares = marker )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.samples.size= }, dtype = { self.samples.dtype= } { self.pluto_rx_ctx= }" if self.pluto_rx_ctx is not None else f"{ self.samples_filename= }"
        )

    def clip_samples_filtered ( self , start : np.uint32 , end : np.uint32 ) -> None :
        if start < 0 or end > ( self.samples_filtered.size - 1 ) :
            raise ValueError ( "Start must be >= 0 & end cannot exceed samples length" )
        if start >= end :
            raise ValueError ( "Start must be < end" )
        #self.samples_filtered = self.samples_filtered [ start : end + 1 ]
        self.samples_filtered = self.samples_filtered [ start : end ]

    def clip_samples_leftovers ( self ) -> None :
        self.samples_leftovers = self.samples [ self.frames.samples_leftovers_start_idx : ]

@dataclass ( slots = True , eq = False )
class RxPluto_v0_1_9 :

    sn : str | None = None
    
    # Pola uzupełnianie w __post_init__
    pluto_rx_ctx : Pluto | None = None
    samples : RxSamples_v0_1_9 = field ( init = False )

    def __post_init__ ( self ) -> None :
        if self.sn is not None :
            self.pluto_rx_ctx = sdr.init_pluto_v3 ( sn = self.sn )
            self.samples = RxSamples_v0_1_9 ( pluto_rx_ctx = self.pluto_rx_ctx )
        else :
            self.samples = RxSamples_v0_1_9 ()

    def __repr__ ( self ) -> str :
        return (
            f"{ self.samples_filename= }" if self.samples_filename is not None else f"{ self.pluto_rx_ctx= }"
        )

@dataclass ( slots = True , eq = False )
class TxPacket_v0_1_8 :
    
    payload : list | tuple | np.ndarray = field ( default_factory = list )
    has_bits : bool = False
    
    # Pola uzupełnianie w __post_init__
    payload_bits : NDArray[ np.uint8 ] = field ( init = False )
    payload_bytes : NDArray[ np.uint8 ] = field ( init = False )
    crc32_bytes : NDArray[ np.uint8 ] = field ( init = False )
    packet_bytes : NDArray[ np.uint8 ] = field ( init = False )
    packet_len : np.uint16 = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.create_payload_bits_and_bytes ()
        self.create_crc32_bytes ()
        self.create_packet_bytes ()
        self.packet_len = np.uint16 ( len ( self.payload_bytes ) + len ( self.crc32_bytes ) )  # payload + crc32

    def create_payload_bits_and_bytes ( self ) -> None :
        if not self.payload or len ( self.payload ) == 0 or self.payload is None :
            raise ValueError ( "Error: Payload is empty!" )
        payload_arr = np.asarray ( self.payload , dtype = np.uint8 ).ravel ()
        if self.has_bits :         
            if payload_arr.max () > 1 :
                raise ValueError ( "Error: Payload has not all values only: zeros or ones!" )
            if len ( payload_arr ) > MAX_ALLOWED_PAYLOAD_LEN_BYTES_LEN * 8 :
                raise ValueError ( "Error: Payload exceeds maximum allowed length!" )
            if len ( payload_arr ) > MAX_RECOMMENDED_PAYLOAD_LEN_BYTES_LEN * 8 :
                raise ValueError ( "Error: Payload exceeds maximum recommended length!" )
            self.payload_bytes = pad_bits2bytes ( payload_arr )
        else :
            # ------------------ payload podany jako bajty -----------------
            if payload_arr.max () > 255 :
                raise ValueError ( "Error: Payload has not all values in 0 - 255!")
            if len ( payload_arr ) > MAX_ALLOWED_PAYLOAD_LEN_BYTES_LEN :
                raise ValueError ( "Error: Payload exceeds maximum allowed length!" )
            if len ( payload_arr ) > MAX_RECOMMENDED_PAYLOAD_LEN_BYTES_LEN :
                raise ValueError ( "Error: Payload exceeds maximum recommended length!" )
            self.payload_bytes = payload_arr

    def create_crc32_bytes ( self ) -> None :
        self.crc32_bytes = create_crc32_bytes ( self.payload_bytes )

    def create_packet_bytes ( self ) -> None:
        self.packet_bytes = np.concatenate ( [ self.payload_bytes , self.crc32_bytes ] )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.payload_bytes= }, { self.crc32_bytes= }, { self.packet_len= }" )

@dataclass ( slots = True , eq = False )
class TxFrame_v0_1_8 :

    packet_len : np.uint16
        
    # Pola uzupełnianie w __post_init__
    sync_sequence_bits : NDArray[ np.uint8 ] = field ( init = False )
    packet_len_bits : NDArray[ np.uint8 ] = field ( init = False )
    frame_bits : NDArray[ np.uint8 ] = field ( init = False )
    frame_bytes : NDArray[ np.uint8 ] = field ( init = False )
    crc32_bytes : NDArray[ np.uint8 ] = field ( init = False )
    tx_packet : TxPacket_v0_1_8 = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.create_sync_sequence_bits ()
        self.create_packet_len_bits ()
        frame_main_bytes = pad_bits2bytes ( np.concatenate ( [ self.sync_sequence_bits , self.packet_len_bits ] ) )
        self.create_crc32_bytes ( frame_main_bytes )
        self.frame_bytes = np.concatenate ( [ frame_main_bytes , self.crc32_bytes ] )

    def create_sync_sequence_bits ( self ) -> None :
        self.sync_sequence_bits = BARKER13_BITS

    def create_packet_len_bits ( self ) -> None :
        self.packet_len_bits = dec2bits ( self.packet_len , PACKET_LEN_LEN_BITS )

    def create_frame_bits ( self ) -> None :
        self.frame_bits = np.concatenate ( [ self.sync_sequence_bits , self.packet_len_bits ] )

    def create_crc32_bytes ( self , frame_main_bytes ) -> None :
        self.crc32_bytes = create_crc32_bytes ( frame_main_bytes )

    def __repr__ ( self ) -> str :
        return (
            f"{ self.frame_bytes= }, { self.frame_bits.size= }, { self.packet_len= }" )

@dataclass ( slots = True , eq = False )
class TxSamples_v0_1_8 :
    
    payload: list | tuple | np.ndarray = field ( default_factory = list )
    has_bits : bool = False
    
    # Pola uzupełnianie w __post_init__
    samples_bytes : NDArray[ np.uint8 ] = field ( init = False )
    samples_bits : NDArray[ np.uint8 ] = field ( init = False )
    samples_bpsk_symbols : NDArray[ np.uint8 ] = field ( init = False )
    samples : NDArray[ np.complex128 ] = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.create_samples_bytes ()
        self.create_samples_bits ()
        self.create_samples_bpsk_symbols ()
        self.create_samples_4pluto ()

    def create_samples_bytes ( self ) -> None :
        tx_packet = TxPacket_v0_1_8 ( payload = self.payload , has_bits = self.has_bits )
        tx_frame = TxFrame_v0_1_8 ( packet_len = tx_packet.packet_len )
        self.samples_bytes = np.concatenate ( ( tx_frame.frame_bytes , tx_packet.packet_bytes ) )

    def create_samples_bits ( self ) -> None:
        self.samples_bits = bytes2bits ( self.samples_bytes )

    def create_samples_bpsk_symbols ( self ) -> None :
        self.samples_bpsk_symbols = modulation.create_bpsk_symbols_v0_1_6_fastest_short ( self.samples_bits )

    def create_samples_4pluto ( self ) -> None :
        self.samples = np.ravel ( filters.apply_tx_rrc_filter_v0_1_6 ( self.samples_bpsk_symbols ) ).astype ( np.complex128 , copy = False )

    def plot_symbols ( self , title = "" ) -> None :
        plot.plot_symbols ( self.samples_bpsk_symbols , f"{title}" )
        #plot.complex_symbols_v0_1_6 ( self.samples_bpsk_symbols , f"{title}" )

    def plot_samples_waveform ( self , title = "" , marker : bool = False ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples , f"{title}" , marker_squares = marker )

    def plot_samples_spectrum ( self , title = "" ) -> None :
        plot.spectrum_occupancy ( self.samples , 1024 , title )

    def __repr__ ( self ) -> str :
        return ( f"{ self.samples_bytes= }, { self.samples.size= }" )

@dataclass ( slots = True , eq = False )
class TxPluto_v0_1_8 :
    
    # Pola uzupełnianie w __post_init__
    tx_samples : TxSamples_v0_1_8 = field ( init = False )

    samples4pluto : NDArray[ np.complex128 ] = field ( init = False )

    payload : list | tuple | np.ndarray = field ( default_factory = list )
    has_bits : bool = False
    pluto_tx_ctx : Pluto = field ( init = False )

    def __post_init__ ( self ) -> None :
        self.init_pluto_tx ()

    def create_samples_4pluto ( self ) -> None :
        if count_bytes ( self.payload , self.has_bits ) > MAX_RECOMMENDED_PAYLOAD_LEN_BYTES_LEN :
            raise ValueError ( "Payload size cannot exceed 1500 bytes (IP over ETHERNET MTU)" )
        self.tx_samples = TxSamples_v0_1_8 ( payload = self.payload , has_bits = self.has_bits )
        self.samples4pluto = sdr.scale_to_pluto_dac ( self.tx_samples.samples )

    def plot_symbols ( self , title = "" ) -> None :
        self.tx_samples.plot_symbols ( title )

    def plot_samples_waveform ( self , title = "" , marker : bool = False ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples4pluto , f"{title}" , marker_squares = marker )

    def plot_samples_spectrum ( self , title = "" ) -> None :
        plot.spectrum_occupancy ( self.samples4pluto , 1024 , title )
    
    def init_pluto_tx ( self ) -> None :
        self.pluto_tx_ctx = sdr.init_pluto_v3 ( sn = sdr.PLUTO_TX_SN )

    # Docelowo powina być tylko funkcja tx() bo jest bezpieczna. Po testach usunąc tx_once () i tx_cyclic ()
    def tx ( self , mode : str , payload : list | tuple | np.ndarray , has_bits : bool = False) :
        self.payload = payload
        self.has_bits = has_bits
        self.create_samples_4pluto ()
        self.pluto_tx_ctx.tx_destroy_buffer ()
        if mode == "once" :
            self.pluto_tx_ctx.tx_cyclic_buffer = False
        elif mode == "cyclic" :
            self.pluto_tx_ctx.tx_cyclic_buffer = True
        else :
            raise ValueError ( "Error: tx mode can be once or cyclic!" )
        self.pluto_tx_ctx.tx ( self.samples4pluto )

    def stop_tx_cyclic ( self ) :
        self.pluto_tx_ctx.tx_destroy_buffer ()
        self.pluto_tx_ctx.tx_cyclic_buffer = False

    def __repr__ ( self ) -> str :
        return ( f"{self.pluto_tx_ctx=}" )
