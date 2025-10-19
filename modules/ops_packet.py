import zlib
import numpy as np

def bits_2_byte_list ( bits : np.ndarray ) :
    """
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


BARKER13_BITS = [ 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 ]
PADDING_BITS = [ 0 , 0 , 0 ] # Padding to 16 bits for BARKER13 (added at the end - LSB)
BARKER13_W_PADDING_BITS = np.array ( BARKER13_BITS + PADDING_BITS , dtype = np.uint8 )
BARKER13_W_PADDING_BYTES = bits_2_byte_list ( BARKER13_W_PADDING_BITS )
BARKER13_W_PADDING_INT = bits_2_int ( BARKER13_W_PADDING_BITS )
#BARKER13_W_PADDING = [ 6 , 80 ] # Jak będzie błąd to zamienić na BARKER13_W_PADDING_BYTES
#BARKER13_W_PADDING_INT = 1616 # Jak będzie błąd to zamienić na BARKER13_W_PADDING_BYTES
#PREAMBLE_BITS_LEN = 16
PREAMBLE_BITS_LEN = len ( BARKER13_W_PADDING_BITS )
PAYLOAD_LENGTH_BITS_LEN = 8
CRC32_BITS_LEN = 32

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
    # Korelacja z surowymi próbkami (na magnitude)
    corr = np.abs ( np.correlate ( samples , sync_seq , mode='full' ) )
    if len ( corr ) == 0 :
        return False
    # Próg na podstawie uśrednionej energii
    max_index = np.argmax ( corr )
    window = samples[ max_index : max_index + len ( sync_seq ) ]
    mean_amplitude = np.mean ( np.abs ( window ) )
    print ( f"{mean_amplitude=}" )
    return mean_amplitude > 100

def fast_energy_gate ( samples , power_threshold_dB = -30 ) :
    # użyj tylko energii w pasmach, bez korelacji:
    # Zlicz średnią moc w całym buforze,
    # Jeśli energia przekracza ustalony próg → prawdopodobnie transmisja.
    avg_power = np.mean ( np.abs ( samples ) **2 )
    power_db = 10 * np.log10 ( avg_power + 1e-12 )
    print ( f"{power_db=}")
    return power_db > power_threshold_dB

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