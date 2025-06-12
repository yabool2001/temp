import zlib
import numpy as np

PREAMBLE = [ 0xAA , 0x0F , 0xAA ]
HEADER = [ 0xAA , 0xAA , 0xAA]

def create_packet ( payload ) :
    length_byte = [ len ( payload ) - 1 ]
    if length_byte[0] < 0 and length_byte[0] > 255 :
        return None
    crc32 = zlib.crc32 ( bytes ( payload ) )
    crc_bytes = list ( crc32.to_bytes ( 4 , 'big' ) )
    return PREAMBLE + HEADER + length_byte + payload + crc_bytes

def create_bpsk_symbols ( packet ) :
    bits = np.unpackbits ( np.array ( packet , dtype = np.uint8 ) )
    return np.array ( [ 1.0 if bit else -1.0 for bit in bits ] , dtype = np.int64 )