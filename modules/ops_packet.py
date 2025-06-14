import zlib
import numpy as np

BARKER13 = [ 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 ]
# Dodaj 3 bity zerowe na końcu
HEADER = [ 1 , 0 , 1 ]
BPSK_PREAMBLE = BARKER13 + HEADER

def gen_bits ( bytes ) :
    return np.unpackbits ( np.array ( bytes , dtype = np.uint8 ) )

def create_packet ( payload ) :
    length_byte = [ len ( payload ) - 1 ]
    if length_byte[0] < 0 and length_byte[0] > 255 :
        return None
    crc32 = zlib.crc32 ( bytes ( payload ) )
    crc_bytes = list ( crc32.to_bytes ( 4 , 'big' ) )
    return BPSK_PREAMBLE + gen_bits ( length_byte + payload + crc_bytes )