import zlib
import numpy as np

BARKER13 = [ 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 ]
PADDING_BITS = [ 0 , 0 , 0 ]
BARKER13_W_PADDING = [ 6 , 80 ]

def gen_bits ( bytes ) :
    return np.unpackbits ( np.array ( bytes , dtype = np.uint8 ) )

def create_packet_bits ( payload ) :
    length_byte = [ len ( payload ) - 1 ]
    crc32 = zlib.crc32 ( bytes ( payload ) )
    crc32_bytes = list ( crc32.to_bytes ( 4 , 'big' ) )
    print ( f"{BARKER13_W_PADDING=}, {length_byte=}, {payload=}, {crc32_bytes=}")
    preamble_bits = gen_bits ( BARKER13_W_PADDING )
    header_bits = gen_bits ( length_byte )
    payload_bits = gen_bits ( payload )
    crc32_bits = gen_bits ( crc32_bytes )
    return np.concatenate ( [ preamble_bits , header_bits , payload_bits , crc32_bits ] )
