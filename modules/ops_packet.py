import zlib
import numpy as np

BARKER13 = [ 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 ]
PADDING_BITS = [ 0 , 0 , 0 ]
BARKER13_W_PADDING = [ 6 , 80 ]
BARKER13_W_PADDING_UINT16 = 1616
PREAMBLE_BITS_LEN = 16
PAYLOAD_LENGTH_BITS_LEN = 1
CRC32_BITS_LEN = 32

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

def create_doubled_payload_packet_bits ( payload ) :
    print ( f"{payload=}")
    payload_bits = gen_bits ( payload )
    return np.concatenate ( [ payload_bits , payload_bits ] )

def get_preamble_uint16_value ( samples , rrc_span , sps ) :
    header_start = rrc_span * sps
    symbols = samples [ header_start // 2 : header_start + ( PREAMBLE_BITS_LEN ) : sps ]
    return ( symbols.real > 0 ).astype ( int )