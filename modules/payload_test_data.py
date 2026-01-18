PAYLOAD_4BYTES_DEC = [ i % 256 for i in range ( 4 ) ]
print ( f"Payload test data initialized: { len ( PAYLOAD_4BYTES_DEC ) } bytes." )
PAYLOAD_10BYTES_DEC = [ i % 256 for i in range ( 10 ) ]
print ( f"{ len ( PAYLOAD_10BYTES_DEC )= } { PAYLOAD_10BYTES_DEC }" )
PAYLOAD_1500BYTES_DEC = [ i % 256 for i in range ( 1500 ) ]
print ( f"Payload test data initialized: { len ( PAYLOAD_1500BYTES_DEC ) } bytes." )
def generate_payload_i_bytes_dec_15 ( i : int ) -> list [ int ] :
    return [ 15 for _ in range ( i ) ]
PAYLOAD_4BYTES_DEC_15 = [ 15, 15 , 15 , 15 ]
PAYLOAD_8BYTES_DEC_15 = [ 15, 15 , 15 , 15 , 15, 15, 15, 15 ]
PAYLOAD_12BYTES_DEC_15 = [ 15, 15 , 15 , 15 , 15, 15, 15, 15, 15, 15, 15, 15 ]
PAYLOAD_16BYTES_DEC_15 = [ 15, 15 , 15 , 15 , 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15 ]
PAYLOAD_32BYTES_DEC_15 = [ 15, 15 , 15 , 15 , 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15 ]