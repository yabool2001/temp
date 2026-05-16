import random , numpy as np , tomllib

from modules import packet
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

PAYLOAD_1BYTE_DEC_15 = [ 15 ]
PAYLOAD_4BYTES_DEC_15 = [ 15, 15 , 15 , 15 ]
PAYLOAD_BYTES = [ 0 ]

PAYLOAD_4BYTES_DEC = [ i % 256 for i in range ( 4 ) ]
if settings["log"]["verbose_2"] : print ( f"Payload test data initialized: { len ( PAYLOAD_4BYTES_DEC ) } bytes." )
PAYLOAD_10BYTES_DEC = [ i % 256 for i in range ( 10 ) ]
if settings["log"]["verbose_2"] : print ( f"{ len ( PAYLOAD_10BYTES_DEC )= } { PAYLOAD_10BYTES_DEC }" )
PAYLOAD_1500BYTES_DEC = [ i % 256 for i in range ( 1500 ) ]
if settings["log"]["verbose_2"] : print ( f"Payload test data initialized: { len ( PAYLOAD_1500BYTES_DEC ) } bytes." )

def generate_payload_i_bytes_dec_15 ( i : int ) -> list [ int ] :
    return [ 15 for _ in range ( i ) ]

def generate_payload_rand_up_2_1500b () -> list [ int ] :
    payload_len = random.randint ( 1 , 1500 )
    return [ random.randint ( 0 , 255 ) for _ in range ( payload_len ) ]

def fill_samples_up_to_max_length ( tx_samples : packet.TxSamples_v0_1_18 , max_samples_size : int ) -> None :
    payload_sizes = np.array ( [] ).astype ( np.uint32 )
    while tx_samples.samples4pluto.size < max_samples_size :
        payload_bytes = generate_payload_rand_up_2_1500b ()
        if not np.any ( payload_sizes == np.uint32 ( len ( payload_bytes ) ) ):
            payload_sizes = np.append ( payload_sizes , np.uint32 ( len ( payload_bytes )  ) )
            tx_samples.add_frame ( payload_bytes = payload_bytes )
