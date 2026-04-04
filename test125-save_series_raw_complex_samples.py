'''
Skrypt do zapisywania do pliku sampli wygenerowanych przez skrypt python test126-tx_large_data.py.

Sekwencja uruchomienia skryptu w ubuntu:
cd ~/python/temp/
source .venv/bin/activate

python test126-tx_large_data.py -10.0

'''
from modules import packet , sdr
from pathlib import Path
import numpy as np
import os , sys , tomllib
import socket
import time as t
import tomllib

Path ( "np.samples_series_01" ).mkdir ( parents = True , exist_ok = True )

debug = True
plt = True
wrt = True
del_old = True

UDP_DEST_IP = "192.168.1.50" # ubuntu
UDP_TARGET_PORT = 10001
ASCII_EOT = b'\x04' # Sygnał zakończenia transmisji danych przez skrypt tx
ASCII_ENQ = b'\x05' # Sygnał do rozpoczęcia transmisji danych przez skrypt tx
ASCII_CAN = b'\x18' # Sygnał do zakończenia pracy skryptu tx
end_rx = False

filename = "np.samples_series_01/rx_samples.npy"
series_len = 10

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

if len ( sys.argv ) > 1 :
    n_o_bytes_uint16 = np.uint16 ( int ( sys.argv[ 1 ] ) )
    n_o_repeats_uint32 = np.uint32 ( int ( sys.argv[ 2 ] ) )
    tx_gain_float = float ( sys.argv[ 3 ] )
else :
    n_o_bytes_uint16 = np.uint16 ( 4 )
    n_o_repeats_uint32 = np.uint32 ( 10 )
    tx_gain_float = float ( toml_settings["ADALM-Pluto"][ "TX_GAIN" ] )

if del_old :
    for file_path in Path ( "np.samples_series_01" ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

rx_pluto = packet.RxPluto_v0_1_17 ( sn = sdr.PLUTO_RX_SN )
# print ( f"\n{ script_filename= } receiving: {rx_pluto=} { rx_pluto.samples.samples.size= }" )
rx_samples = packet.RxSamples_v0_1_17 ( pluto_rx_ctx = rx_pluto.pluto_rx_ctx )
if debug : print ( f"\n{ script_filename= } { rx_samples.samples.size= }" )

udp_sock = socket.socket ( socket.AF_INET , socket.SOCK_DGRAM )
udp_sock.setblocking ( False )
payload_udp = b""

try :
    udp_sock.sendto ( ASCII_ENQ , ( UDP_DEST_IP , UDP_TARGET_PORT ) )
    if debug : print ( f"UDP source socket: { udp_sock.getsockname ()[ 0 ] }:{ udp_sock.getsockname ()[ 1 ] }" )
    if debug : print ( f"Sent ASCII_ENQ to { UDP_DEST_IP }:{ UDP_TARGET_PORT }" )
    while True :
        rx_samples.rx ()
        if wrt :
            rx_samples.save_complex_samples_2_npf ( filename )
        if end_rx :
            if debug : print ( f"End of reception, stopping { script_filename }!" )
            break
        try :
            payload_udp = udp_sock.recv ( 1 )
            if debug : print ( f"\n\r[UDP] Received { len ( payload_udp ) } byte(s): {payload_udp=}" )
        except BlockingIOError :
            t.sleep ( 0.05 )  # odciążenie CPU, gdy nie ma danych do odbioru
            pass
        except Exception :
            pass
        if payload_udp == ASCII_EOT : # END OF TRANSMISSION
            if debug : print ( f"Received ASCII_EOT {payload_udp=}, stopping transmission!" )
            udp_sock.sendto ( ASCII_CAN , ( UDP_DEST_IP , UDP_TARGET_PORT ) )
            if debug : print ( f"Sent ASCII_CAN {ASCII_CAN} to { UDP_DEST_IP }:{ UDP_TARGET_PORT }" )
            end_rx = True
        t.sleep ( 0.05 )  # odciążenie CPU
finally :
    udp_sock.close ()
    exit ( 0 )

        