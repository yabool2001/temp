'''
Skrypt do zapisywania do pliku sampli wygenerowanych przez skrypt python test126-tx_large_data.py.

Sekwencja uruchomienia skryptu w ubuntu:
cd ~/python/temp/
source .venv/bin/activate

python test126-tx_large_data.py -10.0

'''
from modules import ops_os , packet , sdr
from pathlib import Path
import numpy as np
import os , sys , tomllib
import socket
import time as t
import tomllib

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy

debug = True
plt = True
wrt = False
del_old = True

Nof_ATTEMPTS = int ( 1 )
Nof_WRTS = int ( 1 )
UDP_DEST_IP = "192.168.1.50" # ubuntu
UDP_TARGET_PORT = 10001
ASCII_EOT = b'\x04' # Sygnał zakończenia transmisji danych przez skrypt tx
ASCII_ENQ = b'\x05' # Sygnał do rozpoczęcia transmisji danych przez skrypt tx
ASCII_FF = b'\x0c'  # Sygnał do rozpoczęcia pracy skryptu (Form Feed)
ASCII_CAN = b'\x18' # Sygnał do zakończenia pracy skryptu tx
end_rx = False

dir_name = "np.tensors"
filename = "rx_samples.npy"
Path ( dir_name ).mkdir ( parents = True , exist_ok = True )

timestamp_min = int ( ops_os.milis_timestamp () ) - 1000 * 365 * 60 * 60 * 24 # -1Y
timestamp_max = int ( ops_os.milis_timestamp () ) + 1000 * 365 * 60 * 60 * 24 # +1Y

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
    for file_path in Path ( dir_name ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

rx_pluto = packet.RxPluto_v0_1_17 ( sn = sdr.PLUTO_RX_SN )
rx_samples = packet.RxSamples_v0_1_18 ()
if debug : print ( f"\n{ script_filename= } { rx_samples.samples.size= }" )

udp_sock = socket.socket ( socket.AF_INET , socket.SOCK_DGRAM )
#udp_sock.setblocking ( False )

udp_sock.sendto ( ASCII_FF , ( UDP_DEST_IP , UDP_TARGET_PORT ) )
if debug : print ( f"UDP source socket: { udp_sock.getsockname ()[ 0 ] }:{ udp_sock.getsockname ()[ 1 ] }" )
if debug : print ( f"Sent ASCII_FF to { UDP_DEST_IP }:{ UDP_TARGET_PORT }" )

rx_samples.rx ( sdr_ctx = rx_pluto.pluto_rx_ctx )
payload_udp = b""
try :
    j = Nof_ATTEMPTS
    while j :
    
        payload_udp , udp_sender_addr = udp_sock.recvfrom ( 20 )
        if debug : print ( f"\n\r[UDP] Received {len ( payload_udp )=} byte(s): {payload_udp=}" )

        if len ( payload_udp ) >= 13 and int ( payload_udp ) > timestamp_min and int ( payload_udp ) < timestamp_max : # Received a valid timestamp to name the file with received samples
            if debug : print ( f"Valid timestamp received: {payload_udp=}" )
            udp_sock.sendto ( ASCII_ENQ , ( UDP_DEST_IP , UDP_TARGET_PORT ) )
            if debug : print ( f"UDP source socket: { udp_sock.getsockname ()[ 0 ] }:{ udp_sock.getsockname ()[ 1 ] }. Sent ASCII_ENQ to { UDP_DEST_IP }:{ UDP_TARGET_PORT }" )
            i = Nof_WRTS
            while i :
                rx_samples.rx ( sdr_ctx = rx_pluto.pluto_rx_ctx )
                if wrt :
                    rx_samples.save_complex_samples_2_npf ( file_name = f"{payload_udp.decode("utf-8")}_{filename}" , dir_name = dir_name )
                i -= 1
            j -= 1

        elif payload_udp == ASCII_EOT : # Received END OF TRANSMISSION
            if debug : print ( f"Received ASCII_EOT {payload_udp=}, stopping transmission!" )
            if j > 0 :
                udp_sock.sendto ( ASCII_FF , ( UDP_DEST_IP , UDP_TARGET_PORT ) )
                if debug : print ( f"UDP source socket: { udp_sock.getsockname ()[ 0 ] }:{ udp_sock.getsockname ()[ 1 ] }. Sent ASCII_FF to { UDP_DEST_IP }:{ UDP_TARGET_PORT }" )
                rx_samples.rx ( sdr_ctx = rx_pluto.pluto_rx_ctx) # Wyczyszczenie bufora odbiorczego przed rozpoczęciem odbioru próbek, aby nie zapisać starych próbek z poprzedniej transmisji
            
        t.sleep ( 0.1 )  # odciążenie CPU
        rx_samples.rx ( sdr_ctx = rx_pluto.pluto_rx_ctx )
        payload_udp = b""
        print ( f"Waiting for next transmission... {j=}" )

finally :
    udp_sock.sendto ( ASCII_CAN , ( UDP_DEST_IP , UDP_TARGET_PORT ) )
    if debug : print ( f"Sent ASCII_CAN {ASCII_CAN} to { UDP_DEST_IP }:{ UDP_TARGET_PORT }" )
    udp_sock.close ()
    exit ( 0 )

        