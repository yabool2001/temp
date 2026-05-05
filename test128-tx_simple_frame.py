
'''
Skrypt do generowania prostej ramki przez ADALM-Pluto.
Odbiera komendę "ASCII ENQ" (0x05) przez UDP, która jest sygnałem do wysłania wcześniej przygotowanych danych testowych.
Komendę wysyła skrypt test128-rx_and_save_simple_frame.py po tym jak jest gotowy do odbioru danych.
Wysłane dane są zapisywane do pliku w katalogu "np.simple-frames" dla późniejszej analizy.
Po zakończeniu wysyłania danych skrypt wysyła komendę "ASCII EOT" (0x04), że zakończył wysyłanie danych.

Sekwencja uruchomienia skryptu w ubuntu:
cd ~/python/temp/
source .venv/bin/activate
python test128-tx_simple_frame.py -10.0

'''

import socket, numpy as np , os , sys , time as t , tomllib
from modules import ops_os, packet , payload_test_data as ptd , sdr
from pathlib import Path

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

if len ( sys.argv ) > 1 :
    tx_gain_float = float ( sys.argv[ 1 ] )
else :
    tx_gain_float = float ( toml_settings["ADALM-Pluto"][ "TX_GAIN" ] )

np.set_printoptions ( threshold = 10 , edgeitems = 3 , linewidth = np.inf ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy
script_filename = os.path.basename ( __file__ )

debug = True
plt = True
wrt = False
del_old = True

UDP_DEST_IP = "192.168.1.50" # ubuntu
LOCAL_IP_v6_ADDR = "fe80::339e:6cea:f65b:ee40"
UDP_DEST_IP_V6 = "fe80::508d:aae1:d391:439a" # fedora Pro9
INTERFACE = 'wlp1s0'
UDP_TARGET_PORT = 10001
ASCII_FF = b'\x0c'  # Sygnał do rozpoczęcia pracy skryptu (Form Feed)
ASCII_ENQ = b'\x05' # Sygnał do rozpoczęcia transmisji danych przez skrypt tx
ASCII_EOT = b'\x04'  # Sygnał do zakończenia transmisji danych
ASCII_CAN = b'\x18'  # Sygnał do zakończenia pracy skryptu
MAX_SAMPLES_SIZE =  int ( toml_settings["ADALM-Pluto"][ "SAMPLES_BUFFER_SIZE" ] ) * 0.8 # Maksymalna liczba próbek do wysłania w jednej transmisji (80% bufora, aby zostawić miejsce na rozpędzenie się filtra)

tx_samples = []
tx_samples_4pluto = np.array ( [] , dtype = np.complex128 )

dir_name = "np.tensors"

if del_old :
    for file_path in Path ( dir_name ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

tx_pluto = packet.TxPluto_v0_1_17 ( sn = sdr.PLUTO_TX_SN, tx_gain_float = tx_gain_float )
print ( f"\n{ script_filename= } { tx_pluto= }" )

i = 3 # Liczba ramek
total_bytes_len = 0
tx_samples = packet.TxSamples_v0_1_18 ()
while i :
    payload_bytes = ptd.PAYLOAD_4BYTES_DEC_15
    #payload_bytes = ptd.PAYLOAD_1500BYTES_DEC
    total_bytes_len += len ( payload_bytes )
    tx_samples.add_frame ( payload_bytes = payload_bytes )
    i -= 1
print ( f"{tx_samples.samples4pluto.size=}, {total_bytes_len=}" )
print ( f"{tx_samples.frames=}" )

timestamp = ops_os.milis_timestamp ()

if plt :
    tx_samples.plot_complex_samples4pluto ( f"{script_filename}" , marker_peaks = True )

#if wrt :
#    if debug : print ( f"Saving frames to flat tensor file in {dir_name} directory with timestamp {timestamp}..." )
#    tx_samples.save_frames2flat_tensor ( filename = timestamp , dir_name = dir_name )

# Setup UDP Socket
udp_sock = socket.socket ( socket.AF_INET6 , socket.SOCK_DGRAM )
scope_id = socket.if_nametoindex ( INTERFACE )
udp_sock.bind ( ( LOCAL_IP_v6_ADDR , UDP_TARGET_PORT , 0 , scope_id ) )
#udp_sock.setblocking ( False )
print ( f"Czekam na komendy na IPv6 UDP {LOCAL_IP_v6_ADDR}:{UDP_TARGET_PORT}%{INTERFACE}" )
udp_sender_addr = ( UDP_DEST_IP_V6 , UDP_TARGET_PORT , 0 , scope_id )

payload_udp = b""
try :
    while True :
        payload_udp = b""
        payload_udp , udp_sender_addr = udp_sock.recvfrom ( 1 )
        if debug : print ( f"\n\r[UDP] Received { len ( payload_udp ) } byte(s): {payload_udp=}" )
        
        if payload_udp == ASCII_CAN : # ESCAPE
            if debug : print ( f"Received ASCII_CAN {payload_udp=}, stopping transmission & ending script!" )
            break
        
        elif payload_udp == ASCII_ENQ : # ENQUIRY (START OF TRANSMISSION)
            if debug : print ( f"Received ASCII_ENQ {payload_udp=}, starting transmission." )
            tx_samples.tx ( sdr_ctx = tx_pluto.pluto_tx_ctx , repeat = 1 )
            if debug : print ( f"All {tx_samples.samples4pluto.size=} samples transmitted." )
            #tx_samples = None # Zwolnienie pamięci zajmowanej przez próbki po transmisji, aby nie trzymać w pamięci dużej tablicy próbek, która już została wysłana
            udp_sock.sendto ( ASCII_EOT , udp_sender_addr )
            if debug : print ( f"Sent ASCII_EOT to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )

        elif payload_udp == ASCII_FF : # FORM FEED (START OF SCRIPT)
            if debug : print ( f"Received ASCII_FF {payload_udp=}, starting transmission." )
            udp_sock.sendto ( timestamp.encode ( "utf-8" ) , udp_sender_addr ) # Transmisja timestampu do skryptu test125, który go użyje do nazwania pliku z odebranymi próbkami
            if debug : print ( f"Sent {timestamp=} to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )
            if wrt :
                tx_samples.save_frames2flat_tensor ( filename = f"{timestamp}_tx_symbols_flat_tensor" , dir_name = dir_name )
                tx_samples.save_samples_2_flat_tensor ( filename = f"{timestamp}_tx_samples_flat_tensor" , dir_name = dir_name )
                tx_samples.save_complex_samples4pluto_2_npf ( file_name = f"{timestamp}_tx_samples4pluto" , dir_name = dir_name , add_timestamp = False )
                if debug : print ( f"Frames' symbols and samples4pluto saved to flat tensor asd samples4pluto to npf file in {dir_name=} {timestamp=}..." )
        
        t.sleep ( 0.05 )  # odciążenie CPU

finally :
    udp_sock.close ()
