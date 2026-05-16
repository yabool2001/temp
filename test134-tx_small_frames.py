
'''
Skrypt do generowania prostej ramki przez ADALM-Pluto.
Odbiera komendę "ASCII ENQ" (0x05) przez UDP, która jest sygnałem do wysłania wcześniej przygotowanych danych testowych.
Komendę wysyła skrypt test134-rx_small_frames.py po tym jak jest gotowy do odbioru danych.
Wysłane dane są zapisywane do pliku w katalogu "np.simple-frames" dla późniejszej analizy.
Po zakończeniu wysyłania danych skrypt wysyła komendę "ASCII EOT" (0x04), że zakończył wysyłanie danych.

Sekwencja uruchomienia skryptu w ubuntu:
cd ~/python/temp/
source .venv/bin/activate
python test134-tx_small_frames.py -10.0

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

dir_name = "np.tensors"
Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
np.set_printoptions ( threshold = 10 , edgeitems = 3 , linewidth = np.inf ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy
script_filename = os.path.basename ( __file__ )


dbg = True
plt = True
wrt = True
del_old = True

lipkow_ap = True
single_machine = True
legion = True

if lipkow_ap :
    if single_machine :
        if legion :
            IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "Orange9D40" ][ "LEGION" ]
            INTERFACE = toml_settings["IF"][ "LEGION" ]
        else :
            IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "Orange9D40" ][ "SURFACE_PRO9" ]
            INTERFACE = toml_settings["IF"][ "SURFACE_PRO9" ]
    else :
        IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "Orange9D40" ][ "SURFACE_PRO9" ]
        INTERFACE = toml_settings["IF"][ "SURFACE_PRO9" ]
else :
    if single_machine :
        IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "S21_ULTRA" ][ "SURFACE_PRO9" ]
        INTERFACE = toml_settings["IF"][ "SURFACE_PRO9" ]
    else :
        IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "S21_ULTRA" ][ "SURFACE_GO3" ]
        INTERFACE = toml_settings["IF"][ "SURFACE_GO3" ]

UDP_PORT = int ( toml_settings[ "UDP_PORT" ] )
ASCII_ENQ = b'\x05'  # Sygnał do rozpoczęcia transmisji danych
ASCII_EOT = b'\x04'  # Sygnał do zakończenia transmisji danych
ASCII_FF = b'\x0c'  # Sygnał do rozpoczęcia pracy skryptu (Form Feed)
ASCII_CAN = b'\x18'  # Sygnał do zakończenia pracy skryptu

MAX_SAMPLES_SIZE =  int ( toml_settings["ADALM-Pluto"][ "SAMPLES_BUFFER_SIZE" ] ) * 0.8 # Maksymalna liczba próbek do wysłania w jednej transmisji (80% bufora, aby zostawić miejsce na rozpędzenie się filtra)

if del_old :
    for file_path in Path ( dir_name ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

tx_pluto = packet.TxPluto_v0_1_17 ( sn = sdr.PLUTO_TX_SN, tx_gain_float = tx_gain_float )
print ( f"\n{ script_filename= } { tx_pluto= }" )

i = 1 # Liczba ramek
total_bytes_len = 0
tx_samples = packet.TxSamples ()

while i :
    #payload_bytes = ptd.PAYLOAD_4BYTES_DEC_15
    #payload_bytes = ptd.PAYLOAD_1500BYTES_DEC
    #payload_bytes = ptd.PAYLOAD_BYTES
    payload_bytes = [ 15 ]
    total_bytes_len += len ( payload_bytes )
    tx_samples.add_frame ( payload_bytes = payload_bytes )
    i -= 1
print ( f"{tx_samples.samples_4_pluto.size=}, {total_bytes_len=}" )
print ( f"{tx_samples.frames=}" )

timestamp = ops_os.milis_timestamp ()

if plt :
    tx_samples.plot_samples_4_pluto ( f"{script_filename}" )
    tx_samples.plot_active_symbols ( f"{script_filename}" )

#if wrt :
#    if dbg : print ( f"Saving frames to flat tensor file in {dir_name} directory with timestamp {timestamp}..." )
#    tx_samples.save_frames2flat_tensor ( filename = timestamp , dir_name = dir_name )

# Setup UDP Socket
udp_sock = socket.socket ( socket.AF_INET6 , socket.SOCK_DGRAM )
scope_id = socket.if_nametoindex ( INTERFACE )
udp_sock.bind ( ( IP_SRC_ADDR , UDP_PORT , 0 , scope_id ) )
#udp_sock.setblocking ( False )
print ( f"Czekam na komendy na IPv6 UDP {IP_SRC_ADDR}:{UDP_PORT}%{INTERFACE}" )
#udp_sender_addr = ( IP_DST_ADDR , UDP_PORT , 0 , scope_id )

payload_udp = b""
try :
    while True :
        payload_udp = b""
        payload_udp , udp_sender_addr = udp_sock.recvfrom ( 1 )
        if dbg : print ( f"\n\r[UDP] Received { len ( payload_udp ) } byte(s): {payload_udp=}" )
        
        if payload_udp == ASCII_CAN : # ESCAPE
            if dbg : print ( f"Received ASCII_CAN {payload_udp=}, stopping transmission & ending script!" )
            break
        
        elif payload_udp == ASCII_ENQ : # ENQUIRY (START OF TRANSMISSION)
            if dbg : print ( f"Received ASCII_ENQ {payload_udp=}, starting transmission." )
            tx_samples.tx ( sdr_ctx = tx_pluto.pluto_tx_ctx , repeat = 1 )
            if dbg : print ( f"All {tx_samples.samples_4_pluto.size=} samples transmitted." )
            #tx_samples = None # Zwolnienie pamięci zajmowanej przez próbki po transmisji, aby nie trzymać w pamięci dużej tablicy próbek, która już została wysłana
            udp_sock.sendto ( ASCII_EOT , udp_sender_addr )
            if dbg : print ( f"Sent ASCII_EOT to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )

        elif payload_udp == ASCII_FF : # FORM FEED (START OF SCRIPT)
            if dbg : print ( f"Received ASCII_FF {payload_udp=}, starting transmission." )
            udp_sock.sendto ( timestamp.encode ( "utf-8" ) , udp_sender_addr ) # Transmisja timestampu do skryptu test125, który go użyje do nazwania pliku z odebranymi próbkami
            if dbg : print ( f"Sent {timestamp=} to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )
            if wrt :
                tx_samples.save_samples_4_pluto_2_npf ( file_name = f"{timestamp}_tx_samples_4_pluto" , dir_name = dir_name , add_timestamp = False )
                tx_samples.save_samples_2_npf ( file_name = f"{timestamp}_tx_samples" , dir_name = dir_name , add_timestamp = False )
                tx_samples.save_active_symbols_2_npf ( file_name = f"{timestamp}_tx_active_symbols" , dir_name = dir_name , add_timestamp = False )
                tx_samples.plot_samples_4_pluto_spectrum ( f"{script_filename}" )
                if dbg : print ( f"Frames' symbols and samples_4_pluto saved to flat tensor asd samples_4_pluto to npf file in {dir_name=} {timestamp=}..." )
        t.sleep ( 0.05 )  # odciążenie CPU

finally :
    udp_sock.close ()
