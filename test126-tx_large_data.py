
'''
Skrypt do generowania losowej transmisji dużych liczby danych przez ADALM-Pluto.
Odbiera komendę "ASCII ENQ" (0x05) przez UDP, która jest sygnałem do wysłania wcześniej przygotowanych danych testowych.
Komendę wysyła skrypt test125-save_series_raw_complex_samples.py po tym jak jest gotowy do odbioru danych.
Wysłane dane są zapisywane do pliku w katalogu ...... dla późniejszej analizy.
Po zakończeniu wysyłania danych skrypt wysyła komendę "ASCII EOT" (0x04), że zakończył wysyłanie danych.

Sekwencja uruchomienia skryptu w ubuntu:
cd ~/python/temp/
source .venv/bin/activate

python test126-tx_large_data.py -10.0

'''

import curses # Moduł wbudowany w Python do obsługi terminala (obsługa klawiatury)
import socket
import select
import numpy as np
import os
import sys
import time as t
import tomllib

from modules import ops_file, ops_os , packet , payload_test_data as ptd , sdr

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

if len ( sys.argv ) > 1 :
    tx_gain_float = float ( sys.argv[ 1 ] )
else :
    tx_gain_float = float ( toml_settings["ADALM-Pluto"][ "TX_GAIN" ] )

np.set_printoptions ( threshold = np.inf , linewidth = np.inf )

script_filename = os.path.basename ( __file__ )

debug = True
plt = False
wrt = True

UDP_DEST_IP = "192.168.1.50" # ubuntu
UDP_TARGET_PORT = 10001
ASCII_ENQ = b'\x05'  # Sygnał do rozpoczęcia transmisji danych
ASCII_EOT = b'\x04'  # Sygnał do zakończenia transmisji danych
ASCII_CAN = b'\x18'  # Sygnał do zakończenia pracy skryptu
MAX_SAMPLES_SIZE =  int ( toml_settings["ADALM-Pluto"][ "SAMPLES_BUFFER_SIZE" ] ) * 0.8 # Maksymalna liczba próbek do wysłania w jednej transmisji (80% bufora, aby zostawić miejsce na rozpędzenie się filtra)

tx_samples = []
tx_samples_4pluto = np.array ( [] , dtype = np.complex128 )

tx_pluto = packet.TxPluto_v0_1_17 ( sn = sdr.PLUTO_TX_SN, tx_gain_float = tx_gain_float )
print ( f"\n{ script_filename= } { tx_pluto= }" )

total_bytes_len = 0
payload_bytes = ptd.generate_payload_rand_up_2_1500b ()
total_bytes_len += len ( payload_bytes )
tx_samples = packet.TxSamples_v0_1_18 ( payload_bytes = payload_bytes )
print ( f"{tx_samples.samples4pluto.size=}, {len(payload_bytes)=}" )

while tx_samples.samples4pluto.size < MAX_SAMPLES_SIZE :
    payload_bytes = ptd.generate_payload_rand_up_2_1500b ()
    tx_samples.add_frame ( payload_bytes = payload_bytes )
timestamp = ops_os.milis_timestamp ()

if wrt :
    dir_name = "np.tensors"
    if debug : print ( f"Saving frames to flat tensor file in {dir_name} directory with timestamp {timestamp}..." )
    tx_samples.save_frames2flat_tensor ( filename = timestamp , dir_name = dir_name )

print ( f"Final payload bytes length: { total_bytes_len } bytes" )
for frame in tx_samples.frames :
    print ( f"{frame.bpsk_symbols.size=}: {frame.bpsk_symbols[ : 8 ]=}" )

if plt :
    tx_samples.plot_symbols ( f"{script_filename} {tx_samples.bytes.size=}" )
    tx_samples.plot_complex_samples4pluto ( f"{script_filename}" )
    tx_samples.plot_samples_spectrum ( f"{ script_filename } samples4pluto" )

# Setup UDP Socket
udp_sock = socket.socket ( socket.AF_INET , socket.SOCK_DGRAM )
# Automatyczne wykrywanie adresu IP z sieci 192.168.1.x
local_ip = "0.0.0.0" # Domyślny fallback
try:
    # Tworzymy tymczasowy socket żeby sprawdzić routing do sieci 192.168.1.x
    # Łączymy się z przykładowym adresem (np. bramą) w tej sieci aby system wskazał właściwy interfejs
    temp_sock = socket.socket ( socket.AF_INET , socket.SOCK_DGRAM )
    temp_sock.connect ( ( "192.168.1.1" , 1 ) )
    local_ip = temp_sock.getsockname ()[0]
    temp_sock.close()
except Exception:
    pass

print(f"Binding UDP to {local_ip}:10001")
udp_sock.bind ( ( local_ip , 10001 ) )
udp_sock.setblocking ( False )
print ( "Czekam na komendy na porcie UDP 10001" )
payload_udp = b""
udp_sender_addr = ( UDP_DEST_IP , UDP_TARGET_PORT )







# przekazanie timestampu do skryptu test125-save_series_raw_complex_samples.py, po otrzymaniu komendy ASCII_ENQ
while True :
    try :
        payload_udp , udp_sender_addr = udp_sock.recvfrom ( 1 )
    except BlockingIOError :
        t.sleep ( 0.05 )  # odciążenie CPU, gdy nie ma danych do odbioru
        continue
    if payload_udp == ASCII_ENQ : # ENQUIRY (START OF TRANSMISSION)
        if debug : print ( f"Received ASCII_ENQ {payload_udp=}, sending timestamp." )
        payload_udp = b""
        udp_sock.sendto ( timestamp.encode ( "utf-8" ) , udp_sender_addr )
        if debug : print ( f"Sent {timestamp=} to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )
        break
    t.sleep ( 0.05 )  # odciążenie CPU

try :
    while True :
        try :
            payload_udp , udp_sender_addr = udp_sock.recvfrom ( 1 )
            if debug : print ( f"\n\r[UDP] Received { len ( payload_udp ) } byte(s): {payload_udp=}" )
        except BlockingIOError :
            t.sleep ( 0.05 )  # odciążenie CPU, gdy nie ma danych do odbioru
            pass
        except Exception :
            pass
        if payload_udp == ASCII_CAN : # ESCAPE
            if debug : print ( f"Received ASCII_CAN {payload_udp=}, stopping transmission & ending script!" )
            break
        elif payload_udp == ASCII_ENQ : # ENQUIRY (START OF TRANSMISSION)
            if debug : print ( f"Received ASCII_ENQ {payload_udp=}, starting transmission." )
            tx_samples.tx ( sdr_ctx = tx_pluto.pluto_tx_ctx , repeat = 1 )
            if debug : print ( f"All {tx_samples.samples4pluto.size=} samples transmitted." )
            udp_sock.sendto ( ASCII_EOT , udp_sender_addr )
            if debug : print ( f"Sent ASCII_EOT to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )
        payload_udp = b""
        t.sleep ( 0.05 )  # odciążenie CPU
finally :
    udp_sock.close ()
