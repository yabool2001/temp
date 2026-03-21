
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

from modules import packet , payload_test_data as ptd , sdr

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
plt = True

UDP_DEST_IP = "192.168.1.50" # ubuntu
UDP_TARGET_PORT = 10001
ASCII_ENQ = b'\x05'  # Sygnał do rozpoczęcia transmisji danych
ASCII_EOT = b'\x04'  # Sygnał do zakończenia transmisji danych
ASCII_CAN = b'\x18'  # Sygnał do zakończenia pracy skryptu
MAX_SAMPLES_SIZE =  int ( toml_settings["ADALM-Pluto"][ "SAMPLES_BUFFER_SIZE" ] ) * 0.8 # Maksymalna liczba próbek do wysłania w jednej transmisji (80% bufora, aby zostawić miejsce na rozpędzenie się filtra)

tx_samples = []
all_tx_samples_size = 0

tx_pluto = packet.TxPluto_v0_1_17 ( sn = sdr.PLUTO_TX_SN, tx_gain_float = tx_gain_float )
print ( f"\n{ script_filename= } { tx_pluto= }" )

while all_tx_samples_size < MAX_SAMPLES_SIZE :
    tx_samples.append ( packet.TxSamples_v0_1_17 ( pluto_tx_ctx = tx_pluto.pluto_tx_ctx ) )
    tx_samples[ -1 ].create_samples4pluto ( payload_bytes = ptd.generate_payload_rand_up_2_1500b () )
    all_tx_samples_size += len ( tx_samples[-1].samples_filtered )
print ( f"\n{ script_filename= } { all_tx_samples_size= }" )

if plt :
    tx_samples[ 0 ].plot_symbols ( f"{ script_filename } " )
    tx_samples[ 0 ].plot_complex_samples_filtered ( f"{ script_filename } filtered samples" )
    tx_samples[ 0 ].plot_complex_samples4pluto ( f"{ script_filename } samples4pluto" )
    tx_samples[ 0 ].plot_samples_spectrum ( f"{ script_filename } samples4pluto" )

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

try :
    while True :
        try :
            payload_udp = udp_sock.recv ( 1 )
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
            for tx_sample in tx_samples :
                tx_sample.tx_cyclic ()
            if debug : print ( f"All {all_tx_samples_size} samples transmitted." )
            udp_sock.sendto ( ASCII_EOT , ( UDP_DEST_IP , UDP_TARGET_PORT ) )
            if debug : print ( f"Sent ASCII_EOT to { UDP_DEST_IP }:{ UDP_TARGET_PORT }" )
        payload_udp = b""
        t.sleep ( 0.05 )  # odciążenie CPU
finally :
    udp_sock.close ()
