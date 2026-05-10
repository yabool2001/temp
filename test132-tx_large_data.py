
'''
Skrypt do generowania losowej transmisji dużych liczby danych przez ADALM-Pluto.
Odbiera komendę "ASCII ENQ" (0x05) przez UDP, która jest sygnałem do wysłania wcześniej przygotowanych danych testowych.
Komendę wysyła skrypt test125-rx_large_data.py po tym jak jest gotowy do odbioru danych.
Wysłane dane są zapisywane do pliku w katalogu ...... dla późniejszej analizy.
Po zakończeniu wysyłania danych skrypt wysyła komendę "ASCII EOT" (0x04), że zakończył wysyłanie danych.

Sekwencja uruchomienia skryptu w ubuntu:
cd ~/python/temp/
source .venv/bin/activate

python test126-tx_large_data.py -10.0

'''

import numpy as np , os , socket , sys , time as t , tomllib
from pathlib import Path
from modules import ops_os , packet , payload_test_data as ptd , sdr

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

dir_name = "np.tensors"
Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy
script_filename = os.path.basename ( __file__ )

if len ( sys.argv ) > 1 :
    tx_gain_float = float ( sys.argv[ 1 ] )
else :
    tx_gain_float = float ( toml_settings["ADALM-Pluto"][ "TX_GAIN" ] )

debug = True
plt = True
wrt = True
del_old = True
lipkow_ap = True
single_machine = True

if lipkow_ap :
    if single_machine :
        IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "Orange9D40" ][ "LEGION" ]
    else :
        IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "Orange9D40" ][ "LEGION" ]
else :
    if single_machine :
        IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "S21_ULTRA" ][ "SURFACE_PRO9" ]
    else :
        IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "S21_ULTRA" ][ "SURFACE_GO3" ]

UDP_PORT = int ( toml_settings[ "UDP_PORT" ] )
INTERFACE = toml_settings["IF"][ "LEGION" ]
ASCII_ENQ = b'\x05'  # Sygnał do rozpoczęcia transmisji danych
ASCII_EOT = b'\x04'  # Sygnał do zakończenia transmisji danych
ASCII_FF = b'\x0c'  # Sygnał do rozpoczęcia pracy skryptu (Form Feed)
ASCII_CAN = b'\x18'  # Sygnał do zakończenia pracy skryptu

SAMPLES_BUFFER_SIZE_MULTIPLICATOR = 2
SAMPLES_BUFFER_SIZE = int ( toml_settings["ADALM-Pluto"][ "SAMPLES_BUFFER_SIZE" ] )

def wrt_flat_tensor ( tx_samples : packet.TxSamples_v0_1_18 , timestamp_group : str ) -> None :
    tx_samples.save_frames2flat_tensor ( filename = f"{timestamp_group}_tx_symbols_flat_tensor" , dir_name = dir_name )
    #tx_samples.save_samples_2_flat_tensor ( filename = f"{timestamp_group}_tx_samples_flat_tensor" , dir_name = dir_name ) # wersja z rozbiegówką i wygaszeniem (41 sampli) jako  0+j0
    tx_samples.save_symbols_flat_tensor_wo_mute_2_pt ( file_name = f"{timestamp_group}_tx_samples_flat_tensor" , dir_name = dir_name ) # wrersja bez rozbiegówki i wygaszenia, czyli tylko aktywne próbki z ramek, bez 0+j0
    tx_samples.save_complex_samples4pluto_2_npf ( file_name = f"{timestamp_group}_tx_samples4pluto" , dir_name = dir_name , add_timestamp = False )
    if debug : print ( f"Frames' symbols and samples4pluto saved to flat tensor asd samples4pluto to npf file in {dir_name=} {timestamp_group=}..." )

def build_tx_samples_and_timestamp_group ( multiplicator : float = SAMPLES_BUFFER_SIZE_MULTIPLICATOR ) -> tuple [ packet.TxSamples_v0_1_18 , str ] :
    max_samples_size = int ( SAMPLES_BUFFER_SIZE * multiplicator ) # Maksymalna liczba próbek do wysłania w jednej transmisji (80% bufora, aby zostawić miejsce na rozpędzenie się filtra)
    tx_samples = packet.TxSamples_v0_1_18 ()
    ptd.fill_samples_up_to_max_length ( tx_samples = tx_samples , max_samples_size = max_samples_size )
    if debug : print ( f"{tx_samples.samples4pluto.size=}" )
    timestamp_group = ops_os.milis_timestamp ()
    if wrt : wrt_flat_tensor ( tx_samples = tx_samples , timestamp_group = timestamp_group )
    if debug :
        for frame in tx_samples.frames :
            print ( f"{frame.bpsk_symbols.size=}: {frame.bpsk_symbols[ : 8 ]=}" )
    return tx_samples , timestamp_group

if del_old :
    for file_path in Path ( dir_name ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

#tx_samples , timestamp_group = build_tx_samples_and_timestamp_group ( multiplicator = SAMPLES_BUFFER_SIZE_MULTIPLICATOR ) #usuunąć
tx_pluto = packet.TxPluto_v0_1_17 ( sn = sdr.PLUTO_TX_SN, tx_gain_float = tx_gain_float )
if debug : print ( f"\n{ script_filename= } { tx_pluto= }" )
tx_samples = None
timestamp_group = ""

udp_sock = socket.socket ( socket.AF_INET6 , socket.SOCK_DGRAM )
scope_id = socket.if_nametoindex ( INTERFACE )
# Bindowanie gniazda do naszego adresu IPv6 i portu 10001
udp_sock.bind ( ( IP_SRC_ADDR , UDP_PORT , 0 , scope_id ) )

payload_udp = b""
try :
    while True :

        payload_udp = b""
        payload_udp , udp_sender_addr = udp_sock.recvfrom ( 1 ) # Odbieramy dane z bufora UDP, oczekując na komendy od skryptu test132-rx_large_data.py
        if debug : print ( f"\n\r[UDP] Received { len ( payload_udp ) } byte(s): {payload_udp=}" )

        if payload_udp == ASCII_CAN : # ESCAPE - polecenie zamknięcia skryptu
            if debug : print ( f"Received ASCII_CAN {payload_udp=}, stopping transmission & ending script!" )
            payload_udp = b""
            break

        elif payload_udp == ASCII_ENQ : # ENQUIRY TO send A NEW PACKET
            if debug : print ( f"Received ASCII_ENQ {payload_udp=}, sending timestamp_group." )
            tx_samples.tx ( sdr_ctx = tx_pluto.pluto_tx_ctx , repeat = 1 )
            if debug : print ( f"All {tx_samples.samples4pluto.size=} samples transmitted." )
            tx_samples = None
            udp_sock.sendto ( ASCII_EOT , udp_sender_addr )
            if debug : print ( f"Sent ASCII_EOT to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )
            payload_udp = b""


        elif payload_udp == ASCII_FF : # ENQUIRY TO PREPARE A NEW PACKET AND timestamp_group and send timestamp_group
            if debug : print ( f"Received ASCII_FF {payload_udp=}, sending timestamp_group." )
            tx_samples , timestamp_group = build_tx_samples_and_timestamp_group ( multiplicator = SAMPLES_BUFFER_SIZE_MULTIPLICATOR )
            if plt :
                tx_samples.plot_symbols ( f"{script_filename} {tx_samples.bytes.size=}" )
                tx_samples.plot_complex_samples4pluto ( f"{script_filename}" )
                tx_samples.plot_samples_spectrum ( f"{ script_filename } samples4pluto" )
            udp_sock.sendto ( timestamp_group.encode ( "utf-8" ) , udp_sender_addr ) # Transmisja timestamp_groupu do skryptu test125, który go użyje do nazwania pliku z odebranymi próbkami
            if debug : print ( f"Sent {timestamp_group=} to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )

        t.sleep ( 0.05 )  # odciążenie CPU

finally :
    udp_sock.close ()
