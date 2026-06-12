
'''

Skrypt do generowania losowej transmisji dużych liczby danych przez ADALM-Pluto.
Skrypt generuje niepowtarzającą się liczbę bajtów w pakiecie w celu zachowania różnej wartości frame_header.
Odbiera komendę "ASCII ENQ" (0x05) przez UDP, która jest sygnałem do wysłania wcześniej przygotowanych danych testowych.
Komendę wysyła skrypt test134-rx_large_frames.py po tym jak jest gotowy do odbioru danych.
Wysłane dane są zapisywane do pliku w katalogu ...... dla późniejszej analizy.
Po zakończeniu wysyłania danych skrypt wysyła komendę "ASCII EOT" (0x04), że zakończył wysyłanie danych.

Skrypt zapisuje w lokalnym katalogu próbki do plików npy:
{timestamp_group}_tx_samples.npy - próbki tx samples po filtrowaniu z rozbiegówką i wygaszeniem
{timestamp_group}_tx_active_symbols.npy - aktywne próbki wyciągnięte ze składowej real sygnału samples_4_pluto (samples_4_pluto.real) i zaokrąglone do wartości -1+j0 i 1+j0. Wartości obejmują tylo aktywne symbole TX, zaczynające się od  (bez rozbiegówki i wygaszenia, bez 0+j0)

Sekwencja uruchomienia skryptu w ubuntu:
cd ~/python/temp/
source .venv/bin/activate
python test134-tx_large_frames.py -10.0

'''

import numpy as np , os , socket , sys , time as t , tomllib
from pathlib import Path
from modules import ops_os , packet , payload_test_data as ptd , sdr

################
### SETTINGS ###

mode : str = 'test' # Available modes: 'training', 'test' or "inference"

no_frames : int = 1 # Number of frames to transmit in one samples. WORKS ONLY FOR S and M frame sizes

dbg = True
plt = True
wrt = True
del_dst = True

lipkow_ap = True
single_machine = True
legion = True

################
################

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy
script_filename = os.path.basename ( __file__ )

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

dst_dir = f"np.{mode}"
Path ( dst_dir ).mkdir ( parents = True , exist_ok = True )
if del_dst :
    for file_path in Path ( dst_dir ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

if len ( sys.argv ) > 1 :
    tx_gain_float = float ( sys.argv[ 1 ] )
else :
    tx_gain_float = float ( toml_settings["ADALM-Pluto"][ "TX_GAIN" ] )

if mode in ('training', 'test', 'inference') :
    if dbg : print ( f"Running in {mode=}." )
else :
    raise ValueError ( f"Unknown {mode=}. Available modes: 'training', 'test' or 'inference'." )

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
ASCII_S = b'\x53'  # Sygnał do rozpoczęcia odbioru próbek (printable char S)
ASCII_M = b'\x4d'  # Sygnał do rozpoczęcia odbioru próbek (printable char M)
ASCII_L = b'\x4c'  # Sygnał do rozpoczęcia odbioru próbek (printable char L)
ASCII_FRAME_SIZE = b''

SAMPLES_BUFFER_SIZE_MULTIPLICATOR = 2
SAMPLES_BUFFER_SIZE = int ( toml_settings["ADALM-Pluto"][ "SAMPLES_BUFFER_SIZE" ] )

def wrt_flat_tensor ( tx_samples : packet.TxSamples , timestamp_group : str ) -> None :
    tx_samples.save_symbols_from_samples_2_npf ( file_name = f"{timestamp_group}_tx_symbols_from_samples" , dir_name = dst_dir , add_timestamp = False )
    tx_samples.save_samples_2_npf ( file_name = f"{timestamp_group}_tx_samples" , dir_name = dst_dir , add_timestamp = False )
    if dbg : print ( f"Samples and corresponding symbols saved to npf type files in {dst_dir=} {timestamp_group=}..." )

def build_tx_samples_and_timestamp_group ( multiplicator : float = SAMPLES_BUFFER_SIZE_MULTIPLICATOR , frame_size : bytes = ASCII_L , no_frames : int = no_frames ) -> tuple [ packet.TxSamples , str ] :
    tx_samples = packet.TxSamples ()
    match frame_size :
        case b"S" :
            #payload_bytes = ptd.PAYLOAD_1500BYTES_DEC
            payload_bytes = [ 15 ]
            while no_frames :
                tx_samples.add_frame ( payload_bytes = payload_bytes )
                no_frames -= 1
        case b"M" :
            while no_frames :
                payload_bytes = ptd.generate_payload_rand_up_2_1500b ()
                tx_samples.add_frame ( payload_bytes = payload_bytes )
                no_frames -= 1
        case b"L" :
            max_samples_size = int ( SAMPLES_BUFFER_SIZE * multiplicator ) # Maksymalna liczba próbek do wysłania w jednej transmisji (80% bufora, aby zostawić miejsce na rozpędzenie się filtra)
            ptd.fill_samples_up_to_max_length ( tx_samples = tx_samples , max_samples_size = max_samples_size )
    if dbg : print ( f"{tx_samples.samples.size=}" )
    timestamp_group = ops_os.milis_timestamp ()
    if wrt : wrt_flat_tensor ( tx_samples = tx_samples , timestamp_group = timestamp_group )
    return tx_samples , timestamp_group

tx_pluto = packet.TxPluto_v0_1_17 ( sn = sdr.PLUTO_TX_SN, tx_gain_float = tx_gain_float )
if dbg : print ( f"\n{ script_filename= } { tx_pluto= }" )
tx_samples = None
timestamp_group = ""

udp_sock = socket.socket ( socket.AF_INET6 , socket.SOCK_DGRAM )
scope_id = socket.if_nametoindex ( INTERFACE )
# Bindowanie gniazda do naszego adresu IPv6 i portu 10001
udp_sock.bind ( ( IP_SRC_ADDR , UDP_PORT , 0 , scope_id ) )
print ( f"Czekam na komendy na IPv6 UDP {IP_SRC_ADDR}:{UDP_PORT}%{INTERFACE}" )

try :
    while True :

        payload_udp = b''
        payload_udp , udp_sender_addr = udp_sock.recvfrom ( 1 ) # Odbieramy dane z bufora UDP, oczekując na komendy od skryptu test132-rx_large_data.py
        if dbg : print ( f"\n\r[UDP] Received { len ( payload_udp ) } byte(s): {payload_udp=}" )

        if payload_udp == ASCII_CAN : # ESCAPE - polecenie zamknięcia skryptu
            if dbg : print ( f"Received ASCII_CAN {payload_udp=}, stopping transmission & ending script!" )
            payload_udp = b''
            break

        elif payload_udp == ASCII_ENQ : # ENQUIRY TO send A NEW PACKET
            if dbg : print ( f"Received ASCII_ENQ: {payload_udp=}, to send frame(s)." )
            tx_samples.tx ( sdr_ctx = tx_pluto.pluto_tx_ctx , repeat = 1 )
            if dbg : print ( f"All {tx_samples.samples.size=} samples transmitted." )
            tx_samples = None
            udp_sock.sendto ( ASCII_EOT , udp_sender_addr )
            if dbg : print ( f"Sent ASCII_EOT to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )
            payload_udp = b''

        elif payload_udp == ASCII_S or payload_udp == ASCII_M or payload_udp == ASCII_L : # ENQUIRY TO PREPARE A NEW PACKET AND timestamp_group and send timestamp_group
            ASCII_FRAME_SIZE = payload_udp
            if dbg : print ( f"Received ASCII_FRAME_SIZE: {payload_udp=}, sending timestamp_group." )
            tx_samples , timestamp_group = build_tx_samples_and_timestamp_group ( multiplicator = SAMPLES_BUFFER_SIZE_MULTIPLICATOR , frame_size = ASCII_FRAME_SIZE )
            if plt :
                tx_samples.plot_symbols_from_samples ( f"{script_filename} {timestamp_group}" )
                tx_samples.plot_samples ( f"{script_filename} {timestamp_group}" )
            udp_sock.sendto ( timestamp_group.encode ( "utf-8" ) , udp_sender_addr ) # Transmisja timestamp_groupu do skryptu test125, który go użyje do nazwania pliku z odebranymi próbkami
            if dbg : print ( f"Sent {timestamp_group=} to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )

        t.sleep ( 0.05 )  # odciążenie CPU

finally :
    udp_sock.close ()
