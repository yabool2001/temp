
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

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy
if len ( sys.argv ) > 1 :
    tx_gain_float = float ( sys.argv[ 1 ] )
else :
    tx_gain_float = float ( toml_settings["ADALM-Pluto"][ "TX_GAIN" ] )

script_filename = os.path.basename ( __file__ )
dir_name = "np.tensors"

debug = True
plt = False
wrt = True
del_old = True

UDP_DEST_IP = "192.168.1.50" # ubuntu
UDP_TARGET_PORT = 10001
#ASCII_ENQ = b'\x05'  # Sygnał do rozpoczęcia transmisji danych
ASCII_EOT = b'\x04'  # Sygnał do zakończenia transmisji danych
ASCII_FF = b'\x0c'  # Sygnał do rozpoczęcia pracy skryptu (Form Feed)
ASCII_CAN = b'\x18'  # Sygnał do zakończenia pracy skryptu
SAMPLES_BUFFER_SIZE_MULTIPLICATOR = 0.8
SAMPLES_BUFFER_SIZE = int ( toml_settings["ADALM-Pluto"][ "SAMPLES_BUFFER_SIZE" ] )

def wrt_flat_tensor ( tx_samples : packet.TxSamples_v0_1_18 , timestamp : str ) -> None :
    if debug : print ( f"Saving frames to flat tensor file in {dir_name=} {timestamp=}..." )
    tx_samples.save_frames2flat_tensor ( filename = f"{timestamp}_tx_flat_tensor" , dir_name = dir_name )

def build_tx_samples_and_timestamp ( multiplicator : float = SAMPLES_BUFFER_SIZE_MULTIPLICATOR ) -> tuple [ packet.TxSamples_v0_1_18 , str ] :
    max_samples_size = int ( SAMPLES_BUFFER_SIZE * multiplicator ) # Maksymalna liczba próbek do wysłania w jednej transmisji (80% bufora, aby zostawić miejsce na rozpędzenie się filtra)
    tx_samples = packet.TxSamples_v0_1_18 ()
    ptd.fill_samples_up_to_max_length ( tx_samples = tx_samples , max_samples_size = max_samples_size )
    if debug : print ( f"{tx_samples.samples4pluto.size=}" )
    timestamp = ops_os.milis_timestamp ()
    if wrt : wrt_flat_tensor ( tx_samples = tx_samples , timestamp = timestamp )
    if debug :
        for frame in tx_samples.frames :
            print ( f"{frame.bpsk_symbols.size=}: {frame.bpsk_symbols[ : 8 ]=}" )
    return tx_samples , timestamp

if del_old :
    for file_path in Path ( dir_name ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

tx_pluto = packet.TxPluto_v0_1_17 ( sn = sdr.PLUTO_TX_SN, tx_gain_float = tx_gain_float )
if debug : print ( f"\n{ script_filename= } { tx_pluto= }" )
tx_samples = None
timestamp = ""

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
#udp_sock.setblocking ( False )
print ( "Czekam na komendy na porcie UDP 10001" )
udp_sender_addr = ( UDP_DEST_IP , UDP_TARGET_PORT )

payload_udp = b""
try :
    while True :

        payload_udp , udp_sender_addr = udp_sock.recvfrom ( 1 ) # Odbieramy dane z bufora UDP, oczekując na komendy od skryptu test125-save_series_raw_complex_samples.py
        if debug : print ( f"\n\r[UDP] Received { len ( payload_udp ) } byte(s): {payload_udp=}" )

        if payload_udp == ASCII_CAN : # ESCAPE - polecenie zamknięcia skryptu
            if debug : print ( f"Received ASCII_CAN {payload_udp=}, stopping transmission & ending script!" )
            payload_udp = b""
            break

        elif payload_udp == ASCII_FF : # ENQUIRY TO PREPARE and send A NEW PACKET AND SEND TIMESTAMP
            if debug : print ( f"Received ASCII_FF {payload_udp=}, sending timestamp." )
            tx_samples , timestamp = build_tx_samples_and_timestamp ( multiplicator = 1 )
            if plt :
                tx_samples.plot_symbols ( f"{script_filename} {tx_samples.bytes.size=}" )
                tx_samples.plot_complex_samples4pluto ( f"{script_filename}" )
                tx_samples.plot_samples_spectrum ( f"{ script_filename } samples4pluto" )
            udp_sock.sendto ( timestamp.encode ( "utf-8" ) , udp_sender_addr ) # Transmisja timestampu do skryptu test125, który go użyje do nazwania pliku z odebranymi próbkami
            if debug : print ( f"Sent {timestamp=} to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )
            tx_samples.tx ( sdr_ctx = tx_pluto.pluto_tx_ctx , repeat = 1 )
            if debug : print ( f"All {tx_samples.samples4pluto.size=} samples transmitted." )
            tx_samples = None
            udp_sock.sendto ( ASCII_EOT , udp_sender_addr )
            if debug : print ( f"Sent ASCII_EOT to { udp_sender_addr[ 0 ] }:{ udp_sender_addr[ 1 ] }" )
            payload_udp = b""

        payload_udp = b""
        t.sleep ( 0.05 )  # odciążenie CPU

finally :
    udp_sock.close ()
