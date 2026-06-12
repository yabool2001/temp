'''
Skrypt do zapisywania do pliku sampli wygenerowanych przez skrypt python test134-tx_large_frames.py.

Sekwencja uruchomienia skryptu test134-tx_large_frames.py na zdalnej stacji fedora na Surface 9 Pro:

yabool2001@legion:~/python/temp$ ssh yabool2001@192.168.1.60
yabool2001@fedora:~$ cd python/temp/
yabool2001@fedora:~$ git pull
yabool2001@fedora:~$ python test134-tx_large_frames.py -10.0

Sekwencja uruchomienia skryptu test134-rx_large_frames.py lokalnie na ubuntu:
cd ~/python/temp/
source .venv/bin/activate
python test134-rx_large_frames.py -10.0

'''
from modules import ops_file, ops_os , packet, plot , sdr
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import os , sys , tomllib
import socket
import time as t
import tomllib

################
### SETTINGS ###

mode : str = 'training' # Available modes: 'training', 'test' or "inference"

dbg = True
plt = False
wrt = True
del_dst = True

lipkow_ap = True
single_machine = True
legion = True

Nof_ATTEMPTS = int ( 1 )
Nof_WRTS = int ( 3 )
frame_size : str = "S" # Available frame sizes: "S" - small, "M" - medium or "L" - large

################
################

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy
script_filename = os.path.basename ( __file__ )

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

filename = 'rx_samples.npy'
dst_dir = f"np.{mode}"
Path ( dst_dir ).mkdir ( parents = True , exist_ok = True )
if del_dst :
    for file_path in Path ( dst_dir ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

if mode in ('training', 'test', 'inference') :
    if dbg : print ( f"Running in {mode=}." )
else :
    raise ValueError ( f"Unknown {mode=}. Available modes: 'training', 'test' or 'inference'." )

if lipkow_ap :
    if single_machine :
        if legion :
            IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "Orange9D40" ][ "LEGION" ]
            IP_DST_ADDR = toml_settings[ "IP_V6_ADDR" ][ "Orange9D40" ][ "LEGION" ]
            INTERFACE = toml_settings["IF"][ "LEGION" ]
        else :
            IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "Orange9D40" ][ "SURFACE_PRO9" ]
            IP_DST_ADDR = toml_settings[ "IP_V6_ADDR" ][ "Orange9D40" ][ "SURFACE_PRO9" ]
            INTERFACE = toml_settings["IF"][ "SURFACE_PRO9" ]
    else :
        IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "Orange9D40" ][ "LEGION" ]
        IP_DST_ADDR = toml_settings[ "IP_V6_ADDR" ][ "Orange9D40" ][ "SURFACE_PRO9" ]
        INTERFACE = toml_settings["IF"][ "LEGION" ]
else :
    if single_machine :
        IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "S21_ULTRA" ][ "SURFACE_PRO9" ]
        IP_DST_ADDR = toml_settings[ "IP_V6_ADDR" ][ "S21_ULTRA" ][ "SURFACE_PRO9" ]
        INTERFACE = toml_settings["IF"][ "SURFACE_PRO9" ]
    else :
        IP_SRC_ADDR = toml_settings[ "IP_V6_ADDR" ][ "S21_ULTRA" ][ "SURFACE_PRO9" ]
        IP_DST_ADDR = toml_settings[ "IP_V6_ADDR" ][ "S21_ULTRA" ][ "SURFACE_GO3" ]
        INTERFACE = toml_settings["IF"][ "SURFACE_PRO9" ]

UDP_PORT = int ( toml_settings[ "UDP_PORT" ] )
ASCII_ENQ = b'\x05'  # Sygnał do rozpoczęcia transmisji danych
ASCII_EOT = b'\x04'  # Sygnał do zakończenia transmisji danych
#ASCII_FF = b'\x0c'  # Sygnał do rozpoczęcia pracy skryptu (Form Feed)
ASCII_CAN = b'\x18'  # Sygnał do zakończenia pracy skryptu
ASCII_S = b'\x53'  # Sygnał do rozpoczęcia odbioru próbek (printable char S)
ASCII_M = b'\x4d'  # Sygnał do rozpoczęcia odbioru próbek (printable char M)
ASCII_L = b'\x4c'  # Sygnał do rozpoczęcia odbioru próbek (printable char L)
match frame_size :
    case "S" :
        ASCII_FRAME_SIZE = ASCII_S
    case "M" :
        ASCII_FRAME_SIZE = ASCII_M
    case "L" :
        ASCII_FRAME_SIZE = ASCII_L
        
timestamp_min = int ( ops_os.milis_timestamp () ) - 1000 * 365 * 60 * 60 * 24 # -1Y
timestamp_max = int ( ops_os.milis_timestamp () ) + 1000 * 365 * 60 * 60 * 24 # +1Y

def resolve_interface_name ( preferred_interface : str ) -> str :
    available_interfaces = [ name for _ , name in socket.if_nameindex () ]
    if preferred_interface in available_interfaces :
        return preferred_interface
    for interface_name in available_interfaces :
        if interface_name != "lo" :
            if dbg : print ( f"Configured {preferred_interface=} not found, using {interface_name=} for IPv6 UDP." )
            return interface_name
    raise OSError ( "No non-loopback network interface available for IPv6 UDP" )

if len ( sys.argv ) > 1 :
    gain_control_mode_chan0 = str ( sys.argv[ 1 ] )
    rx_gain_chan0_int = int ( sys.argv[ 2 ] )
else :
    gain_control_mode_chan0 = str ( toml_settings[ "ADALM-Pluto" ][ "GAIN_CONTROL" ] )
    rx_gain_chan0_int = float ( toml_settings[ "ADALM-Pluto" ][ "RX_GAIN" ] )

rx_pluto = packet.RxPluto_v0_1_17 ( sn = sdr.PLUTO_RX_SN , gain_control_mode_chan0 = gain_control_mode_chan0 , rx_gain_chan0_int = rx_gain_chan0_int )
rx_samples = packet.RxSamples ()
if dbg : print ( f"\n{script_filename=} {rx_samples.samples_raw.size=}" )

# UDP socket setup
udp_sock = socket.socket ( socket.AF_INET6 , socket.SOCK_DGRAM )
INTERFACE = resolve_interface_name ( INTERFACE )
scope_id = socket.if_nametoindex ( INTERFACE )
udp_target_addr = ( IP_DST_ADDR , UDP_PORT , 0 , scope_id )

# Start UDP communication by sending the frame size to the tx side, so it knows to start transmitting samples of the specified frame size.
udp_sock.sendto ( ASCII_FRAME_SIZE , udp_target_addr )
if dbg : print ( f"UDP source socket: { udp_sock.getsockname ()[ 0 ] }:{ udp_sock.getsockname ()[ 1 ] }" )
if dbg : print ( f"Sent ASCII_FRAME_SIZE to { IP_DST_ADDR }:{ UDP_PORT }" )

# Clean the Pluto RX buffer before starting to receive samples, to avoid saving old samples from previous transmission.
rx_samples.rx ( sdr_ctx = rx_pluto.pluto_rx_ctx )

payload_udp = b""
try :
    j = Nof_ATTEMPTS
    while j :

        payload_udp = b""
        payload_udp , udp_sender_addr = udp_sock.recvfrom ( 20 )
        if dbg : print ( f"UDP Received {len ( payload_udp )=} byte(s): {payload_udp=}" )
        
        if len ( payload_udp ) >= 13 and int ( payload_udp ) > timestamp_min and int ( payload_udp ) < timestamp_max : # Received a valid timestamp to name the file with received samples
            if dbg : print ( f"Valid timestamp received: {payload_udp=}" )
            udp_sock.sendto ( ASCII_ENQ , udp_target_addr )
            if dbg : print ( f"UDP source socket: { udp_sock.getsockname ()[ 0 ] }:{ udp_sock.getsockname ()[ 1 ] }. Sent ASCII_ENQ to { IP_DST_ADDR }:{ UDP_PORT }" )
            i = Nof_WRTS
            while i :
                rx_samples.rx ( sdr_ctx = rx_pluto.pluto_rx_ctx )
                if wrt :
                    rx_samples.save_samples_2_npf ( file_name = f"{payload_udp.decode("utf-8")}_{filename}" , dir_name = dst_dir , add_timestamp = True )
                i -= 1
            j -= 1
        
        elif payload_udp == ASCII_EOT : # Received END OF TRANSMISSION
            if dbg : print ( f"Received ASCII_EOT {payload_udp=}, stopping transmission!" )
            if j > 0 :
                udp_sock.sendto ( ASCII_FRAME_SIZE , udp_target_addr )
                if dbg : print ( f"UDP source socket: { udp_sock.getsockname ()[ 0 ] }:{ udp_sock.getsockname ()[ 1 ] } Sent ASCII_FRAME_SIZE to { IP_DST_ADDR }:{ UDP_PORT }" )
                rx_samples.rx ( sdr_ctx = rx_pluto.pluto_rx_ctx) # Clean the Pluto RX buffer.
        
        t.sleep ( 0.1 )  # odciążenie CPU
        rx_samples.rx ( sdr_ctx = rx_pluto.pluto_rx_ctx )
        if dbg : print ( f"Waiting for next transmission... {j=}" )

finally :
    udp_sock.sendto ( ASCII_CAN , udp_target_addr )
    if dbg : print ( f"Sent ASCII_CAN {ASCII_CAN} to { IP_DST_ADDR }:{ UDP_PORT }" )
    udp_sock.close ()
    if plt :
        samples_files = sorted ( Path ( dst_dir ).glob ( "*_rx_samples_*.npy" ) )
        for samples_file in samples_files :
            samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( samples_file ) )
            plot.complex_waveform_v0_1_6 ( samples , f"{script_filename} {samples_file.name} {samples.size=}")
    exit ( 0 )