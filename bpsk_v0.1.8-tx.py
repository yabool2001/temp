
'''
Sekwencja uruchomienia skryptu:
cd ~/python/temp/
source .venv/bin/activate
python bpsk_v0.1.6-tx.py
'''

import curses # Moduł wbudowany w Python do obsługi terminala (obsługa klawiatury)
import numpy as np
import os
import time as t
import tomllib

#from pathlib import Path
from modules import packet , modulation , plot , sdr

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

tx_packet = packet.TxPacket_v0_1_8 ( payload = settings[ "PAYLOAD_4BYTES_DEC" ] )
print ( f"{ script_filename } utworzono pakiet: { tx_packet.payload_bytes= } { tx_packet.crc32_bytes } { tx_packet.packet_len= }" )
tx_frame = packet.TxFrame_v0_1_8 ( packet_len = tx_packet.packet_len )
print ( f"{ script_filename } utworzono ramkę: { tx_frame.sync_sequence_bits= } { tx_frame.packet_len_bits= } { tx_frame.frame_bits= }" )

'''
pluto_tx = sdr.init_pluto_v3 ( settings["ADALM-Pluto"]["URI"]["SN_TX"] )

stdscr = curses.initscr ()
curses.noecho ()
stdscr.keypad ( True )
print ( "Naciśnij:" )
print ( " - 't' aby wysłać pakiet jednorazowo" )
print ( " - 'c' aby rozpocząć transmisję cykliczną" )
print ( " - 's' aby zatrzymać transmisję cykliczną" )
try :
    while True :
        key = stdscr.getkey ()
        if key ==  't' :
            t.sleep ( 1 )  # anty-dubler
            sdr.tx_once_v0_1_6 ( tx_packet.packet_samples , pluto_tx )
        elif key == 'c' :
            t.sleep ( 1 ) # anty-dubler
            sdr.tx_cyclic_v0_1_6 ( tx_packet.packet_samples , pluto_tx )
        elif key == 's' :
            t.sleep ( 1 ) # anty-dubler
            sdr.stop_tx_cyclic ( pluto_tx )
        t.sleep ( 0.05 )  # odciążenie CPU
finally :
    curses.echo ()
    stdscr.keypad ( False )
    curses.endwin ()
'''