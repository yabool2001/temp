
'''
Issue #8: Tworzenie dokumentacji dla różnych parametrów rrc_filter

Sekwencja uruchomienia skryptu:
cd ~/python/temp/
source .venv/bin/activate
python bpsk-tx_v0.1.5.py
'''

import curses # Moduł wbudowany w Python do obsługi terminala (obsługa klawiatury)
import numpy as np
import os
import time as t
import tomllib

#from pathlib import Path
from modules import filters , sdr , ops_packet , ops_file , modulation , monitor , corrections , plot

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )


tx_bpsk_packet_symbols = modulation.create_bpsk_symbols_v0_1_6 ( np.array ( settings["TEST01_BITS"] , dtype=np.uint8 ) )
tx_packet_upsampled = sdr.TxSamples ( tx_bpsk_packet_symbols )
if settings["log"]["verbose_1"] :
    plot.plot_bpsk_symbols_v2 ( tx_bpsk_packet_symbols , script_filename + f" {tx_bpsk_packet_symbols.size=}" )
    plot.complex_waveform ( tx_packet_upsampled.samples , script_filename + f" {tx_packet_upsampled.samples.size=}" , True )
    plot.spectrum_occupancy ( tx_packet_upsampled.samples , 1024 , script_filename + f" {tx_packet_upsampled.samples.size=}" )

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
            sdr.tx_once ( tx_packet_upsampled , pluto_tx )
        elif key == 'c' :
            t.sleep ( 1 ) # anty-dubler
            sdr.tx_cyclic ( tx_packet_upsampled , pluto_tx )
        elif key == 's' :
            t.sleep ( 1 ) # anty-dubler
            sdr.stop_tx_cyclic ( pluto_tx )
        t.sleep ( 0.05 )  # odciążenie CPU
finally :
    curses.echo ()
    stdscr.keypad ( False )
    curses.endwin ()