
'''
Issue #5: Develop new real-tx to test all important filter's arguments

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


tx_packet_bits = ops_packet.create_packet_bits ( settings[ "PAYLOAD" ] )
if settings["log"]["verbose_1"] : plot.plot_bpsk_symbols_v2 ( tx_packet_bits , script_filename + f" {tx_packet_bits.size=}" )
tx_bpsk_packet_symbols = modulation.create_bpsk_symbols_v0_1_5 ( tx_packet_bits )
if settings["log"]["verbose_1"] : plot.plot_bpsk_symbols_v2 ( tx_bpsk_packet_symbols , script_filename + f" {tx_bpsk_packet_symbols.size=}" )
tx_packet_samples = filters.apply_tx_rrc_filter_v0_1_5 ( tx_bpsk_packet_symbols , False )
if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( tx_packet_samples , script_filename + f" {tx_packet_samples.size=}" )
tx_packet_upsampled = filters.apply_tx_rrc_filter_v0_1_5 ( tx_bpsk_packet_symbols , True )
if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( tx_packet_upsampled , script_filename + f" {tx_packet_upsampled.size=}" )
if settings["log"]["verbose_1"] : plot.spectrum_occupancy ( tx_packet_upsampled , 1024 , script_filename + f" {tx_packet_upsampled.size=}" )

pluto_tx = sdr.init_pluto_v3 ( settings["ADALM-Pluto"]["URI"]["SN_TX"] )
print ( f"{np.max ( np.abs ( tx_packet_upsampled ) )=}" )
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
            sdr.tx_once ( tx_packet_samples , pluto_tx )
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