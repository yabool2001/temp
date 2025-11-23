
'''
Issue #5: Develop new real-tx to test all important filter's arguments
'''

import curses # Moduł wbudowany w Python do obsługi terminala (obsługa klawiatury)
import json
import numpy as np
import os
import time as t

from modules import filters , sdr , ops_packet , ops_file , modulation , monitor , corrections , plot

script_filename = os.path.basename ( __file__ )
# Wczytaj plik JSON z konfiguracją
with open ( "settings.json" , "r" ) as settings_file :
    settings = json.load ( settings_file )

### App settings ###
cuda = True
# monitor.show_spectrum_occupancy ( samples , nperseg = 1024 )

tx_packet_bits = ops_packet.create_packet_bits ( settings[ "PAYLOAD" ] )
if settings["log"]["verbose_1"] : plot.plot_bpsk_symbols_v2 ( tx_packet_bits , script_filename + f" {tx_packet_bits.size=}" )
tx_bpsk_packet_symbols = modulation.create_bpsk_symbols_v0_1_5 ( tx_packet_bits )
if settings["log"]["verbose_1"] : plot.plot_bpsk_symbols_v2 ( tx_bpsk_packet_symbols , script_filename + f" {tx_bpsk_packet_symbols.size=}" )
tx_packet_samples = filters.apply_tx_rrc_filter_v0_1_5 ( tx_bpsk_packet_symbols , False )
if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( tx_packet_samples , script_filename + f" {tx_packet_samples.size=}" )
tx_packet_upsampled = filters.apply_tx_rrc_filter_v0_1_5 ( tx_bpsk_packet_symbols , True )
if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( tx_packet_upsampled , script_filename + f" {tx_packet_upsampled.size=}" )
pluto_tx = sdr.init_pluto_v3 ( settings["ADALM-Pluto"]["URI"]["SN_TX"] )
print ( "Max scaled value:", np.max ( np.abs ( tx_packet_upsampled ) ) )

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
            pluto_tx.tx_destroy_buffer ()
            pluto_tx.tx_cyclic_buffer = True
            pluto_tx.tx ( tx_packet_samples )
            print ( "[c] TX CYCLIC started..." )
        elif key == 's' :
            t.sleep ( 1 ) # anty-dubler
            pluto_tx.tx_destroy_buffer ()
            pluto_tx.tx_cyclic_buffer = False
            print ( f"{pluto_tx.tx_cyclic_buffer=}" )
            print ( "[s] TX CYCLIC stopped" )
        t.sleep ( 0.05 )  # odciążenie CPU
finally :
    curses.echo ()
    stdscr.keypad ( False )
    curses.endwin ()