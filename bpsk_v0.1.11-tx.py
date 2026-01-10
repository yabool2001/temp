
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
from modules import packet , payload_test_data as ptd , sdr

np.set_printoptions ( threshold = np.inf , linewidth = np.inf )

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )


plt = False
tx_pluto = packet.TxPluto_v0_1_11 ( sn = sdr.PLUTO_TX_SN)
print ( f"\n{ script_filename= } { tx_pluto= }" )
tx_pluto.samples.create_samples4pluto ( payload_bytes = settings[ "PAYLOAD_4BYTES_DEC" ] )
if plt :
    tx_pluto.samples.plot_symbols ( f"{script_filename} " )
    tx_pluto.samples.plot_complex_samples_filtered ( f"{script_filename} filtered samples" )
    tx_pluto.samples.plot_complex_samples4pluto ( f"{script_filename} samples4pluto" )
    tx_pluto.samples.plot_samples_spectrum ( f"{script_filename} samples4pluto" )

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
        if key ==  'o' :
            t.sleep ( 1 )  # anty-dubler
            tx_pluto.samples.tx ()
            print ( f"\n{tx_pluto.pluto_tx_ctx.tx_cyclic_buffer=}" )
            print ( f"[o] Sample sent!" )
        elif key == 'c' :
            t.sleep ( 1 ) # anty-dubler
            tx_pluto.samples.tx_cyclic ()
            print ( f"\n{tx_pluto.pluto_tx_ctx.tx_cyclic_buffer=}" )
            print ( f"[c] Tx cyclic started..." )
        elif key == 's' :
            t.sleep ( 1 ) # anty-dubler
            tx_pluto.samples.stop_tx_cyclic ()
            print ( f"\n{tx_pluto.pluto_tx_ctx.tx_cyclic_buffer=}" )
            print ( "[s] Tc cyclic stopped" )
        elif key > '0' and key < '9' : # advanced test mode
            t.sleep ( 1 )  # anty-dubler
            tx_pluto.samples.tx ( repeat = np.uint32 ( key ) )
            print ( f"Samples sent {key} time(s)." )
        elif key == '\x1b':  # ESCAPE
            break
        t.sleep ( 0.05 )  # odciążenie CPU
finally :
    curses.echo ()
    stdscr.keypad ( False )
    curses.endwin ()
