
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

np.set_printoptions ( threshold = np.inf , linewidth = np.inf )

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

plt = False
tx_pluto = packet.TxPluto_v0_1_8 ()
print ( f"\n{ script_filename= } { tx_pluto }" )

payload256bytes = list ( range ( 256 ) )

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
            tx_pluto.tx ( mode = "once" , payload = settings[ "PAYLOAD_4BYTES_DEC" ] )
            print ( f"\n{tx_pluto.pluto_tx_ctx.tx_cyclic_buffer=}" )
            print ( f"\n{tx_pluto.tx_samples.samples_bytes=}" )
            print ( f"[o] Sample sent!" )
            if plt :
                tx_pluto.plot_symbols ( f"{script_filename} {tx_pluto.tx_samples.samples_bytes=}" )
                tx_pluto.plot_samples_waveform ( f"{script_filename} {tx_pluto.tx_samples.samples_bytes=}" , False )
                tx_pluto.plot_samples_spectrum ( f"{script_filename} {tx_pluto.tx_samples.samples_bytes=}" )
        elif key == 'c' :
            t.sleep ( 1 ) # anty-dubler
            tx_pluto.tx ( mode = "cyclic" , payload = settings[ "PAYLOAD_4BYTES_DEC" ] )
            print ( f"\n{tx_pluto.pluto_tx_ctx.tx_cyclic_buffer=}" )
            print ( f"[c] Tx cyclic started..." )
        elif key == 's' :
            t.sleep ( 1 ) # anty-dubler
            tx_pluto.stop_tx_cyclic ()
            print ( f"\n{tx_pluto.pluto_tx_ctx.tx_cyclic_buffer=}" )
            print ( "[s] Tc cyclic stopped" )
        elif key == 't' :
            t.sleep ( 1 )  # anty-dubler
            tx_pluto.tx ( mode = "cyclic" , payload = payload256bytes )
            print ( f"\n{tx_pluto.pluto_tx_ctx.tx_cyclic_buffer=}" )
            print ( f"\n{tx_pluto.tx_samples.samples_bytes=}" )
            print ( "[t] Tester mode send bytes." )
        elif key == '\x1b':  # ESCAPE
            break
        t.sleep ( 0.05 )  # odciążenie CPU
finally :
    curses.echo ()
    stdscr.keypad ( False )
    curses.endwin ()
