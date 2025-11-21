
import adi
import curses # Moduł wbudowany w Python do obsługi terminala (obsługa klawiatury)
import iio
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
real_rx = False # Pobieranie żywych danych z Pluto 
cuda = True

# monitor.show_spectrum_occupancy ( samples , nperseg = 1024 )

# ------------------------ PARAMETRY KONFIGURACJI ------------------------
F_C = int ( 2900000000 )    # częstotliwość nośna [Hz]
BW  = int ( 1000000 )       # szerokość pasma [Hz]
#F_S = 521100               # częstotliwość próbkowania [Hz] >= 521e3 && <
F_S = int ( 3000000 )
SPS = int ( 4 )                # próbek na symbol
TX_GAIN = float ( -10.0 )
URI = "usb:"


RRC_BETA = float ( 0.35 )    # roll-off factor
RRC_SPAN = int ( 11 )    # długość filtru RRC w symbolach


p = settings["PAYLOAD"] # [ 0x0F , 0x0F , 0x0F , 0x0F ]
print (f"{F_C=} {F_S=} {BW=} {SPS=} {RRC_BETA=} {RRC_SPAN=}")

contexts = iio.scan_contexts ()
usb_match = None
for uri, description in contexts.items():
    if "1044739a470b000a090018007ecf7f5ea8" in description:
        uri_tx = uri
tsdr = adi.Pluto ( uri_tx )
tsdr.tx_lo = int ( 2900000000 )
tsdr.rx_lo = int ( 2900000000 )
tsdr.sample_rate = int ( 3000000 )
tsdr.rx_rf_bandwidth = int ( 1000000 )
tsdr.rx_buffer_size = int ( 32768 )
tsdr.tx_hardwaregain_chan0 = float ( -10.0 )
tsdr.gain_control_mode_chan0 = "slow_attack"
tsdr.rx_hardwaregain_chan0 = float ( 50.0 )
tsdr.rx_output_type = "SI"
tsdr.tx_destroy_buffer ()
tsdr.tx_cyclic_buffer = False
t.sleep ( 0.2 ) #delay after setting device parameters

tsdr.tx_destroy_buffer ()
tsdr.tx_cyclic_buffer = False
print ( f"{tsdr.tx_cyclic_buffer=}" )
print ( "Max scaled value:", np.max ( np.abs ( samples ) ) )

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
            tsdr.tx_destroy_buffer () # Dodałem to ale nie wiem czy to jest potrzebne
            tsdr.tx_cyclic_buffer = False
            tsdr.tx ( samples )
            print ( f"Samples sent!" )
        elif key == 'c' :
            t.sleep ( 1 ) # anty-dubler
            tsdr.tx_destroy_buffer ()
            tsdr.tx_cyclic_buffer = True
            tsdr.tx ( samples )
            print ( "[c] TX CYCLIC started..." )
        elif key == 's' :
            t.sleep ( 1 ) # anty-dubler
            tsdr.tx_destroy_buffer ()
            tsdr.tx_cyclic_buffer = False
            print ( f"{tsdr.tx_cyclic_buffer=}" )
            print ( "[s] TX CYCLIC stopped" )
        t.sleep ( 0.05 )  # odciążenie CPU
finally :
    curses.echo ()
    stdscr.keypad ( False )
    curses.endwin ()