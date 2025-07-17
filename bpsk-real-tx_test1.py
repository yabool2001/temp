
import adi
import iio
import json
import keyboard
import numpy as np
import os
import time as t

from modules import filters , sdr , ops_packet , ops_file , modulation , monitor , corrections , plot
#from modules.rrc import rrc_filter
#from modules.clock_sync import polyphase_clock_sync

data = np.loadtxt ( "logs/tx_samples.csv" , delimiter = ',' , skiprows = 1 )
samples = data[ : , 0 ] + 1j * data[ : , 1 ]
samples = samples.astype ( np.complex128 )
samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

script_filename = os.path.basename ( __file__ )

# ------------------------ PARAMETRY KONFIGURACJI ------------------------
F_C = int ( 2900000000 )    # częstotliwość nośna [Hz]
BW  = int ( 1000000 )        # szerokość pasma [Hz]
#F_S = 521100     # częstotliwość próbkowania [Hz] >= 521e3 && <
F_S = int ( 3000000 )
SPS = int ( 4 )                # próbek na symbol
TX_GAIN = float ( -10.0 )
URI = "usb:"


RRC_BETA = float ( 0.35 )    # roll-off factor
RRC_SPAN = int ( 11 )    # długość filtru RRC w symbolach

PAYLOAD = [ 0x0F , 0x0F , 0x0F , 0x0F ]  # można zmieniać dynamicznie
print (f"{F_C=} {F_S=} {BW=} {SPS=} {RRC_BETA=} {RRC_SPAN=}")

contexts = iio.scan_contexts ()
usb_match = None
for uri, description in contexts.items():
    if "1044739a470b000a090018007ecf7f5ea8" in description:
        uri_tx = uri
sdr = adi.Pluto ( uri_tx )
sdr.tx_lo = int ( 2900000000 )
sdr.rx_lo = int ( 2900000000 )
sdr.sample_rate = int ( 3000000 )
sdr.rx_rf_bandwidth = int ( 1000000 )
sdr.rx_buffer_size = int ( 32768 )
sdr.tx_hardwaregain_chan0 = float ( -10.0 )
sdr.gain_control_mode_chan0 = "slow_attack"
sdr.rx_hardwaregain_chan0 = float ( 50.0 )
sdr.rx_output_type = "SI"
sdr.tx_destroy_buffer ()
sdr.tx_cyclic_buffer = False
t.sleep ( 0.2 ) #delay after setting device parameters

sdr.tx_destroy_buffer ()
sdr.tx_cyclic_buffer = False
print ( f"{sdr.tx_cyclic_buffer=}" )
print ( "Max scaled value:", np.max ( np.abs ( samples ) ) )


print ( "Naciśnij:" )
print ( " - 't' aby wysłać pakiet jednorazowo" )
print ( " - 'c' aby rozpocząć transmisję cykliczną" )
print ( " - 's' aby zatrzymać transmisję cykliczną" )
while True :
    
    if keyboard.is_pressed ( "t" ) :
        t.sleep ( 1 )  # anty-dubler
        sdr.tx_destroy_buffer () # Dodałem to ale nie wiem czy to jest potrzebne
        sdr.tx_cyclic_buffer = False
        sdr.tx ( samples )
        print ( f"Samples sent!" )
        
    
    elif keyboard.is_pressed ( "c" ) :
        t.sleep ( 1 ) # anty-dubler
        sdr.tx_destroy_buffer ()
        sdr.tx_cyclic_buffer = True
        sdr.tx ( samples )
        print ( "[c] TX CYCLIC started..." )
        

    elif keyboard.is_pressed ( "s" ) :
        t.sleep ( 1 ) # anty-dubler
        sdr.tx_destroy_buffer ()
        sdr.tx_cyclic_buffer = False
        print ( f"{sdr.tx_cyclic_buffer=}" )
        print ( "[s] TX CYCLIC stopped" )

    t.sleep ( 0.05 )  # odciążenie CPU
