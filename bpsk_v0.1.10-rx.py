
'''
Sekwencja uruchomienia skryptu:
cd ~/python/temp/
source .venv/bin/activate
python bpsk_v0.1.8-tx.py

Po zakończeniu tej wersji wrócić do rozwoju wersji bpsk_v0.1.6-rx
'''
import numpy as np
from numpy.typing import NDArray
import os
import time as t
import tomllib
from pathlib import Path
from numpy import real

#from pathlib import Path
from modules import packet , sdr

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

Path ( "np.samples" ).mkdir ( parents = True , exist_ok = True )

#filename = "np.samples/rx_samples_0.1.7_02_32768.npy"
#filename = "np.samples/rx_samples_0.1.8_08_1s_sat.npy"
#filename = "np.samples/rx_samples_0.1.8_11_1s_sat.npy"
#filename = "np.samples/rx_samples_0.1.8_13_1s_sat.npy"
#filename = "np.samples/rx_samples_0.1.8_15_c_mode.npy"
filename = "np.samples/rx_samples_0.1.8_16_c_mode.npy"
#filename = "np.samples/rx_samples_0.1.8_17_c_mode_full.npy"

        
wrt_filename_npy = "np.samples/rx_samples_log.npy"
wrt_filename_csv = "samples.csv/rx_samples_log.csv"


received_bytes : NDArray[ np.uint8 ] = np.array ( [] , dtype = np.uint8 )
previous_samples_leftovers : NDArray[ np.complex128 ] = np.array ( [] , dtype = np.complex128 )

received_payloads = 0
samples_w_packet : np.uint32 = 0 

real = True

if real :
    rx_pluto = packet.RxPluto_v0_1_11 ( sn = sdr.PLUTO_RX_SN )
else :
    rx_pluto = packet.RxPluto_v0_1_11 ()

print ( f"\n{ script_filename= } receiving: {rx_pluto=} { rx_pluto.samples.samples.size= }" )
while len (received_bytes) < 10000 :
    if real :
        rx_pluto.samples.rx ( previous_samples_leftovers = previous_samples_leftovers )
    else :
        rx_pluto.samples.rx ( previous_samples_leftovers = previous_samples_leftovers , samples_filename = filename )
    if rx_pluto.samples.has_amp_greater_than_ths and settings["log"]["debugging"] : rx_pluto.samples.plot_complex_samples ( title = f"{script_filename}" )
    rx_pluto.samples.detect_frames ()
    #print ( f"\n{ script_filename= } { rx_pluto.samples.samples.size= } { rx_pluto.samples.samples_filtered.size= }" )
    
    if rx_pluto.samples.frames.has_leftovers :
        previous_samples_leftovers = rx_pluto.samples.samples_leftovers
        #print ( f"{rx_pluto.samples.samples_leftovers.size=}\n{rx_pluto.samples.frames.samples_leftovers_start_idx=}")

    if rx_pluto.samples.frames.sync_sequence_peaks.size > 0 :
        rx_pluto.samples.plot_complex_samples_filtered ( title = f"{script_filename}" , peaks = rx_pluto.samples.frames.sync_sequence_peaks )

    if rx_pluto.samples.frames.samples_payloads_bytes.size > 0 :
        samples_w_packet += 1
        print ( f"{samples_w_packet=}")
        print ( f" {rx_pluto.samples.frames.samples_payloads_bytes=}, {rx_pluto.samples.frames.samples_payloads_bytes.size=}" )
        received_bytes = np.concatenate ( [ received_bytes , rx_pluto.samples.frames.samples_payloads_bytes ] )
        print ( f"{received_bytes.size=}" )