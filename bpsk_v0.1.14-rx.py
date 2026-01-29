
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

samples_filename = "np.samples/rx_samples_0.1.14_128Bx20_missed_last_frames.npy"
#samples_filename = "np.samples/rx_samples_0.1.15_no_samples.npy"
#samples_filename = "np.samples/rx_samples_0.1.14_1500B_01.npy"
#samples_filename = "np.samples/rx_samples_0.1.8_16_c_mode.npy"
        
wrt_filename_npy = "np.samples/rx_samples_last.npy"
wrt_filename_csv = "samples.csv/rx_samples_last.csv"
wrt_filename_log = "logs/rx_samples_last_log.csv"

with open ( wrt_filename_log , "w" ) as wrt_file :
    wrt_file.write ( "time,idx,has_sync_sequence,has_frame,has_packet\n" )
    wrt_file.write ( packet.log_packet )

received_bytes : NDArray[ np.uint8 ] = np.array ( [] , dtype = np.uint8 )
previous_samples_leftovers : NDArray[ np.complex128 ] = np.array ( [] , dtype = np.complex128 )

real = False
wrt = False

if real :
    rx_pluto = packet.RxPluto_v0_1_13 ( sn = sdr.PLUTO_RX_SN )
else :
    rx_pluto = packet.RxPluto_v0_1_13 ()

print ( f"\n{ script_filename= } receiving: {rx_pluto=} { rx_pluto.samples.samples.size= }" )
while ( len ( received_bytes ) < 10000 and real ) or ( not real and received_bytes.size == 0 ) :
    if real :
        rx_pluto.samples.rx ( previous_samples_leftovers = previous_samples_leftovers )
    else :
        rx_pluto.samples.rx ( samples_filename = samples_filename )
    if settings[ "log" ][ "debugging" ] :
        if rx_pluto.samples.has_amp_greater_than_ths : rx_pluto.samples.plot_complex_samples ( title = f"{ script_filename }" )
    rx_pluto.samples.detect_frames ()
    #print ( f"\n{ script_filename= } { rx_pluto.samples.samples.size= } { rx_pluto.samples.samples_filtered.size= }" )

    if rx_pluto.samples.has_amp_greater_than_ths :
        print ( f"{ script_filename } { rx_pluto.samples.has_amp_greater_than_ths= }" )
    
    if rx_pluto.samples.frames.has_leftovers :
        previous_samples_leftovers = rx_pluto.samples.samples_leftovers
        #print ( f"{rx_pluto.samples.samples_leftovers.size=}\n{rx_pluto.samples.frames.samples_leftovers_start_idx=}")

    if rx_pluto.samples.frames.sync_sequence_peaks.size > 0 :
        rx_pluto.samples.plot_complex_samples_filtered ( title = f"{ script_filename } {rx_pluto.samples.frames.sync_sequence_peaks.size=}" , peaks = rx_pluto.samples.frames.sync_sequence_peaks )
    #rx_pluto.samples.plot_complex_samples_filtered ( title = f"{ script_filename } {rx_pluto.samples.frames.sync_sequence_peaks.size=}" , peaks = rx_pluto.samples.frames.sync_sequence_peaks )

    if rx_pluto.samples.frames.samples_payloads_bytes.size > 0 :
        print ( f" { rx_pluto.samples.frames.samples_payloads_bytes[0]= }, { rx_pluto.samples.frames.samples_payloads_bytes.size= }" )
        received_bytes = np.concatenate ( [ received_bytes , rx_pluto.samples.frames.samples_payloads_bytes ] )
        print ( f"{ received_bytes.size= }" )
        if settings["log"]["debugging"] : rx_pluto.samples.analyze ()
        if wrt and real:
            rx_pluto.samples.save_complex_samples_2_npf ( wrt_filename_npy )
        if packet.log_packet != "" :
            with open ( wrt_filename_log , "a" ) as wrt_file :
                wrt_file.write ( packet.log_packet )
            packet.log_packet = ""

    #t.sleep ( 5 )

    if not real :
        break