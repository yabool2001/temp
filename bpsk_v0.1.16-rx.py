
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
import threading
import time as t
import tomllib
from pathlib import Path
from numpy import real

#from pathlib import Path
from modules import ops_file, packet , sdr

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

Path ( "np.samples" ).mkdir ( parents = True , exist_ok = True )

#samples_filename = "np.samples/rx_samples_0.1.15_no_samples.npy"
samples_filename = "np.samples/rx_samples_0.1.14_128Bx20.npy"
#samples_filename = "np.samples/rx_samples_0.1.15_1500B_d00x1.npy"
#samples_filename = "np.samples/rx_samples_0.1.15_1500B_d15x1.npy"
#samples_filename = "np.samples/rx_samples_0.1.15_1500B_d0-4.npy"
#samples_filename = "np.samples/rx_samples_0.1.15_1500B_d15x5.npy"

wrt_filename_npy = "np.samples/rx_samples_0.1.16.npy"
wrt_filename_csv = "samples.csv/rx_samples_last.csv"
wrt_filename_log = "logs/rx_perf_log.csv"

with open ( wrt_filename_log , "w" ) as wrt_file :
    wrt_file.write ( "time,log_name\n" )
    wrt_file.write ( packet.log_packet )

received_bytes : NDArray[ np.uint8 ] = np.array ( [] , dtype = np.uint8 )
previous_samples_leftovers : NDArray[ np.complex128 ] = np.array ( [] , dtype = np.complex128 )

real = True
debug = False
plt = True
wrt = False

if real :
    rx_pluto = packet.RxPluto_v0_1_16 ( sn = sdr.PLUTO_RX_SN )
else :
    rx_pluto = packet.RxPluto_v0_1_16 ()

print ( f"\n{ script_filename= } receiving: {rx_pluto=}" )

while ( len ( received_bytes ) < 100000 and real ) or ( not real and received_bytes.size == 0 ) :
    if real :
        rx_pluto_samples = packet.RxSamples_v0_1_16 ( pluto_rx_ctx = rx_pluto.pluto_rx_ctx )
        rx_pluto_samples.rx ( previous_samples_leftovers = previous_samples_leftovers )
    else :
        rx_pluto_samples.rx ( samples_filename = samples_filename )
        rx_pluto_samples = packet.RxSamples_v0_1_16 ()
    if debug :
        if rx_pluto_samples.has_amp_greater_than_ths : rx_pluto_samples.plot_complex_samples ( title = f"{ script_filename }" )
    rx_pluto_samples.detect_frames ( deep = False )
    #print ( f"\n{ script_filename= } { rx_pluto.samples.samples.size= } { rx_pluto.samples.samples_filtered.size= }" )

    if rx_pluto_samples.has_amp_greater_than_ths :
        if debug : print ( f"{ script_filename } { rx_pluto_samples.has_amp_greater_than_ths= }" )
    
    if rx_pluto_samples.frames.has_leftovers :
        previous_samples_leftovers = rx_pluto_samples.samples_leftovers
        #print ( f"{rx_pluto_samples.samples_leftovers.size=}\n{rx_pluto_samples.frames.samples_leftovers_start_idx=}")
    if rx_pluto_samples.frames.sync_sequence_peaks.size > 0 :
        if plt : rx_pluto_samples.plot_complex_samples_filtered ( title = f"{ script_filename } {rx_pluto_samples.frames.sync_sequence_peaks.size=}" , peaks = rx_pluto_samples.frames.sync_sequence_peaks )
    #rx_pluto_samples.plot_complex_samples_filtered ( title = f"{ script_filename } {rx_pluto_samples.frames.sync_sequence_peaks.size=}" , peaks = rx_pluto_samples.frames.sync_sequence_peaks )

    if rx_pluto_samples.frames.samples_payloads_bytes.size > 0 :
        received_bytes = np.concatenate ( [ received_bytes , rx_pluto_samples.frames.samples_payloads_bytes ] )
        print ( f"{ rx_pluto_samples.frames.samples_payloads_bytes[0]= }, { rx_pluto_samples.frames.samples_payloads_bytes.size= } { received_bytes.size= }" )
        if debug : rx_pluto_samples.analyze ()
        if wrt and real:
            rx_pluto_samples.save_complex_samples_2_npf ( wrt_filename_npy )
    
    if packet.log_packet != "" :
        # To jest najlepszy i najprostszy wybór dla aplikacji SDR działającej w pętli.
        # Pozwala na płynny odbiór próbek bez dławienia się przy zapisie na dysk.
        # Ryzyko, że "wątek zginie" przy zamykaniu programu jest minimalne w porównaniu do korzyści z płynności działania,
        # a systemowy bufor pliku i tak zazwyczaj zdąży się opróżnić.
        log_thread = threading.Thread ( target = ops_file.save_log_thread , args = ( wrt_filename_log , packet.log_packet ) , daemon = True  )
        log_thread.start ()
        packet.log_packet = ""

    if not real :
        break
