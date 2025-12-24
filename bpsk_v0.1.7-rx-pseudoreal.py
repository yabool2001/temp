
'''
Sekwencja uruchomienia skryptu:
cd ~/python/temp/
source .venv/bin/activate
python bpsk_v0.1.6-tx.py

Po zakończeniu tej wersji wrócić do rozwoju wersji bpsk_v0.1.6-rx
'''

import os
import time as t
import tomllib

#from pathlib import Path
from modules import ops_file , packet , sdr

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

filename_samples_1 = "logs/rx_samples_32768_1.npy"
filename_samples_2 = "logs/rx_samples_32768_2.npy"
filename_samples_3 = "logs/rx_samples_32768_3_1sample.npy"
filename_samples_9 = "logs/rx_samples_32768_9_empty.npy"
samples_1 = ops_file.open_samples_from_npf ( filename_samples_1 )
samples_2 = ops_file.open_samples_from_npf ( filename_samples_2 )
samples_3 = ops_file.open_samples_from_npf ( filename_samples_3 )
samples_9 = ops_file.open_samples_from_npf ( filename_samples_9 )

rx_samples = packet.RxSamples_v0_1_7 ( samples = samples_3 )

if rx_samples.sync_seguence_peaks is not None :
    print ( rx_samples.sync_seguence_peaks.size )
    rx_samples.plot_complex_waveform ( f"{script_filename}" , marker = False , peaks = True )
    for idx in rx_samples.sync_seguence_peaks :
        rx_frame = packet.RxFrame_v0_1_7 ( rx_samples.samples_filtered [ : idx ] )
        has_sync = rx_frame.has_sync_sequence ()
        print ( f"Frame ending at sample idx {idx}: has_sync_sequence = {has_sync}" )
else :
    print ( "No sync sequence peaks found" )


'''
tx_packet = packet.TxPacket ( payload = settings[ "PAYLOAD_4BYTES_DEC" ] )
tx_packet.plot_symbols ( tx_packet.packet_symbols , script_filename + " BPSK packet symbols" )
tx_packet.plot_waveform ( tx_packet.payload_samples , script_filename + " BPSK payload waveform samples" , True)
tx_packet.plot_waveform ( tx_packet.packet_samples , script_filename + " BPSK packet waveform samples" )
tx_packet.plot_spectrum ( tx_packet.packet_samples , script_filename + " BPSK packet spectrum occupancy" )
'''

