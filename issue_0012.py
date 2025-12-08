
'''
'''
import numpy as np
import os
import time as t
import tomllib
from numpy.typing import NDArray

#from pathlib import Path
from modules import modulation , ops_file , packet , sdr

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )
filename = "logs/rx_samples_32768_3_1sample.npy"
samples : NDArray[np.complex128] = ops_file.open_samples_from_npf ( filename )
sync_sequence = modulation.get_barker13_bpsk_samples_v0_1_3 ( clipped = True )
rx_packets = packet.RxPackets ( samples = samples )

t0 = t.perf_counter_ns ()

m = len ( samples )
n = len ( sync_sequence )
if m < n or n == 0:
    print ( "Błąd: Za krótki ciąg samples < sync_sequence" )

# valid cross-correlation (tpl reversed) -> length m-n+1
# use complex conjugate on template for proper correlation with complex samples
corr = np.abs ( np.correlate ( samples , sync_sequence.conj ()[ : : -1 ] , mode = 'valid' ) )

t1 = t.perf_counter_ns ()
print ( f"has_sync perf: {( t1 - t0 ) / 1e3 : .3f} µs" )

#rx_packets.plot_waveform ( rx_packets.samples_filtered , script_filename + " BPSK packet waveform samples" )


'''
tx_packet = packet.TxPacket ( payload = settings[ "PAYLOAD_4BYTES_DEC" ] )
tx_packet.plot_symbols ( tx_packet.packet_symbols , script_filename + " BPSK packet symbols" )
tx_packet.plot_waveform ( tx_packet.payload_samples , script_filename + " BPSK payload waveform samples" , True)
tx_packet.plot_waveform ( tx_packet.packet_samples , script_filename + " BPSK packet waveform samples" )
tx_packet.plot_spectrum ( tx_packet.packet_samples , script_filename + " BPSK packet spectrum occupancy" )
'''

