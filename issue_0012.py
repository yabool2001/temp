
'''
'''
import numpy as np
import os
import time as t
import tomllib
from numpy.typing import NDArray

#from pathlib import Path
from modules import filters , modulation , ops_file , packet , plot

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )
filename = "logs/rx_samples_32768_3_1sample_clipped.npy"
samples : NDArray[np.complex128] = ops_file.open_samples_from_npf ( filename )
samples = filters.apply_rrc_rx_filter_v0_1_3 ( samples , False ) ; plot.complex_waveform ( samples , f"{script_filename} | {samples.size=}" , False )
sync_sequence = modulation.get_barker13_bpsk_samples_v0_1_3 ( clipped = True ) ; plot.complex_waveform ( sync_sequence , f"{script_filename} | {sync_sequence.size=}" , False )

t0 = t.perf_counter_ns ()

m = len ( samples )
n = len ( sync_sequence )
if m < n or n == 0:
    print ( "Błąd: Za krótki ciąg samples < sync_sequence" )
#sync_sequence_conj =  sync_sequence.conj () ; plot.complex_waveform ( sync_sequence_conj , f"{script_filename} | {sync_sequence_conj.size=}" , False )
#sync_sequence_fliped = np.flip ( sync_sequence ) ; plot.complex_waveform ( sync_sequence_fliped , f"{script_filename} | {sync_sequence_fliped.size=}" , False )
#sync_sequence_conj_fliped =  np.flip ( sync_sequence.conj () ) ; plot.complex_waveform ( sync_sequence_conj_fliped , f"{script_filename} | {sync_sequence_conj_fliped.size=}" , False )
corr = np.correlate ( samples , np.flip ( sync_sequence.conj () ) , mode = 'valid' ) ; plot.complex_waveform ( corr , f"{script_filename} | {corr.size=}" , False )
#correlate_abs = np.abs ( correlate ) ; plot.complex_waveform ( correlate_abs , f"{script_filename} | {correlate_abs.size=}" , False )
corr_abs = np.abs ( np.correlate ( samples , np.flip ( sync_sequence.conj () ) , mode = 'valid' ) ) ; plot.real_waveform ( corr_abs , f"{script_filename} | {corr_abs.size=}" , False )

# sync_sequence energy
sync_sequence_energy = np.sum ( np.abs ( sync_sequence ) ** 2 ); print ( f" {sync_sequence_energy.size=} {sync_sequence_energy=}" )

# 3. Używając np.flip() – trochę bardziej „oficjalne”
#sync_sequence_rev = np.flip ( sync_sequence )
#plot.complex_waveform ( sync_sequence_rev , script_filename + " sync_sequence.conj ()[ : : -1 ]" , False )
#plot.complex_waveform ( sync_sequence.conj ()[ : : -1 ] , script_filename + " sync_sequence.conj ()[ : : -1 ]" , False )

# valid cross-correlation (tpl reversed) -> length m-n+1
# use complex conjugate on template for proper correlation with complex samples
#sync_sequence =  sync_sequence.conj ()[ : : -1 ]
#corr = np.abs ( np.correlate ( samples , sync_sequence.conj ()[ : : -1 ] , mode = 'valid' ) )
#plot.complex_waveform ( sync_sequence , script_filename + " sync_sequence" , False )
#plot.complex_waveform ( sync_sequence.conj ()[ : : -1 ] , script_filename + " sync_sequence.conj ()[ : : -1 ]" , False )
t1 = t.perf_counter_ns ()
print ( f"has_sync perf: {( t1 - t0 ) / 1e3 : .3f} µs" )

#rx_packets.plot_waveform ( rx_packets.samples_filtered , script_filename + " BPSK packet waveform samples" )


'''
tx_packet = packet.TxPacket ( payload = settings[ "PAYLOAD_4BYTES_DEC" ] )
plot_symbols ( tx_packet.packet_symbols , script_filename + " BPSK packet symbols" )
'''

