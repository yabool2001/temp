from modules import filters , ops_file , modulation , packet , plot
import numba
from numba import jit, prange
import numpy as np
from numpy.typing import NDArray
import os
from pathlib import Path
import time as t

Path ( "logs" ).mkdir ( parents = True , exist_ok = True )
script_filename = os.path.basename ( __file__ )

filename_samples_1k = "np.samples/complex_samples_1k.npy"
filename_samples_32768_2 = "logs/rx_samples_32768_2.npy" # caly przebieg zawiera pakiety
filename_samples_32768_3_1sample = "logs/rx_samples_32768_3_1sample.npy" # caly przebieg zawiera tylko 1 pakiet
filename_samples_32768_6 = "logs/rx_samples_32768_6.npy" # dwukrotny przegieg i caly zawiera pakiety

filename_sync_sequence = "np.samples/complex_sync_sqeuence_4_1k.npy"

#samples  = ops_file.open_samples_from_npf ( filename_samples_1k )
samples  = ops_file.open_samples_from_npf ( filename_samples_32768_3_1sample )
sync_sequence = ops_file.open_samples_from_npf ( filename_sync_sequence )

t0 = t.perf_counter_ns ()
filters.has_sync_sequence ( samples , modulation.get_barker13_bpsk_samples_v0_1_3 ( clipped = True ) )

plot.complex_waveform ( samples , f"{script_filename} | {samples.size=}" , False )
plot.complex_waveform ( sync_sequence , f"{script_filename} | {sync_sequence.size=}" , False )
m = len ( samples )
n = len ( sync_sequence )
corr = np.correlate ( samples , np.flip ( sync_sequence.conj () ) , mode = 'valid' ) ; plot.complex_waveform ( corr , f"{script_filename} | {corr.size=}" , False )
corr_abs = np.abs ( np.correlate ( samples , np.flip ( sync_sequence.conj () ) , mode = 'valid' ) ) ; plot.real_waveform ( corr_abs , f"{script_filename} | {corr_abs.size=}" , False )

sync_sequence_power = np.abs ( sync_sequence ) ** 2 ; plot.real_waveform ( sync_sequence_power , f"{script_filename} | {sync_sequence_power.size=}" , False )
sync_sequence_energy = np.sum ( sync_sequence_power ) ; print ( f" {sync_sequence_energy.size=} {sync_sequence_energy=}" )

# rolling window energy for received (efficient via cumsum)
samples_power = np.abs ( samples ) ** 2 ; plot.real_waveform ( samples_power , f"{script_filename} | {samples_power.size=}" , False )
cumsum = np.concatenate ( ( [ 0.0 ] , np.cumsum ( samples_power ) ) ) ; plot.real_waveform ( cumsum , f"{script_filename} | {cumsum.size=}" , False )
window_energy = cumsum[ n: ] - cumsum[ :-n ]

t1 = t.perf_counter_ns ()
print ( f"Detekcja sekwencji synchronizacji tj. w filters.has_sync_sequence: {(t1 - t0)/1e3:.1f} µs ")
