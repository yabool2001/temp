from modules import filters , ops_file , modulation , packet , plot
import numba
from numba import jit, prange
import numpy as np
from numpy.typing import NDArray
import os
from pathlib import Path
import time as t

#Path ( "logs" ).mkdir ( parents = True , exist_ok = True )
script_filename = os.path.basename ( __file__ )

plt = False

filename_samples = "logs/rx_samples_32768_3_1sample_clipped.npy"
filename_sync_sequence = "logs/sync_sequence_barker13_bpsk_clipped.npy"

samples  = ops_file.open_samples_from_npf ( filename_samples )
sync_sequence = ops_file.open_samples_from_npf ( filename_sync_sequence )

#t0 = t.perf_counter_ns ()

m = len ( samples )
n = len ( sync_sequence )
corr = np.correlate ( samples , np.flip ( sync_sequence.conj () ) , mode = 'valid' )
corr_abs = np.abs ( np.correlate ( samples , np.flip ( sync_sequence.conj () ) , mode = 'valid' ) )


peak_idx_corr = int ( np.argmax ( corr ) ) ; peak_val = float ( np.abs ( corr[ peak_idx_corr ] ) ) ; print ( f"corr: {peak_idx_corr=}, {peak_val=}" )
peak_idx_corr_abs = int ( np.argmax ( corr_abs ) ) ; peak_val = float ( corr_abs[ peak_idx_corr_abs ] ) ; print ( f"corr_abs: {peak_idx_corr_abs=}, {peak_val=}" )
# rolling window energy for received (efficient via cumulative_sample_power_sum)
samples_power = np.abs ( samples ) ** 2
cumulative_sample_power_sum = np.concatenate ( ( [ 0.0 ] , np.cumsum ( samples_power ) ) )
window_energy = cumulative_sample_power_sum[ n: ] - cumulative_sample_power_sum[ :-n ]

# normalized correlation: corr / (sqrt(E_window * E_template))
sync_sequence_energy = np.vdot ( sync_sequence , sync_sequence ) .real ; print (f"{sync_sequence_energy=}")  # = ||sync_sequence||² obliczenie energii sekwencji synchronizacji
norm_corr = np.abs(corr) / ( np.sqrt ( window_energy * sync_sequence_energy ) + 1e-12 )
peak_idx_norm_corr = int ( np.argmax ( norm_corr ) ) ; peak_val = float ( norm_corr[ peak_idx_norm_corr ] ) ; print ( f"norm_corr: {peak_idx_norm_corr=}, {peak_val=}" )
# UWAGA! UWAGA! Podobno powinno być tak:
norm_corr_best = (np.abs(corr) ** 2) / (window_energy * sync_sequence_energy + 1e-12)
peak_idx_norm_corr_best = int ( np.argmax ( norm_corr_best ) ) ; peak_val = float ( norm_corr_best[ peak_idx_norm_corr_best ] ) ; print ( f"norm_corr_best: {peak_idx_norm_corr_best=}, {peak_val=}" )
# lub tak
# POPRAWNA WERSJA – CA-PHD (Correlation with Amplitude and Phase Homomorphic Detection)
norm_corr_better = np.abs(corr) / (np.sqrt(window_energy * sync_sequence_energy) + 1e-12)
peak_idx_norm_corr_better = int ( np.argmax ( norm_corr_better ) ) ; peak_val = float ( norm_corr_better[ peak_idx_norm_corr_better ] ) ; print ( f"norm_corr_better: {peak_idx_norm_corr_better=}, {peak_val=}" )
#t1 = t.perf_counter_ns ()
#print ( f"Detekcja sekwencji synchronizacji tj. w filters.has_sync_sequence: {(t1 - t0)/1e3:.1f} µs ")
if plt :
    plot.complex_waveform_v0_1_6 ( samples , f"{script_filename} | {samples.size=}" , False )
    plot.complex_waveform_v0_1_6 ( samples , f"{script_filename} | {samples.size=}" , False , np.array([peak_idx_corr]) )
    plot.complex_waveform_v0_1_6 ( samples , f"{script_filename} | {samples.size=}" , False , np.array([peak_idx_corr_abs]) )
    plot.complex_waveform_v0_1_6 ( samples , f"{script_filename} | {samples.size=}" , False , np.array([peak_idx_norm_corr]) )
    plot.complex_waveform_v0_1_6 ( samples , f"{script_filename} | {samples.size=}" , False , np.array([peak_idx_norm_corr_best]) )
    plot.complex_waveform_v0_1_6 ( samples , f"{script_filename} | {samples.size=}" , False , np.array([peak_idx_norm_corr_better]) )
    plot.complex_waveform ( sync_sequence , f"{script_filename} | {sync_sequence.size=}" , False )
    plot.complex_waveform ( corr , f"{script_filename} | {corr.size=}" , False )
    plot.real_waveform ( corr_abs , f"{script_filename} | {corr_abs.size=}" , False )
    plot.real_waveform ( samples_power , f"{script_filename} | {samples_power.size=}" , False )
    plot.real_waveform ( cumulative_sample_power_sum , f"{script_filename} | {cumulative_sample_power_sum.size=}" , False )
    plot.real_waveform ( window_energy , f"{script_filename} | {window_energy.size=}" , False )
    plot.real_waveform ( norm_corr , f"{script_filename} | {norm_corr.size=}" , False )
    plot.real_waveform ( norm_corr_best , f"{script_filename} | {norm_corr_best.size=}" , False )
    plot.real_waveform ( norm_corr_better , f"{script_filename} | {norm_corr_better.size=}" , False )   