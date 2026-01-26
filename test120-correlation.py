from modules import filters , ops_file , modulation , packet , plot , correlation
import numba
from numba import jit, prange
import numpy as np
from numpy.typing import NDArray
import os
from pathlib import Path
import time as t
import csv

#Path ( "logs" ).mkdir ( parents = True , exist_ok = True )
script_filename = os.path.basename ( __file__ )

plt = True

filename_sync_sequence = "correlation/test120_sync_sequence.npy"
filename_samples_1_npy = "correlation/test120_samples_1.npy"
filename_samples_2_npy = "correlation/test120_samples_2.npy"
filename_samples_3_npy = "correlation/test120_samples_3.npy"
filename_samples_4_npy = "correlation/test120_samples_4.npy"

sync_sequence = ops_file.open_real_float64_samples_from_npf ( filename_sync_sequence )
samples_1 = ops_file.open_real_float64_samples_from_npf ( filename_samples_1_npy )
samples_2  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2_npy )
samples_3  = ops_file.open_real_float64_samples_from_npf ( filename_samples_3_npy )
samples_4  = ops_file.open_real_float64_samples_from_npf ( filename_samples_4_npy )

if plt :
    plot.real_waveform_v0_1_6 ( sync_sequence , f"{sync_sequence.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_1 , f"{samples_1.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_2 , f"{samples_2.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_3 , f"{samples_3.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_4 , f"{samples_4.size=}" , True )

scenarios = [
    { "name" : "s3 corr" , "desc" : "s3_and_ss3 1 sample" , "sample" : samples_3_bpsk_1 , "sync_sequence" : sync_sequence_3 , "mode": "valid" } ,
    { "name" : "s3 corr" , "desc" : "s3_and_ss3 8 samples" , "sample" : samples_3_bpsk_2 , "sync_sequence" : sync_sequence_3 , "mode": "valid" }
]

conjugate = [ False , True ]
flip = [ False , True ]
magnitude_mode = [ False , True ]

for scenario in scenarios :
    has_sync = correlation.correlation_v8 ( scenario )
if has_sync is not None :
    print ( f"{has_sync=}" )

'''
#t0 = t.perf_counter_ns ()

m1 = len ( samples_1 )
m2 = len ( samples_2 )
n1 = len ( sync_sequence_1 )
n2 = len ( sync_sequence_2 )


corr_abs_1 = np.abs ( np.correlate ( samples_1 , sync_sequence_1 , mode = 'valid' ) )
peak_idx_corr_abs_1 = int ( np.argmax ( corr_abs_1 ) ) ; peak_val_1 = corr_abs_1[ peak_idx_corr_abs_1 ] ; print ( f"corr_abs_1: {peak_idx_corr_abs_1=}, {peak_val_1=}" )

corr_abs_1_fc = np.abs ( np.correlate ( samples_1 , np.flip ( sync_sequence_1.conj () ) , mode = 'valid' ) )
peak_idx_corr_abs_1_fc = int ( np.argmax ( corr_abs_1_fc ) ) ; peak_val_1_fc = corr_abs_1_fc[ peak_idx_corr_abs_1_fc ] ; print ( f"corr_abs_1_fc: {peak_idx_corr_abs_1_fc=}, {peak_val_1_fc=}" )



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
'''