from modules import ops_file , plot
import numpy as np
from numpy.typing import NDArray
import os
from scipy.signal import find_peaks
import time as t
import csv

script_filename = os.path.basename ( __file__ )

plt = True

filename_sync_sequence = "np.samples/barker13_samples_clipped.npy"
filename_samples_1_npy = "np.samples/rx_samples_0.1.14_128Bx20_missed_last_frames.npy"
filename_samples_2_npy = "np.samples/rx_samples_0.1.15_no_samples.npy"

sync_sequence = ops_file.open_samples_from_npf ( filename_sync_sequence )
samples_frames = ops_file.open_samples_from_npf ( filename_samples_1_npy )
samples_noise  = ops_file.open_samples_from_npf ( filename_samples_2_npy )


#if plt :
    #plot.complex_waveform_v0_1_6 ( sync_sequence , f"{sync_sequence.size=}" , False )
    #plot.complex_waveform_v0_1_6 ( samples_frames , f"{samples_frames.size=}" , False )
    #plot.complex_waveform_v0_1_6 ( samples_noise , f"{samples_noise.size=}" , False )

scenarios = [
            { "name" : "samples_frames" , "desc" : "20 frames" , "samples" : samples_frames , "sync_sequence" : sync_sequence } ,
            { "name" : "samples_noise" , "desc" : "only noise" , "samples" : samples_noise , "sync_sequence" : sync_sequence }
            ]

def my_correlation ( scenario : dict ) -> None :

    corr_2_amp_min_ratio = 12.0
    min_peak_height_ratio = 0.8

    sync_sequence : NDArray[ np.float64 ] = scenario["sync_sequence"]
    samples : NDArray[ np.float64 ] = scenario["samples"]

    peaks = np.array ( [] ).astype ( np.uint32 )

    # W BPSK Q=0 teoretycznie, ale jeśli plik 'sync_sequence' jest typu complex,
    # musimy użyć sprzężenia (conj), inaczej korelacja będzie błędna. To bezpieczne.
    # Różnica w czasie jest pomijalna a więc zostawię conjugate
    corr = np.abs ( np.correlate ( samples , np.conj(sync_sequence) , mode = "valid" ) )
    #corr = np.abs ( np.correlate ( samples , sync_sequence , mode = "valid" ) )

    ones = np.ones ( len ( sync_sequence ) )
    # Fix: Use abs(samples)**2 for calculating energy of complex signal
    local_energy = np.correlate ( np.abs(samples)**2 , ones , mode = "valid" )
    sync_seq_norm = np.linalg.norm ( sync_sequence )
    local_signal_norm = np.sqrt ( np.maximum ( local_energy , 1e-10 ) )
    corr_norm = corr / ( local_signal_norm * sync_seq_norm )

    max_peak_val_normalized = np.max ( corr_norm )
    peaks , _ = find_peaks ( corr_norm , height = max_peak_val_normalized * min_peak_height_ratio )

    max_amplitude_real = np.max ( np.abs ( samples  ) )
    max_peak_real_val = np.max ( corr_norm )
    corr_2_amp = max_peak_real_val / max_amplitude_real
    #if corr_2_amp > corr_2_amp_min_ratio :

    if plt :
        plot.real_waveform_v0_1_6 ( corr_norm , f"corr normalized2 {scenario['name']} {peaks.size=} {corr_norm.size=}" , False , peaks )
        plot.complex_waveform_v0_1_6 ( samples , f"samples normalized {scenario['name']} {peaks.size=} {samples.size=}" , False , peaks )

#my_correlation ( scenarios[0] )
for scenario in scenarios :
    my_correlation ( scenario )