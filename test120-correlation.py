from modules import ops_file , plot
import numpy as np
from numpy.typing import NDArray
import os
from scipy.signal import find_peaks
import time as t
import csv

script_filename = os.path.basename ( __file__ )

plt = False

filename_sync_sequence = "correlation/test120_sync_sequence.npy"
filename_samples_1_npy = "correlation/test120_samples_1.npy"
filename_samples_2_npy = "correlation/test120_samples_2.npy"
filename_samples_3_npy = "correlation/test120_samples_3.npy"
filename_samples_4_npy = "correlation/test120_samples_4.npy"
filename_samples_2_1_3_4_npy = "correlation/test120_samples_2_1_3_4.npy"


sync_sequence = ops_file.open_real_float64_samples_from_npf ( filename_sync_sequence )
samples_1 = ops_file.open_real_float64_samples_from_npf ( filename_samples_1_npy )
samples_2  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2_npy )
samples_3  = ops_file.open_real_float64_samples_from_npf ( filename_samples_3_npy )
samples_4  = ops_file.open_real_float64_samples_from_npf ( filename_samples_4_npy )
samples_2_1_3_4  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2_1_3_4_npy )

if plt :
    plot.real_waveform_v0_1_6 ( sync_sequence , f"{sync_sequence.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_1 , f"{samples_1.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_2 , f"{samples_2.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_3 , f"{samples_3.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_4 , f"{samples_4.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_2_1_3_4 , f"{samples_2_1_3_4.size=}" , False )

scenarios = [
    { "name" : "samples_1" , "desc" : "identicall" , "samples" : samples_1 , "sync_sequence" : sync_sequence } ,
    { "name" : "samples_2" , "desc" : "amplitude x 10" , "samples" : samples_2 , "sync_sequence" : sync_sequence } ,
    { "name" : "samples_3" , "desc" : "1 position different" , "samples" : samples_3 , "sync_sequence" : sync_sequence } ,
    { "name" : "samples_4" , "desc" : "amplitude x 10 and 1 position different" , "samples" : samples_4 , "sync_sequence" : sync_sequence } ,
    { "name" : "samples_2_1_3_4" , "desc" : "combined samples 2,1,3,4" , "samples" : samples_2_1_3_4 , "sync_sequence" : sync_sequence }
]

def my_correlation ( scenario : dict ) -> None :

    corr_2_amp_min_ratio = 12.0
    min_peak_height_ratio = 0.4  # Ten cudowanie pokazuje liczbę sampli na plot i chyba też dobrą w print liczbę bajtów!!!

    samples : NDArray[ np.float64 ] = scenario["samples"]
    sync_sequence : NDArray[ np.float64 ] = scenario["sync_sequence"]

    peaks = np.array ( [] ).astype ( np.uint32 )
    max_amplitude = np.max ( np.abs ( samples ) )

    #print ( f"{max_amplitude_real=} at {max_amplitude_real_idx=}, {max_amplitude_imag=} at {max_amplitude_imag_idx=}, {max_amplitude_abs=} at {max_amplitude_abs_idx=}" )
    #avg_amplitude = np.mean(np.abs(scenario['sample']))
    #percentile_95 = np.percentile(np.abs(scenario['sample']), 95)
    #rms_amplitude = np.sqrt(np.mean(np.abs(scenario['sample'])**2))

    corr = np.correlate ( samples , sync_sequence , mode = "valid" )

    max_peak_val = np.max ( corr )
    print (f"{max_peak_val=}, {max_amplitude=}, {scenario['name']=}, {scenario['desc']=}")

    corr_2_amp = max_peak_val / max_amplitude

    peaks , _ = find_peaks ( corr , height = max_peak_val * min_peak_height_ratio )

    plot.real_waveform_v0_1_6 ( corr , f"corr {scenario['name']} {samples.size=}" , False , peaks )
    plot.real_waveform_v0_1_6 ( samples , f"samples corr {scenario['name']} {samples.size=}" , False , peaks )

for scenario in scenarios :
    has_sync = my_correlation ( scenario )

