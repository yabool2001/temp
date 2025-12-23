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
#plt = False

filename_samples_1 = "correlation/samples_1.npy"
filename_samples_2 = "correlation/samples_2.npy"
filename_samples_2_noisy_1 = "correlation/samples_2_noisy_1.npy"
filename_samples_2_noisy_2 = "correlation/samples_2_noisy_2.npy"
filename_samples_2_noisy_3 = "correlation/samples_2_noisy_3.npy"
filename_samples_3_bpsk_1 = "logs/rx_samples_32768_3_1sample.npy"
filename_samples_3_bpsk_2 = "logs/rx_samples_32768_1.npy"
filename_samples_4_bpsk_1 = "logs/rx_samples_32768_2.npy"
filename_samples_5_bpsk_1 = "logs/rx_samples_32768_9_empty.npy"
filename_sync_sequence_1 = "correlation/sync_sequence_1.npy"
filename_sync_sequence_2 = "correlation/sync_sequence_2.npy"
filename_sync_sequence_3 = "logs/barker13_samples_clipped.npy"
filename_sync_sequence_3_noclipping = "logs/barker13_samples.npy"
filename_results_csv = "correlation/correlation_results.csv"

samples_1 = ops_file.open_real_float64_samples_from_npf ( filename_samples_1 )
samples_2  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2 )
samples_2_noisy_1  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2_noisy_1 )
samples_2_noisy_2  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2_noisy_2 )
samples_2_noisy_3  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2_noisy_3 )
samples_3_bpsk_1  = ops_file.open_samples_from_npf ( filename_samples_3_bpsk_1 )
samples_3_bpsk_2  = ops_file.open_samples_from_npf ( filename_samples_3_bpsk_2 )
samples_4_bpsk_1  = ops_file.open_samples_from_npf ( filename_samples_4_bpsk_1 )
samples_5_bpsk_1  = ops_file.open_samples_from_npf ( filename_samples_5_bpsk_1 )
sync_sequence_3_noclipping = ops_file.open_samples_from_npf ( filename_sync_sequence_3_noclipping )
sync_sequence_3 = ops_file.open_samples_from_npf ( filename_sync_sequence_3 )
sync_sequence_1 = ops_file.open_real_float64_samples_from_npf ( filename_sync_sequence_1 )
sync_sequence_2 = ops_file.open_real_float64_samples_from_npf ( filename_sync_sequence_2 )


######### samples muszą być filtrowane RRC przed korelacją! #########


#plot.complex_waveform_v0_1_6 ( sync_sequence_3 , f"{sync_sequence_3.size=}" , True )
#plot.complex_waveform_v0_1_6 ( samples_3_bpsk_1 , f"{samples_3_bpsk_1.size=}" , False )
#plot.complex_waveform_v0_1_6 ( samples_3_bpsk_2 , f"{samples_3_bpsk_2.size=}" , False )
#plot.real_waveform_v0_1_6 ( samples_1 , f"samples_1" , True )
#plot.real_waveform_v0_1_6 ( samples_2 , f"samples_2" , True )
#plot.real_waveform_v0_1_6 ( samples_2_noisy_1 , f"samples_2_noisy_1" , True )

scenarios_old = [
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,

    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "valid" , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "same"  , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "full"  , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "valid" , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "same"  , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "full"  , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,

    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "valid" , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "same"  , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "full"  , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "valid" , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "same"  , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,
    { "name" : "s1 corr" , "desc" : "samples_1 & sync_sequence_1" , "sample" : samples_1 , "sync_sequence" : sync_sequence_1 , "mode": "full"  , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,

    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,

    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,

    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2 & sync_sequence_2" , "sample" : samples_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,

    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,

    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,

    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_1 & sync_sequence_2" , "sample" : samples_2_noisy_1 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,
    
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,

    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,

    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_2 & sync_sequence_2" , "sample" : samples_2_noisy_2 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,

    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : False , "magnitude_mode" : True } ,
    
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : True , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : True , "magnitude_mode" : True }
    ]

samples_3_bpsk_1 = filters.apply_rrc_rx_filter_v0_1_6 ( samples_3_bpsk_1 )
samples_3_bpsk_2 = filters.apply_rrc_rx_filter_v0_1_6 ( samples_3_bpsk_2 )

scenarios_old2 = [
    { "name" : "s3 corr" , "desc" : "s3_and_ss3 1 sample" , "sample" : samples_3_bpsk_1 , "sync_sequence" : sync_sequence_3 , "mode": "valid" } ,
    { "name" : "s3 corr" , "desc" : "s3_and_ss3 8 samples" , "sample" : samples_3_bpsk_2 , "sync_sequence" : sync_sequence_3 , "mode": "valid" }
]
scenarios_old2_nc = [
    { "name" : "s3 corr" , "desc" : "s3_and_ss3_noclipping 1 sample" , "sample" : samples_3_bpsk_1 , "sync_sequence" : sync_sequence_3_noclipping , "mode": "valid" } ,
    { "name" : "s3 corr" , "desc" : "s3_and_ss3_noclipping 8 samples" , "sample" : samples_3_bpsk_2 , "sync_sequence" : sync_sequence_3_noclipping , "mode": "valid" }
]
scenarios_1_samples = [
    { "name" : "s3 corr" , "desc" : "s3_and_ss3 1 sample" , "sample" : samples_3_bpsk_1 , "sync_sequence" : sync_sequence_3 , "mode": "valid" } 
]
scenarios_8_samples = [
    { "name" : "s3 corr" , "desc" : "s3_and_ss3 8 samples" , "sample" : samples_3_bpsk_2 , "sync_sequence" : sync_sequence_3 , "mode": "valid" } 
]
scenarios_full_samples = [
    { "name" : "s4 corr" , "desc" : "s4_and_ss3 many samples" , "sample" : samples_4_bpsk_1 , "sync_sequence" : sync_sequence_3 , "mode": "valid" } 
]
scenarios_0_samples = [
    { "name" : "s5 corr" , "desc" : "s5_and_ss3 empty samples" , "sample" : samples_5_bpsk_1 , "sync_sequence" : sync_sequence_3 , "mode": "valid" } 
]

conjugate = [ False , True ]
flip = [ False , True ]
magnitude_mode = [ False , True ]

for scenario in scenarios_1_samples :
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

if plt :
    plot.real_waveform_v0_1_6 ( sync_sequence_1 , f"{script_filename} | {sync_sequence_1.size=}" , True )
    plot.real_waveform_v0_1_6 ( s1_corr_1_valid , f"{script_filename} | {s1_corr_1_valid.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_1 , f"{script_filename} | {samples_1.size=} s1_corr_1_valid" , True , np.array([s1_corr_1_valid_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s1_corr_1_same , f"{script_filename} | {s1_corr_1_same.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_1 , f"{script_filename} | {samples_1.size=} s1_corr_1_same" , True , np.array([s1_corr_1_same_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s1_corr_1_full , f"{script_filename} | {s1_corr_1_full.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_1 , f"{script_filename} | {samples_1.size=} s1_corr_1_full" , True , np.array([s1_corr_1_full_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s1_corr_1_fc_valid , f"{script_filename} | {s1_corr_1_fc_valid.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_1 , f"{script_filename} | {samples_1.size=} s1_corr_1_fc_valid" , True , np.array([s1_corr_1_fc_valid_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s1_corr_1_fc_same , f"{script_filename} | {s1_corr_1_fc_same.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_1 , f"{script_filename} | {samples_1.size=} s1_corr_1_fc_same" , True , np.array([s1_corr_1_fc_same_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s1_corr_1_fc_full , f"{script_filename} | {s1_corr_1_fc_full.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_1 , f"{script_filename} | {samples_1.size=} s1_corr_1_fc_full" , True , np.array([s1_corr_1_fc_full_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s2_corr_1_valid , f"{script_filename} | {s2_corr_1_valid.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_2 , f"{script_filename} | {samples_2.size=} s2_corr_1_valid" , True , np.array([s2_corr_1_valid_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s2_corr_1_same , f"{script_filename} | {s2_corr_1_same.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_2 , f"{script_filename} | {samples_2.size=} s2_corr_1_same" , True , np.array([s2_corr_1_same_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s2_corr_1_full , f"{script_filename} | {s2_corr_1_full.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_2 , f"{script_filename} | {samples_2.size=} s2_corr_1_full" , True , np.array([s2_corr_1_full_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s2_corr_1_fc_valid , f"{script_filename} | {s2_corr_1_fc_valid.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_2 , f"{script_filename} | {samples_2.size=} s2_corr_1_fc_valid" , True , np.array([s2_corr_1_fc_valid_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s2_corr_1_fc_same , f"{script_filename} | {s2_corr_1_fc_same.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_2 , f"{script_filename} | {samples_2.size=} s2_corr_1_fc_same" , True , np.array([s2_corr_1_fc_same_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s2_corr_1_fc_full , f"{script_filename} | {s2_corr_1_fc_full.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_2 , f"{script_filename} | {samples_2.size=} s2_corr_1_fc_full" , True , np.array([s2_corr_1_fc_full_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s2_noisy_1_corr_1_valid , f"{script_filename} | {s2_noisy_1_corr_1_valid.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_2_noisy_1 , f"{script_filename} | {samples_2_noisy_1.size=} s2_noisy_1_corr_1_valid" , True , np.array([s2_noisy_1_corr_1_valid_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s2_noisy_1_corr_1_same , f"{script_filename} | {s2_noisy_1_corr_1_same.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_2_noisy_1 , f"{script_filename} | {samples_2_noisy_1.size=} s2_noisy_1_corr_1_same" , True , np.array([s2_noisy_1_corr_1_same_peak_idx]) )
    plot.real_waveform_v0_1_6 ( s2_noisy_1_corr_1_full , f"{script_filename} | {s2_noisy_1_corr_1_full.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_2_noisy_1 , f"{script_filename} | {samples_2_noisy_1.size=} s2_noisy_1_corr_1_full" , True , np.array([s2_noisy_1_corr_1_full_peak_idx]) )
    
    plot.real_waveform_v0_1_6 ( corr_abs_1.astype(np.float64) , f"{script_filename} | {corr_abs_1.size=}" , True )
    plot.real_waveform_v0_1_6 ( samples_1 , f"{script_filename} | {samples_1.size=} corr_abs" , True , np.array([peak_idx_corr_abs_1]) )
    
    plot.complex_waveform_v0_1_6 ( samples_1_float64 , f"{script_filename} | {samples_1_float64.size=}" , True , np.array([peak_idx_norm_corr_1]) )
    plot.complex_waveform_v0_1_6 ( samples_1_float64 , f"{script_filename} | {samples_1_float64.size=}" , True , np.array([peak_idx_norm_corr_best_1]) )
    plot.complex_waveform_v0_1_6 ( samples_1_float64 , f"{script_filename} | {samples_1_float64.size=}" , True , np.array([peak_idx_norm_corr_better_1]) )
    plot.complex_waveform ( sync_sequence_1_float64 , f"{script_filename} | {sync_sequence_1_float64.size=}" , True )
    
    plot.real_waveform ( corr_abs_1 , f"{script_filename} | {corr_abs_1.size=}" , True )
    plot.real_waveform ( samples_power , f"{script_filename} | {samples_power.size=}" , True )
    plot.real_waveform ( cumulative_sample_power_sum , f"{script_filename} | {cumulative_sample_power_sum.size=}" , True )
    plot.real_waveform ( window_energy , f"{script_filename} | {window_energy.size=}" , True )
    plot.real_waveform ( norm_corr_best , f"{script_filename} | {norm_corr_best.size=}" , True )
    plot.real_waveform ( norm_corr_better , f"{script_filename} | {norm_corr_better.size=}" , True )
    '''