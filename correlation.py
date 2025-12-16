from modules import filters , ops_file , modulation , packet , plot
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

#plt = True
plt = False

filename_samples_1 = "correlation/samples_1.npy"
filename_samples_2 = "correlation/samples_2.npy"
filename_samples_2_noisy_1 = "correlation/samples_2_noisy_1.npy"
filename_samples_2_noisy_2 = "correlation/samples_2_noisy_2.npy"
filename_samples_2_noisy_3 = "correlation/samples_2_noisy_3.npy"
filename_samples_2_noisy_4 = "correlation/samples_2_noisy_4.npy"
filename_sync_sequence_1 = "correlation/sync_sequence_1.npy"
filename_sync_sequence_2 = "correlation/sync_sequence_2.npy"
filename_results_csv = "correlation/correlation_results.csv"

samples_1 = ops_file.open_real_float64_samples_from_npf ( filename_samples_1 )
samples_2  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2 )
samples_2_noisy_1  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2_noisy_1 )
samples_2_noisy_2  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2_noisy_2 )
samples_2_noisy_3  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2_noisy_3 )
samples_2_noisy_4  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2_noisy_4 )
sync_sequence_1 = ops_file.open_real_float64_samples_from_npf ( filename_sync_sequence_1 )
sync_sequence_2 = ops_file.open_real_float64_samples_from_npf ( filename_sync_sequence_2 )

#plot.real_waveform_v0_1_6 ( sync_sequence_1 , f"sync_sequence_1" , True )
#plot.real_waveform_v0_1_6 ( sync_sequence_2 , f"sync_sequence_2" , True )
#plot.real_waveform_v0_1_6 ( samples_1 , f"samples_1" , True )
#plot.real_waveform_v0_1_6 ( samples_2 , f"samples_2" , True )
#plot.real_waveform_v0_1_6 ( samples_2_noisy_1 , f"samples_2_noisy_1" , True )

scenarios = [
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
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_3 & sync_sequence_2" , "sample" : samples_2_noisy_3 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : True , "flip" : True , "magnitude_mode" : True } ,

    { "name" : "s2 corr" , "desc" : "samples_2_noisy_4 & sync_sequence_2" , "sample" : samples_2_noisy_4 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_4 & sync_sequence_2" , "sample" : samples_2_noisy_4 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_4 & sync_sequence_2" , "sample" : samples_2_noisy_4 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : False } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_4 & sync_sequence_2" , "sample" : samples_2_noisy_4 , "sync_sequence" : sync_sequence_2 , "mode": "valid" , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_4 & sync_sequence_2" , "sample" : samples_2_noisy_4 , "sync_sequence" : sync_sequence_2 , "mode": "same"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True } ,
    { "name" : "s2 corr" , "desc" : "samples_2_noisy_4 & sync_sequence_2" , "sample" : samples_2_noisy_4 , "sync_sequence" : sync_sequence_2 , "mode": "full"  , "conjugate" : False , "flip" : False , "magnitude_mode" : True }
    ]



results = []
for scenario in scenarios:
    if scenario[ "conjugate" ] :
        scenario[ "sync_sequence" ] = np.conj ( scenario[ "sync_sequence" ] )
    if scenario[ "flip" ] :
        scenario[ "sync_sequence" ] = np.flip ( scenario[ "sync_sequence" ] )
    corr = np.correlate ( scenario[ "sample" ] , scenario[ "sync_sequence" ] , mode = scenario[ "mode" ] )
    if scenario[ "magnitude_mode" ] :
        corr = np.abs ( corr )
    peak_idx = int ( np.argmax ( corr ) )
    peak_val = np.abs ( corr[ peak_idx ] )
    name = f"{script_filename} {scenario[ 'desc' ]} | {scenario[ 'name' ]} {scenario[ 'mode' ]} : {'conjugated' if scenario[ 'conjugate' ] else ''} {'fliped' if scenario[ 'flip' ] else ''} {'magnitued' if scenario[ 'magnitude_mode' ] else ''}"
    print ( f"{name} {scenario[ 'desc' ]}: {peak_idx=}, {peak_val=}" )
    results.append ( {
        'name' : scenario['name'],
        'description' : scenario['desc'],
        'correlation_mode' : scenario['mode'],
        'conjugate' : scenario['conjugate'],
        'flip' : scenario['flip'],
        'magnitude' : scenario['magnitude_mode'],
        'peak_idx' : peak_idx,
        'peak_val' : peak_val
    } )
    if plt :
        plot.real_waveform_v0_1_6 ( corr , f"{name}" , True )
        plot.real_waveform_v0_1_6 ( scenario[ "sample" ] , f"{name}" , True , np.array([peak_idx]) )
    

with open ( filename_results_csv , 'w' , newline='' ) as csvfile :
    fieldnames = ['name', 'description', 'correlation_mode', 'conjugate', 'flip', 'magnitude', 'peak_idx', 'peak_val']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)

#t0 = t.perf_counter_ns ()

m1 = len ( samples_1 )
m2 = len ( samples_2 )
n1 = len ( sync_sequence_1 )
n2 = len ( sync_sequence_2 )


corr_abs_1 = np.abs ( np.correlate ( samples_1 , sync_sequence_1 , mode = 'valid' ) )
peak_idx_corr_abs_1 = int ( np.argmax ( corr_abs_1 ) ) ; peak_val_1 = corr_abs_1[ peak_idx_corr_abs_1 ] ; print ( f"corr_abs_1: {peak_idx_corr_abs_1=}, {peak_val_1=}" )

corr_abs_1_fc = np.abs ( np.correlate ( samples_1 , np.flip ( sync_sequence_1.conj () ) , mode = 'valid' ) )
peak_idx_corr_abs_1_fc = int ( np.argmax ( corr_abs_1_fc ) ) ; peak_val_1_fc = corr_abs_1_fc[ peak_idx_corr_abs_1_fc ] ; print ( f"corr_abs_1_fc: {peak_idx_corr_abs_1_fc=}, {peak_val_1_fc=}" )


'''
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