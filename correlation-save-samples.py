from modules import corrections , ops_file , plot , modulation
from pathlib import Path
import numpy as np

Path ( "correlation" ).mkdir ( parents = True , exist_ok = True )

plt = True
save = True

filename_sync_sequence = "correlation/test120_sync_sequence.npy"
filename_samples_1_npy = "correlation/test120_samples_1.npy"
filename_samples_2_npy = "correlation/test120_samples_2.npy"
filename_samples_3_npy = "correlation/test120_samples_3.npy"
filename_samples_4_npy = "correlation/test120_samples_4.npy"

sync_sequence = np.array ( [ 1. , 1. , 0. , 0. , -2. , -2. , 0. , 0. , 1. , 1.  ] , dtype = np.float64 )


samples_1 = np.array ( [    .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 ,
                            .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 ,
                            1. , 1. , 0. , 0. , -2. , -2. , 0. , 0. , 1. , 1. ,
                            0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                            .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 ]
                            , dtype = np.float64 )
samples_2 = np.array ( [    .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 ,
                            .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 ,
                            10. , 10. , 0. , 0. , -20. , -20. , 0. , 0. , 10. , 10. ,
                            0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                            .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 ]
                              , dtype = np.float64 )
samples_3 = np.array ( [    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                            .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 ,
                            1. , 1. , 0. , 0. , -2. , 0. , 0. , 0. , 1. , 1. ,
                            0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                            .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 ]
                            , dtype = np.float64 )
samples_4 = np.array ( [    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                            .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 ,
                            10. , 10. , 0. , 0. , -20. , 0. , 0. , 0. , 10. , 10. ,
                            0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                            .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 , .0 ]
                            , dtype = np.float64 )

if plt :
    plot.real_waveform_v0_1_6 ( sync_sequence , f"{sync_sequence.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_1 , f"{samples_1.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_2 , f"{samples_2.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_3 , f"{samples_3.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_4 , f"{samples_4.size=}" , False )

if save :
    ops_file.save_samples_2_npf ( filename_sync_sequence , sync_sequence )
    ops_file.save_samples_2_npf ( filename_samples_1_npy , samples_1 )
    ops_file.save_samples_2_npf ( filename_samples_2_npy , samples_2 )
    ops_file.save_samples_2_npf ( filename_samples_3_npy , samples_3 )
    ops_file.save_samples_2_npf ( filename_samples_4_npy , samples_4 )