from modules import ops_file , plot , modulation
from pathlib import Path
import numpy as np

Path ( "correlation" ).mkdir ( parents = True , exist_ok = True )

plt = True
save = False

filename_sync_sequence_1 = "correlation/sync_sequence_1.npy"
filename_sync_sequence_2 = "correlation/sync_sequence_2.npy"
filename_samples_1 = "correlation/samples_1.npy"
filename_samples_2 = "correlation/samples_2.npy"
filename_barker13 = "correlation/barker13.npy"
filename_barker13_clipped = "correlation/barker13_clipped.npy"

sync_sequence_1 = np.array ( [ 0 , 100 , 100 , -100 , -100 , 0  ] , dtype = np.int32 )
sync_sequence_2 = np.array ( [ 0 , 100 , 0 , -100 , 0 , 200 , 0 , -200 , 0 , 1000 , 0 , -200 , 0 , 200 , 0 , -100 , 0 , 100 , 0 ] , dtype = np.int32 )

samples_1 = np.array ( [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 
                              0 , 100 , 100 , -100 , -100 , 0  ,
                              0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
                             , dtype = np.int32 )
samples_2 = np.array ( [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 100 , 0 , -100 , 0 , 200 , 0 , -200 , 0 , 1000 , 0 , -200 , 0 , 200 , 0 , -100 , 0 , 100 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
                               , dtype = np.int32 )

barker13 = modulation.get_barker13_bpsk_samples_v0_1_3 ( clipped = False )
barker13_clipped = modulation.get_barker13_bpsk_samples_v0_1_3 ( clipped = True )

if plt :
    plot.complex_waveform_v0_1_6 ( barker13 , f"{barker13.size=}" , True )
    plot.complex_waveform_v0_1_6 ( barker13_clipped , f"{barker13_clipped.size=}" , True )
    #plot.real_waveform ( sync_sequence_1 , f"{sync_sequence_1.size=}" , True )
    #plot.real_waveform ( samples_1 , f"{samples_1.size=}" , True )
    #plot.real_waveform ( sync_sequence_2 , f"{sync_sequence_2.size=}" , True )
    #plot.real_waveform ( samples_2 , f"{samples_2.size=}" , True )

if save :
    ops_file.save_complex_samples_2_npf ( filename_barker13 , barker13 )
    ops_file.save_complex_samples_2_npf ( filename_barker13_clipped , barker13_clipped )
    #ops_file.save_samples_2_npf ( filename_sync_sequence_1 , sync_sequence_1 )
    #ops_file.save_samples_2_npf ( filename_sync_sequence_2 , sync_sequence_2 )
    #ops_file.save_samples_2_npf ( filename_samples_1 , samples_1 )
    #ops_file.save_samples_2_npf ( filename_samples_2 , samples_2 )