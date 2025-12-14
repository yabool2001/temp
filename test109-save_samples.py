from modules import ops_file , plot , modulation
from pathlib import Path
import numpy as np

Path ( "normalization" ).mkdir ( parents = True , exist_ok = True )

plt = True

filename_samples_square_sync_sequence = "normalization/samples_square_sync_sequence.npy"
filename_samples_triangle_sync_sequence = "normalization/samples_triangle_sync_sequence.npy"
filename_samples_square = "normalization/samples_square.npy"
filename_samples_triangle = "normalization/samples_triangle.npy"

samples_square_sync_sequence = np.array ( [ 0 , 100 , 100 , -100 , -100 , 0  ] , dtype = np.int32 )
samples_triangle_sync_sequence = np.array ( [ 0 , 100 , 0 , -100 , 0 , 200 , 0 , -200 , 0 , 1000 , 0 , -200 , 0 , 200 , 0 , -100 , 0 , 100 , 0 ] , dtype = np.int32 )

samples_square = np.array ( [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 
                              0 , 100 , 100 , -100 , -100 , 0  ,
                              0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
                             , dtype = np.int32 )
samples_triangle = np.array ( [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 100 , 0 , -100 , 0 , 200 , 0 , -200 , 0 , 1000 , 0 , -200 , 0 , 200 , 0 , -100 , 0 , 100 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
                               , dtype = np.int32 )



barker13 = modulation.get_barker13_bpsk_samples_v0_1_3 ()
barker13_clipped = modulation.get_barker13_bpsk_samples_v0_1_3 ( clipped = True )


if plt :
    #plot.complex_waveform_v0_1_6 ( barker13 , f"{barker13.size=}" , True )
    #plot.complex_waveform_v0_1_6 ( barker13_clipped , f"{barker13_clipped.size=}" , True )
    plot.real_waveform ( samples_square_sync_sequence , f"{samples_square_sync_sequence.size=}" , True )
    plot.real_waveform ( samples_square , f"{samples_square.size=}" , True )
    plot.real_waveform ( samples_triangle_sync_sequence , f"{samples_triangle_sync_sequence.size=}" , True )
    plot.real_waveform ( samples_triangle , f"{samples_triangle.size=}" , True )

#ops_file.save_samples_2_npf ( filename , barker13 )
#ops_file.save_samples_2_npf ( filename_clipped , barker13_clipped )
ops_file.save_samples_2_npf ( filename_samples_square_sync_sequence , samples_square_sync_sequence )
ops_file.save_samples_2_npf ( filename_samples_triangle_sync_sequence , samples_triangle_sync_sequence )
ops_file.save_samples_2_npf ( filename_samples_square , samples_square )
ops_file.save_samples_2_npf ( filename_samples_triangle , samples_triangle )