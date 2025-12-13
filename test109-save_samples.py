from modules import ops_file , plot , modulation
from pathlib import Path
import numpy as np

Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

plt = True

samples_tri = np.array ( [ 0 , 1 , 0 , -1 , 0 , 2 , 0 , -2 , 0 , 8 , 0 , -2 , 0 , 2 , 0 , -1 , 0 , 1 , 0 ,  ] , dtype = np.int8 )
samples_sq = np.array ( [ 0 , 1 , 1 , -1 , -1 , 0  ] , dtype = np.int8 )

filename_tri = "np.samples/real_tri_1.npy"
filename_sq = "np.samples/real_sq_1.npy"
filename_clipped = "np.samples/barker13_samples_clipped.npy"
filename = "np.samples/barker13_samples.npy"

barker13 = modulation.get_barker13_bpsk_samples_v0_1_3 ()
barker13_clipped = modulation.get_barker13_bpsk_samples_v0_1_3 ( clipped = True )


if plt :
    #plot.complex_waveform_v0_1_6 ( barker13 , f"{barker13.size=}" , True )
    #plot.complex_waveform_v0_1_6 ( barker13_clipped , f"{barker13_clipped.size=}" , True )
    plot.real_waveform ( samples_tri , f"{samples_tri.size=}" , True )
    plot.real_waveform ( samples_sq , f"{samples_sq.size=}" , True )

#ops_file.save_samples_2_npf ( filename , barker13 )
#ops_file.save_samples_2_npf ( filename_clipped , barker13_clipped )
ops_file.save_samples_2_npf ( filename_tri , samples_tri )
ops_file.save_samples_2_npf ( filename_sq , samples_sq )
